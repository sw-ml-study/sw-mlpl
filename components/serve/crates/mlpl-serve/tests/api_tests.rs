//! Saga 21 step 001: REST surface integration
//! tests. Spin `mlpl-serve` up on a random localhost
//! port via `server::build_app(...)` + a manual
//! `axum::serve(listener, app)`, then drive it with
//! `reqwest`.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::{RunConfig, ServeConfig};
use mlpl_serve::server::{ServerError, build_app, build_app_with_peers_cors, run};
use serde_json::Value as JsonValue;

/// Spin up a server in the background on a random
/// loopback port. Returns the bound address; the
/// task runs for the duration of the test.
async fn start_server(auth_mode: AuthMode) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(auth_mode);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn create_session(addr: SocketAddr) -> (String, String) {
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "session create should be 200");
    let body: JsonValue = resp.json().await.unwrap();
    let id = body["session_id"].as_str().unwrap().to_string();
    let token = body["token"].as_str().unwrap().to_string();
    assert!(!id.is_empty(), "session_id must be non-empty");
    assert!(!token.is_empty(), "token must be non-empty");
    (id, token)
}

#[tokio::test]
async fn post_sessions_returns_id_and_token() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    // 32 alphanumeric is the contract; check both the
    // length and that no two sessions reuse a token.
    assert_eq!(token.len(), 32, "token should be 32 chars");
    let (id2, token2) = create_session(addr).await;
    assert_ne!(id, id2, "session ids must be unique");
    assert_ne!(token, token2, "tokens must be unique");
}

#[tokio::test]
async fn eval_runs_program_against_session_env() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(5)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["kind"], "array", "iota returns an array");
    let value = body["value"].as_str().unwrap();
    // iota(5) prints as something containing 0..4 in
    // some bracketed form; we don't pin the exact
    // formatting, but the digits must be there.
    for digit in ['0', '1', '2', '3', '4'] {
        assert!(
            value.contains(digit),
            "iota(5) value {value:?} should contain digit {digit}"
        );
    }
}

#[tokio::test]
async fn eval_returns_viz_shape_and_values() {
    // Phase 1c: the eval response carries shape + flat values so the
    // connect-mode web UI can emit a 3D sculpture for a result that
    // was evaluated server-side.
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "iota(6)"}))
        .send()
        .await
        .unwrap();
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["shape"], serde_json::json!([6]), "shape rides back");
    assert_eq!(
        body["values"],
        serde_json::json!([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "flat values ride back for 3D"
    );
}

#[tokio::test]
async fn eval_model_returns_sankey_viz_node() {
    // Phase 1c part 2: a model evaluated server-side carries its
    // Sankey decomposition so connect-mode renders the diagram.
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "chain(linear(4, 3, 0), relu_layer())"}))
        .send()
        .await
        .unwrap();
    let body: JsonValue = resp.json().await.unwrap();
    let nodes = body["viz_node"]["sankey"]["nodes"].as_array();
    assert!(
        nodes.is_some_and(|n| n.len() >= 3),
        "model eval should carry a Sankey viz_node (input + layers + output): {body}"
    );
}

#[tokio::test]
async fn eval_persists_state_across_calls() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let url = format!("http://{addr}/v1/sessions/{id}/eval");
    let client = reqwest::Client::new();

    // First call binds `x = 7` (assignment returns
    // a sentinel; we don't care about the value).
    let r1 = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "x = 7"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);

    // Second call reads `x` -- proves the env
    // survived between requests.
    let r2 = client
        .post(&url)
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let body: JsonValue = r2.json().await.unwrap();
    assert!(
        body["value"].as_str().unwrap().contains('7'),
        "second eval should see x=7 from the first call"
    );
}

#[tokio::test]
async fn eval_without_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
    let body: JsonValue = resp.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("authorization"),
        "401 body should mention authorization"
    );
}

#[tokio::test]
async fn eval_with_wrong_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth("not-the-real-token-xxxxxxxxxxxxx")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn eval_unknown_session_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let bogus = "00000000-0000-0000-0000-000000000000";

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{bogus}/eval"))
        .bearer_auth("anything")
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn eval_program_error_returns_400_with_message() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .bearer_auth(&token)
        .json(&serde_json::json!({"program": "undefined_var_xyz"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: JsonValue = resp.json().await.unwrap();
    let err = body["error"].as_str().unwrap();
    assert!(
        err.contains("undefined_var_xyz") || err.to_lowercase().contains("undefined"),
        "400 body should reference the missing variable: {err}"
    );
}

#[tokio::test]
async fn health_returns_ok_and_version() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(
        !body["version"].as_str().unwrap().is_empty(),
        "version should be set by CARGO_PKG_VERSION"
    );
}

#[tokio::test]
async fn devices_reports_compiled_capabilities() {
    let addr = start_server(AuthMode::Required).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/devices"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();
    let names: Vec<&str> = body["devices"]
        .as_array()
        .expect("devices array")
        .iter()
        .map(|d| d.as_str().unwrap())
        .collect();
    assert!(names.contains(&"cpu"), "cpu always available: {names:?}");
    // `/v1/devices` reflects the build: `cuda` is present iff this
    // server was compiled with the cuda feature on Linux/x86_64.
    let cuda_built = cfg!(all(
        feature = "cuda",
        target_os = "linux",
        target_arch = "x86_64"
    ));
    assert_eq!(
        names.contains(&"cuda"),
        cuda_built,
        "cuda capability tracks the build: {names:?}"
    );
}

#[tokio::test]
async fn run_rejects_non_loopback_with_auth_disabled() {
    // Pick an arbitrary non-loopback address; we
    // don't actually expect to bind it. The safety
    // check should fail before bind.
    let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let err = run(RunConfig {
        addr,
        auth_mode: AuthMode::Disabled,
        peers: mlpl_serve::peers::empty_registry(),
        tls: None,
        serve: ServeConfig::default(),
    })
    .await
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        matches!(err, ServerError::InsecureBind { .. }),
        "expected InsecureBind, got {err:?}"
    );
    assert!(
        msg.contains("--auth required"),
        "error message should mention --auth required: {msg}"
    );
}

#[tokio::test]
async fn inspect_returns_var_snapshot() {
    let addr = start_server(AuthMode::Required).await;
    let (id, token) = create_session(addr).await;
    let url_eval = format!("http://{addr}/v1/sessions/{id}/eval");
    let client = reqwest::Client::new();

    // Bind two vars; one a model.
    for prog in [
        "x = iota(5)",
        "y = reshape(iota(12), [3, 4])",
        "m = linear(3, 4, 0)",
    ] {
        let resp = client
            .post(&url_eval)
            .bearer_auth(&token)
            .json(&serde_json::json!({"program": prog}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "setup eval `{prog}` should succeed");
    }

    let resp = client
        .get(format!("http://{addr}/v1/sessions/{id}/inspect"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: JsonValue = resp.json().await.unwrap();

    let var_names: Vec<&str> = body["vars"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert!(
        var_names.contains(&"x"),
        "x should be in vars: {var_names:?}"
    );
    assert!(
        var_names.contains(&"y"),
        "y should be in vars: {var_names:?}"
    );

    // y has shape [3, 4]
    let y = body["vars"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == "y")
        .unwrap();
    assert_eq!(y["shape"], serde_json::json!([3, 4]));

    // Models list contains m
    let models: Vec<&str> = body["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(models.contains(&"m"), "m should be in models: {models:?}");

    // No truncation expected for a tiny snapshot
    assert_eq!(body["more"], 0);
}

#[tokio::test]
async fn inspect_without_bearer_returns_401() {
    let addr = start_server(AuthMode::Required).await;
    let (id, _token) = create_session(addr).await;

    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{id}/inspect"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn inspect_unknown_session_returns_404() {
    let addr = start_server(AuthMode::Required).await;
    let bogus = "00000000-0000-0000-0000-000000000000";

    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/v1/sessions/{bogus}/inspect"))
        .bearer_auth("anything")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// Saga 25 step A: with `--static-dir <path>` set,
/// the server mounts a `ServeDir` at `/sw-mlpl/`. Smoke
/// test it with a tempdir + a tiny `index.html`. The
/// /v1 routes must keep working alongside.
#[tokio::test]
async fn static_dir_serves_index_at_sw_mlpl_prefix() {
    use std::io::Write;
    let tmp = std::env::temp_dir().join(format!(
        "mlpl-serve-static-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let index_path = tmp.join("index.html");
    let mut f = std::fs::File::create(&index_path).unwrap();
    writeln!(f, "<!DOCTYPE html><title>mlpl-test</title>").unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app_with_peers_cors(
        AuthMode::Required,
        mlpl_serve::peers::empty_registry(),
        ServeConfig {
            static_dir: Some(tmp.clone()),
            ..Default::default()
        },
    );
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Static side: GET /sw-mlpl/ should return the index.
    let html = reqwest::get(format!("http://{addr}/sw-mlpl/index.html"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("mlpl-test"), "served body was {html:?}");

    // API side: /v1/health must still work alongside.
    let health = reqwest::get(format!("http://{addr}/v1/health"))
        .await
        .unwrap();
    assert_eq!(health.status(), 200);

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Saga 25 step B: `--self-signed` mode terminates TLS
/// in-process and serves the same routes over https. Smoke
/// test it by wiring the server's `run()` into an
/// `axum_server::bind_rustls` flow with a freshly-generated
/// loopback cert, then call /v1/health with a TLS client
/// configured to accept any cert (this is a self-signed
/// loopback test, not a trust check).
#[tokio::test]
async fn self_signed_serves_health_over_tls() {
    use std::net::TcpListener as StdListener;
    let listener = StdListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let (tls_config, fingerprint) = mlpl_serve::tls::self_signed_loopback().await.unwrap();
    assert!(fingerprint.contains(':'), "fingerprint format wrong");
    tokio::spawn(async move {
        let _ = run(RunConfig {
            addr,
            auth_mode: AuthMode::Required,
            peers: mlpl_serve::peers::empty_registry(),
            tls: Some(tls_config),
            serve: ServeConfig::default(),
        })
        .await;
    });
    // Brief settle for the listener.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();
    let resp = client
        .get(format!("https://{addr}/v1/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "TLS /v1/health must return 200");
    let body: JsonValue = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn auth_disabled_skips_bearer_check() {
    let addr = start_server(AuthMode::Disabled).await;
    let (id, _token) = create_session(addr).await;

    // Note: NO bearer header.
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions/{id}/eval"))
        .json(&serde_json::json!({"program": "iota(3)"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "with AuthMode::Disabled, missing bearer should still succeed"
    );
}
