//! Saga 21.5 step 004: viz-storage URL surfacing in `--connect`
//! mode. When the server's `/eval` (or `/eval_stream`) returns
//! an SVG, the response now carries a `viz_url` field pointing
//! into the server's `POST /v1/viz` storage. The connect client
//! exposes this on the parsed `EvalResponse` so the REPL can
//! print `viz: <url>` alongside the locally-cached `viz: <path>`.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn blocking_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap()
}

/// `eval_remote` parses out `viz_url` for an SVG-returning
/// program so the connect REPL can render `viz: <url>` before
/// the value.
#[tokio::test(flavor = "multi_thread")]
async fn connect_eval_surfaces_viz_url_for_svg() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = mlpl_repl_connect::connect::create_session(&client, &base).unwrap();
        let resp = mlpl_repl_connect::connect::eval_remote(
            &client,
            &base,
            &id,
            &token,
            "hist(iota(10), 5)",
        )
        .unwrap();
        assert_eq!(resp.kind, "string");
        let url = resp
            .viz_url
            .as_deref()
            .expect("expected viz_url on SVG-returning eval");
        assert!(
            url.starts_with("/v1/viz/"),
            "viz_url should be a /v1/viz/ path: {url}"
        );
    })
    .await;
    result.unwrap();
}

/// Non-SVG `eval_remote` responses leave `viz_url` as `None`.
#[tokio::test(flavor = "multi_thread")]
async fn connect_eval_leaves_viz_url_none_for_non_svg() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let client = blocking_client();
        let (id, token) = mlpl_repl_connect::connect::create_session(&client, &base).unwrap();
        let resp =
            mlpl_repl_connect::connect::eval_remote(&client, &base, &id, &token, "iota(5) + 1")
                .unwrap();
        assert!(
            resp.viz_url.is_none(),
            "viz_url should be None for non-SVG: {:?}",
            resp.viz_url
        );
    })
    .await;
    result.unwrap();
}
