//! Static-mount cache policy: every `/sw-mlpl` response must carry
//! `Cache-Control: no-cache` so browsers REVALIDATE on each load
//! instead of heuristically caching index.html -- a cached index
//! keeps serving a stale (days-old) bundle after every deploy.

use std::net::SocketAddr;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::ServeConfig;
use mlpl_serve::server::build_app_with_peers_cors;

async fn start_with_static(dir: std::path::PathBuf) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let serve = ServeConfig {
        static_dir: Some(dir),
        ..ServeConfig::default()
    };
    let app = build_app_with_peers_cors(
        AuthMode::Required,
        mlpl_serve::peers::empty_registry(),
        serve,
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

#[tokio::test]
async fn static_responses_are_no_cache() {
    let dir = std::env::temp_dir().join(format!("mlpl-static-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("index.html"), "<html>hi</html>").unwrap();
    let addr = start_with_static(dir).await;
    let resp = reqwest::Client::new()
        .get(format!("http://{addr}/sw-mlpl/index.html"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let cc = resp
        .headers()
        .get("cache-control")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        cc, "no-cache",
        "static files must revalidate every load, got {cc:?}"
    );
}
