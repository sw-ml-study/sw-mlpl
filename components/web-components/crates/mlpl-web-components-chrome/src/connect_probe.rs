//! Connect-button probe helpers: the mount-time reachability check
//! behind the button label, and the per-backend availability summary
//! for the connected panel. Split from `connect_button` to keep that
//! module inside the function-count and function-LOC budgets.

use yew::prelude::*;

/// Whether the `?connect=` server answered the devices probe.
/// `None` while the (retrying) probe is still running; `Some(false)`
/// once it gave up -- the button shows a warning instead of a lying
/// check mark. Never probes when the page has no `?connect=`.
#[hook]
pub fn use_reachable() -> Option<bool> {
    let reachable = use_state(|| None::<bool>);
    {
        let reachable = reachable.clone();
        use_effect_with((), move |()| {
            wasm_bindgen_futures::spawn_local(async move {
                if mlpl_web_eval::eval_url::is_connected() {
                    let names = mlpl_web_eval::devices::fetch_devices_with_retry().await;
                    reachable.set(Some(!names.is_empty()));
                }
            });
            || ()
        });
    }
    *reachable
}

/// One JSON GET, `None` on any transport or non-2xx failure --
/// probes must degrade, never throw.
pub(crate) async fn get_json(url: &str) -> Option<serde_json::Value> {
    let resp = gloo::net::http::Request::get(url).send().await.ok()?;
    resp.ok().then_some(())?;
    resp.json().await.ok()
}

/// Per-backend availability summary for the connected panel, from
/// the `/v1/devices` body: the raw device list plus an explicit
/// available/unavailable verdict for CUDA, MLX, and Ollama.
#[must_use]
pub fn backend_status(body: &serde_json::Value) -> String {
    let devs: Vec<&str> = body["devices"]
        .as_array()
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    let avail = |b: bool| if b { "available" } else { "unavailable" };
    format!(
        "{} -- CUDA: {}, MLX: {}, Ollama: {}",
        devs.join(", "),
        avail(devs.contains(&"cuda")),
        avail(devs.contains(&"mlx")),
        avail(body["ollama"].as_bool() == Some(true)),
    )
}

/// The connected server's build stamp from `GET /v1/health`
/// (`commit` + `built_at`), so the splash can show it beside the UI
/// build and flag a UI-vs-server skew.
#[derive(Clone, PartialEq)]
pub struct ServerBuild {
    pub commit: String,
    pub built_at: String,
}

/// Fetch the connected server's `/v1/health` build stamp once on
/// mount. `None` when not connected or until it answers (so the UI
/// simply omits the server line rather than lying).
#[hook]
pub fn use_server_build() -> Option<ServerBuild> {
    let build = use_state(|| None::<ServerBuild>);
    {
        let build = build.clone();
        use_effect_with((), move |()| {
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(base) = mlpl_web_eval::eval_url::current_connect_url_from_window() {
                    let url = format!("{}/v1/health", base.trim_end_matches('/'));
                    if let Some(body) = get_json(&url).await
                        && let (Some(commit), Some(built_at)) =
                            (body["commit"].as_str(), body["built_at"].as_str())
                    {
                        build.set(Some(ServerBuild {
                            commit: commit.to_string(),
                            built_at: built_at.to_string(),
                        }));
                    }
                }
            });
            || ()
        });
    }
    (*build).clone()
}
