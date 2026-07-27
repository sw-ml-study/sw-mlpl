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
