//! Probe the connected `mlpl-serve` for its in-process device set
//! (`GET /v1/devices`), so the UI can gate GPU demos by the peer's
//! REAL capability (CUDA on a Linux peer, MLX on an Apple peer) rather
//! than a static "connected => all GPUs" guess.
//!
//! WASM reads the `?connect=` URL and fetches via `gloo::net`; native
//! is a stub (no browser, no probe).

/// Fetch the connected peer's `GET /v1/devices` list (e.g.
/// `["cpu", "cuda"]`). Reads the active `?connect=<url>` itself; returns
/// an empty vec when not connected or on any error, so the caller
/// degrades to "no GPU demos runnable" rather than panicking.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_devices() -> Vec<String> {
    let Some(base) = crate::eval_url::current_connect_url_from_window() else {
        return Vec::new();
    };
    let url = format!("{}/v1/devices", base.trim_end_matches('/'));
    let Ok(resp) = gloo::net::http::Request::get(&url).send().await else {
        return Vec::new();
    };
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    body["devices"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|d| d.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Native stub: no browser, so no live probe.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub async fn fetch_devices() -> Vec<String> {
    Vec::new()
}
