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

/// Delay in ms before probe retry `attempt` (0-based), or `None` to
/// give up. Bounded backoff totalling ~30s: a page loaded while the
/// server is restarting still lights up its GPU demos once the server
/// answers, without polling a dead (or absent) server forever.
#[must_use]
pub fn retry_delay_ms(attempt: u32) -> Option<u32> {
    [1_000, 2_000, 4_000, 8_000, 15_000]
        .get(attempt as usize)
        .copied()
}

/// [`fetch_devices`], retried per [`retry_delay_ms`] until the peer
/// answers with a non-empty device set. Skips the whole schedule when
/// the page has no `?connect=` URL -- nothing to probe.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_devices_with_retry() -> Vec<String> {
    if crate::eval_url::current_connect_url_from_window().is_none() {
        return Vec::new();
    }
    let mut attempt = 0;
    loop {
        let names = fetch_devices().await;
        if !names.is_empty() {
            return names;
        }
        let Some(delay) = retry_delay_ms(attempt) else {
            return names;
        };
        attempt += 1;
        gloo::timers::future::TimeoutFuture::new(delay).await;
    }
}

/// Native stub: no browser, so no probe and nothing to retry.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub async fn fetch_devices_with_retry() -> Vec<String> {
    Vec::new()
}
