//! Probe the connected `mlpl-serve` for its in-process device set
//! (`GET /v1/devices`), so the UI can gate GPU demos by the peer's
//! REAL capability (CUDA on a Linux peer, MLX on an Apple peer) rather
//! than a static "connected => all GPUs" guess.
//!
//! WASM reads the `?connect=` URL and fetches via `gloo::net`; native
//! is a stub (no browser, no probe).

/// Fetch a server's `GET /v1/devices` list (e.g. `["cpu", "cuda"]`)
/// from an explicit base URL; empty vec on any error, so callers
/// degrade to "no GPU demos runnable" rather than panicking. Used by
/// the retrying probe below and by same-origin autoconnect, which
/// probes the page's own origin before any `?connect=` exists.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_devices_at(base: &str) -> Vec<String> {
    let url = format!("{}/v1/devices", base.trim_end_matches('/'));
    let Ok(resp) = gloo::net::http::Request::get(&url).send().await else {
        return Vec::new();
    };
    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    parse_devices_body(&body)
}

/// Map a `/v1/devices` body to the peer-capability name set: the
/// `devices` array verbatim, plus a synthetic `"ollama"` entry when
/// the server reports its configured Ollama host alive (absent field
/// -- an older server -- means not alive). Pure for native tests.
#[must_use]
pub fn parse_devices_body(body: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = body["devices"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|d| d.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if body["ollama"].as_bool() == Some(true) {
        names.push("ollama".to_string());
    }
    names
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

/// Probe the `?connect=` target, retried per [`retry_delay_ms`] until
/// it answers with a non-empty device set. Skips the whole schedule
/// when the page has no `?connect=` URL -- nothing to probe. After a
/// couple of failed attempts, self-heals: if the PAGE's own server is
/// alive (and is not the dead target), the `?connect=` is rewritten
/// to the origin and the page reloads connected -- a stale or
/// mistyped target (e.g. `localhost` from another machine) fixes
/// itself instead of stranding the user on a diagnostic.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_devices_with_retry() -> Vec<String> {
    let Some(base) = crate::eval_url::current_connect_url_from_window() else {
        return Vec::new();
    };
    let mut attempt = 0;
    loop {
        let names = fetch_devices_at(&base).await;
        if !names.is_empty() {
            return names;
        }
        if attempt == 1 && fallback_to_origin(&base).await {
            return Vec::new();
        }
        let Some(delay) = retry_delay_ms(attempt) else {
            return names;
        };
        attempt += 1;
        gloo::timers::future::TimeoutFuture::new(delay).await;
    }
}

/// When the connect target is dead but the page's own origin answers
/// `/v1/devices`, rewrite `?connect=` to the origin (page reloads
/// connected). Returns whether the rewrite was issued.
#[cfg(target_arch = "wasm32")]
async fn fallback_to_origin(target: &str) -> bool {
    let Some(win) = web_sys::window() else {
        return false;
    };
    let Ok(origin) = win.location().origin() else {
        return false;
    };
    if origin.trim_end_matches('/') == target.trim_end_matches('/') {
        return false;
    }
    if fetch_devices_at(&origin).await.is_empty() {
        return false;
    }
    web_sys::console::warn_1(
        &format!("[mlpl-web] ?connect={target} is not responding; reconnecting to {origin}").into(),
    );
    win.location()
        .set_search(&format!("connect={origin}"))
        .is_ok()
}

/// Native stub: no browser, so no probe and nothing to retry.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub async fn fetch_devices_with_retry() -> Vec<String> {
    Vec::new()
}
