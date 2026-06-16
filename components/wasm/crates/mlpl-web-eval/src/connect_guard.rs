//! Whether connect mode can actually reach the server from THIS page.
//!
//! The public demo is served over HTTPS. A `?connect=http://...` to a
//! non-local server is blocked by the browser (mixed content: an https://
//! page may not fetch a plain http:// origin), and a cross-origin server
//! also needs CORS. When that's the case the UI should treat the page as
//! NOT connected -- disable connect demos, keep the button on "Connect",
//! and explain the block -- rather than claim "Connected" and then fail
//! every request with a cryptic "Failed to fetch".

/// Only the wasm `connect_blocked_reason` consumes this; native builds
/// never reach it, so gate the const to avoid a dead-code warning there.
#[cfg(target_arch = "wasm32")]
const MIXED_CONTENT_MSG: &str = "Connect mode is unavailable from the public HTTPS demo. \
Browsers block requests from an https:// page to a plain http:// server (mixed content), and a \
cross-origin server also needs CORS. To use connect mode, open the playground from the server \
itself -- http://<host>:6464/sw-mlpl -- or, on the same machine, append \
?connect=http://127.0.0.1:6464.";

/// The pure mixed-content rule, testable without a browser: an https page
/// cannot reach a plain-http, non-local connect server. `localhost` /
/// `127.0.0.1` are exempt -- browsers treat them as potentially trustworthy.
#[must_use]
pub fn is_mixed_content_blocked(page_is_https: bool, connect_url: &str) -> bool {
    let local = connect_url.contains("localhost")
        || connect_url.contains("127.0.0.1")
        || connect_url.contains("[::1]");
    page_is_https && connect_url.starts_with("http://") && !local
}

/// The user-facing reason connect mode is blocked from this page, or `None`
/// when it should work (same-scheme page/server, or a localhost URL, or no
/// `?connect=` at all).
#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn connect_blocked_reason() -> Option<String> {
    let url = crate::eval_url::current_connect_url_from_window()?;
    let https = web_sys::window()?.location().protocol().ok().as_deref() == Some("https:");
    is_mixed_content_blocked(https, &url).then(|| MIXED_CONTENT_MSG.to_string())
}

/// Native builds have no browser and no connect URL, so nothing is blocked.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn connect_blocked_reason() -> Option<String> {
    None
}
