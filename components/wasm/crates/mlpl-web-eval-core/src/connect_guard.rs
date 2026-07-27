//! Whether connect mode can actually reach the server from THIS page.
//!
//! The public demo is served over HTTPS. A `?connect=http://...` to a
//! non-local server is blocked by the browser (mixed content: an https://
//! page may not fetch a plain http:// origin), and a cross-origin server
//! also needs CORS. When that's the case the UI should treat the page as
//! NOT connected -- disable connect demos, keep the button on "Connect",
//! and explain the block -- rather than claim "Connected" and then fail
//! every request with a cryptic "Failed to fetch".

// Compat re-export: the pure rule moved to eval_url (URL-scheme
// logic); callers keep using connect_guard::is_mixed_content_blocked.
pub use crate::eval_url::is_mixed_content_blocked;

/// Only the wasm `connect_blocked_reason` consumes this; native builds
/// never reach it, so gate the const to avoid a dead-code warning there.
#[cfg(target_arch = "wasm32")]
const MIXED_CONTENT_MSG: &str = "Connect mode is unavailable from the public HTTPS demo. \
Browsers block requests from an https:// page to a plain http:// server (mixed content), and a \
cross-origin server also needs CORS. To use connect mode, open the playground from the server \
itself -- http://<host>:6464/sw-mlpl -- or, on the same machine, append \
?connect=http://127.0.0.1:6464.";

/// The user-facing reason connect mode is blocked from this page, or `None`
/// when it should work (same-scheme page/server, or a localhost URL, or no
/// `?connect=` at all).
#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn connect_blocked_reason() -> Option<String> {
    let url = crate::eval_url::current_connect_url_from_window()?;
    let https = web_sys::window()?.location().protocol().ok().as_deref() == Some("https:");
    crate::eval_url::is_mixed_content_blocked(https, &url).then(|| MIXED_CONTENT_MSG.to_string())
}

/// Native builds have no browser and no connect URL, so nothing is blocked.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn connect_blocked_reason() -> Option<String> {
    None
}

/// Whether `program` contains a `train N { ... }` block -- the only
/// construct that emits per-step `*_metric` SSE frames, and therefore
/// the only kind of connect eval routed through the streaming endpoint.
/// Plain evals keep the JSON `/eval` path, which carries the 3D-viz
/// payload the stream's `done` frame does not.
#[must_use]
pub fn program_streams_metrics(program: &str) -> bool {
    let t = program.trim_start();
    if t.starts_with(':') || t.starts_with("llm_call(") {
        return false;
    }
    // `emit_frame(...)` lines stream live tensor frames (Game of
    // Life saga step 4) exactly like train blocks stream metrics.
    has_train_block(t) || t.contains("emit_frame(")
}

/// `train` as a standalone word with a `{` somewhere after it, so
/// `retrain_flag` / `trainx` (and "train" inside an `:ask` question,
/// which has no block) do not count.
fn has_train_block(t: &str) -> bool {
    let bytes = t.as_bytes();
    for (i, _) in t.match_indices("train") {
        let boundary_before =
            i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        let after = bytes.get(i + 5);
        let boundary_after = matches!(after, Some(c) if c.is_ascii_whitespace() || *c == b'{');
        if boundary_before && boundary_after && t[i..].contains('{') {
            return true;
        }
    }
    false
}
