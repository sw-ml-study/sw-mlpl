//! Saga 21.5 step 006: `?connect=<url>` query-string parsing.
//!
//! Lifted from `eval.rs` in step 008 so the parent module stays
//! under its 500-line file budget. Pure functions -- the
//! `current_connect_url_from_window` accessor is cfg-gated for
//! WASM since `web_sys::window()` doesn't exist on native.

/// Parse a `?connect=<url>` parameter out of a
/// `window.location.search` query string. Returns `None` when
/// the parameter is missing, empty, or the search string itself
/// is malformed. Pure function so unit tests can drive it
/// without a browser.
#[must_use]
pub fn parse_connect_url(search: &str) -> Option<String> {
    let trimmed = search.strip_prefix('?').unwrap_or(search);
    for pair in trimmed.split('&') {
        if let Some(v) = pair.strip_prefix("connect=") {
            let decoded = url_decode(v);
            if !decoded.is_empty() {
                return Some(decoded);
            }
        }
    }
    None
}

/// Browser-only convenience: read `window.location.search` and
/// run it through `parse_connect_url`. Returns `None` outside a
/// browser context.
#[must_use]
pub fn current_connect_url_from_window() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window()?;
        let search = window.location().search().ok()?;
        parse_connect_url(&search)
    }
    // Native builds have no browser window, hence no connect URL --
    // the `None` (same shape as `is_connected` below) lets UI crates
    // that read the connect URL compile and test natively.
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

/// True when the page is in connect mode (`?connect=<url>` set).
/// Cross-target: always `false` on native builds (no browser, no
/// connect URL) so demo capability-gating compiles everywhere.
#[must_use]
pub fn is_connected() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        current_connect_url_from_window().is_some()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Minimal `%`-decoder for the `connect=` value. Web URLs use a
/// small subset (`%3A` -> `:`, `%2F` -> `/`); pulling in a full
/// `urlencoding` crate for one query parameter is overkill.
fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(byte as char);
            i += 3;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}
