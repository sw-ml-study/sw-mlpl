//! Saga 33 step 007: connect-mode URL log helper extracted from
//! `App::app`. Saga 21.5 step 006: parse `?connect=<url>` from
//! `window.location.search` and log it for visibility. The
//! actual evaluator swap into the REPL flow lands in step 007
//! alongside the streaming SSE plumbing.

/// Log the connect-mode URL if one is present in the page
/// query string. WASM-only side-effect; on native this is a
/// no-op stub so call sites stay target-agnostic.
pub fn log_connect_mode() {
    #[cfg(target_arch = "wasm32")]
    if let Some(url) = mlpl_web_eval::eval::current_connect_url_from_window() {
        web_sys::console::log_1(
            &format!("[mlpl-web] ?connect={url} parsed (wiring in step 007)").into(),
        );
    }
}
