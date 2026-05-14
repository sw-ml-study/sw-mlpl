//! Saga 21.5 step 006: evaluator abstraction for the web REPL.
//!
//! `Evaluator` is callback-based so the WASM (in-process) and
//! REST (remote `mlpl-serve`) impls share one trait surface. The
//! WASM impl invokes the callback synchronously; the REST impl
//! returns immediately and the callback fires once the HTTP
//! round-trip completes -- on WASM via `wasm_bindgen_futures::spawn_local`
//! + `gloo::net`, on native via `reqwest::blocking`.
//!
//! The full UI wiring (`?connect=<url>` swap of the REPL
//! evaluator, full `handlers.rs` refactor onto the trait, the
//! Playwright round-trip test) lands as step 006 ships and step
//! 007 extends it with SSE streaming. Step 006 itself delivers:
//!
//!   1. The trait + both impls (this file).
//!   2. Native unit tests against an in-process `mlpl-serve`
//!      (`apps/mlpl-web/tests/connect_mode_tests.rs`).
//!   3. CORS support on the server
//!      (`mlpl-serve --cors-allow <origin>`).
//!   4. URL parsing for `?connect=<url>` from
//!      `window.location.search`.

use std::cell::RefCell;
use std::rc::Rc;

/// Callback fired with the formatted result of one eval. Matches
/// the existing `WasmSession::eval(&str) -> String` shape: errors
/// arrive as `"error: <msg>"` strings, not as a separate Err
/// variant, so call sites can keep their existing
/// `result.starts_with("error:")` test for the red-text UI.
pub type ResultCb = Box<dyn FnOnce(String) + 'static>;

/// One-line evaluator. The WASM impl runs in-process; the REST
/// impl POSTs to a remote `mlpl-serve`. Both invoke `on_result`
/// exactly once.
pub trait Evaluator {
    /// Evaluate `program` and surface the formatted result via
    /// `on_result`. Implementations decide whether the callback
    /// fires immediately (WASM) or later (REST).
    fn eval(&self, program: &str, on_result: ResultCb);

    /// Reset the evaluator's session state. For the REST impl,
    /// drops the cached session id + token so the next `eval`
    /// will mint a fresh server-side session.
    fn clear(&self);
}

/// In-process WASM evaluator: thin wrapper over `WasmSession`.
/// Reuses an existing session (`from_session`) so the rest of
/// the web app keeps direct access for demos / tutorials that
/// run many lines in a tight loop.
pub struct WasmEvaluator {
    session: Rc<RefCell<mlpl_wasm::WasmSession>>,
}

impl WasmEvaluator {
    #[must_use]
    pub fn from_session(session: Rc<RefCell<mlpl_wasm::WasmSession>>) -> Self {
        Self { session }
    }
}

impl Evaluator for WasmEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let result = self.session.borrow().eval(program);
        on_result(result);
    }
    fn clear(&self) {
        self.session.borrow().clear();
    }
}

/// REST evaluator backed by a remote `mlpl-serve`. Lazily creates
/// a session on first `eval`. Native build uses
/// `reqwest::blocking`; the WASM build uses `gloo::net` + a
/// `wasm_bindgen_futures::spawn_local` to keep the main thread
/// free.
pub struct RemoteEvaluator {
    base_url: String,
    state: Rc<RefCell<Option<RemoteSession>>>,
}

/// Server-side session id + bearer token, minted on first eval.
#[derive(Clone)]
pub(crate) struct RemoteSession {
    pub(crate) session_id: String,
    pub(crate) token: String,
}

impl RemoteEvaluator {
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            state: Rc::new(RefCell::new(None)),
        }
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

// ---- Native impl (tests, reqwest::blocking) ----

#[cfg(not(target_arch = "wasm32"))]
impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let result = native_eval(&self.base_url, &self.state, program);
        on_result(result);
    }
    fn clear(&self) {
        *self.state.borrow_mut() = None;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_eval(base_url: &str, state: &RefCell<Option<RemoteSession>>, program: &str) -> String {
    if state.borrow().is_none() {
        match native_create_session(base_url) {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return format!("error: {e}"),
        }
    }
    let s = state
        .borrow()
        .as_ref()
        .expect("session created above")
        .clone();
    let url = format!(
        "{}/v1/sessions/{}/eval",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let resp = match reqwest::blocking::Client::new()
        .post(&url)
        .bearer_auth(&s.token)
        .json(&serde_json::json!({"program": program}))
        .send()
    {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let body: serde_json::Value = match resp.json() {
        Ok(j) => j,
        Err(e) => return format!("error: decode: {e}"),
    };
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

#[cfg(not(target_arch = "wasm32"))]
fn native_create_session(base_url: &str) -> Result<RemoteSession, String> {
    let url = format!("{}/v1/sessions", base_url.trim_end_matches('/'));
    let body: serde_json::Value = reqwest::blocking::Client::new()
        .post(&url)
        .send()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or("missing session_id")?
        .to_string();
    let token = body
        .get("token")
        .and_then(|v| v.as_str())
        .ok_or("missing token")?
        .to_string();
    Ok(RemoteSession { session_id, token })
}

// ---- WASM impl (browser, gloo::net::http) ----

#[cfg(target_arch = "wasm32")]
impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let base = self.base_url.clone();
        let state = self.state.clone();
        let program = program.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_eval(&base, &state, &program).await;
            on_result(result);
        });
    }
    fn clear(&self) {
        *self.state.borrow_mut() = None;
    }
}

#[cfg(target_arch = "wasm32")]
async fn wasm_eval(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
) -> String {
    if state.borrow().is_none() {
        match wasm_create_session(base_url).await {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return format!("error: {e}"),
        }
    }
    let s = state
        .borrow()
        .as_ref()
        .expect("session created above")
        .clone();
    let url = format!(
        "{}/v1/sessions/{}/eval",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    let req = match gloo::net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", s.token))
        .header("Content-Type", "application/json")
        .body(serde_json::json!({"program": program}).to_string())
    {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => return format!("error: decode: {e}"),
    };
    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return format!("error: {err}");
    }
    body.get("value")
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("error: missing value: {body}"), str::to_string)
}

#[cfg(target_arch = "wasm32")]
async fn wasm_create_session(base_url: &str) -> Result<RemoteSession, String> {
    let url = format!("{}/v1/sessions", base_url.trim_end_matches('/'));
    let resp = gloo::net::http::Request::post(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or("missing session_id")?
        .to_string();
    let token = body
        .get("token")
        .and_then(|v| v.as_str())
        .ok_or("missing token")?
        .to_string();
    Ok(RemoteSession { session_id, token })
}

/// Saga 21.5 step 006: parse a `?connect=<url>` parameter out of
/// a `window.location.search` query string. Returns `None` when
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
#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn current_connect_url_from_window() -> Option<String> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    parse_connect_url(&search)
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
