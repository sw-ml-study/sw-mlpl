//! Saga 21.5 step 006: evaluator abstraction for the web REPL.
//!
//! `Evaluator` is callback-based so the WASM (in-process) and
//! REST (remote `mlpl-serve`) impls share one trait surface. The
//! WASM impl invokes the callback synchronously; the REST impl
//! returns immediately and the callback fires once the HTTP
//! round-trip completes -- on WASM via `wasm_bindgen_futures::spawn_local`
//! + `gloo::net` (`eval_wasm.rs`), on native via `reqwest::blocking`.
//!
//! Saga 21.5 step 008 split out three sibling modules to defend
//! the per-file LOC budget: `eval_sse.rs` (shared SSE parser),
//! `eval_url.rs` (`?connect=` query helpers), and `eval_wasm.rs`
//! (browser-only impls). This file keeps the trait + shared
//! types + the native REST impl + the public `fetch_viz` helper
//! used by both step 008's tests and step 007's UI work.

use std::cell::RefCell;
use std::rc::Rc;

#[allow(unused_imports)]
pub use crate::eval_url::current_connect_url_from_window;
#[allow(unused_imports)]
pub use crate::eval_url::parse_connect_url;

/// Callback fired with the formatted result of one eval. Matches
/// the existing `WasmSession::eval(&str) -> String` shape: errors
/// arrive as `"error: <msg>"` strings, not as a separate Err
/// variant, so call sites can keep their existing
/// `result.starts_with("error:")` test for the red-text UI.
pub type ResultCb = Box<dyn FnOnce(String) + 'static>;

/// Saga 21.5 step 007: per-iteration metric frame surfaced by
/// `RemoteEvaluator::eval_stream`. Mirrors the server-side
/// `SseEvent::Metric` payload.
#[derive(Debug, Clone)]
pub struct RemoteMetric {
    pub name: String,
    pub step: usize,
    pub value: f64,
}

/// Saga 21.5 step 007: terminal SSE frame. `Done` carries the
/// final value + kind (matching the non-streaming `/eval`);
/// `Cancelled` carries the step + partial loss curve from a
/// `train { }` that observed a cancel; `Error` covers both
/// HTTP-level failures (auth, lex/parse) and runtime
/// `EvalError`s emitted as `event: error`.
#[derive(Debug)]
pub enum StreamOutcome {
    Done {
        value: String,
        kind: String,
    },
    Cancelled {
        step: usize,
        partial_losses: Vec<f64>,
    },
    Error {
        message: String,
    },
}

/// Callback fired once per `event: metric` SSE frame during
/// streaming eval.
pub type MetricCb = Box<dyn FnMut(&RemoteMetric) + 'static>;

/// Callback fired exactly once with the terminal outcome of a
/// streaming eval.
pub type StreamCb = Box<dyn FnOnce(StreamOutcome) + 'static>;

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
/// `reqwest::blocking`; the WASM build uses `gloo::net` (see
/// `eval_wasm.rs`).
pub struct RemoteEvaluator {
    base_url: String,
    state: Rc<RefCell<Option<RemoteSession>>>,
}

/// Server-side session id + bearer token, minted on first eval.
#[derive(Clone)]
pub struct RemoteSession {
    pub session_id: String,
    pub token: String,
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

    /// Snapshot of the current server-side session id, if one
    /// has been minted. Used by `cancel` to address the in-flight
    /// stream.
    pub(crate) fn current_session_id(&self) -> Option<String> {
        self.state.borrow().as_ref().map(|s| s.session_id.clone())
    }

    pub(crate) fn current_token(&self) -> Option<String> {
        self.state.borrow().as_ref().map(|s| s.token.clone())
    }

    /// Internal hook for `eval_wasm.rs` to receive the shared
    /// state cell without exposing it to external callers.
    #[allow(dead_code)]
    pub(crate) fn state_handle(&self) -> Rc<RefCell<Option<RemoteSession>>> {
        self.state.clone()
    }

    pub(crate) fn clear_state(&self) {
        *self.state.borrow_mut() = None;
    }
}

/// Resolve a viz URL (`/v1/viz/<id>` path OR
/// `http://host/v1/viz/<id>` absolute) into a fully-qualified URL
/// against `base_url`. Used by both native `fetch_viz` and WASM
/// `fetch_viz_async`.
pub(crate) fn resolve_viz_url(base_url: &str, viz_url: &str) -> String {
    if viz_url.starts_with("http://") || viz_url.starts_with("https://") {
        viz_url.to_string()
    } else {
        format!("{}{}", base_url.trim_end_matches('/'), viz_url)
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
        self.clear_state();
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

/// Saga 21.5 step 007: cheap-to-clone Send-able handle for
/// firing a cancel POST from a different thread (native test).
/// Native-only: the fields are read solely by `cancel()`
/// (`cfg(not(wasm32))`), and the constructor `cancel_handle()`
/// lives in the native `RemoteEvaluator` impl. WASM cancels
/// in-tick on the single-threaded JS event loop, so the struct
/// is not compiled there (it would be dead code).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct CancelHandle {
    base_url: String,
    session_id: String,
    token: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl CancelHandle {
    /// Best-effort POST `/v1/sessions/<id>/cancel`. Errors are
    /// swallowed because cancel-from-side-thread inherently
    /// races with the in-flight stream; the eval result already
    /// surfaces via `StreamOutcome::Cancelled`.
    pub fn cancel(&self) {
        let url = format!(
            "{}/v1/sessions/{}/cancel",
            self.base_url.trim_end_matches('/'),
            self.session_id
        );
        let _ = reqwest::blocking::Client::new()
            .post(&url)
            .bearer_auth(&self.token)
            .send();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RemoteEvaluator {
    /// Native streaming-eval impl. POSTs to `/eval_stream`, reads
    /// the SSE body line-by-line, fires `on_metric` per metric
    /// frame, and fires `on_result` exactly once with the
    /// terminal outcome.
    pub fn eval_stream(&self, program: &str, on_metric: MetricCb, on_result: StreamCb) {
        let outcome = native_eval_stream(&self.base_url, &self.state, program, on_metric);
        on_result(outcome);
    }

    /// In-tick cancel for the browser UI (single-threaded JS
    /// event loop). Tests cross threads via `cancel_handle()`
    /// instead. No-op when no session has been minted yet.
    pub fn cancel(&self) {
        if let Some(h) = self.cancel_handle() {
            h.cancel();
        }
    }

    /// Snapshot a `Send`-able handle that can fire `cancel`
    /// from a different thread. `None` until the first eval
    /// has minted a server session.
    pub fn cancel_handle(&self) -> Option<CancelHandle> {
        let session_id = self.current_session_id()?;
        let token = self.current_token()?;
        Some(CancelHandle {
            base_url: self.base_url.clone(),
            session_id,
            token,
        })
    }

    /// Saga 21.5 step 008: native viz fetch. `GET /v1/viz/<id>`
    /// with bearer auth, returns the raw bytes + content type so
    /// callers can render an `<img>` (for `image/*`) or an
    /// `<iframe srcdoc>` (for `text/html`). Accepts both bare
    /// paths and absolute URLs. The WASM equivalent
    /// (`fetch_viz_async`) lives in `eval_wasm.rs`.
    pub fn fetch_viz(&self, viz_url: &str, bearer: &str) -> Result<(Vec<u8>, String), String> {
        let url = resolve_viz_url(&self.base_url, viz_url);
        let resp = reqwest::blocking::Client::new()
            .get(&url)
            .bearer_auth(bearer)
            .send()
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("fetch_viz: {}", resp.status()));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();
        Ok((bytes, content_type))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_eval_stream(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    mut on_metric: MetricCb,
) -> StreamOutcome {
    if state.borrow().is_none() {
        match native_create_session(base_url) {
            Ok(s) => *state.borrow_mut() = Some(s),
            Err(e) => return StreamOutcome::Error { message: e },
        }
    }
    let s = state.borrow().as_ref().expect("session created").clone();
    let url = format!(
        "{}/v1/sessions/{}/eval_stream",
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
        Err(e) => {
            return StreamOutcome::Error {
                message: e.to_string(),
            };
        }
    };
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return StreamOutcome::Error { message };
    }
    let reader = std::io::BufReader::new(resp);
    crate::eval_sse::parse_sse_stream(reader, &mut on_metric)
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
