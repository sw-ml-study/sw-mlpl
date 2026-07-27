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

// Wire types (callbacks, metric frames, stream outcomes) moved to
// mlpl-web-eval-core (spike step 015); re-exported so crate::eval::X
// and mlpl_web_eval::eval::X paths keep working.
pub use mlpl_web_eval_core::wire::{MetricCb, RemoteMetric, ResultCb, StreamCb, StreamOutcome};

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

// The native (non-wasm) REST + streaming impls live in the
// `eval_native` / `eval_native_stream` siblings (connect-telemetry
// step 006 split); `CancelHandle` re-exported so callers keep the
// `eval::CancelHandle` path.
#[cfg(not(target_arch = "wasm32"))]
pub use crate::eval_native_stream::CancelHandle;
