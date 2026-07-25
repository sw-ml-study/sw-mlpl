//! Saga 21.5 step 008: browser-only `RemoteEvaluator` impls.
//!
//! Lifted out of `eval.rs` so the parent module stays under its
//! 500-line file budget. Every fn is `#[cfg(target_arch =
//! "wasm32")]` because the entire module body would otherwise
//! force `gloo::net` + `wasm_bindgen_futures` on native builds.
//! `cargo test` compiles this file but elides every function;
//! the WASM build (`scripts/build-pages.sh`) wires them in.
//!
//! Saga 77: HTTP helper bodies (`wasm_create_session`,
//! `wasm_eval_stream`, `wasm_eval_stream_inner`) moved to
//! `eval_wasm_helpers.rs` so the parent stays under the
//! module-function-count limit.

#![cfg(target_arch = "wasm32")]

use crate::eval::current_connect_url_from_window;
use crate::eval::{Evaluator, MetricCb, RemoteEvaluator, ResultCb, StreamCb};
use crate::eval_wasm_helpers::wasm_eval;
use crate::eval_wasm_stream::wasm_eval_stream;

thread_local! {
    // One persistent RemoteEvaluator per page so the server-side
    // session (workspace state) is reused across submitted lines.
    static CONNECT_EVALUATOR: std::cell::RefCell<Option<RemoteEvaluator>> =
        const { std::cell::RefCell::new(None) };
}

/// Connect-mode one-shot eval. When `?connect=<url>` is present in
/// the page URL, route `program` to that `mlpl-serve` over the
/// REST eval API (async, server-side) and fire `on_result` with
/// the server's value -- so `llm_call` / MLX-GPU work run on the
/// server and never block (or panic) the browser. Returns true
/// when it took the eval; false (leaving `on_result` uncalled)
/// when no connect URL is set, so the caller does local eval.
pub fn connect_eval(program: &str, on_result: ResultCb) -> bool {
    let Some(url) = current_connect_url_from_window() else {
        return false;
    };
    // The browser will block this fetch (https page -> http server) before it
    // leaves the tab. Short-circuit with a clear explanation instead of a
    // cryptic "Failed to fetch".
    if let Some(reason) = crate::connect_guard::connect_blocked_reason() {
        on_result(format!("error: {reason}"));
        return true;
    }
    CONNECT_EVALUATOR.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(RemoteEvaluator::new(url));
        }
        cell.borrow()
            .as_ref()
            .expect("evaluator set above")
            .eval(program, on_result);
    });
    true
}

/// Connect eval that STREAMS per-step metrics into `loss_trace` when
/// the program contains a train block -- feeding the live loss panel --
/// and takes the plain JSON path otherwise (which carries the 3D-viz
/// payload the stream's `done` frame does not). Same contract as
/// [`connect_eval`]: true when it took the eval. Metric frames land in
/// THIS eval's generation, so concurrent evals never intermix curves.
pub fn connect_eval_auto(program: &str, on_result: ResultCb) -> bool {
    if !crate::connect_guard::program_streams_metrics(program) {
        return connect_eval(program, on_result);
    }
    let Some(url) = current_connect_url_from_window() else {
        return false;
    };
    if let Some(reason) = crate::connect_guard::connect_blocked_reason() {
        on_result(format!("error: {reason}"));
        return true;
    }
    let gen_id = crate::telemetry_trace::current_gen();
    CONNECT_EVALUATOR.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(RemoteEvaluator::new(url));
        }
        let ev = cell.borrow();
        let evaluator = ev.as_ref().expect("evaluator set above");
        crate::eval_wasm_stream::stream_into_loss_trace(evaluator, program, gen_id, on_result);
    });
    true
}

impl Evaluator for RemoteEvaluator {
    fn eval(&self, program: &str, on_result: ResultCb) {
        let base = self.base_url().to_string();
        let state = self.state_handle();
        let program = program.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_eval(&base, &state, &program).await;
            on_result(result);
        });
    }
    fn clear(&self) {
        self.clear_state();
    }
}

impl RemoteEvaluator {
    /// Browser streaming-eval. Spawns a local future that POSTs
    /// to `/eval_stream`, reads the response body as a UTF-8
    /// stream, parses SSE frames, fires `on_metric` per metric,
    /// and finally fires `on_result` with the terminal outcome.
    pub fn eval_stream(&self, program: &str, on_metric: MetricCb, on_result: StreamCb) {
        let base = self.base_url().to_string();
        let state = self.state_handle();
        let program = program.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let outcome = wasm_eval_stream(&base, &state, &program, on_metric).await;
            on_result(outcome);
        });
    }

    /// Browser cancel: fire `/v1/sessions/<id>/cancel`. No-op
    /// when no session has been minted yet.
    pub fn cancel(&self) {
        let Some(session_id) = self.current_session_id() else {
            return;
        };
        let Some(token) = self.current_token() else {
            return;
        };
        let url = format!(
            "{}/v1/sessions/{}/cancel",
            self.base_url().trim_end_matches('/'),
            session_id
        );
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(req) = gloo::net::http::Request::post(&url)
                .header("Authorization", &format!("Bearer {token}"))
                .body(String::new())
            {
                let _ = req.send().await;
            }
        });
    }

    /// Browser viz fetch: `GET /v1/viz/<id>` with bearer auth.
    /// Returns `(bytes, content_type)` so the caller can render
    /// an `<img>` (for `image/*`) or an `<iframe srcdoc>` (for
    /// `text/html`). Native equivalent lives in `eval.rs`.
    pub async fn fetch_viz_async(
        &self,
        viz_url: &str,
        bearer: &str,
    ) -> Result<(Vec<u8>, String), String> {
        let url = crate::eval::resolve_viz_url(self.base_url(), viz_url);
        let resp = gloo::net::http::Request::get(&url)
            .header("Authorization", &format!("Bearer {bearer}"))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.ok() {
            return Err(format!("fetch_viz: {}", resp.status()));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let bytes = resp.binary().await.map_err(|e| e.to_string())?;
        Ok((bytes, content_type))
    }
}
