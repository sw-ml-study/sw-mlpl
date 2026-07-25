//! Browser streaming-eval transport (connect-telemetry step 002).
//!
//! POSTs to `/v1/sessions/<id>/eval_stream` and reads the SSE body
//! INCREMENTALLY via the fetch `ReadableStream`, feeding chunks through
//! `SseFeed` so `event: metric` frames fire as they arrive -- this is
//! what makes the live loss panel live. (The old implementation pulled
//! the whole body first, so every frame batched at body end.)

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use wasm_bindgen::JsCast;

use crate::eval::{MetricCb, RemoteEvaluator, RemoteSession, ResultCb, StreamOutcome};
use crate::eval_sse::{SseFeed, parse_sse_stream};
use crate::eval_wasm_helpers::wasm_create_session;

/// Stream `program` on `evaluator`, landing each metric frame in
/// generation `gen_id`'s loss trace and the terminal outcome on
/// `on_result` as a display string. The glue between
/// `connect_eval_auto` and the live loss panel.
pub(crate) fn stream_into_loss_trace(
    evaluator: &RemoteEvaluator,
    program: &str,
    gen_id: u32,
    on_result: ResultCb,
) {
    evaluator.eval_stream(
        program,
        Box::new(move |m| crate::loss_trace::push(gen_id, &m.name, m.value)),
        Box::new(move |outcome| on_result(outcome.into_display().0)),
    );
}

pub(crate) async fn wasm_eval_stream(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    mut on_metric: MetricCb,
) -> StreamOutcome {
    match wasm_eval_stream_inner(base_url, state, program, &mut on_metric).await {
        Ok(outcome) => outcome,
        Err(message) => StreamOutcome::Error { message },
    }
}

// Inner helper returning `Result` so `?` collapses the fallible
// `gloo::net::http` steps into one error path.
async fn wasm_eval_stream_inner(
    base_url: &str,
    state: &RefCell<Option<RemoteSession>>,
    program: &str,
    on_metric: &mut MetricCb,
) -> Result<StreamOutcome, String> {
    if state.borrow().is_none() {
        *state.borrow_mut() = Some(wasm_create_session(base_url).await?);
    }
    let s = state.borrow().as_ref().expect("session").clone();
    let resp = post_stream_request(base_url, &s, program).await?;
    if resp.ok() {
        read_sse_stream(&resp, on_metric).await
    } else {
        Ok(StreamOutcome::Error {
            message: resp.text().await.unwrap_or_default(),
        })
    }
}

/// POST the `/eval_stream` request (bearer-authed JSON body).
async fn post_stream_request(
    base_url: &str,
    s: &RemoteSession,
    program: &str,
) -> Result<gloo::net::http::Response, String> {
    let url = format!(
        "{}/v1/sessions/{}/eval_stream",
        base_url.trim_end_matches('/'),
        s.session_id
    );
    gloo::net::http::Request::post(&url)
        .header("Authorization", &format!("Bearer {}", s.token))
        .header("Content-Type", "application/json")
        .body(serde_json::json!({"program": program}).to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())
}

/// Drive the response `ReadableStream` chunk by chunk through an
/// `SseFeed`. Falls back to a whole-body read (frames batch at body
/// end) when the browser exposes no stream.
async fn read_sse_stream(
    resp: &gloo::net::http::Response,
    on_metric: &mut MetricCb,
) -> Result<StreamOutcome, String> {
    let Some(stream) = resp.body() else {
        let text = resp.text().await.unwrap_or_default();
        return Ok(parse_sse_stream(std::io::Cursor::new(text), on_metric));
    };
    let reader: web_sys::ReadableStreamDefaultReader = stream
        .get_reader()
        .dyn_into()
        .map_err(|_| "no stream reader".to_string())?;
    let mut feed = SseFeed::default();
    loop {
        let chunk = wasm_bindgen_futures::JsFuture::from(reader.read())
            .await
            .map_err(|_| "stream read failed".to_string())?;
        let Some(bytes) = chunk_bytes(&chunk)? else {
            let message = "stream ended without terminal frame".to_string();
            return Ok(StreamOutcome::Error { message });
        };
        if let Some(outcome) = feed.push(&bytes, on_metric) {
            return Ok(outcome);
        }
    }
}

/// One `reader.read()` result -> its `Uint8Array` bytes, or `None`
/// when the stream reports done.
fn chunk_bytes(chunk: &wasm_bindgen::JsValue) -> Result<Option<Vec<u8>>, String> {
    let done = js_sys::Reflect::get(chunk, &"done".into())
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if done {
        return Ok(None);
    }
    let value = js_sys::Reflect::get(chunk, &"value".into())
        .map_err(|_| "stream chunk missing value".to_string())?;
    Ok(Some(js_sys::Uint8Array::new(&value).to_vec()))
}
