//! Saga 21.5 step 006: connect-mode evaluator-trait tests.
//!
//! The web REPL's `Evaluator` trait has two impls: `WasmEvaluator`
//! (in-process) and `RemoteEvaluator` (REST against a remote
//! `mlpl-serve`). This file unit-tests both impls on a native
//! (non-WASM) target by spinning `mlpl-serve` up in-process on a
//! random loopback port and asserting that the same MLPL
//! program produces the same display string through both paths.
//!
//! `eval.rs` lives inside the `mlpl-web` binary, so the
//! integration test re-includes its source via `#[path]` (same
//! pattern `mlpl-repl` uses for its `connect_stream.rs` tests).
//! `parse_connect_url` is exercised here too; it is pure and
//! does not need a server.

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval.rs"]
mod eval;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_native.rs"]
mod eval_native;
#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_sse.rs"]
mod eval_sse;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_native_stream.rs"]
mod eval_native_stream;

#[allow(dead_code)]
#[path = "../../../../wasm/crates/mlpl-web-eval/src/eval_url.rs"]
mod eval_url;

use std::cell::RefCell;
use std::net::SocketAddr;
use std::rc::Rc;

use eval::{Evaluator, RemoteEvaluator, WasmEvaluator, parse_connect_url};
use mlpl_serve::auth::AuthMode;
use mlpl_serve::server::build_app;

async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_app(AuthMode::Required);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

/// Synchronous helper: invoke `Evaluator::eval` and collect the
/// result string from the callback. Works for the WASM impl
/// (sync callback) and the native REST impl (blocking-then-sync
/// callback) without any async glue.
fn eval_sync(evaluator: &dyn Evaluator, program: &str) -> String {
    let slot = Rc::new(RefCell::new(None::<String>));
    let slot_clone = slot.clone();
    evaluator.eval(
        program,
        Box::new(move |result| {
            *slot_clone.borrow_mut() = Some(result);
        }),
    );
    slot.borrow().clone().expect("callback should have fired")
}

/// `WasmEvaluator` and `RemoteEvaluator` produce the same final
/// display string for `iota(5) + 1`. The contract: both impls
/// route through the same `mlpl_eval::eval_program_value` +
/// `Display` formatting, so the string output matches
/// byte-for-byte.
#[tokio::test(flavor = "multi_thread")]
async fn wasm_and_remote_produce_same_display_for_iota_plus_one() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let wasm =
            WasmEvaluator::from_session(Rc::new(RefCell::new(mlpl_wasm::WasmSession::new())));
        let remote = RemoteEvaluator::new(&base);
        let from_wasm = eval_sync(&wasm, "iota(5) + 1");
        let from_remote = eval_sync(&remote, "iota(5) + 1");
        assert_eq!(
            from_wasm, from_remote,
            "WASM and REST evaluators should produce identical displays"
        );
        // Sanity-check the display: should be a row of digits 1..=5.
        for digit in ['1', '2', '3', '4', '5'] {
            assert!(
                from_wasm.contains(digit),
                "iota(5)+1 display should contain {digit}: {from_wasm}"
            );
        }
    })
    .await;
    result.unwrap();
}

/// `RemoteEvaluator::clear` drops the cached server session so
/// the next eval mints a fresh one. State that was bound before
/// the clear is no longer visible.
#[tokio::test(flavor = "multi_thread")]
async fn remote_clear_drops_session_state() {
    let addr = start_server().await;
    let base = format!("http://{addr}");

    let result = tokio::task::spawn_blocking(move || {
        let remote = RemoteEvaluator::new(&base);
        eval_sync(&remote, "x = iota(3)");
        let before = eval_sync(&remote, "x");
        assert!(
            !before.starts_with("error:"),
            "x should be bound before clear: {before}"
        );
        remote.clear();
        let after = eval_sync(&remote, "x");
        assert!(
            after.contains("undefined"),
            "x should be unbound after clear: {after}"
        );
    })
    .await;
    result.unwrap();
}

/// `RemoteEvaluator` surfaces server-side eval errors as
/// `"error: ..."` strings, matching the `WasmSession::eval`
/// shape so existing call sites can keep their
/// `result.starts_with("error:")` test for the red-text UI.
#[tokio::test(flavor = "multi_thread")]
async fn remote_eval_error_starts_with_error_prefix() {
    let addr = start_server().await;
    let base = format!("http://{addr}");
    let result = tokio::task::spawn_blocking(move || {
        let remote = RemoteEvaluator::new(&base);
        let out = eval_sync(&remote, "undefined_var");
        assert!(
            out.starts_with("error:"),
            "remote error should surface with the same prefix: {out}"
        );
    })
    .await;
    result.unwrap();
}

// ---- parse_connect_url (pure fn) ----

#[test]
fn parse_connect_url_extracts_simple_query() {
    assert_eq!(
        parse_connect_url("?connect=http://localhost:6464"),
        Some("http://localhost:6464".to_string())
    );
}

#[test]
fn parse_connect_url_handles_missing_question_mark() {
    assert_eq!(
        parse_connect_url("connect=http://localhost:6464"),
        Some("http://localhost:6464".to_string())
    );
}

#[test]
fn parse_connect_url_finds_connect_among_other_params() {
    assert_eq!(
        parse_connect_url("?lesson=3&connect=http://x:1&debug=1"),
        Some("http://x:1".to_string())
    );
}

#[test]
fn parse_connect_url_returns_none_when_absent() {
    assert_eq!(parse_connect_url("?lesson=3"), None);
    assert_eq!(parse_connect_url(""), None);
}

#[test]
fn parse_connect_url_returns_none_for_empty_value() {
    assert_eq!(parse_connect_url("?connect="), None);
}

#[test]
fn parse_connect_url_percent_decodes_value() {
    // %3A -> ':'
    assert_eq!(
        parse_connect_url("?connect=http%3A//x%3A6464"),
        Some("http://x:6464".to_string())
    );
}
