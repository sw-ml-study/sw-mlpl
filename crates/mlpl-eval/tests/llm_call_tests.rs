//! Saga 19 step 001: `llm_call(url, prompt, model)`
//! language-level builtin.
//!
//! Pure HTTP path lives in `mlpl-runtime`; eval-side
//! dispatch evaluates string args, calls the runtime
//! helper, and wraps the reply in `Value::Str`. Tests
//! drive a `mockito` server so no real Ollama needed.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, mlpl_eval::EvalError> {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program_value(&stmts, env)
}

fn run_ok(env: &mut Environment, src: &str) -> Value {
    run(env, src).expect("eval")
}

fn as_str(v: Value) -> String {
    match v {
        Value::Str(s) => s,
        other => panic!("expected Value::Str, got {other:?}"),
    }
}

#[test]
fn llm_call_happy_path_returns_response_field() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/generate")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::PartialJsonString(
            r#"{"model":"llama3.2","prompt":"say hi","stream":false}"#.into(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"response":"hello"}"#)
        .create();

    let mut env = Environment::new();
    let src = format!(r#"llm_call("{}", "say hi", "llama3.2")"#, server.url());
    let v = run_ok(&mut env, &src);
    assert_eq!(as_str(v), "hello");
    mock.assert();
}

#[test]
fn llm_call_appends_api_generate_when_missing() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/generate")
        .with_status(200)
        .with_body(r#"{"response":"ok"}"#)
        .create();

    let mut env = Environment::new();
    // Pass the bare base URL with a trailing slash --
    // `resolve_url` should strip the slash AND append
    // `/api/generate`.
    let src = format!(
        r#"llm_call("{}/", "x", "m")"#,
        server.url().trim_end_matches('/')
    );
    let v = run_ok(&mut env, &src);
    assert_eq!(as_str(v), "ok");
    mock.assert();
}

#[test]
fn llm_call_does_not_double_append_full_url() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/api/generate")
        .with_status(200)
        .with_body(r#"{"response":"ok"}"#)
        .create();

    let mut env = Environment::new();
    // Pass the full endpoint URL; should NOT become
    // `/api/generate/api/generate`.
    let src = format!(
        r#"llm_call("{}/api/generate", "x", "m")"#,
        server.url().trim_end_matches('/')
    );
    let v = run_ok(&mut env, &src);
    assert_eq!(as_str(v), "ok");
    mock.assert();
}

#[test]
fn llm_call_non_2xx_errors_with_status_and_body_preview() {
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/api/generate")
        .with_status(500)
        .with_body("boom: model not found")
        .create();

    let mut env = Environment::new();
    let src = format!(r#"llm_call("{}", "x", "m")"#, server.url());
    let err = run(&mut env, &src).expect_err("expected non-2xx error");
    let msg = format!("{err}");
    assert!(
        msg.contains("llm_call"),
        "msg should mention llm_call: {msg}"
    );
    assert!(msg.contains("500"), "msg should mention status 500: {msg}");
    assert!(
        msg.contains("boom"),
        "msg should include body preview 'boom': {msg}"
    );
}

#[test]
fn llm_call_missing_response_field_errors() {
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/api/generate")
        .with_status(200)
        .with_body(r#"{"foo":"bar"}"#)
        .create();

    let mut env = Environment::new();
    let src = format!(r#"llm_call("{}", "x", "m")"#, server.url());
    let err = run(&mut env, &src).expect_err("expected missing-field error");
    let msg = format!("{err}");
    assert!(
        msg.contains("response"),
        "msg should mention `response`: {msg}"
    );
}

#[test]
fn llm_call_invalid_json_errors() {
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/api/generate")
        .with_status(200)
        .with_body("this is not json")
        .create();

    let mut env = Environment::new();
    let src = format!(r#"llm_call("{}", "x", "m")"#, server.url());
    let err = run(&mut env, &src).expect_err("expected json parse error");
    let msg = format!("{err}");
    assert!(
        msg.contains("llm_call"),
        "msg should mention llm_call: {msg}"
    );
}

#[test]
fn llm_call_wrong_arity_errors() {
    let mut env = Environment::new();
    let err = run(&mut env, r#"llm_call("http://h", "x")"#).expect_err("arity");
    let msg = format!("{err}");
    assert!(
        msg.contains("llm_call") && msg.contains("3"),
        "msg should mention llm_call + expected arity 3: {msg}"
    );
}

#[test]
fn llm_call_non_string_argument_errors() {
    let mut env = Environment::new();
    let err =
        run(&mut env, r#"llm_call("http://h", iota(3), "m")"#).expect_err("non-string prompt");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("string"),
        "msg should mention expected string: {msg}"
    );
}
