use mlpl_wasm::WasmSession;

#[test]
fn eval_scalar_arithmetic() {
    let session = WasmSession::new();
    assert_eq!(session.eval("1 + 2"), "3");
    assert_eq!(session.eval("3 * 4"), "12");
    assert_eq!(session.eval("10 / 3"), "3.3333333333333335");
}

#[test]
fn eval_array_ops() {
    let session = WasmSession::new();
    assert_eq!(session.eval("[1, 2, 3] * 10"), "10 20 30");
    assert_eq!(session.eval("[1, 2, 3] + [4, 5, 6]"), "5 7 9");
}

#[test]
fn eval_variable_persistence() {
    let session = WasmSession::new();
    assert_eq!(session.eval("x = 42"), "42");
    assert_eq!(session.eval("x + 1"), "43");
}

#[test]
fn eval_builtin_functions() {
    let session = WasmSession::new();
    assert_eq!(session.eval("iota(5)"), "0 1 2 3 4");
    assert_eq!(session.eval("shape([1,2,3])"), "3");
    assert_eq!(session.eval("reduce_add([1,2,3,4,5])"), "15");
}

#[test]
fn eval_error_returns_error_prefix() {
    let session = WasmSession::new();
    let result = session.eval("foo(1)");
    assert!(
        result.starts_with("error: "),
        "expected error prefix, got: {result}"
    );
}

#[test]
fn eval_parse_error() {
    let session = WasmSession::new();
    let result = session.eval("[1, 2,");
    assert!(
        result.starts_with("error: "),
        "expected error prefix, got: {result}"
    );
}

#[test]
fn eval_matrix_display() {
    let session = WasmSession::new();
    session.eval("x = iota(6)");
    let result = session.eval("reshape(x, [2, 3])");
    assert_eq!(result, "0 1 2\n3 4 5");
}

#[test]
fn eval_empty_input() {
    let session = WasmSession::new();
    assert_eq!(session.eval(""), "");
}

#[test]
fn eval_ml_functions() {
    let session = WasmSession::new();
    assert_eq!(session.eval("sigmoid(0)"), "0.5");
    assert_eq!(session.eval("mean([2, 4, 6])"), "4");
}

#[test]
fn separate_sessions_isolated() {
    let s1 = WasmSession::new();
    let s2 = WasmSession::new();
    s1.eval("x = 100");
    let result = s2.eval("x");
    assert!(
        result.starts_with("error: "),
        "sessions should be isolated, got: {result}"
    );
}

#[test]
fn eval_line_stateless() {
    assert_eq!(mlpl_wasm::eval_line("2 + 3"), "5");
    assert_eq!(mlpl_wasm::eval_line("[1,2] * 3"), "3 6");
}

#[test]
fn eval_string_literal_returns_string() {
    let session = WasmSession::new();
    assert_eq!(session.eval(r#""hello""#), "hello");
    assert_eq!(session.eval(r#""scatter""#), "scatter");
}

#[test]
fn eval_string_in_arithmetic_is_error() {
    let session = WasmSession::new();
    let result = session.eval(r#""x" + 1"#);
    assert!(result.starts_with("error: "));
}
