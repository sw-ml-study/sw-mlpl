fn run(src: &str) -> String {
    let mut env = mlpl_eval::Environment::new();
    let toks = mlpl_parser::lex(src).unwrap();
    let stmts = mlpl_parser::parse(&toks).unwrap();
    match mlpl_eval::eval_program(&stmts, &mut env) {
        Ok(v) => format!("{v}"),
        Err(e) => format!("ERROR: {e}"),
    }
}

#[test]
fn simple_call() {
    assert_eq!(run("def u:double(x) { x * 2 }\nu:double(21)"), "42");
}

#[test]
fn multi_param() {
    assert_eq!(run("def u:add(a, b) { a + b }\nu:add(10, 32)"), "42");
}

#[test]
fn zero_param() {
    assert_eq!(run("def u:greet() { 42 }\nu:greet()"), "42");
}

#[test]
fn return_early() {
    let src = "def u:f(x) { if gt(x, 0) { return x * 10 } else { return 0 - x }; 999 }\nu:f(5)";
    assert_eq!(run(src), "50");
}

#[test]
fn return_negative_branch() {
    let src = "def u:f(x) { if gt(x, 0) { return x * 10 } else { return 0 - x }; 999 }\nu:f(-3)";
    assert_eq!(run(src), "3");
}

#[test]
fn recursion_fibonacci() {
    let src = "def u:fib(n) { if gt(n, 1) { u:fib(n - 1) + u:fib(n - 2) } else { n } }\nu:fib(10)";
    assert_eq!(run(src), "55");
}

#[test]
fn lexical_scoping_reads_outer() {
    assert_eq!(run("x = 100\ndef u:f(y) { x + y }\nu:f(5)"), "105");
}

#[test]
fn parameter_shadow_does_not_leak() {
    assert_eq!(run("x = 100\ndef u:f(x) { x * 2 }\nu:f(7)\nx"), "100");
}

#[test]
fn arity_mismatch() {
    let out = run("def u:f(a, b) { a + b }\nu:f(1)");
    assert!(out.contains("ERROR"), "expected arity error, got: {out}");
}

#[test]
fn undefined_fn() {
    let out = run("u:nonexistent(1)");
    assert!(out.contains("ERROR"), "expected error, got: {out}");
}
