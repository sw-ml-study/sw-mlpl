//! Compiler parity for the readable-scripts surface: `format`, `write`,
//! variadic `str_concat`, `and` / `or` / `not` (with the interpreter's
//! scalar short-circuit), and record destructuring.

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let stmts = parse(&lex(src).expect("lex ok")).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

#[test]
fn format_write_and_concat_lower_to_variadic_runtime_calls() {
    let s = lower_src("format(\"{} = {:.2f}\", \"x\", 1.5)").unwrap();
    assert!(s.contains("format (vec !"), "{s}");
    let s = lower_src("write(\"a\", 1, \"b\")").unwrap();
    assert!(s.contains("write (vec !"), "{s}");
    let s = lower_src("str_concat(\"a\", \"b\", \"c\", \"d\")").unwrap();
    assert!(s.contains("str_concat (vec !"), "{s}");
}

#[test]
fn too_few_arguments_is_a_compile_diagnostic() {
    for src in ["str_concat(\"a\")", "format()", "write()"] {
        match lower_src(src) {
            Err(LowerError::Unsupported(msg)) => assert!(msg.contains('/'), "{src}: {msg}"),
            other => panic!("{src}: expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
fn scalar_and_or_short_circuit() {
    let s = lower_src("x = 0\nx and exit(3)").unwrap();
    assert!(s.contains("rank () == 0"), "short-circuit guard: {s}");
    let s = lower_src("not 1").unwrap();
    assert!(s.contains("apply_binop"), "{s}");
}

#[test]
fn destructuring_lowers_to_field_reads() {
    let s = lower_src("{a, b: x} = {a: 1, b: 2}\na + x").unwrap();
    assert!(
        s.contains("field (\"a\")") && s.contains("field (\"b\")"),
        "{s}"
    );
    assert!(s.contains("let mut x"), "{s}");
}

#[test]
fn destructuring_inside_user_function() {
    // Compiled user-function parameters are arrays, so the record is
    // built in the body.
    let src = "def u:area(n) {\n  {w, h} = {w: n, h: n + 1}\n  w * h\n}\nu:area(3)";
    let s = lower_src(src).unwrap();
    assert!(s.contains("field (\"w\")"), "{s}");
}

#[test]
fn destructuring_as_a_function_tail_is_a_diagnostic() {
    let src = "def u:f(n) {\n  {a} = {a: n}\n}\nu:f(1)";
    match lower_src(src) {
        Err(LowerError::Unsupported(msg)) => assert!(msg.contains("destructuring"), "{msg}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
