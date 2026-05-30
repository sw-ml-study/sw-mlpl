//! `print(v)` / `eprint(v)` builtins. Saga 31 step 001.
//!
//! Behavior contract:
//! - `print(v)` writes `v`'s Display form to stdout followed by
//!   a newline.
//! - `eprint(v)` writes the same to stderr.
//! - Both return `v` unchanged so calls compose into expressions
//!   (e.g. `x = print(some_computation)` binds AND shows).
//!
//! Tests here pin the composability contract (return value
//! equals input) via direct in-process eval. The stdout/stderr
//! emission is best verified by spawning the binary against a
//! /tmp script; that surface lands in saga 31 step 003 alongside
//! the args() + CLI passthrough work.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

#[test]
fn print_returns_scalar_unchanged() {
    let mut env = Environment::new();
    let r = run("print(42)", &mut env);
    assert_eq!(r.shape(), &Shape::scalar());
    assert_eq!(r.data(), &[42.0]);
}

#[test]
fn eprint_returns_scalar_unchanged() {
    let mut env = Environment::new();
    let r = run("eprint(42)", &mut env);
    assert_eq!(r.shape(), &Shape::scalar());
    assert_eq!(r.data(), &[42.0]);
}

#[test]
fn print_returns_vector_unchanged() {
    let mut env = Environment::new();
    let r = run("print([1, 2, 3])", &mut env);
    assert_eq!(r.shape().dims(), &[3]);
    assert_eq!(r.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn print_composes_into_binding() {
    // The whole point of returning the input: `x = print(...)`
    // both surfaces the value and binds it for downstream use.
    let mut env = Environment::new();
    let r = run("x = print(reduce_add([1, 2, 3]))\nx * 2", &mut env);
    assert_eq!(r.shape(), &Shape::scalar());
    assert_eq!(r.data(), &[12.0]);
}

#[test]
fn print_rejects_zero_args() {
    let mut env = Environment::new();
    let err = eval_program(&parse(&lex("print()").unwrap()).unwrap(), &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("print") && msg.contains("expects"),
        "expected arity error mentioning print, got: {msg}"
    );
}

#[test]
fn print_is_variadic_and_returns_last() {
    // print/eprint are variadic + space-joined (println-style) so
    // labelled output like `print("count:", n)` works without string
    // concatenation. The return value is the last argument, so
    // single-arg `print(x)` still yields x.
    let mut env = Environment::new();
    let r = run("print(\"count:\", 21, 2)", &mut env);
    assert_eq!(r.shape(), &Shape::scalar());
    assert_eq!(r.data(), &[2.0]);
}
