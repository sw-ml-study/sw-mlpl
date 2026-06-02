//! Regression for issue #6 / C1: a user-function call must get a fresh
//! local scope -- locals created in the body must NOT leak into the
//! caller or sibling frames, so recursion that reads a local after a
//! recursive call is correct.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Vec<f64> {
    let mut env = Environment::new();
    eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env)
        .unwrap()
        .data()
        .to_vec()
}

#[test]
fn recursion_does_not_clobber_caller_locals() {
    // `keep = n` then recurse then return `keep`: each call must see its
    // OWN keep, not the deepest frame's.
    let r = eval(
        "def u:f(n) { if gt(n, 0) { keep = n; tmp = u:f(n - 1); keep } else { 0 } }\n\
[u:f(3), u:f(5), u:f(1)]\n",
    );
    assert_eq!(r, vec![3.0, 5.0, 1.0]);
}

#[test]
fn recursive_factorial_is_correct() {
    let r = eval(
        "def u:fact(n) { if gt(n, 1) { n * u:fact(n - 1) } else { 1 } }\n\
[u:fact(1), u:fact(4), u:fact(5)]\n",
    );
    assert_eq!(r, vec![1.0, 24.0, 120.0]);
}

#[test]
fn function_local_does_not_leak_to_caller() {
    // `tmp` is local to u:g; after the call the caller's own `tmp` (if
    // any) is untouched and `tmp` is not defined at top level.
    let r = eval(
        "def u:g(x) { tmp = x * 10; tmp + 1 }\n\
tmp = 7\n\
y = u:g(2)\n\
[y, tmp]\n",
    );
    assert_eq!(
        r,
        vec![21.0, 7.0],
        "u:g's local tmp must not overwrite the outer tmp"
    );
}
