//! A `u:` call must cost O(names the call writes), not O(all globals)
//! (microgpt-mlpl issue e: a 2.28M-element global made every call 0.64 ms,
//! so scripts `expunge`d big globals before hot loops). The frame keeps its
//! exact scoping semantics: locals and rebound params vanish on return,
//! across every value kind, through errors and recursion, while explicit
//! global writes and optimizer updates persist.

use std::time::{Duration, Instant};

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, EvalError> {
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), env)?;
    }
    Ok(last)
}

fn nums(env: &mut Environment, src: &str) -> Vec<f64> {
    match run(env, src) {
        Ok(Value::Array(a)) => a.data().to_vec(),
        other => panic!("{src}: {other:?}"),
    }
}

fn time_calls(env: &mut Environment) -> Duration {
    let start = Instant::now();
    run(env, "repeat 1000 { u:ts() }").unwrap();
    start.elapsed()
}

#[test]
fn call_cost_does_not_scale_with_big_globals() {
    let mut env = Environment::new();
    run(&mut env, "def u:ts() { \"tiny body\"; 6 }").unwrap();
    let small = time_calls(&mut env);
    run(&mut env, "big = zeros([2000000])").unwrap();
    let big = time_calls(&mut env);
    // Deep-cloning 16 MB per call made `big` ~100x `small`; generous
    // headroom for a noisy debug build.
    assert!(
        big < small * 5 + Duration::from_millis(50),
        "small {small:?} vs big {big:?}"
    );
}

#[test]
fn locals_and_params_vanish_on_return() {
    let mut env = Environment::new();
    run(
        &mut env,
        "x = [1, 2]\ndef u:f(x) { \"shadow\"; y = x * 10; y }",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "u:f([3])"), vec![30.0]);
    assert_eq!(nums(&mut env, "x"), vec![1.0, 2.0]);
    assert!(run(&mut env, "y").is_err(), "local y leaked");
}

#[test]
fn a_local_of_another_kind_restores_the_global() {
    let mut env = Environment::new();
    run(
        &mut env,
        "m = linear(2, 2, 0)\nn = [5]\ndef u:g() { \"shadow kinds\"; m = [1]; n = \"text\"; 0 }\nu:g()",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "shape(apply(m, [[1, 1]]))"), vec![1.0, 2.0]);
    assert_eq!(nums(&mut env, "n"), vec![5.0]);
}

#[test]
fn an_error_inside_a_call_still_unwinds_its_writes() {
    let mut env = Environment::new();
    run(
        &mut env,
        "x = [1]\ndef u:bad() { \"fails\"; x = [99]; undefined_fn(1) }",
    )
    .unwrap();
    assert!(run(&mut env, "u:bad()").is_err());
    assert_eq!(nums(&mut env, "x"), vec![1.0]);
}

#[test]
fn recursion_and_nesting_keep_each_frames_bindings() {
    let mut env = Environment::new();
    run(
        &mut env,
        "def u:fact(n) { \"factorial\"; if n < 2 { 1 } else { n * u:fact(n - 1) } }\n\
         def u:outer(a) { \"nest\"; b = a + 1; c = u:inner(b); a + c }\n\
         def u:inner(a) { \"inner\"; a * 100 }",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "u:fact(5)"), vec![120.0]);
    assert_eq!(nums(&mut env, "u:outer(1)"), vec![201.0]);
    assert!(run(&mut env, "b").is_err());
}

#[test]
fn global_set_and_optimizer_writes_persist() {
    let mut env = Environment::new();
    run(
        &mut env,
        "g = [0]\ndef u:set() { \"global write\"; global_set(\"g\", [7]); 0 }\nu:set()\n\
         W = param[1]\nW = [2]\n\
         def u:step() { \"train\"; momentum_sgd(reduce_add(W * W), W, 0.1, 0.0) }\nu:step()",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "g"), vec![7.0]);
    assert_eq!(nums(&mut env, "W"), vec![1.6]);
}
