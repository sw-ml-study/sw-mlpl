//! Shape metadata (`shape`, `rank`, `len`) inside grad on a user function's
//! PARAMETER. These read structure, not values, so they are constant leaves --
//! but they must resolve the argument through the traced scope, not the global
//! env (where a function parameter is unbound).

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Vec<f64>, EvalError> {
    let prog = parse(&lex(src).expect("lex")).expect("parse");
    eval_program_value(&prog, env).map(|v| match v {
        Value::Array(a) => a.data().to_vec(),
        _ => Vec::new(),
    })
}

fn grad_of(def: &str) -> Result<Vec<f64>, EvalError> {
    let mut env = Environment::new();
    for line in [
        "W = param[2, 3]",
        "W = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]",
        def,
    ] {
        run(&mut env, line).expect("setup");
    }
    run(&mut env, "grad(u:g(W), W)")
}

#[test]
fn shape_of_function_parameter_is_a_constant() {
    let g = grad_of("def u:g(a) { reduce_add(a) * take(shape(a), 0, 0) }").expect("grad");
    assert_eq!(g, vec![2.0; 6]);
}

#[test]
fn rank_and_len_of_function_parameter_are_constants() {
    let g = grad_of("def u:g(a) { reduce_add(a) * rank(a) }").expect("rank");
    assert_eq!(g, vec![2.0; 6]);
    let g = grad_of("def u:g(a) { reduce_add(a) * len(a) }").expect("len");
    assert_eq!(g, vec![2.0; 6]);
}

#[test]
fn shape_derived_size_of_parameter_folds_through_locals() {
    // reduce_mul is not on the tape: the whole size folds, and the fold must
    // see the function's local binding for `a`.
    let g = grad_of("def u:g(a) { reduce_add(a) / reduce_mul(shape(a)) }").expect("fold");
    for v in g {
        assert!((v - 1.0 / 6.0).abs() < 1e-12, "{v}");
    }
}
