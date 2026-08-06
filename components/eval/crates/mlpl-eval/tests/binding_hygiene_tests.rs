//! Cross-kind binding hygiene (mlplunit sequencing bug,
//! 2026-08-05): a fresh binding must SHADOW the old kind in every
//! value table, and a u: call frame must restore the RESULTS
//! table like every other table.

use mlpl_array::DenseArray;
use mlpl_eval::Environment;

fn eval_in(env: &mut Environment, src: &str) -> Result<DenseArray, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program(&stmts, env).map_err(|e| e.to_string())
}

#[test]
fn rebinding_across_kinds_shadows_everywhere() {
    let mut env = Environment::new();
    assert_eq!(
        eval_in(&mut env, "x = \"hi\"; x = [1, 2]; x + 1")
            .unwrap()
            .data(),
        &[2.0, 3.0]
    );
    assert_eq!(
        eval_in(&mut env, "r = ok(1); r = [5]; r + 1")
            .unwrap()
            .data(),
        &[6.0]
    );
    assert_eq!(
        eval_in(&mut env, "m = {a: 1}; m = 7; m + 1")
            .unwrap()
            .data(),
        &[8.0]
    );
}

#[test]
fn result_arguments_do_not_leak_out_of_the_call_frame() {
    // The mlplunit fixture shape: a helper taking ok(record)
    // arguments must not corrupt a LATER helper that reuses the
    // same parameter names for arrays.
    let mut env = Environment::new();
    let out = eval_in(
        &mut env,
        "def u:match(actual, expected) { equal(actual, expected) }\n\
         def u:approx(actual, expected, tol, message) { lt(reduce_add(abs(actual - expected)), tol) }\n\
         def u:outer() {\n\
           a = u:match(ok({value: 42}), ok({value: 42}))\n\
           b = u:approx([1, 2.001], [1, 2], 0.01, \"post-equality array arithmetic\")\n\
           a + b\n\
         }\n\
         u:outer()",
    )
    .unwrap();
    assert_eq!(out.data(), &[2.0], "both helpers succeed in sequence");
}

#[test]
fn caller_result_binding_is_restored_after_a_call() {
    let mut env = Environment::new();
    let out = eval_in(
        &mut env,
        "r = ok(41)\n\
         def u:shadow(r) { unwrap(r) * 0 }\n\
         u:shadow(ok(1))\n\
         unwrap(r) + 1",
    )
    .unwrap();
    assert_eq!(out.data(), &[42.0], "caller's r survives the callee's r");
}
