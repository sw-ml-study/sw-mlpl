//! Regression: a model bound by `m = chain(...)` / `lora(...)` must
//! be resolvable as a BARE identifier -- not only via `apply(m, x)`.
//!
//! The env-table refactor split models into a sibling table and the
//! bare-`Ident` dispatch lost its `models` case, so `m`, `is_model(m)`,
//! model arithmetic, and any user function that references a global
//! model all failed with `undefined variable: m` even though
//! `apply(m, x)` still worked. This broke the connect MLX demos
//! (tic-tac-toe's `u:play_o` reads the global `m`). These tests pin
//! bare-identifier resolution so the regression can't return silently.

use mlpl_eval::{Environment, Value};

fn eval(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

const BUILD_M: &str = "m = chain(linear(3, 4, 0), relu_layer(), linear(4, 2, 1))";

#[test]
fn bare_model_identifier_resolves_not_undefined() {
    let mut env = Environment::new();
    eval(&mut env, BUILD_M).unwrap();
    // The exact breakage: a bare `m` must not be `undefined variable`.
    let v = eval(&mut env, "m").expect("bare model identifier `m` must resolve");
    assert!(
        matches!(v, Value::Model(_)),
        "bare `m` should be a Value::Model, got {v:?}"
    );
}

#[test]
fn model_arithmetic_reads_the_bound_model() {
    // A bare model name used past `apply` -- here comparing two
    // predictions -- must resolve the model, not error.
    let mut env = Environment::new();
    eval(&mut env, BUILD_M).unwrap();
    let v = eval(
        &mut env,
        "equal(apply(m, zeros([1, 3])), apply(m, zeros([1, 3])))",
    )
    .expect("re-reading global `m` twice must resolve both times");
    assert!(matches!(v, Value::Array(_)));
}

#[test]
fn apply_by_name_still_works() {
    // The path that never broke -- guard it stays working alongside
    // the bare-identifier fix.
    let mut env = Environment::new();
    eval(&mut env, BUILD_M).unwrap();
    let v = eval(&mut env, "apply(m, zeros([1, 3]))").expect("apply(m, x)");
    assert!(
        matches!(v, Value::Array(_)),
        "apply output should be an array"
    );
}

#[test]
fn user_fn_local_shadow_does_not_destroy_global_model() {
    // The tic-tac-toe breaker: `u:encode` binds a LOCAL `m`
    // (`m = eq(board, mover)`) while a global model `m` exists.
    // A per-call frame that failed to snapshot the `models` table
    // let that local assignment's `clear_binding` wipe the global
    // model, so the model vanished after the call (`undefined
    // variable: m`). The global must survive an identically-named
    // local.
    let mut env = Environment::new();
    eval(&mut env, BUILD_M).unwrap();
    eval(
        &mut env,
        "def u:clob(v) { \"local m shadows the global model\"; m = eq(v, 0); reduce_add(m, 0) }",
    )
    .unwrap();
    eval(&mut env, "u:clob([0, 1, 0])").expect("call with a local `m`");
    let v = eval(&mut env, "m").expect("global model must survive a user fn's local `m`");
    assert!(
        matches!(v, Value::Model(_)),
        "global `m` should still be a model, got {v:?}"
    );
    assert!(matches!(
        eval(&mut env, "apply(m, zeros([1, 3]))").unwrap(),
        Value::Array(_)
    ));
}

#[test]
fn user_fn_reading_a_global_model_resolves_it() {
    // Mirrors the tic-tac-toe demo: a user fn references the global
    // model `m` (here through `apply`), which requires the name to
    // resolve inside the call.
    let mut env = Environment::new();
    eval(&mut env, BUILD_M).unwrap();
    eval(
        &mut env,
        "def u:run(x) { \"apply the global model\"; apply(m, x) }",
    )
    .unwrap();
    let v = eval(&mut env, "u:run(zeros([1, 3]))").expect("user fn referencing global `m`");
    assert!(matches!(v, Value::Array(_)));
}
