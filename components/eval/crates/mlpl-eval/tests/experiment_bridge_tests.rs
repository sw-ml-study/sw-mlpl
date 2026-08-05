//! The experiment -> array bridge: `param_count(m)` (the size
//! axis of a quality-vs-size frontier) and
//! `experiment_metric("name")` (one recorded metric across the
//! in-memory experiment log, in run order, skipping runs that
//! did not record it -- design resolution 2).

use mlpl_array::DenseArray;
use mlpl_eval::Environment;

fn eval_in(env: &mut Environment, src: &str) -> Result<DenseArray, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program(&stmts, env).map_err(|e| e.to_string())
}

#[test]
fn param_count_totals_every_trainable_array() {
    let mut env = Environment::new();
    // 2*4 + 4 + 4*2 + 2 = 22
    let n = eval_in(
        &mut env,
        "m = chain(linear(2, 4, 0), relu_layer(), linear(4, 2, 1)); param_count(m)",
    )
    .unwrap();
    assert_eq!(n.data(), &[22.0]);
}

#[test]
fn experiment_metric_collects_in_run_order_and_skips_gaps() {
    let mut env = Environment::new();
    // Run "a" predates the metric: a metric first recorded in a
    // later run is absent from earlier records, and those runs are
    // skipped. (Once bound, a *_metric var lingers in the session
    // and is recorded by every later run -- recording semantics.)
    eval_in(&mut env, "experiment \"a\" { other_metric = 9 }").unwrap();
    eval_in(&mut env, "experiment \"b\" { loss_metric = 3 }").unwrap();
    eval_in(&mut env, "experiment \"c\" { loss_metric = 1 }").unwrap();
    let v = eval_in(&mut env, "experiment_metric(\"loss_metric\")").unwrap();
    assert_eq!(v.data(), &[3.0, 1.0], "run order, gap skipped");
    let empty = eval_in(&mut env, "experiment_metric(\"nope_metric\")").unwrap();
    assert_eq!(empty.shape().dims(), &[0]);
}

#[test]
fn the_frontier_pipeline_composes() {
    let mut env = Environment::new();
    eval_in(
        &mut env,
        "experiment \"w2\" { m = chain(linear(2, 2, 0), linear(2, 2, 1)); \
         loss_metric = 0.9; params_metric = param_count(m) }",
    )
    .unwrap();
    eval_in(
        &mut env,
        "experiment \"w8\" { m = chain(linear(2, 8, 2), linear(8, 2, 3)); \
         loss_metric = 0.3; params_metric = param_count(m) }",
    )
    .unwrap();
    let mask = eval_in(
        &mut env,
        "n = tally(experiment_metric(\"loss_metric\")); \
         P = concat(reshape(experiment_metric(\"params_metric\"), [n, 1]), \
                    reshape(experiment_metric(\"loss_metric\"), [n, 1]), 1); \
         pareto_front(P, [-1, -1])",
    )
    .unwrap();
    // Both runs sit on the frontier: fewer params vs lower loss.
    assert_eq!(mask.data(), &[1.0, 1.0]);
}

#[test]
fn bridge_errors_tutor() {
    let mut env = Environment::new();
    let e = eval_in(&mut env, "param_count(5)").unwrap_err();
    assert!(e.contains("model"), "{e}");
    let e = eval_in(&mut env, "x = 1; experiment_metric(x)").unwrap_err();
    assert!(e.contains("string literal"), "{e}");
}
