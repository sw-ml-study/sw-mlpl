//! Record field reads inside `grad` are constant leaves (demo-decision-model
//! Q5): records are data -- a param is identified by name, never by a record
//! field -- so `batch.ids` / `batch.wmask` read eagerly and the gradient
//! flows through the surrounding ops. A record-valued `u:` argument binds
//! for the body's field reads. The Q5 probe written with a `{ids, wmask}`
//! record must equal the plain-array gradient exactly.

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

const Q5: &str = "E = param[10, 4]\nE = randn(1, [10, 4])\n\
                  ids = reshape([1, 2, 3, 4], [2, 2])\nwm = fill([2, 2, 1], 0.5)\n\
                  batch = {ids: ids, wmask: wm}\n\
                  def u:pool(ids, wmask) { \"arrays\"; reduce_add(reduce_add(gather_rows(E, ids) * wmask, 1)) }\n\
                  def u:pool_rec(b) { \"record\"; reduce_add(reduce_add(gather_rows(E, b.ids) * b.wmask, 1)) }";

#[test]
fn q5_record_argument_matches_the_plain_array_gradient() {
    let mut env = Environment::new();
    run(&mut env, Q5).unwrap();
    let plain = nums(&mut env, "grad(u:pool(ids, wm), E)");
    let record = nums(&mut env, "grad(u:pool_rec(batch), E)");
    assert_eq!(plain, record);
    assert!(plain.iter().any(|v| *v != 0.0));
}

#[test]
fn field_reads_at_top_level_and_through_nested_calls() {
    let mut env = Environment::new();
    run(&mut env, Q5).unwrap();
    let plain = nums(&mut env, "grad(u:pool(ids, wm), E)");
    let top = nums(
        &mut env,
        "grad(reduce_add(reduce_add(gather_rows(E, batch.ids) * batch.wmask, 1)), E)",
    );
    assert_eq!(plain, top);
    run(&mut env, "def u:outer(b) { \"nested\"; u:pool_rec(b) * 2 }").unwrap();
    let nested = nums(&mut env, "grad(u:outer(batch), E)");
    assert_eq!(nested, plain.iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

#[test]
fn a_record_param_shadows_a_traced_name_and_does_not_leak() {
    let mut env = Environment::new();
    run(&mut env, Q5).unwrap();
    // `ids` is both a global array and u:pool_rec2's record param.
    run(
        &mut env,
        "def u:pool_rec2(ids) { \"shadow\"; reduce_add(reduce_add(gather_rows(E, ids.ids) * ids.wmask, 1)) }",
    )
    .unwrap();
    let plain = nums(&mut env, "grad(u:pool(ids, wm), E)");
    assert_eq!(nums(&mut env, "grad(u:pool_rec2(batch), E)"), plain);
    assert_eq!(nums(&mut env, "ids"), vec![1.0, 2.0, 3.0, 4.0]);
}
