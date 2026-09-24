//! Built-in layer weight access (microgpt-mlpl; also pretrained-weight
//! loading): `params(model)`, `get_param` / `set_param` by ROLE name,
//! `rms_norm(dim, {eps})` (any rank >= 2), and a bias-free
//! `linear(in, out, seed, {bias: 0})`.

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

fn err_text(env: &mut Environment, src: &str) -> String {
    match run(env, src) {
        Ok(v) => panic!("expected an error from {src}, got {v:?}"),
        Err(e) => e.to_string(),
    }
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

#[test]
fn params_lists_a_models_parameter_names() {
    let mut env = Environment::new();
    run(
        &mut env,
        "m = chain(linear(2, 3, 0), relu_layer(), linear(3, 1, 1))",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "list_len(params(m))"), vec![4.0]);
    match run(&mut env, "params(m)").unwrap() {
        Value::StrList { items } => assert_eq!(items.len(), 4),
        other => panic!("expected a string list, got {other:?}"),
    }
}

#[test]
fn set_param_then_apply_matches_the_hand_written_forward() {
    let mut env = Environment::new();
    run(&mut env, "L = linear(2, 3, 0, {bias: 0})").unwrap();
    assert_eq!(nums(&mut env, "list_len(params(L))"), vec![1.0]);
    run(&mut env, "set_param(L, \"W\", [[1, 2, 3], [4, 5, 6]])").unwrap();
    assert_eq!(nums(&mut env, "apply(L, [[1, 1]])"), vec![5.0, 7.0, 9.0]);
    assert_eq!(
        nums(&mut env, "get_param(L, \"W\")"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn set_param_checks_shape_and_role() {
    let mut env = Environment::new();
    run(&mut env, "L = linear(2, 3, 0)").unwrap();
    let msg = err_text(&mut env, "set_param(L, \"W\", zeros([3, 2]))");
    assert!(msg.contains("[2, 3]"), "names the expected shape: {msg}");
    let msg = err_text(&mut env, "get_param(L, \"Wq\")");
    assert!(
        msg.contains("W") && msg.contains("b"),
        "lists the roles: {msg}"
    );
}

#[test]
fn chains_address_the_kth_layer_owning_a_role() {
    let mut env = Environment::new();
    run(&mut env, "m = chain(linear(2, 3, 0), linear(3, 1, 1))").unwrap();
    assert_eq!(nums(&mut env, "shape(get_param(m, \"W\"))"), vec![2.0, 3.0]);
    assert_eq!(
        nums(&mut env, "shape(get_param(m, \"W\", 1))"),
        vec![3.0, 1.0]
    );
}

#[test]
fn set_param_attention_weights_then_train() {
    let mut env = Environment::new();
    run(
        &mut env,
        "A = causal_attention(4, 2, 7)\nset_param(A, \"Wq\", zeros([4, 4]))\n\
         X = randn(1, [5, 4])\nT = randn(2, [5, 4])",
    )
    .unwrap();
    assert_eq!(
        nums(&mut env, "reduce_add(abs(get_param(A, \"Wq\")))"),
        vec![0.0]
    );
    run(
        &mut env,
        "train 20 { adam(mean((apply(A, X) - T) * (apply(A, X) - T)), A, 0.01, 0.9, 0.999, 0.00000001) }",
    )
    .unwrap();
    let l = nums(&mut env, "last_losses");
    assert!(l[19] < l[0], "{} -> {}", l[0], l[19]);
    assert!(nums(&mut env, "reduce_add(abs(get_param(A, \"Wq\")))")[0] > 0.0);
}

#[test]
fn set_param_inside_a_user_function_persists() {
    let mut env = Environment::new();
    run(
        &mut env,
        "L = linear(1, 1, 0)\ndef u:load() { \"load weights\"; set_param(L, \"W\", [[7]]); 0 }\nu:load()",
    )
    .unwrap();
    assert_eq!(nums(&mut env, "get_param(L, \"W\")"), vec![7.0]);
}

#[test]
fn rms_norm_eps_option_and_rank3_input() {
    let mut env = Environment::new();
    run(&mut env, "N1 = rms_norm(2, {eps: 1})\nN0 = rms_norm(2)").unwrap();
    let got = nums(&mut env, "apply(N1, [[1, 1]])");
    assert!(close(&got, &[0.5_f64.sqrt(), 0.5_f64.sqrt()]), "{got:?}");
    let got = nums(&mut env, "apply(N0, [[1, 1]])");
    assert!(close(&got, &[1.0 / (1.0_f64 + 1e-8).sqrt(); 2]), "{got:?}");
    // [B, T, d] = [2, 1, 2]: each row normalized independently, eager and
    // on the tape (grad through a rank-3 input).
    let got = nums(
        &mut env,
        "shape(apply(N0, reshape([3, 4, 3, 4], [2, 1, 2])))",
    );
    assert_eq!(got, vec![2.0, 1.0, 2.0]);
    run(
        &mut env,
        "X3 = param[2, 1, 2]\nX3 = reshape([3, 4, 1, 2], [2, 1, 2])",
    )
    .unwrap();
    let g = nums(&mut env, "grad(reduce_add(apply(N0, X3) * [1, 0]), X3)");
    assert_eq!(g.len(), 4);
    let msg = err_text(&mut env, "rms_norm(2, {epsilon: 1})");
    assert!(
        msg.contains("unknown option") && msg.contains("eps"),
        "{msg}"
    );
}

#[test]
fn bias_free_linear_trains_under_adam() {
    let mut env = Environment::new();
    run(
        &mut env,
        "L = linear(2, 1, 3, {bias: 0})\nX = [[1, 2], [3, 4]]\nY = [[1], [2]]\n\
         train 30 { adam(mean((apply(L, X) - Y) * (apply(L, X) - Y)), L, 0.05, 0.9, 0.999, 0.00000001) }",
    )
    .unwrap();
    let l = nums(&mut env, "last_losses");
    assert!(l[29] < l[0], "{} -> {}", l[0], l[29]);
}
