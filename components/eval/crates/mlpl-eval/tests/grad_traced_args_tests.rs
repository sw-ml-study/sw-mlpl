//! Non-differentiable arguments of differentiable builtins (cross_entropy
//! targets, rotate's shift, pow's exponent, transpose_axes' permutation) must
//! resolve through grad's traced scope, so a user function's PARAMETER works
//! -- not only globals (microgpt-mlpl: `def u:ce(x, y)` failed with
//! "undefined variable: y"). Also: comparison operators inside grad are
//! constant 0/1 masks, exactly like the lt/gt/eq builtins.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Vec<f64>, EvalError> {
    let prog = parse(&lex(src).expect("lex")).expect("parse");
    eval_program_value(&prog, env).map(|v| match v {
        Value::Array(a) => a.data().to_vec(),
        _ => Vec::new(),
    })
}

/// Evaluate setup lines, then return (via-u:-function, inline) results of the
/// two grad expressions; they must agree.
fn both(setup: &[&str], via_fn: &str, inline: &str) -> (Vec<f64>, Vec<f64>) {
    let mut env = Environment::new();
    for line in setup {
        run(&mut env, line).expect("setup");
    }
    let a = run(&mut env, via_fn).expect("grad through u: function");
    let b = run(&mut env, inline).expect("inline grad");
    (a, b)
}

const W33: &[&str] = &[
    "W = param[3, 3]",
    "W = randn(1, [3, 3])",
    "X = randn(2, [2, 3])",
];

#[test]
fn cross_entropy_targets_bound_to_a_function_parameter() {
    let mut setup = W33.to_vec();
    setup.push("def u:ce(x, y) { cross_entropy(matmul(x, W), y) }");
    let (a, b) = both(
        &setup,
        "grad(u:ce(X, [0, 2]), W)",
        "grad(cross_entropy(matmul(X, W), [0, 2]), W)",
    );
    assert_eq!(a, b);
}

#[test]
fn rotate_shift_bound_to_a_function_parameter() {
    let mut setup = W33.to_vec();
    setup.push("def u:r(k) { reduce_add(rotate(W, k, 1) * X) }");
    setup.push("X = randn(3, [3, 3])");
    let (a, b) = both(
        &setup,
        "grad(u:r(1), W)",
        "grad(reduce_add(rotate(W, 1, 1) * X), W)",
    );
    assert_eq!(a, b);
}

#[test]
fn pow_exponent_bound_to_a_function_parameter() {
    let mut setup = W33.to_vec();
    setup.push("def u:p(k) { reduce_add(pow(W, k)) }");
    let (a, b) = both(&setup, "grad(u:p(3), W)", "grad(reduce_add(pow(W, 3)), W)");
    assert_eq!(a, b);
}

#[test]
fn transpose_axes_perm_bound_to_a_function_parameter() {
    let mut setup = W33.to_vec();
    setup.push("X = randn(3, [3, 3])");
    setup.push("def u:t(p) { reduce_add(transpose_axes(W, p) * X) }");
    let (a, b) = both(
        &setup,
        "grad(u:t([1, 0]), W)",
        "grad(reduce_add(transpose_axes(W, [1, 0]) * X), W)",
    );
    assert_eq!(a, b);
}

#[test]
fn comparison_operators_are_constant_masks_inside_grad() {
    let setup = [
        "W = param[3, 3]",
        "W = randn(1, [3, 3])",
        "r = reshape(iota(3), [3, 1])",
        "c = reshape(iota(3), [1, 3])",
    ];
    let (a, b) = both(
        &setup,
        "grad(reduce_add(W * (c < r + 1)), W)",
        "grad(reduce_add(W * lt(c, r + 1)), W)",
    );
    assert_eq!(a, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    assert_eq!(a, b);
    let mut env = Environment::new();
    for line in setup {
        run(&mut env, line).expect("setup");
    }
    for op in ["<=", ">", ">=", "==", "!="] {
        let src = format!("grad(reduce_add(W * (c {op} r)), W)");
        run(&mut env, &src).unwrap_or_else(|e| panic!("{op}: {e}"));
    }
}

#[test]
fn multi_head_causal_attention_trains() {
    let mut env = Environment::new();
    for line in [
        "A = causal_attention(16, 4, 7)",
        "X = randn(1, [8, 16])",
        "T = randn(2, [8, 16])",
        "train 30 { adam(mean((apply(A, X) - T) * (apply(A, X) - T)), A, 0.01, 0.9, 0.999, 0.00000001) }",
    ] {
        run(&mut env, line).expect("setup");
    }
    let l = run(&mut env, "last_losses").expect("losses");
    assert!(
        l[29] < l[0] * 0.1,
        "4-head attention trains: {} -> {}",
        l[0],
        l[29]
    );
}
