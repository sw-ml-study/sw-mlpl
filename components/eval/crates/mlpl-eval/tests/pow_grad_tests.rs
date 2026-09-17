//! RS1-pow (../reasoning-from-scratch): pow(x, k) with a constant positive
//! integer exponent differentiates (exact repeated-product rule); other
//! exponents error loudly.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(src: &str, env: &mut Environment) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

fn param_env(vals: Vec<f64>) -> Environment {
    let mut env = Environment::new();
    let n = vals.len();
    env.set_param("x".into(), DenseArray::new(Shape::vector(n), vals).unwrap());
    env
}

#[test]
fn grad_of_pow_square_is_2x() {
    let x = [1.5f64, 2.0, 3.0];
    let mut env = param_env(x.to_vec());
    let g = run("grad(sum(pow(x, 2)), x)", &mut env);
    for (i, xi) in x.iter().enumerate() {
        assert!((g.data()[i] - 2.0 * xi).abs() < 1e-9, "d x^2 at {xi}");
    }
}

#[test]
fn grad_of_pow_cube_is_3x2() {
    let x = [1.5f64, 2.0, 3.0];
    let mut env = param_env(x.to_vec());
    let g = run("grad(sum(pow(x, 3)), x)", &mut env);
    for (i, xi) in x.iter().enumerate() {
        assert!((g.data()[i] - 3.0 * xi * xi).abs() < 1e-9, "d x^3 at {xi}");
    }
}

#[test]
fn pow_square_matches_finite_differences() {
    let base = [0.7f64, 2.5];
    let mut env = param_env(base.to_vec());
    let g = run("grad(sum(pow(x, 4)), x)", &mut env);
    let eps = 1e-6;
    for (i, xi) in base.iter().enumerate() {
        let fd = ((xi + eps).powi(4) - (xi - eps).powi(4)) / (2.0 * eps);
        assert!((g.data()[i] - fd).abs() < 1e-4, "fd x^4 at {xi}");
    }
}

#[test]
fn fractional_exponent_errors_loudly() {
    let mut env = param_env(vec![4.0]);
    let toks = lex("grad(sum(pow(x, 0.5)), x)").unwrap();
    let stmts = parse(&toks).unwrap();
    let err = eval_program(&stmts, &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("sqrt"), "error should point to sqrt: {msg}");
}

#[test]
fn eager_pow_unchanged() {
    let mut env = Environment::new();
    let v = run("pow(2, 10)", &mut env);
    assert!((v.data()[0] - 1024.0).abs() < 1e-9);
}
