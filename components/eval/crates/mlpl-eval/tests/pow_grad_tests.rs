//! RS1-pow (../reasoning-from-scratch): pow(x, k) with ANY constant exponent
//! differentiates via the PowConst tape node (d/dx x^k = k*x^(k-1)); a
//! differentiable (param-dependent) exponent is rejected.

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
fn grad_of_pow_half_is_finite_diff() {
    // d/dx x^0.5 = 0.5 * x^-0.5.
    let base = [0.25f64, 1.0, 4.0];
    let mut env = param_env(base.to_vec());
    let g = run("grad(sum(pow(x, 0.5)), x)", &mut env);
    for (i, xi) in base.iter().enumerate() {
        assert!(
            (g.data()[i] - 0.5 * xi.powf(-0.5)).abs() < 1e-9,
            "d x^0.5 at {xi}"
        );
    }
}

#[test]
fn grad_of_pow_negative_and_fractional_match_finite_diff() {
    let base = [0.7f64, 2.5, 3.3];
    for (expo, src) in [
        (-1.0, "grad(sum(pow(x, 0 - 1)), x)"),
        (2.5, "grad(sum(pow(x, 2.5)), x)"),
    ] {
        let mut env = param_env(base.to_vec());
        let g = run(src, &mut env);
        let eps = 1e-6;
        for (i, xi) in base.iter().enumerate() {
            let fd = ((xi + eps).powf(expo) - (xi - eps).powf(expo)) / (2.0 * eps);
            assert!(
                (g.data()[i] - fd).abs() < 1e-3,
                "fd x^{expo} at {xi}: {} vs {fd}",
                g.data()[i]
            );
        }
    }
}

#[test]
fn differentiable_exponent_is_rejected() {
    let mut env = param_env(vec![4.0]);
    env.set_param("k".into(), DenseArray::from_scalar(2.0));
    let toks = lex("grad(sum(pow(x, k)), x)").unwrap();
    let stmts = parse(&toks).unwrap();
    let err = eval_program(&stmts, &mut env).unwrap_err();
    assert!(
        format!("{err}").contains("exponent"),
        "rejects a differentiable exponent"
    );
}

#[test]
fn eager_pow_unchanged() {
    let mut env = Environment::new();
    let v = run("pow(2, 10)", &mut env);
    assert!((v.data()[0] - 1024.0).abs() < 1e-9);
}
