//! RS1 (../reasoning-from-scratch): sqrt / sin / cos are differentiable
//! inside `grad`. Each gradient is checked against its closed form
//! (d/dx sqrt = 0.5/sqrt(x), d/dx sin = cos, d/dx cos = -sin) and, for
//! sqrt, a finite-difference cross-check. Eager (non-grad) values are
//! unchanged by the tape wiring.

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
fn grad_of_sqrt_is_half_over_sqrt() {
    let mut env = param_env(vec![0.25, 1.0, 4.0, 9.0]);
    let g = run("grad(sum(sqrt(x)), x)", &mut env);
    let x = [0.25f64, 1.0, 4.0, 9.0];
    for (i, xi) in x.iter().enumerate() {
        assert!(
            (g.data()[i] - 0.5 / xi.sqrt()).abs() < 1e-9,
            "d sqrt at {xi}: got {}",
            g.data()[i]
        );
    }
}

#[test]
fn grad_of_sqrt_matches_finite_differences() {
    let base = [0.5f64, 2.0, 7.5];
    let mut env = param_env(base.to_vec());
    let g = run("grad(sum(sqrt(x)), x)", &mut env);
    let eps = 1e-6;
    for (i, xi) in base.iter().enumerate() {
        let fd = ((xi + eps).sqrt() - (xi - eps).sqrt()) / (2.0 * eps);
        assert!((g.data()[i] - fd).abs() < 1e-5, "fd mismatch at {xi}");
    }
}

#[test]
fn grad_of_sin_is_cos() {
    let x = [0.0f64, 0.5, 1.0, -0.7];
    let mut env = param_env(x.to_vec());
    let g = run("grad(sum(sin(x)), x)", &mut env);
    for (i, xi) in x.iter().enumerate() {
        assert!((g.data()[i] - xi.cos()).abs() < 1e-9, "d sin at {xi}");
    }
}

#[test]
fn grad_of_cos_is_neg_sin() {
    let x = [0.0f64, 0.5, 1.0, -0.7];
    let mut env = param_env(x.to_vec());
    let g = run("grad(sum(cos(x)), x)", &mut env);
    for (i, xi) in x.iter().enumerate() {
        assert!((g.data()[i] + xi.sin()).abs() < 1e-9, "d cos at {xi}");
    }
}

#[test]
fn eager_values_unchanged() {
    let mut env = Environment::new();
    let s = run("sqrt(4)", &mut env);
    assert!((s.data()[0] - 2.0).abs() < 1e-12);
    let si = run("sin(0)", &mut env);
    assert!(si.data()[0].abs() < 1e-12);
    let co = run("cos(0)", &mut env);
    assert!((co.data()[0] - 1.0).abs() < 1e-12);
}
