//! Stable-diffusion saga, step 1: the 2D diffusion denoiser learns.
//!
//! Forward-noise the two-moons dataset at every timestep over a
//! linspace/running_product schedule, train a tiny MLP to predict the added
//! noise, and assert the noise-prediction loss falls -- i.e. the
//! denoiser genuinely learns to undo the forward (noising) process,
//! which is what makes the reverse (denoising) sampling work. Sizes are
//! small so it runs fast in the default (dev) profile.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const DEMO: &str = "\
M = moons(1, 60, 0.08)\n\
X0 = matmul(M, [[1,0],[0,1],[0,0]])\n\
N = 60\n\
T = 8 ; betas = linspace(0.0001, 0.08, T) ; alphas = 1 - betas ; abar = running_product(alphas)\n\
started = 0 ; TX = fill([1,3], 0) ; TY = fill([1,2], 0) ; t = 0\n\
while gt(T, t) { ab = take(abar,0,t) ; eps = randn(t+7,[N,2]) ; Xt = sqrt(ab)*X0 + sqrt(1-ab)*eps ; tcol = fill([N,1], t/T) ; xin = concat(Xt, tcol, 1) ; if eq(started,0) { TX = xin ; TY = eps ; started = 1 } else { TX = concat(TX, xin, 0) ; TY = concat(TY, eps, 0) } ; t = t+1 }\n\
m = chain(linear(3,32,0), relu_layer(), linear(32,2,1))\n\
experiment \"d\" { train 80 { adam(mean((apply(m,TX)-TY)*(apply(m,TX)-TY)), m, 0.01, 0.9, 0.999, 0.00000001) ; loss_metric = mean((apply(m,TX)-TY)*(apply(m,TX)-TY)) } }\n";

#[test]
fn diffusion_denoiser_learns_to_predict_noise() {
    let mut env = Environment::new();
    eval_program(&parse(&lex(DEMO).unwrap()).unwrap(), &mut env).unwrap();
    let losses = env.get("last_losses").expect("last_losses bound").clone();
    let d = losses.data();
    let (first, last) = (d[0], *d.last().unwrap());
    assert!(
        last < first * 0.5,
        "noise-prediction loss should fall as the denoiser learns: {first} -> {last}"
    );
}

#[test]
fn linspace_running_product_build_a_valid_alpha_bar_schedule() {
    let mut env = Environment::new();
    let src = "T = 5 ; betas = linspace(0.0, 0.4, T) ; abar = running_product(1 - betas) ; abar";
    let r = eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env).unwrap();
    let v = r.data();
    assert_eq!(v.len(), 5);
    // alpha-bar is the running product of (1 - beta); strictly decreasing
    // after the first step and in (0, 1].
    assert!((v[0] - 1.0).abs() < 1e-12, "first alpha-bar is 1.0: {v:?}");
    assert!(
        v.windows(2).all(|w| w[1] <= w[0]),
        "alpha-bar decreasing: {v:?}"
    );
    assert!(
        v.iter().all(|&x| x > 0.0 && x <= 1.0),
        "alpha-bar in (0,1]: {v:?}"
    );
}
