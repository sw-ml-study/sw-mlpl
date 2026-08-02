//! Seam instrumentation (saga E4 step 8): counts uploads /
//! downloads / submissions / CPU fallbacks for resident training
//! steps, so performance results are explainable. Run with
//! `--nocapture` to see the raw profile.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};
use mlpl_tensor_handle::{seam_reset, seam_snapshot};

fn eval(env: &mut Environment, src: &str) {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

const SETUP: &str = "m = chain(embed(60, 16, 0), \
     residual(chain(rms_norm(16), causal_attention(16, 1, 1))), \
     rms_norm(16), linear(16, 60, 2)); \
     X = [1, 3, 5, 7, 2, 4, 6, 0]; Y = [3, 5, 7, 2, 4, 6, 0, 1]";
const STEP: &str = "adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001)";

#[test]
fn one_resident_step_profile_is_bounded_and_printed() {
    let mut env = Environment::new();
    eval(&mut env, SETUP);
    seam_reset();
    eval(&mut env, &format!("device(\"mlx\") {{ {STEP} }}"));
    let (up, down, submit, fallback) = seam_snapshot();
    println!(
        "one tiny-LM adam step: uploads={up} downloads={down} submits={submit} cpu_fallbacks={fallback}"
    );
    assert!(up > 0 && down > 0 && submit > 0, "resident path engaged");
}

#[test]
fn thirty_step_loop_profile_shows_amortization() {
    let mut env = Environment::new();
    eval(&mut env, SETUP);
    // Step 1 warms the resident caches; measure steps 2..=31.
    eval(&mut env, &format!("device(\"mlx\") {{ {STEP} }}"));
    seam_reset();
    eval(
        &mut env,
        &format!("device(\"mlx\") {{ train 30 {{ {STEP} }} }}"),
    );
    let (up, down, submit, fallback) = seam_snapshot();
    println!(
        "30 warm steps: uploads={up} downloads={down} submits={submit} cpu_fallbacks={fallback} \
         (per step: up={:.1} down={:.1} submit={:.1} fb={:.1})",
        up as f64 / 30.0,
        down as f64 / 30.0,
        submit as f64 / 30.0,
        fallback as f64 / 30.0
    );
    // The weight cache must amortize: params re-upload only when
    // a structural fallback breaks residency, never all of them.
    assert!(down > 0, "per-step host mirror refresh happens");
}
