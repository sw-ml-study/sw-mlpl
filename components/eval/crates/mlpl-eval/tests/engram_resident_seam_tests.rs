//! Seam profile for one warm engram-in-chain resident train step
//! (saga E5 step 001 baseline). Own binary: the counters are
//! process-global. The numbers this prints are the E5 optimization
//! ledger -- the known residual seam crossings are the concat in
//! the gate path (CPU structural fallback, forward + backward),
//! the fused cross-entropy backward and the per-step upload of
//! the host-built selection one-hot (concat runs resident since
//! dev-concat landed).

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};
use mlpl_tensor_handle::{seam_reset, seam_snapshot};

fn eval(env: &mut Environment, src: &str) {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    eval_program(&stmts, env).expect("eval");
}

#[test]
fn warm_engram_step_profile_is_bounded_and_printed() {
    let mut env = Environment::new();
    eval(
        &mut env,
        "e = engram(4, [2], 1, 8, 4, 7); \
         m = chain(embed(16, 4, 0), residual(chain(rms_norm(4), causal_attention(4, 1, 1))), \
                   e, rms_norm(4), linear(4, 16, 4)); \
         ids = [1, 2, 3, 1, 2]; Y = [2, 3, 1, 2, 3]; 0",
    );
    let step = "adam(cross_entropy(apply(m, ids), Y), m, 0.001, 0.9, 0.999, 0.00000001)";
    // Warm up (uploads weights, seeds resident moments).
    eval(&mut env, &format!("device(\"mlx\") {{ {step} }}; 0"));
    seam_reset();
    eval(
        &mut env,
        &format!("device(\"mlx\") {{ train 10 {{ {step} }} }}; 0"),
    );
    let (up, down, submit, fallback) = seam_snapshot();
    let per = |v: u64| v as f64 / 10.0;
    println!(
        "warm engram step: uploads={:.1} downloads={:.1} submits={:.1} cpu_fallbacks={:.1}",
        per(up),
        per(down),
        per(submit),
        per(fallback)
    );
    // dev-concat landed: the only CPU fallback left per step is
    // fused cross-entropy backward.
    assert!(per(fallback) <= 1.0, "unexplained fallback growth");
    assert!(per(down) <= 30.0, "unexplained download growth");
}
