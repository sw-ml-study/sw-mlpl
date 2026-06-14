//! Coverage for the `train_val_curve` analysis builtin and the
//! concat-accumulation pattern the "Watch a Model Learn (overfitting)"
//! demo uses to record per-epoch train/validation loss. The full demo's
//! training cost lives in the heavy web-demo smoke; here the dispatch and
//! data-flow plumbing are checked cheaply (zero or 4 steps), so this runs
//! in milliseconds even on the dev profile.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(src: &str) -> Value {
    let toks = lex(src).expect("lex");
    let prog = parse(&toks).expect("parse");
    let mut env = Environment::default();
    eval_program_value(&prog, &mut env).expect("eval")
}

fn run_str(src: &str) -> String {
    match run(src) {
        Value::Str(s) => s,
        other => panic!("expected an SVG string, got {other:?}"),
    }
}

#[test]
fn train_val_curve_dispatches_to_two_polylines() {
    let svg = run_str("train_val_curve([2, 1.2, 0.6, 0.3], [2, 1.3, 1.1, 1.4])");
    assert!(svg.starts_with("<svg"), "svg: {svg}");
    assert_eq!(svg.matches("<polyline").count(), 2, "svg: {svg}");
    assert!(svg.contains(">train<") && svg.contains(">val<"), "legend: {svg}");
}

#[test]
fn concat_accumulates_a_loss_history() {
    // The demo primes a [1] vector then concats one scalar per epoch.
    let v = run("h = [1.0]; h = concat(h, [0.8]); h = concat(h, [0.6]); h");
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[1.0, 0.8, 0.6]),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn watch_model_learn_structure_runs_in_miniature() {
    // The exact data-flow of the demo -- separate train/val sets, an
    // over-capacity chain, chunked `train`, per-chunk concat of train+val
    // loss, then `train_val_curve` -- but 2 chunks x 2 steps so it is fast.
    let svg = run_str(
        "Mtr = moons(7, 12, 0.3)\n\
         Xtr = matmul(Mtr, [[1,0],[0,1],[0,0]])\n\
         ytr = reshape(matmul(Mtr, [[0],[0],[1]]), [12])\n\
         Mva = moons(101, 20, 0.3)\n\
         Xva = matmul(Mva, [[1,0],[0,1],[0,0]])\n\
         yva = reshape(matmul(Mva, [[0],[0],[1]]), [20])\n\
         mdl = chain(linear(2, 8, 1), tanh_layer(), linear(8, 2, 2))\n\
         trh = [cross_entropy(apply(mdl, Xtr), ytr)]\n\
         vah = [cross_entropy(apply(mdl, Xva), yva)]\n\
         train 2 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.05, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }\n\
         trh = concat(trh, [cross_entropy(apply(mdl, Xtr), ytr)])\n\
         vah = concat(vah, [cross_entropy(apply(mdl, Xva), yva)])\n\
         train 2 { adam(cross_entropy(apply(mdl, Xtr), ytr), mdl, 0.05, 0.9, 0.999, 0.00000001); cross_entropy(apply(mdl, Xtr), ytr) }\n\
         trh = concat(trh, [cross_entropy(apply(mdl, Xtr), ytr)])\n\
         vah = concat(vah, [cross_entropy(apply(mdl, Xva), yva)])\n\
         train_val_curve(trh, vah)",
    );
    assert_eq!(svg.matches("<polyline").count(), 2, "svg: {svg}");
}
