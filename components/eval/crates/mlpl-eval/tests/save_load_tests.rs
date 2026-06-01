//! `save_model(model, path)` / `load_model(path)`: persist a trained
//! model's spec + every param value to a JSON snapshot and restore it.
//! Pins the contract that a loaded model reproduces the original's
//! forward pass exactly -- in-session and into a fresh Environment
//! (i.e. genuine train-once / load-many persistence).

use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

const BUILD_AND_TRAIN: &str = "\
m = chain(embed(8, 4, 0), linear(4, 8, 1))\n\
X = [0, 1, 2, 3]\n\
experiment \"e\" { train 3 { adam(cross_entropy(apply(m, X), [1, 2, 3, 0]), m, 0.05, 0.9, 0.999, 0.00000001) } }\n";

#[test]
fn save_then_load_reproduces_forward_in_session() {
    let path = std::env::temp_dir().join("mlpl_saveload_insession.json");
    let p = path.to_str().unwrap();
    let mut env = Environment::new();
    run(
        &mut env,
        &format!(
            "{BUILD_AND_TRAIN}\
before = apply(m, X)\n\
save_model(m, \"{p}\")\n\
m2 = load_model(\"{p}\")\n\
after = apply(m2, X)\n\
match = mean(eq(reshape(before, [32]), reshape(after, [32])))\n"
        ),
    );
    assert_eq!(
        env.get("match").unwrap().data()[0],
        1.0,
        "loaded model's forward must match the original exactly"
    );
    let params = model_params(&env, "m2").expect("m2 registered as a model");
    assert!(!params.is_empty());
    for name in &params {
        assert!(env.get(name).is_some(), "loaded param {name} present");
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn load_into_fresh_environment_matches_original() {
    let path = std::env::temp_dir().join("mlpl_saveload_freshenv.json");
    let p = path.to_str().unwrap();

    // Train + save in one Environment, capturing its forward.
    // (End on an array statement -- a program ending on save_model's
    // Model return value would trip eval_program's array expectation.)
    let mut env1 = Environment::new();
    run(
        &mut env1,
        &format!("{BUILD_AND_TRAIN}save_model(m, \"{p}\")\norig = apply(m, X)\n"),
    );
    let orig = env1.get("orig").expect("orig forward").data().to_vec();

    // Load into a brand-new Environment -- no prior knowledge of the
    // architecture -- and reproduce the forward.
    let mut env2 = Environment::new();
    run(
        &mut env2,
        &format!("X = [0, 1, 2, 3]\nm = load_model(\"{p}\")\nout = apply(m, X)\n"),
    );
    let loaded = env2.get("out").expect("loaded forward").data().to_vec();

    // JSON serialization of f64 round-trips to ~15 significant digits
    // (well below training noise), so compare within tolerance rather
    // than bit-for-bit.
    assert_eq!(orig.len(), loaded.len());
    for (o, l) in orig.iter().zip(&loaded) {
        assert!(
            (o - l).abs() < 1e-9,
            "fresh-env forward diverged: {o} vs {l}"
        );
    }
    std::fs::remove_file(&path).ok();
}
