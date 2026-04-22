//! Saga 20 step 002: `perturb_params(m, family, sigma, seed)`.
//!
//! `perturb_params` walks a model's parameter set, filters by the
//! named family, and adds `sigma * randn(seed, shape)` to each
//! matching parameter in place. It is the core of the Neural
//! Thickets workflow: clone a base, perturb the clone, evaluate.
//!
//! These tests pin:
//! - Each family touches exactly the intended parameter subset.
//! - Parameters outside the family are bit-identical to the source.
//! - The structural head rule for `mlp_only` and `embed_and_head`
//!   (the last top-level `linear` child of the outermost `chain`).
//! - Determinism on (seed, sigma) and a practical magnitude bound
//!   on Gaussian deltas.
//! - The source model is never affected by a perturb on its clone.
//! - Error handling for unknown families, wrong arity, non-model
//!   arguments.

use std::collections::{HashMap, HashSet};

use mlpl_array::DenseArray;
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

/// Build the spec fixture: embedding + causal attention block +
/// MLP block + final projection head. Contains every parameter
/// family needed to exercise the four family strings.
fn build_fixture(env: &mut Environment) {
    let src = "base = chain(embed(8, 4, 0),\
                             residual(chain(rms_norm(4), causal_attention(4, 1, 1))),\
                             residual(chain(rms_norm(4),\
                                            linear(4, 16, 2),\
                                            relu_layer(),\
                                            linear(16, 4, 3))),\
                             rms_norm(4),\
                             linear(4, 8, 4))";
    run(env, src);
    run(env, "copy = clone_model(base)");
}

/// Snapshot the tensor values for a model's params, keyed by name.
fn snapshot(env: &Environment, model_ident: &str) -> HashMap<String, Vec<f64>> {
    let names = model_params(env, model_ident).unwrap();
    let mut out = HashMap::new();
    for n in names {
        let v = env.get(&n).unwrap().data().to_vec();
        out.insert(n, v);
    }
    out
}

/// Return the subset of param names whose values differ between
/// `before` and `after` (both assumed to be same-keyed maps).
fn changed_params(
    before: &HashMap<String, Vec<f64>>,
    after: &HashMap<String, Vec<f64>>,
) -> HashSet<String> {
    let mut out = HashSet::new();
    for (name, old) in before {
        if old != after.get(name).unwrap() {
            out.insert(name.clone());
        }
    }
    out
}

/// Largest absolute delta between two same-shape tensor data vecs.
fn max_abs_delta(before: &[f64], after: &[f64]) -> f64 {
    before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn all_layers_touches_every_param() {
    let mut env = Environment::new();
    build_fixture(&mut env);

    let base_before = snapshot(&env, "base");
    let copy_before = snapshot(&env, "copy");
    run(&mut env, "perturb_params(copy, \"all_layers\", 0.02, 42)");
    let base_after = snapshot(&env, "base");
    let copy_after = snapshot(&env, "copy");

    // Base is untouched.
    assert_eq!(base_before, base_after, "base must not be affected");

    // Every param of the copy is now different.
    let changed = changed_params(&copy_before, &copy_after);
    assert_eq!(
        changed.len(),
        copy_before.len(),
        "all_layers must perturb every parameter"
    );

    // Gaussian noise * sigma stays inside a practical bound. We use
    // 10*sigma rather than 6*sigma because `randn` goes through
    // Box-Muller on a 53-bit uniform; the theoretical max magnitude
    // is ~sqrt(-2 * ln(2^-53)) = sqrt(2 * 37 * ln 2) ~= 8.6, so 10x
    // is a safe practical ceiling while still catching any serious
    // scaling bug (an outright `sigma` mis-application would blow
    // past this easily).
    let ten_sigma = 10.0 * 0.02;
    for (name, before) in &copy_before {
        let after = copy_after.get(name).unwrap();
        let d = max_abs_delta(before, after);
        assert!(
            d <= ten_sigma,
            "delta on {name} should stay within 10*sigma, got {d}"
        );
    }
}

#[test]
fn attention_only_touches_only_attention_projections() {
    let mut env = Environment::new();
    build_fixture(&mut env);

    let before = snapshot(&env, "copy");
    run(
        &mut env,
        "perturb_params(copy, \"attention_only\", 0.02, 7)",
    );
    let after = snapshot(&env, "copy");

    let changed = changed_params(&before, &after);
    assert!(
        !changed.is_empty(),
        "attention_only must affect at least one param"
    );
    for name in &changed {
        assert!(
            name.starts_with("__attn_"),
            "attention_only changed a non-attention param: {name}"
        );
    }
    // Spot-check: at least one Wq name was touched (attention spec
    // holds four projections under the same id).
    assert!(changed.iter().any(|n| n.starts_with("__attn_Wq_")));
}

#[test]
fn mlp_only_excludes_final_projection_head() {
    let mut env = Environment::new();
    build_fixture(&mut env);

    // Identify the head linear's params (the last top-level linear
    // child of the outermost chain): that's the `linear(4, 8, 4)`
    // call, whose W shape is [4, 8].
    let copy_names = model_params(&env, "copy").unwrap();
    let head_w_b: HashSet<String> = copy_names
        .iter()
        .filter(|n| {
            let v = env.get(n).unwrap();
            let d = v.shape().dims();
            // W: [4, 8]; b: [1, 8]
            (n.starts_with("__linear_W_") && d == [4, 8])
                || (n.starts_with("__linear_b_") && d == [1, 8])
        })
        .cloned()
        .collect();
    assert_eq!(head_w_b.len(), 2, "head linear contributes W and b");

    let before = snapshot(&env, "copy");
    run(&mut env, "perturb_params(copy, \"mlp_only\", 0.02, 11)");
    let after = snapshot(&env, "copy");

    let changed = changed_params(&before, &after);
    assert!(!changed.is_empty(), "mlp_only must touch MLP linears");
    for name in &changed {
        assert!(
            name.starts_with("__linear_W_") || name.starts_with("__linear_b_"),
            "mlp_only changed a non-linear param: {name}"
        );
        assert!(
            !head_w_b.contains(name),
            "mlp_only must NOT touch the head linear: {name}"
        );
    }
    // The head params are untouched; spot-check.
    for h in &head_w_b {
        assert_eq!(
            before.get(h).unwrap(),
            after.get(h).unwrap(),
            "head param {h} should be bit-identical after mlp_only"
        );
    }
}

#[test]
fn embed_and_head_touches_embedding_and_final_linear_only() {
    let mut env = Environment::new();
    build_fixture(&mut env);

    let copy_names = model_params(&env, "copy").unwrap();
    // Head params: last top-level linear's W [4,8] and b [1,8].
    let head_w_b: HashSet<String> = copy_names
        .iter()
        .filter(|n| {
            let d = env.get(n).unwrap().shape().dims().to_vec();
            (n.starts_with("__linear_W_") && d == [4, 8])
                || (n.starts_with("__linear_b_") && d == [1, 8])
        })
        .cloned()
        .collect();
    let embed_names: HashSet<String> = copy_names
        .iter()
        .filter(|n| n.starts_with("__embed_E_"))
        .cloned()
        .collect();
    assert!(!embed_names.is_empty(), "fixture has an embedding");
    assert_eq!(head_w_b.len(), 2);

    let expected: HashSet<String> = embed_names.union(&head_w_b).cloned().collect();

    let before = snapshot(&env, "copy");
    run(
        &mut env,
        "perturb_params(copy, \"embed_and_head\", 0.02, 19)",
    );
    let after = snapshot(&env, "copy");

    let changed = changed_params(&before, &after);
    assert_eq!(
        changed, expected,
        "embed_and_head must touch exactly the embedding plus the head linear"
    );
}

#[test]
fn perturb_is_deterministic_for_same_seed_and_family() {
    // Two independent clones of the same base, perturbed with the
    // same (family, sigma, seed), must end up with identical deltas
    // applied across the affected params (in the natural walk order).
    let mut env = Environment::new();
    build_fixture(&mut env);
    run(&mut env, "copy2 = clone_model(base)");

    run(&mut env, "perturb_params(copy, \"all_layers\", 0.02, 123)");
    run(&mut env, "perturb_params(copy2, \"all_layers\", 0.02, 123)");

    let copy_names = model_params(&env, "copy").unwrap();
    let copy2_names = model_params(&env, "copy2").unwrap();
    assert_eq!(
        copy_names.len(),
        copy2_names.len(),
        "clones should have matching param counts"
    );
    let base_names = model_params(&env, "base").unwrap();
    // For each corresponding slot in the walk order, the (copy -
    // base) delta and (copy2 - base) delta must match bit-for-bit.
    for i in 0..copy_names.len() {
        let base_v = env.get(&base_names[i]).unwrap().data();
        let a = env.get(&copy_names[i]).unwrap().data();
        let b = env.get(&copy2_names[i]).unwrap().data();
        let da: Vec<f64> = a.iter().zip(base_v).map(|(x, y)| x - y).collect();
        let db: Vec<f64> = b.iter().zip(base_v).map(|(x, y)| x - y).collect();
        assert_eq!(da, db, "same seed must produce the same delta on slot {i}");
    }
}

#[test]
fn perturb_with_different_seed_produces_different_deltas() {
    let mut env = Environment::new();
    build_fixture(&mut env);
    run(&mut env, "copy2 = clone_model(base)");

    let copy_before = snapshot(&env, "copy");
    run(&mut env, "perturb_params(copy, \"all_layers\", 0.02, 42)");
    run(&mut env, "perturb_params(copy2, \"all_layers\", 0.02, 99)");
    let copy_after = snapshot(&env, "copy");

    // Pull copy2's tensors aligned to copy's walk order.
    let copy_names = model_params(&env, "copy").unwrap();
    let copy2_names = model_params(&env, "copy2").unwrap();
    let mut any_differs = false;
    for i in 0..copy_names.len() {
        let a = env.get(&copy_names[i]).unwrap().data();
        let b = env.get(&copy2_names[i]).unwrap().data();
        if a != b {
            any_differs = true;
            break;
        }
    }
    assert!(
        any_differs,
        "different seeds should produce different values"
    );
    // And the first copy did in fact change vs. its pre-perturb snapshot.
    assert_ne!(copy_before, copy_after);
}

#[test]
fn perturb_params_returns_a_scalar_unit() {
    // Aligning with to_device's convention: model-mutating builtins
    // return a scalar zero so they can sit in statement position.
    let mut env = Environment::new();
    build_fixture(&mut env);
    let stmts = parse(&lex("r = perturb_params(copy, \"all_layers\", 0.02, 1)").unwrap()).unwrap();
    let _ = eval_program(&stmts, &mut env).unwrap();
    let r = env.get("r").expect("r bound");
    assert_eq!(r.shape().dims(), &[] as &[usize], "r should be a scalar");
    assert_eq!(r.data(), &[0.0]);
}

#[test]
fn perturb_params_sigma_zero_is_a_no_op() {
    let mut env = Environment::new();
    build_fixture(&mut env);
    let before = snapshot(&env, "copy");
    run(&mut env, "perturb_params(copy, \"all_layers\", 0.0, 1)");
    let after = snapshot(&env, "copy");
    assert_eq!(before, after, "sigma=0 must leave every param untouched");
}

#[test]
fn perturb_params_rejects_unknown_family() {
    let mut env = Environment::new();
    build_fixture(&mut env);
    let stmts = parse(&lex("perturb_params(copy, \"banana\", 0.02, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("unknown family should error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("all_layers")
            && msg.contains("attention_only")
            && msg.contains("mlp_only")
            && msg.contains("embed_and_head"),
        "error message should list the accepted families, got: {msg}"
    );
    assert!(
        msg.to_ascii_lowercase().contains("banana"),
        "error message should include the bad name, got: {msg}"
    );
}

#[test]
fn perturb_params_rejects_wrong_arity() {
    let mut env = Environment::new();
    build_fixture(&mut env);
    let stmts = parse(&lex("perturb_params(copy, \"all_layers\", 0.02)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("arity mismatch");
    let msg = format!("{err:?}");
    assert!(
        msg.to_ascii_lowercase().contains("perturb_params")
            || msg.to_ascii_lowercase().contains("arity"),
        "error should reference perturb_params or arity, got: {msg}"
    );
}

#[test]
fn perturb_params_rejects_non_model_argument() {
    let mut env = Environment::new();
    run(&mut env, "x = 1");
    let stmts = parse(&lex("perturb_params(x, \"all_layers\", 0.02, 1)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("x is not a model");
    let msg = format!("{err:?}");
    assert!(
        msg.to_ascii_lowercase().contains("perturb_params")
            || msg.to_ascii_lowercase().contains("model"),
        "error should reference perturb_params or model, got: {msg}"
    );
}

#[test]
fn perturb_arbitrary_chain_without_top_level_linear_skips_head() {
    // When the outermost spec has no top-level linear child, there
    // is no head; `mlp_only` still works (touches all linears since
    // none are excluded), and `embed_and_head` only touches embed
    // params.
    let mut env = Environment::new();
    let src = "base = chain(embed(8, 4, 0),\
                             residual(chain(rms_norm(4),\
                                            linear(4, 16, 2),\
                                            relu_layer(),\
                                            linear(16, 4, 3))))";
    run(&mut env, src);
    run(&mut env, "copy = clone_model(base)");

    // embed_and_head -> only __embed_E_* changes.
    let before = snapshot(&env, "copy");
    run(
        &mut env,
        "perturb_params(copy, \"embed_and_head\", 0.02, 1)",
    );
    let after = snapshot(&env, "copy");
    let changed = changed_params(&before, &after);
    assert!(!changed.is_empty());
    for n in &changed {
        assert!(
            n.starts_with("__embed_E_"),
            "without a top-level linear head, embed_and_head must touch only embed params; got {n}"
        );
    }

    // mlp_only -> every __linear_* param (none excluded, since no head).
    let before = snapshot(&env, "copy");
    run(&mut env, "perturb_params(copy, \"mlp_only\", 0.02, 2)");
    let after = snapshot(&env, "copy");
    let changed = changed_params(&before, &after);
    let linear_count = before.keys().filter(|n| n.starts_with("__linear_")).count();
    assert_eq!(
        changed.len(),
        linear_count,
        "without a head, mlp_only should touch every linear param"
    );
}

fn _keep_densearray_import(_: &DenseArray) {}
