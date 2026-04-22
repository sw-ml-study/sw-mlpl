//! Saga 20 step 004: end-to-end Neural Thickets integration test.
//!
//! This composes every Saga 20 builtin (`clone_model`,
//! `perturb_params`, `argtop_k`, `scatter`) against a Saga 13
//! Tiny LM style model on a tiny synthetic corpus. It is the
//! cut-down mirror of `demos/neural_thicket.mlpl`: same structure,
//! smaller sizes so the test runs in seconds on CPU.
//!
//! Asserts:
//! - 16-wide losses vector is fully finite.
//! - `argtop_k` returns 4 in-range indices.
//! - `heat` is shaped `[4, 4]`.
//! - The ensemble logits keep the same shape as a single variant's
//!   forward output.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

#[test]
fn neural_thicket_cut_down_runs_end_to_end() {
    let mut env = Environment::new();

    // Small byte-level corpus (mirrors the tiny_lm_train_tests
    // pattern): predictable vocab, plenty of tokens for a context
    // of 4, trains in a few gradient steps.
    let src = "\
        corpus = \"abcabcabcabcabcabcabcabcabcabcabcabc\"\n\
        ids    = tokenize_bytes(corpus)\n\
        X_all  = shift_pairs_x(ids, 4)\n\
        Y_all  = shift_pairs_y(ids, 4)\n\
        X      = reshape(X_all, [reduce_mul(shape(X_all))])\n\
        Y      = reshape(Y_all, [reduce_mul(shape(Y_all))])\n\
        V = 256 ; d = 8 ; h = 1\n\
        base = chain(embed(V, d, 0),\
                     residual(chain(rms_norm(d), causal_attention(d, h, 1))),\
                     residual(chain(rms_norm(d), linear(d, 16, 2),\
                                    relu_layer(), linear(16, d, 3))),\
                     rms_norm(d),\
                     linear(d, V, 4))\n\
        train 5 {\n\
          adam(cross_entropy(apply(base, X), Y), base,\
               0.01, 0.9, 0.999, 0.00000001);\n\
          loss_metric = cross_entropy(apply(base, X), Y)\n\
        }\n\
        val_X = X\n\
        val_Y = Y\n\
        sigma  = 0.1\n\
        losses = zeros([16])\n\
    ";
    run(&mut env, src);

    // Four family loops. Keeping them as separate statements so a
    // single parse failure doesn't mask structural bugs.
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"all_layers\", sigma, i + 100);\
           losses = scatter(losses, i, cross_entropy(apply(v, val_X), val_Y))\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"attention_only\", sigma, i + 200);\
           losses = scatter(losses, 4 + i, cross_entropy(apply(v, val_X), val_Y))\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"mlp_only\", sigma, i + 300);\
           losses = scatter(losses, 8 + i, cross_entropy(apply(v, val_X), val_Y))\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"embed_and_head\", sigma, i + 400);\
           losses = scatter(losses, 12 + i, cross_entropy(apply(v, val_X), val_Y))\n\
         }",
    );

    // Heatmap shape + finite losses.
    run(&mut env, "heat = reshape(losses, [4, 4])");
    let heat = env.get("heat").expect("heat bound");
    assert_eq!(heat.shape().dims(), &[4, 4]);
    let losses = env.get("losses").expect("losses bound");
    assert_eq!(losses.shape().dims(), &[16]);
    assert_eq!(losses.data().len(), 16);
    for (i, &v) in losses.data().iter().enumerate() {
        assert!(v.is_finite(), "loss at slot {i} should be finite, got {v}");
    }

    // argtop_k on negated losses -> 4 indices in [0, 16).
    run(&mut env, "neg_losses = -1.0 * losses");
    run(&mut env, "best_idx = argtop_k(neg_losses, 4)");
    let best_idx = env.get("best_idx").expect("best_idx bound");
    assert_eq!(best_idx.shape().dims(), &[4]);
    let mut seen = std::collections::HashSet::new();
    for &idx_f in best_idx.data() {
        let idx = idx_f as usize;
        assert!(idx < 16, "best_idx entry {idx} out of range");
        assert!(
            seen.insert(idx),
            "best_idx should have distinct entries, saw {idx} twice"
        );
    }

    // Capture the single-variant output shape for the ensemble check.
    run(
        &mut env,
        "v0 = clone_model(base); one_logits = apply(v0, val_X)",
    );
    let one_shape = env
        .get("one_logits")
        .expect("one_logits bound")
        .shape()
        .dims()
        .to_vec();

    // Ensemble over all 16 variants (same logic as the demo).
    run(&mut env, "ens_logits = zeros(shape(apply(base, val_X)))");
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"all_layers\", sigma, i + 100);\
           ens_logits = ens_logits + apply(v, val_X)\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"attention_only\", sigma, i + 200);\
           ens_logits = ens_logits + apply(v, val_X)\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"mlp_only\", sigma, i + 300);\
           ens_logits = ens_logits + apply(v, val_X)\n\
         }",
    );
    run(
        &mut env,
        "for i in [0, 1, 2, 3] {\n\
           v = clone_model(base);\
           perturb_params(v, \"embed_and_head\", sigma, i + 400);\
           ens_logits = ens_logits + apply(v, val_X)\n\
         }",
    );
    run(&mut env, "ens_logits = ens_logits * (1.0 / 16.0)");

    let ens = env.get("ens_logits").expect("ens_logits bound");
    assert_eq!(
        ens.shape().dims(),
        one_shape.as_slice(),
        "ensemble logits must match a single variant's forward shape"
    );
    for (i, &v) in ens.data().iter().enumerate() {
        assert!(v.is_finite(), "ensemble logits at {i} should be finite");
    }
}
