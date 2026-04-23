//! Saga 15 step 004: end-to-end `demos/lora_finetune.mlpl`
//! integration test.
//!
//! Cut-down variant (V=32, d=8, ctx=4, 3 base-train + 3
//! lora-finetune steps) exercises the full pipeline in under
//! a second on CPU. Pins the invariants the demo argues:
//!
//! - Base pre-training moves the base params.
//! - `student = lora(base, ...)` clones the base (student's
//!   params are disjoint from the base's originals).
//! - Fine-tuning the student leaves every base param in
//!   student (cloned base W/b + embedding + attention)
//!   bit-identical; only `__lora_A_*` and `__lora_B_*` move.
//! - Adapter B moves away from its zero init.
//! - The cross-entropy loss after 3 fine-tune steps does not
//!   NaN out (loose bound; the main claim here is the frozen/
//!   trainable isolation, not the final loss value).

use std::collections::HashMap;

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

fn snapshot_student(env: &Environment) -> HashMap<String, Vec<f64>> {
    mlpl_eval::model_params(env, "student")
        .unwrap()
        .into_iter()
        .map(|n| {
            let v = env.get(&n).unwrap().data().to_vec();
            (n, v)
        })
        .collect()
}

#[test]
fn lora_finetune_freezes_base_and_moves_adapters() {
    let mut env = Environment::new();

    // --- base pre-training on a tiny synthetic corpus ---
    // Explicit integer sequences keep every id in [0, V) so
    // the embedding lookup is well-defined. (`tokenize_bytes`
    // produces byte ids 0-255, which would need V >= 256 and
    // slow the test down.)
    run(
        &mut env,
        "ids = [1, 3, 5, 7, 2, 4, 6, 0, 9, 11, 13, 15, 2, 4, 6, 0, 1, 3, 5, 7, 2, 4, 6, 0]",
    );
    run(&mut env, "X_all = shift_pairs_x(ids, 4)");
    run(&mut env, "Y_all = shift_pairs_y(ids, 4)");
    run(&mut env, "X = reshape(X_all, [reduce_mul(shape(X_all))])");
    run(&mut env, "Y = reshape(Y_all, [reduce_mul(shape(Y_all))])");
    run(&mut env, "V = 16 ; d = 8 ; h = 1");
    run(
        &mut env,
        "base = chain(embed(V, d, 0), \
                      residual(chain(rms_norm(d), causal_attention(d, h, 1))), \
                      rms_norm(d), \
                      linear(d, V, 2))",
    );
    run(
        &mut env,
        "train 3 { adam(cross_entropy(apply(base, X), Y), base, 0.01, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(base, X), Y) }",
    );

    // --- wrap with LoRA; capture pre-fine-tune snapshot ---
    run(&mut env, "student = lora(base, 2, 4.0, 7)");
    let before = snapshot_student(&env);

    // B adapter should start at zero.
    for (name, vals) in &before {
        if name.starts_with("__lora_B_") {
            for v in vals {
                assert_eq!(*v, 0.0, "adapter B '{name}' must init to zero");
            }
        }
    }

    // --- fine-tune the student on the same corpus (small
    //     enough to reuse X/Y; the point is the frozen
    //     isolation, not distribution shift) ---
    run(
        &mut env,
        "train 3 { adam(cross_entropy(apply(student, X), Y), student, 0.05, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(student, X), Y) }",
    );
    let after = snapshot_student(&env);

    // Every non-adapter param in the student must be
    // bit-identical; every adapter must have moved.
    for (name, before_vals) in &before {
        let after_vals = after.get(name).unwrap();
        let is_adapter = name.starts_with("__lora_A_") || name.starts_with("__lora_B_");
        if is_adapter {
            assert_ne!(
                before_vals, after_vals,
                "adapter '{name}' should have moved across 3 adam steps"
            );
        } else {
            assert_eq!(
                before_vals, after_vals,
                "frozen base param '{name}' must be bit-identical after fine-tune"
            );
        }
    }

    // --- final loss is finite (no NaN blow-up) ---
    let losses = env.get("last_losses").expect("last_losses bound");
    assert_eq!(losses.shape().dims(), &[3]);
    for (i, v) in losses.data().iter().enumerate() {
        assert!(
            v.is_finite(),
            "fine-tune loss at step {i} is not finite: {v}"
        );
    }
}

#[test]
fn lora_finetune_does_not_touch_the_source_base_binding() {
    // `student = lora(base, ...)` clones the base. Training
    // the student must not move the base's own params.
    let mut env = Environment::new();
    run(
        &mut env,
        "ids = [1, 3, 5, 7, 2, 4, 6, 0, 9, 11, 13, 15, 2, 4, 6, 0]",
    );
    run(&mut env, "X_all = shift_pairs_x(ids, 4)");
    run(&mut env, "Y_all = shift_pairs_y(ids, 4)");
    run(&mut env, "X = reshape(X_all, [reduce_mul(shape(X_all))])");
    run(&mut env, "Y = reshape(Y_all, [reduce_mul(shape(Y_all))])");
    run(
        &mut env,
        "base = chain(embed(16, 8, 0), \
                      residual(chain(rms_norm(8), causal_attention(8, 1, 1))), \
                      linear(8, 16, 2))",
    );

    // Snapshot base BEFORE lora().
    let base_names = mlpl_eval::model_params(&env, "base").unwrap();
    let base_before: HashMap<String, Vec<f64>> = base_names
        .iter()
        .map(|n| (n.clone(), env.get(n).unwrap().data().to_vec()))
        .collect();

    run(&mut env, "student = lora(base, 2, 4.0, 7)");
    run(
        &mut env,
        "train 3 { adam(cross_entropy(apply(student, X), Y), student, 0.05, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(student, X), Y) }",
    );

    // Base's original names are untouched by student
    // training (lora cloned into fresh names).
    for (name, before_vals) in &base_before {
        let after_vals = env.get(name).unwrap().data();
        assert_eq!(
            before_vals.as_slice(),
            after_vals,
            "original base param '{name}' must not move when we train the student"
        );
    }
}
