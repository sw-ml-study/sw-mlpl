//! Saga 15 step 002: `lora(m, rank, alpha, seed)` +
//! `ModelSpec::LinearLora` variant.
//!
//! `lora` clones `m` and replaces every `Linear` node with a
//! `LinearLora` that owns two fresh adapter matrices alongside
//! the cloned base W, b. A is `randn * (1 / sqrt(in))`, B is
//! all zeros. The base W, b of each LinearLora are
//! automatically marked frozen at rewrite time so
//! `adam(loss, student, ...)` (step 004 demo) only moves the
//! adapters; users can `unfreeze(student)` to also train the
//! base.
//!
//! This step ships the rewrite and the param allocation only.
//! Forward + autograd for `LinearLora` is step 003; `apply`
//! on a LoRA-wrapped model returns a clear "not yet
//! supported" error until step 003 lands.

use std::collections::HashSet;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, env).unwrap();
}

fn _arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

#[test]
fn lora_wraps_single_linear_with_four_params() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");

    let names = model_params(&env, "student").expect("student bound");
    assert_eq!(names.len(), 4, "LinearLora contributes W, b, A, B");

    // Shapes: W [4, 8], b [1, 8], A [4, 2], B [2, 8].
    let mut shapes: Vec<(String, Vec<usize>)> = names
        .iter()
        .map(|n| (n.clone(), env.get(n).unwrap().shape().dims().to_vec()))
        .collect();
    shapes.sort_by_key(|(n, _)| n.clone());

    let expected_w = shapes
        .iter()
        .any(|(n, d)| n.starts_with("__linear_W_") && d == &[4, 8]);
    let expected_b = shapes
        .iter()
        .any(|(n, d)| n.starts_with("__linear_b_") && d == &[1, 8]);
    let expected_a = shapes
        .iter()
        .any(|(n, d)| n.starts_with("__lora_A_") && d == &[4, 2]);
    let expected_b_adapter = shapes
        .iter()
        .any(|(n, d)| n.starts_with("__lora_B_") && d == &[2, 8]);

    assert!(
        expected_w,
        "student should have a __linear_W_* param of shape [4, 8]; got {shapes:?}"
    );
    assert!(
        expected_b,
        "student should have a __linear_b_* param of shape [1, 8]"
    );
    assert!(
        expected_a,
        "student should have a __lora_A_* param of shape [4, 2]"
    );
    assert!(
        expected_b_adapter,
        "student should have a __lora_B_* param of shape [2, 8]"
    );
}

#[test]
fn lora_b_adapter_is_zero_initialized() {
    // The LoRA convention: B is zero-init so the
    // pre-training-step adapter delta is zero and the
    // wrapped forward matches the base exactly before any
    // gradient step. Forward identity is tested in step 003
    // (once apply supports LinearLora); step 002 just pins
    // the storage invariant.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    let names = model_params(&env, "student").unwrap();
    let b_adapter_name = names
        .iter()
        .find(|n| n.starts_with("__lora_B_"))
        .expect("student has a __lora_B_* param");
    let b_adapter = env.get(b_adapter_name).unwrap();
    for &v in b_adapter.data() {
        assert_eq!(v, 0.0, "every entry of the LoRA B matrix must init to zero");
    }
}

#[test]
fn lora_a_adapter_is_scaled_randn() {
    // A is `randn(seed + i) * (1 / sqrt(in))`. At in=4 that
    // scale is 0.5. We do not pin exact bit values (the
    // randn path goes through the runtime and could shift
    // if the PRNG changes); we only assert that A is not
    // all-zero and that its magnitude is bounded by a
    // sensible multiple of the scale.
    let mut env = Environment::new();
    run(&mut env, "m = linear(16, 8, 0)");
    run(&mut env, "student = lora(m, 4, 4.0, 7)");

    let names = model_params(&env, "student").unwrap();
    let a_name = names
        .iter()
        .find(|n| n.starts_with("__lora_A_"))
        .expect("student has a __lora_A_* param");
    let a = env.get(a_name).unwrap();
    let scale = 1.0 / (16.0_f64).sqrt();

    let mut any_nonzero = false;
    for &v in a.data() {
        if v != 0.0 {
            any_nonzero = true;
        }
        assert!(
            v.abs() <= 10.0 * scale,
            "A entry {v} should stay inside 10 * (1 / sqrt(in)) = {}",
            10.0 * scale
        );
    }
    assert!(any_nonzero, "A should not be all-zero");
}

#[test]
fn lora_auto_freezes_base_but_not_adapters() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");

    let names = model_params(&env, "student").unwrap();
    for n in &names {
        let is_base_w = n.starts_with("__linear_W_");
        let is_base_b = n.starts_with("__linear_b_");
        let is_adapter = n.starts_with("__lora_A_") || n.starts_with("__lora_B_");

        if is_base_w || is_base_b {
            assert!(
                env.is_frozen(n),
                "base param '{n}' should auto-freeze at lora() rewrite"
            );
        }
        if is_adapter {
            assert!(
                !env.is_frozen(n),
                "adapter param '{n}' must NOT be frozen (adapters are what train)"
            );
        }
    }
}

#[test]
fn lora_walks_chain_and_wraps_every_linear() {
    let mut env = Environment::new();
    // Two linears with a relu between them; lora should wrap both.
    run(
        &mut env,
        "m = chain(linear(4, 8, 0), relu_layer(), linear(8, 4, 1))",
    );
    run(&mut env, "student = lora(m, 2, 4.0, 7)");

    let names = model_params(&env, "student").unwrap();
    let a_count = names.iter().filter(|n| n.starts_with("__lora_A_")).count();
    let b_count = names.iter().filter(|n| n.starts_with("__lora_B_")).count();
    assert_eq!(a_count, 2, "two linears -> two A adapters");
    assert_eq!(b_count, 2, "two linears -> two B adapters");
}

#[test]
fn lora_leaves_non_linear_nodes_unchanged() {
    let mut env = Environment::new();
    // embed + causal_attention + rms_norm should NOT be
    // wrapped; only the trailing Linear gets a LoRA adapter.
    run(
        &mut env,
        "m = chain(embed(16, 8, 0), \
                   residual(chain(rms_norm(8), causal_attention(8, 1, 1))), \
                   rms_norm(8), \
                   linear(8, 16, 2))",
    );
    run(&mut env, "student = lora(m, 2, 4.0, 7)");

    let names = model_params(&env, "student").unwrap();
    // exactly one linear => exactly one adapter pair.
    let a_count = names.iter().filter(|n| n.starts_with("__lora_A_")).count();
    let b_count = names.iter().filter(|n| n.starts_with("__lora_B_")).count();
    assert_eq!(a_count, 1);
    assert_eq!(b_count, 1);

    // Attention + embedding still present.
    let has_attn = names.iter().any(|n| n.starts_with("__attn_"));
    let has_embed = names.iter().any(|n| n.starts_with("__embed_"));
    assert!(
        has_attn,
        "attention params should be present in the wrapped spec"
    );
    assert!(
        has_embed,
        "embedding param should be present in the wrapped spec"
    );
}

#[test]
fn repeated_lora_calls_produce_disjoint_param_names() {
    let mut env = Environment::new();
    run(&mut env, "base = linear(4, 8, 0)");
    run(&mut env, "s1 = lora(base, 2, 4.0, 7)");
    run(&mut env, "s2 = lora(base, 2, 4.0, 7)");

    let n1: HashSet<String> = model_params(&env, "s1").unwrap().into_iter().collect();
    let n2: HashSet<String> = model_params(&env, "s2").unwrap().into_iter().collect();
    assert!(
        n1.is_disjoint(&n2),
        "two lora() calls on the same base must produce fully disjoint param sets"
    );
}

#[test]
fn lora_rejects_rank_zero() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    let stmts = parse(&lex("lora(m, 0, 4.0, 7)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank=0 should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("lora") && msg.contains("rank"), "got: {msg}");
}

#[test]
fn lora_rejects_rank_too_large() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    let stmts = parse(&lex("lora(m, 5, 4.0, 7)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("rank > min(in=4, out=8) should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("lora") && msg.contains("rank"), "got: {msg}");
}

#[test]
fn lora_rejects_wrong_arity() {
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    let stmts = parse(&lex("lora(m, 2, 4.0)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("3-arg form should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("lora") || msg.contains("arity"), "got: {msg}");
}

#[test]
fn lora_rejects_non_model_argument() {
    let mut env = Environment::new();
    run(&mut env, "x = 1");
    let stmts = parse(&lex("lora(x, 2, 4.0, 7)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("x is not a model");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(msg.contains("lora") || msg.contains("model"), "got: {msg}");
}

#[test]
fn lora_rejects_nested_lora() {
    // Applying lora() to an already-lora'd model is not yet
    // supported; the rewrite would allocate adapters for the
    // cloned base W, b which also have hidden adapters --
    // meaning is ambiguous. Surface it as an error.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    run(&mut env, "s1 = lora(m, 2, 4.0, 7)");
    let stmts = parse(&lex("s2 = lora(s1, 2, 4.0, 7)").unwrap()).unwrap();
    let err = eval_program(&stmts, &mut env).expect_err("nested lora should error");
    let msg = format!("{err:?}").to_ascii_lowercase();
    assert!(
        msg.contains("lora"),
        "error should mention lora, got: {msg}"
    );
}

#[test]
fn apply_on_lora_model_runs_forward_in_step_003() {
    // Originally a step-002 test that pinned the "forward
    // not yet implemented" stub. Step 003 replaced that
    // stub with the real forward; the test now asserts the
    // positive: apply(lora_m, X) runs and produces the
    // expected shape. Full forward semantics (identity
    // before training, formula correctness, MLX parity,
    // frozen-base training isolation) live in
    // `lora_forward_tape_tests.rs`.
    let mut env = Environment::new();
    run(&mut env, "m = linear(4, 8, 0)");
    run(&mut env, "student = lora(m, 2, 4.0, 7)");
    env.set(
        "X".into(),
        DenseArray::new(Shape::new(vec![2, 4]), vec![1.0; 8]).unwrap(),
    );
    let stmts = parse(&lex("apply(student, X)").unwrap()).unwrap();
    let out = eval_program(&stmts, &mut env).expect("apply on LinearLora should work in step 003");
    assert_eq!(out.shape().dims(), &[2, 8]);
}
