//! Saga 13 step 001: token embedding layer.
//!
//! `embed(vocab_size, d_model, seed)` constructs a `Value::Model` whose
//! single trainable parameter is a `[vocab, d_model]` lookup table.
//! `apply(emb, tokens)` gathers rows from the table by integer token
//! ids: input `[N]` -> output `[N, d_model]`. Gradients flow back into
//! the table through the autograd tape so `adam(loss, emb, ...)`
//! trains it like any other layer.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

#[test]
fn embed_creates_model_with_one_table_parameter() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(5, 3, 7)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "E").expect("E is a registered model");
    assert_eq!(names.len(), 1, "embedding owns a single table parameter");
    let table = env.get(&names[0]).expect("table bound");
    assert_eq!(table.shape().dims(), &[5, 3]);
    assert!(
        env.is_param(&names[0]),
        "table must be tracked as trainable"
    );
}

#[test]
fn apply_embed_gathers_rows_by_token_id() {
    // Build an embedding then overwrite the table with a known matrix.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(5, 3, 1)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "E").unwrap();
    let table = arr(
        vec![5, 3],
        vec![
            0.0, 0.1, 0.2, // row 0
            1.0, 1.1, 1.2, // row 1
            2.0, 2.1, 2.2, // row 2
            3.0, 3.1, 3.2, // row 3
            4.0, 4.1, 4.2, // row 4
        ],
    );
    env.set(names[0].clone(), table);
    env.set("toks".into(), arr(vec![3], vec![0.0, 2.0, 4.0]));
    let out = run("apply(E, toks)", &mut env);
    assert_eq!(out.shape().dims(), &[3, 3]);
    assert_eq!(out.data(), &[0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 4.0, 4.1, 4.2]);
}

#[test]
fn apply_embed_handles_repeated_tokens() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(3, 2, 2)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "E").unwrap();
    env.set(
        names[0].clone(),
        arr(vec![3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    );
    env.set("toks".into(), arr(vec![4], vec![1.0, 1.0, 0.0, 1.0]));
    let out = run("apply(E, toks)", &mut env);
    assert_eq!(out.shape().dims(), &[4, 2]);
    assert_eq!(
        out.data(),
        &[30.0, 40.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0]
    );
}

#[test]
fn apply_embed_rejects_out_of_range_token() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(3, 2, 0)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    env.set("toks".into(), arr(vec![2], vec![0.0, 5.0])); // 5 >= vocab 3
    let result = eval_program(&parse(&lex("apply(E, toks)").unwrap()).unwrap(), &mut env);
    assert!(result.is_err(), "out-of-range token must surface an error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("token") || msg.contains("vocab") || msg.contains("range"),
        "error should mention token / vocab / range, got: {msg}"
    );
}

#[test]
fn grad_through_apply_embed_matches_one_hot_form() {
    // grad(loss, table) for an embedding lookup must equal grad of the
    // hand-written `O @ table` form, where O is the [N, V] one-hot of
    // the token ids. This is the gradcheck for the embedding layer.
    let mut env = Environment::new();
    let setup = "\
        E = embed(4, 2, 5)\n\
        toks = [0.0, 2.0, 1.0, 2.0]\n\
        ONE = [[1.0, 0.0, 0.0, 0.0], \
               [0.0, 0.0, 1.0, 0.0], \
               [0.0, 1.0, 0.0, 0.0], \
               [0.0, 0.0, 1.0, 0.0]]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();
    let names = model_params(&env, "E").unwrap();
    let table = names[0].clone();

    let g_apply = run(&format!("grad(sum(apply(E, toks)), {table})"), &mut env);
    let g_hand = run(
        &format!("grad(sum(matmul(ONE, {table})), {table})"),
        &mut env,
    );
    assert_eq!(g_apply.shape(), g_hand.shape());
    for (a, b) in g_apply.data().iter().zip(g_hand.data().iter()) {
        assert!((a - b).abs() < 1e-10, "grad mismatch: {a} vs {b}");
    }
    // Repeated token id 2 must accumulate: row 2 of the gradient should
    // be twice the contribution of a single row.
    let g = g_apply.data();
    assert!((g[2 * 2] - 2.0).abs() < 1e-10, "row 2 col 0 expected 2.0");
    assert!(
        (g[2 * 2 + 1] - 2.0).abs() < 1e-10,
        "row 2 col 1 expected 2.0"
    );
}

#[test]
fn adam_step_reduces_embedding_loss() {
    // sum(apply(E, toks) * apply(E, toks)) is a quadratic in the rows
    // of the table touched by `toks` (rows 0, 1, 2). 50 Adam steps must
    // shrink those rows toward zero. Row 3 is untouched and so its
    // norm should be unchanged. We can't read sum() outside grad/opt,
    // so we verify by inspecting the table directly after training.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(4, 2, 11)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "E").unwrap();
    let table_name = names[0].clone();
    env.set(
        table_name.clone(),
        arr(vec![4, 2], vec![1.0, -1.0, 0.5, 0.5, -0.5, 0.25, 9.0, -9.0]),
    );
    env.set("toks".into(), arr(vec![3], vec![0.0, 1.0, 2.0]));

    let row_norm = |t: &DenseArray, r: usize| -> f64 {
        let d = t.data();
        (d[r * 2] * d[r * 2] + d[r * 2 + 1] * d[r * 2 + 1]).sqrt()
    };
    let table0 = env.get(&table_name).unwrap().clone();
    let n0_0 = row_norm(&table0, 0);
    let n0_1 = row_norm(&table0, 1);
    let n0_2 = row_norm(&table0, 2);
    let n0_3 = row_norm(&table0, 3);

    eval_program(
        &parse(&lex(
            "train 50 { adam(sum(apply(E, toks) * apply(E, toks)), E, 0.05, 0.9, 0.999, 0.00000001) }",
        ).unwrap()).unwrap(),
        &mut env,
    ).unwrap();

    let table1 = env.get(&table_name).unwrap().clone();
    let n1_0 = row_norm(&table1, 0);
    let n1_1 = row_norm(&table1, 1);
    let n1_2 = row_norm(&table1, 2);
    let n1_3 = row_norm(&table1, 3);

    assert!(n1_0 < n0_0, "row 0 expected to shrink, {n0_0} -> {n1_0}");
    assert!(n1_1 < n0_1, "row 1 expected to shrink, {n0_1} -> {n1_1}");
    assert!(n1_2 < n0_2, "row 2 expected to shrink, {n0_2} -> {n1_2}");
    assert!(
        (n1_3 - n0_3).abs() < 1e-12,
        "row 3 (untouched by toks) must be unchanged, {n0_3} -> {n1_3}"
    );
}

#[test]
fn describe_embed_renders_vocab_and_d() {
    // :describe should pretty-print embedding shape parameters.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("E = embed(7, 4, 0)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let desc = mlpl_eval::inspect(&env, ":describe E").expect("inspect returns a String");
    assert!(
        desc.contains("embed[vocab=7, d=4]"),
        ":describe E should include 'embed[vocab=7, d=4]', got:\n{desc}"
    );
}
