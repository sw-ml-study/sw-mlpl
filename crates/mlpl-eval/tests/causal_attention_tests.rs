//! Saga 13 step 003: causal masking for attention.
//!
//! `causal_attention(d_model, heads, seed)` builds a model identical
//! to `attention(d_model, heads, seed)` but applies a lower-triangular
//! causal mask to the pre-softmax scores so position `t` cannot attend
//! to positions `> t`.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, model_params};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

#[test]
fn causal_attention_registers_same_four_params_as_attention() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = causal_attention(4, 1, 5)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let names = model_params(&env, "A").unwrap();
    assert_eq!(names.len(), 4, "Wq Wk Wv Wo");
    for n in &names {
        assert_eq!(env.get(n).unwrap().shape().dims(), &[4, 4]);
        assert!(env.is_param(n));
    }
}

#[test]
fn causal_attention_same_param_set_as_attention_same_seed() {
    // With the same seed, causal_attention and attention should
    // initialise their four projections to identical values -- the
    // only difference is the mask applied at forward time.
    let mut env_c = Environment::new();
    let mut env_u = Environment::new();
    eval_program(
        &parse(&lex("A = causal_attention(4, 1, 11)").unwrap()).unwrap(),
        &mut env_c,
    )
    .unwrap();
    eval_program(
        &parse(&lex("A = attention(4, 1, 11)").unwrap()).unwrap(),
        &mut env_u,
    )
    .unwrap();
    let names_c = model_params(&env_c, "A").unwrap();
    let names_u = model_params(&env_u, "A").unwrap();
    assert_eq!(names_c.len(), names_u.len());
    for (nc, nu) in names_c.iter().zip(names_u.iter()) {
        let wc = env_c.get(nc).unwrap();
        let wu = env_u.get(nu).unwrap();
        assert_eq!(wc.shape().dims(), wu.shape().dims());
        assert_eq!(wc.data(), wu.data());
    }
}

#[test]
fn causal_attention_position0_output_independent_of_later_positions() {
    // Position 0's output row must depend only on position 0's input
    // row. Perturbing position 1 must leave position 0's output
    // unchanged (within epsilon). Without the causal mask, the
    // softmax row for position 0 mixes all positions, so a
    // perturbation at position 1 would leak into position 0's output.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = causal_attention(4, 1, 3)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();

    // Base input: [T=3, d=4].
    let x_base = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, // position 0
            0.2, 0.8, -0.1, 0.3, // position 1
            -0.3, 0.4, 0.9, 0.1, // position 2
        ],
    );
    env.set("X".into(), x_base);
    let out_base = run("apply(A, X)", &mut env);
    assert_eq!(out_base.shape().dims(), &[3, 4]);
    let row0_base: Vec<f64> = out_base.data()[0..4].to_vec();

    // Perturb only position 1.
    let x_perturbed = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, // position 0 (unchanged)
            5.0, -3.0, 2.0, 4.0, // position 1 (very different)
            -0.3, 0.4, 0.9, 0.1, // position 2 (unchanged)
        ],
    );
    env.set("X".into(), x_perturbed);
    let out_perturbed = run("apply(A, X)", &mut env);
    let row0_perturbed: Vec<f64> = out_perturbed.data()[0..4].to_vec();

    for (a, b) in row0_base.iter().zip(row0_perturbed.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "causal mask leak: position-0 output changed when position 1 was \
             perturbed. base={row0_base:?}, perturbed={row0_perturbed:?}"
        );
    }
}

#[test]
fn unmasked_attention_position0_output_does_change_with_later_positions() {
    // Sanity check: without the causal mask, perturbing position 1
    // DOES change position 0's output. This pins the contrast the
    // causal test above relies on -- if this stops being true, the
    // mask test would trivially pass for the wrong reasons.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = attention(4, 1, 3)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let x_base = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, 0.2, 0.8, -0.1, 0.3, -0.3, 0.4, 0.9, 0.1,
        ],
    );
    env.set("X".into(), x_base);
    let out_base = run("apply(A, X)", &mut env);
    let row0_base: Vec<f64> = out_base.data()[0..4].to_vec();

    let x_perturbed = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, 5.0, -3.0, 2.0, 4.0, -0.3, 0.4, 0.9, 0.1,
        ],
    );
    env.set("X".into(), x_perturbed);
    let out_perturbed = run("apply(A, X)", &mut env);
    let row0_perturbed: Vec<f64> = out_perturbed.data()[0..4].to_vec();

    let max_delta = row0_base
        .iter()
        .zip(row0_perturbed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > 1e-6,
        "unmasked attention should leak position-1 into position-0 output; got delta={max_delta}"
    );
}

#[test]
fn causal_attention_lower_triangle_respected_at_all_positions() {
    // For T=3, position 1 must depend on positions 0 and 1, but not
    // on position 2. Perturbing position 2 must leave positions 0
    // and 1 unchanged.
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = causal_attention(4, 1, 7)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let x_base = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, 0.2, 0.8, -0.1, 0.3, -0.3, 0.4, 0.9, 0.1,
        ],
    );
    env.set("X".into(), x_base);
    let out_base = run("apply(A, X)", &mut env);
    let rows01_base: Vec<f64> = out_base.data()[0..8].to_vec();

    let x_perturbed = arr(
        vec![3, 4],
        vec![
            1.0, 0.0, 0.5, -0.2, 0.2, 0.8, -0.1, 0.3, 9.0, -9.0, 9.0, -9.0,
        ],
    );
    env.set("X".into(), x_perturbed);
    let out_perturbed = run("apply(A, X)", &mut env);
    let rows01_perturbed: Vec<f64> = out_perturbed.data()[0..8].to_vec();

    for (a, b) in rows01_base.iter().zip(rows01_perturbed.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "causal mask violated: rows 0..1 changed when position 2 was perturbed"
        );
    }
}

#[test]
fn grad_through_apply_causal_attention_produces_nonzero_grad() {
    // Gradcheck-style: the masked forward is still differentiable
    // wrt Wq via the tape. Loss is mean squared error to a fixed
    // target; gradient must be non-trivially non-zero.
    let mut env = Environment::new();
    let setup = "\
        attn = causal_attention(4, 1, 41)\n\
        X = [[1.0, 0.0, 0.5, -0.2], [0.2, 0.8, -0.1, 0.3], [-0.3, 0.4, 0.9, 0.1]]\n\
        Y = [[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.0], [0.2, 0.3, 0.0, 0.1]]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();

    let names = model_params(&env, "attn").expect("attn is a model");
    assert_eq!(names.len(), 4);
    let wq = names[0].clone();
    let src = format!("grad(mean((apply(attn, X) - Y) * (apply(attn, X) - Y)), {wq})");
    let g = run(&src, &mut env);
    assert_eq!(g.shape().dims(), &[4, 4]);
    assert!(g.data().iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn grad_matches_finite_difference_on_causal_attention() {
    // Finite-difference gradcheck on a single Wq entry. The analytic
    // grad from the tape must agree with the numerical central
    // difference to a few decimal places.
    let mut env = Environment::new();
    let setup = "\
        attn = causal_attention(4, 1, 19)\n\
        X = [[1.0, 0.0, 0.5, -0.2], [0.2, 0.8, -0.1, 0.3], [-0.3, 0.4, 0.9, 0.1]]\n\
        Y = [[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.0], [0.2, 0.3, 0.0, 0.1]]\n";
    eval_program(&parse(&lex(setup).unwrap()).unwrap(), &mut env).unwrap();
    let names = model_params(&env, "attn").unwrap();
    let wq = names[0].clone();

    // Analytic gradient via the tape.
    let loss_src = "mean((apply(attn, X) - Y) * (apply(attn, X) - Y))";
    let grad_src = format!("grad({loss_src}, {wq})");
    let g = run(&grad_src, &mut env);
    let g_entry = g.data()[0]; // d loss / d Wq[0, 0]

    // Central finite-difference wrt Wq[0, 0].
    let h = 1e-4_f64;
    let wq_arr = env.get(&wq).unwrap().clone();
    let mut plus = wq_arr.data().to_vec();
    plus[0] += h;
    let mut minus = wq_arr.data().to_vec();
    minus[0] -= h;

    env.set(
        wq.clone(),
        DenseArray::new(wq_arr.shape().clone(), plus).unwrap(),
    );
    let l_plus = run(loss_src, &mut env).data()[0];

    env.set(
        wq.clone(),
        DenseArray::new(wq_arr.shape().clone(), minus).unwrap(),
    );
    let l_minus = run(loss_src, &mut env).data()[0];

    let fd = (l_plus - l_minus) / (2.0 * h);
    assert!(
        (g_entry - fd).abs() < 1e-4,
        "analytic grad {g_entry} disagreed with finite-difference {fd}"
    );
}

#[test]
fn describe_causal_attention_renders_flag() {
    // `:describe` should make it obvious that a layer is causal.
    use mlpl_eval::inspect;
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("A = causal_attention(4, 1, 2)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let out = inspect(&env, ":describe A").unwrap();
    assert!(
        out.contains("causal_attention(d=4, heads=1)"),
        "expected causal_attention render, got: {out}"
    );
}
