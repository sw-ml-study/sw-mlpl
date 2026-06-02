//! `play_vs_random(model, n)` builtin (tic-tac-toe fine-tune demo):
//! play the named model as O against a random opponent for `n` games
//! and return outcome counts `[losses, ties, wins]` -- the input to
//! the `svg(_, "waffle")` outcome viz (red loss / gray tie / green
//! win). The board->move policy is scored by the model's forward
//! pass; the game loop itself lives in `mlpl-tictactoe`.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_tictactoe::{Board, encode, play_vs_random_counts};

use crate::env::Environment;
use crate::model_apply::apply_model;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_types::{EvalError, Value};

/// Deterministic opponent seed so the before/after comparison is
/// reproducible across runs.
const SEED: u64 = 0x07ac_70e5;

pub(crate) fn eval_play_vs_random(args: &[Expr], env: &Environment) -> Result<Value, EvalError> {
    let model = resolve_model(&args[0], env)?;
    let n = parse_n(&args[1])?;
    // Preflight: surface a model/shape error up front rather than
    // silently degrading every move in the loop.
    apply_model(&model, &board_input(&Board::default()), env)?;
    let counts = play_vs_random_counts(n, SEED, |b: &Board| best_legal(&model, env, b));
    let data = counts.iter().map(|&c| c as f64).collect();
    Ok(Value::Array(
        DenseArray::new(Shape::new(vec![3]), data).unwrap(),
    ))
}

fn resolve_model(arg: &Expr, env: &Environment) -> Result<ModelSpec, EvalError> {
    let Expr::Ident(name, _) = arg else {
        return Err(EvalError::Unsupported(
            "play_vs_random: first argument must be a model name".into(),
        ));
    };
    env.get_model(name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(name.clone()))
}

fn parse_n(arg: &Expr) -> Result<usize, EvalError> {
    match arg {
        Expr::IntLit(k, _) if *k >= 0 => Ok(*k as usize),
        _ => Err(EvalError::Unsupported(
            "play_vs_random: second argument must be a non-negative game count".into(),
        )),
    }
}

/// Encode a board from O's (the model's) perspective as a `[1, 27]`
/// one-hot batch ready for the model forward pass.
fn board_input(b: &Board) -> DenseArray {
    let enc = encode(&b.cells, -1).to_vec();
    DenseArray::new(Shape::new(vec![1, 27]), enc).unwrap()
}

/// The model's move: score every cell, pick the highest-scoring legal
/// one. Falls back to the first legal cell if the forward pass fails
/// (the preflight already ruled that out for a well-formed model).
fn best_legal(model: &ModelSpec, env: &Environment, b: &Board) -> usize {
    let logits = apply_model(model, &board_input(b), env)
        .map(|a| a.data().to_vec())
        .unwrap_or_default();
    let legal = b.legal();
    if logits.is_empty() {
        return legal.first().copied().unwrap_or(0);
    }
    legal
        .into_iter()
        .max_by(|&i, &j| {
            logits[i]
                .partial_cmp(&logits[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}
