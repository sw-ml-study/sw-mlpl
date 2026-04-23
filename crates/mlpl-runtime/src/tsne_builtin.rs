//! Saga 16 step 002: `tsne(X, perplexity, iters, seed)`
//! public entry + orchestrator.
//!
//! Validation lives in `tsne_validate`; the high-dim
//! affinity matrix in `tsne_affinities`; the per-step
//! gradient descent in `tsne_gradient`. This module is
//! the thin facade: dispatch name -> validate args ->
//! build P -> init Y -> loop the gradient step -> return
//! `[N, 2]` result.
//!
//! See `contracts/eval-contract/tsne.md` for every
//! hyperparameter choice and the rotational / reflection
//! ambiguity in the output.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;
use crate::prng::Xorshift64;
use crate::tsne_affinities::compute_p_matrix;
use crate::tsne_gradient::tsne_step;
use crate::tsne_validate::{TsneArgs, validate_tsne_args};

const EARLY_EXAG: f64 = 4.0;
const EARLY_EXAG_END: usize = 100;
const MOMENTUM_SWITCH: usize = 250;
const MOMENTUM_EARLY: f64 = 0.5;
const MOMENTUM_LATE: f64 = 0.8;

pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "tsne" => Some(builtin_tsne(args)),
        _ => None,
    }
}

/// `tsne(X, perplexity, iters, seed) -> Y [N, 2]`.
/// Orchestrator. Each named step delegates to a
/// single-responsibility helper in a sibling module.
fn builtin_tsne(args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    let TsneArgs {
        n,
        d,
        perplexity,
        iters,
        seed,
        xs,
    } = validate_tsne_args(args)?;
    let p = compute_p_matrix(&xs, n, d, perplexity);
    let mut y = init_y(n, seed);
    let mut prev_update = vec![0.0_f64; n * 2];
    for step in 0..iters {
        let (exag, momentum) = schedule(step);
        tsne_step(&mut y, &p, &mut prev_update, n, exag, momentum);
    }
    Ok(DenseArray::new(Shape::new(vec![n, 2]), y)?)
}

/// Step-dependent hyperparameter schedule. van der Maaten
/// defaults: early-exaggeration factor 4 for the first
/// 100 iterations, then 1. Momentum 0.5 for the first 250
/// iterations, then 0.8.
fn schedule(step: usize) -> (f64, f64) {
    let exag = if step < EARLY_EXAG_END {
        EARLY_EXAG
    } else {
        1.0
    };
    let momentum = if step < MOMENTUM_SWITCH {
        MOMENTUM_EARLY
    } else {
        MOMENTUM_LATE
    };
    (exag, momentum)
}

/// Initial `Y`: `randn(seed, [N, 2]) * 1e-4`. Uses the
/// same Xorshift64 path `mlpl_runtime::randn` uses so the
/// t-SNE init is deterministic and seed-consistent with
/// the rest of the language.
fn init_y(n: usize, seed: f64) -> Vec<f64> {
    let raw_seed = seed as i64 as u64;
    let mut rng = Xorshift64::new(raw_seed);
    (0..n * 2).map(|_| rng.next_normal() * 1e-4).collect()
}
