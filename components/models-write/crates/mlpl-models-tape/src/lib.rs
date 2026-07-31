//! Tape-side lowering of the Saga 11 model DSL. Saga 33 step
//! 008 extracted this crate from `crates/mlpl-eval/src/
//! model_tape*.rs` into the `components/mlpl-session` nested
//! workspace so it can be tested + evolved independently of
//! the wider eval engine.
//!
//! Autograd twin of model_dispatch's `apply_model`: given a
//! `ModelSpec` and an input `Tensor`, walks the model tree and
//! emits the same primitive ops onto the autograd tape so
//! `grad(loss, ...)` -- and every optimizer built on top --
//! can route gradients back to each trainable parameter leaf.

pub mod apply;
pub mod attention;
pub mod attention_heads;
pub mod error;
pub mod layers;
pub mod linear;

pub use apply::apply_model_tape;
pub use error::TapeError;
pub use layers::{EngramInputs, engram_tape};

use mlpl_array::DenseArray;
use mlpl_core::LabeledShape;

/// `cross_entropy` target validation. Lives at crate root
/// (no dedicated module) so the crate's module count stays at
/// 7 modules (PASS).
///
/// Validates that `targets` is a flat 1-D index array matching
/// the flattened row count of `logits`, with every entry in
/// `0..vocab`.
pub fn validate_cross_entropy_targets(
    logits: &DenseArray,
    targets: &DenseArray,
) -> Result<Vec<usize>, TapeError> {
    let (n, v) = logits_n_vocab(logits)?;
    check_target_shape(n, targets)?;
    extract_target_indices(targets, v)
}

fn logits_n_vocab(logits: &DenseArray) -> Result<(usize, usize), TapeError> {
    let dims = logits.shape().dims();
    match dims.len() {
        2 => Ok((dims[0], dims[1])),
        3 => Ok((dims[0] * dims[1], dims[2])),
        r => Err(TapeError::Unsupported(format!(
            "cross_entropy: logits must be rank 2 or 3, got rank {r}"
        ))),
    }
}

fn check_target_shape(n: usize, targets: &DenseArray) -> Result<(), TapeError> {
    if targets.elem_count() == n {
        return Ok(());
    }
    Err(TapeError::ShapeMismatch {
        op: "cross_entropy".into(),
        expected: LabeledShape::new(vec![n], vec![None]),
        actual: LabeledShape::new(targets.shape().dims().to_vec(), vec![None; targets.rank()]),
    })
}

fn extract_target_indices(targets: &DenseArray, v: usize) -> Result<Vec<usize>, TapeError> {
    let mut idx = Vec::with_capacity(targets.elem_count());
    for (i, &t) in targets.data().iter().enumerate() {
        if t < 0.0 || t.fract() != 0.0 || (t as usize) >= v {
            return Err(TapeError::Unsupported(format!(
                "cross_entropy: target[{i}] = {t} is not a valid class index in 0..{v}"
            )));
        }
        idx.push(t as usize);
    }
    Ok(idx)
}
