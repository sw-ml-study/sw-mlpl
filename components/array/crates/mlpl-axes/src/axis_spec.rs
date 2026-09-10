//! Shared axis-selection types: name a set of axes once, resolve everywhere.
//!
//! Every builtin that selects axes (`reduce`, `compress`, `drop`, ...) or
//! attaches names (`label`, `relabel`, `reshape_labeled`) funnels through the
//! types here, so the accepted forms cannot drift between builtins or between
//! the interpreter and the compile-to-Rust path. This module is pure: it maps
//! an already-parsed selection to concrete axis positions and owns no parsing
//! of `Value`/`CVal` (that lives in each side's thin adapter).

use mlpl_array::DenseArray;

/// A selection of axes to operate over -- by name (resolved against an
/// array's labels) or by position. `#[non_exhaustive]` so a future form
/// (e.g. a range) cannot break downstream constructors.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AxisSpec {
    /// Axes named by label, e.g. `["channel", "kernel_y"]`.
    Names(Vec<String>),
    /// Axes by 0-based position, e.g. `[2, 3]`.
    Indices(Vec<usize>),
}

/// A list of axis names to ATTACH (`label` / `relabel` / `reshape_labeled`).
/// Inner `None` leaves that axis positional, matching `DenseArray`'s
/// `Option<Vec<Option<String>>>` label representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisNames(pub Vec<Option<String>>);

/// Why an [`AxisSpec`] could not be resolved against an array.
/// `#[non_exhaustive]` so new variants do not break downstream `match`es.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AxisError {
    /// A name was given but the array carries no labels.
    NamedAxisButNoLabels,
    /// No axis carries the given label.
    NoAxisNamed(String),
    /// A positional axis is >= the array's rank.
    IndexOutOfRank { index: usize, rank: usize },
    /// The same axis was selected more than once.
    DuplicateAxis(usize),
}

impl AxisSpec {
    /// Resolve this selection to concrete, distinct, in-rank axis indices.
    ///
    /// # Errors
    /// Returns an [`AxisError`] for an unlabeled array named by label, an
    /// unknown label, a positional axis outside the rank, or a repeated axis.
    pub fn resolve(&self, arr: &DenseArray) -> Result<Vec<usize>, AxisError> {
        let indices = match self {
            AxisSpec::Indices(ix) => resolve_indices(ix, arr.shape().rank())?,
            AxisSpec::Names(names) => resolve_names(names, arr.labels())?,
        };
        reject_duplicates(&indices)?;
        Ok(indices)
    }
}

/// Validate positional axes against the array rank (order preserved).
fn resolve_indices(indices: &[usize], rank: usize) -> Result<Vec<usize>, AxisError> {
    for &index in indices {
        if index >= rank {
            return Err(AxisError::IndexOutOfRank { index, rank });
        }
    }
    Ok(indices.to_vec())
}

/// Look each name up in the array's labels, preserving the given order.
fn resolve_names(
    names: &[String],
    labels: Option<&[Option<String>]>,
) -> Result<Vec<usize>, AxisError> {
    let labels = labels.ok_or(AxisError::NamedAxisButNoLabels)?;
    names
        .iter()
        .map(|name| {
            labels
                .iter()
                .position(|l| l.as_deref() == Some(name.as_str()))
                .ok_or_else(|| AxisError::NoAxisNamed(name.clone()))
        })
        .collect()
}

/// Reject a selection that names the same axis twice.
fn reject_duplicates(indices: &[usize]) -> Result<(), AxisError> {
    for (i, &axis) in indices.iter().enumerate() {
        if indices[..i].contains(&axis) {
            return Err(AxisError::DuplicateAxis(axis));
        }
    }
    Ok(())
}
