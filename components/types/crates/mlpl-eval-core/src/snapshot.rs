//! Serializable model snapshot for `save_model` / `load_model`.
//!
//! A snapshot is everything needed to reconstruct a trained model: its
//! [`ModelSpec`](crate::ModelSpec) (architecture + the param names it
//! references) plus the current value of every one of those params.
//! Param arrays are stored as plain `{dims, data}` so this crate need
//! not depend on `mlpl-array` -- the eval layer converts to/from
//! `DenseArray` via that crate's public accessors. JSON (de)serialization
//! itself lives in the eval layer, which already carries `serde_json`.

use crate::ModelSpec;
use serde::{Deserialize, Serialize};

/// One parameter's value: flat data plus its shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamEntry {
    /// Environment name the model's spec references this param by.
    pub name: String,
    /// Row-major shape.
    pub dims: Vec<usize>,
    /// Flat row-major values.
    pub data: Vec<f64>,
}

/// A persisted model: architecture + all its parameter values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    /// Format version, for forward-compatible loading.
    pub version: u32,
    /// The model architecture (also lists the param names).
    pub spec: ModelSpec,
    /// Every param the spec owns, in `spec.params()` order.
    pub params: Vec<ParamEntry>,
}

impl ModelSnapshot {
    /// Current snapshot format version.
    pub const VERSION: u32 = 1;
}
