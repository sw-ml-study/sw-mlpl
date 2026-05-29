//! Sankey-flow payload for composite objects (chain, residual,
//! larger model wrappers). The evaluator builds this when a
//! sculpture represents a composite; renderers (saga D and
//! later) hand it to Plotly's Sankey trace type without
//! further transformation.
//!
//! Edge widths carry a numeric magnitude (tensor element
//! count, token count, |grad|, ...) so visual ribbon thickness
//! reads as "amount of stuff flowing through this op". The
//! `width` field is `f64` rather than `usize` to support
//! gradient magnitudes in the future backward-pass renderer.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SankeyViz {
    pub nodes: Vec<SankeyNode>,
    pub edges: Vec<SankeyEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SankeyNode {
    pub id: String,
    /// Pretty label rendered on the Sankey node, e.g.
    /// `"embed (V=280, d=32)"`.
    pub label: String,
    /// Coarse op family ("embed", "attention", "linear",
    /// "softmax", ...). Drives node coloring + filtering.
    pub op_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SankeyEdge {
    pub from: String,
    pub to: String,
    /// Numeric magnitude (typically post-op tensor element
    /// count). Renderers normalize to a visual ribbon width.
    pub width: f64,
    /// Optional shape string shown on hover ("[batch=32, d=32]").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}
