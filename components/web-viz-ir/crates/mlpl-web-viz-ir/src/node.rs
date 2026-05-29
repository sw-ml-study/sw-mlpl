//! The IR root: one `VizNode` per inspected sculpture. Plugs
//! into `mlpl-web-viz3d::events::ShapeInfo` via an
//! `Option<VizNode>` field so the JS dispatch site can read
//! `userData.viz?.kind` and pick a renderer.
//!
//! `VizNode` deliberately omits `stats` / `values` / `elements`
//! -- those stay on `ShapeInfo` (which the dispatcher already
//! sees). This split keeps `viz-ir` a pure leaf crate (no dep
//! on `viz3d`) and avoids a cycle.

use serde::{Deserialize, Serialize};

use crate::attention::AttentionViz;
use crate::axis::AxisDim;
use crate::kind::VizKind;
use crate::sankey::SankeyViz;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VizNode {
    /// Unique within a trace; renderers may reference siblings
    /// (Sankey breadcrumbs, derivation-step back-links).
    pub id: String,
    /// User-typed variable name (`A`, `Q`, `scores`, ...) when
    /// the tensor was bound to one; `None` for intermediates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Pretty title for the dialog header.
    pub label: String,
    pub kind: VizKind,
    /// Per-axis sizes + optional labels. Mirrors `ShapeInfo`'s
    /// raw `Vec<usize>` but adds the semantic labels.
    pub shape: Vec<AxisDim>,
    /// Source expression that produced this tensor
    /// (`"softmax(scores, 1)"`). Renderers heuristic-detect
    /// the semantic role from this when no `viz_hint` was
    /// provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    /// Populated when `kind == VizKind::Attention`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionViz>,
    /// Populated when `kind == VizKind::Composite`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sankey: Option<SankeyViz>,
}
