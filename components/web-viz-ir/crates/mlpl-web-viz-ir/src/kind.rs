//! The discriminator the dialog dispatches on when picking a
//! renderer. Saga A ships every variant the saga plan currently
//! anticipates; sagas B-G each add one matching renderer and
//! populate one variant. `Other` is the escape hatch for
//! one-off hints from MLPL code (e.g. `viz_hint("my-thing")`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VizKind {
    /// Scalar / vector / matrix tensor with values + stats.
    /// Default; renderer is today's text body.
    Tensor,
    /// Rank-2/3/4 matrix the dialog should render as an
    /// attention heatmap. Carries an `AttentionViz` payload on
    /// the parent `VizNode`.
    Attention,
    /// Composite object (chain, residual, model). Carries a
    /// `SankeyViz` payload.
    Composite,
    /// Loss / metric series. Reserved for the loss-curve
    /// renderer in a future saga.
    Series,
    /// 2D projection of high-dim points (PCA / UMAP / t-SNE).
    /// Reserved for the projection renderer.
    Projection,
}
