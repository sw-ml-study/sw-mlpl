//! Visualization IR for the web playground's introspect dialog.
//!
//! Sits between MLPL evaluation and the dialog renderers: each
//! sculpture on the 3D stage carries an optional `VizNode` that
//! the dispatch site in `stage3d.js` reads to pick the right
//! 2D renderer (attention heatmap, composite Sankey, derivation
//! steps, ...). When absent, the dialog falls back to the
//! existing text body.
//!
//! Per `docs/viz-ir-plan.md` saga A: this crate ships the
//! types only -- no MLPL changes populate it yet (that lands
//! in sagas B+), and no renderer reads it yet either.
//!
//! Design note: `VizNode` deliberately does NOT carry the
//! tensor stats / values / element count that `ShapeInfo` (in
//! `mlpl-web-viz3d::events`) already serializes. Stats are
//! universal; viz-ir adds only the semantic payload (kind +
//! optional attention or sankey extras). Keeping the split
//! avoids a `viz-ir <-> viz3d` cycle.

pub mod attention;
pub mod axis;
pub mod kind;
pub mod node;
pub mod sankey;

pub use attention::{AttentionLayout, AttentionViz, Token};
pub use axis::AxisDim;
pub use kind::VizKind;
pub use node::VizNode;
pub use sankey::{SankeyEdge, SankeyNode, SankeyViz};
