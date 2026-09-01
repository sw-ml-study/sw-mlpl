//! `dataflow(nodes, edges)` -- a structural SVG renderer for layered
//! DAGs (boxes, directed edges, edge labels, group bands, edge widths,
//! node/edge highlight), sibling to the quantitative chart marks. The
//! MLPL-record extraction lives in `mlpl-eval`; this crate takes plain
//! slices so it never depends on the interpreter value model. See
//! docs/dataflow-renderer-design.md.

mod build;
mod groups;
mod layout;
mod model;
mod render;
mod widths;

pub use mlpl_viz_core::VizError;

/// Everything a dataflow diagram needs, as columnar slices. `labels`
/// names each node (index = id) and `from[i] -> to[i]` are the directed
/// edges; every other field is an OPTIONAL channel that is either empty
/// (unused) or exactly one-per-node / one-per-edge.
#[derive(Default)]
pub struct Dataflow<'a> {
    /// One box label per node.
    pub labels: &'a [String],
    /// Edge source ids.
    pub from: &'a [usize],
    /// Edge target ids.
    pub to: &'a [usize],
    /// Per-edge label, or empty.
    pub edge_labels: &'a [String],
    /// Group id per node (banded together), or empty.
    pub groups: &'a [usize],
    /// Per-node highlight flag, or empty.
    pub node_highlight: &'a [bool],
    /// Per-edge stroke width, or empty for the default.
    pub edge_widths: &'a [f64],
    /// Per-edge highlight flag, or empty.
    pub edge_highlight: &'a [bool],
    /// Interpret `edge_widths` on a log scale so extreme ratios read as
    /// an orders-of-magnitude contrast; default is clamped-linear.
    pub width_log: bool,
}

/// Render a [`Dataflow`] to an SVG string. Errors (never panics) on an
/// empty graph, a length mismatch, or an out-of-range endpoint.
pub fn render_dataflow(d: &Dataflow) -> Result<String, VizError> {
    let graph = build::build(d)?;
    Ok(render::render(&layout::layout(graph)))
}
