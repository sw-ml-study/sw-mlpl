//! `dataflow(nodes, edges)` -- a structural SVG renderer for layered
//! DAGs (boxes, directed edges, edge labels), sibling to the
//! quantitative chart marks. The MLPL-record extraction lives in
//! `mlpl-eval`; this crate takes plain slices so it never depends on
//! the interpreter value model. See docs/dataflow-renderer-design.md.

mod build;
mod layout;
mod model;
mod render;

pub use mlpl_viz_core::VizError;

/// Render a dataflow diagram to an SVG string. `labels` names each node
/// (index = node id); `from[i] -> to[i]` are the directed edges;
/// `edge_labels` is empty (no labels) or one string per edge. Errors on
/// an empty graph, a length mismatch, or an out-of-range endpoint.
pub fn render_dataflow(
    labels: &[String],
    from: &[usize],
    to: &[usize],
    edge_labels: &[String],
) -> Result<String, VizError> {
    let graph = build::build(labels, from, to, edge_labels)?;
    Ok(render::render(&layout::layout(graph)))
}
