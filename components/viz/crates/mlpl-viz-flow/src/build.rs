//! Validate plain slices into a typed [`Graph`]. The MLPL-record
//! extraction (record fields -> these slices) lives one layer up in
//! `mlpl-eval`; this crate stays free of the interpreter value model.

use mlpl_viz_core::VizError;

use crate::model::Graph;

/// Build a validated graph from node labels, edge endpoints, and
/// optional per-edge labels. Errors (never panics) on an empty graph,
/// a `from`/`to`/`edge_labels` length mismatch, or an out-of-range
/// endpoint.
pub fn build(
    labels: &[String],
    from: &[usize],
    to: &[usize],
    edge_labels: &[String],
) -> Result<Graph, VizError> {
    validate(labels, from, to, edge_labels)?;
    Ok(Graph {
        labels: labels.to_vec(),
        edges: from.iter().copied().zip(to.iter().copied()).collect(),
        edge_labels: edge_labels.to_vec(),
    })
}

/// The four validity checks: at least one node, matching `from`/`to`
/// lengths, an edge-label count that is 0 or one-per-edge, and every
/// endpoint in range.
fn validate(
    labels: &[String],
    from: &[usize],
    to: &[usize],
    edge_labels: &[String],
) -> Result<(), VizError> {
    let bad = |m: String| Err(VizError::InvalidShape(m));
    if labels.is_empty() {
        return bad("dataflow needs at least one node".into());
    }
    if from.len() != to.len() {
        return bad(format!(
            "dataflow edges: from has {} entries, to has {}",
            from.len(),
            to.len()
        ));
    }
    if !edge_labels.is_empty() && edge_labels.len() != from.len() {
        return bad(format!(
            "dataflow edge labels: {} labels for {} edges",
            edge_labels.len(),
            from.len()
        ));
    }
    if let Some(id) = from.iter().chain(to).find(|&&id| id >= labels.len()) {
        return bad(format!(
            "dataflow edge endpoint {id} is out of range (only {} nodes)",
            labels.len()
        ));
    }
    Ok(())
}
