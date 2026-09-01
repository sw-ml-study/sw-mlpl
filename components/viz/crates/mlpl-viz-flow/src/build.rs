//! Validate a [`Dataflow`] spec into a typed [`Graph`]. The MLPL-record
//! extraction (record fields -> the spec slices) lives one layer up in
//! `mlpl-eval`; this crate stays free of the interpreter value model.

use mlpl_viz_core::VizError;

use crate::Dataflow;
use crate::model::Graph;

/// Build a validated graph, or a clean `VizError` (never a panic) on an
/// empty graph, a length mismatch, or an out-of-range endpoint.
pub fn build(d: &Dataflow) -> Result<Graph, VizError> {
    validate_edges(d)?;
    validate_channels(d)?;
    Ok(Graph {
        labels: d.labels.to_vec(),
        edges: d.from.iter().copied().zip(d.to.iter().copied()).collect(),
        edge_labels: d.edge_labels.to_vec(),
        groups: d.groups.to_vec(),
        node_highlight: d.node_highlight.to_vec(),
        edge_widths: d.edge_widths.to_vec(),
        edge_highlight: d.edge_highlight.to_vec(),
        width_log: d.width_log,
    })
}

fn bad<T>(m: String) -> Result<T, VizError> {
    Err(VizError::InvalidShape(m))
}

/// A non-empty node set, matching `from`/`to` lengths, a `0`-or-per-edge
/// label count, and every endpoint in range.
fn validate_edges(d: &Dataflow) -> Result<(), VizError> {
    if d.labels.is_empty() {
        return bad("dataflow needs at least one node".into());
    }
    if d.from.len() != d.to.len() {
        return bad(format!(
            "dataflow edges: from has {} entries, to has {}",
            d.from.len(),
            d.to.len()
        ));
    }
    check_len("edge labels", d.edge_labels.len(), d.from.len())?;
    if let Some(id) = d.from.iter().chain(d.to).find(|&&id| id >= d.labels.len()) {
        return bad(format!(
            "dataflow edge endpoint {id} is out of range (only {} nodes)",
            d.labels.len()
        ));
    }
    Ok(())
}

/// Each optional channel is empty or exactly one-per-node / one-per-edge.
fn validate_channels(d: &Dataflow) -> Result<(), VizError> {
    check_len("groups", d.groups.len(), d.labels.len())?;
    check_len("node highlight", d.node_highlight.len(), d.labels.len())?;
    check_len("edge widths", d.edge_widths.len(), d.from.len())?;
    check_len("edge highlight", d.edge_highlight.len(), d.from.len())
}

/// An optional channel is valid when empty or exactly `expected` long.
fn check_len(what: &str, got: usize, expected: usize) -> Result<(), VizError> {
    if got == 0 || got == expected {
        Ok(())
    } else {
        bad(format!("dataflow {what}: {got} entries for {expected}"))
    }
}
