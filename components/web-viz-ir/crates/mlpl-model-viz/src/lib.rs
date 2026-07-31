//! Translate a `ModelSpec` into a viz-IR `VizNode` (a Composite
//! Sankey diagram) for the model-decomposition view. Shared by
//! mlpl-wasm (local eval) and mlpl-serve (connect-mode eval) so a
//! model evaluated either place renders the same Sankey -- part of
//! "the 3D view shows everything regardless of where work ran"
//! (Phase 1c part 2). Extracted from mlpl-wasm's eval_impl.
//!
//! Sequential composition (`Chain`) flattens into one ribbon;
//! `Residual(inner)` annotates a passthrough on the inner walk (no
//! bypass edge in v0). Leaf layers become one node each; edge width
//! is a coarse parameter-count proxy so a bottleneck reads visually.

use mlpl_eval::{ActKind, ModelSpec};
use mlpl_web_viz_ir::{SankeyEdge, SankeyNode, SankeyViz, VizKind, VizNode};

/// Accumulates the Sankey graph as the model tree is walked. Bundles
/// what used to be three positional `&mut` args (nodes/edges/counter)
/// so `emit_leaf` stays within the argument budget.
struct SankeyBuilder {
    nodes: Vec<SankeyNode>,
    edges: Vec<SankeyEdge>,
    counter: usize,
}

impl SankeyBuilder {
    fn emit_leaf(
        &mut self,
        op_kind: &str,
        label: &str,
        prev_id: &str,
        in_residual: bool,
        width: f64,
    ) -> String {
        self.counter += 1;
        let id = format!("{op_kind}_{}", self.counter);
        let suffix = if in_residual { " *res" } else { "" };
        self.nodes.push(SankeyNode {
            id: id.clone(),
            label: format!("{label}{suffix}"),
            op_kind: op_kind.to_string(),
        });
        self.edges.push(SankeyEdge {
            from: prev_id.to_string(),
            to: id.clone(),
            width,
            label: None,
        });
        id
    }
}

/// Walk a `ModelSpec` into a Composite `VizNode` carrying a Sankey.
pub fn model_to_viz_node(spec: &ModelSpec) -> VizNode {
    let mut b = SankeyBuilder {
        nodes: vec![SankeyNode {
            id: "input".to_string(),
            label: "input".to_string(),
            op_kind: "input".to_string(),
        }],
        edges: Vec::new(),
        counter: 0,
    };
    let last_id = walk_model(spec, "input", &mut b, false);
    b.nodes.push(SankeyNode {
        id: "output".to_string(),
        label: "output".to_string(),
        op_kind: "output".to_string(),
    });
    b.edges.push(SankeyEdge {
        from: last_id,
        to: "output".to_string(),
        width: 1.0,
        label: None,
    });
    VizNode {
        id: "model".to_string(),
        name: None,
        label: "model".to_string(),
        kind: VizKind::Composite,
        shape: vec![],
        producer: None,
        attention: None,
        sankey: Some(SankeyViz {
            nodes: b.nodes,
            edges: b.edges,
        }),
    }
}

fn walk_model(spec: &ModelSpec, prev_id: &str, b: &mut SankeyBuilder, in_residual: bool) -> String {
    match spec {
        ModelSpec::Chain(children) => {
            let mut cur = prev_id.to_string();
            for child in children {
                cur = walk_model(child, &cur, b, in_residual);
            }
            cur
        }
        ModelSpec::Residual(inner) => walk_model(inner, prev_id, b, true),
        leaf => {
            let (op_kind, label, width) = leaf_meta(leaf);
            b.emit_leaf(op_kind, &label, prev_id, in_residual, width)
        }
    }
}

/// (op_kind, label, ribbon width) for a leaf layer. `Chain` and
/// `Residual` are composites handled by `walk_model`; the exhaustive
/// match here is what forces a new `ModelSpec` variant to pick its
/// Sankey appearance.
fn leaf_meta(spec: &ModelSpec) -> (&'static str, String, f64) {
    match spec {
        ModelSpec::Chain(_) | ModelSpec::Residual(_) => {
            unreachable!("composites are walked, not emitted as leaves")
        }
        ModelSpec::Linear { .. } => ("linear", "linear".into(), 64.0),
        ModelSpec::Embedding { vocab, d_model, .. } => (
            "embed",
            format!("embed (V={vocab}, d={d_model})"),
            (*d_model as f64).max(1.0),
        ),
        ModelSpec::Attention { causal, .. } => {
            let n = if *causal {
                "causal_attention"
            } else {
                "attention"
            };
            (n, n.to_string(), 48.0)
        }
        ModelSpec::RmsNorm { dim } => ("rms_norm", format!("rms_norm ({dim})"), 16.0),
        ModelSpec::Activation(kind) => {
            let label = match kind {
                ActKind::Tanh => "tanh",
                ActKind::Relu => "relu",
                ActKind::Softmax => "softmax",
            };
            ("activation", label.to_string(), 4.0)
        }
        ModelSpec::LinearLora { .. } => ("linear_lora", "linear (LoRA)".into(), 64.0),
        ModelSpec::Engram {
            ngram_orders,
            heads,
            slots,
            ..
        } => (
            "engram",
            format!("engram ({}x{heads}x{slots})", ngram_orders.len()),
            32.0,
        ),
    }
}
