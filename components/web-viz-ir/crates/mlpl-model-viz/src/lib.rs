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
        ModelSpec::Linear { .. } => b.emit_leaf("linear", "linear", prev_id, in_residual, 64.0),
        ModelSpec::Embedding { vocab, d_model, .. } => b.emit_leaf(
            "embed",
            &format!("embed (V={vocab}, d={d_model})"),
            prev_id,
            in_residual,
            (*d_model as f64).max(1.0),
        ),
        ModelSpec::Attention { causal, .. } => {
            let n = if *causal {
                "causal_attention"
            } else {
                "attention"
            };
            b.emit_leaf(n, n, prev_id, in_residual, 48.0)
        }
        ModelSpec::RmsNorm { dim } => b.emit_leaf(
            "rms_norm",
            &format!("rms_norm ({dim})"),
            prev_id,
            in_residual,
            16.0,
        ),
        ModelSpec::Activation(kind) => {
            let label = match kind {
                ActKind::Tanh => "tanh",
                ActKind::Relu => "relu",
                ActKind::Softmax => "softmax",
            };
            b.emit_leaf("activation", label, prev_id, in_residual, 4.0)
        }
        ModelSpec::LinearLora { .. } => {
            b.emit_leaf("linear_lora", "linear (LoRA)", prev_id, in_residual, 64.0)
        }
    }
}
