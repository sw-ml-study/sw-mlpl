//! Saga 76: free-function eval helpers extracted from lib.rs.

use mlpl_eval::{Environment, ModelSpec};
use mlpl_web_viz_ir::{SankeyEdge, SankeyNode, SankeyViz, VizKind, VizNode};

pub struct EvalResult {
    pub display: String,
    pub values: Option<Vec<f64>>,
    pub shape: Vec<usize>,
    /// Saga BPE-1: populated when the evaluated value is a
    /// `Value::StrList` (produced by `decode_each(...)` or
    /// `[...]` string literals). Lets the viz layer attach
    /// per-token labels to attention sculptures.
    pub string_list: Option<Vec<String>>,
    /// Saga D: populated when the evaluated value is a
    /// `Value::Model`. Carries a pre-built `VizNode` with a
    /// composite-Sankey payload describing the chain's flow
    /// (one node per layer, one edge per layer-to-layer
    /// connection). The JS dispatch site hands it to the
    /// `composite` renderer.
    pub viz_node: Option<VizNode>,
}

pub(crate) fn eval_input_with_values(input: &str, env: &mut Environment) -> EvalResult {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return empty_result();
    }
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return text_result(out);
    }
    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return text_result(format!("error: {e}")),
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return empty_result(),
        Ok(s) => s,
        Err(e) => return text_result(format!("error: {e}")),
    };
    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => value_to_result(value),
        Err(e) => text_result(format!("error: {e}")),
    }
}

fn empty_result() -> EvalResult {
    EvalResult {
        display: String::new(),
        values: None,
        shape: vec![],
        string_list: None,
        viz_node: None,
    }
}

fn text_result(display: String) -> EvalResult {
    EvalResult {
        display,
        values: None,
        shape: vec![],
        string_list: None,
        viz_node: None,
    }
}

fn value_to_result(value: mlpl_eval::Value) -> EvalResult {
    match value {
        mlpl_eval::Value::Array(ref arr) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Array(arr.clone())),
            values: Some(arr.data().to_vec()),
            shape: arr.shape().dims().to_vec(),
            string_list: None,
            viz_node: None,
        },
        mlpl_eval::Value::StrList { ref items } => EvalResult {
            display: format!("{}", mlpl_eval::Value::StrList { items: items.clone() }),
            values: None,
            shape: vec![items.len()],
            string_list: Some(items.clone()),
            viz_node: None,
        },
        mlpl_eval::Value::Model(ref spec) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Model(spec.clone())),
            values: None,
            shape: vec![],
            string_list: None,
            viz_node: Some(model_to_viz_node(spec)),
        },
        other => text_result(format!("{other}")),
    }
}

/// Saga D: walk a `ModelSpec` into a `VizNode { kind:
/// Composite, sankey: SankeyViz }`. Sequential composition
/// (`Chain`) flattens into one ribbon; `Residual(inner)`
/// annotates a passthrough on the inner walk (no bypass edge
/// in v0 -- the visual gets busy fast). Leaf layers
/// (Linear/Embed/Attention/RmsNorm/Activation) become one
/// node each. Edge width = a coarse parameter-count proxy so
/// the bottleneck in an autoencoder reads visually.
fn model_to_viz_node(spec: &ModelSpec) -> VizNode {
    let mut nodes = vec![SankeyNode {
        id: "input".to_string(),
        label: "input".to_string(),
        op_kind: "input".to_string(),
    }];
    let mut edges = Vec::new();
    let mut counter = 0usize;
    let last_id = walk_model(spec, "input", &mut nodes, &mut edges, &mut counter, false);
    nodes.push(SankeyNode {
        id: "output".to_string(),
        label: "output".to_string(),
        op_kind: "output".to_string(),
    });
    edges.push(SankeyEdge {
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
        sankey: Some(SankeyViz { nodes, edges }),
    }
}

fn walk_model(
    spec: &ModelSpec,
    prev_id: &str,
    nodes: &mut Vec<SankeyNode>,
    edges: &mut Vec<SankeyEdge>,
    counter: &mut usize,
    in_residual: bool,
) -> String {
    match spec {
        ModelSpec::Chain(children) => {
            let mut cur = prev_id.to_string();
            for child in children {
                cur = walk_model(child, &cur, nodes, edges, counter, in_residual);
            }
            cur
        }
        ModelSpec::Residual(inner) => {
            // v0: flatten residual into its inner chain; tag
            // the resulting nodes so the renderer can stripe
            // them. The bypass edge is deferred.
            walk_model(inner, prev_id, nodes, edges, counter, true)
        }
        ModelSpec::Linear { .. } => emit_leaf(
            "linear",
            "linear",
            prev_id,
            nodes,
            edges,
            counter,
            in_residual,
            64.0,
        ),
        ModelSpec::Embedding { vocab, d_model, .. } => emit_leaf(
            "embed",
            &format!("embed (V={vocab}, d={d_model})"),
            prev_id,
            nodes,
            edges,
            counter,
            in_residual,
            (*d_model as f64).max(1.0),
        ),
        ModelSpec::Attention { causal, .. } => emit_leaf(
            if *causal { "causal_attention" } else { "attention" },
            if *causal { "causal_attention" } else { "attention" },
            prev_id,
            nodes,
            edges,
            counter,
            in_residual,
            48.0,
        ),
        ModelSpec::RmsNorm { dim } => emit_leaf(
            "rms_norm",
            &format!("rms_norm ({dim})"),
            prev_id,
            nodes,
            edges,
            counter,
            in_residual,
            16.0,
        ),
        ModelSpec::Activation(kind) => {
            let label = match kind {
                mlpl_eval::ActKind::Tanh => "tanh",
                mlpl_eval::ActKind::Relu => "relu",
                mlpl_eval::ActKind::Softmax => "softmax",
            };
            emit_leaf("activation", label, prev_id, nodes, edges, counter, in_residual, 4.0)
        }
        // LoRA-augmented linear layer: same Sankey shape as a
        // plain Linear for v0 (a future saga can dedicated-render
        // the side-branch low-rank adapter).
        ModelSpec::LinearLora { .. } => emit_leaf(
            "linear_lora",
            "linear (LoRA)",
            prev_id,
            nodes,
            edges,
            counter,
            in_residual,
            64.0,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_leaf(
    op_kind: &str,
    label: &str,
    prev_id: &str,
    nodes: &mut Vec<SankeyNode>,
    edges: &mut Vec<SankeyEdge>,
    counter: &mut usize,
    in_residual: bool,
    width: f64,
) -> String {
    *counter += 1;
    let id = format!("{op_kind}_{}", *counter);
    let suffix = if in_residual { " *res" } else { "" };
    nodes.push(SankeyNode {
        id: id.clone(),
        label: format!("{label}{suffix}"),
        op_kind: op_kind.to_string(),
    });
    edges.push(SankeyEdge {
        from: prev_id.to_string(),
        to: id.clone(),
        width,
        label: None,
    });
    id
}

pub(crate) fn eval_input(input: &str, env: &mut Environment) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Introspection commands (`:vars`, `:models`, `:fns`, `:wsid`,
    // `:describe <name>`) short-circuit evaluation so they work
    // identically in the terminal and web REPLs.
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return out;
    }

    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return format!("error: {e}"),
    };

    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return String::new(),
        Ok(s) => s,
        Err(e) => return format!("error: {e}"),
    };

    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => format!("{value}"),
        Err(e) => format!("error: {e}"),
    }
}
