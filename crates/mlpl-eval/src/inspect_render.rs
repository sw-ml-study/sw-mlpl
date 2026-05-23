//! Saga 32 step 005: rendering helpers extracted from
//! `inspect.rs` so the orchestrator stays under the
//! sw-checklist function-count budget. Pure string-shaping
//! helpers; no env mutation.

use mlpl_array::DenseArray;
use mlpl_core::LabeledShape;
use mlpl_eval_core::inspect_groups::BUILTIN_GROUPS;
use mlpl_eval_core::model::{ActKind, ModelSpec};

pub(crate) fn format_builtins() -> String {
    let mut out = String::new();
    for (group, fns) in BUILTIN_GROUPS {
        out.push_str(group);
        out.push('\n');
        for (_, sig, doc) in *fns {
            out.push_str(&format!("  {sig:<40} {doc}\n"));
        }
        out.push('\n');
    }
    out.truncate(out.trim_end().len());
    out
}

pub(crate) fn format_shape(arr: &DenseArray) -> String {
    let dims = arr.shape().dims();
    if dims.is_empty() {
        return "scalar".into();
    }
    // Labeled (fully or partially) arrays render through `LabeledShape`
    // Display: `[seq=6, d_model=4]`, `[6, d_model=4]`. Unlabeled
    // arrays keep the positional `[6, 4]` rendering so existing
    // :vars/:describe output is unchanged for pre-labels demos.
    if let Some(labels) = arr.labels() {
        return LabeledShape::new(dims.to_vec(), labels.to_vec()).to_string();
    }
    let inner: Vec<String> = dims.iter().map(usize::to_string).collect();
    format!("[{}]", inner.join(", "))
}

pub(crate) fn render_spec(spec: &ModelSpec) -> String {
    match spec {
        ModelSpec::Linear { .. } => "linear".into(),
        ModelSpec::Chain(children) => {
            let parts: Vec<String> = children.iter().map(render_spec).collect();
            format!("chain({})", parts.join(" -> "))
        }
        ModelSpec::Activation(k) => match k {
            ActKind::Tanh => "tanh".into(),
            ActKind::Relu => "relu".into(),
            ActKind::Softmax => "softmax".into(),
        },
        ModelSpec::Residual(inner) => format!("residual({})", render_spec(inner)),
        ModelSpec::RmsNorm { dim } => format!("rms_norm({dim})"),
        ModelSpec::Attention {
            d_model,
            heads,
            causal,
            ..
        } => {
            let name = if *causal {
                "causal_attention"
            } else {
                "attention"
            };
            format!("{name}(d={d_model}, heads={heads})")
        }
        ModelSpec::Embedding { vocab, d_model, .. } => {
            format!("embed[vocab={vocab}, d={d_model}]")
        }
        ModelSpec::LinearLora {
            in_dim,
            out_dim,
            rank,
            alpha,
            ..
        } => format!("lora[linear({in_dim} -> {out_dim}), rank={rank}, alpha={alpha}]"),
    }
}
