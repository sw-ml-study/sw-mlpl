//! Recognizer for the tic-tac-toe board-policy LoRA model
//! (`Chain[LinearLora, relu, LinearLora]` -- the shape
//! `lora(chain(linear, relu_layer(), linear))` produces): the per-layer
//! frozen-base + adapter param names the GPU fine-tune step needs. This
//! recognizer is device-agnostic and shared by both backends; the
//! on-device MLP step lives in the sibling compute crates
//! (`mlpl-mlx-eval::mlx_mlp`, `mlpl-cuda-eval::cuda_mlp`).

use crate::env::Environment;
use mlpl_eval_core::ModelSpec;
use mlpl_parser::Expr;

// The names type moved to mlpl-eval-state (env-types-out step);
// re-exported so existing paths keep working. The RECOGNIZER below
// stays here -- it reads models and is interpreter-coupled.
pub use mlpl_eval_state::LoraNames;

/// Recognize `Chain[LinearLora, Activation(Relu), LinearLora]` and
/// return both layers' param names; `None` for any other model.
pub(crate) fn mlp_layout(model_arg: &Expr, env: &Environment) -> Option<(LoraNames, LoraNames)> {
    let Expr::Ident(name, _) = model_arg else {
        return None;
    };
    let ModelSpec::Chain(ls) = env.get_model(name)? else {
        return None;
    };
    match ls.as_slice() {
        [l1, mid, l2] if is_relu(mid) => Some((lora_names(l1)?, lora_names(l2)?)),
        _ => None,
    }
}

fn is_relu(spec: &ModelSpec) -> bool {
    matches!(spec, ModelSpec::Activation(mlpl_eval_core::ActKind::Relu))
}

fn lora_names(spec: &ModelSpec) -> Option<LoraNames> {
    let ModelSpec::LinearLora {
        w,
        b,
        a,
        b_adapter,
        alpha,
        rank,
        ..
    } = spec
    else {
        return None;
    };
    Some(LoraNames {
        w: w.clone(),
        b: b.clone(),
        a: a.clone(),
        b_adapter: b_adapter.clone(),
        scale: (alpha / *rank as f64) as f32,
    })
}
