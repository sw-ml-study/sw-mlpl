//! The Model DSL cluster (eval decomposition cluster peel 1;
//! docs/eval-env-design.md): layer constructors, apply/forward
//! lowering, attention weights, `LoRA` composition, freeze/mutate/io,
//! inspection, and feasibility checks. Split from the mlpl-eval hub;
//! the evaluator and device dispatch are reached through the
//! mlpl-eval-env hooks (installed at eval entry).

pub mod model_apply;
pub mod model_apply_attention;
pub mod model_apply_compose;
pub mod model_apply_embed;
pub mod model_apply_engram;
pub(crate) mod model_apply_engram_chain;
pub mod model_apply_lora;
pub mod model_apply_simple;
pub mod model_attn_weights;
pub mod model_dispatch;
pub mod model_dispatch_scalar;
pub(crate) mod model_engram_math;
pub mod model_eval_apply;
pub mod model_eval_attention;
pub mod model_eval_compose;
pub mod model_eval_engram;
pub mod model_eval_layers;
pub mod model_feasibility;
pub mod model_freeze;
pub mod model_inspect;
pub mod model_io;
pub mod model_lora;
pub mod model_mutate;

pub use model_apply_embed::tokens_to_onehot;

/// Capability-trait prelude, mirroring the hub's `env_api` so the
/// moved modules keep their `use crate::env_api::*;` imports.
pub(crate) mod env_api {
    #[allow(unused_imports)]
    pub use mlpl_eval_envc_bindings::*;
    #[allow(unused_imports)]
    pub use mlpl_eval_envc_exec::*;
    #[allow(unused_imports)]
    pub use mlpl_eval_envc_ml::*;
    #[allow(unused_imports)]
    pub use mlpl_eval_envc_obs::*;
}
