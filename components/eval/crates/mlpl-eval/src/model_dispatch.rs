//! Built-in dispatch for the Saga 11 model DSL.
//!
//! Saga 33 step 004 split the 905-line file into topical
//! siblings; this file is now a facade that re-exports the
//! `pub` API expected by `eval.rs`.
//!
//! Split: `model_eval_layers` (`linear`, `embed`, `rms_norm`),
//! `model_eval_compose` (`chain`, `residual`, `activation_kind`),
//! `model_eval_attention` (`attention` / `causal_attention`),
//! `model_eval_apply` (`apply`, `predict_batch`,
//! `attention_weights`, `check_device_agreement`),
//! `model_dispatch_scalar` (shared scalar extraction),
//! `model_apply` + siblings (forward-pass implementation), and
//! `model_attn_weights` (read-only attention-weight extraction).

pub(crate) use crate::model_eval_apply::{eval_apply, eval_attention_weights, eval_predict_batch};
pub(crate) use crate::model_eval_attention::eval_attention;
pub(crate) use crate::model_eval_compose::{activation_kind, eval_chain, eval_residual};
pub(crate) use crate::model_eval_layers::{eval_embedding, eval_linear, eval_rms_norm};
