//! `lora(m, rank, alpha, seed)`: wrap every `Linear` in `m`
//! with trainable low-rank adapters. Saga 33 step 014.
//!
//! Reuses `mlpl_models_mutate::clone_spec` to deep-clone the
//! source tree, then walks the clone and rewrites each Linear
//! into a `ModelSpec::LinearLora` with freshly allocated A/B
//! adapter parameters. Auto-freezes every non-adapter param so
//! `adam(loss, lora_m, ...)` only moves the adapters.
//!
//! Forward + autograd for `LinearLora` lives in the existing
//! `mlpl-models-apply` (forward) and `mlpl-models-tape` (autograd)
//! crates -- this crate is *only* the structural rewrite.
//!
//! C+D loose-coupling boundary: generic over the env capability
//! traits the rewriter actually needs; eval-loop deps injected
//! as caller-supplied closures; local `TuneError` vocabulary.

pub mod error;
pub mod lora;
pub mod lora_helpers;

pub use error::TuneError;
pub use lora::lora_inner;
