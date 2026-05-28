//! Pure BPE training + apply + decode algorithm extracted
//! from `crates/mlpl-eval/src/bpe.rs`. Saga 33 step 018.
//!
//! - `train(corpus, vocab_size) -> merges`
//! - `apply_trained(bytes, merges) -> token_ids`
//! - `decode_token(id, merges, out_bytes)`
//!
//! All pure: bytes-in, ids-out (or vice versa). No env, no
//! Value, no parser. The mlpl-eval wrapper handles the
//! Value::Str / Value::Array marshalling at the boundary.

pub mod apply;
pub mod train;

pub use apply::{apply_trained, decode_token};
pub use train::train;
