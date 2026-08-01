//! Backend-neutral Engram core (saga E1, docs/engram-sagas-plan.md):
//! the spec + validation, memory accounting for `:describe`, and the
//! exact-arithmetic n-gram hash REFERENCE that every backend -- CPU
//! tensor pipeline, MLX (saga E5), later CUDA (saga E10) -- must
//! reproduce bit-for-bit. No tensor types live here.

pub mod accounting;
pub mod hash;
pub mod spec;

/// Largest token id the exact-arithmetic hash contract accepts
/// (2^21 - 1): keeps every `id x multiplier` product below 2^52 so
/// f64 tensor pipelines match the u64 reference exactly. Covers
/// every real tokenizer vocabulary by a wide margin.
pub const MAX_TOKEN_ID: u64 = (1 << 21) - 1;

pub use accounting::DType;
pub use hash::{
    AddressingStats, HASH_PRIME, HashSpec, addressing_stats, hash_multipliers, head_offset,
    ngram_hashes,
};
pub use spec::{EngramError, EngramSpec};
