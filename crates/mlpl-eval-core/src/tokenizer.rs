//! Tokenizer spec types (Saga 33 step 019: moved from
//! `mlpl-eval/src/tokenizer.rs`). Sibling to `ModelSpec` --
//! both are leaf data types that compose into the larger
//! `Value` variant set. Lives here so downstream crates
//! (including future `Value` consumers) can use it without
//! pulling in the full mlpl-eval crate.

/// Internal representation of a tokenizer. Sibling to
/// `ModelSpec`.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizerSpec {
    /// Identity byte-level tokenizer: each byte 0..256 is its
    /// own token. Vocab size is implicitly 256.
    ByteLevel,
    /// Trained byte-level BPE tokenizer (Saga 12 step 005).
    /// Vocab starts at 256 bytes; each merge adds one entry
    /// with the next free id. `merges[i]` = `(left_id,
    /// right_id)` pair that produced the new token at id
    /// `256 + i`.
    BpeMerges {
        /// Ordered list of `(left_id, right_id)` merges. Apply
        /// in training order (step 006 spec).
        merges: Vec<(u32, u32)>,
        /// Total vocab size = 256 + merges.len().
        vocab_size: u32,
        /// Number of bytes in the training corpus.
        corpus_byte_count: usize,
        /// Seed threaded through to training (currently
        /// unused by the deterministic algorithm; reserved
        /// for future randomized sub-sampling at larger
        /// scales).
        seed: u64,
    },
}

impl TokenizerSpec {
    /// Human-readable one-line description used by `:describe`.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::ByteLevel => "byte-level tokenizer (vocab=256)".into(),
            Self::BpeMerges {
                merges,
                vocab_size,
                corpus_byte_count,
                seed,
            } => format!(
                "BPE tokenizer (vocab={vocab_size}, merges={}, trained from {corpus_byte_count} bytes, seed={seed})",
                merges.len()
            ),
        }
    }
}
