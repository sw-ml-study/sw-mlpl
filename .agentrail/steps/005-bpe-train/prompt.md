Phase 2 step 005: BPE trainer. Add train_bpe(corpus, vocab_size, seed) returning Value::Tokenizer with a BpeMerges variant.

Algorithm: byte-level Byte-Pair Encoding. Start with a 256-entry vocab (one per byte). At each merge step, count adjacent token pairs across the corpus, pick the most frequent pair, add it as a new vocab entry with the next free id, and rewrite the corpus replacing that pair. Stop when either vocab_size is reached or no pair has frequency >= 2.

Deterministic tie-breaking: on ties in pair count, pick the lexicographically smallest pair, comparing as (left_id, right_id) tuples. This keeps training fully deterministic given the same corpus.

Seed threading: unused in the core algorithm (which is deterministic), but accepted and stored in the resulting tokenizer's metadata for later sub-sampling use (document that the current implementation uses the seed only for any future randomized pre-processing; training is deterministic).

corpus argument: accepts either a Value::Str (treated as UTF-8 bytes) or a rank-1 DenseArray of byte indices (pre-tokenized output of tokenize_bytes).

TokenizerSpec::BpeMerges { merges: Vec<(u32, u32)>, vocab_size: u32, corpus_byte_count: usize, seed: u64 }. Value::Tokenizer wrapping this variant.

:describe prints something like "BPE tokenizer (vocab=1024, merges=768, trained from 49152 bytes, seed=42)".

TDD:
(1) train_bpe(s, 260, 7) on a tiny string s ends with vocab_size >= 256 and <= 260; merges table has 0..=4 entries.
(2) re-running train_bpe with the same input + seed produces identical merges list (determinism).
(3) a constructed pathological tie case exercises the lexicographic tie-breaker and matches expected ordering.
(4) early exit when no pair has count >= 2.
(5) passing a Value::Str and the equivalent pre-tokenized rank-1 array produce identical results.
(6) :describe on a trained BPE tokenizer includes vocab_size and merge count.
