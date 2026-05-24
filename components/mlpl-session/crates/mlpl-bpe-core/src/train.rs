//! BPE training: bytes -> ordered (left, right) merges.
//!
//! Greedy algorithm: start with one token per byte (ids 0..256),
//! at each step count adjacent pairs, pick the most frequent
//! (lex-smallest tiebreak), add it as a new token id, rewrite
//! the sequence, repeat until vocab is reached or no adjacent
//! pairs remain.

use std::collections::HashMap;

/// Train BPE merges from a byte corpus. Returns the merge list
/// in training order; `merges[i]` becomes token id `256 + i`.
pub fn train(corpus: &[u8], vocab_size: u32) -> Vec<(u32, u32)> {
    let mut tokens: Vec<u32> = corpus.iter().map(|&b| u32::from(b)).collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut next_id: u32 = 256;
    while next_id < vocab_size && tokens.len() >= 2 {
        let Some(pair) = pick_merge_pair(&tokens) else {
            break;
        };
        merges.push(pair);
        tokens = apply_merge(&tokens, pair, next_id);
        next_id += 1;
    }
    merges
}

/// Count adjacent pairs in `tokens` and return the winning
/// pair (highest count, lex-smallest on ties) with count >= 1.
/// Returns `None` if no adjacent pairs exist.
pub(crate) fn pick_merge_pair(tokens: &[u32]) -> Option<(u32, u32)> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for w in tokens.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|(pa, ca), (pb, cb)| ca.cmp(cb).then_with(|| pb.cmp(pa)))
        .map(|(p, _)| p)
}

/// Produce a new token sequence with every occurrence of
/// `pair` replaced by `new_id`. Left-to-right greedy.
pub(crate) fn apply_merge(tokens: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(tokens[i]);
            i += 1;
        }
    }
    out
}
