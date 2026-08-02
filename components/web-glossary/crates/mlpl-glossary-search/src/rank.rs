//! Query normalization + per-term match ranking for the glossary
//! search: the alias table, plural folding, and the tier rules.

/// Cross-spelling synonyms -> the canonical glossary term (both lowercased).
/// Only needed for queries that are NOT a substring of the term; plurals and
/// embedded names (`imatrix`, `ptq`) are handled by the tier rules below.
const ALIASES: &[(&str, &str)] = &[
    ("half precision", "fp16"),
    ("single precision", "fp32"),
    ("bfloat16", "bf16"),
    ("kquant", "k-quant"),
    ("k quant", "k-quant"),
    ("iquant", "i-quant"),
    ("i quant", "i-quant"),
    ("bpw", "bits per weight"),
    ("ggml", "gguf"),
];

/// Match strength of a lowercased `term` against query `q` (and its
/// depluralized form `qs`). Lower is better; `u8::MAX` means no match.
pub(crate) fn rank(term: &str, q: &str, qs: &str) -> u8 {
    let first_word_hit = term
        .split_whitespace()
        .next()
        .is_some_and(|w| w.starts_with(q));
    if term == q {
        0
    } else if first_word_hit {
        1
    } else if term.split_whitespace().any(|w| w.starts_with(q)) {
        2
    } else if term.contains(q) {
        3
    } else if qs != q && term.contains(qs) {
        4
    } else {
        u8::MAX
    }
}

/// Strip a single trailing plural `s` (keeping a stem of >= 3 chars), so
/// `k-quants` folds to `k-quant`, `modifiers` to `modifier`.
pub(crate) fn depluralize(q: &str) -> &str {
    match q.strip_suffix('s') {
        Some(stem) if stem.len() >= 3 => stem,
        _ => q,
    }
}

/// The canonical term for a cross-spelling `q`, if any.
pub(crate) fn alias_target(q: &str) -> Option<&'static str> {
    ALIASES.iter().find(|(k, _)| *k == q).map(|(_, v)| *v)
}
