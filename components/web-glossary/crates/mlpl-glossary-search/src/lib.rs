//! Fuzzy matching for the glossary search box. A standalone, dependency-free
//! crate: it operates on term strings only, so it has no yew / wasm ties and
//! can be unit-tested directly. The web UI (`mlpl-web-glossary`) calls
//! [`best_match`] with its entry terms.
//!
//! A typed query is matched in tiers, best first: exact term, first-word
//! prefix (the type-ahead feel), any-word prefix, substring anywhere, then a
//! plural-folded substring. A small alias table maps cross-spellings
//! (`bfloat16` -> `bf16`) that no substring rule would catch. [`best_match`]
//! returns the index of the best entry, ties broken by position (the entry
//! list is sorted, so that is the alphabetically-first match).

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

/// The index of the entry that best matches `query`, or `None`.
pub fn best_match<'a>(query: &str, terms: impl Iterator<Item = &'a str>) -> Option<usize> {
    let raw = query.trim().to_lowercase();
    if raw.is_empty() {
        return None;
    }
    let q = alias_target(&raw).unwrap_or(&raw);
    let qs = depluralize(q);
    terms
        .enumerate()
        .map(|(i, t)| (rank(&t.to_lowercase(), q, qs), i))
        .filter(|(r, _)| *r != u8::MAX)
        .min()
        .map(|(_, i)| i)
}

/// Match strength of a lowercased `term` against query `q` (and its
/// depluralized form `qs`). Lower is better; `u8::MAX` means no match.
fn rank(term: &str, q: &str, qs: &str) -> u8 {
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
fn depluralize(q: &str) -> &str {
    match q.strip_suffix('s') {
        Some(stem) if stem.len() >= 3 => stem,
        _ => q,
    }
}

/// The canonical term for a cross-spelling `q`, if any.
fn alias_target(q: &str) -> Option<&'static str> {
    ALIASES.iter().find(|(k, _)| *k == q).map(|(_, v)| *v)
}
