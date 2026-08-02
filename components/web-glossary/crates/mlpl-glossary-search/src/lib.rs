//! Fuzzy matching for the glossary search box. A standalone, dependency-free
//! crate: it operates on term strings only, so it has no yew / wasm ties and
//! can be unit-tested directly. The web UI (`mlpl-web-glossary`) calls
//! [`best_match`] with its entry terms.
//!
//! A typed query is matched in tiers, best first: exact term, first-word
//! prefix (the type-ahead feel), any-word prefix, substring anywhere, then a
//! plural-folded substring. A small alias table maps cross-spellings
//! (`bfloat16` -> `bf16`) that no substring rule would catch. [`all_matches`]
//! returns EVERY matching index, best first (ties broken by position; the
//! entry list is sorted, so ties fall alphabetically); [`best_match`] is its
//! head, kept for single-target callers.

/// The index of the entry that best matches `query`, or `None`.
pub fn best_match<'a>(query: &str, terms: impl Iterator<Item = &'a str>) -> Option<usize> {
    all_matches(query, terms).first().copied()
}

/// EVERY matching entry index, best first (rank, then list position
/// -- the list is sorted, so ties fall alphabetically). Empty query
/// matches nothing: the caller shows the full unfiltered list.
pub fn all_matches<'a>(query: &str, terms: impl Iterator<Item = &'a str>) -> Vec<usize> {
    let raw = query.trim().to_lowercase();
    if raw.is_empty() {
        return Vec::new();
    }
    let q = alias_target(&raw).unwrap_or(&raw);
    let qs = depluralize(q);
    let mut hits: Vec<(u8, usize)> = terms
        .enumerate()
        .map(|(i, t)| (rank(&t.to_lowercase(), q, qs), i))
        .filter(|(r, _)| *r != u8::MAX)
        .collect();
    hits.sort_unstable();
    hits.into_iter().map(|(_, i)| i).collect()
}

mod rank;
use rank::{alias_target, depluralize, rank};
