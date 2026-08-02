//! Glossary fuzzy-matcher tests.

use mlpl_glossary_search::best_match;

const TERMS: &[&str] = &[
    "Activation outlier",
    "BF16",
    "Bits per weight",
    "FP16",
    "Importance matrix (imatrix)",
    "K-quant",
    "Post-training quantization (PTQ)",
    "Q4_K_M",
    "Quant size modifiers (S, M, L)",
    "Ternary weights",
];

fn term(q: &str) -> Option<&'static str> {
    best_match(q, TERMS.iter().copied()).map(|i| TERMS[i])
}

#[test]
fn exact_prefix_and_empty() {
    assert_eq!(term("fp16"), Some("FP16"));
    assert_eq!(term("q4"), Some("Q4_K_M"));
    assert_eq!(term(""), None);
    assert_eq!(term("zzzznotaterm"), None);
}

#[test]
fn substring_anywhere() {
    // not first-word prefixes, but appear inside the term
    assert_eq!(term("imatrix"), Some("Importance matrix (imatrix)"));
    assert_eq!(term("ptq"), Some("Post-training quantization (PTQ)"));
    assert_eq!(term("outlier"), Some("Activation outlier"));
}

#[test]
fn plurals_fold() {
    assert_eq!(term("k-quants"), Some("K-quant"));
    assert_eq!(term("modifiers"), Some("Quant size modifiers (S, M, L)"));
    assert_eq!(term("weights"), Some("Ternary weights"));
}

#[test]
fn aliases_resolve() {
    assert_eq!(term("bfloat16"), Some("BF16"));
    assert_eq!(term("half precision"), Some("FP16"));
    assert_eq!(term("bpw"), Some("Bits per weight"));
}

#[test]
fn all_matches_returns_every_match_best_first() {
    let terms = [
        "Attention",
        "Cross-attention",
        "Self-attention",
        "Adam",
        "Flash attention",
    ];
    let hits = mlpl_glossary_search::all_matches("attention", terms.iter().copied());
    // Exact match first, then any-word prefixes in list order, then
    // substrings -- and EVERY matching entry is present.
    assert_eq!(hits.first(), Some(&0), "exact match leads");
    assert_eq!(hits.len(), 4, "all attention entries match: {hits:?}");
    assert!(!hits.contains(&3), "Adam does not match");
    // Consistency: best_match is all_matches' head.
    let best = mlpl_glossary_search::best_match("attention", terms.iter().copied());
    assert_eq!(best, hits.first().copied());
}

#[test]
fn all_matches_empty_query_is_empty() {
    let terms = ["Adam", "Attention"];
    assert!(mlpl_glossary_search::all_matches("  ", terms.iter().copied()).is_empty());
}
