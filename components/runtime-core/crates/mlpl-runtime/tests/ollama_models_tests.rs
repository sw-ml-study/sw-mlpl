//! `median_model` -- the "not too small (weak), not too big (slow)"
//! auto-pick used as the `:ask` default. Pure (no network), so it is
//! unit-testable directly.

use mlpl_runtime::median_model;

#[test]
fn median_picks_the_middle_by_size() {
    // sorted by size: a(1), c(3), b(5); median index 3/2 = 1 -> c.
    let models = vec![
        ("a".to_string(), 1u64),
        ("b".to_string(), 5),
        ("c".to_string(), 3),
    ];
    assert_eq!(median_model(&models).as_deref(), Some("c"));
}

#[test]
fn median_of_empty_is_none() {
    assert_eq!(median_model(&[]), None);
}

#[test]
fn median_is_size_order_independent_of_input_order() {
    let a = vec![
        ("x".to_string(), 10u64),
        ("y".to_string(), 20),
        ("z".to_string(), 30),
    ];
    let mut b = a.clone();
    b.reverse();
    assert_eq!(median_model(&a), median_model(&b));
}

const GB: u64 = 1_000_000_000;

#[test]
fn median_skips_models_below_the_viable_floor() {
    // Two toy models drag the raw median onto "bad" (3 GB). With the
    // viable-size floor, the median is taken over {bad, good, big} ->
    // "good", so :ask never lands on the toy-pulled middle.
    let models = vec![
        ("toy-a".to_string(), 200_000_000), // 0.2 GB
        ("toy-b".to_string(), 400_000_000), // 0.4 GB
        ("bad".to_string(), 3 * GB),
        ("good".to_string(), 4 * GB),
        ("big".to_string(), 6 * GB),
    ];
    assert_eq!(median_model(&models).as_deref(), Some("good"));
}

#[test]
fn median_falls_back_to_full_list_when_all_below_floor() {
    // A host with only small models still gets a pick (median of all),
    // not None.
    let models = vec![
        ("a".to_string(), 200_000_000),
        ("b".to_string(), 400_000_000),
        ("c".to_string(), 600_000_000),
    ];
    assert_eq!(median_model(&models).as_deref(), Some("b"));
}

#[test]
fn median_never_picks_an_embedding_model() {
    // Embedding models cannot answer prompts; excluded even when large
    // (here the embed model is the biggest, so the raw median would be
    // the embed model).
    let models = vec![
        ("llama".to_string(), 4 * GB),
        ("nomic-embed-text".to_string(), 9 * GB),
    ];
    assert_eq!(median_model(&models).as_deref(), Some("llama"));
}

#[test]
fn median_of_only_embedding_models_is_none() {
    let models = vec![("nomic-embed-text".to_string(), 300_000_000)];
    assert_eq!(median_model(&models), None);
}
