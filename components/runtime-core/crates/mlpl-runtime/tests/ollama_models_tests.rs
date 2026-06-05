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
