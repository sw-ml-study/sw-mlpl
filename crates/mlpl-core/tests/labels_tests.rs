use mlpl_core::LabeledShape;

#[test]
fn positional_has_no_labels() {
    let ls = LabeledShape::positional(vec![2, 3]);
    assert_eq!(ls.dims(), &[2, 3]);
    assert_eq!(ls.labels(), &[None, None]);
    assert!(!ls.is_labeled());
}

#[test]
fn positional_scalar_is_empty() {
    let ls = LabeledShape::positional(vec![]);
    assert!(ls.labels().is_empty());
    assert!(!ls.is_labeled());
}

#[test]
fn new_with_labels_round_trips() {
    let ls = LabeledShape::new(vec![6, 4], vec![Some("seq".into()), Some("d_model".into())]);
    assert_eq!(ls.dims(), &[6, 4]);
    assert_eq!(ls.labels()[0].as_deref(), Some("seq"));
    assert_eq!(ls.labels()[1].as_deref(), Some("d_model"));
    assert!(ls.is_labeled());
}

#[test]
fn mixed_labels_count_as_labeled() {
    let ls = LabeledShape::new(vec![2, 3], vec![None, Some("cols".into())]);
    assert!(ls.is_labeled());
}

#[test]
#[should_panic(expected = "dims and labels must have the same length")]
fn new_panics_on_rank_mismatch() {
    let _ = LabeledShape::new(vec![2, 3], vec![Some("rows".into())]);
}
