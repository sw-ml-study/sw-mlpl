use mlpl_core::Span;

#[test]
fn span_new() {
    let s = Span::new(0, 5);
    assert_eq!(s.start, 0);
    assert_eq!(s.end, 5);
}

#[test]
fn span_len() {
    assert_eq!(Span::new(3, 7).len(), 4);
}

#[test]
fn span_empty() {
    assert!(Span::new(5, 5).is_empty());
    assert!(!Span::new(0, 1).is_empty());
}

#[test]
fn span_display() {
    assert_eq!(Span::new(10, 20).to_string(), "10..20");
}

#[test]
fn span_equality() {
    assert_eq!(Span::new(0, 5), Span::new(0, 5));
    assert_ne!(Span::new(0, 5), Span::new(0, 6));
}

#[test]
#[should_panic(expected = "start")]
fn span_invalid_panics() {
    let _ = Span::new(10, 5);
}
