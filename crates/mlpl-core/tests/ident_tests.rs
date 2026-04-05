use mlpl_core::Identifier;

#[test]
fn valid_simple() {
    assert!(Identifier::new("x").is_some());
    assert!(Identifier::new("my_var").is_some());
    assert!(Identifier::new("result2").is_some());
}

#[test]
fn valid_underscore_start() {
    assert!(Identifier::new("_temp").is_some());
    assert!(Identifier::new("_").is_some());
    assert!(Identifier::new("__double").is_some());
}

#[test]
fn invalid_empty() {
    assert!(Identifier::new("").is_none());
}

#[test]
fn invalid_starts_with_digit() {
    assert!(Identifier::new("2fast").is_none());
    assert!(Identifier::new("0x").is_none());
}

#[test]
fn invalid_special_chars() {
    assert!(Identifier::new("my-var").is_none());
    assert!(Identifier::new("hello!").is_none());
    assert!(Identifier::new("a b").is_none());
    assert!(Identifier::new("x+y").is_none());
}

#[test]
fn invalid_unicode() {
    assert!(Identifier::new("caf\u{00e9}").is_none());
}

#[test]
fn as_str_roundtrip() {
    let id = Identifier::new("foo").unwrap();
    assert_eq!(id.as_str(), "foo");
}

#[test]
fn display() {
    let id = Identifier::new("my_var").unwrap();
    assert_eq!(id.to_string(), "my_var");
}

#[test]
fn equality() {
    let a = Identifier::new("x").unwrap();
    let b = Identifier::new("x").unwrap();
    let c = Identifier::new("y").unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn clone_ident() {
    let a = Identifier::new("x").unwrap();
    let b = a.clone();
    assert_eq!(a, b);
}
