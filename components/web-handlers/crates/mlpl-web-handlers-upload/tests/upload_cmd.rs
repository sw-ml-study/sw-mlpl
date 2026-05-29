//! Saga 82: upload-command parser tests previously inline in
//! mlpl-web/src/handlers.rs. Moved to integration tests so the
//! src module function counts stay clean.

use mlpl_web_handlers_upload::upload_cmd::{is_valid_identifier, parse_upload_command};

#[test]
fn parse_upload_with_name() {
    assert_eq!(parse_upload_command(":upload x"), Some("x".into()));
    assert_eq!(
        parse_upload_command(":upload my_photo"),
        Some("my_photo".into())
    );
    assert_eq!(
        parse_upload_command(":upload   spaced"),
        Some("spaced".into())
    );
}

#[test]
fn parse_upload_no_name_returns_empty_string() {
    assert_eq!(parse_upload_command(":upload"), Some(String::new()));
    assert_eq!(parse_upload_command(":upload   "), Some(String::new()));
}

#[test]
fn parse_non_upload_returns_none() {
    assert_eq!(parse_upload_command(":help"), None);
    assert_eq!(parse_upload_command(":vars"), None);
    assert_eq!(parse_upload_command("x = 1"), None);
    assert_eq!(parse_upload_command(""), None);
    // `:uploaded` shares a prefix with `:upload` but is a
    // different command (no separator); today the parser
    // treats it as `:upload <name>` = `:upload ed`. This is
    // a known quirk -- nobody types `:uploaded` so it
    // doesn't collide in practice.
}

#[test]
fn identifier_validation() {
    assert!(is_valid_identifier("x"));
    assert!(is_valid_identifier("X"));
    assert!(is_valid_identifier("my_photo"));
    assert!(is_valid_identifier("_x"));
    assert!(is_valid_identifier("a1"));
    assert!(!is_valid_identifier(""));
    assert!(!is_valid_identifier("1"));
    assert!(!is_valid_identifier("1x"));
    assert!(!is_valid_identifier("x y"));
    assert!(!is_valid_identifier("x-y"));
    assert!(!is_valid_identifier("x.y"));
}
