//! Character-based string ops (algebra work-order B3):
//! `str_len`, `str_slice`, `str_find`, `str_split`.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program_value(&stmts, &mut Environment::new())
}

fn scalar(src: &str) -> f64 {
    match eval(src).unwrap() {
        Value::Array(a) => a.data()[0],
        other => panic!("expected a scalar, got {other:?}"),
    }
}

fn string(src: &str) -> String {
    match eval(src).unwrap() {
        Value::Str(s) => s,
        other => panic!("expected a string, got {other:?}"),
    }
}

fn list(src: &str) -> Vec<String> {
    match eval(src).unwrap() {
        Value::StrList { items } => items,
        other => panic!("expected a string list, got {other:?}"),
    }
}

#[test]
fn str_len_counts_characters_not_bytes() {
    assert_eq!(scalar("str_len(\"abcd\")"), 4.0);
    // é is 2 UTF-8 bytes but one character.
    assert_eq!(scalar("str_len(\"héllo\")"), 5.0);
    assert_eq!(scalar("str_len(\"\")"), 0.0);
}

#[test]
fn str_slice_is_character_indexed() {
    assert_eq!(string("str_slice(\"abcdef\", 1, 3)"), "bcd");
    assert_eq!(string("str_slice(\"héllo\", 1, 1)"), "é");
    // A length past the end clamps to what's available.
    assert_eq!(string("str_slice(\"abc\", 1, 10)"), "bc");
}

#[test]
fn str_find_returns_char_index_or_minus_one() {
    assert_eq!(scalar("str_find(\"<rect/><rect/>\", \"<rect\")"), 0.0);
    assert_eq!(scalar("str_find(\"abc\", \"z\")"), -1.0);
    // char index, not byte index (past the 2-byte é).
    assert_eq!(scalar("str_find(\"héllo\", \"llo\")"), 2.0);
}

#[test]
fn str_split_yields_a_string_list() {
    assert_eq!(list("str_split(\"a,b,c\", \",\")"), vec!["a", "b", "c"]);
    assert_eq!(list("str_split(\"abc\", \",\")"), vec!["abc"]);
}

#[test]
fn str_ops_reject_a_non_string() {
    assert!(matches!(
        eval("str_len(123)"),
        Err(EvalError::Unsupported(_))
    ));
    assert!(matches!(
        eval("str_find(\"a\", 1)"),
        Err(EvalError::Unsupported(_))
    ));
}
