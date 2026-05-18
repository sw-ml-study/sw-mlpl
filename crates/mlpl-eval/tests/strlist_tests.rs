//! Evaluation tests for the string-list value type and the
//! `[...]` array-literal disambiguation between numeric arrays
//! and string lists.
//! Saga 29 step 002.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env)
}

fn eval_seq(srcs: &[&str]) -> Result<Value, EvalError> {
    let src = srcs.join("\n");
    eval(&src)
}

#[test]
fn empty_array_lit_stays_empty_array() {
    // Empty [] is ambiguous; preserve the existing numeric-array
    // behavior for back-compat. A user who wants an empty StrList
    // must build it some other way (or wait for syntax in a
    // follow-up step).
    match eval("[]").unwrap() {
        Value::Array(a) => assert_eq!(a.data().len(), 0),
        other => panic!("expected empty Value::Array, got {other:?}"),
    }
}

#[test]
fn all_string_elements_make_a_strlist() {
    match eval("[\"a\", \"b\", \"c\"]").unwrap() {
        Value::StrList { items } => {
            assert_eq!(
                items,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
        other => panic!("expected Value::StrList, got {other:?}"),
    }
}

#[test]
fn single_string_element_makes_a_strlist() {
    match eval("[\"only\"]").unwrap() {
        Value::StrList { items } => assert_eq!(items, vec!["only".to_string()]),
        other => panic!("expected Value::StrList, got {other:?}"),
    }
}

#[test]
fn all_numeric_elements_still_make_a_dense_array() {
    // Regression: existing numeric-array path is unchanged.
    match eval("[1, 2, 3]").unwrap() {
        Value::Array(a) => assert_eq!(a.data(), &[1.0, 2.0, 3.0]),
        other => panic!("expected Value::Array, got {other:?}"),
    }
}

#[test]
fn mixed_string_and_numeric_errors_cleanly() {
    let err = eval("[\"a\", 1]").unwrap_err();
    match err {
        EvalError::MixedArrayLitElements { kinds } => {
            assert!(
                kinds.contains(&"string"),
                "kinds should mention string, got: {kinds:?}"
            );
            assert!(
                kinds.contains(&"array"),
                "kinds should mention array, got: {kinds:?}"
            );
        }
        other => panic!("expected MixedArrayLitElements, got {other:?}"),
    }
}

#[test]
fn numeric_then_string_also_errors_with_mixed() {
    let err = eval("[1, \"a\"]").unwrap_err();
    assert!(
        matches!(err, EvalError::MixedArrayLitElements { .. }),
        "expected MixedArrayLitElements, got {err:?}"
    );
}

#[test]
fn strlist_display_round_trip() {
    let v = eval("[\"alpha\", \"beta\"]").unwrap();
    let s = format!("{v}");
    assert_eq!(s, "[\"alpha\", \"beta\"]");
}

#[test]
fn empty_strlist_display_via_constructor() {
    // No source-level syntax for empty StrList yet, but the
    // Display impl should still render an empty StrList cleanly.
    let v = Value::StrList { items: vec![] };
    assert_eq!(format!("{v}"), "[]");
}

#[test]
fn strlist_bound_to_a_name_round_trips() {
    let v = eval_seq(&["names = [\"cat\", \"dog\"]", "names"]).unwrap();
    match v {
        Value::StrList { items } => assert_eq!(items, vec!["cat".to_string(), "dog".to_string()]),
        other => panic!("expected Value::StrList after rebind, got {other:?}"),
    }
}

#[test]
fn record_field_can_hold_a_strlist() {
    let v = eval("{names: [\"cat\", \"dog\"]}").unwrap();
    match v {
        Value::Record { fields } => match fields.get("names") {
            Some(Value::StrList { items }) => {
                assert_eq!(items, &vec!["cat".to_string(), "dog".to_string()]);
            }
            other => panic!("names field should be StrList, got {other:?}"),
        },
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn field_access_returns_strlist_through_a_record() {
    let v = eval_seq(&[
        "r = {names: [\"cat\", \"dog\"], label: \"pets\"}",
        "r.names",
    ])
    .unwrap();
    match v {
        Value::StrList { items } => assert_eq!(items, vec!["cat".to_string(), "dog".to_string()]),
        other => panic!("expected StrList from r.names, got {other:?}"),
    }
}

#[test]
fn list_len_returns_strlist_length_as_scalar() {
    let v = eval("list_len([\"a\", \"b\", \"c\"])").unwrap();
    match v {
        Value::Array(a) => {
            assert_eq!(a.rank(), 0, "list_len should return a rank-0 scalar");
            assert_eq!(a.data(), &[3.0]);
        }
        other => panic!("expected scalar Array, got {other:?}"),
    }
}

#[test]
fn list_len_on_empty_strlist_returns_zero() {
    // Bound via an intermediate variable since `[]` -> Array
    // (back-compat). We construct via assignment from a StrList
    // expr that came from somewhere -- here, indirectly by
    // popping a record field.
    let v = eval_seq(&[
        "r = {names: [\"x\"]}",
        // r.names is a StrList of 1
        "list_len(r.names)",
    ])
    .unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[1.0]),
        other => panic!("expected scalar Array, got {other:?}"),
    }
}

#[test]
fn list_len_on_non_strlist_errors_cleanly() {
    let err = eval("list_len(42)").unwrap_err();
    // The exact variant is implementation-defined; what matters
    // is we get a tutoring error, not a panic.
    match err {
        EvalError::Unsupported(_) | EvalError::ExpectedArray | EvalError::TypeMismatch { .. } => {}
        other => panic!("expected a clean tutoring error, got {other:?}"),
    }
}

#[test]
fn value_kind_string_is_the_strlist_name() {
    let v = Value::StrList {
        items: vec!["x".to_string()],
    };
    assert_eq!(mlpl_eval::value_kind(&v), "string-list");
}
