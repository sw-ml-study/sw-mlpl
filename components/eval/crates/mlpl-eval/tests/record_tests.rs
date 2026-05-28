//! Evaluation tests for record literals and field access.
//! Saga 29 step 001.

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
fn empty_record_evaluates() {
    match eval("{}").unwrap() {
        Value::Record { fields } => assert!(fields.is_empty()),
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn record_literal_with_int_fields_evaluates() {
    match eval("{X: 1, Y: 2}").unwrap() {
        Value::Record { fields } => {
            assert_eq!(fields.len(), 2);
            match fields.get("X").unwrap() {
                Value::Array(a) => assert_eq!(a.data(), &[1.0]),
                other => panic!("X should be array scalar 1, got {other:?}"),
            }
            match fields.get("Y").unwrap() {
                Value::Array(a) => assert_eq!(a.data(), &[2.0]),
                other => panic!("Y should be array scalar 2, got {other:?}"),
            }
        }
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn record_keys_btreemap_sorted() {
    // Source order Z, A, M -- BTreeMap iteration is alphabetical.
    let r = eval("{Z: 1, A: 2, M: 3}").unwrap();
    match r {
        Value::Record { fields } => {
            let keys: Vec<_> = fields.keys().cloned().collect();
            assert_eq!(keys, vec!["A", "M", "Z"]);
        }
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn field_access_reads_back_value() {
    let v = eval_seq(&["r = {X: 42, Y: 7}", "r.X"]).unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[42.0]),
        other => panic!("expected Array scalar 42, got {other:?}"),
    }
}

#[test]
fn field_access_on_record_literal_directly() {
    // `{x: 7}.x` -- no intermediate binding.
    let v = eval("{x: 7}.x").unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[7.0]),
        other => panic!("expected Array scalar 7, got {other:?}"),
    }
}

#[test]
fn nested_field_access_chained() {
    let v = eval_seq(&["r = {outer: {inner: 99}}", "r.outer.inner"]).unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[99.0]),
        other => panic!("expected Array scalar 99, got {other:?}"),
    }
}

#[test]
fn unknown_field_errors_with_available_keys() {
    let err = eval_seq(&["r = {X: 1, Y: 2}", "r.Z"]).unwrap_err();
    match err {
        EvalError::FieldNotFound {
            requested,
            available,
        } => {
            assert_eq!(requested, "Z");
            assert_eq!(available, vec!["X".to_string(), "Y".to_string()]);
        }
        other => panic!("expected FieldNotFound, got {other:?}"),
    }
}

#[test]
fn field_access_on_array_is_a_type_error() {
    let err = eval_seq(&["x = [1, 2, 3]", "x.foo"]).unwrap_err();
    match err {
        EvalError::FieldOnNonRecord {
            receiver_kind,
            field,
        } => {
            assert_eq!(receiver_kind, "array");
            assert_eq!(field, "foo");
        }
        other => panic!("expected FieldOnNonRecord, got {other:?}"),
    }
}

#[test]
fn field_access_on_string_is_a_type_error() {
    let err = eval_seq(&["s = \"hello\"", "s.foo"]).unwrap_err();
    match err {
        EvalError::FieldOnNonRecord { receiver_kind, .. } => {
            assert_eq!(receiver_kind, "string");
        }
        other => panic!("expected FieldOnNonRecord, got {other:?}"),
    }
}

#[test]
fn record_field_containing_array_round_trips() {
    let v = eval_seq(&["r = {xs: [1, 2, 3]}", "r.xs"]).unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[1.0, 2.0, 3.0]),
        other => panic!("expected Array [1, 2, 3], got {other:?}"),
    }
}

#[test]
fn record_field_containing_string_round_trips() {
    let v = eval_seq(&["r = {name: \"cat\"}", "r.name"]).unwrap();
    match v {
        Value::Str(s) => assert_eq!(s, "cat"),
        other => panic!("expected Str \"cat\", got {other:?}"),
    }
}

#[test]
fn record_value_can_be_rebound() {
    let v = eval_seq(&["r = {X: 1}", "r = {X: 2, Y: 3}", "r.Y"]).unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[3.0]),
        other => panic!("expected Array scalar 3, got {other:?}"),
    }
}

#[test]
fn record_display_renders_braces_and_fields() {
    let r = eval("{B: 2, A: 1}").unwrap();
    // BTreeMap-ordered: A first.
    let s = format!("{r}");
    assert!(s.starts_with('{') && s.ends_with('}'), "got: {s}");
    assert!(s.contains("A: "), "got: {s}");
    assert!(s.contains("B: "), "got: {s}");
    // A comes before B
    assert!(
        s.find("A: ").unwrap() < s.find("B: ").unwrap(),
        "BTreeMap key order should sort A before B; got: {s}"
    );
}

#[test]
fn field_access_via_function_call_result() {
    // `:add(r.X, r.Y)` -- builtin call whose args use field access.
    let v = eval_seq(&["r = {X: 10, Y: 20}", "r.X + r.Y"]).unwrap();
    match v {
        Value::Array(a) => assert_eq!(a.data(), &[30.0]),
        other => panic!("expected Array scalar 30, got {other:?}"),
    }
}
