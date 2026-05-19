//! Evaluation tests for the Result<val, err> value type and
//! its accessors: ok(), err(), is_ok(), is_err(), unwrap(),
//! unwrap_or(default, r), err_message(r).
//!
//! Saga 29 step 012.

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

fn scalar(v: &Value) -> f64 {
    match v {
        Value::Array(a) => {
            assert_eq!(
                a.data().len(),
                1,
                "expected scalar, got shape {:?}",
                a.shape().dims()
            );
            a.data()[0]
        }
        other => panic!("expected scalar Value::Array, got {other:?}"),
    }
}

#[test]
fn ok_constructor_wraps_a_scalar() {
    match eval("ok(42)").unwrap() {
        Value::Result { ok, payload } => {
            assert!(ok);
            assert_eq!(scalar(&payload), 42.0);
        }
        other => panic!("expected Value::Result, got {other:?}"),
    }
}

#[test]
fn err_constructor_wraps_a_string() {
    match eval("err(\"bad file\")").unwrap() {
        Value::Result { ok, payload } => {
            assert!(!ok);
            match *payload {
                Value::Str(s) => assert_eq!(s, "bad file"),
                other => panic!("expected Str payload, got {other:?}"),
            }
        }
        other => panic!("expected Value::Result, got {other:?}"),
    }
}

#[test]
fn ok_and_err_round_trip_through_assign() {
    let v = eval_seq(&["r = ok(7)", "r"]).unwrap();
    match v {
        Value::Result { ok, payload } => {
            assert!(ok);
            assert_eq!(scalar(&payload), 7.0);
        }
        other => panic!("expected Value::Result after rebind, got {other:?}"),
    }
}

#[test]
fn is_ok_returns_one_for_ok_zero_for_err() {
    assert_eq!(scalar(&eval("is_ok(ok(1))").unwrap()), 1.0);
    assert_eq!(scalar(&eval("is_ok(err(\"x\"))").unwrap()), 0.0);
}

#[test]
fn is_err_is_the_inverse() {
    assert_eq!(scalar(&eval("is_err(ok(1))").unwrap()), 0.0);
    assert_eq!(scalar(&eval("is_err(err(\"x\"))").unwrap()), 1.0);
}

#[test]
fn unwrap_returns_the_payload_for_ok() {
    let v = eval("unwrap(ok(99))").unwrap();
    assert_eq!(scalar(&v), 99.0);
}

#[test]
fn unwrap_on_err_raises_unwrap_on_err_with_payload_message() {
    let e = eval("unwrap(err(\"cancelled\"))").unwrap_err();
    match e {
        EvalError::UnwrapOnErr { message } => assert_eq!(message, "cancelled"),
        other => panic!("expected UnwrapOnErr, got {other:?}"),
    }
}

#[test]
fn err_message_returns_the_payload_for_err() {
    match eval("err_message(err(\"bad file\"))").unwrap() {
        Value::Str(s) => assert_eq!(s, "bad file"),
        other => panic!("expected Str, got {other:?}"),
    }
}

#[test]
fn err_message_on_ok_raises_unsupported() {
    // err_message on Ok(_) is a logic error: there is no message.
    let e = eval("err_message(ok(1))").unwrap_err();
    match e {
        EvalError::Unsupported(_) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn unwrap_or_returns_payload_for_ok() {
    let v = eval("unwrap_or(ok(5), 0)").unwrap();
    assert_eq!(scalar(&v), 5.0);
}

#[test]
fn unwrap_or_returns_default_for_err() {
    let v = eval("unwrap_or(err(\"nope\"), 42)").unwrap();
    assert_eq!(scalar(&v), 42.0);
}

#[test]
fn is_ok_on_non_result_raises_not_a_result() {
    let e = eval("is_ok(1)").unwrap_err();
    match e {
        EvalError::NotAResult {
            receiver_kind,
            accessor,
        } => {
            assert_eq!(accessor, "is_ok");
            assert_eq!(receiver_kind, "array");
        }
        other => panic!("expected NotAResult, got {other:?}"),
    }
}

#[test]
fn unwrap_on_non_result_raises_not_a_result() {
    let e = eval("unwrap(\"hi\")").unwrap_err();
    match e {
        EvalError::NotAResult {
            receiver_kind,
            accessor,
        } => {
            assert_eq!(accessor, "unwrap");
            assert_eq!(receiver_kind, "string");
        }
        other => panic!("expected NotAResult, got {other:?}"),
    }
}

#[test]
fn bad_arity_for_unwrap() {
    let e = eval("unwrap()").unwrap_err();
    match e {
        EvalError::BadArity {
            func,
            expected,
            got,
        } => {
            assert_eq!(func, "unwrap");
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
        }
        other => panic!("expected BadArity, got {other:?}"),
    }
}

#[test]
fn bad_arity_for_unwrap_or() {
    let e = eval("unwrap_or(ok(1))").unwrap_err();
    match e {
        EvalError::BadArity {
            func,
            expected,
            got,
        } => {
            assert_eq!(func, "unwrap_or");
            assert_eq!(expected, 2);
            assert_eq!(got, 1);
        }
        other => panic!("expected BadArity, got {other:?}"),
    }
}

#[test]
fn display_for_ok_and_err() {
    let ok_v = eval("ok(3)").unwrap();
    assert_eq!(format!("{ok_v}"), "Ok(3)");
    let err_v = eval("err(\"cancelled\")").unwrap();
    assert_eq!(format!("{err_v}"), "Err(cancelled)");
}

#[test]
fn result_can_wrap_a_record() {
    // Records can be wrapped too -- the upload() path returns
    // ok({pixels: [3,H,W], h: H, w: W}) or err("...").
    let v = eval_seq(&["r = ok({a: 1, b: 2})", "unwrap(r).a"]).unwrap();
    assert_eq!(scalar(&v), 1.0);
}
