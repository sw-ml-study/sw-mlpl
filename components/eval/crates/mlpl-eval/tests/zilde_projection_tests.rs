//! `get_value` / `get_error`: Rust's `.ok()`/`.err()` with the APL2
//! zilde flavor -- 0-or-1 element vectors, `tally` as the tag
//! (docs/option-result-design.md, review-round-2 synthesis).

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_eval::Value, String> {
    let prog = parse(&lex(src).map_err(|e| format!("{e:?}"))?).map_err(|e| format!("{e:?}"))?;
    eval_program_value(&prog, &mut Environment::new()).map_err(|e| format!("{e:?}"))
}

fn data(src: &str) -> Vec<f64> {
    match eval(src).expect("eval ok") {
        mlpl_eval::Value::Array(a) => a.data().to_vec(),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn get_value_of_ok_scalar_is_one_element() {
    assert_eq!(data("get_value(ok(42))"), vec![42.0]);
    assert_eq!(data("tally(get_value(ok(42)))"), vec![1.0]);
}

#[test]
fn get_value_of_err_is_zilde() {
    assert_eq!(data("get_value(err(7))"), Vec::<f64>::new());
    assert_eq!(data("tally(get_value(err(7)))"), vec![0.0]);
}

#[test]
fn get_error_projects_the_other_side() {
    assert_eq!(data("get_error(err(7))"), vec![7.0]);
    assert_eq!(data("tally(get_error(ok(1)))"), vec![0.0]);
}

#[test]
fn projections_are_complementary_by_construction() {
    assert_eq!(
        data("r = ok(5); tally(get_value(r)) + tally(get_error(r))"),
        vec![1.0]
    );
}

#[test]
fn unwrap_or_is_derivable_from_the_projection() {
    // take(concat(get_value(r), [d]), 0, 0) == unwrap_or(r, d)
    assert_eq!(
        data("take(concat(get_value(ok(5)), [9], 0), 0, 0)"),
        vec![5.0]
    );
    assert_eq!(
        data("take(concat(get_value(err(0)), [9], 0), 0, 0)"),
        vec![9.0]
    );
}

#[test]
fn non_scalar_payload_names_the_stage6_gap() {
    let msg = eval("get_value(ok([1, 2, 3]))").expect_err("needs enclose");
    assert!(msg.contains("enclose"), "{msg}");
}
