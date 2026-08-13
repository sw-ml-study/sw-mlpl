//! `str_concat` / `str_join` (B1). String results are checked by
//! tokenizing them to their bytes (the program value is a
//! `DenseArray`, so `tokenize_bytes` gives a comparable array).

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn bytes(src: &str) -> Vec<f64> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
        .unwrap()
        .data()
        .to_vec()
}

fn err(src: &str) -> mlpl_eval::EvalError {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new()).unwrap_err()
}

#[test]
fn str_concat_joins_two_strings() {
    // "abcd" -> [97, 98, 99, 100]
    assert_eq!(
        bytes("tokenize_bytes(str_concat(\"ab\", \"cd\"))"),
        vec![97.0, 98.0, 99.0, 100.0]
    );
    // empty operands
    assert_eq!(
        bytes("tokenize_bytes(str_concat(\"\", \"x\"))"),
        vec![120.0]
    );
    assert_eq!(
        bytes("tokenize_bytes(str_concat(\"x\", \"\"))"),
        vec![120.0]
    );
}

#[test]
fn str_concat_preserves_unicode_bytes() {
    // "éx" -> é is UTF-8 [0xC3, 0xA9] = [195, 169], then x = 120.
    assert_eq!(
        bytes("tokenize_bytes(str_concat(\"é\", \"x\"))"),
        vec![195.0, 169.0, 120.0]
    );
}

#[test]
fn str_join_folds_a_string_list() {
    // "a, b, c" -> [97, 44, 32, 98, 44, 32, 99]
    assert_eq!(
        bytes("tokenize_bytes(str_join([\"a\", \"b\", \"c\"], \", \"))"),
        vec![97.0, 44.0, 32.0, 98.0, 44.0, 32.0, 99.0]
    );
    // empty separator, single element, and empty list.
    assert_eq!(
        bytes("tokenize_bytes(str_join([\"a\", \"b\", \"c\"], \"\"))"),
        vec![97.0, 98.0, 99.0]
    );
    assert_eq!(
        bytes("tokenize_bytes(str_join([\"only\"], \"-\"))"),
        vec![111.0, 110.0, 108.0, 121.0]
    );
    assert!(bytes("tokenize_bytes(str_join([], \"-\"))").is_empty());
}

#[test]
fn to_string_is_the_shortest_round_trip_of_to_number() {
    // Integral values print bare; the byte forms confirm the text.
    assert_eq!(bytes("tokenize_bytes(to_string(0))"), vec![48.0]); // "0"
    assert_eq!(bytes("tokenize_bytes(to_string(12))"), vec![49.0, 50.0]); // "12"
    assert_eq!(bytes("tokenize_bytes(to_string(0 - 3))"), vec![45.0, 51.0]); // "-3"
    assert_eq!(bytes("tokenize_bytes(to_string(8 / 2))"), vec![52.0]); // "4", not "4.0"
    assert_eq!(
        bytes("tokenize_bytes(to_string(1.5))"),
        vec![49.0, 46.0, 53.0]
    ); // "1.5"
    // Round-trip: to_number(to_string(sqrt(2))) recovers the value.
    let rt = bytes("unwrap(to_number(to_string(sqrt(2))))");
    assert_eq!(rt.len(), 1);
    assert!((rt[0] - 2.0_f64.sqrt()).abs() < 1e-15, "{rt:?}");
}

#[test]
fn string_list_is_accepted_as_a_user_fn_argument() {
    // B5: a string list ([...] of strings -> StrList) passed to a
    // `def u:` function was rejected by the arg-domain check.
    assert_eq!(
        bytes("def u:f(xs) { list_len(xs) }\nu:f([\"a\", \"b\", \"c\"])"),
        vec![3.0]
    );
    assert_eq!(bytes("def u:f(xs) { list_len(xs) }\nu:f([])"), vec![0.0]);
}

#[test]
fn no_coercion_number_argument_is_an_error() {
    assert!(matches!(
        err("str_concat(\"a\", 1)"),
        mlpl_eval::EvalError::Unsupported(_)
    ));
    // a numeric array is not a string list.
    assert!(matches!(
        err("str_join([1, 2], \"-\")"),
        mlpl_eval::EvalError::Unsupported(_)
    ));
}
