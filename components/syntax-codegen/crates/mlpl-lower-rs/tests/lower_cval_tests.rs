//! Lowering of the compiled value model: string literals and the
//! write_stdout / args / arg builtins produce CVal; a numeric
//! program result is wrapped as CVal::Arr.

use mlpl_lower_rs::lower;
use mlpl_parser::{lex, parse};

fn lowered(src: &str) -> String {
    let toks = lex(src).expect("lex");
    let stmts = parse(&toks).expect("parse");
    lower(&stmts).expect("lower").to_string()
}

#[test]
fn string_literal_lowers_to_cval_str() {
    let s = lowered("\"hello\"");
    assert!(s.contains("CVal :: Str") || s.contains("CVal::Str"), "{s}");
    assert!(s.contains("hello"), "{s}");
}

#[test]
fn write_stdout_lowers_to_runtime_call() {
    let s = lowered("write_stdout(\"hi\")");
    assert!(s.contains("write_stdout"), "{s}");
    assert!(s.contains("CVal :: Str") || s.contains("CVal::Str"), "{s}");
}

#[test]
fn args_and_arg_lower_to_runtime_calls() {
    assert!(
        lowered("args()").contains("cli_args"),
        "args() should lower to cli_args"
    );
    let s = lowered("arg(0)");
    assert!(s.contains("arg"), "{s}");
}

#[test]
fn numeric_result_is_wrapped_as_cval_arr() {
    // a numeric program result is wrapped so lower() always yields a CVal
    let s = lowered("1 + 2");
    assert!(s.contains("CVal :: Arr") || s.contains("CVal::Arr"), "{s}");
    assert!(s.contains("apply_binop"), "{s}");
}

#[test]
fn write_stdout_of_byte_array_wraps_arg_as_cval_arr() {
    // the array-literal argument is wrapped as CVal::Arr for the sink
    let s = lowered("write_stdout([72, 105])");
    assert!(s.contains("write_stdout"), "{s}");
    assert!(s.contains("CVal :: Arr") || s.contains("CVal::Arr"), "{s}");
}
