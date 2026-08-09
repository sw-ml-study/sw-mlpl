//! The compiled value model (CVal) + string/args helpers.

use mlpl_array::DenseArray;
use mlpl_rt_value::{CVal, arg, cli_args, write_stdout};

#[test]
fn arr_accessor_and_from() {
    let v: CVal = DenseArray::from_scalar(42.0).into();
    assert_eq!(v.arr().data(), &[42.0]);
}

#[test]
fn display_renders_each_variant() {
    assert_eq!(format!("{}", CVal::Str("hi".into())), "hi");
    assert_eq!(
        format!("{}", CVal::StrList(vec!["a".into(), "b".into()])),
        "a\nb"
    );
    assert_eq!(format!("{}", CVal::Arr(DenseArray::from_scalar(7.0))), "7");
}

#[test]
fn write_stdout_returns_byte_count() {
    // string -> its UTF-8 byte count
    assert_eq!(write_stdout(&CVal::Str("hi".into())).arr().data(), &[2.0]);
    // array -> cell count (each cell a byte)
    let bytes = CVal::Arr(DenseArray::from_vec(vec![72.0, 105.0, 33.0]));
    assert_eq!(write_stdout(&bytes).arr().data(), &[3.0]);
}

#[test]
fn cli_args_is_a_string_list() {
    // Under `cargo test` argv is the test harness; just assert the
    // shape (a StrList), not its contents.
    assert!(matches!(cli_args(), CVal::StrList(_)));
}

#[test]
fn arg_out_of_range_is_empty() {
    // index 9999 is certainly past argv
    assert_eq!(
        arg(&CVal::Arr(DenseArray::from_scalar(9999.0))),
        CVal::Str(String::new())
    );
}
