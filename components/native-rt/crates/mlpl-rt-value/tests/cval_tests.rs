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
fn write_stdout_ok_returns_byte_count() {
    // string -> ok(UTF-8 byte count)
    let r = write_stdout(&CVal::Str("hi".into()));
    assert_eq!(
        r,
        CVal::result(true, CVal::Arr(DenseArray::from_scalar(2.0)))
    );
    // array of valid bytes -> ok(cell count)
    let bytes = CVal::Arr(DenseArray::from_vec(vec![72.0, 105.0, 33.0]));
    assert_eq!(
        write_stdout(&bytes),
        CVal::result(true, CVal::Arr(DenseArray::from_scalar(3.0)))
    );
}

#[test]
fn write_stdout_rejects_invalid_bytes_with_err() {
    // out-of-range 256 -> err (rejected, NOT truncated to 0), with a
    // descriptive interpreter-parity message.
    let over = CVal::Arr(DenseArray::from_vec(vec![256.0]));
    match write_stdout(&over) {
        CVal::Result { ok: false, payload } => {
            let msg = format!("{payload}");
            assert!(msg.contains("256") && msg.contains("0..=255"), "{msg}");
        }
        other => panic!("expected err Result, got {other:?}"),
    }
    // non-integer and negative are likewise rejected.
    let frac = CVal::Arr(DenseArray::from_vec(vec![1.5]));
    assert!(matches!(
        write_stdout(&frac),
        CVal::Result { ok: false, .. }
    ));
    let neg = CVal::Arr(DenseArray::from_vec(vec![-1.0]));
    assert!(matches!(write_stdout(&neg), CVal::Result { ok: false, .. }));
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
