//! Compiled-runtime `format` / variadic `str_concat` / `write`, matching
//! the interpreter: a scalar formats as a number, a string as text, any
//! other value as its display; `str_concat` takes strings only.

use mlpl_array::DenseArray;
use mlpl_rt_value::{CVal, format, str_concat, write};

fn s(x: &str) -> CVal {
    CVal::Str(x.into())
}

fn n(x: f64) -> CVal {
    CVal::Arr(DenseArray::from_scalar(x))
}

#[test]
fn format_uses_the_shared_spec_formatter() {
    assert_eq!(
        format(vec![
            s("{} = {:.2f} ({:>4d})"),
            s("loss"),
            n(0.12345),
            n(7.0)
        ]),
        s("loss = 0.12 (   7)")
    );
}

#[test]
#[should_panic(expected = "format")]
fn format_bad_spec_panics_like_a_hard_error() {
    let _ = format(vec![s("{:d}"), s("text")]);
}

#[test]
fn str_concat_joins_any_number_of_strings() {
    assert_eq!(str_concat(vec![s("a"), s("b"), s("c")]), s("abc"));
}

#[test]
#[should_panic(expected = "argument 1 must be a string")]
fn str_concat_names_the_bad_argument() {
    let _ = str_concat(vec![s("a"), n(1.0)]);
}

#[test]
fn write_returns_the_text_written() {
    assert_eq!(write(vec![s("step "), n(3.0)]), s("step 3"));
}
