//! `format(template, args)` against Python's `str.format` for the supported
//! spec subset. Each expected string is what `CPython` 3 prints for the same
//! template and arguments (MLPL numbers are f64; an integral value formats
//! with `{}` as MLPL displays it -- `5`, not Python's `5.0`).

use std::f64::consts::PI;

use mlpl_format::{FmtArg, format};

fn n(v: f64) -> FmtArg {
    FmtArg::Num(v)
}
fn s(v: &str) -> FmtArg {
    FmtArg::Text(v.into())
}
fn ok(t: &str, args: &[FmtArg]) -> String {
    format(t, args).unwrap_or_else(|e| panic!("{t}: {e}"))
}

#[test]
fn fields_numbering_and_escapes() {
    assert_eq!(ok("{} and {}", &[s("a"), s("b")]), "a and b");
    assert_eq!(ok("{0} {1} {0}", &[s("a"), s("b")]), "a b a");
    assert_eq!(ok("{{x}} = {}", &[n(5.0)]), "{x} = 5");
    assert_eq!(ok("no fields", &[]), "no fields");
}

#[test]
fn default_display_matches_mlpl() {
    assert_eq!(ok("{}", &[n(5.0)]), "5");
    assert_eq!(ok("{}", &[n(2.5)]), "2.5");
    assert_eq!(ok("{}", &[n(-0.125)]), "-0.125");
    assert_eq!(ok("{}", &[s("text")]), "text");
}

#[test]
fn width_fill_and_alignment() {
    assert_eq!(ok("{:>8}", &[s("ab")]), "      ab");
    assert_eq!(ok("{:<5}|", &[s("ab")]), "ab   |");
    assert_eq!(ok("{:^6}", &[s("ab")]), "  ab  ");
    assert_eq!(ok("{:*^7}", &[s("ab")]), "**ab***");
    assert_eq!(ok("{:5}|", &[s("ab")]), "ab   |"); // strings default left
    assert_eq!(ok("{:5}|", &[n(42.0)]), "   42|"); // numbers default right
    assert_eq!(ok("{:1}", &[s("wide")]), "wide"); // width is a minimum
}

#[test]
fn integer_types() {
    assert_eq!(ok("{:4d}", &[n(7.0)]), "   7");
    assert_eq!(ok("{:04d}", &[n(7.0)]), "0007");
    assert_eq!(ok("{:04d}", &[n(-7.0)]), "-007");
    assert_eq!(ok("{:+d}", &[n(5.0)]), "+5");
    assert_eq!(ok("{: d}", &[n(5.0)]), " 5");
    assert_eq!(ok("{:x}", &[n(255.0)]), "ff");
    assert_eq!(ok("{:X}", &[n(255.0)]), "FF");
    assert_eq!(ok("{:#x}", &[n(255.0)]), "0xff");
    assert_eq!(ok("{:b}", &[n(5.0)]), "101");
    assert_eq!(ok("{:o}", &[n(8.0)]), "10");
    assert_eq!(ok("{:,}", &[n(1_234_567.0)]), "1,234,567");
    assert_eq!(ok("{:,d}", &[n(-1000.0)]), "-1,000");
}

#[test]
fn fixed_point() {
    assert_eq!(ok("{:.4f}", &[n(PI)]), "3.1416");
    assert_eq!(ok("{:8.2f}", &[n(PI)]), "    3.14");
    assert_eq!(ok("{:f}", &[n(1.5)]), "1.500000");
    assert_eq!(ok("{:08.2f}", &[n(-PI)]), "-0003.14");
    assert_eq!(ok("{:+.1f}", &[n(2.0)]), "+2.0");
    assert_eq!(ok("{:,.2f}", &[n(1_234_567.891)]), "1,234,567.89");
    assert_eq!(ok("{:.1%}", &[n(0.256)]), "25.6%");
    assert_eq!(ok("{:.2f}", &[n(f64::NAN)]), "nan");
    assert_eq!(ok("{:.2f}", &[n(f64::INFINITY)]), "inf");
}

#[test]
fn exponent_and_general() {
    assert_eq!(ok("{:e}", &[n(12345.678)]), "1.234568e+04");
    assert_eq!(ok("{:.2e}", &[n(0.000_123)]), "1.23e-04");
    assert_eq!(ok("{:E}", &[n(1.0)]), "1.000000E+00");
    assert_eq!(ok("{:g}", &[n(0.0001)]), "0.0001");
    assert_eq!(ok("{:g}", &[n(0.000_01)]), "1e-05");
    assert_eq!(ok("{:g}", &[n(1_234_567.0)]), "1.23457e+06");
    assert_eq!(ok("{:g}", &[n(100.0)]), "100");
    assert_eq!(ok("{:.3g}", &[n(PI)]), "3.14");
    assert_eq!(ok("{:g}", &[n(1.5)]), "1.5");
}

#[test]
fn errors_name_the_field_and_argument() {
    let e = format("{} {} {}", &[n(1.0), n(2.0)]).unwrap_err();
    assert!(e.contains("field 2") && e.contains("2 given"), "{e}");
    let e = format("x = {:d}", &[n(2.5)]).unwrap_err();
    assert!(
        e.contains("{:d}") && e.contains("argument 0") && e.contains("integer"),
        "{e}"
    );
    let e = format("{:d}", &[s("seven")]).unwrap_err();
    assert!(e.contains("string") && e.contains("argument 0"), "{e}");
    let e = format("{0} {}", &[n(1.0), n(2.0)]).unwrap_err();
    assert!(e.contains("automatic") && e.contains("manual"), "{e}");
    assert!(format("{:q}", &[n(1.0)]).unwrap_err().contains("'q'"));
    assert!(format("{", &[]).unwrap_err().contains("unmatched"));
}
