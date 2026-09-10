//! `svg(text, "equation")` at the interpreter level: a Unicode math
//! string renders to a high-contrast SVG -- the math-behind-the-code
//! visual used by the convolution / moving-average demos.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run_string(src: &str) -> String {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    match eval_program_value(&stmts, &mut Environment::new()).unwrap() {
        Value::Str(s) => s,
        other => panic!("expected Value::Str, got {other:?}"),
    }
}

#[test]
fn equation_type_renders_a_unicode_math_svg() {
    // (I * K)[y,x] = SUM_c  with a real summation sign and a subscript.
    let svg = run_string("svg(\"(I \u{2217} K)[y,x] = \u{2211}_c\", \"equation\")");
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("fill=\"#1e1e2e\"")); // dark, high-contrast canvas
    assert!(svg.contains('\u{2211}')); // the summation sign survives lex->render
    assert!(svg.contains("baseline-shift=\"sub\"")); // the _c subscript
}

#[test]
fn equation_type_rejects_a_non_string_argument() {
    let stmts = parse(&lex("svg([1, 2, 3], \"equation\")").unwrap()).unwrap();
    assert!(eval_program_value(&stmts, &mut Environment::new()).is_err());
}
