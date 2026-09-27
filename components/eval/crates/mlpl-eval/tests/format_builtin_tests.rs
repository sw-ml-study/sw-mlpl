//! `format(template, args...)`: Python-style replacement fields in MLPL
//! (the formatter itself is pinned in mlpl-format's tests). This pins the
//! builtin surface: variadic arguments, scalars as numbers, strings, a
//! non-scalar value formatting as its display text, and errors.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<Value, EvalError> {
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), env)?;
    }
    Ok(last)
}

fn text(src: &str) -> String {
    match run(&mut Environment::new(), src) {
        Ok(Value::Str(s)) => s,
        other => panic!("{src}: {other:?}"),
    }
}

#[test]
fn a_training_progress_line() {
    let got = text(
        "step = 9\nnum_steps = 1000\nloss = 2.345678\n\
         format(\"step {:4d} / {:4d} | loss {:.4f}\", step + 1, num_steps, loss)",
    );
    assert_eq!(got, "step   10 / 1000 | loss 2.3457");
}

#[test]
fn strings_numbers_and_escapes() {
    assert_eq!(
        text("format(\"{} has {} rows\", \"corpus\", 32033)"),
        "corpus has 32033 rows"
    );
    assert_eq!(
        text("format(\"{:>6}|{:<6}|\", \"ab\", \"cd\")"),
        "    ab|cd    |"
    );
    assert_eq!(
        text("format(\"{{literal}} {:.1%}\", 0.25)"),
        "{literal} 25.0%"
    );
    assert_eq!(text("format(\"no fields\")"), "no fields");
}

#[test]
fn a_non_scalar_formats_as_its_display_text() {
    assert_eq!(text("format(\"v = {}\", [1, 2, 3])"), "v = 1 2 3");
    let err = run(&mut Environment::new(), "format(\"{:.2f}\", [1, 2])").unwrap_err();
    assert!(err.to_string().contains("argument 0"), "{err}");
}

#[test]
fn errors_are_catchable_and_named() {
    let err = run(&mut Environment::new(), "format(\"{} {}\", 1)").unwrap_err();
    assert!(err.to_string().contains("only 1 given"), "{err}");
    let err = run(&mut Environment::new(), "format(3)").unwrap_err();
    assert!(err.to_string().contains("template"), "{err}");
}
