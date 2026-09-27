//! `write(...)` -- `print` without the newline (arguments concatenated,
//! stdout flushed so `\r` progress lines appear) -- and variadic
//! `str_concat(a, b, c, ...)`. Replaces
//! `unwrap(write_stdout(tokenize_bytes(s)))` and nested `str_concat` towers.

use mlpl_eval::{Environment, EvalError, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(src: &str) -> Result<Value, EvalError> {
    let mut env = Environment::new();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in src.lines().filter(|l| !l.trim().is_empty()) {
        last = eval_program_value(&parse(&lex(line).unwrap()).unwrap(), &mut env)?;
    }
    Ok(last)
}

#[test]
fn str_concat_takes_any_number_of_strings() {
    assert_eq!(
        run("str_concat(\"a\", \"b\", \"c\", \"d\")").unwrap(),
        Value::Str("abcd".into())
    );
    assert_eq!(
        run("str_concat(\"a\", \"b\")").unwrap(),
        Value::Str("ab".into())
    );
    let err = run("str_concat(\"a\")").unwrap_err().to_string();
    assert!(err.contains("str_concat"), "{err}");
    let err = run("str_concat(\"a\", 1, \"c\")").unwrap_err().to_string();
    assert!(
        err.contains("argument 1") && err.contains("string"),
        "{err}"
    );
}

#[test]
fn write_returns_the_text_it_wrote_without_a_newline() {
    assert_eq!(
        run("write(\"step \", 10, \" of \", 1000, \"\\r\")").unwrap(),
        Value::Str("step 10 of 1000\r".into())
    );
    assert_eq!(
        run("write(format(\"{:.2f}\", 2.5))").unwrap(),
        Value::Str("2.50".into())
    );
    assert!(run("write()").is_err());
}
