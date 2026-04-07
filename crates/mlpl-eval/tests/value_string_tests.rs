use mlpl_eval::{Environment, EvalError, Value, eval_program, eval_program_value};

fn parse(input: &str) -> Vec<mlpl_parser::Expr> {
    let toks = mlpl_parser::lex(input).unwrap();
    mlpl_parser::parse(&toks).unwrap()
}

#[test]
fn eval_string_literal_via_value() {
    let stmts = parse(r#""hello""#);
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    assert_eq!(v, Value::Str("hello".to_string()));
}

#[test]
fn eval_string_literal_via_array_returns_error() {
    let stmts = parse(r#""hello""#);
    let mut env = Environment::new();
    let r = eval_program(&stmts, &mut env);
    assert!(matches!(r, Err(EvalError::ExpectedArray)));
}

#[test]
fn array_result_works_via_value() {
    let stmts = parse("1 + 2");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    let arr = v.into_array().unwrap();
    assert_eq!(arr.data(), &[3.0]);
}

#[test]
fn string_in_arithmetic_is_error() {
    let stmts = parse(r#""x" + 1"#);
    let mut env = Environment::new();
    let r = eval_program_value(&stmts, &mut env);
    assert!(matches!(r, Err(EvalError::ExpectedArray)));
}

#[test]
fn last_statement_decides_type() {
    // First stmt is array, last is string
    let stmts = parse(
        r#"x = 42
"hello""#,
    );
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    assert_eq!(v, Value::Str("hello".to_string()));
}
