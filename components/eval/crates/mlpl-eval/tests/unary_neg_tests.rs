use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{Expr, lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

fn eval_env(src: &str, env: &mut Environment) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    eval_program(&stmts, env).unwrap()
}

// -- Parser tests --

#[test]
fn parse_neg_ident() {
    let tokens = lex("-x").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert_eq!(stmts.len(), 1);
    assert!(matches!(stmts[0], Expr::UnaryNeg { .. }));
}

#[test]
fn parse_neg_paren_expr() {
    let tokens = lex("-(1 + 2)").unwrap();
    let stmts = parse(&tokens).unwrap();
    assert!(matches!(stmts[0], Expr::UnaryNeg { .. }));
}

// -- Eval tests --

#[test]
fn eval_neg_scalar() {
    let arr = eval("-5").unwrap();
    assert_eq!(arr.data(), &[-5.0]);
}

#[test]
fn eval_neg_variable() {
    let mut env = Environment::new();
    eval_env("x = 3", &mut env);
    let arr = eval_env("-x", &mut env);
    assert_eq!(arr.data(), &[-3.0]);
}

#[test]
fn eval_neg_array() {
    let arr = eval("-[1, 2, 3]").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[-1.0, -2.0, -3.0]);
}

#[test]
fn eval_add_neg() {
    let arr = eval("1 + -2").unwrap();
    assert_eq!(arr.data(), &[-1.0]);
}

#[test]
fn eval_neg_paren() {
    let arr = eval("-(1 + 2)").unwrap();
    assert_eq!(arr.data(), &[-3.0]);
}

#[test]
fn eval_neg_literal_unchanged() {
    // "-3" should still work (lexer produces IntLit(-3))
    let arr = eval("-3").unwrap();
    assert_eq!(arr.data(), &[-3.0]);
}
