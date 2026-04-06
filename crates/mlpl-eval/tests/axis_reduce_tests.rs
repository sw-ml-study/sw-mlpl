use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

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

// -- Column sums (axis 0) --

#[test]
fn reduce_add_axis_0() {
    // [[1,2,3],[4,5,6]] reduced along axis 0 -> [5, 7, 9]
    let mut env = Environment::new();
    eval_env("m = [[1, 2, 3], [4, 5, 6]]", &mut env);
    let arr = eval_env("reduce_add(m, 0)", &mut env);
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[5.0, 7.0, 9.0]);
}

// -- Row sums (axis 1) --

#[test]
fn reduce_add_axis_1() {
    // [[1,2,3],[4,5,6]] reduced along axis 1 -> [6, 15]
    let mut env = Environment::new();
    eval_env("m = [[1, 2, 3], [4, 5, 6]]", &mut env);
    let arr = eval_env("reduce_add(m, 1)", &mut env);
    assert_eq!(arr.shape(), &Shape::vector(2));
    assert_eq!(arr.data(), &[6.0, 15.0]);
}

// -- reduce_mul along axis --

#[test]
fn reduce_mul_axis_0() {
    // [[1,2],[3,4]] reduced along axis 0 -> [3, 8]
    let mut env = Environment::new();
    eval_env("m = [[1, 2], [3, 4]]", &mut env);
    let arr = eval_env("reduce_mul(m, 0)", &mut env);
    assert_eq!(arr.data(), &[3.0, 8.0]);
}

// -- 1-arg still works (reduce all) --

#[test]
fn reduce_add_one_arg_unchanged() {
    let arr = eval("reduce_add([1, 2, 3])").unwrap();
    assert_eq!(arr.data(), &[6.0]);
}

#[test]
fn reduce_mul_one_arg_unchanged() {
    let arr = eval("reduce_mul([2, 3, 4])").unwrap();
    assert_eq!(arr.data(), &[24.0]);
}

// -- Invalid axis --

#[test]
fn reduce_add_invalid_axis() {
    let mut env = Environment::new();
    eval_env("v = [1, 2, 3]", &mut env);
    let tokens = lex("reduce_add(v, 1)").unwrap();
    let stmts = parse(&tokens).unwrap();
    let result = eval_program(&stmts, &mut env);
    assert!(result.is_err());
}

// -- With iota + reshape --

#[test]
fn reduce_add_axis_with_reshape() {
    let mut env = Environment::new();
    eval_env("m = reshape(iota(12), [3, 4])", &mut env);
    // axis 0: sum each column -> [12, 15, 18, 21]
    let arr = eval_env("reduce_add(m, 0)", &mut env);
    assert_eq!(arr.shape(), &Shape::vector(4));
    assert_eq!(arr.data(), &[12.0, 15.0, 18.0, 21.0]);
}
