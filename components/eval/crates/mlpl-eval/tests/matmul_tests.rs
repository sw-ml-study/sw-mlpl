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

#[test]
fn matmul_2x2() {
    let arr = eval("matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(arr.data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_identity() {
    let arr = eval("matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]])").unwrap();
    assert_eq!(arr.data(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_mat_vec() {
    let arr = eval("matmul([[1, 2], [3, 4]], [5, 6])").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(2));
    assert_eq!(arr.data(), &[17.0, 39.0]);
}

#[test]
fn matmul_dimension_mismatch() {
    let result = eval("matmul([[1, 2], [3, 4]], [[1, 2, 3]])");
    assert!(result.is_err());
}

#[test]
fn matmul_with_iota_reshape() {
    let mut env = Environment::new();
    eval_env("a = reshape(iota(6), [2, 3])", &mut env);
    eval_env("b = reshape(iota(6), [3, 2])", &mut env);
    let arr = eval_env("matmul(a, b)", &mut env);
    // [2,3] * [3,2] -> [2,2]
    // a = [[0,1,2],[3,4,5]], b = [[0,1],[2,3],[4,5]]
    // [0][0] = 0*0+1*2+2*4 = 10, [0][1] = 0*1+1*3+2*5 = 13
    // [1][0] = 3*0+4*2+5*4 = 28, [1][1] = 3*1+4*3+5*5 = 40
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(arr.data(), &[10.0, 13.0, 28.0, 40.0]);
}

#[test]
fn matmul_non_square() {
    // [2,3] * [3,1] -> [2,1]
    let arr = eval("matmul([[1, 2, 3], [4, 5, 6]], [[1], [0], [1]])").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 1]));
    assert_eq!(arr.data(), &[4.0, 10.0]);
}
