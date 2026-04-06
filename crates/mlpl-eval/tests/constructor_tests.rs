use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

#[test]
fn zeros_vector() {
    let arr = eval("zeros([3])").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn zeros_matrix() {
    let arr = eval("zeros([2, 3])").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0; 6]);
}

#[test]
fn ones_vector() {
    let arr = eval("ones([3])").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(3));
    assert_eq!(arr.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn ones_matrix() {
    let arr = eval("ones([2, 2])").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(arr.data(), &[1.0; 4]);
}

#[test]
fn fill_vector() {
    let arr = eval("fill([3], 5)").unwrap();
    assert_eq!(arr.data(), &[5.0, 5.0, 5.0]);
}

#[test]
fn fill_matrix() {
    let arr = eval("fill([2, 2], 0.1)").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 2]));
    for &v in arr.data() {
        assert!((v - 0.1).abs() < 1e-10);
    }
}
