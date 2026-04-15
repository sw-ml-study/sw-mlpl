use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, Value, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env)
}

fn eval_value(src: &str) -> Result<Value, EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env)
}

#[test]
fn iota_5() {
    let arr = eval("iota(5)").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(5));
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn shape_vector() {
    let arr = eval("shape([1, 2, 3])").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(1));
    assert_eq!(arr.data(), &[3.0]);
}

#[test]
fn shape_matrix() {
    let arr = eval("shape([[1, 2, 3], [4, 5, 6]])").unwrap();
    assert_eq!(arr.data(), &[2.0, 3.0]);
}

#[test]
fn rank_vector() {
    let arr = eval("rank([1, 2, 3])").unwrap();
    assert_eq!(arr.shape(), &Shape::scalar());
    assert_eq!(arr.data(), &[1.0]);
}

#[test]
fn rank_scalar() {
    let arr = eval("rank(42)").unwrap();
    assert_eq!(arr.data(), &[0.0]);
}

#[test]
fn reshape_iota() {
    let arr = eval("reshape(iota(6), [2, 3])").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn transpose_matrix() {
    let arr = eval("transpose(reshape(iota(6), [2, 3]))").unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![3, 2]));
    assert_eq!(arr.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn unknown_function() {
    let result = eval("unknown(1)");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("unknown"), "error: {err_msg}");
}

#[test]
fn reshape_mismatch() {
    let result = eval("reshape(iota(6), [2, 2])");
    assert!(result.is_err());
}

#[test]
fn iota_with_variable() {
    let tokens = lex("n = 4\niota(n)").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let arr = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn compose_reshape_arithmetic() {
    let tokens = lex("x = iota(6)\nm = reshape(x, [2, 3])\nm + 1").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let arr = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn labels_of_scalar_is_empty() {
    let v = eval_value("labels(42)").unwrap();
    assert_eq!(v, Value::Str(String::new()));
}

#[test]
fn labels_of_unlabeled_vector_is_empty() {
    let v = eval_value("labels(iota(3))").unwrap();
    assert_eq!(v, Value::Str(String::new()));
}

#[test]
fn labels_of_unlabeled_matrix_is_one_comma() {
    let v = eval_value("labels(reshape(iota(6), [2, 3]))").unwrap();
    assert_eq!(v, Value::Str(",".into()));
}

#[test]
fn labels_of_unlabeled_rank3_is_two_commas() {
    let v = eval_value("labels(reshape(iota(12), [2, 2, 3]))").unwrap();
    assert_eq!(v, Value::Str(",,".into()));
}

#[test]
fn shape_still_works_on_matrix_after_labels_exist() {
    let arr = eval("shape(reshape(iota(6), [2, 3]))").unwrap();
    assert_eq!(arr.data(), &[2.0, 3.0]);
}

// -- label() builtin and annotation syntax (Saga 11.5 Phase 2) --

#[test]
fn label_builtin_round_trips_single_axis() {
    let v = eval_value("labels(label(iota(5), [\"seq\"]))").unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

#[test]
fn label_builtin_round_trips_matrix() {
    let src = "labels(label(reshape(iota(6), [2, 3]), [\"rows\", \"cols\"]))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("rows,cols".into()));
}

#[test]
fn label_preserves_shape_and_data() {
    let src = "label(reshape(iota(6), [2, 3]), [\"rows\", \"cols\"])";
    let arr = eval(src).unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn label_rank_mismatch_errors() {
    // Matrix (rank 2) but only one label provided.
    let result = eval("label(reshape(iota(6), [2, 3]), [\"only_one\"])");
    assert!(result.is_err(), "expected rank-mismatch error");
}

#[test]
fn label_rejects_non_string_entries() {
    // Numeric label entries are not labels.
    let result = eval("label(iota(3), [1, 2, 3])");
    assert!(result.is_err(), "expected non-string label error");
}

#[test]
fn label_leaves_original_unlabeled() {
    // Round-trip: labeling a bound variable produces a new value;
    // the original binding is untouched because `label` is pure.
    let src = "x = iota(3)\ny = label(x, [\"seq\"])\nlabels(x)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str(String::new()));
}

#[test]
fn annotation_syntax_attaches_labels() {
    let src = "x : [batch, dim] = reshape(iota(6), [2, 3])\nlabels(x)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("batch,dim".into()));
}

#[test]
fn annotation_syntax_preserves_shape_and_data() {
    let src = "x : [batch, dim] = reshape(iota(6), [2, 3])\nx";
    let arr = eval(src).unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn annotation_syntax_single_axis() {
    let src = "v : [seq] = iota(4)\nlabels(v)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

#[test]
fn annotation_rank_mismatch_errors() {
    // Declared 1 axis but value is rank 2 -- should error at evaluation.
    let result = eval("x : [only_one] = reshape(iota(6), [2, 3])");
    assert!(result.is_err(), "expected rank-mismatch error");
}

#[test]
fn unlabeled_assign_still_works() {
    // Without annotation, behavior is unchanged (positional).
    let src = "x = reshape(iota(6), [2, 3])\nlabels(x)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str(",".into()));
}
