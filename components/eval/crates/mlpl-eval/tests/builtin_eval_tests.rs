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
fn linspace_inclusive() {
    let arr = eval("linspace(0.0, 1.0, 5)").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(5));
    assert_eq!(arr.data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn linspace_single_is_start() {
    assert_eq!(eval("linspace(3.0, 9.0, 1)").unwrap().data(), &[3.0]);
}

#[test]
fn running_product_accumulates() {
    let arr = eval("running_product([1.0, 2.0, 3.0, 4.0])").unwrap();
    assert_eq!(arr.shape(), &Shape::vector(4));
    assert_eq!(arr.data(), &[1.0, 2.0, 6.0, 24.0]);
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

// -- Label propagation through shape ops (Saga 11.5 Phase 2 cont.) --

#[test]
fn transpose_swaps_labels_in_repl() {
    let src = "x : [rows, cols] = reshape(iota(6), [2, 3])\nlabels(transpose(x))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("cols,rows".into()));
}

#[test]
fn transpose_unlabeled_stays_unlabeled() {
    let src = "labels(transpose(reshape(iota(6), [2, 3])))";
    let v = eval_value(src).unwrap();
    // Unlabeled rank-2 shows as "," (two positional axes).
    assert_eq!(v, Value::Str(",".into()));
}

#[test]
fn reshape_drops_labels_in_repl() {
    let src = "x : [rows, cols] = reshape(iota(6), [2, 3])\nlabels(reshape(x, [6]))";
    let v = eval_value(src).unwrap();
    // After reshape, labels cleared; rank-1 unlabeled renders as "".
    assert_eq!(v, Value::Str(String::new()));
}

#[test]
fn reshape_labeled_attaches_new_labels() {
    let src = "labels(reshape_labeled(iota(6), [2, 3], [\"r\", \"c\"]))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("r,c".into()));
}

#[test]
fn reshape_labeled_preserves_shape_and_data() {
    let src = "reshape_labeled(iota(6), [2, 3], [\"r\", \"c\"])";
    let arr = eval(src).unwrap();
    assert_eq!(arr.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn reshape_labeled_rank_mismatch_errors() {
    // 2-d shape but only 1 label.
    let result = eval("reshape_labeled(iota(6), [2, 3], [\"only_one\"])");
    assert!(result.is_err(), "expected label-rank mismatch error");
}

#[test]
fn reshape_labeled_shape_mismatch_errors() {
    // Element count does not match.
    let result = eval("reshape_labeled(iota(6), [2, 2], [\"r\", \"c\"])");
    assert!(result.is_err(), "expected shape mismatch error");
}

#[test]
fn reshape_labeled_from_labeled_source() {
    // Labels on the source do not leak through; only the supplied
    // labels matter. This matches the `reshape clears labels` rule.
    let src = "x : [rows, cols] = reshape(iota(6), [2, 3])\n\
               labels(reshape_labeled(x, [3, 2], [\"a\", \"b\"]))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("a,b".into()));
}

#[test]
fn reshape_labeled_rejects_non_string_labels() {
    let result = eval("reshape_labeled(iota(6), [2, 3], [1, 2])");
    assert!(result.is_err(), "expected non-string label error");
}

// -- Elementwise label propagation in REPL (Saga 11.5 Phase 3) --

#[test]
fn binop_matching_labels_propagate_in_repl() {
    let src = "x : [seq] = iota(3)\n\
               y : [seq] = iota(3)\n\
               labels(x + y)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

#[test]
fn binop_mismatched_labels_error_in_repl() {
    let src = "x : [seq] = iota(3)\n\
               y : [batch] = iota(3)\n\
               x + y";
    let result = eval(src);
    assert!(result.is_err(), "expected label mismatch error");
}

#[test]
fn binop_labeled_plus_unlabeled_in_repl() {
    let src = "x : [seq] = iota(3)\n\
               y = iota(3)\n\
               labels(x + y)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

#[test]
fn binop_scalar_times_labeled_in_repl() {
    let src = "x : [seq] = iota(3)\nlabels(2 * x)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

#[test]
fn binop_both_unlabeled_in_repl_unchanged() {
    let src = "a = iota(3)\nb = iota(3)\nlabels(a + b)";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str(String::new()));
}

#[test]
fn relabel_overrides_labels() {
    let src = "x : [seq] = iota(3)\nlabels(relabel(x, [\"time\"]))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("time".into()));
}

#[test]
fn relabel_escape_hatch_resolves_mismatch() {
    // Labels would clash on `x + y`, but relabel resolves it.
    let src = "x : [seq] = iota(3)\n\
               y : [batch] = iota(3)\n\
               labels(x + relabel(y, [\"seq\"]))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq".into()));
}

// -- matmul / reduce / softmax by axis name (Saga 11.5 Phase 3 cont.) --

#[test]
fn matmul_propagates_labels_in_repl() {
    let src = "a = label(reshape(iota(6), [2, 3]), [\"seq\", \"d\"])\n\
               b = label(reshape(iota(12), [3, 4]), [\"d\", \"heads\"])\n\
               labels(matmul(a, b))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("seq,heads".into()));
}

#[test]
fn matmul_mismatched_contraction_errors_in_repl() {
    let src = "a = label(reshape(iota(6), [2, 3]), [\"seq\", \"d\"])\n\
               b = label(reshape(iota(12), [3, 4]), [\"time\", \"heads\"])\n\
               matmul(a, b)";
    let result = eval(src);
    assert!(result.is_err(), "expected contraction label mismatch");
}

#[test]
fn reduce_add_by_axis_name_matches_int() {
    let mut env_int = Environment::new();
    let mut env_name = Environment::new();
    let tokens =
        lex("m = label(reshape(iota(6), [2, 3]), [\"batch\", \"feat\"])\nreduce_add(m, 1)")
            .unwrap();
    let stmts = parse(&tokens).unwrap();
    let by_int = eval_program(&stmts, &mut env_int).unwrap();

    let tokens =
        lex("m = label(reshape(iota(6), [2, 3]), [\"batch\", \"feat\"])\nreduce_add(m, \"feat\")")
            .unwrap();
    let stmts = parse(&tokens).unwrap();
    let by_name = eval_program(&stmts, &mut env_name).unwrap();

    assert_eq!(by_int.data(), by_name.data());
    assert_eq!(by_int.shape(), by_name.shape());
}

#[test]
fn reduce_add_by_axis_name_drops_label() {
    let src = "m = label(reshape(iota(6), [2, 3]), [\"batch\", \"feat\"])\n\
               labels(reduce_add(m, \"feat\"))";
    let v = eval_value(src).unwrap();
    // Only "batch" remains after reducing "feat".
    assert_eq!(v, Value::Str("batch".into()));
}

#[test]
fn reduce_mul_by_axis_name() {
    let src = "m = label(reshape(iota(6), [2, 3]) + 1, [\"batch\", \"feat\"])\n\
               labels(reduce_mul(m, \"batch\"))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("feat".into()));
}

#[test]
fn argmax_by_axis_name_matches_int() {
    let src_int = "m = label(reshape(iota(6), [2, 3]), [\"batch\", \"class\"])\nargmax(m, 1)";
    let src_name =
        "m = label(reshape(iota(6), [2, 3]), [\"batch\", \"class\"])\nargmax(m, \"class\")";
    let by_int = eval(src_int).unwrap();
    let by_name = eval(src_name).unwrap();
    assert_eq!(by_int.data(), by_name.data());
}

#[test]
fn softmax_by_axis_name_matches_int() {
    let src_int = "m = label(reshape(iota(6), [2, 3]) + 0.0, [\"batch\", \"cls\"])\nsoftmax(m, 1)";
    let src_name =
        "m = label(reshape(iota(6), [2, 3]) + 0.0, [\"batch\", \"cls\"])\nsoftmax(m, \"cls\")";
    let by_int = eval(src_int).unwrap();
    let by_name = eval(src_name).unwrap();
    assert_eq!(by_int.data(), by_name.data());
}

#[test]
fn reduce_add_by_axis_int_still_works_on_labeled() {
    let src = "m = label(reshape(iota(6), [2, 3]), [\"batch\", \"feat\"])\n\
               labels(reduce_add(m, 0))";
    let v = eval_value(src).unwrap();
    assert_eq!(v, Value::Str("feat".into()));
}

#[test]
fn reduce_add_unknown_axis_name_errors() {
    let src = "m = label(reshape(iota(6), [2, 3]), [\"batch\", \"feat\"])\n\
               reduce_add(m, \"unknown\")";
    let result = eval(src);
    assert!(result.is_err(), "expected unknown axis name error");
}

// -- Structured ShapeMismatch error (Saga 11.5 Phase 4) --

#[test]
fn binop_label_mismatch_yields_structured_error() {
    let src = "x : [seq] = iota(3)\n\
               y : [batch] = iota(3)\n\
               x + y";
    let err = eval(src).unwrap_err();
    match err {
        EvalError::ShapeMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, "add");
            assert_eq!(expected.dims(), &[3]);
            assert_eq!(expected.labels(), &[Some("seq".into())]);
            assert_eq!(actual.labels(), &[Some("batch".into())]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn binop_broadcast_shape_mismatch_yields_structured_error() {
    // Same-rank but incompatible shapes: [3] vs [4].
    let src = "a = iota(3)\nb = iota(4)\na + b";
    let err = eval(src).unwrap_err();
    match err {
        EvalError::ShapeMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, "add");
            assert_eq!(expected.dims(), &[3]);
            assert_eq!(actual.dims(), &[4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn matmul_label_mismatch_yields_structured_error() {
    let src = "a = label(reshape(iota(6), [2, 3]), [\"seq\", \"d\"])\n\
               b = label(reshape(iota(12), [3, 4]), [\"time\", \"heads\"])\n\
               matmul(a, b)";
    let err = eval(src).unwrap_err();
    match err {
        EvalError::ShapeMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, "matmul");
            assert_eq!(expected.labels(), &[Some("seq".into()), Some("d".into())]);
            assert_eq!(
                actual.labels(),
                &[Some("time".into()), Some("heads".into())]
            );
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn matmul_shape_mismatch_yields_structured_error() {
    // [2,3] @ [4,5] -> contraction axis size mismatch
    let src = "a = reshape(iota(6), [2, 3])\n\
               b = reshape(iota(20), [4, 5])\n\
               matmul(a, b)";
    let err = eval(src).unwrap_err();
    assert!(
        matches!(err, EvalError::ShapeMismatch { ref op, .. } if op == "matmul"),
        "expected ShapeMismatch for matmul, got {err:?}"
    );
}

#[test]
fn shape_mismatch_display_names_op_and_shapes() {
    let src = "x : [seq] = iota(3)\n\
               y : [batch] = iota(3)\n\
               x + y";
    let err = eval(src).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("add"), "{msg}");
    assert!(msg.contains("seq"), "{msg}");
    assert!(msg.contains("batch"), "{msg}");
}

#[test]
fn running_sum_accumulates() {
    let arr = eval("running_sum([1.0, 2.0, 3.0, 4.0])").unwrap();
    assert_eq!(arr.data(), &[1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn scans_compose_with_lenses_and_reject_higher_ranks() {
    // GET-lens composition: focus a row or column, then scan it.
    let row = eval("M = reshape(range(6), [2, 3]); running_sum(take(M, 0, 1))").unwrap();
    assert_eq!(row.data(), &[3.0, 7.0, 12.0]);
    let col = eval("M = reshape(range(6), [2, 3]); running_product(take(M, 1, 2) + 1)").unwrap();
    assert_eq!(col.data(), &[3.0, 18.0]);
    // The ravel lens makes the whole-array scan EXPLICIT.
    let whole = eval("M = reshape(range(6), [2, 3]); running_sum(flatten(M))").unwrap();
    assert_eq!(whole.data(), &[0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    // A bare higher-rank input is an error that names both lenses.
    let err = eval("M = reshape(range(6), [2, 3]); running_sum(M)").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("take(") && msg.contains("flatten("),
        "hinting error: {msg}"
    );
}

#[test]
fn grade_up_and_down_are_stable_argsorts() {
    let up = eval("grade_up([3.0, 1.0, 2.0])").unwrap();
    assert_eq!(up.data(), &[1.0, 2.0, 0.0]);
    let down = eval("grade_down([3.0, 1.0, 2.0])").unwrap();
    assert_eq!(down.data(), &[0.0, 2.0, 1.0]);
    // Stability: ties keep original order in BOTH directions.
    let up_t = eval("grade_up([1.0, 1.0, 0.0])").unwrap();
    assert_eq!(up_t.data(), &[2.0, 0.0, 1.0]);
    let down_t = eval("grade_down([1.0, 1.0, 0.0])").unwrap();
    assert_eq!(down_t.data(), &[0.0, 1.0, 2.0]);
    // The curriculum idiom: gather_rows(X, grade_up(d)).
    let sorted =
        eval("X = reshape(range(6), [3, 2]); d = [5.0, 1.0, 3.0]; gather_rows(X, grade_up(d))")
            .unwrap();
    assert_eq!(sorted.data(), &[2.0, 3.0, 4.0, 5.0, 0.0, 1.0]);
}

#[test]
fn compress_keeps_masked_slices() {
    let v = eval("compress([1, 0, 1], [10.0, 20.0, 30.0])").unwrap();
    assert_eq!(v.data(), &[10.0, 30.0]);
    // Rows (default axis 0) and columns (axis 1) of a matrix.
    let rows = eval("M = reshape(range(6), [3, 2]); compress([0, 1, 1], M)").unwrap();
    assert_eq!(rows.shape().dims(), &[2, 2]);
    assert_eq!(rows.data(), &[2.0, 3.0, 4.0, 5.0]);
    let cols = eval("M = reshape(range(6), [2, 3]); compress([1, 0, 1], M, 1)").unwrap();
    assert_eq!(cols.shape().dims(), &[2, 2]);
    assert_eq!(cols.data(), &[0.0, 2.0, 3.0, 5.0]);
    // The rejection idiom: keep candidates over a threshold.
    let kept =
        eval("C = reshape(range(8), [4, 2]); s = [0.2, 0.95, 0.5, 0.99]; compress(gt(s, 0.9), C)")
            .unwrap();
    assert_eq!(kept.data(), &[2.0, 3.0, 6.0, 7.0]);
    // Length mismatch is an error.
    assert!(eval("compress([1, 0], [1.0, 2.0, 3.0])").is_err());
}
