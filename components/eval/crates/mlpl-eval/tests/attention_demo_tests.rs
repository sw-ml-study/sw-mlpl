//! Integration test for the attention pattern demo.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> mlpl_array::DenseArray {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program(&stmts, &mut env).unwrap()
}

#[test]
fn attention_rows_sum_to_one_and_diagonal_dominates_when_k_equals_q() {
    // Q = K so each token most attends to itself; the diagonal of
    // softmax(Q K^T / sqrt(d)) should dominate.
    let src = r#"
Q = randn(17, [6, 4])
K = Q
S = matmul(Q, transpose(K)) / sqrt(4)
A = softmax(S, 1)
row_sums = reduce_add(A, 1)
diag = reduce_add(A * eq(matmul(reshape(iota(6), [6, 1]), ones([1, 6])) - matmul(ones([6, 1]), reshape(iota(6), [1, 6])), 0), 1)
diag_mean = mean(diag)
row_sums
"#;
    let rs = eval(src);
    assert_eq!(rs.shape().dims(), &[6]);
    for &s in rs.data() {
        assert!((s - 1.0).abs() < 1e-9, "row sum {s}");
    }

    // Diagonal mean should be well above the uniform value 1/6.
    let src2 = r#"
Q = randn(17, [6, 4])
K = Q
S = matmul(Q, transpose(K)) / sqrt(4)
A = softmax(S, 1)
I = eq(matmul(reshape(iota(6), [6, 1]), ones([1, 6])) - matmul(ones([6, 1]), reshape(iota(6), [1, 6])), 0)
mean(reduce_add(A * I, 1))
"#;
    let dm = eval(src2).data()[0];
    assert!(
        dm > 0.35,
        "diagonal mean {dm} should dominate (uniform would be ~0.167)"
    );
}
