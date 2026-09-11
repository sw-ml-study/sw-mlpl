//! axis-naming-unification parity: `label` and `reduce` accept the SAME
//! axis-name forms -- a bracketed list of names, an equivalent
//! comma-string, and (for label) a computed value. This is the guard that
//! the two places you name axes cannot drift apart again.

use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn label_and_reduce_share_the_bracketed_and_comma_forms() {
    // reshape(range(6),[2,3]) = [[0,1,2],[3,4,5]]; reduce over row axis "r"
    // -> column sums [3,5,7]. Every combination of label-form x reduce-form
    // must succeed and agree.
    let cases = [
        "reduce(:add, label(M, [\"r\", \"c\"]), [\"r\"])",
        "reduce(:add, label(M, \"r,c\"),        [\"r\"])",
        "reduce(:add, label(M, [\"r\", \"c\"]), \"r\")",
        "reduce(:add, label(M, \"r,c\"),        \"r\")",
    ];
    for c in cases {
        let src = format!("M = reshape(range(6), [2, 3]); {c}");
        let y = eval(&src).unwrap_or_else(|e| panic!("case {c}: {e:?}"));
        assert_eq!(y.data(), &[3.0, 5.0, 7.0], "case: {c}");
    }
}

#[test]
fn label_accepts_a_computed_name_vector() {
    // The new capability: label's argument is EVALUATED, so a variable
    // holding the names works -- names are first-class data.
    let y = eval("n = [\"r\", \"c\"]; reduce(:add, label(reshape(range(6), [2, 3]), n), [\"r\"])")
        .unwrap();
    assert_eq!(y.data(), &[3.0, 5.0, 7.0]);
}

#[test]
fn reshape_labeled_accepts_a_comma_string() {
    // reshape_labeled's name arg is unified too.
    let y = eval("reduce(:add, reshape_labeled(range(6), [2, 3], \"r,c\"), [\"c\"])").unwrap();
    // reduce over "c" (=axis 1): row sums [0+1+2, 3+4+5] = [3, 12].
    assert_eq!(y.data(), &[3.0, 12.0]);
}

#[test]
fn label_still_rejects_non_name_forms() {
    // A numeric list is not a set of names.
    assert!(eval("label(reshape(range(6), [2, 3]), [1, 2])").is_err());
    // A scalar is not a set of names.
    assert!(eval("label(reshape(range(6), [2, 3]), 5)").is_err());
}
