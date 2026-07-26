//! `feasible(estimate, budget) -> 0/1`: gate-pattern check
//! used to guard a real `train { }` call. Pure arithmetic
//! over two rank-1 arrays.

use mlpl_array::DenseArray;
use mlpl_parser::Expr;

use crate::error::FeasibilityError;

pub fn feasible_inner<E, A, Err>(args: &[Expr], env: &mut E, array: A) -> Result<DenseArray, Err>
where
    A: Fn(&Expr, &mut E) -> Result<DenseArray, Err>,
    Err: From<FeasibilityError>,
{
    let [a0, a1] = args else {
        return Err(FeasibilityError::BadArity {
            func: "feasible".into(),
            expected: 2,
            got: args.len(),
        }
        .into());
    };
    let est = array(a0, env)?;
    let budget = array(a1, env)?;
    check_shapes(&est, &budget)?;
    let e = est.data();
    let b = budget.data();
    let passes = (b[0] == 0.0 || e[1] <= b[0])
        && (b[1] == 0.0 || e[2] <= b[1])
        && (b[2] == 0.0 || e[4] <= b[2]);
    Ok(DenseArray::from_scalar(if passes { 1.0 } else { 0.0 }))
}

/// `est` must be `[5]` and `budget` `[3]`.
fn check_shapes(est: &DenseArray, budget: &DenseArray) -> Result<(), FeasibilityError> {
    if est.shape().dims() != [5] {
        return Err(FeasibilityError::BadEstimateShape(
            est.shape().dims().to_vec(),
        ));
    }
    if budget.shape().dims() != [3] {
        return Err(FeasibilityError::BadBudgetShape(
            budget.shape().dims().to_vec(),
        ));
    }
    Ok(())
}
