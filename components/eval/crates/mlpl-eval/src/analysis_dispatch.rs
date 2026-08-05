//! Dispatch for the `mlpl_viz` high-level analysis builtins (`hist`,
//! `scatter_labeled`, `loss_curve`, `train_val_curve`, `loss_landscape`,
//! `confusion_matrix`, `boundary_2d`). Split out of `eval_ops` so the
//! dispatcher stays within the function-LOC budget. `name` is already
//! validated by `eval_ops::eval_analysis_helper`, so the catch-all arms
//! only ever see their own remaining builtin.

use mlpl_array::DenseArray;
use mlpl_eval_types::EvalError;

/// Assert the analysis builtin `name` received exactly `n` arguments.
fn arity(name: &str, a: &[DenseArray], n: usize) -> Result<(), EvalError> {
    if a.len() == n {
        Ok(())
    } else {
        Err(EvalError::BadArity {
            func: name.into(),
            expected: n,
            got: a.len(),
        })
    }
}

/// Render one analysis builtin to its SVG string, fanning out by argument
/// shape (1-2 arg curves/scatters vs 3-4 arg grids/surfaces).
pub(crate) fn call_analysis(name: &str, a: &[DenseArray]) -> Result<String, EvalError> {
    match name {
        "hist" | "scatter_labeled" | "loss_curve" | "train_val_curve" => curves(name, a),
        _ => grids(name, a),
    }
}

/// The 1-2 argument vector renderers.
fn curves(name: &str, a: &[DenseArray]) -> Result<String, EvalError> {
    Ok(match name {
        "loss_curve" => {
            arity(name, a, 1)?;
            mlpl_viz::analysis_loss_curve(&a[0])?
        }
        "scatter_labeled" => {
            arity(name, a, 2)?;
            mlpl_viz::analysis_scatter_labeled(&a[0], &a[1])?
        }
        "train_val_curve" => {
            arity(name, a, 2)?;
            mlpl_viz::analysis_train_val_curve(&a[0], &a[1])?
        }
        _ => {
            arity(name, a, 2)?;
            if a[1].rank() != 0 {
                return Err(EvalError::ExpectedString);
            }
            mlpl_viz::analysis_hist(&a[0], a[1].data()[0] as usize)?
        }
    })
}

/// The 3-4 argument grid/surface renderers.
fn grids(name: &str, a: &[DenseArray]) -> Result<String, EvalError> {
    Ok(match name {
        "loss_landscape" => {
            arity(name, a, 3)?;
            mlpl_viz::analysis_loss_landscape(&a[0], &a[1], &a[2])?
        }
        "confusion_matrix" => {
            arity(name, a, 2)?;
            mlpl_viz::analysis_confusion_matrix(&a[0], &a[1])?
        }
        "pareto_plot" => {
            arity(name, a, 2)?;
            let mask =
                mlpl_runtime::call_builtin("pareto_front", vec![a[0].clone(), a[1].clone()])?;
            mlpl_viz::analysis_pareto_plot(&a[0], &mask)?
        }
        _ => {
            arity(name, a, 4)?;
            mlpl_viz::analysis_boundary_2d(&a[0], &a[1], &a[2], &a[3])?
        }
    })
}
