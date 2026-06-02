//! Tic-tac-toe policy-dataset builtins for the fine-tune demo.
//! `ttt_boards()` returns the `[N, 27]` one-hot board features and
//! `ttt_moves()` the `[N]` optimal-move labels -- together the complete
//! supervised dataset the literate page trains the classifier on. The
//! game tree + labelling live in `mlpl-tictactoe`; these are the
//! language-level data primitives (analogous to `load_preloaded`).

use mlpl_array::{DenseArray, Shape};
use mlpl_tictactoe::policy_dataset;

use mlpl_eval_types::{EvalError, Value};

pub(crate) fn eval_ttt_boards() -> Result<Value, EvalError> {
    let (x, _y, n) = policy_dataset();
    Ok(Value::Array(
        DenseArray::new(Shape::new(vec![n, 27]), x).unwrap(),
    ))
}

pub(crate) fn eval_ttt_moves() -> Result<Value, EvalError> {
    let (_x, y, n) = policy_dataset();
    Ok(Value::Array(
        DenseArray::new(Shape::new(vec![n]), y).unwrap(),
    ))
}
