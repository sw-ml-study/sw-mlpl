//! Argument validation helpers for dataset builtins. Extracted
//! out of `dataset_builtins.rs` to keep that module's user-
//! visible builtin function count under the per-module budget.

use mlpl_array::DenseArray;

use crate::error::RuntimeError;

/// Validate the args to `shift_pairs_x` / `shift_pairs_y`:
/// `(ids: rank-1, block_size: scalar > 0)`. Returns
/// `(block_size, num_batches)` on success.
pub(crate) fn validate_shift_pairs(
    name: &str,
    args: &[DenseArray],
) -> Result<(usize, usize), RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[0].rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("ids must be rank-1, got rank {}", args[0].rank()),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "block_size must be a scalar".into(),
        });
    }
    let bs = args[1].data()[0] as usize;
    if bs == 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "block_size must be > 0".into(),
        });
    }
    let n = args[0].shape().dims()[0];
    let b = n / (bs + 1);
    if b == 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!(
                "ids length {n} is too short for block_size {bs} (need at least {} tokens)",
                bs + 1
            ),
        });
    }
    Ok((bs, b))
}
