//! Small array UTILITIES: `concat(a, b)` (rank-0/1 vector concatenation)
//! and `last_row(m)` (the final row of a rank-2 matrix).

use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub(crate) fn array_util(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if name == "concat" {
        concat_1d(name, args)
    } else {
        last_row(name, args)
    }
}

fn concat_1d(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    for (i, a) in args.iter().enumerate() {
        if a.rank() > 1 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("argument {i} must be rank 0 or 1, got rank {}", a.rank()),
            });
        }
    }
    let mut data = Vec::with_capacity(args[0].data().len() + args[1].data().len());
    data.extend_from_slice(args[0].data());
    data.extend_from_slice(args[1].data());
    Ok(DenseArray::from_vec(data))
}

fn last_row(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    if args[0].rank() != 2 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("expected rank-2 matrix, got rank {}", args[0].rank()),
        });
    }
    let cols = args[0].shape().dims()[1];
    let data = args[0].data();
    Ok(DenseArray::from_vec(data[data.len() - cols..].to_vec()))
}
