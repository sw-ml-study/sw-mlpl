//! Shape-driven array CONSTRUCTORS: `zeros(dims)`, `ones(dims)`, and
//! `fill(dims, value)`. Each reads a dims vector and produces a dense
//! array of that shape.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::error::RuntimeError;

pub(crate) fn constructor(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if name == "fill" {
        return fill_constructor(name, args);
    }
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let dims: Vec<usize> = args[0].data().iter().map(|&d| d as usize).collect();
    let val = if name == "zeros" { 0.0 } else { 1.0 };
    let count = dims.iter().product();
    Ok(DenseArray::new(Shape::new(dims), vec![val; count])?)
}

fn fill_constructor(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "value must be scalar".into(),
        });
    }
    let dims: Vec<usize> = args[0].data().iter().map(|&d| d as usize).collect();
    let count = dims.iter().product();
    Ok(DenseArray::new(
        Shape::new(dims),
        vec![args[1].data()[0]; count],
    )?)
}
