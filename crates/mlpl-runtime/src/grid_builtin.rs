//! `grid` built-in: build a 2D grid of evaluation points.

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;

/// `grid(bounds, n)` returns an `(n*n) x 2` matrix of `(x, y)` points
/// covering `[xmin, xmax] x [ymin, ymax]`. `bounds` is the length-4
/// vector `[xmin, xmax, ymin, ymax]`.
pub fn builtin_grid(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    let bounds = args[0].data();
    if args[0].rank() != 1 || bounds.len() != 4 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "bounds must be a length-4 vector [xmin, xmax, ymin, ymax]".into(),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "n must be a scalar".into(),
        });
    }
    let n = args[1].data()[0] as usize;
    if n == 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "n must be > 0".into(),
        });
    }
    let (xmin, xmax, ymin, ymax) = (bounds[0], bounds[1], bounds[2], bounds[3]);
    let denom = if n == 1 { 1.0 } else { (n - 1) as f64 };
    let mut data = Vec::with_capacity(n * n * 2);
    for i in 0..n {
        let y = ymin + (ymax - ymin) * (i as f64) / denom;
        for j in 0..n {
            let x = xmin + (xmax - xmin) * (j as f64) / denom;
            data.push(x);
            data.push(y);
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n * n, 2]), data)?)
}
