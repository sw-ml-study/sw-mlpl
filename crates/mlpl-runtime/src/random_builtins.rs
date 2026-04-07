//! Seeded random built-ins: `random` (uniform) and `randn` (normal).

use mlpl_array::{DenseArray, Shape};

use crate::error::RuntimeError;
use crate::prng::Xorshift64;

/// Dispatch random built-ins. Returns None if not matched.
pub(crate) fn try_call(
    name: &str,
    args: Vec<DenseArray>,
) -> Option<Result<DenseArray, RuntimeError>> {
    match name {
        "random" => Some(build(name, args, |rng| rng.next_f64())),
        "randn" => Some(build(name, args, |rng| rng.next_normal())),
        "blobs" => Some(builtin_blobs(name, args)),
        _ => None,
    }
}

fn builtin_blobs(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 3,
            got: args.len(),
        });
    }
    if args[0].rank() != 0 || args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "seed and n_per_class must be scalars".into(),
        });
    }
    let seed = args[0].data()[0] as i64 as u64;
    let n_per_class = args[1].data()[0] as usize;
    // centers: Kx2 matrix OR length-2K vector.
    let centers = &args[2];
    let cdims = centers.shape().dims();
    let (k, pairs): (usize, Vec<(f64, f64)>) = match cdims.len() {
        2 if cdims[1] == 2 => {
            let k = cdims[0];
            let mut v = Vec::with_capacity(k);
            for i in 0..k {
                v.push((centers.data()[i * 2], centers.data()[i * 2 + 1]));
            }
            (k, v)
        }
        1 if cdims[0] % 2 == 0 => {
            let k = cdims[0] / 2;
            let mut v = Vec::with_capacity(k);
            for i in 0..k {
                v.push((centers.data()[i * 2], centers.data()[i * 2 + 1]));
            }
            (k, v)
        }
        _ => {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: "centers must be Kx2 matrix or length-2K vector".into(),
            });
        }
    };

    let sigma = 0.15;
    let total = k * n_per_class;
    let mut data = Vec::with_capacity(total * 3);
    let mut rng = Xorshift64::new(seed);
    for (label, &(cx, cy)) in pairs.iter().enumerate().take(k) {
        for _ in 0..n_per_class {
            let x = cx + sigma * rng.next_normal();
            let y = cy + sigma * rng.next_normal();
            data.push(x);
            data.push(y);
            data.push(label as f64);
        }
    }
    Ok(DenseArray::new(Shape::new(vec![total, 3]), data)?)
}

fn build(
    name: &str,
    args: Vec<DenseArray>,
    mut draw: impl FnMut(&mut Xorshift64) -> f64,
) -> Result<DenseArray, RuntimeError> {
    if args.len() != 2 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 2,
            got: args.len(),
        });
    }
    if args[0].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("seed must be scalar, got rank {}", args[0].rank()),
        });
    }
    let seed = args[0].data()[0] as i64 as u64;
    let dims: Vec<usize> = args[1].data().iter().map(|&d| d as usize).collect();
    let count: usize = dims.iter().product();
    let mut rng = Xorshift64::new(seed);
    let data: Vec<f64> = (0..count).map(|_| draw(&mut rng)).collect();
    Ok(DenseArray::new(Shape::new(dims), data)?)
}
