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
        _ => None,
    }
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
