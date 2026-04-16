//! Seeded random built-ins: `random` / `randn` and the Saga 13 sampling
//! primitives `sample(logits, temperature, seed)` and `top_k(logits, k)`.

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
        "moons" | "circles" => Some(synthetic_2d(name, args)),
        "sample" => Some(builtin_sample(name, args)),
        "top_k" => Some(builtin_top_k(name, args)),
        _ => None,
    }
}

/// `sample(logits, temperature, seed)` — categorical sample from a
/// 1-D `[V]` logit vector. Returns a scalar integer token id.
///
/// Temperature `0.0` collapses to `argmax(logits)` (the most likely
/// token). Otherwise the sampling distribution is `softmax(logits /
/// temperature)`, drawn via inverse-CDF on a single `Xorshift64`
/// uniform. `-inf` logits (the masked entries left by `top_k`) are
/// supported because the max-subtraction step makes them go to zero
/// after `exp`. The same `(logits, temperature, seed)` always yields
/// the same id.
fn builtin_sample(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 3,
            got: args.len(),
        });
    }
    if args[0].rank() != 1 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("logits must be a vector, got rank {}", args[0].rank()),
        });
    }
    if args[1].rank() != 0 || args[2].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "temperature and seed must be scalars".into(),
        });
    }
    let logits = args[0].data();
    let temperature = args[1].data()[0];
    if temperature < 0.0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("temperature must be non-negative, got {temperature}"),
        });
    }
    if logits.is_empty() {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "logits must be non-empty".into(),
        });
    }
    let raw_seed = args[2].data()[0] as i64 as u64;
    Ok(DenseArray::from_scalar(
        categorical_sample(logits, temperature, raw_seed) as f64,
    ))
}

/// Pure sampling math behind `sample`. `temperature == 0.0` collapses
/// to argmax; otherwise draws via inverse-CDF on a SplitMix64-mixed
/// xorshift uniform. The mixing decorrelates adjacent small seeds so
/// single-shot sampling is well-distributed.
fn categorical_sample(logits: &[f64], temperature: f64, raw_seed: u64) -> usize {
    if temperature == 0.0 {
        let mut best = 0usize;
        let mut bv = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        return best;
    }
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|x| ((x - m) / temperature).exp())
        .collect();
    let z: f64 = exps.iter().sum();
    let mut s = raw_seed.wrapping_add(0x9E3779B97F4A7C15);
    s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
    s ^= s >> 31;
    let mut rng = Xorshift64::new(s);
    let u = rng.next_f64() * z;
    let mut acc = 0.0_f64;
    for (i, &e) in exps.iter().enumerate() {
        acc += e;
        if u < acc {
            return i;
        }
    }
    exps.len() - 1
}

/// `top_k(logits, k)` — return a `[V]` logit vector with all but the
/// top-`k` entries replaced by `-inf`. Pure (no randomness). Composed
/// with `sample` to implement top-k sampling: the `-inf` entries
/// vanish after softmax. Ties are broken by index (lower wins) so the
/// result is deterministic.
fn builtin_top_k(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
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
            reason: format!("logits must be a vector, got rank {}", args[0].rank()),
        });
    }
    if args[1].rank() != 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "k must be a scalar".into(),
        });
    }
    let logits = args[0].data();
    let k_raw = args[1].data()[0];
    if k_raw < 1.0 || k_raw.fract() != 0.0 || (k_raw as usize) > logits.len() {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("k must be an integer in 1..={}, got {k_raw}", logits.len()),
        });
    }
    let k = k_raw as usize;
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut keep = vec![false; logits.len()];
    for &i in order.iter().take(k) {
        keep[i] = true;
    }
    let out: Vec<f64> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| if keep[i] { x } else { f64::NEG_INFINITY })
        .collect();
    Ok(DenseArray::new(Shape::new(vec![logits.len()]), out)?)
}

/// Shared implementation for the `moons(seed, n, noise)` and
/// `circles(seed, n, noise)` synthetic 2D classification datasets.
/// Returns an `[n, 3]` matrix of `[x, y, label]` rows. Labels are
/// 0 / 1 in two roughly balanced groups; layout is deterministic
/// given `seed`.
///
/// - `moons`: two interleaving half-circles, the classic
///   `make_moons` shape.
/// - `circles`: a small inner ring (radius 0.5, label 0) inside a
///   larger outer ring (radius 1.0, label 1).
fn synthetic_2d(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 3 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 3,
            got: args.len(),
        });
    }
    for (i, a) in args.iter().enumerate() {
        if a.rank() != 0 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("argument {i} must be scalar, got rank {}", a.rank()),
            });
        }
    }
    let seed = args[0].data()[0] as i64 as u64;
    let n = args[1].data()[0] as usize;
    let noise = args[2].data()[0];
    let n0 = n / 2;
    let n1 = n - n0;
    let mut rng = Xorshift64::new(seed);
    let mut data = Vec::with_capacity(n * 3);
    let pi = std::f64::consts::PI;
    for k in 0..n {
        let label = if k < n0 { 0.0 } else { 1.0 };
        let local_n = if k < n0 { n0 } else { n1 };
        let local_i = if k < n0 { k } else { k - n0 };
        let denom = local_n.max(1) as f64;
        let (mut x, mut y) = if name == "moons" {
            let theta = pi * (local_i as f64) / denom;
            if label == 0.0 {
                (theta.cos(), theta.sin())
            } else {
                (1.0 - theta.cos(), 0.5 - theta.sin())
            }
        } else {
            // circles
            let theta = 2.0 * pi * (local_i as f64) / denom;
            let r = if label == 0.0 { 0.5 } else { 1.0 };
            (r * theta.cos(), r * theta.sin())
        };
        x += noise * rng.next_normal();
        y += noise * rng.next_normal();
        data.push(x);
        data.push(y);
        data.push(label);
    }
    Ok(DenseArray::new(Shape::new(vec![n, 3]), data)?)
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
        1 if cdims[0].is_multiple_of(2) => {
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
