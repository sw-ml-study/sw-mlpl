//! Seeded path sampling + the entity-disjoint split core.

use mlpl_array::{DenseArray, Shape};
use mlpl_runtime_core::{RuntimeError, Xorshift64};

use crate::edges;
use crate::ops::arity;

/// `kg_paths(edges, hops, n, seed)` -- `[n, hops+1]` valid paths
/// sampled by seeded random walk (uniform start edge, uniform
/// outgoing edge at each node; dead ends restart the walk).
pub(crate) fn kg_paths(name: &str, args: &[DenseArray]) -> Result<DenseArray, RuntimeError> {
    if args.len() != 4 {
        return Err(arity(name, 4, args.len()));
    }
    let e = edges::parse(name, &args[0])?;
    let (hops, n) = (args[1].data()[0] as usize, args[2].data()[0] as usize);
    let mut rng = Xorshift64::new(args[3].data()[0] as u64);
    if e.0.is_empty() || hops == 0 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: "need a non-empty edge list and hops >= 1".into(),
        });
    }
    let mut out = Vec::with_capacity(n * (hops + 1));
    let mut attempts = 0usize;
    let mut done = 0usize;
    while done < n {
        attempts += 1;
        if attempts > n * 200 {
            return Err(RuntimeError::InvalidArgument {
                func: name.into(),
                reason: format!("could not sample {n} paths of {hops} hops (too many dead ends)"),
            });
        }
        if let Some(path) = walk(&e, hops, &mut rng) {
            out.extend(path.into_iter().map(|v| v as f64));
            done += 1;
        }
    }
    Ok(DenseArray::new(Shape::new(vec![n, hops + 1]), out)?)
}

/// One random walk of `hops` edges; `None` on a dead end.
fn walk(e: &edges::Edges, hops: usize, rng: &mut Xorshift64) -> Option<Vec<i64>> {
    let start = e.0[(rng.next_u64() as usize) % e.0.len()].0;
    let mut path = vec![start];
    let mut cur = start;
    for _ in 0..hops {
        let outs: Vec<i64> =
            e.0.iter()
                .filter(|(s, _, _)| *s == cur)
                .map(|(_, _, d)| *d)
                .collect();
        if outs.is_empty() {
            return None;
        }
        cur = outs[(rng.next_u64() as usize) % outs.len()];
        path.push(cur);
    }
    Some(path)
}

/// Entity-disjoint split core: entities are shuffled by seed; the
/// first `frac` become the train set. An edge whose endpoints are
/// BOTH train entities goes to `train`; every other edge goes to
/// `eval` -- so eval paths must visit entities training never saw.
pub fn split_edges(
    name: &str,
    edges_arr: &DenseArray,
    frac: f64,
    seed: u64,
) -> Result<(DenseArray, DenseArray), RuntimeError> {
    let e = edges::parse(name, edges_arr)?;
    let mut ents: Vec<i64> = e.0.iter().flat_map(|(s, _, d)| [*s, *d]).collect();
    ents.sort_unstable();
    ents.dedup();
    let mut rng = Xorshift64::new(seed);
    for i in (1..ents.len()).rev() {
        ents.swap(i, (rng.next_u64() as usize) % (i + 1));
    }
    let cut = ((ents.len() as f64) * frac).round() as usize;
    let train_set: std::collections::HashSet<i64> =
        ents[..cut.min(ents.len())].iter().copied().collect();
    let (tr, ev): (Vec<_>, Vec<_>) =
        e.0.iter()
            .partition(|(s, _, d)| train_set.contains(s) && train_set.contains(d));
    Ok((crate::ops::edges_array(&tr)?, crate::ops::edges_array(&ev)?))
}
