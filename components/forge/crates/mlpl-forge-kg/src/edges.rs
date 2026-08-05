//! Edge-array validation + the adjacency views the ops share.

use mlpl_array::DenseArray;
use mlpl_runtime_core::RuntimeError;

/// One parsed edge list: `(src, rel, dst)` triples as integers.
pub(crate) struct Edges(pub(crate) Vec<(i64, i64, i64)>);

/// Validate an `[E, 3]` edge array into integer triples.
pub(crate) fn parse(name: &str, a: &DenseArray) -> Result<Edges, RuntimeError> {
    let dims = a.shape().dims();
    if dims.len() != 2 || dims[1] != 3 {
        return Err(RuntimeError::InvalidArgument {
            func: name.into(),
            reason: format!("edges must be [E, 3] (src, rel, dst) rows, got shape {dims:?}"),
        });
    }
    let mut out = Vec::with_capacity(dims[0]);
    for row in a.data().chunks_exact(3) {
        out.push((row[0] as i64, row[1] as i64, row[2] as i64));
    }
    Ok(Edges(out))
}

impl Edges {
    /// `dst` ids one hop from `node`, optionally restricted to one
    /// relation; sorted, deduplicated.
    pub(crate) fn neighbors(&self, node: i64, rel: Option<i64>) -> Vec<i64> {
        let mut out: Vec<i64> = self
            .0
            .iter()
            .filter(|(s, r, _)| *s == node && rel.is_none_or(|want| *r == want))
            .map(|(_, _, d)| *d)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether SOME edge connects `a -> b` (any relation).
    pub(crate) fn connects(&self, a: i64, b: i64) -> bool {
        self.0.iter().any(|(s, _, d)| *s == a && *d == b)
    }
}
