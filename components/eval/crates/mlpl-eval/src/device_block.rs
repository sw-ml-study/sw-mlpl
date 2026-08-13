//! `device(...)` block-source helpers: render a block body to source
//! text and collect the array bindings it references, for a peer
//! dispatcher. Split out of `device` (per docs/code_metrics.md, split
//! by responsibility).

use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use std::collections::HashMap;

use crate::env::Environment;
use crate::env_api::*;

/// Render a block body back to newline-joined source text.
pub(crate) fn block_source(body: &[Expr]) -> String {
    body.iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The array-valued bindings whose names appear in `source`, so a peer
/// dispatcher receives exactly the arrays the block can reference.
pub(crate) fn collect_array_bindings(
    env: &Environment,
    source: &str,
) -> HashMap<String, DenseArray> {
    env.vars_iter()
        .filter(|(name, _)| source.contains(name.as_str()))
        .map(|(name, arr)| (name.clone(), arr.clone()))
        .collect()
}

/// Whether this build can dispatch through MLX (Apple aarch64): the
/// `mlx` Cargo feature plus the target OS/arch gate.
pub(crate) const fn mlx_available() -> bool {
    cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

/// Whether this build can dispatch through CUDA (Linux x86_64).
pub(crate) const fn cuda_available() -> bool {
    cfg!(all(
        feature = "cuda",
        target_os = "linux",
        target_arch = "x86_64"
    ))
}
