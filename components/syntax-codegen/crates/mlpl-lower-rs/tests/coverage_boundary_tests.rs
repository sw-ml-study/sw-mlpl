//! Compiler-coverage boundary gate. The compile-to-Rust path lowers a
//! defined SUBSET of MLPL; the rest is interpreter-only (by design or
//! not-yet-lowered). This test pins a representative set of
//! interpreter-only builtins and asserts the compiler does NOT lower
//! them -- so the boundary (docs/compiler-coverage.md) stays visible
//! and honest. When the compiler gains one of these (e.g. `list_len`
//! via the du saga), the test fails: remove it here and update the doc.

use mlpl_lower_rs::supported_builtin_names;

/// Builtins the interpreter has but the compiler does not lower today,
/// grouped by reason. Not exhaustive -- a representative gate; growing
/// it strengthens the check. See docs/compiler-coverage.md.
const INTERPRETER_ONLY: &[&str] = &[
    // Visualization -- no render target in a headless binary.
    "svg",
    "dataflow",
    "hist",
    "scatter_labeled",
    "loss_curve",
    "confusion_matrix",
    "boundary_2d",
    // ML / autograd / training -- tape-based; not lowered.
    "grad",
    "adam",
    "momentum_sgd",
    "chain",
    "linear",
    "apply",
    "cross_entropy",
    "softmax",
    // Array ops not yet lowered.
    "gather_rows",
    "compress",
    "grade_up",
    "grade_down",
    "rotate",
    "concat",
    "flatten",
    // StrList / du -- queued.
    "list_len",
    "list_get",
    "fs_walk",
    // Streaming filesystem -- interpreter-only.
    "scan_length_prefixed",
    "read_bytes_packed",
    // Codecs / reflection / engram / ports.
    "to_json",
    "from_json",
    "to_toml",
    "from_toml",
    "engram_hash",
];

#[test]
fn interpreter_only_builtins_are_not_compiled() {
    let compiled: std::collections::HashSet<&str> = supported_builtin_names().into_iter().collect();
    let leaked: Vec<&str> = INTERPRETER_ONLY
        .iter()
        .copied()
        .filter(|n| compiled.contains(n))
        .collect();
    assert!(
        leaked.is_empty(),
        "these are listed INTERPRETER_ONLY but the compiler now lowers them -- remove them \
         from INTERPRETER_ONLY and update docs/compiler-coverage.md: {leaked:?}"
    );
}

#[test]
fn the_compiler_surface_is_non_trivial() {
    // A sanity floor so an accidental empty registry is caught.
    assert!(
        supported_builtin_names().len() >= 40,
        "compiler builtin surface unexpectedly small"
    );
}
