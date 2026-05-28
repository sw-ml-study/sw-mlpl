//! Benchmark harness for interpreter-vs-compiled MLPL.
//!
//! Every entry in [`WORKLOADS`] is an MLPL source string that the
//! interpreter (`mlpl-eval`) and the compile path (`mlpl-lower-rs`
//! -> Rust -> `mlpl-rt`) both accept. At build time, `build.rs`
//! lowers each workload to a free `fn case_<name>() -> DenseArray`
//! in `$OUT_DIR/compiled_cases.rs`; the Criterion bench harness in
//! `benches/interp_vs_compiled.rs` `include!`s that file and
//! compares the two code paths on the same input.
//!
//! Constraint: every source must stay within the lowered subset
//! (see `docs/compiling-mlpl.md` "Out of scope"). Adding `param`,
//! `grad`, `repeat`, the Model DSL, or string-named axis args here
//! will make `build.rs` fail with `LowerError::Unsupported`.

/// `(name, mlpl_source)` pairs benched by the interp-vs-compiled
/// harness. Names are used as Rust identifiers in generated code
/// (via `case_<name>`) and as Criterion group names, so they must
/// be snake_case and unique.
pub const WORKLOADS: &[(&str, &str)] = &[
    ("scalar_tight", "1 + 2 * 3 - 4"),
    (
        "small_array_arith",
        "reduce_add([1, 2, 3, 4, 5] * 10 + [0, 1, 2, 3, 4])",
    ),
    ("iota_reduce_100", "reduce_add(iota(100))"),
    (
        "reshape_reduce_100x100",
        "m = reshape(iota(10000), [100, 100]); \
         rows = reduce_add(m, 0); \
         cols = reduce_add(m, 1); \
         reduce_add(rows) + reduce_add(cols)",
    ),
    (
        "matmul_16x16",
        "a = reshape(iota(256), [16, 16]); \
         b = reshape(iota(256) + 1, [16, 16]); \
         reduce_add(matmul(a, b))",
    ),
    (
        "transpose_chain_10x10",
        "m = reshape(iota(100), [10, 10]); \
         reduce_add(transpose(m) + m)",
    ),
];
