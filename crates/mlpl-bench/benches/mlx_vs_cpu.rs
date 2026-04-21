//! Saga 14 step 008: Criterion harness comparing the interpreter
//! CPU path against the interpreter+`device("mlx") { ... }` path
//! on two workloads:
//!
//! 1. `reshape_reduce_100x100` -- the same 100x100 reshape+reduce
//!    program from the step-003 / compile-to-rust harness. Dense
//!    matmul-free arithmetic; measures raw dispatch overhead plus
//!    MLX's ability to fuse the two axis reductions.
//! 2. `tiny_lm_train_step` -- one Adam step (forward + CE +
//!    backward + Adam update) on a Saga 13 Tiny-LM-shaped slice
//!    scaled down so the bench is interactive: V=60, d=16, T=8,
//!    single-head causal attention. Measures the warm-path cost
//!    of the full training loop that the end-to-end
//!    `demos/tiny_lm_mlx.mlpl` runs 200 of.
//!
//! Each group runs `interp_cpu` vs `interp_mlx`. Criterion's
//! warm-up period (3 s default) amortizes MLX's first-call compile
//! overhead; the reported times are warm-path steady-state. The
//! first iteration of the outer criterion loop is also timed
//! separately as `cold_*` so the Saga 14 plan's "cold vs warm"
//! request is satisfied.
//!
//! Triple-gated to Apple Silicon + the `mlx` feature.
//!
//! Run with `cargo bench -p mlpl-bench --features mlx`.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use std::time::Instant;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{Expr, lex, parse};

const RESHAPE_REDUCE: &str = "m = reshape(iota(10000), [100, 100]); \
                              rows = reduce_add(m, 0); \
                              cols = reduce_add(m, 1); \
                              reduce_add(rows) + reduce_add(cols)";

// One Adam step on a Saga 13 Tiny LM-shaped slice. Smaller than
// demos/tiny_lm_mlx.mlpl (which is V=280, d=32, T=32, 200 steps)
// so a single warm iteration is under a second even on CPU.
const TINY_LM_TRAIN_CPU: &str = "\
m = chain(embed(60, 16, 0), \
          residual(chain(rms_norm(16), causal_attention(16, 1, 1))), \
          rms_norm(16), \
          linear(16, 60, 2)) ; \
X = [1, 3, 5, 7, 2, 4, 6, 0] ; \
Y = [3, 5, 7, 2, 4, 6, 0, 1] ; \
adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001)";

const TINY_LM_TRAIN_MLX: &str = "\
device(\"mlx\") { \
  m = chain(embed(60, 16, 0), \
            residual(chain(rms_norm(16), causal_attention(16, 1, 1))), \
            rms_norm(16), \
            linear(16, 60, 2)) \
} ; \
X = [1, 3, 5, 7, 2, 4, 6, 0] ; \
to_device(X, \"mlx\") ; \
Y = [3, 5, 7, 2, 4, 6, 0, 1] ; \
device(\"mlx\") { \
  adam(cross_entropy(apply(m, X), Y), m, 0.001, 0.9, 0.999, 0.00000001) \
}";

fn parse_or_die(src: &str, label: &str) -> Vec<Expr> {
    let tokens = lex(src).unwrap_or_else(|e| panic!("bench lex {label}: {e}"));
    parse(&tokens).unwrap_or_else(|e| panic!("bench parse {label}: {e}"))
}

fn run(stmts: &[Expr]) {
    let mut env = Environment::new();
    let _ = eval_program(stmts, &mut env).expect("bench eval");
}

fn cold_time_once(stmts: &[Expr], label: &str) {
    // Deliberately untimed-by-criterion cold measurement: print a
    // one-shot wall-clock to stdout so the bench output includes
    // the MLX-first-call compile overhead next to the warm numbers
    // Criterion reports.
    let t0 = Instant::now();
    run(stmts);
    let dt = t0.elapsed();
    println!("  cold/{label}: {dt:?}");
}

fn bench_reshape_reduce(c: &mut Criterion) {
    let cpu_src = RESHAPE_REDUCE;
    let mlx_src = format!("device(\"mlx\") {{ {RESHAPE_REDUCE} }}");
    let cpu_stmts = parse_or_die(cpu_src, "reshape_reduce cpu");
    let mlx_stmts = parse_or_die(&mlx_src, "reshape_reduce mlx");

    println!("reshape_reduce_100x100 cold timings:");
    cold_time_once(&cpu_stmts, "cpu");
    cold_time_once(&mlx_stmts, "mlx");

    let mut group = c.benchmark_group("reshape_reduce_100x100");
    group.bench_function("interp_cpu", |b| {
        b.iter(|| {
            let mut env = Environment::new();
            black_box(eval_program(&cpu_stmts, &mut env).expect("cpu eval"));
        });
    });
    group.bench_function("interp_mlx", |b| {
        b.iter(|| {
            let mut env = Environment::new();
            black_box(eval_program(&mlx_stmts, &mut env).expect("mlx eval"));
        });
    });
    group.finish();
}

fn bench_tiny_lm_train_step(c: &mut Criterion) {
    let cpu_stmts = parse_or_die(TINY_LM_TRAIN_CPU, "tiny_lm_train cpu");
    let mlx_stmts = parse_or_die(TINY_LM_TRAIN_MLX, "tiny_lm_train mlx");

    println!("tiny_lm_train_step cold timings:");
    cold_time_once(&cpu_stmts, "cpu");
    cold_time_once(&mlx_stmts, "mlx");

    let mut group = c.benchmark_group("tiny_lm_train_step");
    // One training step does a lot of work; keep Criterion's
    // iteration count reasonable by relaxing the measurement
    // time. The default 5s is plenty for a robust mean.
    group.sample_size(30);
    group.bench_function("interp_cpu", |b| {
        b.iter(|| {
            let mut env = Environment::new();
            black_box(eval_program(&cpu_stmts, &mut env).expect("cpu eval"));
        });
    });
    group.bench_function("interp_mlx", |b| {
        b.iter(|| {
            let mut env = Environment::new();
            black_box(eval_program(&mlx_stmts, &mut env).expect("mlx eval"));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_reshape_reduce, bench_tiny_lm_train_step);
criterion_main!(benches);
