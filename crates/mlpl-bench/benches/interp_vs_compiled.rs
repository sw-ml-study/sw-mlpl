//! Criterion harness: for every `mlpl_bench::WORKLOADS` entry,
//! bench the interpreter path against the compiled path on the
//! same input. Parsing happens once up front (outside the
//! measurement) because the compiled path's parsing happens at
//! Rust compile time and shouldn't be double-counted.
//!
//! Run with `cargo bench -p mlpl-bench`. Criterion writes HTML
//! reports to `target/criterion/`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use mlpl_bench::WORKLOADS;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{Expr, lex, parse};

#[allow(clippy::type_complexity)]
mod compiled {
    include!(concat!(env!("OUT_DIR"), "/compiled_cases.rs"));
}
use compiled::COMPILED;

fn bench_all(c: &mut Criterion) {
    assert_eq!(
        WORKLOADS.len(),
        COMPILED.len(),
        "src/lib.rs WORKLOADS and build.rs list have diverged"
    );

    for (name, src) in WORKLOADS {
        let tokens = lex(src).expect("bench lex");
        let stmts: Vec<Expr> = parse(&tokens).expect("bench parse");
        let compiled_fn = COMPILED
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| *f)
            .unwrap_or_else(|| panic!("no compiled fn for {name}"));

        let mut group = c.benchmark_group(*name);
        group.bench_function("interp", |b| {
            b.iter(|| {
                let mut env = Environment::new();
                black_box(eval_program(&stmts, &mut env).expect("interp eval"));
            });
        });
        group.bench_function("compiled", |b| {
            b.iter(|| {
                black_box(compiled_fn());
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
