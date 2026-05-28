//! Run a curated set of MLPL expressions through both the
//! interpreter (`mlpl-eval::eval_program`) and the compile path
//! (`mlpl-lower-rs::lower` + rustc + execute), and assert the
//! numeric outputs match bit-for-bit (within f64 epsilon) for
//! deterministic ops.
//!
//! The parity harness amortizes rustc cost across cases by
//! generating one big Rust program with one `fn test_N()` per
//! case, compiling it once, running it once, and parsing the
//! `TESTNAME=<number>` lines from stdout.
//!
//! Gated by `MLPL_PARITY_TESTS=1` so day-to-day `cargo test`
//! runs skip the rustc invocation:
//!
//! ```sh
//! MLPL_PARITY_TESTS=1 cargo test -p mlpl-parity-tests
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use mlpl_eval::{Environment, eval_program};
use mlpl_lower_rs::lower;
use mlpl_parser::{lex, parse};

/// (name, MLPL source) pairs. Every source must produce a scalar
/// or a single-element array whose `data()[0]` is the comparison
/// point. Names become Rust fn identifiers, so they must be
/// snake_case and unique.
const PARITY_CASES: &[(&str, &str)] = &[
    ("scalar_add", "1 + 2"),
    ("scalar_precedence", "2 + 3 * 4 - 1"),
    ("int_div_truncates", "10 / 3"),
    ("neg_of_iota_sum", "reduce_add(-iota(5))"),
    ("iota_reduce_flat", "reduce_add(iota(10))"),
    (
        "iota_reduce_axis",
        "reduce_add(reshape(iota(6), [2, 3]), 1)",
    ),
    (
        "transpose_roundtrip",
        "reduce_add(transpose(reshape(iota(6), [2, 3])))",
    ),
    (
        "labeled_matmul",
        "a : [seq, d] = reshape(iota(6) + 1, [2, 3]); \
         b : [d, h] = reshape([1, 0, 0, 1, 1, 1], [3, 2]); \
         reduce_add(matmul(a, b))",
    ),
    (
        "variable_reuse",
        "x = iota(8); y = x + 1; reduce_add(y * y) - reduce_add(x * x)",
    ),
];

fn should_run() -> bool {
    std::env::var("MLPL_PARITY_TESTS").is_ok()
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn tempdir(tag: &str) -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let p = base.join(format!("mlpl-parity-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn interpreter_scalar(src: &str) -> f64 {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let arr = eval_program(&stmts, &mut env).expect("eval");
    arr.data()[0]
}

fn lower_body(src: &str) -> String {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    lower(&stmts).expect("lower").to_string()
}

#[test]
fn interpreter_and_compiler_agree_on_all_parity_cases() {
    if !should_run() {
        eprintln!("skipping parity test suite; set MLPL_PARITY_TESTS=1 to run");
        return;
    }

    // 1. Run interpreter once per case.
    let interpreter_values: Vec<f64> = PARITY_CASES
        .iter()
        .map(|(_, src)| interpreter_scalar(src))
        .collect();

    // 2. Codegen one Rust program containing every case as a
    //    separate function + a main that prints `name=value` per line.
    let ws = workspace_root();
    let tmp = tempdir("suite");
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    let cargo_toml = format!(
        "[package]\n\
         name = \"mlpl_parity_run\"\n\
         edition = \"2024\"\n\
         version = \"0.0.0\"\n\
         \n\
         [dependencies]\n\
         mlpl-rt = {{ path = \"{}/crates/mlpl-rt\" }}\n",
        ws.display()
    );
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml).unwrap();

    let mut rust_src = String::new();
    rust_src.push_str("use mlpl_rt::DenseArray;\n\n");
    for (name, src) in PARITY_CASES {
        let body = lower_body(src);
        rust_src.push_str(&format!(
            "fn case_{name}() -> DenseArray {{\n    {body}\n}}\n\n"
        ));
    }
    rust_src.push_str("fn main() {\n");
    for (name, _) in PARITY_CASES {
        rust_src.push_str(&format!(
            "    println!(\"{name}={{}}\", case_{name}().data()[0]);\n"
        ));
    }
    rust_src.push_str("}\n");
    std::fs::write(tmp.join("src/main.rs"), rust_src).unwrap();

    let build = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(&tmp)
        .output()
        .expect("cargo build");
    assert!(
        build.status.success(),
        "parity suite cargo build failed:\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(tmp.join("target/release/mlpl_parity_run"))
        .output()
        .expect("run parity binary");
    assert!(run.status.success(), "parity binary exited non-zero");
    let stdout = String::from_utf8(run.stdout).expect("utf8");

    // 3. Parse `name=value` lines and compare against interpreter.
    let mut found = std::collections::HashMap::new();
    for line in stdout.lines() {
        let (name, value) = line.split_once('=').expect("name=value line");
        let v: f64 = value.parse().expect("f64 parse");
        found.insert(name.to_string(), v);
    }
    for (i, (name, src)) in PARITY_CASES.iter().enumerate() {
        let got = *found
            .get(*name)
            .unwrap_or_else(|| panic!("missing compiled output for {name}"));
        let expected = interpreter_values[i];
        assert!(
            (got - expected).abs() < 1e-9,
            "{name}: interpreter={expected}, compiled={got}\n  src: {src}"
        );
    }
}

/// A deliberately heavier workload: build a medium matrix, run
/// several transpose+reduce chains, and report the runtime of the
/// compiled binary vs the interpreter on the same source. Records
/// the speedup ratio to stderr for the release step's saga summary.
/// Not a correctness gate -- just a data point.
#[test]
fn compiled_speedup_measurement() {
    if !should_run() {
        eprintln!("skipping speedup measurement; set MLPL_PARITY_TESTS=1 to run");
        return;
    }
    // Source: build a 100x100 matrix via iota+reshape, reduce along
    // both axes a few times. Deterministic, integer-valued, fast.
    let src = "m = reshape(iota(10000), [100, 100]); \
               rows = reduce_add(m, 0); \
               cols = reduce_add(m, 1); \
               reduce_add(rows) + reduce_add(cols)";

    // Interpreter timing: run 5x, take median.
    let mut interp_nanos: Vec<u128> = (0..5)
        .map(|_| {
            let start = Instant::now();
            let _ = interpreter_scalar(src);
            start.elapsed().as_nanos()
        })
        .collect();
    interp_nanos.sort_unstable();
    let interp_median = interp_nanos[interp_nanos.len() / 2];

    // Compiled timing: build once, run 5x inside the binary.
    let ws = workspace_root();
    let tmp = tempdir("speedup");
    std::fs::create_dir_all(tmp.join("src")).unwrap();
    let cargo_toml = format!(
        "[package]\n\
         name = \"mlpl_speedup_bin\"\n\
         edition = \"2024\"\n\
         version = \"0.0.0\"\n\
         \n\
         [dependencies]\n\
         mlpl-rt = {{ path = \"{}/crates/mlpl-rt\" }}\n",
        ws.display()
    );
    std::fs::write(tmp.join("Cargo.toml"), cargo_toml).unwrap();
    let body = lower_body(src);
    let main_rs = format!(
        "use std::time::Instant;\n\
         use mlpl_rt::DenseArray;\n\
         \n\
         fn case() -> DenseArray {{\n    {body}\n}}\n\
         \n\
         fn main() {{\n\
             let mut samples = Vec::with_capacity(5);\n\
             for _ in 0..5 {{\n\
                 let start = Instant::now();\n\
                 let _ = std::hint::black_box(case());\n\
                 samples.push(start.elapsed().as_nanos());\n\
             }}\n\
             samples.sort_unstable();\n\
             println!(\"{{}}\", samples[samples.len() / 2]);\n\
         }}\n"
    );
    std::fs::write(tmp.join("src/main.rs"), main_rs).unwrap();
    let build = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(&tmp)
        .output()
        .expect("cargo build");
    assert!(build.status.success(), "speedup cargo build failed");
    let run = Command::new(tmp.join("target/release/mlpl_speedup_bin"))
        .output()
        .expect("run speedup binary");
    assert!(run.status.success(), "speedup binary exited non-zero");
    let compiled_median: u128 = String::from_utf8(run.stdout)
        .expect("utf8")
        .trim()
        .parse()
        .expect("u128 parse");

    let ratio = interp_median as f64 / compiled_median as f64;
    eprintln!(
        "speedup (workspace reshape-iota-reduce-100x100): \
         interpreter={interp_median}ns, compiled={compiled_median}ns, ratio={ratio:.2}x"
    );
    // Don't assert on the ratio -- workloads vary on different
    // machines. Just require the compiled run is not grossly
    // slower (within 3x of the interpreter is fine as a sanity
    // check; realistic hardware should see compiled win).
    assert!(
        compiled_median < interp_median * 3,
        "compiled runtime {compiled_median}ns dwarfs interpreter {interp_median}ns"
    );
}
