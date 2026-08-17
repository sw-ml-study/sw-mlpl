//! Integration tests for the `mlpl-build` binary.
//!
//! Each test writes an MLPL source file to a temp dir, invokes the
//! compiled `mlpl-build` binary via `cargo run`, and asserts on
//! either the produced binary's output or on the reported error.
//!
//! Gated by `MLPL_BUILD_TESTS=1` because every test case shells out
//! to `cargo build --release` and takes several seconds. Run with:
//!
//! ```sh
//! MLPL_BUILD_TESTS=1 cargo test -p mlpl-build
//! ```

use std::path::PathBuf;
use std::process::Command;

fn should_run() -> bool {
    std::env::var("MLPL_BUILD_TESTS").is_ok()
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
    let p = base.join(format!("mlpl-build-it-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Run `mlpl-build` with the given args, using `cargo run -p` so we
/// exercise the locally-built binary.
fn run_mlpl_build(args: &[&str]) -> std::process::Output {
    let ws = workspace_root();
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "mlpl-build", "--"])
        .current_dir(&ws);
    for a in args {
        cmd.arg(a);
    }
    cmd.output().expect("run mlpl-build")
}

#[test]
fn builds_native_binary_that_prints_reduce_add() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("reduce");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "reduce_add(range(10))\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert!(run.status.success(), "produced binary exited non-zero");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "45");
}

#[test]
fn include_is_resolved_and_compiled() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("include");
    // The whole program value comes from the included file, so a
    // correct 45 proves the include was resolved and lowered.
    std::fs::write(tmp.join("answer.mlpl"), "reduce_add(range(10))\n").unwrap();
    std::fs::write(tmp.join("prog.mlpl"), "include \"answer.mlpl\"\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[
        tmp.join("prog.mlpl").to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert!(run.status.success(), "produced binary exited non-zero");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "45");
}

#[test]
fn user_function_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("userfn");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(
        &src_path,
        "def u:add3(a, b, c) { a + b + c }\nu:add3(10, 20, 15)\n",
    )
    .unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n--- stderr ---\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert!(run.status.success(), "produced binary exited non-zero");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "45");
}

#[test]
fn if_expression_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("ifexpr");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(
        &src_path,
        "def u:pick(c) { if c { 42 } else { 7 } }\nu:pick(1)\n",
    )
    .unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "42");
}

#[test]
fn while_loop_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("while");
    let src_path = tmp.join("prog.mlpl");
    // Countdown accumulator: 5+4+3+2+1 = 15. Mutable acc + param n.
    std::fs::write(
        &src_path,
        "def u:sumdown(n) { acc = 0; while n { acc = acc + n; n = n - 1 } acc }\nu:sumdown(5)\n",
    )
    .unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "15");
}

#[test]
fn record_field_access_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("record");
    let src_path = tmp.join("prog.mlpl");
    // Build a record, read a numeric field back: {X: 42, Y: 7}.X -> 42.
    std::fs::write(&src_path, "r = {X: 42, Y: 7}\nr.X\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "42");
}

#[test]
fn result_propagation_and_field_access_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("result");
    let src_path = tmp.join("prog.mlpl");
    // u:fit returns ok(record); u:run unwraps with `?` then reads a
    // field. fit(5) = ok({slope: 6, ...}); run unwraps -> {slope: 6};
    // f.slope = 6.
    std::fs::write(
        &src_path,
        "def u:fit(n) { ok({slope: n + 1, intercept: n - 1}) }\n\
         def u:run(n) { f = u:fit(n)?; f.slope }\n\
         u:run(5)\n",
    )
    .unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .output()
        .expect("run produced binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "6");
}

#[test]
fn bit_ops_compile_and_match_golden_values() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    // One compiled program per golden case (interpreter parity):
    // band(12,10)=8, and the from_bits(bits(...)) round-trip = 165.
    for (src, expected) in [
        ("band(12, 10)\n", "8"),
        ("from_bits(bits(165, 8))\n", "165"),
        ("shl(15, 4, 8)\n", "240"),
    ] {
        let tmp = tempdir("bits");
        let src_path = tmp.join("prog.mlpl");
        std::fs::write(&src_path, src).unwrap();
        let out_path = tmp.join("prog");
        let result =
            run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
        assert!(
            result.status.success(),
            "mlpl-build failed for {src:?}:\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let run = Command::new(&out_path)
            .output()
            .expect("run produced binary");
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            expected,
            "wrong result for {src:?}"
        );
    }
}

#[test]
fn write_stdout_writes_valid_bytes_and_returns_ok() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("wstdout-ok");
    let src_path = tmp.join("prog.mlpl");
    // bytes 72,105 = "Hi"; write_stdout writes them and returns ok(2).
    // main prints the result (ok(2)) after the bytes -> "Hiok(2)".
    std::fs::write(&src_path, "write_stdout([72, 105])\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path).output().expect("run binary");
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(out.contains("Hi"), "expected written bytes 'Hi' in {out:?}");
    assert!(out.contains("ok("), "expected ok Result in {out:?}");
}

#[test]
fn write_stdout_rejects_out_of_range_byte_no_truncation() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("wstdout-reject");
    let src_path = tmp.join("prog.mlpl");
    // 256 is out of 0..=255: the compiled writer REJECTS (interpreter
    // parity) instead of truncating to byte 0, and returns err.
    std::fs::write(&src_path, "write_stdout([256])\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path).output().expect("run binary");
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(out.starts_with("err("), "expected err Result, got {out:?}");
    assert!(
        out.contains("256") && out.contains("0..=255"),
        "expected descriptive reject message, got {out:?}"
    );
    // No truncated byte should precede the err message.
    assert!(
        !run.stdout.contains(&0u8),
        "no null byte should be written: {out:?}"
    );
}

#[test]
fn file_size_compiles_and_reads_metadata() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("filesize");
    std::fs::write(tmp.join("data.bin"), [65u8, 66, 67]).unwrap();
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "file_size(\"data.bin\")\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    // Run with cwd = the temp dir, which is the compiled sandbox root.
    let run = Command::new(&out_path)
        .current_dir(&tmp)
        .output()
        .expect("run binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "ok(3)");
}

#[test]
fn read_bytes_whole_and_range_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("readbytes");
    std::fs::write(tmp.join("data.bin"), [65u8, 66, 67]).unwrap();
    let build_run = |src: &str, tag: &str| -> String {
        let sp = tmp.join(format!("{tag}.mlpl"));
        std::fs::write(&sp, src).unwrap();
        let op = tmp.join(tag);
        let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
        assert!(
            r.status.success(),
            "mlpl-build failed for {src:?}:\n{}",
            String::from_utf8_lossy(&r.stderr)
        );
        let run = Command::new(&op).current_dir(&tmp).output().expect("run");
        String::from_utf8_lossy(&run.stdout).trim().to_string()
    };
    // Whole file -> ok([65, 66, 67]).
    let whole = build_run("read_bytes(\"data.bin\")\n", "whole");
    assert!(whole.starts_with("ok("), "{whole}");
    assert!(
        whole.contains("65") && whole.contains("66") && whole.contains("67"),
        "{whole}"
    );
    // offset 1, length 5 -> EOF-clamped to [66, 67] (no byte 65).
    let range = build_run("read_bytes(\"data.bin\", 1, 5)\n", "range");
    assert!(range.starts_with("ok("), "{range}");
    assert!(
        range.contains("66") && range.contains("67") && !range.contains("65"),
        "{range}"
    );
}

#[test]
fn write_append_read_bytes_roundtrip_compiles_and_runs() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("writebytes");
    let src_path = tmp.join("prog.mlpl");
    // write [65], append [66,67], read back -> ok([65, 66, 67]).
    std::fs::write(
        &src_path,
        "def u:rt(x) { write_bytes(\"out.bin\", [65])?; \
         append_bytes(\"out.bin\", [66, 67])?; read_bytes(\"out.bin\") }\nu:rt(0)\n",
    )
    .unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .current_dir(&tmp)
        .output()
        .expect("run binary");
    let s = String::from_utf8_lossy(&run.stdout);
    assert!(s.starts_with("ok("), "{s}");
    assert!(
        s.contains("65") && s.contains("66") && s.contains("67"),
        "{s}"
    );
    // The file really holds the written + appended bytes.
    assert_eq!(
        std::fs::read(tmp.join("out.bin")).unwrap(),
        vec![65u8, 66, 67]
    );
}

#[test]
fn write_bytes_rejects_out_of_range_and_writes_nothing() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("writebytes-reject");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "write_bytes(\"out.bin\", [256])\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .current_dir(&tmp)
        .output()
        .expect("run binary");
    let s = String::from_utf8_lossy(&run.stdout);
    assert!(s.starts_with("err("), "{s}");
    assert!(s.contains("256"), "{s}");
    // Validation fails before any write, so no file is created.
    assert!(
        !tmp.join("out.bin").exists(),
        "reject must not create the file"
    );
}

#[test]
fn text_conversions_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("textops");
    let build_run = |src: &str, tag: &str| -> String {
        let sp = tmp.join(format!("{tag}.mlpl"));
        std::fs::write(&sp, src).unwrap();
        let op = tmp.join(tag);
        let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
        assert!(
            r.status.success(),
            "mlpl-build failed for {src:?}:\n{}",
            String::from_utf8_lossy(&r.stderr)
        );
        String::from_utf8_lossy(&Command::new(&op).output().expect("run").stdout)
            .trim()
            .to_string()
    };
    // tokenize_bytes exposes the raw UTF-8 byte cells: "Hi" -> [72, 105].
    let cells = build_run("tokenize_bytes(\"Hi\")\n", "cells");
    assert!(
        cells.contains("72") && cells.contains("105"),
        "tokenize_bytes cells: {cells}"
    );
    // str -> bytes -> str round-trip.
    assert_eq!(
        build_run("decode_bytes(tokenize_bytes(\"Hi\"))\n", "rt"),
        "Hi"
    );
    // to_int parse success -> ok(42).
    assert_eq!(build_run("to_int(\"42\")\n", "toi"), "ok(42)");
    // to_int parse failure -> an err( Result the program can branch on
    // (NOT a panic -- to_int returns a CVal::Result).
    let bad = build_run("to_int(\"xyz\")\n", "toierr");
    assert!(
        bad.starts_with("err(") && bad.contains("xyz"),
        "to_int err branch: {bad}"
    );
}

#[test]
fn print_and_eprint_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("procprint");
    // Compile `src`, run the binary, and return (stdout, stderr).
    let build_run = |src: &str, tag: &str| -> (String, String) {
        let sp = tmp.join(format!("{tag}.mlpl"));
        std::fs::write(&sp, src).unwrap();
        let op = tmp.join(tag);
        let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
        assert!(
            r.status.success(),
            "mlpl-build failed for {src:?}:\n{}",
            String::from_utf8_lossy(&r.stderr)
        );
        let out = Command::new(&op).output().expect("run");
        (
            String::from_utf8_lossy(&out.stdout).to_string(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        )
    };
    // print writes to stdout AND returns its argument: the program's
    // result is print's return value, which the generated main echoes.
    // So "kept" appears twice -- once from print's side effect, once
    // from main printing the returned value. That proves both.
    let (out, err) = build_run("print(\"kept\")\n", "p");
    assert_eq!(
        out.trim().lines().collect::<Vec<_>>(),
        vec!["kept", "kept"],
        "print writes then returns its argument: {out}"
    );
    assert!(err.is_empty(), "print must not touch stderr: {err}");
    // eprint writes to STDERR; the distinct final result goes to stdout.
    let (out, err) = build_run("eprint(\"to-stderr\")\ndisp(\"to-stdout\")\n", "e");
    assert!(err.contains("to-stderr"), "stderr: {err}");
    assert!(
        out.contains("to-stdout") && !out.contains("to-stderr"),
        "eprint must not reach stdout; stdout: {out}"
    );
}

#[test]
fn exit_sets_process_status_code() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("procexit");
    // Compile `src`, run the binary, and return its process exit code.
    let exit_code = |src: &str, tag: &str| -> Option<i32> {
        let sp = tmp.join(format!("{tag}.mlpl"));
        std::fs::write(&sp, src).unwrap();
        let op = tmp.join(tag);
        let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
        assert!(
            r.status.success(),
            "mlpl-build failed for {src:?}:\n{}",
            String::from_utf8_lossy(&r.stderr)
        );
        Command::new(&op).status().expect("run").code()
    };
    // exit(code) ends the process with that status.
    assert_eq!(exit_code("exit(3)\n", "e3"), Some(3));
    // A normal program exits 0.
    assert_eq!(exit_code("iota(3)\n", "ok0"), Some(0));
}

#[test]
fn decode_bytes_loud_rejects_out_of_range_cell() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("decodebytes-reject");
    let src_path = tmp.join("prog.mlpl");
    // 256 is out of 0..=255. decode_bytes returns a String (not a
    // Result), so this is a HARD error -- the compiled binary must
    // FAIL loudly (non-zero exit), never truncate 256 to a byte.
    // Interpreter parity: mlpl-eval decode_bytes -> array_to_bytes
    // raises EvalError, aborting the program.
    std::fs::write(&src_path, "decode_bytes([256])\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path)
        .current_dir(&tmp)
        .output()
        .expect("run binary");
    assert!(
        !run.status.success(),
        "decode_bytes([256]) must abort, not exit 0"
    );
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(
        err.contains("decode_bytes") && err.contains("256"),
        "expected a loud decode_bytes reject naming 256, got: {err}"
    );
}

#[test]
fn parse_error_reports_source_location_not_rustc_cascade() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("parse-err");
    let src_path = tmp.join("bad.mlpl");
    // `@` is not a valid MLPL character -- the eager lex check
    // should catch this before we ever shell to cargo.
    std::fs::write(&src_path, "1 @ 2\n").unwrap();
    let out_path = tmp.join("bad");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(!result.status.success(), "expected failure");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("mlpl-build:") && stderr.contains("bad.mlpl"),
        "error should mention mlpl-build and source path, got:\n{stderr}"
    );
}

#[test]
fn wasm_target_produces_wasm_output() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    // Skip if the wasm target isn't installed on this machine --
    // we don't want the test suite to fail on a clean dev env.
    let check = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output();
    match check {
        Ok(out) if String::from_utf8_lossy(&out.stdout).contains("wasm32-unknown-unknown") => {}
        _ => {
            eprintln!("skipping wasm test: wasm32-unknown-unknown target not installed");
            return;
        }
    }

    let tmp = tempdir("wasm");
    let src_path = tmp.join("prog.mlpl");
    std::fs::write(&src_path, "reduce_add(range(5))\n").unwrap();
    let out_path = tmp.join("prog.wasm");
    let result = run_mlpl_build(&[
        src_path.to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
        "--target",
        "wasm32-unknown-unknown",
    ]);
    assert!(
        result.status.success(),
        "wasm mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    // The WASM magic number is 0x00 0x61 0x73 0x6d ("\0asm").
    let bytes = std::fs::read(&out_path).expect("read wasm output");
    assert!(
        bytes.starts_with(&[0x00, 0x61, 0x73, 0x6d]),
        "output does not start with WASM magic bytes"
    );
}
