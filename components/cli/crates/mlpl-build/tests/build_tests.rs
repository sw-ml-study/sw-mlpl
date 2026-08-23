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
    // stdout is PRISTINE: exactly the bytes, no trailing ok(2) line (the
    // wrapper suppresses the ok Result -- the write already happened).
    std::fs::write(&src_path, "write_stdout([72, 105])\n").unwrap();
    let out_path = tmp.join("prog");
    let result = run_mlpl_build(&[src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run = Command::new(&out_path).output().expect("run binary");
    assert_eq!(run.stdout, b"Hi", "stdout not pristine: {:?}", run.stdout);
    assert_eq!(run.status.code(), Some(0));
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
    // The reject is a final err: message to stderr, exit 1, stdout empty
    // (no truncated byte written).
    assert!(
        run.stdout.is_empty(),
        "stdout must be empty: {:?}",
        run.stdout
    );
    assert_eq!(run.status.code(), Some(1));
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(
        err.contains("256") && err.contains("0..=255"),
        "expected descriptive reject message on stderr, got {err:?}"
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
    // disp renders the ok Result to stdout (a bare Result is suppressed).
    std::fs::write(&src_path, "disp(file_size(\"data.bin\"))\n").unwrap();
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
    // Whole file -> ok([65, 66, 67]). disp renders the Result to stdout.
    let whole = build_run("disp(read_bytes(\"data.bin\"))\n", "whole");
    assert!(whole.starts_with("ok("), "{whole}");
    assert!(
        whole.contains("65") && whole.contains("66") && whole.contains("67"),
        "{whole}"
    );
    // offset 1, length 5 -> EOF-clamped to [66, 67] (no byte 65).
    let range = build_run("disp(read_bytes(\"data.bin\", 1, 5))\n", "range");
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
         append_bytes(\"out.bin\", [66, 67])?; read_bytes(\"out.bin\") }\ndisp(u:rt(0))\n",
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
    // The reject is a final err: message to stderr, exit 1, empty stdout.
    assert!(
        run.stdout.is_empty(),
        "stdout must be empty: {:?}",
        run.stdout
    );
    assert_eq!(run.status.code(), Some(1));
    let s = String::from_utf8_lossy(&run.stderr);
    assert!(s.contains("256"), "stderr: {s}");
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
    // to_int parse success -> ok(42). disp renders the Result to stdout
    // (a bare ok Result is suppressed by the pristine-stdout wrapper).
    assert_eq!(build_run("disp(to_int(\"42\"))\n", "toi"), "ok(42)");
    // to_int parse failure -> an err( Result the program can branch on
    // (NOT a panic -- to_int returns a CVal::Result). disp renders it.
    let bad = build_run("disp(to_int(\"xyz\"))\n", "toierr");
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
fn compiled_stdout_is_pristine_and_err_sets_exit_code() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("pristine");
    // Compile `src`, run it, return (stdout bytes, stderr string, exit code).
    let run = |src: &str, tag: &str| -> (Vec<u8>, String, Option<i32>) {
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
            out.stdout,
            String::from_utf8_lossy(&out.stderr).to_string(),
            out.status.code(),
        )
    };
    // write_stdout emits EXACTLY the bytes -- no trailing ok(N) text.
    let (out, err, code) = run("write_stdout([65, 66, 67])\n", "ws");
    assert_eq!(
        out,
        b"ABC",
        "stdout not pristine: {:?}",
        String::from_utf8_lossy(&out)
    );
    assert!(err.is_empty(), "stderr: {err}");
    assert_eq!(code, Some(0));
    // A plain (non-Result) value program still shows its result.
    let (out, _, code) = run("iota(3)\n", "val");
    assert_eq!(String::from_utf8_lossy(&out).trim(), "0 1 2");
    assert_eq!(code, Some(0));
    // A final err(...) prints its message to STDERR and exits 1 -- not
    // to stdout.
    let (out, err, code) = run("err(\"boom\")\n", "er");
    assert!(out.is_empty(), "err must not reach stdout: {:?}", out);
    assert!(err.contains("boom"), "stderr: {err}");
    assert_eq!(code, Some(1));
    // A final ok(value) is suppressed (pristine); wrap in disp to show it.
    let (out, _, _) = run("to_int(\"42\")\n", "okq");
    assert!(out.is_empty(), "ok Result must be suppressed: {:?}", out);
    let (out, _, _) = run("disp(to_int(\"42\"))\n", "okd");
    assert_eq!(String::from_utf8_lossy(&out).trim(), "ok(42)");
}

#[test]
fn read_bytes_unwrapped_flows_into_array_ops() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("readunwrap");
    // 5 bytes incl two newlines (10): "H\ni\n!" -> [72, 10, 105, 10, 33].
    std::fs::write(tmp.join("data.bin"), [72u8, 10, 105, 10, 33]).unwrap();
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
        // cwd = tmp is the compiled sandbox root for read_bytes.
        String::from_utf8_lossy(
            &Command::new(&op)
                .current_dir(&tmp)
                .output()
                .expect("run")
                .stdout,
        )
        .trim()
        .to_string()
    };
    // Via a CVal binding: b = read_bytes(...)? then compare + reduce.
    // A wc-newline-count shape: count bytes equal to 10 -> 2. (`?` is
    // valid only inside a Result-returning fn; the path is inline
    // because a compiled user-fn parameter is DenseArray-typed -- a
    // string/CVal parameter is a separate rung.)
    let count = build_run(
        "def u:nl() { b = read_bytes(\"data.bin\")? ; reduce_add(eq(b, 10)) }\nu:nl()\n",
        "nl",
    );
    assert_eq!(count, "2", "newline count");
    // Directly: reduce_add over the unwrapped read -> byte sum 230.
    let sum = build_run(
        "def u:sum() { reduce_add(read_bytes(\"data.bin\")?) }\nu:sum()\n",
        "sum",
    );
    assert_eq!(sum, "230", "byte sum");
}

#[test]
fn string_ops_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("strops");
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
    assert_eq!(build_run("str_len(\"hello\")\n", "sl"), "5");
    assert_eq!(build_run("str_concat(\"ab\", \"cd\")\n", "sc"), "abcd");
    // str_find returns a CHAR index, or -1.
    assert_eq!(build_run("str_find(\"hello\", \"ll\")\n", "sf"), "2");
    assert_eq!(build_run("str_find(\"hello\", \"z\")\n", "sfn"), "-1");
    // str_slice(s, start, len): len chars from char index start.
    assert_eq!(build_run("str_slice(\"hello\", 1, 3)\n", "ss"), "ell");
    // str_split -> StrList (displays as newline-joined). Empty sep -> chars.
    assert_eq!(build_run("str_split(\"a,b,c\", \",\")\n", "sp"), "a\nb\nc");
    assert_eq!(build_run("str_split(\"xyz\", \"\")\n", "spc"), "x\ny\nz");
}

#[test]
fn type_of_and_equal_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("typeeq");
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
    // type_of: value kind as a string (interpreter value_kind parity).
    assert_eq!(build_run("type_of([1, 2])\n", "toa"), "array");
    assert_eq!(build_run("type_of(\"hi\")\n", "tos"), "string");
    // equal: structural equality -> scalar 1/0.
    assert_eq!(build_run("equal([1, 2, 3], [1, 2, 3])\n", "eqt"), "1");
    assert_eq!(build_run("equal([1, 2], [1, 3])\n", "eqf"), "0");
    assert_eq!(build_run("equal(\"a\", \"a\")\n", "eqs"), "1");
    // The range-reader idiom: equal(type_of(v), "array").
    assert_eq!(
        build_run("equal(type_of([1, 2]), \"array\")\n", "combo"),
        "1"
    );
}

#[test]
fn take_and_floor_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("takefloor");
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
    // take(a, axis, idx): drop `axis` at index `idx` (interpreter parity)
    // -- take(iota(10), 0, 3) selects the element at index 3 -> 3.
    assert_eq!(build_run("take(iota(10), 0, 3)\n", "tk"), "3");
    // floor: elementwise. iota(5)/2 = [0,0.5,1,1.5,2] -> [0,0,1,1,2].
    assert_eq!(build_run("floor(iota(5) / 2)\n", "fl"), "0 0 1 1 2");
}

#[test]
fn tally_compiles_and_counts_major_cells() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("tally");
    std::fs::write(tmp.join("data.bin"), [65u8, 66, 67, 68, 69]).unwrap();
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
        String::from_utf8_lossy(
            &Command::new(&op)
                .current_dir(&tmp)
                .output()
                .expect("run")
                .stdout,
        )
        .trim()
        .to_string()
    };
    // tally = leading-axis length (major cells), as a scalar.
    assert_eq!(build_run("tally(iota(4))\n", "iv"), "4");
    assert_eq!(build_run("tally([10, 20, 30])\n", "lit"), "3");
    // tally over a ?-unwrapped read: number of bytes read (5).
    assert_eq!(
        build_run(
            "def u:n() { tally(read_bytes(\"data.bin\")?) }\nu:n()\n",
            "rd"
        ),
        "5"
    );
}

#[test]
fn comparisons_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("cmp");
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
    // Scalar comparisons -> 1.0 / 0.0 (interpreter parity).
    assert_eq!(build_run("gt(5, 3)\n", "gt"), "1");
    assert_eq!(build_run("lt(5, 3)\n", "lt"), "0");
    assert_eq!(build_run("eq(4, 4)\n", "eqt"), "1");
    assert_eq!(build_run("eq(4, 5)\n", "eqf"), "0");
    // Elementwise with scalar broadcasting.
    assert_eq!(build_run("eq([1, 2, 3], [1, 0, 3])\n", "eqv"), "1 0 1");
    assert_eq!(build_run("gt([1, 5, 3], 2)\n", "gtv"), "0 1 1");
    // gt/lt/eq + arithmetic compose the rest of 0/1 logic:
    // not(x) = eq(x, 0); ge(a,b) = eq(lt(a,b), 0); and = mul.
    assert_eq!(build_run("eq(lt(5, 3), 0)\n", "ge"), "1"); // 5 >= 3
}

#[test]
fn read_stdin_echoes_piped_input() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    use std::io::Write;
    use std::process::Stdio;
    let tmp = tempdir("procstdin");
    // A compiled program that echoes all of stdin back out.
    let sp = tmp.join("echo.mlpl");
    std::fs::write(&sp, "disp(read_stdin())\n").unwrap();
    let op = tmp.join("echo");
    let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
    assert!(
        r.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let mut child = Command::new(&op)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"piped input line\n")
        .unwrap();
    let out = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("piped input line"), "stdout: {stdout}");
}

#[test]
fn infix_comparison_operators_compile_and_run() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    let tmp = tempdir("infixcmp");
    let sp = tmp.join("cmp.mlpl");
    // Count how many of 5..8 satisfy each comparison against 6, summing
    // the six masks: reduce_add(iota(9) applied) -- but keep it scalar:
    // 5>6=0, 5<6=1, 6>=6=1, 6<=6=1, 6==6=1, 5!=6=1 -> 0+1+1+1+1+1 = 5.
    // Precedence check folded in: `2 + 3 > 4` is (2+3)>4 = 1, added -> 6.
    std::fs::write(
        &sp,
        "def u:score() { \"Sum of six comparison masks plus a precedence check.\" \
         (5 > 6) + (5 < 6) + (6 >= 6) + (6 <= 6) + (6 == 6) + (5 != 6) + (2 + 3 > 4) }\n\
         u:score()\n",
    )
    .unwrap();
    let op = tmp.join("cmp");
    let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
    assert!(
        r.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let run = Command::new(&op).output().expect("run binary");
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "6");
}

#[test]
fn read_stdin_chunk_counts_bytes_incrementally() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    use std::io::Write;
    use std::process::Stdio;
    let tmp = tempdir("stdinchunk");
    // A bounded-memory byte counter: loop `read_stdin_chunk(4)` (a
    // 4-byte budget that FORCES short reads over a 14-byte input),
    // accumulate `tally(chunk.bytes)`, and stop when `chunk.eof` flips
    // to 1 on the terminal empty read. Proves incremental reads, the
    // EOF terminator, record field access, and `?`-unwrapped Results
    // all compose in compiled code.
    let sp = tmp.join("count.mlpl");
    std::fs::write(
        &sp,
        "def u:count() { total = 0; more = 1; \
         while more { chunk = read_stdin_chunk(4)?; \
         total = total + tally(chunk.bytes); more = 1 - chunk.eof } \
         total }\nu:count()\n",
    )
    .unwrap();
    let op = tmp.join("count");
    let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
    assert!(
        r.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    // 14-byte payload; with a 4-byte budget the loop reads several
    // short chunks then the empty EOF chunk. The count must be exact.
    let mut child = Command::new(&op)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hello, stdin!\n")
        .unwrap();
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "counter must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.trim(), "14", "byte count (stdout: {stdout:?})");
}

#[test]
fn read_stdin_chunk_empty_input_is_immediate_eof() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    use std::process::Stdio;
    let tmp = tempdir("stdinchunk-eof");
    // Empty stdin: the very first `read_stdin_chunk` returns
    // `{bytes: [], eof: 1}`, so the loop body never accumulates and
    // the count is 0.
    let sp = tmp.join("empty.mlpl");
    std::fs::write(
        &sp,
        "def u:count() { total = 0; more = 1; \
         while more { chunk = read_stdin_chunk(8)?; \
         total = total + tally(chunk.bytes); more = 1 - chunk.eof } \
         total }\nu:count()\n",
    )
    .unwrap();
    let op = tmp.join("empty");
    let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
    assert!(
        r.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let out = Command::new(&op)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert!(out.status.success(), "empty-input counter must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.trim(), "0", "empty stdin -> 0 (stdout: {stdout:?})");
}

#[test]
fn read_stdin_chunk_rejects_bad_budget_without_consuming() {
    if !should_run() {
        eprintln!("skipping mlpl-build e2e test; set MLPL_BUILD_TESTS=1 to run");
        return;
    }
    use std::io::Write;
    use std::process::Stdio;
    let tmp = tempdir("stdinchunk-badbudget");
    // `read_stdin_chunk(0)` is an invalid budget: the `?` propagates
    // the err, so the program aborts with a non-zero exit and never
    // consumes stdin.
    let sp = tmp.join("bad.mlpl");
    std::fs::write(
        &sp,
        "def u:go() { chunk = read_stdin_chunk(0)?; print(tally(chunk.bytes)) }\nu:go()\n",
    )
    .unwrap();
    let op = tmp.join("bad");
    let r = run_mlpl_build(&[sp.to_str().unwrap(), "-o", op.to_str().unwrap()]);
    assert!(
        r.status.success(),
        "mlpl-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let mut child = Command::new(&op)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let _ = child.stdin.take().unwrap().write_all(b"unconsumed");
    let out = child.wait_with_output().expect("wait");
    assert!(
        !out.status.success(),
        "bad budget must abort (non-zero exit)"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("read_stdin_chunk") && err.contains("positive integer"),
        "expected a budget-rejection message, got: {err}"
    );
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
