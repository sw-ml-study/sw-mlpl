//! Step 3 of extensions-dynamic-load: the `load_extension` builtin +
//! `MLPL_EXTENSION_PATH` discovery. From MLPL, load a native extension
//! by name (resolved from the env var the run script sets) and call it.
//! In its own test binary so the env var and the process registry are
//! isolated.

use std::path::PathBuf;
use std::process::Command;

use mlpl_eval::{Environment, Value};

fn eval(env: &mut Environment, src: &str) -> Result<Value, String> {
    let toks = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&toks).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

/// Build the fixture cdylib and return the directory that holds it.
fn fixture_dir() -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let _ = Command::new(cargo)
        .args(["build", "-p", "mlpl-ext-testdylib"])
        .status();
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../target/debug")
}

#[test]
fn load_extension_discovers_by_name_and_invokes() {
    let dir = fixture_dir();
    // The env var a justfile / run script points at the built extensions.
    unsafe { std::env::set_var("MLPL_EXTENSION_PATH", &dir) };

    let mut env = Environment::new();
    // Discover by library name; the descriptor declares namespace
    // "testext", which is what load_extension returns.
    match eval(&mut env, "load_extension(\"mlpl_ext_testdylib\")").unwrap() {
        Value::Str(ns) => assert_eq!(ns, "testext"),
        other => panic!("expected namespace string, got {other:?}"),
    }
    // The dynamically loaded function is now callable from MLPL.
    match eval(&mut env, "testext:answer()").unwrap() {
        Value::Array(a) => assert_eq!(a.data()[0], 42.0),
        other => panic!("expected 42, got {other:?}"),
    }
    // A non-scalar value crosses the dlopen edge: an array goes IN.
    // sum([10, 20, 30, 40]) = 100.
    match eval(&mut env, "testext:sum([10, 20, 30, 40])").unwrap() {
        Value::Array(a) => assert_eq!(a.data()[0], 100.0),
        other => panic!("expected 100, got {other:?}"),
    }
    // A name with no matching library is a clean err Result (no crash,
    // no registration).
    match eval(&mut env, "load_extension(\"does_not_exist\")").unwrap() {
        Value::Result { ok, .. } => assert!(!ok, "missing extension should err"),
        other => panic!("expected err Result, got {other:?}"),
    }

    unsafe { std::env::remove_var("MLPL_EXTENSION_PATH") };
}
