//! B3 dynamic-loader: `load_c_extension` opens the cdylib fixture,
//! registers it, and reports its namespace; a bad path is a clean
//! error, not a crash.

use std::path::{Path, PathBuf};
use std::process::Command;

use mlpl_extension_abi::{ExtValue, call_contained};
use mlpl_extension_loader::load_c_extension;

/// Build the fixture cdylib and return its shared-library path
/// (repo-root target, platform prefix/extension aware).
fn fixture_dylib() -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let _ = Command::new(cargo)
        .args(["build", "-p", "mlpl-ext-testdylib"])
        .status();
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../target/debug");
    let file = format!(
        "{}mlpl_ext_testdylib.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    );
    target.join(file)
}

#[test]
fn loads_registers_and_reports_the_namespace() {
    let name = unsafe { load_c_extension(&fixture_dylib()) }.expect("load");
    assert_eq!(name, "testext");

    // The dynamically loaded function dispatches like a static one.
    let f = mlpl_extension_registry::lookup("testext:answer").expect("registered");
    assert_eq!(
        call_contained(&f.func, &[]).expect("invoke"),
        ExtValue::F64(42.0)
    );

    // A missing library is a clear error, not a panic.
    let bad = unsafe { load_c_extension(Path::new("/no/such/extension.dylib")) };
    assert!(bad.is_err(), "bad path should error");
}
