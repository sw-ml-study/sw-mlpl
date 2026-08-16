//! B3 loader-gate: `dlopen` the cdylib fixture, register it through the
//! C descriptor ABI, and invoke its function -- proving the dynamic
//! extension-loading round-trip works in CI.

use std::path::PathBuf;
use std::process::Command;

use libloading::{Library, Symbol};
use mlpl_ext_testdylib::ANSWER;
use mlpl_extension_abi::{ExtValue, call_contained};
use mlpl_extension_cabi::{ExtensionDescriptorV1, register_c_extension};

/// The shared library artifact in the repo-root `target/debug`, with
/// the platform's prefix/extension (`libmlpl_ext_testdylib.dylib` /
/// `.so`).
fn dylib_path() -> PathBuf {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../target/debug");
    let file = format!(
        "{}mlpl_ext_testdylib.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    );
    target.join(file)
}

#[test]
fn dlopen_registers_and_invokes_the_cdylib_extension() {
    // Ensure the cdylib artifact exists even under a bare `test -p`.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let _ = Command::new(cargo)
        .args(["build", "-p", "mlpl-ext-testdylib"])
        .status();
    let path = dylib_path();

    unsafe {
        let lib = Library::new(&path).unwrap_or_else(|e| panic!("dlopen {}: {e}", path.display()));
        let entry: Symbol<unsafe extern "C" fn() -> *const ExtensionDescriptorV1> = lib
            .get(b"sw_mlpl_extension_v1")
            .expect("resolve entry symbol");
        register_c_extension(entry()).expect("register descriptor");
        std::mem::forget(lib); // keep the library mapped for the process
    }

    let f = mlpl_extension_registry::lookup("testext:answer").expect("registered");
    assert_eq!(
        call_contained(&f.func, &[]).expect("invoke"),
        ExtValue::F64(ANSWER)
    );
}
