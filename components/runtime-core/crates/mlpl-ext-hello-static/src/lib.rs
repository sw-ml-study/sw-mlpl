//! In-repo statically linked `hello` extension -- the provider
//! that proves the host registry/invoke path (demo-extensions
//! upstream contract, first slice). It builds an
//! `ExtensionDescriptorV1` and registers it. `register()` is
//! idempotent (a duplicate registration is ignored) so the
//! binaries and tests can call it freely.

use mlpl_extension_abi::{ExtError, ExtFnDesc, ExtValue, ExtensionDescriptorV1};
use mlpl_extension_registry::{RegistryError, register as registry_register};

fn answer(_: &[ExtValue]) -> Result<ExtValue, ExtError> {
    Ok(ExtValue::I64(42))
}

/// Always fails with a domain error (proves the err-Result path).
fn fail(_: &[ExtValue]) -> Result<ExtValue, ExtError> {
    Err(ExtError::new("hello.fail always fails"))
}

/// Always panics (proves host-side panic containment).
fn boom(_: &[ExtValue]) -> Result<ExtValue, ExtError> {
    panic!("hello.boom intentional panic");
}

const SIG_ANSWER: &str =
    "returns = \"i64\"\ndocumentation = \"Return the canonical extension answer.\"";
const SIG_PLAIN: &str = "returns = \"i64\"";

/// Register the `hello` extension. Idempotent: a duplicate
/// registration (already registered) is treated as success.
pub fn register() {
    let f =
        |name: &str, sig: &str, func: fn(&[ExtValue]) -> Result<ExtValue, ExtError>| ExtFnDesc {
            name: name.to_string(),
            arity: 0,
            signature_toml: sig.to_string(),
            func: std::sync::Arc::new(func),
        };
    let descriptor = ExtensionDescriptorV1 {
        name: "hello".to_string(),
        private_namespace: "_hello".to_string(),
        facade_mlpl: String::new(),
        functions: vec![
            f("answer", SIG_ANSWER, answer),
            f("fail", SIG_PLAIN, fail),
            f("boom", SIG_PLAIN, boom),
        ],
    };
    match registry_register(&descriptor) {
        Ok(()) | Err(RegistryError::DuplicateNamespace(_)) => {}
    }
}
