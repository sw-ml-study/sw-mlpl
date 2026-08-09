//! The process-global extension registry. Each test uses a
//! DISTINCT namespace because the registry is a shared global and
//! tests run in one process (register is fail-closed on dup).

use mlpl_extension_abi::{ExtError, ExtFnDesc, ExtValue, ExtensionDescriptorV1};
use mlpl_extension_registry::{RegistryError, lookup, register, signatures};

fn answer(_: &[ExtValue]) -> Result<ExtValue, ExtError> {
    Ok(ExtValue::I64(42))
}

fn desc(ns: &str) -> ExtensionDescriptorV1 {
    ExtensionDescriptorV1 {
        name: ns.to_string(),
        private_namespace: format!("_{ns}"),
        facade_mlpl: String::new(),
        functions: vec![ExtFnDesc {
            name: "answer".to_string(),
            arity: 0,
            signature_toml: "returns = \"i64\"".to_string(),
            func: answer,
        }],
    }
}

#[test]
fn register_then_lookup_returns_the_function() {
    register(&desc("regtest_a")).unwrap();
    let f = lookup("regtest_a:answer").expect("registered");
    assert_eq!(f.arity, 0);
    assert_eq!((f.func)(&[]), Ok(ExtValue::I64(42)));
}

#[test]
fn lookup_miss_is_none() {
    register(&desc("regtest_b")).unwrap();
    assert!(lookup("regtest_b:nope").is_none());
    assert!(lookup("nosuchns:answer").is_none());
}

#[test]
fn duplicate_namespace_is_fail_closed() {
    register(&desc("regtest_c")).unwrap();
    let e = register(&desc("regtest_c")).unwrap_err();
    assert_eq!(
        e,
        RegistryError::DuplicateNamespace("regtest_c".to_string())
    );
}

#[test]
fn signatures_lists_registered_functions() {
    register(&desc("regtest_d")).unwrap();
    let sigs = signatures();
    assert!(
        sigs.iter()
            .any(|(k, s)| k == "regtest_d:answer" && s.contains("i64")),
        "expected regtest_d:answer signature in {sigs:?}"
    );
}
