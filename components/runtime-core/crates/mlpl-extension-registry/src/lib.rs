//! Public host registration + lookup registry for native
//! extensions (demo-extensions upstream contract A1). ONE
//! process-global registry (mirroring the `dispatch_hook`
//! `OnceLock` precedent) shared by the interpreter, statically
//! linked providers, and the binaries -- so the REPL, scripts,
//! and (later) compiled programs all resolve against the same
//! registry via one common `register` path (dynamic loading will
//! call the same entry point).
//!
//! Registration is FAIL-CLOSED (a duplicate public namespace is
//! rejected, no partial state) and copies every function
//! descriptor into host-owned memory. Functions are keyed by the
//! qualified `namespace:function` name (the colon call spelling
//! the parser already supports).

use std::collections::BTreeMap;
use std::sync::{OnceLock, RwLock};

use mlpl_extension_abi::{ExtFnDesc, ExtensionDescriptorV1};

#[derive(Default)]
struct Registry {
    namespaces: BTreeMap<String, ()>,
    functions: BTreeMap<String, ExtFnDesc>,
}

static STORE: OnceLock<RwLock<Registry>> = OnceLock::new();

fn store() -> &'static RwLock<Registry> {
    STORE.get_or_init(|| RwLock::new(Registry::default()))
}

/// A registration failure.
#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// The public namespace is already registered.
    DuplicateNamespace(String),
}

/// Register a validated extension. Fail-closed on a duplicate
/// public namespace; each function is copied host-owned and keyed
/// as `<namespace>:<function>`.
///
/// # Errors
/// Returns `DuplicateNamespace` if the descriptor's public
/// namespace is already registered.
pub fn register(desc: &ExtensionDescriptorV1) -> Result<(), RegistryError> {
    let mut reg = store().write().expect("extension registry poisoned");
    if reg.namespaces.contains_key(&desc.name) {
        return Err(RegistryError::DuplicateNamespace(desc.name.clone()));
    }
    for f in &desc.functions {
        reg.functions
            .insert(format!("{}:{}", desc.name, f.name), f.clone());
    }
    reg.namespaces.insert(desc.name.clone(), ());
    Ok(())
}

/// Look up a registered function by its qualified
/// `namespace:function` name, returning a host-owned clone.
#[must_use]
pub fn lookup(qualified: &str) -> Option<ExtFnDesc> {
    store()
        .read()
        .expect("extension registry poisoned")
        .functions
        .get(qualified)
        .cloned()
}

/// Every registered `(qualified_name, signature_toml)` pair, for
/// `help` / `:describe`.
#[must_use]
pub fn signatures() -> Vec<(String, String)> {
    store()
        .read()
        .expect("extension registry poisoned")
        .functions
        .iter()
        .map(|(k, f)| (k.clone(), f.signature_toml.clone()))
        .collect()
}
