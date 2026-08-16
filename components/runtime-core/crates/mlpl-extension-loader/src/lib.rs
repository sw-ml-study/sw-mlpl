//! The dynamic extension loader (B3). `load_c_extension` opens a
//! provider shared library, resolves its `sw_mlpl_extension_v1` entry,
//! and registers the returned descriptor through the SAME C ABI the
//! static path uses -- so a `dlopen`ed extension and a statically
//! linked one dispatch identically. The loaded library is held mapped
//! for the process (no `dlclose` in v1), since the descriptor's
//! function pointers live inside it. `libloading` stays confined here.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use mlpl_extension_cabi::{ExtensionDescriptorV1, register_c_extension};

/// Loaded libraries, kept mapped for the process lifetime.
static LIBS: OnceLock<Mutex<Vec<Library>>> = OnceLock::new();

fn libs() -> &'static Mutex<Vec<Library>> {
    LIBS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Load a provider cdylib at `path`, register its extension, and return
/// the registered namespace name.
///
/// # Safety
/// `path` must name a trusted, correctly built sw-mlpl extension: the
/// loader runs its `sw_mlpl_extension_v1` entry and calls its
/// function pointers. Loading arbitrary native code is inherently
/// unsafe.
///
/// # Errors
/// Returns a clear message if the library cannot be opened, has no V1
/// entry, or exports an invalid descriptor -- never a crash.
pub unsafe fn load_c_extension(path: &Path) -> Result<String, String> {
    let lib =
        unsafe { Library::new(path) }.map_err(|e| format!("dlopen {}: {e}", path.display()))?;
    let name = unsafe { register_from(&lib) }?;
    libs().lock().expect("loaded-libs lock").push(lib);
    Ok(name)
}

/// Resolve the V1 entry, register the descriptor, and read its
/// namespace name. Split out so `load_c_extension` only owns the
/// dlopen + hold-handle bookkeeping.
unsafe fn register_from(lib: &Library) -> Result<String, String> {
    let entry: Symbol<unsafe extern "C" fn() -> *const ExtensionDescriptorV1> =
        unsafe { lib.get(b"sw_mlpl_extension_v1") }
            .map_err(|e| format!("no sw_mlpl_extension_v1 entry: {e}"))?;
    let desc = unsafe { entry() };
    unsafe { register_c_extension(desc) }?;
    let d = unsafe { &*desc };
    if d.name.data.is_null() {
        return Err("descriptor has a null name".to_string());
    }
    let bytes = unsafe { std::slice::from_raw_parts(d.name.data, d.name.len) };
    String::from_utf8(bytes.to_vec()).map_err(|_| "extension name is not UTF-8".to_string())
}
