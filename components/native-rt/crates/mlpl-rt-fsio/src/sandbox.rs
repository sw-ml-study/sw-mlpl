//! The sandbox-root path check + numeric-argument validation shared
//! by the read and write functions.

use std::path::PathBuf;

use mlpl_array::DenseArray;

/// Resolve `rel` inside the sandbox root (`MLPL_FS_ROOT` or the cwd)
/// and reject escapes, mirroring the interpreter's `contained`:
/// canonicalize the longest existing prefix, re-append the missing
/// tail, and require the result to stay under the canonical root.
pub(crate) fn contained(rel: &str) -> Result<PathBuf, String> {
    let root = match std::env::var_os("MLPL_FS_ROOT") {
        Some(r) => PathBuf::from(r),
        None => std::env::current_dir().map_err(|e| format!("cwd: {e}"))?,
    };
    let canon_root = root
        .canonicalize()
        .map_err(|e| format!("sandbox root {}: {e}", root.display()))?;
    let mut probe = root.join(rel);
    let mut popped = Vec::new();
    let canon = loop {
        match probe.canonicalize() {
            Ok(c) => break c,
            Err(_) => match (probe.parent(), probe.file_name()) {
                (Some(parent), Some(name)) => {
                    popped.push(name.to_owned());
                    probe = parent.to_path_buf();
                }
                _ => return Err(format!("{rel}: outside the sandbox")),
            },
        }
    };
    let mut resolved = canon;
    resolved.extend(popped.iter().rev());
    if resolved.starts_with(&canon_root) {
        Ok(resolved)
    } else {
        Err(format!("{rel}: outside the sandbox"))
    }
}

/// A scalar non-negative integer, or a hard panic (interpreter
/// parity: an invalid offset/length is a hard error, not an `err`).
pub(crate) fn nonneg(a: &DenseArray, who: &str) -> u64 {
    let x = a.data()[0];
    assert!(
        a.rank() == 0 && x >= 0.0 && x.fract() == 0.0,
        "read_bytes: {who} must be a non-negative integer"
    );
    x as u64
}
