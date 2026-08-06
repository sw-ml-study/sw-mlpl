//! `fs_walk(root, {recursive, kind, pattern})` -- deterministic
//! lexical-order directory walking inside the sandbox. Symlinks
//! are NOT followed (explicit policy: cycle-safe by
//! construction); `pattern` is a simple `*` wildcard.

use std::collections::BTreeMap;
use std::path::Path;

use crate::fncall_fs::{contained, fs_err, fs_ok};
use mlpl_eval_types::Value;

pub(crate) fn walk(root: &Path, rel: &str, opts: &BTreeMap<String, Value>) -> Value {
    let recursive = matches!(opts.get("recursive"), Some(Value::Array(a)) if a.data() == [1.0]);
    let want_dirs = matches!(opts.get("kind"), Some(Value::Str(k)) if k == "dir");
    let pattern = match opts.get("pattern") {
        Some(Value::Str(p)) => p.clone(),
        _ => "*".to_string(),
    };
    let base = match contained(root, rel) {
        Ok(p) => p,
        Err(e) => return fs_err(format!("fs_walk: {e}")),
    };
    let mut found = Vec::new();
    if let Err(e) = collect(&base, recursive, want_dirs, &pattern, &mut found) {
        return fs_err(format!("fs_walk: {e}"));
    }
    let canon_root = match root.canonicalize() {
        Ok(c) => c,
        Err(e) => return fs_err(format!("fs_walk: {e}")),
    };
    let mut items: Vec<String> = found
        .iter()
        .filter_map(|p| p.strip_prefix(&canon_root).ok())
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .collect();
    items.sort();
    fs_ok(Value::StrList { items })
}

fn collect(
    dir: &Path,
    recursive: bool,
    want_dirs: bool,
    pattern: &str,
    out: &mut Vec<std::path::PathBuf>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        if ty.is_symlink() {
            continue;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if ty.is_dir() {
            if want_dirs && glob_match(pattern, &name) {
                out.push(path.clone());
            }
            if recursive {
                collect(&path, recursive, want_dirs, pattern, out)?;
            }
        } else if !want_dirs && glob_match(pattern, &name) {
            out.push(path);
        }
    }
    Ok(())
}

/// Simple `*` wildcard: the non-star segments must appear in
/// order, anchored at both ends.
fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut rest = name;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match rest.find(part) {
            Some(pos) => {
                if i == 0 && pos != 0 {
                    return false;
                }
                rest = &rest[pos + part.len()..];
            }
            None => return false,
        }
    }
    parts
        .last()
        .is_none_or(|last| last.is_empty() || rest.is_empty())
}
