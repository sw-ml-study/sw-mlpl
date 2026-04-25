//! Saga 21 step 003: CLI viz cache strategy.
//!
//! Replaces `mlpl-repl`'s old raw-`<svg>`-XML print
//! path with a write-to-disk + print-the-path
//! strategy. Both the local CLI and the
//! `--connect <url>` client route every
//! `Value::Str` through `transform_value` before
//! printing; SVG strings end up at
//! `$MLPL_CACHE_DIR/<sha256-prefix>.svg` (default
//! `dirs::cache_dir().join("mlpl")`), and non-SVG
//! strings pass through unchanged.
//!
//! Server-side `mlpl-serve` does NOT transform --
//! it returns raw SVG strings; clients decide where
//! to put them. A future saga may add a server-side
//! visualization-storage URL endpoint (out of
//! scope here).
//!
//! Other formats (PNG, HTML, JSON) are deferred --
//! `is_svg_string` is the only detector today.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const CACHE_DIR_ENV_VAR: &str = "MLPL_CACHE_DIR";
const CACHE_DIR_NAME: &str = "mlpl";
const HASH_PREFIX_LEN: usize = 12;

/// True if `s` looks like an SVG document: optional
/// XML prolog + leading `<svg`. Whitespace before
/// either token is allowed; we strip it before the
/// check.
#[must_use]
pub fn is_svg_string(s: &str) -> bool {
    let trimmed = s.trim_start();
    if trimmed.starts_with("<svg") {
        return true;
    }
    if let Some(rest) = trimmed.strip_prefix("<?xml") {
        return rest.trim_start().contains("<svg");
    }
    false
}

/// Content-addressed cache path for `content` under
/// `cache_dir`. The filename is the first
/// `HASH_PREFIX_LEN` (12) hex chars of the SHA-256
/// of the bytes plus an `.svg` suffix. Deterministic
/// (same input -> same path) and content-addressed
/// (different input -> different path).
#[must_use]
pub fn cache_path_for_content(content: &str, cache_dir: &Path) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hex = format!("{:x}", hasher.finalize());
    let prefix: String = hex.chars().take(HASH_PREFIX_LEN).collect();
    cache_dir.join(format!("{prefix}.svg"))
}

/// Resolve the cache directory + write `content`
/// there, creating the directory if missing. Returns
/// the path on success. The caller is expected to
/// call this only when `is_svg_string(content)` is
/// true.
///
/// # Errors
/// Bubbles up `io::Error` from `create_dir_all` or
/// `write` -- caller should fall back to printing
/// the raw content if the write fails.
pub fn write_to_cache(content: &str, cache_dir: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    let path = cache_path_for_content(content, cache_dir);
    fs::write(&path, content)?;
    Ok(path)
}

/// Run `s` through the cache transform: SVG strings
/// land on disk and come back as `viz: <path>`,
/// non-SVG strings pass through unchanged.
///
/// `cache_dir_override` lets a caller force a
/// specific directory (e.g., `--svg-out <dir>` on
/// the legacy CLI flag). When `None`, resolves via
/// `$MLPL_CACHE_DIR` -> `dirs::cache_dir()/mlpl/`
/// -> a tmp-dir fallback (so `transform_value`
/// never panics in environments without a valid
/// cache dir, e.g. some CI sandboxes).
///
/// On a write error we silently fall back to the
/// raw string -- the user still sees something
/// useful, and the error is the cache dir's, not
/// the program's.
#[must_use]
pub fn transform_value(s: &str, cache_dir_override: Option<&Path>) -> String {
    if !is_svg_string(s) {
        return s.to_string();
    }
    let cache_dir = cache_dir_override
        .map(Path::to_path_buf)
        .or_else(resolve_cache_dir)
        .unwrap_or_else(std::env::temp_dir);
    match write_to_cache(s, &cache_dir) {
        Ok(path) => format!("viz: {}", path.display()),
        Err(_) => s.to_string(),
    }
}

fn resolve_cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(CACHE_DIR_ENV_VAR) {
        return Some(PathBuf::from(dir));
    }
    dirs::cache_dir().map(|d| d.join(CACHE_DIR_NAME))
}
