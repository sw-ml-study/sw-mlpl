//! The in-memory provider: virtual paths for tests and web/WASM
//! registries, enforcing the same sandbox rules as the filesystem.

use std::collections::BTreeMap;

use crate::provider::{IncludeError, SourceId, SourceProvider};

/// In-memory provider: virtual paths with `/` separators. Used by
/// the loader tests and available to web/WASM registries. Applies
/// the same sandbox RULES as the filesystem provider: no absolute
/// paths, no escape above the virtual root.
#[derive(Default)]
pub struct MemoryProvider {
    files: BTreeMap<String, String>,
}

impl MemoryProvider {
    /// Register `text` under virtual path `path`.
    #[must_use]
    pub fn with(mut self, path: &str, text: &str) -> Self {
        self.files.insert(path.to_string(), text.to_string());
        self
    }
}

impl SourceProvider for MemoryProvider {
    fn resolve(&self, from: &SourceId, rel: &str) -> Result<SourceId, IncludeError> {
        let bad = |reason: &str| IncludeError::Unresolved {
            from: from.0.clone(),
            rel: rel.to_string(),
            reason: reason.to_string(),
        };
        if rel.starts_with('/') {
            return Err(bad("absolute paths are rejected; includes are relative"));
        }
        let dir = from.0.rsplit_once('/').map_or("", |(d, _)| d);
        let mut parts: Vec<&str> = Vec::new();
        for seg in dir.split('/').chain(rel.split('/')) {
            match seg {
                "" | "." => {}
                ".." => {
                    if parts.pop().is_none() {
                        return Err(bad("path escapes the source root"));
                    }
                }
                s => parts.push(s),
            }
        }
        Ok(SourceId(parts.join("/")))
    }

    fn read(&self, id: &SourceId) -> Result<String, IncludeError> {
        self.files
            .get(&id.0)
            .cloned()
            .ok_or_else(|| IncludeError::Unresolved {
                from: id.0.clone(),
                rel: id.0.clone(),
                reason: "no such source file".to_string(),
            })
    }
}
