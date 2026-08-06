//! Filesystem `SourceProvider` for script mode: the include
//! sandbox lives HERE (the loader never touches IO). Root
//! containment is checked on canonicalized paths, so symlink
//! escapes reject too.

use std::path::{Path, PathBuf};

use mlpl_source_loader::{IncludeError, SourceId, SourceProvider};

pub struct FsProvider {
    root: PathBuf,
}

impl FsProvider {
    /// Sandbox rooted at `root` (canonicalized).
    ///
    /// # Errors
    /// A rendered message when `root` does not exist.
    pub fn new(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("--source-dir {}: {e}", root.display()))?;
        Ok(Self { root })
    }

    /// Canonical id for the ROOT script. The root script itself
    /// may live outside the sandbox (mlplunit runs a combined
    /// file from a temp dir with --source-dir pointing at the
    /// real sources); only its INCLUDES are confined to root.
    ///
    /// # Errors
    /// A rendered message when the script path cannot resolve.
    pub fn script_id(&self, script: &Path) -> Result<SourceId, String> {
        let c = script
            .canonicalize()
            .map_err(|e| format!("{}: {e}", script.display()))?;
        Ok(SourceId(c.to_string_lossy().into_owned()))
    }
}

impl SourceProvider for FsProvider {
    fn resolve(&self, from: &SourceId, rel: &str) -> Result<SourceId, IncludeError> {
        let bad = |reason: String| IncludeError::Unresolved {
            from: from.0.clone(),
            rel: rel.to_string(),
            reason,
        };
        if Path::new(rel).is_absolute() {
            return Err(bad("absolute include paths are rejected".into()));
        }
        // Files under the root resolve relative to themselves; a
        // root script OUTSIDE the sandbox resolves against root.
        let base = match Path::new(&from.0).parent() {
            Some(d) if d.starts_with(&self.root) => d.to_path_buf(),
            _ => self.root.clone(),
        };
        let canonical = base
            .join(rel)
            .canonicalize()
            .map_err(|e| bad(format!("no such source file ({e})")))?;
        canonical
            .starts_with(&self.root)
            .then(|| SourceId(canonical.to_string_lossy().into_owned()))
            .ok_or_else(|| bad("path escapes the source root".into()))
    }

    fn read(&self, id: &SourceId) -> Result<String, IncludeError> {
        let text = std::fs::read_to_string(&id.0).map_err(|e| IncludeError::Unresolved {
            from: id.0.clone(),
            rel: id.0.clone(),
            reason: format!("cannot read source ({e})"),
        })?;
        // Same full-line `#` stripping script mode always applied
        // (shebangs etc.); blank replacements keep line numbers.
        Ok(text
            .lines()
            .map(|l| {
                if l.trim_start().starts_with('#') {
                    ""
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }
}
