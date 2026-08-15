//! Include resolution for the compiler front-end. Runs the SAME
//! `mlpl-source-loader::expand()` the interpreter's script mode
//! uses, so a compiled program resolves `include` identically to
//! the interpreted one (load-once, cycle chains, and the sandbox
//! below). The `FsProvider` sandbox is replicated from
//! `mlpl-cli/src/fs_provider.rs` (the loader itself never touches
//! IO); keep the two in sync.

use std::path::{Path, PathBuf};

use mlpl_parser_ast::Expr;
use mlpl_source_loader::{IncludeError, SourceId, SourceProvider, expand};

/// Load `input` and expand every reachable `include` into one
/// flat statement list, in source order. The include sandbox root
/// is `source_dir` when given, else the input file's own directory.
pub fn load_stmts(input: &Path, source_dir: Option<&Path>) -> Result<Vec<Expr>, String> {
    let root_dir = crate::project::resolve_root_dir(input, source_dir);
    let provider = FsProvider::new(&root_dir)?;
    // Canonical id for the root script (it may live outside the
    // sandbox; only its includes are confined to root).
    let canon = input
        .canonicalize()
        .map_err(|e| format!("{}: {e}", input.display()))?;
    let root = SourceId(canon.to_string_lossy().into_owned());
    let (chunks, _table) = match expand(&root, &provider) {
        Ok(out) => out,
        // Surface the offending source path (any involved file), so
        // a lex/parse error names its file instead of cascading into
        // a rustc error inside the temp project.
        Err(IncludeError::Parse { source, error }) => {
            return Err(format!("{source}: {error:?}"));
        }
        Err(other) => return Err(format!("include error: {other}")),
    };
    Ok(chunks.into_iter().flat_map(|c| c.stmts).collect())
}

/// Filesystem `SourceProvider`: the include sandbox. Root
/// containment is checked on canonicalized paths, so symlink
/// escapes reject too. The root script may live outside the
/// sandbox; only its INCLUDES are confined to root.
struct FsProvider {
    root: PathBuf,
}

impl FsProvider {
    fn new(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("--source-dir {}: {e}", root.display()))?;
        Ok(Self { root })
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
