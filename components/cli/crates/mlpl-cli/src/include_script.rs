//! Script loading through the include expander: paths in,
//! evaluated-ready chunks out, with `path:line:column` rendering
//! for anything that fails before evaluation.

use std::path::{Path, PathBuf};

use mlpl_source_loader::{Chunk, IncludeError, SourceTable, expand};

use crate::fs_provider::FsProvider;

pub struct LoadedScript {
    pub chunks: Vec<Chunk>,
    pub table: SourceTable,
}

/// Load `path` and expand its includes. The sandbox root is
/// `source_dir` when given, else the script's own directory.
///
/// # Errors
/// A fully rendered message (sandbox violations, cycles, or
/// `path:line:column` parse errors in any involved file).
pub fn load_script(path: &Path, source_dir: Option<&Path>) -> Result<LoadedScript, String> {
    let root_dir = resolve_root_dir(path, source_dir);
    let provider = FsProvider::new(&root_dir)?;
    let root = provider.script_id(path)?;
    match expand(&root, &provider) {
        Ok((chunks, table)) => Ok(LoadedScript { chunks, table }),
        Err(IncludeError::Parse { source, error }) => {
            Err(render_parse_error(&source, &error, path))
        }
        Err(other) => Err(format!("error: {other}")),
    }
}

/// The sandbox root for a script: `source_dir` when given, else the
/// script's own directory, else the current directory. A bare
/// filename's `parent()` is `Some("")` (not `None`), which would make
/// the root the empty path and fail `FsProvider::new("")` with a
/// misleading `--source-dir` error; treat an empty parent as absent so
/// `mlpl-repl mini.mlpl` resolves to `.`.
pub(crate) fn resolve_root_dir(path: &Path, source_dir: Option<&Path>) -> PathBuf {
    source_dir
        .map(Path::to_path_buf)
        .or_else(|| {
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
        })
        .unwrap_or_else(|| Path::new(".").to_path_buf())
}

fn render_parse_error(source: &str, error: &mlpl_lexer::ParseError, root: &Path) -> String {
    // Re-read is fine here: this is the error path, and the text
    // already loaded once.
    let text = std::fs::read_to_string(source).unwrap_or_default();
    let (line, col) = line_col(&text, error_span_start(error));
    let shown = Path::new(source)
        .strip_prefix(root.parent().unwrap_or_else(|| Path::new("")))
        .unwrap_or_else(|_| Path::new(source));
    format!("error: {}:{line}:{col}: {error:?}", shown.display())
}

fn error_span_start(e: &mlpl_lexer::ParseError) -> usize {
    use mlpl_lexer::ParseError as P;
    match e {
        P::UnexpectedCharacter { span, .. }
        | P::InvalidNumber { span }
        | P::UnexpectedToken { span, .. }
        | P::UnclosedDelimiter { span, .. }
        | P::DuplicateRecordField { span, .. } => span.start,
        P::InvalidUtf8 { .. } => 0,
    }
}

/// 1-based line and column for a byte offset.
#[must_use]
pub fn line_col(text: &str, byte: usize) -> (usize, usize) {
    let clamped = byte.min(text.len());
    let before = &text[..clamped];
    let line = before.matches('\n').count() + 1;
    let col = clamped - before.rfind('\n').map_or(0, |i| i + 1) + 1;
    (line, col)
}

#[cfg(test)]
mod root_dir_tests {
    use super::resolve_root_dir;
    use std::path::Path;

    /// A1 regression: a bare filename must resolve to `.`, never the
    /// empty path (which fails `FsProvider::new("")`).
    #[test]
    fn bare_filename_root_is_dot_not_empty() {
        let root = resolve_root_dir(Path::new("mini.mlpl"), None);
        assert_eq!(root, Path::new("."));
        assert!(!root.as_os_str().is_empty());
    }

    /// The other spellings are unchanged.
    #[test]
    fn subdir_and_source_dir_unchanged() {
        assert_eq!(
            resolve_root_dir(Path::new("sub/mini.mlpl"), None),
            Path::new("sub")
        );
        assert_eq!(
            resolve_root_dir(Path::new("mini.mlpl"), Some(Path::new("/x"))),
            Path::new("/x")
        );
    }
}
