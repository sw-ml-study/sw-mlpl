//! The provider seam: who may turn an include path into text.

/// Canonical identity of one source: for the filesystem provider
/// a canonicalized path string, for the memory provider its key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub String);

/// Everything that can go wrong resolving or loading includes.
#[derive(Debug, Clone, PartialEq)]
pub enum IncludeError {
    /// Path rejected or unreadable: sandbox violations, missing
    /// files, absolute paths, traversal escapes.
    Unresolved {
        /// The file whose include failed.
        from: String,
        /// The include argument as written.
        rel: String,
        /// Human reason (names the rule that rejected it).
        reason: String,
    },
    /// An include chain revisited a file; the chain is complete,
    /// root first, offender last.
    Cycle {
        /// Display names along the cycle.
        chain: Vec<String>,
    },
    /// A file failed to lex or parse; the error stays relative
    /// to THAT file's own text.
    Parse {
        /// Display name of the failing file.
        source: String,
        /// The underlying parse error.
        error: mlpl_lexer::ParseError,
    },
}

impl std::fmt::Display for IncludeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved { from, rel, reason } => {
                write!(f, "include \"{rel}\" (from {from}): {reason}")
            }
            Self::Cycle { chain } => {
                write!(f, "include cycle: {}", chain.join(" -> "))
            }
            Self::Parse { source, error } => write!(f, "{source}: {error:?}"),
        }
    }
}

/// Resolve include paths and read source text. Implementations
/// own ALL sandbox policy; the loader never touches IO.
pub trait SourceProvider {
    /// Turn an include argument, relative to the including file,
    /// into a canonical id -- or reject it with the rule it broke.
    ///
    /// # Errors
    /// [`IncludeError::Unresolved`] naming the violated rule.
    fn resolve(&self, from: &SourceId, rel: &str) -> Result<SourceId, IncludeError>;
    /// Read a source's full text.
    ///
    /// # Errors
    /// [`IncludeError::Unresolved`] when the source cannot be read.
    fn read(&self, id: &SourceId) -> Result<String, IncludeError>;
}
