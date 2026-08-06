//! Static-include expansion (docs/static-include-design-mlpl.md).
//!
//! Each source file is lexed and parsed INDEPENDENTLY, so its
//! spans stay valid byte offsets into its own text; the loader
//! splices per-file chunks at the include site and keeps a
//! `SourceTable` for `path:line:column` diagnostics. The
//! evaluator never sees a filesystem: all IO hides behind
//! `SourceProvider`, and the in-memory provider gives web/WASM
//! parity plus filesystem-free tests.

mod expand;
mod memory;
mod provider;

pub use expand::{Chunk, SourceTable, expand};
pub use memory::MemoryProvider;
pub use provider::{IncludeError, SourceId, SourceProvider};
