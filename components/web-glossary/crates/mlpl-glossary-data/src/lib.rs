//! Glossary data layer: parses `docs/glossary.md` into looked-up entries.
//! Pure (no yew/wasm), so it is a leaf the web UI crate depends on. See
//! docs/modular-build.md.

mod model;
mod parse;

pub use model::{GlossaryDoc, GlossaryEntry};
pub use parse::{doc, find_by_term};
