//! Shared low-level types for MLPL.
//!
//! This crate is deliberately small. It holds only types that two or
//! more MLPL crates genuinely need: source spans, identifiers, and
//! axis-labeled shapes.

mod ident;
mod labels;
mod span;

pub use ident::Identifier;
pub use labels::LabeledShape;
pub use span::Span;
