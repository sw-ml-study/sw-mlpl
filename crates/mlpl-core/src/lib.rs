//! Shared low-level types for MLPL.
//!
//! This crate is deliberately small. It holds only types that two or
//! more MLPL crates genuinely need: source spans and identifiers.

mod ident;
mod span;

pub use ident::Identifier;
pub use span::Span;
