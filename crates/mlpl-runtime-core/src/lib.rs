//! Saga 32 step 002: foundation crate for mlpl-runtime.
//!
//! Holds the leaf types every builtin needs:
//! - `RuntimeError`: error variant used by every dispatch arm
//! - `Xorshift64`: deterministic PRNG seeded from a scalar
//!
//! Both modules have ZERO outbound `use` statements, so this
//! crate sits at the bottom of the dependency DAG. Sibling
//! crates (`mlpl-runtime`, `mlpl-runtime-dimred`, ...) all
//! depend on it; nothing depends back upward, so there are no
//! cycles.

pub mod error;
pub mod prng;

pub use error::RuntimeError;
pub use prng::Xorshift64;
