//! Pure-inspection ops over a `ModelSpec`. Saga 33 step 013.
//!
//! - `embed_table(model)`: pull the first Embedding layer's
//!   table out of `env`.
//! - `estimate_train(model, steps, batch, seq [, dtype_bytes])`:
//!   honest-approximate cost estimator (params, VRAM, FLOPs,
//!   wall-clock).
//!
//! Both ops follow the C+D pattern: generic over the env
//! capability traits they need, with a local `InspectError`
//! vocabulary and caller-injected resolver closures so this
//! crate never imports the wider eval engine.

pub mod embed_table;
pub mod error;
pub mod estimate;
pub mod estimate_compute;
pub mod estimate_walk;

pub use embed_table::embed_table_inner;
pub use error::InspectError;
pub use estimate::estimate_train_inner;
