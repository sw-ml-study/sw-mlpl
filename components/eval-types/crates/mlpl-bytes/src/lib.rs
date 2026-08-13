//! Typed packed byte-buffer support: element dtypes + canonical
//! little-endian packing. A leaf crate (no dependencies) so
//! `mlpl-eval-types` can hold a `Value::Bytes { dtype, .. }` variant
//! without growing its own module count.

mod dtype;
mod pack;
mod read;

pub use dtype::ByteDtype;
pub use pack::pack_f64s;
pub use read::read_le;
