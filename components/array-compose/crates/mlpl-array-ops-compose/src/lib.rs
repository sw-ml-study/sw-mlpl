//! Composition extension traits + free functions for DenseArray:
//! `concat`, `stack`, `patchify`, `take`. Body extracted from
//! mlpl-array in saga 53.

mod concat;
mod patchify;
mod stack;
mod take;
mod windows;

pub use concat::ConcatExt;
pub use patchify::PatchifyExt;
pub use stack::{RotateExt, stack};
pub use take::TakeExt;
pub use windows::WindowsExt;

pub mod prelude {
    pub use super::{ConcatExt, PatchifyExt, RotateExt, TakeExt, WindowsExt, stack};
}
