//! Shared SVG scaffolding for the mlpl-viz crate family: canvas
//! constants, data scaling, document open/close helpers, and the
//! `VizError` type every renderer returns.

pub mod error;
pub mod scaffold;

pub use error::VizError;
pub use scaffold::{
    H, PAD, W, bounds, data_range, scale, write_corner_scale_labels, write_svg_close,
    write_svg_open, write_svg_open_with_size,
};
