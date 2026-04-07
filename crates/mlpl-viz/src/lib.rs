//! Visualization for MLPL: SVG diagram rendering.

pub mod svg;

pub use svg::{VizError, render, render_bar, render_line, render_scatter};
