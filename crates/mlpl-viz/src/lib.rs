//! Visualization for MLPL: SVG diagram rendering.

pub mod svg;

pub use svg::{VizError, render, render_bar, render_heatmap, render_line, render_scatter};
