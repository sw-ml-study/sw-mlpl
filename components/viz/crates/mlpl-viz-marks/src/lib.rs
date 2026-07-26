//! Chart-mark primitives for MLPL SVG rendering.

pub mod charts;
pub mod heatmap;
pub mod heatmap_grid;
mod life;
pub mod scatter;
pub mod waffle;

pub use charts::{render_bar, render_line};
pub use heatmap::render_heatmap;
pub use heatmap_grid::render_heatmap_grid;
pub use life::render_life;
pub use scatter::{render_scatter, render_scatter3d};
pub use waffle::render_waffle;
