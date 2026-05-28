//! Visualization for MLPL: SVG diagram rendering + interactive
//! HTML/JS (Plotly) for the dim-reduction milestone's 3D
//! scatter viewer.

pub mod analysis;
mod boundary_2d_validate;
mod plotly3d;
pub mod svg;

pub use analysis::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_scatter_labeled,
};
pub use plotly3d::render_plotly3d;
pub use svg::{
    VizError, render, render_attention_overlay, render_bar, render_critical_dimensions,
    render_decision_boundary, render_gallery, render_heatmap, render_heatmap_grid, render_line,
    render_scatter, render_scatter3d, render_with_aux,
};
