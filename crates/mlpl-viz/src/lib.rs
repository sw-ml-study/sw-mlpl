//! Visualization for MLPL: SVG diagram rendering.

pub mod analysis;
pub mod svg;

pub use analysis::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_scatter_labeled,
};
pub use svg::{
    VizError, render, render_bar, render_decision_boundary, render_heatmap, render_line,
    render_scatter, render_scatter3d, render_with_aux,
};
