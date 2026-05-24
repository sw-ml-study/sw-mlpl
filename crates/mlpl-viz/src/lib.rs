//! Visualization for MLPL: SVG diagram rendering.

pub mod analysis;
mod boundary_2d_validate;
pub mod svg;

pub use analysis::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_scatter_labeled,
};
pub use svg::{
    VizError, render, render_attention_overlay, render_bar, render_decision_boundary,
    render_gallery, render_heatmap, render_heatmap_grid, render_line, render_scatter,
    render_scatter3d, render_with_aux,
};
