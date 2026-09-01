//! Visualization facade for MLPL: SVG diagram rendering + interactive
//! HTML/JS (Plotly). The implementation lives in the sibling crates
//! (mlpl-viz-core / -marks / -analysis, tech-debt spike step 001);
//! this crate keeps the dispatch, the gallery/plotly renderers, and
//! the stable `mlpl_viz::` re-export surface.

mod gallery;
mod gallery_layout;
mod gallery_overlay;
mod plotly3d;
pub mod svg;

pub use mlpl_viz_analysis::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_loss_landscape, analysis_pareto_plot, analysis_scatter_labeled,
    analysis_train_val_curve,
};
pub use plotly3d::render_plotly3d;
pub use svg::{
    VizError, render, render_attention_overlay, render_bar, render_critical_dimensions,
    render_dataflow, render_decision_boundary, render_gallery, render_heatmap, render_heatmap_grid,
    render_line, render_scatter, render_scatter3d, render_with_aux,
};
