//! Analysis-flavored charts for MLPL: hist / confusion / loss
//! curves, train-vs-val, loss landscape, decision boundary, and PCA
//! loadings.

mod analysis;
mod boundary_2d_validate;
mod critical_dimensions;
mod decision_boundary;
mod loss_landscape;
mod train_curve;

pub use analysis::{
    analysis_boundary_2d, analysis_confusion_matrix, analysis_hist, analysis_loss_curve,
    analysis_scatter_labeled,
};
pub use critical_dimensions::render_critical_dimensions;
pub use decision_boundary::render_decision_boundary;
pub use loss_landscape::analysis_loss_landscape;
pub use train_curve::{analysis_pareto_plot, analysis_train_val_curve};
