//! SVG dispatch: type-name -> renderer, spanning the mark and
//! analysis crates plus this crate's gallery/plotly renderers.

use mlpl_array::DenseArray;

pub use crate::gallery::render_gallery;
pub use crate::gallery_overlay::render_attention_overlay;
pub use mlpl_viz_analysis::{render_critical_dimensions, render_decision_boundary};
pub use mlpl_viz_core::VizError;
pub use mlpl_viz_flow::{Dataflow, render_dataflow};
pub use mlpl_viz_marks::{
    render_bar, render_heatmap, render_heatmap_grid, render_life, render_line, render_scatter,
    render_scatter3d, render_waffle,
};

/// Dispatch on a diagram type name. Returns the rendered SVG string.
pub fn render(data: &DenseArray, type_name: &str) -> Result<String, VizError> {
    render_with_aux(data, type_name, None)
}

/// Dispatch on a diagram type name with an optional auxiliary array.
/// Diagrams that need extra inputs (e.g. `decision_boundary` needs the
/// training points) read them from `aux`.
pub fn render_with_aux(
    data: &DenseArray,
    type_name: &str,
    aux: Option<&DenseArray>,
) -> Result<String, VizError> {
    match type_name {
        "scatter" => render_scatter(data),
        "scatter3d" => render_scatter3d(data),
        "plotly3d" => crate::plotly3d::render_plotly3d(data, aux),
        "line" => render_line(data),
        "bar" => render_bar(data),
        "heatmap" => render_heatmap(data),
        "heatmap_grid" => render_heatmap_grid(data),
        "life" => render_life(data),
        "waffle" => render_waffle(data),
        "critical_dimensions" => render_critical_dimensions(data, aux),
        "gallery" => render_gallery(data, aux),
        "attention_overlay" => render_attention_overlay(data, aux),
        "decision_boundary" => {
            render_decision_boundary(data, aux.ok_or_else(decision_boundary_missing_aux)?)
        }
        other => Err(VizError::UnknownType(other.to_string())),
    }
}

fn decision_boundary_missing_aux() -> VizError {
    VizError::InvalidShape(
        "decision_boundary requires a third argument (training points Nx3)".into(),
    )
}
