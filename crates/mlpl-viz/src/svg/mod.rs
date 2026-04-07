//! SVG diagram rendering for MLPL arrays.

mod bar;
mod helpers;
mod line;
mod scatter;

use std::fmt;

use mlpl_array::DenseArray;

pub use bar::render_bar;
pub use line::render_line;
pub use scatter::render_scatter;

pub(crate) use helpers::{H, PAD, W, bounds, scale, write_svg_close, write_svg_open};

/// Errors produced by SVG rendering.
#[derive(Clone, Debug, PartialEq)]
pub enum VizError {
    /// The data shape is not valid for the requested diagram.
    InvalidShape(String),
    /// The diagram type name is not recognized.
    UnknownType(String),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid shape for diagram: {s}"),
            Self::UnknownType(s) => write!(f, "unknown svg type: '{s}'"),
        }
    }
}

impl std::error::Error for VizError {}

/// Dispatch on a diagram type name. Returns the rendered SVG string.
pub fn render(data: &DenseArray, type_name: &str) -> Result<String, VizError> {
    match type_name {
        "scatter" => render_scatter(data),
        "line" => render_line(data),
        "bar" => render_bar(data),
        other => Err(VizError::UnknownType(other.to_string())),
    }
}
