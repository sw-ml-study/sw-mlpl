//! Rendering errors shared by every mlpl-viz renderer.

use std::fmt;

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
