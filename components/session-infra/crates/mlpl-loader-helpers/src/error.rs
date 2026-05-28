//! Local error vocabulary for the loader helpers.

use mlpl_array::ArrayError;

#[derive(Debug)]
pub enum LoaderHelperError {
    AbsolutePathRejected {
        path: String,
        root: String,
    },
    SandboxEscape {
        path: String,
        root: String,
    },
    RootedComponent {
        path: String,
        root: String,
    },
    NoDataRows {
        path: String,
    },
    HeaderOnly {
        path: String,
    },
    RaggedRow {
        path: String,
        row_idx: usize,
        got_cols: usize,
        expected_cols: usize,
    },
    NonNumericCell {
        path: String,
        row_idx: usize,
        cell: String,
    },
    ArrayError(ArrayError),
}

impl From<ArrayError> for LoaderHelperError {
    fn from(e: ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl std::fmt::Display for LoaderHelperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AbsolutePathRejected { path, root } => write!(
                f,
                "load(\"{path}\"): absolute paths are rejected; paths are relative to the sandbox root {root}"
            ),
            Self::SandboxEscape { path, root } => {
                write!(f, "load(\"{path}\"): path escapes sandbox root {root}")
            }
            Self::RootedComponent { path, root } => write!(
                f,
                "load(\"{path}\"): rooted components not permitted inside sandbox {root}"
            ),
            Self::NoDataRows { path } => {
                write!(f, "load(\"{path}\"): file contains no data rows")
            }
            Self::HeaderOnly { path } => {
                write!(f, "load(\"{path}\"): header-only file with no data rows")
            }
            Self::RaggedRow {
                path,
                row_idx,
                got_cols,
                expected_cols,
            } => write!(
                f,
                "load(\"{path}\"): ragged rows (row {row_idx} has {got_cols} cols, expected {expected_cols})"
            ),
            Self::NonNumericCell {
                path,
                row_idx,
                cell,
            } => write!(
                f,
                "load(\"{path}\"): non-numeric cell \"{cell}\" at row {row_idx}"
            ),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
        }
    }
}

impl std::error::Error for LoaderHelperError {}
