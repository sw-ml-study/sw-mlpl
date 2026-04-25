//! Saga 21 step 003: thin holder for the optional
//! `--svg-out <dir>` override flag. Actual SVG ->
//! cache transformation now lives in
//! `mlpl_cli::viz_cache::transform_value`, which both
//! local and `--connect` modes route through.

use std::path::PathBuf;

/// Override directory for SVG cache writes. When
/// unset, `viz_cache::transform_value` resolves
/// `$MLPL_CACHE_DIR` -> `dirs::cache_dir()/mlpl/`.
pub struct SvgOut {
    pub dir: Option<PathBuf>,
}

impl SvgOut {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }
}
