//! Data-directory sandbox root (None = filesystem disabled,
//! the WASM surface).

use mlpl_eval_env::Environment;
use std::path::PathBuf;

impl EnvDataDir for Environment {
    fn set_data_dir(&mut self, dir: PathBuf) {
        self.data_dir = Some(dir);
    }

    fn data_dir(&self) -> Option<&PathBuf> {
        self.data_dir.as_ref()
    }
}

/// Filesystem sandbox root for `load("relative-path")`.
pub trait EnvDataDir {
    /// Set the sandbox root for `load("relative-path")` (Saga 12
    /// step 001). The terminal REPL calls this from a `--data-dir`
    /// CLI flag; the web REPL never calls this, leaving fs access
    /// disabled.
    fn set_data_dir(&mut self, dir: PathBuf);
    /// Borrow the current sandbox root, if any.
    fn data_dir(&self) -> Option<&PathBuf>;
}
