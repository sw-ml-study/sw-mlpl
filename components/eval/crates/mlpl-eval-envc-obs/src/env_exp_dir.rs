//! Experiment output directory for `experiment` records.

use mlpl_eval_env::Environment;
use std::path::PathBuf;

impl EnvExpDir for Environment {
    fn set_exp_dir(&mut self, dir: PathBuf) {
        self.exp_dir = Some(dir);
    }

    fn exp_dir(&self) -> Option<&PathBuf> {
        self.exp_dir.as_ref()
    }
}

/// Experiment output directory (None disables disk writes).
pub trait EnvExpDir {
    /// Set the output directory for `experiment` records. Saga 12
    /// step 007. The terminal REPL calls this from a `--exp-dir`
    /// CLI flag; the web REPL leaves it unset so nothing is
    /// written to disk.
    fn set_exp_dir(&mut self, dir: PathBuf);
    /// Borrow the configured experiment output dir, if any.
    fn exp_dir(&self) -> Option<&PathBuf>;
}
