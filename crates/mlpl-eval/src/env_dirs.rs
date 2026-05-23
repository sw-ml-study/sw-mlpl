//! Saga 33 step 003: data-dir / experiment-dir methods
//! extracted from `env.rs`. Experiment-log append/borrow lives
//! in the sibling `env_exp_log.rs`. The terminal REPL sets
//! `data_dir` from `--data-dir` and `exp_dir` from `--exp-dir`;
//! the web REPL leaves both `None` so disk access stays off.

use std::path::PathBuf;

use crate::env::Environment;

impl Environment {
    /// Set the sandbox root for `load("relative-path")` (Saga 12
    /// step 001). The terminal REPL calls this from a `--data-dir`
    /// CLI flag; the web REPL never calls this, leaving fs access
    /// disabled.
    pub fn set_data_dir(&mut self, dir: PathBuf) {
        self.data_dir = Some(dir);
    }

    /// Borrow the current sandbox root, if any.
    #[must_use]
    pub fn data_dir(&self) -> Option<&PathBuf> {
        self.data_dir.as_ref()
    }

    /// Set the output directory for `experiment` records. Saga 12
    /// step 007. The terminal REPL calls this from a `--exp-dir`
    /// CLI flag; the web REPL leaves it unset so nothing is
    /// written to disk.
    pub fn set_exp_dir(&mut self, dir: PathBuf) {
        self.exp_dir = Some(dir);
    }

    /// Borrow the configured experiment output dir, if any.
    #[must_use]
    pub fn exp_dir(&self) -> Option<&PathBuf> {
        self.exp_dir.as_ref()
    }
}
