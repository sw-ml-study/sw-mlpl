//! Dataset fetch cluster (eval decomposition cluster peel 2): the
//! `fetch("name")` builtin's lookup + tarball download / extract /
//! verify path. IO-shaped, no tape or evaluator dependencies -- the
//! hub calls in through `mlpl_eval_fetch::eval`. Test modules ride
//! along as `#[path]` siblings of `fetch_dataset`.

mod fetch_dataset;
mod fetch_io;

pub use fetch_dataset::eval;
