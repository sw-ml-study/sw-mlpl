//! Mode-switching callbacks for the web playground:
//! `path`, `select`, and the `callbacks` bundle that wires
//! them together. Extracted from mlpl-web during saga 82.
//! mlpl-web re-exports each module under its original
//! `mode_*` namespace.

pub mod callbacks;
pub mod path;
pub mod select;
