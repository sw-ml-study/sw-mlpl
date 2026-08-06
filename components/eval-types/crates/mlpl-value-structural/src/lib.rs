//! `equal(a, b)` / `repr(value)` cores: total structural
//! comparison and bounded deterministic rendering over every
//! MLPL `Value` kind. Pure functions -- no environment, no IO --
//! so the terminal, web, and server surfaces share one behavior.
//! Upstream contract: mlplunit's sw-MLPL-changes-needed.md item 2.

mod eq;
mod fmt_util;
mod repr;

pub use eq::value_equal;
pub use repr::value_repr;
