//! Connect-mode client for the MLPL REPL: remote sessions against
//! `mlpl-serve` (eval, streaming, cancel, reattach) plus the
//! `:ask` / `:connect` command surface and their terminal
//! rendering. Extracted from the `mlpl-repl` bin (spike step 013).

pub mod ask;
pub mod ask_model;
pub mod connect;
pub mod connect_reattach;
pub mod connect_repl;
pub mod connect_stream;
