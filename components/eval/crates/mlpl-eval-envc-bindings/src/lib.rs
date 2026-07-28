//! Environment BINDING capabilities as traits (eval decomposition,
//! first capability peel; design in docs/eval-env-design.md). Each
//! trait is implemented here for `mlpl_eval_env::Environment` -- the
//! orphan rule allows the impl in the trait's own crate -- so call
//! sites keep method syntax by importing the trait (the hub re-exports
//! them all through its `env_api` prelude).

pub mod env_records;
pub mod env_results;
pub mod env_scope;
pub mod env_string_lists;
pub mod env_strings;
pub mod env_vars;

pub use env_records::EnvRecords;
pub use env_results::EnvResults;
pub use env_scope::{EnvScope, ScopeSnapshot};
pub use env_string_lists::EnvStringLists;
pub use env_strings::EnvStrings;
pub use env_vars::EnvVars;
