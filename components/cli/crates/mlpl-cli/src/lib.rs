//! CLI support library for MLPL.

pub mod viz_cache;
pub mod viz_format;

// Re-export so existing `use mlpl_cli::viz_cache::VizFormat`
// imports keep compiling -- the type was previously defined
// inside `viz_cache`. Saga 21.5 step 005 split it into its own
// module to defend the per-module function-count budget.
pub use viz_format::VizFormat;

pub fn crate_name() -> &'static str {
    "mlpl-cli"
}
pub mod fs_provider;
pub mod include_script;
pub mod script_exec;
