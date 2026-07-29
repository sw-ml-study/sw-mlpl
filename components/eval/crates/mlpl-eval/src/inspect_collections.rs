//! REPL listing helpers: `:vars`, `:models`, `:wsid`.
//!
//! Lifted out of `inspect.rs` so that the dispatcher there stays
//! under the 25-LOC function-LOC budget (saga 33 step 024). Each
//! helper takes `&Environment` immutably and returns a rendered
//! string ready to print.

use crate::env::Environment;
use crate::env_api::*;

/// Shown by `:fns` and the `:introspect` `## :fns` section when the
/// session has no user-defined functions yet.
pub(crate) const NO_USER_FNS_MSG: &str = "(no user-defined functions yet)\n\
define with: def u:name(args) { body }";

/// The user-defined function signatures (`:fns`), one per line, or the
/// "none yet" hint. Shared by the `:fns` command and `:introspect`.
pub(crate) fn format_fns(env: &Environment) -> String {
    let sigs = env.user_fn_signatures();
    if sigs.is_empty() {
        NO_USER_FNS_MSG.to_string()
    } else {
        sigs.join("\n")
    }
}

pub(crate) fn format_vars(env: &Environment) -> String {
    if env.vars.is_empty() {
        return "(no variables bound)".into();
    }
    let mut names: Vec<&String> = env.vars.keys().collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let arr = &env.vars[name];
        let shape = crate::inspect_render::format_shape(arr);
        let param_marker = if env.params.contains(name) {
            " [param]"
        } else {
            ""
        };
        let tag_marker = match env.get_tag(name) {
            Some(t) => format!("  {}", crate::tag_render::header_line(t)),
            None => String::new(),
        };
        out.push_str(&format!("  {name}: {shape}{param_marker}{tag_marker}\n"));
    }
    out.truncate(out.trim_end().len());
    out
}

pub(crate) fn format_models(env: &Environment) -> String {
    // ":models" lists the MLPL model objects YOU built in this workspace
    // (chain(...), etc.) -- distinct from ":connect list", which lists the
    // connected server's Ollama LLMs for ":ask".
    if env.models.is_empty() {
        return "(no models in this workspace yet -- build one with chain(...). \
                For the server's LLMs, see :connect list.)"
            .into();
    }
    let mut names: Vec<&String> = env.models.keys().collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let spec = &env.models[name];
        let param_count = spec.params().len();
        out.push_str(&format!(
            "  {name}: {} ({param_count} params)\n",
            crate::inspect_render::render_spec(spec)
        ));
    }
    out.truncate(out.trim_end().len());
    out
}

pub(crate) fn format_wsid(env: &Environment) -> String {
    format!(
        "workspace:\n  variables:       {}\n  parameters:      {}\n  \
         models:          {}\n  optimizer slots: {}",
        env.vars.len(),
        env.params.len(),
        env.models.len(),
        env.optim_state.buffers.len()
    )
}
