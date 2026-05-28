//! REPL listing helpers: `:vars`, `:models`, `:wsid`.
//!
//! Lifted out of `inspect.rs` so that the dispatcher there stays
//! under the 25-LOC function-LOC budget (saga 33 step 024). Each
//! helper takes `&Environment` immutably and returns a rendered
//! string ready to print.

use crate::env::Environment;

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
    if env.models.is_empty() {
        return "(no models bound)".into();
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
