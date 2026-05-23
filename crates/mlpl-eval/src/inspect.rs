//! REPL introspection commands: `:vars`, `:models`, `:fns`, `:wsid`,
//! and `:describe <name>`. Shared between the terminal REPL
//! (`mlpl-repl`) and the web REPL (`mlpl-web` via `mlpl-wasm`) so
//! that both surfaces behave identically.
//!
//! These are inspired by APL's workspace conventions (`)VARS`,
//! `)FNS`, `)WSID`) but delivered as REPL commands rather than
//! language-level built-ins, so they stay out of the expression
//! grammar and never need to return a value.

use mlpl_array::DenseArray;
use mlpl_eval_core::inspect_groups::BUILTIN_GROUPS;

use crate::env::Environment;

/// If `input` is a recognized introspection command, returns the
/// rendered output. Returns `None` when the command is not one of
/// ours -- the caller should pass it through its normal handling
/// path (error for unknown commands, etc.).
pub fn inspect(env: &mut Environment, input: &str) -> Option<String> {
    let trimmed = input.trim();
    if !trimmed.starts_with(':') {
        return None;
    }
    let mut parts = trimmed.split_whitespace();
    let head = parts.next()?;
    let arg = parts.next();
    match head {
        ":vars" | ":variables" => Some(format_vars(env)),
        ":models" => Some(format_models(env)),
        ":fns" | ":functions" => Some(
            "(no user-defined functions)\n\
             user functions are not yet a language feature; for the \
             built-in surface, use :builtins"
                .into(),
        ),
        ":builtins" | ":built-ins" => Some(crate::inspect_render::format_builtins()),
        ":experiments" => Some(crate::experiment::format_registry(env)),
        ":version" => Some(format!(
            "MLPL v{} -- Array Programming Language for ML\n  \
             target: {}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::ARCH,
        )),
        ":wsid" => Some(format!(
            "workspace:\n  variables:       {}\n  parameters:      {}\n  \
             models:          {}\n  optimizer slots: {}",
            env.vars.len(),
            env.params.len(),
            env.models.len(),
            env.optim_state.buffers.len()
        )),
        ":describe" => Some(match arg {
            Some(name) => format_describe(env, name),
            None => "usage: :describe <name>".into(),
        }),
        ":tags" => Some(crate::tag_render::format_tags(env)),
        ":untag" => Some(handle_untag(env, arg)),
        ":help" => arg.and_then(|topic| help_topic(topic, env)),
        _ => None,
    }
}

fn handle_untag(env: &mut Environment, arg: Option<&str>) -> String {
    let Some(name) = arg else {
        return "usage: :untag <name>".into();
    };
    if env.get_tag(name).is_some() {
        env.clear_tag(name);
        format!("untagged {name}")
    } else {
        format!("{name} had no tag")
    }
}

/// Resolve `:help <topic>` to a corresponding inspector output.
/// `:help` with no topic is handled by the REPL itself (it prints
/// the long-form cheatsheet); only `:help <topic>` lands here.
fn help_topic(topic: &str, env: &Environment) -> Option<String> {
    match topic {
        "vars" | "variables" => Some(format_vars(env)),
        "models" => Some(format_models(env)),
        "fns" | "functions" => Some(
            "(no user-defined functions)\n\
             user functions are not yet a language feature; for the \
             built-in surface, use :builtins"
                .into(),
        ),
        "builtins" | "built-ins" => Some(crate::inspect_render::format_builtins()),
        "wsid" | "workspace" => Some(format!(
            "workspace:\n  variables:       {}\n  parameters:      {}\n  \
             models:          {}\n  optimizer slots: {}",
            env.vars.len(),
            env.params.len(),
            env.models.len(),
            env.optim_state.buffers.len()
        )),
        "describe" => Some(
            ":describe <name>\n  print the shape and a values preview \
             for a variable, the layer tree for a model, or the signature \
             and one-line doc for a built-in"
                .into(),
        ),
        _ => None,
    }
}

fn format_vars(env: &Environment) -> String {
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

fn format_models(env: &Environment) -> String {
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

fn describe_array(env: &Environment, name: &str, arr: &DenseArray) -> String {
    let shape = crate::inspect_render::format_shape(arr);
    let param_marker = if env.params.contains(name) {
        " (trainable param)"
    } else {
        ""
    };
    let data = arr.data();
    let preview = if data.is_empty() {
        "(empty)".to_string()
    } else {
        let take = 8.min(data.len());
        let head: Vec<String> = data[..take].iter().map(|v| format!("{v:.4}")).collect();
        if data.len() > take {
            format!("{} ... ({} total)", head.join(" "), data.len())
        } else {
            head.join(" ")
        }
    };
    let header = match env.get_tag(name) {
        Some(t) => format!("{name} -- {}", crate::tag_render::header_line(t)),
        None => format!("{name} -- array"),
    };
    let mut out = format!("{header}\n  shape: {shape}{param_marker}\n  values: {preview}");
    if let Some(t) = env.get_tag(name) {
        for line in crate::tag_render::body_lines(t, Some(arr)) {
            out.push_str(&format!("\n  {line}"));
        }
    }
    out
}

fn format_describe(env: &Environment, name: &str) -> String {
    if let Some(tok) = env.tokenizers.get(name) {
        return format!("{name} -- tokenizer\n  {}", tok.describe());
    }
    if let Some(spec) = env.models.get(name) {
        let mut out = format!(
            "{name} -- model\n  shape: {}\n",
            crate::inspect_render::render_spec(spec)
        );
        let ps = spec.params();
        if ps.is_empty() {
            out.push_str("  params: (none)");
        } else {
            out.push_str("  params:\n");
            for p in ps {
                if let Some(arr) = env.vars.get(&p) {
                    out.push_str(&format!(
                        "    {p}: {}\n",
                        crate::inspect_render::format_shape(arr)
                    ));
                }
            }
            out.truncate(out.trim_end().len());
        }
        return out;
    }
    if let Some(arr) = env.vars.get(name) {
        return describe_array(env, name, arr);
    }
    if let Some(s) = env.get_string(name) {
        // Web-UI demos bind `_demo` here; multi-line indented.
        let body = s
            .lines()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        return format!("{name} -- string ({} chars)\n{body}", s.len());
    }
    // Last fallback: flatten the grouped built-in list and look up by name.
    for (_, entries) in BUILTIN_GROUPS {
        if let Some(doc) = entries.iter().find(|(n, _, _)| *n == name) {
            return format!("{} -- built-in\n  {}\n  {}", doc.0, doc.1, doc.2);
        }
    }
    format!("'{name}' is not a bound variable, model, or built-in.")
}

// The `BUILTIN_GROUPS` data table moved to `inspect_groups.rs`
// to keep this file under the file-LOC budget. The remaining code
// below stays here because it consumes the table for output
// formatting (private helper) -- the data and the formatting are
// orthogonal axes of change.
