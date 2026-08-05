//! REPL introspection commands: `:vars`, `:models`, `:fns`, `:wsid`,
//! and `:describe <name>`. Shared between the terminal REPL
//! (`mlpl-repl`) and the web REPL (`mlpl-web` via `mlpl-wasm`) so
//! that both surfaces behave identically.
//!
//! These are inspired by APL's workspace conventions (`)VARS`,
//! `)FNS`, `)WSID`) but delivered as REPL commands rather than
//! language-level built-ins, so they stay out of the expression
//! grammar and never need to return a value.
//!
//! Saga 33 step 024 split the rendering helpers out to
//! `inspect_collections.rs` (`format_vars` / `format_models` /
//! `format_wsid`) and `inspect_describe.rs` (`format_describe`
//! and its per-kind helpers). What's left here is the thin
//! command parser + `topic_output` table, which the `:cmd` and
//! `:help cmd` forms share so the two stay in lock-step.

use crate::env::Environment;
use crate::env_api::*;

const HELP_DESCRIBE_MSG: &str = ":describe <name>\n  \
print the shape and a values preview \
for a variable, the layer tree for a model, or the signature \
and one-line doc for a built-in";

/// If `input` is a recognized introspection command, returns the
/// rendered output. Returns `None` when the command is not one of
/// ours -- the caller should pass it through its normal handling
/// path (error for unknown commands, etc.).
pub fn inspect(env: &mut Environment, input: &str) -> Option<String> {
    let trimmed = input.trim();
    if let Some(out) = dash_help(env, trimmed) {
        return Some(out);
    }
    let topic = trimmed.strip_prefix(':')?;
    let mut parts = topic.split_whitespace();
    let head = parts.next()?;
    // Accept the colon spelling of a name argument everywhere:
    // `:describe :disp` == `:describe disp`.
    let arg = parts.next().map(|a| a.strip_prefix(':').unwrap_or(a));
    topic_output(env, head).or_else(|| match head {
        "version" => Some(crate::inspect_render::version_string()),
        "experiments" => Some(crate::experiment::format_registry(env)),
        "tags" => Some(crate::tag_render::format_tags(env)),
        "describe" => Some(match arg {
            Some(name) => crate::inspect_describe::format_describe(env, name),
            None => "usage: :describe <name>".into(),
        }),
        "list" => Some(crate::inspect_list::list_or_usage(env, arg)),
        "untag" => Some(handle_untag(env, arg)),
        "help" => arg.map(|t| help_topic(t, env).unwrap_or_else(|| help_unknown_topic(t))),
        _ => None,
    })
}

/// `:<name> --help` (or `-h`): the command's one-line brief from
/// the shared registry, or the builtin's describe body. Lives at
/// this layer so every surface (terminal, web local, server)
/// answers identically -- the Usage Guide advertises the form.
fn dash_help(env: &Environment, trimmed: &str) -> Option<String> {
    let head = trimmed
        .strip_suffix(" --help")
        .or_else(|| trimmed.strip_suffix(" -h"))?
        .trim();
    let name = head.strip_prefix(':')?;
    if let Some((cmd, brief)) = crate::inspect_colon::REPL_COMMANDS
        .iter()
        .find(|(n, _)| *n == name)
    {
        return Some(format!(":{cmd} -- {brief}"));
    }
    Some(crate::inspect_describe::format_describe(env, name))
}

/// `:help <topic>` with a topic nobody recognizes: list what IS
/// available instead of falling through to evaluation.
fn help_unknown_topic(topic: &str) -> String {
    format!(
        "no help topic '{topic}'. Topics: vars, models, fns, builtins, describe, wsid, \
         introspect. For a builtin use `:describe <name>`; for a command's usage add \
         `--help` (e.g. `:trace --help`)."
    )
}

/// Topic names shared between `:topic` and `:help topic` so both
/// forms emit identical output. New no-arg listings should be
/// added here so they show up under both surfaces at once.
fn topic_output(env: &Environment, topic: &str) -> Option<String> {
    match topic {
        "vars" | "variables" => Some(crate::inspect_collections::format_vars(env)),
        "models" => Some(crate::inspect_collections::format_models(env)),
        "fns" | "functions" => Some(crate::inspect_collections::format_fns(env)),
        "builtins" | "built-ins" => Some(crate::inspect_render::format_builtins()),
        "wsid" | "workspace" => Some(crate::inspect_collections::format_wsid(env)),
        "introspect" => Some(crate::inspect_introspect::format_introspect(env)),
        _ => None,
    }
}

/// `:help <topic>` falls back to the shared `topic_output` table,
/// then adds the `describe` help string that's specific to the
/// `:help` surface (`:describe` itself needs an argument).
fn help_topic(topic: &str, env: &Environment) -> Option<String> {
    topic_output(env, topic).or_else(|| match topic {
        "describe" => Some(HELP_DESCRIBE_MSG.into()),
        _ => None,
    })
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
