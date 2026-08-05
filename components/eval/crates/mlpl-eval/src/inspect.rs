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
    let (head, args) = parse_colon_line(trimmed)?;
    if let Some(msg) = name_arg_guard(head, &args) {
        return Some(msg);
    }
    let arg = args.first().copied();
    topic_output(env, head).or_else(|| match head {
        "version" => Some(crate::inspect_render::version_string()),
        "experiments" => Some(crate::experiment::format_registry(env)),
        "tags" => Some(crate::tag_render::format_tags(env)),
        "describe" => Some(crate::inspect_list::describe_names(env, &args)),
        "list" => Some(crate::inspect_list::list_or_usage(env, arg)),
        "untag" => Some(crate::inspect_list::handle_untag(env, arg)),
        "help" => arg.map(|t| help_topic(t, env).unwrap_or_else(|| help_unknown_topic(t))),
        _ => None,
    })
}

/// Split `:head arg arg...`. Name arguments accept the colon
/// spelling (`:describe :disp` == `:describe disp`) and trailing
/// commas (`:describe x, y`).
fn parse_colon_line(trimmed: &str) -> Option<(&str, Vec<&str>)> {
    let mut parts = trimmed.strip_prefix(':')?.split_whitespace();
    let head = parts.next()?;
    let args = parts
        .map(|a| {
            let a = a.trim_end_matches(',');
            a.strip_prefix(':').unwrap_or(a)
        })
        .filter(|a| !a.is_empty())
        .collect();
    Some((head, args))
}

/// The name-taking commands reject expressions LOUDLY: `:describe
/// x + y` must not silently describe `x`, and `:describe (x + y)`
/// must not report "'(x' is not bound". `:describe` accepts
/// several names; `:list` / `:untag` take exactly one.
fn name_arg_guard(head: &str, args: &[&str]) -> Option<String> {
    if !["describe", "list", "untag"].contains(&head) {
        return None;
    }
    let name_ok = |a: &&str| a.chars().all(|c| c.is_alphanumeric() || "_:.".contains(c));
    let bad = args.iter().any(|a| !name_ok(a)) || (head != "describe" && args.len() > 1);
    bad.then(|| {
        format!(
            "`:{head}` takes names, not expressions: `:{head} x`. To inspect a \
             computed value, bind it first (`t = x + y` then `:{head} t`) or view it \
             with `:disp(x + y)`."
        )
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
