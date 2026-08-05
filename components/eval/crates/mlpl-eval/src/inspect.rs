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
    let topic = trimmed.strip_prefix(':')?;
    let mut parts = topic.split_whitespace();
    let head = parts.next()?;
    let arg = parts.next();
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
        "help" => arg.and_then(|t| help_topic(t, env)),
        _ => None,
    })
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

/// Whether a colon-prefixed line is a builtin-reference CALL
/// expression (`:disp(g)`) rather than a REPL command: `:` + an
/// identifier + an immediate `(`. Such lines parse and evaluate as
/// programs -- routing them to the command handler was the bug that
/// made `:disp(g)` "unknown" while `disp(g)` worked.
#[must_use]
pub fn is_colon_call_expr(line: &str) -> bool {
    let t = line.trim_start();
    let Some(rest) = t.strip_prefix(':') else {
        return false;
    };
    let ident_len = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    ident_len > 0 && rest[ident_len..].starts_with('(')
}

/// A tailored hint for `:name ...` lines where `name` is a documented
/// BUILTIN (not a command): explains the quote/call trichotomy
/// instead of a bare "unknown command".
#[must_use]
pub fn colon_ref_hint(line: &str) -> Option<String> {
    let word = line.trim().strip_prefix(':')?.split_whitespace().next()?;
    if !mlpl_eval_core::inspect_groups::documented_builtin_names().any(|n| n == word) {
        return None;
    }
    Some(format!(
        "`:{word}` is a builtin REFERENCE (the quoted, first-class form of `{word}`). \
         To call it, write `{word}(...)` or `:{word}(...)` -- `:{word} x` is not a command."
    ))
}

/// Error text for a colon line that is neither a recognized command
/// nor a colon-call expression. Callers must not let such lines fall
/// through to program evaluation: `:disp x` parses as a bare `:disp`
/// reference followed by `x`, silently printing `x`. Keeps every
/// REPL surface (terminal, web local, server) answering alike.
#[must_use]
pub fn colon_fallthrough_error(line: &str) -> Option<String> {
    let t = line.trim();
    if !t.starts_with(':') || is_colon_call_expr(t) {
        return None;
    }
    Some(colon_ref_hint(t).unwrap_or_else(|| format!("unknown command: {t}")))
}
