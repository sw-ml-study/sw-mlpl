//! Colon-line classification shared by every REPL surface
//! (terminal, web local, server): the colon-call test, the
//! builtin-REFERENCE trichotomy hint, and the fall-through guard
//! that keeps unrecognized colon lines out of program evaluation.
//! Split from `inspect.rs` for the module function budget.

/// Every REPL command word with a one-line brief, across all
/// surfaces (some are handled client-side in the web playground,
/// some server-side, some in the terminal binary). Used to catch
/// `:history()`-style lines (the parenthesized form looks like a
/// builtin call, but commands take no parentheses) and to answer
/// `:describe <command>`.
pub const REPL_COMMANDS: &[(&str, &str)] = &[
    ("vars", "list bound variables with shape and tag"),
    ("variables", "alias of :vars"),
    ("models", "list bound models with layer structure"),
    (
        "fns",
        "list your def u: functions with signatures and doc-strings",
    ),
    ("functions", "alias of :fns"),
    ("builtins", "list built-in functions by category"),
    ("built-ins", "alias of :builtins"),
    ("wsid", "workspace summary"),
    ("workspace", "alias of :wsid"),
    ("introspect", "run all no-arg inspectors at once"),
    ("version", "sw-MLPL version + target arch"),
    ("experiments", "list captured experiment runs"),
    ("tags", "list every binding's ValueTag"),
    (
        "describe",
        "describe a variable, model, tokenizer, built-in, or REPL command: :describe <name>",
    ),
    ("list", "print a u: function back verbatim: :list <u:name>"),
    (
        "untag",
        "clear a binding's auto-attached tag: :untag <name>",
    ),
    (
        "help",
        "command list and syntax summary; :help <topic> for focused help",
    ),
    (
        "history",
        "list recent REPL command lines (also given to :ask as context)",
    ),
    (
        "status",
        "connected backend devices + live telemetry; :status watch keeps updating",
    ),
    ("clear", "reset all variables, models, and session state"),
    (
        "reset",
        "cancel ALL in-flight work on the connected backend (y/N prompt)",
    ),
    (
        "ask",
        "send a question to the connected Ollama model: :ask <question>",
    ),
    (
        "connect",
        "list or pick the server's Ollama model: :connect list | :connect set <m>",
    ),
    ("upload", "bind a photo as a variable (web): :upload <name>"),
    ("tokenizers", "list bound tokenizers"),
    (
        "trace",
        "execution tracing: :trace on | off, :trace, :trace json [file]",
    ),
    ("2d", "close the 3D visualization stage"),
    (
        "3d",
        "open the 3D visualization stage; :3d reset re-centers the camera",
    ),
    ("exit", "quit the terminal REPL (also plain exit or quit)"),
    ("quit", "alias of :exit"),
];

/// Whether a colon-prefixed line is a builtin-reference CALL
/// expression (`:disp(g)`) rather than a REPL command: `:` + an
/// identifier + an immediate `(`. Such lines parse and evaluate as
/// programs -- routing them to the command handler was the bug that
/// made `:disp(g)` "unknown" while `disp(g)` worked.
#[must_use]
pub fn is_colon_call_expr(line: &str) -> bool {
    colon_call_ident(line).is_some()
}

/// The identifier of a colon-call line (`:disp(g)` -> `disp`), or
/// `None` when the line is not a colon call.
fn colon_call_ident(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix(':')?;
    let ident_len = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .count();
    (ident_len > 0 && rest[ident_len..].starts_with('(')).then(|| &rest[..ident_len])
}

/// A tailored hint for `:name ...` lines where `name` is a documented
/// BUILTIN (not a command): explains the quote/call trichotomy
/// instead of a bare "unknown command".
#[must_use]
pub fn colon_ref_hint(line: &str) -> Option<String> {
    let word = line.trim().strip_prefix(':')?.split_whitespace().next()?;
    if let Some(rest) = word.strip_prefix("u:") {
        return Some(format!(
            "`:u:{rest}` is a user-function REFERENCE. Bind it (`f = :u:{rest}`), \
             store it in a record, or call the function directly: `u:{rest}(...)`."
        ));
    }
    if !mlpl_eval_core::inspect_groups::documented_builtin_names().any(|n| n == word) {
        return None;
    }
    Some(format!(
        "`:{word}` is a builtin REFERENCE (the quoted, first-class form of `{word}`). \
         To call it, write `{word}(...)` or `:{word}(...)` -- `:{word} x` is not a command."
    ))
}

/// Error text for a colon line that is neither a recognized command
/// nor a legitimate colon-call expression. Callers must not let such
/// lines fall through to program evaluation: `:disp x` parses as a
/// bare `:disp` reference followed by `x`, silently printing `x`.
/// Also catches `:history()`-style command-with-parentheses lines.
/// Keeps every REPL surface answering alike.
#[must_use]
pub fn colon_fallthrough_error(line: &str) -> Option<String> {
    let t = line.trim();
    if !t.starts_with(':') {
        return None;
    }
    let Some(ident) = colon_call_ident(t) else {
        return Some(colon_ref_hint(t).unwrap_or_else(|| format!("unknown command: {t}")));
    };
    // `:history()` / `:describe(x)`: the name before `(` is a REPL
    // command, so the call form cannot work. Genuine builtin calls
    // pass through to program evaluation.
    if !REPL_COMMANDS.iter().any(|(n, _)| *n == ident)
        || mlpl_eval_core::inspect_groups::documented_builtin_names().any(|n| n == ident)
    {
        return None;
    }
    Some(format!(
        "`:{ident}` is a REPL command and takes no parentheses -- type `:{ident}`, \
         with any argument after a space (e.g. `:describe name`)."
    ))
}
