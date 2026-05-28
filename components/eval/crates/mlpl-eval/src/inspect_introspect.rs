//! Saga 33 step 037d: `:introspect` bundles every no-arg
//! inspector into one large markdown-headered output.
//!
//! Sections in fixed order: `:version`, `:wsid`, `:builtins`,
//! `:vars`, `:models`, `:experiments`, `:tags`. Each section
//! prints under a `## :<topic>` header so a long scroll
//! stays scannable. Arg-taking inspectors (`:describe`,
//! `:untag`, `:help`) are NOT included -- they need user
//! input and don't fit the "dump everything" intent.
//!
//! Used as the optional final line of any demo to capture
//! full workspace state at the end (the Workspace
//! Introspection demo wires it in by default in step 037d).

use crate::env::Environment;
use crate::experiment::format_registry;
use crate::inspect_collections::{format_models, format_vars, format_wsid};
use crate::inspect_render::{format_builtins, version_string};
use crate::tag_render::format_tags;

/// Render the bundled introspection dump as one string.
/// Sections are separated by `\n\n## :<topic>\n\n` headers.
pub(crate) fn format_introspect(env: &Environment) -> String {
    let sections: [(&str, String); 7] = [
        (":version", version_string()),
        (":wsid", format_wsid(env)),
        (":builtins", format_builtins()),
        (":vars", format_vars(env)),
        (":models", format_models(env)),
        (":experiments", format_registry(env)),
        (":tags", format_tags(env)),
    ];
    let mut out = String::new();
    for (name, body) in &sections {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("## {name}\n\n{body}"));
    }
    out
}
