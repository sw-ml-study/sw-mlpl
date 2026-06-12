//! Saga 33 step 037d: `:introspect` bundles every no-arg
//! inspector into one large markdown-headered output.
//!
//! Sections in fixed order: `:version`, `:wsid`, `:builtins`,
//! `:vars`, `:models`, `:fns`, `:experiments`, `:tags`. Each
//! section prints under a `## :<topic>` header so a long scroll
//! stays scannable. Arg-taking inspectors (`:describe`,
//! `:untag`, `:help`) are NOT included -- they need user
//! input and don't fit the "dump everything" intent. A trailing
//! `## connect-mode introspection` section names the inspectors
//! that need a connected server (`:status`, `:connect list`,
//! `:ask`): they cannot run in-process, so they are named, not
//! dumped, and the note disambiguates `:models` (your workspace
//! models) from `:connect list` (the server's LLMs).
//!
//! Used as the optional final line of any demo to capture
//! full workspace state at the end (the Workspace
//! Introspection demo wires it in by default in step 037d).

use crate::env::Environment;
use crate::experiment::format_registry;
use crate::inspect_collections::{format_fns, format_models, format_vars, format_wsid};
use crate::inspect_render::{format_builtins, version_string};
use crate::tag_render::format_tags;

/// Server-side introspectors that cannot run in this in-process dump;
/// named (not dumped) so the surface is complete and `:models` above is
/// not confused with `:connect list`.
const CONNECT_MODE_NOTE: &str = "\n\n## connect-mode introspection\n\n\
(live server-side commands -- named here, not captured in this in-process snapshot; run each directly)\n  \
:status        backend self-test: devices + live CPU/GPU/RAM\n  \
:connect list  the server's installed Ollama LLMs, for :ask -- NOT the same as :models above\n  \
:ask <prompt>  ask the connected LLM about your session";

/// Render the bundled introspection dump as one string.
/// Sections are separated by `\n\n## :<topic>\n\n` headers.
pub(crate) fn format_introspect(env: &Environment) -> String {
    let sections: [(&str, String); 8] = [
        (":version", version_string()),
        (":wsid", format_wsid(env)),
        (":builtins", format_builtins()),
        (":vars", format_vars(env)),
        (":models", format_models(env)),
        (":fns", format_fns(env)),
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
    out.push_str(CONNECT_MODE_NOTE);
    out
}
