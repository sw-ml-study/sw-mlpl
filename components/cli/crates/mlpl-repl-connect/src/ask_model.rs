//! Ollama-model selection shared by `:ask` (standalone + connect REPL).
//!
//! Resolution precedence, highest first:
//!   1. an explicit `:connect set <name>` for this session,
//!   2. `$OLLAMA_MODEL`,
//!   3. a median-by-size auto-pick from the installed models (not too
//!      small / weak, not too big / slow),
//!   4. a small built-in fallback name.

use std::sync::Mutex;

static OVERRIDE: Mutex<Option<String>> = Mutex::new(None);
const FALLBACK_MODEL: &str = "llama3.2";

/// Set the session model (`:connect set <name>`).
pub fn set_model(name: &str) {
    *OVERRIDE.lock().unwrap() = Some(name.to_string());
}

/// Resolve the effective model for an `:ask` against `host`.
pub fn resolve(host: &str) -> String {
    if let Some(m) = OVERRIDE.lock().unwrap().clone() {
        return m;
    }
    if let Ok(m) = std::env::var("OLLAMA_MODEL")
        && !m.is_empty()
    {
        return m;
    }
    mlpl_runtime::list_models(host)
        .ok()
        .as_deref()
        .and_then(mlpl_runtime::median_model)
        .unwrap_or_else(|| FALLBACK_MODEL.to_string())
}

/// Handle `:connect list` / `:connect set <model>` (Ollama model
/// management) against `host`. Returns the text to print.
pub fn connect_cmd(arg: &str, host: &str) -> String {
    let arg = arg.trim();
    if arg == "list" {
        list(host)
    } else if let Some(name) = arg.strip_prefix("set ").map(str::trim) {
        if name.is_empty() {
            "usage: :connect set <model>   (see :connect list)".to_string()
        } else {
            set_model(name);
            format!("ask model set to {name}")
        }
    } else if arg.is_empty() {
        format!(
            "current :ask model: {}\nusage: :connect list  |  :connect set <model>",
            resolve(host)
        )
    } else {
        format!("unknown :connect subcommand '{arg}' (try: list  |  set <model>)")
    }
}

/// Render the `:connect list` model listing for `host`, marking the pick.
pub fn list(host: &str) -> String {
    let mut models = match mlpl_runtime::list_models(host) {
        Ok(m) if !m.is_empty() => m,
        Ok(_) => return format!("no Ollama models at {host} (try: ollama pull qwen2.5:0.5b)"),
        Err(e) => return format!("could not list models at {host}: {e}"),
    };
    let current = resolve(host);
    models.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    let rows = models.iter().map(|(name, size)| {
        let mark = if *name == current {
            "   <- current"
        } else {
            ""
        };
        format!("  {name:<30} {:>5.1} GB{mark}", *size as f64 / 1e9)
    });
    let body = rows.collect::<Vec<_>>().join("\n");
    format!("Ollama models at {host}:\n{body}\n(select with `:connect set <name>`)")
}

// ---- Terminal rendering for connect-mode results (from the old
// bin-side render.rs). Lives here beside the :connect command
// formatting -- both are "turn a server response into terminal
// text" -- and keeps every module in this crate at the
// sw-checklist budgets. ----
use crate::connect::InspectResponse;

pub fn format_inspect(command: &str, snap: &InspectResponse) -> String {
    let mut out = String::new();
    let render_names = |out: &mut String, label: &str, names: &[String]| {
        if names.is_empty() {
            out.push_str(&format!("(no {label})"));
        } else {
            for n in names {
                out.push_str(&format!("  {n}\n"));
            }
            out.truncate(out.trim_end().len());
        }
    };
    match command {
        ":vars" => out.push_str(&format_vars(snap)),
        ":models" => render_names(&mut out, "models", &snap.models),
        ":tokenizers" => render_names(&mut out, "tokenizers", &snap.tokenizers),
        ":experiments" => render_names(&mut out, "experiments", &snap.experiments),
        ":wsid" => {
            out.push_str(&format_wsid(snap));
        }
        _ => unreachable!("dispatch_slash filters before format_inspect"),
    }
    out
}

/// The `:vars` listing: name, shape, `[param]` tag, truncation note.
fn format_vars(snap: &InspectResponse) -> String {
    let mut out = String::new();
    if snap.vars.is_empty() {
        out.push_str("(no variables)");
        return out;
    }
    for v in &snap.vars {
        let tag = if v.is_param { " [param]" } else { "" };
        let dims: Vec<String> = v.shape.iter().map(|d| d.to_string()).collect();
        out.push_str(&format!("  {}: [{}]{tag}\n", v.name, dims.join(", ")));
    }
    if snap.more > 0 {
        out.push_str(&format!("  ... ({} more)\n", snap.more));
    }
    out.truncate(out.trim_end().len());
    out
}

/// The `:wsid` workspace summary counts.
fn format_wsid(snap: &InspectResponse) -> String {
    format!(
        "workspace (remote):\n  variables: {}\n  models:    {}\n  tokenizers: {}\n  experiments: {}",
        snap.vars.len() + snap.more,
        snap.models.len(),
        snap.tokenizers.len(),
        snap.experiments.len()
    )
}
