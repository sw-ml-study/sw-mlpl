//! Ollama-model selection shared by `:ask` (standalone + connect REPL).
//!
//! Resolution precedence, highest first:
//!   1. an explicit `:ask use <name>` for this session,
//!   2. `$OLLAMA_MODEL`,
//!   3. a median-by-size auto-pick from the installed models (not too
//!      small / weak, not too big / slow),
//!   4. a small built-in fallback name.

use std::sync::Mutex;

static OVERRIDE: Mutex<Option<String>> = Mutex::new(None);
const FALLBACK_MODEL: &str = "llama3.2";

/// Set the session model (`:ask use <name>`).
pub fn set_model(name: &str) {
    *OVERRIDE.lock().unwrap() = Some(name.to_string());
}

/// Resolve the effective model for an `:ask` against `host`.
pub fn resolve(host: &str) -> String {
    if let Some(m) = OVERRIDE.lock().unwrap().clone() {
        return m;
    }
    if let Ok(m) = std::env::var("OLLAMA_MODEL") {
        if !m.is_empty() {
            return m;
        }
    }
    mlpl_runtime::list_models(host)
        .ok()
        .as_deref()
        .and_then(mlpl_runtime::median_model)
        .unwrap_or_else(|| FALLBACK_MODEL.to_string())
}

/// Render the `:ask models` listing for `host`, marking the current pick.
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
    format!("Ollama models at {host}:\n{body}\n(select with `:ask use <name>`)")
}
