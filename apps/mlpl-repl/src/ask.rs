//! `:ask <question>` REPL command.
//!
//! Saga 19 preview: calls a local Ollama server at
//! `OLLAMA_HOST` (default `http://localhost:11434`) with a prompt
//! template that includes (a) a system framing so the model knows
//! it is explaining MLPL, (b) the current workspace summary (so
//! "what's in my session?" questions work), and (c) the user's
//! question. The caller is the CLI REPL; the web REPL stays out
//! of this path because (i) CORS requires the user to set
//! `OLLAMA_ORIGINS` before the browser's fetch is allowed, and
//! (ii) this feature is primarily useful in the interactive CLI
//! where the user is mid-session anyway. Saga 19 ships the real
//! REST integration story across web + CLI + codegen.
//!
//! Model selection is via `OLLAMA_MODEL` (default `llama3.2`).
//! The command blocks the REPL until Ollama responds; streaming
//! is a Saga 19 follow-up.

use mlpl_eval::Environment;

const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "llama3.2";
const TIMEOUT_SECS: u64 = 120;

/// Dispatch `:ask <question>` -- called from the main REPL
/// command table. Prints the answer to stdout or an
/// actionable-to-fix error to stderr.
pub fn dispatch(question: &str, env: &Environment) {
    let question = question.trim();
    if question.is_empty() {
        eprintln!("usage: :ask <question>");
        eprintln!("  example: :ask what did I just train?");
        return;
    }
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.into());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    match call_ollama(&host, &model, question, env) {
        Ok(answer) => println!("{}", answer.trim_end()),
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("  :ask needs a running Ollama server at {host}.");
            eprintln!("  Start one with: ollama serve && ollama pull {model}");
            eprintln!("  Override host/model with OLLAMA_HOST / OLLAMA_MODEL env vars.");
        }
    }
}

/// Build the request payload and POST to Ollama's `/api/chat`.
/// Returns the model's reply text on success; the error string
/// surfaces the connection failure or non-200 body so the REPL
/// can print an actionable fix hint.
fn call_ollama(
    host: &str,
    model: &str,
    question: &str,
    env: &Environment,
) -> Result<String, String> {
    let system = build_system_prompt();
    let user_context = build_user_context(env, question);
    let body = serde_json::json!({
        "model": model,
        "stream": false,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_context},
        ],
    });
    let url = format!("{}/api/chat", host.trim_end_matches('/'));
    let resp = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .build()
        .post(&url)
        .set("Content-Type", "application/json")
        .send_json(body)
        .map_err(|e| format!("POST {url} failed: {e}"))?;
    let json: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("bad JSON from {url}: {e}"))?;
    json.pointer("/message/content")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("no /message/content in Ollama response: {json}"))
}

/// System prompt framing the model's task. Short on purpose --
/// a small local model (llama3.2, qwen2.5-coder) needs a
/// concrete role, not a wall of instructions.
fn build_system_prompt() -> String {
    "You are a concise assistant inside MLPL, a Rust-first array \
     programming language for machine learning inspired by APL \
     and J. Answer questions about the user's current MLPL \
     session briefly (under 200 words, plain prose). If the \
     user asks about code, explain what it does in MLPL terms \
     (label propagation, shape discipline, scoped forms like \
     `device(\"mlx\") { }`, `experiment \"name\" { }`, \
     `train N { }`). Never fabricate builtins; if you are not \
     sure a name exists, say so."
        .into()
}

/// User-message context: the question plus a compact snapshot
/// of the current workspace, assembled from the same inspectors
/// `:vars` / `:models` / `:experiments` already expose. The
/// `_demo` string (if bound) gets prepended so "what did this
/// demo do?" questions have the narration text as grounding.
fn build_user_context(env: &Environment, question: &str) -> String {
    let mut parts = Vec::new();
    if let Some(demo) = env.get_string("_demo") {
        parts.push(format!("Current demo:\n{demo}\n"));
    }
    let vars_summary = var_summary(env);
    if !vars_summary.is_empty() {
        parts.push(format!("Workspace variables:\n{vars_summary}"));
    }
    parts.push(format!("Question: {question}"));
    parts.join("\n\n")
}

/// Compact one-line-per-var summary. Shows shape + "[param]"
/// tag for trainables; keeps values out (a large tensor would
/// blow past any reasonable context window). Capped at 40
/// entries; the LLM does not need to see every intermediate.
fn var_summary(env: &Environment) -> String {
    let mut pairs: Vec<(String, String, bool)> = env
        .vars_iter()
        .map(|(name, arr)| {
            let shape = format!("{:?}", arr.shape().dims());
            (name.clone(), shape, env.is_param(name))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let total = pairs.len();
    let mut out = String::new();
    for (name, shape, is_param) in pairs.iter().take(40) {
        let tag = if *is_param { " [param]" } else { "" };
        out.push_str(&format!("  {name}: {shape}{tag}\n"));
    }
    if total > 40 {
        out.push_str(&format!("  ... ({} more)\n", total - 40));
    }
    out.truncate(out.trim_end().len());
    out
}
