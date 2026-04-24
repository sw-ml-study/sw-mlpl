//! `:ask <question>` REPL command.
//!
//! Saga 19 step 002: the HTTP path now lives in
//! `mlpl_runtime::call_ollama` (the same path the
//! language-level `llm_call` builtin uses). `:ask`
//! still owns the workspace-aware framing: a system
//! prompt that tells the model what MLPL is, plus a
//! user-context message built from `:vars` /
//! `:models` / the optional `_demo` narration. That
//! framing gets concatenated into a single prompt
//! and sent through the `/api/generate` endpoint --
//! the model loses the role distinction it would
//! get from `/api/chat`, but the context is
//! preserved.
//!
//! TODO(saga-19-followup): a future `llm_chat(history,
//! prompt)` variant -- listed in step 001's deferred
//! non-goals -- would let `:ask` keep the
//! system+user role pair via `/api/chat` while still
//! sharing the underlying HTTP machinery. Not
//! shipping today.
//!
//! `OLLAMA_HOST` (default `http://localhost:11434`)
//! and `OLLAMA_MODEL` (default `llama3.2`) override
//! the endpoint. CLI-only -- the web REPL stays out
//! of this path because (i) CORS requires
//! `OLLAMA_ORIGINS` be set before the browser's fetch
//! is allowed, and (ii) Saga 21 ships the proper
//! server-side proxy.

use mlpl_eval::Environment;

const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "llama3.2";

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
    let prompt = build_prompt(env, question);
    match mlpl_runtime::call_ollama(&host, &prompt, &model) {
        Ok(answer) => println!("{}", answer.trim_end()),
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("  :ask needs a running Ollama server at {host}.");
            eprintln!("  Start one with: ollama serve && ollama pull {model}");
            eprintln!("  Override host/model with OLLAMA_HOST / OLLAMA_MODEL env vars.");
        }
    }
}

/// Concatenate the system framing and the user
/// context into a single `/api/generate` prompt
/// string. Order: system block first so the model
/// sees the role definition before the workspace
/// dump and the question.
fn build_prompt(env: &Environment, question: &str) -> String {
    let system = build_system_prompt();
    let user = build_user_context(env, question);
    format!("{system}\n\n{user}")
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
