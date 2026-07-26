//! The `:ask` system prompt and program builder. Browser-only (the
//! context readers eval JS against the live page); native builds
//! compile the module away, matching the origin `connect.rs` gating.

#![cfg(target_arch = "wasm32")]

use mlpl_web_eval::state::{EntryKind, HistoryEntry};

/// System/meta preamble prepended to every `:ask` so even broad
/// questions ("describe this environment") are grounded in what
/// MLPL is and what context the user can surface, rather than the
/// model guessing about generic OS/web environments.
const ASK_SYSTEM: &str = "You are an assistant embedded INSIDE the sw-MLPL REPL -- an APL/J/BQN-inspired array and tensor language for machine learning, with a 3D visualization playground (the REPL renders each result as a 3D sculpture: tensors as grids/bars, attention as heatmaps, models as Sankey diagrams). You are NOT a generic cloud/AWS/web assistant; EVERY question is about sw-MLPL. \
CRITICAL: Do NOT invent, guess, or hallucinate commands, syntax, model names, or features. If you are unsure, say so plainly and tell the user to run `:help`. The ONLY REPL commands are exactly these -- never claim any other exists: :help, :help <topic>, :<cmd> --help, :vars, :models, :tokenizers, :fns, :builtins, :describe <name>, :history, :experiments, :wsid, :status, :status watch, :reset, :ask <prompt>, :connect list, :connect set <model>, :3d, :2d, :clear, :upload <name>. There is NO :load_model, NO :model, NO :search. \
DO NOT WRITE MLPL CODE unless you are CERTAIN of the exact syntax (ideally only code you can see in this session's context): MLPL is an array language, NOT Python -- it has NO `+=`/`-=`, NO `==` (use `eq(a, b)`), NO lambdas or `->`, NO `filter`/`map`/`print`/`length`/`strcat`/`append`, and `device(\"...\")`, `experiment`, and `train`/`repeat` ALWAYS take a `{ ... }` block (a bare `device(\"mlx\")` is a syntax error). If you are not sure code will run, DESCRIBE the approach in plain words and tell the user to read the demo/its literate walkthrough and `:help` -- never emit a code block you have not actually seen run. When the user asks how to do something in the REPL: for command help tell them to run `:help` (lists all commands) or `:<cmd> --help` (help for one command, e.g. `:ask --help`); to see or change which LLM answers their `:ask`, tell them `:connect list` (lists the installed Ollama models) then `:connect set <name>`. \
The user's recent REPL activity and any selected 3D sculpture are provided below as your context -- use them. Answer concisely and specifically about sw-MLPL.";

/// Compact MLPL syntax cheat-sheet prepended to the builtin signatures in
/// [`mlpl_reference`]. The CORRECT forms (vs the Python-isms the prompt
/// forbids), so when code is warranted the model writes valid MLPL.
const MLPL_SYNTAX: &str = " MLPL quick reference -- use EXACTLY these forms. \
Assign `name = expr`; comment `# ...`; compare with `eq(a,b)` / `gt(a,b)` / `lt(a,b)` (there is no `==`/`<`/`>`); \
EVERY block uses braces: `device(\"mlx\") { ... }`, `experiment \"name\" { ... }`, `train N { ... }`, `repeat N { ... }`, `if c { ... } else { ... }`; \
define a function with `def u:name(a, b) { body }` -- the `u:` prefix is REQUIRED and there is NO `return` (the block's last expression is its value) -- then call it `u:name(args)`; iterate with `for x in iota(n) { ... }` or `while cond { ... }`; index/slice a tensor with `take(x, axis, i)`. \
Statements inside a block are separated by `;`. The COMPLETE builtin set follows (call ONLY these exact signatures -- there is no filter/map/print/length/strcat/append):";

/// Build the full `:ask` program: the question as the user prompt,
/// with the grounding/context riding in `llm_call`'s `system` field.
#[must_use]
pub fn ask_program(question: &str, history: &[HistoryEntry]) -> String {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let question = esc(question.trim().trim_matches('"').trim());
    let system = esc(&build_ask_system(history));
    let (url, model) = crate::context::ask_endpoint();
    format!("llm_call(\"{url}\", \"{question}\", \"{model}\", \"{system}\")")
}

/// The compact MLPL reference for the `:ask` system prompt: the syntax
/// cheat-sheet plus every builtin's signature, grouped, sourced from the
/// curated `BUILTIN_GROUPS` table so it never drifts from the real set.
fn mlpl_reference() -> String {
    let mut r = MLPL_SYNTAX.to_string();
    for (group, entries) in mlpl_eval_core::inspect_groups::BUILTIN_GROUPS {
        let sigs: Vec<&str> = entries.iter().map(|&(_, sig, _)| sig).collect();
        r.push_str(&format!(" [{group}] {}.", sigs.join(", ")));
    }
    r
}

/// Summarize the recent REPL activity (last few command/result
/// pairs) so `:ask` can answer questions about "what is being run
/// in the REPL" -- an in-context REPL assistant. Newest entries
/// are kept; long outputs are truncated char-safely.
fn repl_history_context(history: &[HistoryEntry]) -> String {
    let mut recent: Vec<String> = history
        .iter()
        .rev()
        .filter(|e| matches!(e.kind, EntryKind::Command))
        .take(6)
        .map(|e| {
            let out: String = e.output.trim().chars().take(180).collect();
            format!("mlpl> {} => {}", e.input.trim(), out)
        })
        .collect();
    recent.reverse();
    recent.join(" | ")
}

/// Ground the model in the demos run THIS session. Demos now run in
/// the same REPL (the transcript is not cleared between them), so
/// every "About this demo -- <name>" narration is listed oldest
/// first, and the newest one -- the active demo -- also gets its
/// intro body, so the model doesn't guess (e.g. Othello for
/// tic-tac-toe) and can answer "what has been done in this REPL"
/// across demos.
fn demo_context(history: &[HistoryEntry]) -> String {
    let intros: Vec<&HistoryEntry> = history
        .iter()
        .filter(|e| {
            matches!(e.kind, EntryKind::Narration) && e.input.starts_with("About this demo")
        })
        .collect();
    let Some(active) = intros.last() else {
        return String::new();
    };
    let mut p = String::new();
    if intros.len() > 1 {
        let names: Vec<&str> = intros
            .iter()
            .map(|e| e.input.trim_start_matches("About this demo -- ").trim())
            .collect();
        p.push_str(&format!(
            " Demos run this session, oldest first (earlier ones' variables and results are still in the workspace): {}.",
            names.join(", ")
        ));
    }
    let body: String = active.output.trim().chars().take(400).collect();
    p.push_str(&format!(" Active demo -- {}: {body}.", active.input.trim()));
    p
}

/// Build the `:ask` system message: meta preamble + recent REPL
/// activity + the selected sculpture (if any). This goes in
/// Ollama's `system` role (not the prompt), which weak models
/// follow far better. The question is sent as the user prompt.
fn build_ask_system(history: &[HistoryEntry]) -> String {
    let mut p = ASK_SYSTEM.to_string();
    // Compact MLPL reference (syntax + builtin signatures) -- real forms.
    p.push_str(&mlpl_reference());
    p.push_str(&demo_context(history));
    let recent = repl_history_context(history);
    if !recent.is_empty() {
        p.push_str(&format!(" Recent REPL activity (oldest first): {recent}."));
    }
    let sel = crate::context::selection_context();
    if !sel.is_empty() {
        p.push_str(&format!(" Selected 3D sculpture: {sel}."));
    }
    p
}
