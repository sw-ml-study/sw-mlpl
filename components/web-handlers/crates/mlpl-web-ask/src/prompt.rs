//! The `:ask` system prompt and program builder. Browser-only (the
//! context readers eval JS against the live page); native builds
//! compile the module away, matching the origin `connect.rs` gating.

#![cfg(target_arch = "wasm32")]

use crate::prompt_text::{ASK_SYSTEM, mlpl_reference};
use mlpl_web_eval::state::{EntryKind, HistoryEntry};

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
    // Last words win with weak local models: restate the code rules
    // observed being violated (kwargs, in-fn models, echo artifacts).
    p.push_str(crate::prompt_text::STRICT_CODE_RECAP);
    p
}
