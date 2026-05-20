#[derive(Clone, PartialEq)]
pub struct HistoryEntry {
    pub input: String,
    pub output: String,
    pub is_error: bool,
    /// When true, this is a narration panel (demo intro or
    /// takeaway), not an MLPL command + result. The renderer
    /// drops the `mlpl>` prompt and styles it as prose.
    pub kind: EntryKind,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EntryKind {
    /// Regular REPL input + output pair.
    Command,
    /// Demo narration (intro before the run, takeaway after).
    Narration,
    /// Saga 29 step 018: "this line is currently evaluating"
    /// placeholder. Pushed before a blocking eval so the
    /// browser paints a CSS-animated spinner while WASM is
    /// busy; replaced with a `Command` entry once the eval
    /// returns. The CSS animation continues even while the
    /// JS thread is blocked (the browser compositor handles
    /// it), so the user gets a visible "still alive"
    /// indicator without needing Web Workers.
    Running,
}

#[derive(Clone, Copy, PartialEq)]
pub enum DocTab {
    LangReference,
    Usage,
    Glossary,
    Diagrams,
}
