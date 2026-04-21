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
}

#[derive(Clone, Copy, PartialEq)]
pub enum DocTab {
    LangReference,
    Usage,
}
