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

/// Split an editor Run group into NARRATION (its leading
/// full-line `#` comments, markers stripped) and CODE (the
/// rest, verbatim -- interior comments inside a multi-line
/// construct stay put). Narration renders as the demo-style
/// prose block; code evaluates as the transcript's input line.
/// Either side may be absent: a comment-only group (a document's
/// closing summary) has no code; ordinary input has no narration.
#[must_use]
pub fn split_leading_comments(group: &str) -> (Option<String>, Option<String>) {
    let mut narration: Vec<&str> = Vec::new();
    let mut rest: Vec<&str> = Vec::new();
    let mut in_code = false;
    for line in group.lines() {
        let t = line.trim();
        if !in_code && t.starts_with('#') {
            narration.push(t.trim_start_matches('#').trim_start());
        } else if !t.is_empty() || in_code {
            in_code = true;
            rest.push(line);
        }
    }
    let join = |v: Vec<&str>| {
        if v.is_empty() {
            None
        } else {
            Some(v.join("\n"))
        }
    };
    (join(narration), join(rest))
}
