//! Split an editor Run group into NARRATION (its leading
//! full-line `#` comments, markers stripped) and CODE (the rest,
//! verbatim -- interior comments inside a multi-line construct
//! stay put). Narration renders as the demo-style prose block;
//! code evaluates as the transcript's input line.

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
