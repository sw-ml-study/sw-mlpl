//! `split_inline_comment` -- splits a tutorial-example line
//! on the first unescaped `#` into `(code, Option<comment>)`.
//! Moved from mlpl-web/src/entry_render.rs during saga 82 so
//! the tutorial crate doesn't pull in render code; entry_render
//! (still in mlpl-web) routes through this public function.

/// Split a statement GROUP (an entry, which may be several lines) into
/// one `(code, comment)` pair PER LINE. `split_inline_comment` is a
/// per-LINE contract: applied to a whole multi-line group it splits at
/// the FIRST `#`, so a leading full-line `# comment` swallows the code on
/// every line after it into the comment span (upstream-asks #16). An
/// empty group still yields one empty line so the `mlpl>` prompt renders.
pub fn split_group_lines(group: &str) -> Vec<(&str, Option<&str>)> {
    if group.is_empty() {
        return vec![("", None)];
    }
    group.lines().map(split_inline_comment).collect()
}

pub fn split_inline_comment(line: &str) -> (&str, Option<&str>) {
    let mut in_str = false;
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_str = !in_str,
            b'#' if !in_str => {
                let code = line[..i].trim_end();
                let comment = line[i + 1..].trim();
                let comment_opt = if comment.is_empty() {
                    None
                } else {
                    Some(comment)
                };
                return (code, comment_opt);
            }
            _ => {}
        }
    }
    (line, None)
}
