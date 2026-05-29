//! `split_inline_comment` -- splits a tutorial-example line
//! on the first unescaped `#` into `(code, Option<comment>)`.
//! Moved from mlpl-web/src/entry_render.rs during saga 82 so
//! the tutorial crate doesn't pull in render code; entry_render
//! (still in mlpl-web) routes through this public function.

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
