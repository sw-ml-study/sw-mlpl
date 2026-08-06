//! Group editor text into complete statements for Run: lines
//! buffer until every bracket opened outside a string literal or
//! `#` comment is closed, so multi-line `def` / `if` / record
//! constructs evaluate WHOLE while the transcript still gets one
//! entry (and one viz) per statement.

/// Split `text` into balanced statement groups. Blank lines and
/// full-line `#` comments vanish; a trailing unbalanced group is
/// still emitted (the evaluator reports the unclosed delimiter).
#[must_use]
pub fn group_statements(text: &str) -> Vec<String> {
    let mut groups: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut depth: i64 = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(trimmed);
        depth += bracket_delta(trimmed);
        if depth <= 0 {
            groups.push(std::mem::take(&mut buf));
            depth = 0;
        }
    }
    if !buf.is_empty() {
        groups.push(buf);
    }
    groups
}

/// Net bracket depth of one line, ignoring string literals and
/// everything after an (unquoted) `#` comment marker.
fn bracket_delta(line: &str) -> i64 {
    let (mut depth, mut in_str, mut escaped) = (0i64, false, false);
    for c in line.chars() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '#' => break,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    depth
}
