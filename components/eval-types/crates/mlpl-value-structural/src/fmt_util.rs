//! Small shared formatting helpers for the repr renderer.

pub(crate) fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars().take(120) {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    if s.chars().count() > 120 {
        out.push_str("...");
    }
    out.push('"');
}

pub(crate) fn join(out: &mut String, n: usize, mut f: impl FnMut(&mut String, usize)) {
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        f(out, i);
    }
}
