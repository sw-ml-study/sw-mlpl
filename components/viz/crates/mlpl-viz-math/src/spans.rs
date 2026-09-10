//! Inline parsing: convert `_`/`^` runs in an equation line into
//! sub/superscript spans, and emit them (XML-escaped) as `<tspan>`s.

use std::iter::Peekable;
use std::str::Chars;

/// Baseline placement of an inline run.
pub(crate) enum Kind {
    Normal,
    Sub,
    Sup,
}

/// A contiguous run of the line rendered at one baseline.
pub(crate) struct Span {
    pub text: String,
    pub kind: Kind,
}

/// Split a line into spans: `_x`/`^x` (single char) and `_{...}`/`^{...}`
/// (braced) become Sub/Sup; everything else is a Normal run.
pub(crate) fn spans(line: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let push_normal = |out: &mut Vec<Span>, buf: &mut String| {
        if !buf.is_empty() {
            out.push(Span {
                text: std::mem::take(buf),
                kind: Kind::Normal,
            });
        }
    };
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        let kind = match c {
            '_' => Kind::Sub,
            '^' => Kind::Sup,
            _ => {
                buf.push(c);
                continue;
            }
        };
        push_normal(&mut out, &mut buf);
        out.push(Span {
            text: take_script(&mut chars),
            kind,
        });
    }
    push_normal(&mut out, &mut buf);
    out
}

/// Read a script argument: a `{...}` group or a single following char.
pub(crate) fn take_script(chars: &mut Peekable<Chars>) -> String {
    if chars.peek() == Some(&'{') {
        chars.next();
        return chars.by_ref().take_while(|&c| c != '}').collect();
    }
    chars.next().map(String::from).unwrap_or_default()
}

/// Emit a span as a `<tspan>`, shifting and shrinking sub/superscripts.
/// XML-escapes the three characters that matter inside `<text>`.
pub(crate) fn push_span(out: &mut String, span: &Span) {
    let esc = span
        .text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    match span.kind {
        Kind::Normal => out.push_str(&format!("<tspan>{esc}</tspan>")),
        Kind::Sub => out.push_str(&format!(
            "<tspan baseline-shift=\"sub\" font-size=\"0.7em\">{esc}</tspan>"
        )),
        Kind::Sup => out.push_str(&format!(
            "<tspan baseline-shift=\"super\" font-size=\"0.7em\">{esc}</tspan>"
        )),
    }
}
