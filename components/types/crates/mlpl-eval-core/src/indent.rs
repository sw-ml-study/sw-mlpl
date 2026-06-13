//! A simple brace/`;`-aware indenter for MLPL source text.
//!
//! `Display for Expr` renders flat (`def u:f(x) { a; if c { b } else { d }
//! }`), which is unreadable for any function with control flow. This
//! re-indents that string: one 4-space level per `{`, a line break after
//! each in-block `;`, and `}` dedented onto its own line. String literals
//! are copied verbatim so `{`/`}`/`;` inside them are not treated as
//! structure. A display aid (used by `:list <fn>` and the 3D step view),
//! not a full formatter. Operates purely on text, so it lives here rather
//! than in the parser.

use std::iter::Peekable;
use std::str::Chars;

const STEP: &str = "    ";

/// Copy a double-quoted string literal verbatim (opening `"` consumed),
/// honoring `\"`/`\\` escapes, through the closing quote.
fn copy_str_lit(out: &mut String, chars: &mut Peekable<Chars>) {
    out.push('"');
    while let Some(c) = chars.next() {
        out.push(c);
        match c {
            '\\' => {
                if let Some(e) = chars.next() {
                    out.push(e);
                }
            }
            '"' => break,
            _ => {}
        }
    }
}

/// End the current line (dropping its trailing spaces) and start a fresh
/// one indented to `depth`.
fn line(out: &mut String, depth: usize) {
    while out.ends_with(' ') {
        out.pop();
    }
    out.push('\n');
    for _ in 0..depth {
        out.push_str(STEP);
    }
}

/// Emit a brace and re-indent: `open` adds a level and breaks after `{`;
/// otherwise dedent first, then place `}` on its own fresh line.
fn brace(out: &mut String, depth: &mut usize, open: bool) {
    if open {
        *depth += 1;
        out.push('{');
        line(out, *depth);
    } else {
        *depth = depth.saturating_sub(1);
        line(out, *depth);
        out.push('}');
    }
}

/// Re-indent flat MLPL `src`. The preceding source space (collapsed by the
/// `' '` arm) is what separates `head` from its `{`.
#[must_use]
pub fn indent_source(src: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => copy_str_lit(&mut out, &mut chars),
            '{' => brace(&mut out, &mut depth, true),
            '}' => brace(&mut out, &mut depth, false),
            ';' => {
                out.push(';');
                line(&mut out, depth);
            }
            ' ' if out.ends_with([' ', '\n']) => {}
            _ => out.push(c),
        }
    }
    out.trim_end().to_string()
}
