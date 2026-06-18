//! Tiny markdown-subset renderer for `Lesson.intro` blocks.
//!
//! Lesson intros used to render as one wall of plain text.
//! The "How models learn" lesson hit this limit immediately:
//! a single ~200-word paragraph was hard to scan even with
//! em-dash setup. This renderer adds just enough structure
//! to break those walls without pulling in a full markdown
//! crate:
//!
//! - `## heading` (line-level)         -> `<h3 class="lesson-section">`
//! - `- bullet item` (line-level)      -> contributes to `<ul class="lesson-list">`
//! - `---` (line-level, alone)         -> `<hr class="lesson-separator"/>`
//! - blank line                        -> paragraph break
//! - `**bold**` (inline)               -> `<strong>`
//! - `*emph*` (inline, word-boundary)  -> `<em>`
//! - `` `code` `` (inline)             -> `<code>`
//! - everything else                   -> `<p class="lesson-paragraph">`
//!
//! `*emph*` opens only at a word boundary (start, or after whitespace /
//! an opening bracket), so mid-word asterisks stay literal -- e.g. the
//! `Q*_K` / `IQ*` quant-name wildcards are NOT turned into italics.
//!
//! Lessons that don't use any markers render exactly as
//! before: one paragraph, no surprise styling. Authors opt
//! in incrementally.
//!
//! HTML escaping: `<`, `>`, `&` are escaped everywhere
//! (including inside inline code) since lesson content
//! routinely uses things like `[T, T]` and `Q @ K^T` that
//! would otherwise look like garbled tags.

use yew::AttrValue;
use yew::prelude::Html;

/// Parse a small markdown subset and return rendered HTML.
pub fn render(intro: &str) -> Html {
    let mut out = String::new();
    let mut para: Vec<String> = Vec::new();
    let mut bullets: Vec<String> = Vec::new();
    for line in intro.lines() {
        let t = line.trim();
        if t.is_empty() {
            flush_para(&mut out, &mut para);
            flush_bullets(&mut out, &mut bullets);
        } else if let Some(rest) = t.strip_prefix("## ") {
            flush_para(&mut out, &mut para);
            flush_bullets(&mut out, &mut bullets);
            out.push_str("<h3 class=\"lesson-section\">");
            out.push_str(&inline(rest));
            out.push_str("</h3>");
        } else if t == "---" {
            flush_para(&mut out, &mut para);
            flush_bullets(&mut out, &mut bullets);
            out.push_str("<hr class=\"lesson-separator\"/>");
        } else if let Some(rest) = t.strip_prefix("- ") {
            flush_para(&mut out, &mut para);
            bullets.push(inline(rest));
        } else {
            flush_bullets(&mut out, &mut bullets);
            para.push(inline(t));
        }
    }
    flush_para(&mut out, &mut para);
    flush_bullets(&mut out, &mut bullets);
    Html::from_html_unchecked(AttrValue::from(out))
}

fn flush_para(out: &mut String, p: &mut Vec<String>) {
    if !p.is_empty() {
        out.push_str("<p class=\"lesson-paragraph\">");
        out.push_str(&p.join(" "));
        out.push_str("</p>");
        p.clear();
    }
}

fn flush_bullets(out: &mut String, b: &mut Vec<String>) {
    if !b.is_empty() {
        out.push_str("<ul class=\"lesson-list\">");
        for item in b.iter() {
            out.push_str("<li>");
            out.push_str(item);
            out.push_str("</li>");
        }
        out.push_str("</ul>");
        b.clear();
    }
}

/// Inline-format one segment: escape `<` `>` `&`, recognize
/// `**bold**` and `` `code` ``. Bold contents are recursively
/// inline-formatted (so escaping stays consistent); code
/// contents are only HTML-escaped (no nested formatting).
pub(crate) fn inline(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut prev: Option<char> = None;
    while let Some(c) = chars.next() {
        let at_boundary = prev.is_none_or(|p| p.is_whitespace() || "([{\"'".contains(p));
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '*' if chars.peek() == Some(&'*') => {
                chars.next();
                let inner = inline(&scan_to(&mut chars, '*', true));
                out.push_str(&format!("<strong>{inner}</strong>"));
            }
            '*' if at_boundary => {
                let inner = inline(&scan_to(&mut chars, '*', false));
                out.push_str(&format!("<em>{inner}</em>"));
            }
            '`' => out.push_str(&format!(
                "<code>{}</code>",
                escape_html(&scan_to(&mut chars, '`', false))
            )),
            _ => out.push(c),
        }
        prev = Some(c);
    }
    out
}

/// Consume chars from `chars` up to and including the next
/// matching delimiter. With `double = true`, the closer is
/// `delim delim` (e.g. `**`); a single `delim` followed by
/// non-delim is treated as a literal.
fn scan_to(chars: &mut std::iter::Peekable<std::str::Chars>, delim: char, double: bool) -> String {
    let mut buf = String::new();
    while let Some(&nc) = chars.peek() {
        chars.next();
        if nc == delim {
            if !double {
                return buf;
            }
            if chars.peek() == Some(&delim) {
                chars.next();
                return buf;
            }
            buf.push(delim);
        } else {
            buf.push(nc);
        }
    }
    buf
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}
