//! Inline-span (backtick code, **bold**, _emph_) parsing
//! for the markdown-ish path-body renderer.

use yew::prelude::*;

#[derive(Debug)]
pub(crate) enum Span {
    Text(String),
    Code(String),
    Bold(String),
    Emph(String),
}

pub(crate) fn render_span(span: Span) -> Html {
    match span {
        Span::Text(s) => html! { { s } },
        Span::Code(s) => html! { <code class="path-body-code">{ s }</code> },
        Span::Bold(s) => html! { <strong>{ s }</strong> },
        Span::Emph(s) => html! { <em>{ s }</em> },
    }
}

/// Tokenize inline span markers. Greedy + first-match wins;
/// no nesting (a span runs from delimiter to its mate, and
/// the inner text is emitted as a literal string).
pub(crate) fn split(text: &str) -> Vec<Span> {
    let bytes = text.as_bytes();
    let mut spans: Vec<Span> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some((span, advance)) = try_match_span(bytes, i, c) {
            if !buf.is_empty() {
                spans.push(Span::Text(std::mem::take(&mut buf)));
            }
            spans.push(span);
            i += advance;
            continue;
        }
        buf.push(c as char);
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::Text(buf));
    }
    spans
}

fn try_match_span(bytes: &[u8], i: usize, c: u8) -> Option<(Span, usize)> {
    if c == b'`' {
        return match_until(bytes, i + 1, b'`').map(|(s, end)| (Span::Code(s), end - i));
    }
    if c == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
        return match_until_pair(bytes, i + 2, b'*', b'*').map(|(s, end)| (Span::Bold(s), end - i));
    }
    if c == b'_'
        && let Some((s, end)) = match_until(bytes, i + 1, b'_')
    {
        return Some((Span::Emph(s), end - i));
    }
    None
}

fn match_until(bytes: &[u8], start: usize, delim: u8) -> Option<(String, usize)> {
    let mut j = start;
    while j < bytes.len() {
        if bytes[j] == delim {
            let s = std::str::from_utf8(&bytes[start..j]).ok()?;
            return Some((s.to_string(), j + 1));
        }
        j += 1;
    }
    None
}

fn match_until_pair(bytes: &[u8], start: usize, d1: u8, d2: u8) -> Option<(String, usize)> {
    let mut j = start;
    while j + 1 < bytes.len() {
        if bytes[j] == d1 && bytes[j + 1] == d2 {
            let s = std::str::from_utf8(&bytes[start..j]).ok()?;
            return Some((s.to_string(), j + 2));
        }
        j += 1;
    }
    None
}
