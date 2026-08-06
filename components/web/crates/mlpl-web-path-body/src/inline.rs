//! Inline-span (backtick code, **bold**, _emph_,
//! `[[glossary-term]]`) parsing for the markdown-ish
//! path-body renderer. Rendering side lives in
//! `inline_render.rs`.

#[derive(Debug)]
pub(crate) enum Span {
    Text(String),
    Code(String),
    Bold(String),
    Emph(String),
    /// Saga 29 step 024: `[[term]]` sigil. The renderer
    /// turns this into a clickable button that dispatches
    /// a window-level `mlpl-glossary-open` CustomEvent;
    /// the top-level `GlossaryPopupHost` component
    /// listens for that event and opens the popup.
    Glossary(String),
    /// `[label](url)` -- a plain hyperlink (opens in a new tab).
    /// Used e.g. to link a demo's intro to its literate walkthrough.
    Link {
        text: String,
        url: String,
    },
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
        // Accumulate WHOLE characters: all span delimiters are
        // ASCII, but the text between them may be any UTF-8
        // (APL glyphs in narration) -- a byte-as-char push
        // shatters multi-byte sequences into mojibake.
        let ch = text[i..].chars().next().expect("in-bounds char");
        buf.push(ch);
        i += ch.len_utf8();
    }
    if !buf.is_empty() {
        spans.push(Span::Text(buf));
    }
    spans
}

fn try_match_span(bytes: &[u8], i: usize, c: u8) -> Option<(Span, usize)> {
    if c == b'[' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
        return match_until_pair(bytes, i + 2, b']', b']')
            .map(|(s, end)| (Span::Glossary(s), end - i));
    }
    // `[label](url)` -- a single `[` (not the `[[` glossary sigil)
    // followed by `](`. A "url" containing whitespace is prose
    // ("[a, b] (note)" punctuation), not a link.
    if c == b'['
        && let Some((text, after)) = match_until(bytes, i + 1, b']')
        && after < bytes.len()
        && bytes[after] == b'('
        && let Some((url, end)) = match_until(bytes, after + 1, b')')
        && !url.contains(char::is_whitespace)
    {
        return Some((Span::Link { text, url }, end - i));
    }
    if c == b'`' {
        return match_until(bytes, i + 1, b'`').map(|(s, end)| (Span::Code(s), end - i));
    }
    if c == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
        return match_until_pair(bytes, i + 2, b'*', b'*').map(|(s, end)| (Span::Bold(s), end - i));
    }
    // `_emph_` only at WORD BOUNDARIES (CommonMark rule): a `_`
    // inside snake_case is literal, so identifiers in prose
    // survive verbatim (the transpose_axes class bug).
    if c == b'_'
        && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
        && let Some((s, end)) = match_emph_close(bytes, i + 1)
    {
        return Some((Span::Emph(s), end - i));
    }
    None
}

/// Find the closing `_` of an emphasis span: the first `_` NOT
/// followed by an alphanumeric (an intra-word `_` cannot close).
fn match_emph_close(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let mut j = start;
    while j < bytes.len() {
        if bytes[j] == b'_' && (j + 1 == bytes.len() || !bytes[j + 1].is_ascii_alphanumeric()) {
            let s = std::str::from_utf8(&bytes[start..j]).ok()?;
            return Some((s.to_string(), j + 1));
        }
        j += 1;
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
