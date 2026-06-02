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
        buf.push(c as char);
        i += 1;
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
    // followed by `](`. Reuses `match_until` for both halves.
    if c == b'['
        && let Some((text, after)) = match_until(bytes, i + 1, b']')
        && after < bytes.len()
        && bytes[after] == b'('
        && let Some((url, end)) = match_until(bytes, after + 1, b')')
    {
        return Some((Span::Link { text, url }, end - i));
    }
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

#[cfg(test)]
mod tests {
    use super::{Span, split};

    #[test]
    fn glossary_sigil_round_trips() {
        // Core: [[term]] -> Span::Glossary("term"). Also
        // covers leading text + trailing text segmenting.
        let spans = split("see [[take (builtin)]] for more");
        assert_eq!(spans.len(), 3);
        assert!(matches!(&spans[0], Span::Text(t) if t == "see "));
        assert!(matches!(&spans[1], Span::Glossary(t) if t == "take (builtin)"));
        assert!(matches!(&spans[2], Span::Text(t) if t == " for more"));
    }

    #[test]
    fn unterminated_or_single_bracket_stays_text() {
        // `[term]` (single) and `[[unterminated` (no closer)
        // both render as literal text rather than crashing.
        assert!(matches!(&split("[not a link]")[0], Span::Text(_)));
        assert!(matches!(&split("[[unterminated")[0], Span::Text(_)));
    }

    #[test]
    fn markdown_link_parses() {
        // [label](url) -> Span::Link; surrounding text segments.
        let spans = split("see [the doc](literate/x.html) now");
        assert!(matches!(&spans[0], Span::Text(t) if t == "see "));
        assert!(
            matches!(&spans[1], Span::Link { text, url } if text == "the doc" && url == "literate/x.html")
        );
        assert!(matches!(&spans[2], Span::Text(t) if t == " now"));
        // A bare [bracketed] with no (url) stays literal text.
        assert!(matches!(&split("[just text]")[0], Span::Text(_)));
    }

    #[test]
    fn code_and_glossary_coexist() {
        // The dispatcher must not confuse `code` with [[term]].
        let spans = split("`code` and [[Stack (tape op)]]");
        assert!(matches!(&spans[0], Span::Code(t) if t == "code"));
        assert!(matches!(&spans[2], Span::Glossary(t) if t == "Stack (tape op)"));
    }
}
