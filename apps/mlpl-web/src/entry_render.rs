//! Rendering helpers for REPL/tutorial history entries.
//! Extracted from main.rs to keep that module under the
//! sw-checklist file-LOC budget.

use yew::prelude::*;

use crate::state::{EntryKind, HistoryEntry};
use crate::summary;

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let safe = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Split an MLPL line at the first inline `#` comment that
/// is not inside a string literal. Returns `(code, comment)`
/// with comment as `Some(text)` when present (without the
/// leading `#`, trimmed). MLPL's parser already drops `#`
/// comments; this is purely for the UI to render the
/// commentary as an annotation alongside the code.
pub(crate) fn split_inline_comment(line: &str) -> (&str, Option<&str>) {
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

fn render_input_line(input: &str) -> Html {
    let (code, comment) = split_inline_comment(input);
    let comment_html = match comment {
        Some(c) => {
            html! { <span class="line-comment">{ format!(" # {c}") }</span> }
        }
        None => html! {},
    };
    html! {
        <div class="input-line">
            <span class="prompt">{"mlpl> "}</span>
            { code }
            { comment_html }
        </div>
    }
}

fn render_svg_body(svg: &str) -> Html {
    let svg_html = Html::from_html_unchecked(AttrValue::from(svg.to_string()));
    let href = format!("data:image/svg+xml;charset=utf-8,{}", percent_encode(svg));
    html! {
        <div class="svg-output">
            { svg_html }
            <a class="svg-download" href={href} download="mlpl.svg" title="Download SVG" aria-label="Download SVG">{"\u{2b07}"}</a>
        </div>
    }
}

pub(crate) fn render_entry(entry: &HistoryEntry) -> Html {
    if entry.kind == EntryKind::Narration {
        // Demo narration: prose framing around the code output.
        // No `mlpl>` prompt, no output pre-formatting; `input` is
        // the heading (e.g. "About this demo" / "What just
        // happened"), `output` is the narration body.
        return html! {
            <div class="narration">
                <div class="narration-heading">{ &entry.input }</div>
                <div class="narration-body">{ &entry.output }</div>
            </div>
        };
    }
    let body = if !entry.is_error && entry.output.trim_start().starts_with("<svg") {
        render_svg_body(&entry.output)
    } else if entry.is_error {
        html! { <pre class={"output-line error"}>{ &entry.output }</pre> }
    } else if let Some(s) = summary::summarize(&entry.output) {
        let summary_text = format!(
            "{}  min={}  max={}  mean={}  median={}  std={}",
            s.shape,
            summary::fmt_stat(s.min),
            summary::fmt_stat(s.max),
            summary::fmt_stat(s.mean),
            summary::fmt_stat(s.median),
            summary::fmt_stat(s.std),
        );
        html! {
            <details class="output-summary">
                <summary>{ summary_text }</summary>
                <pre class={"output-line"}>{ &entry.output }</pre>
            </details>
        }
    } else {
        html! { <pre class={"output-line"}>{ &entry.output }</pre> }
    };
    html! {
        <div class="entry">
            { render_input_line(&entry.input) }
            { body }
        </div>
    }
}
