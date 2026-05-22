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
    // Saga 29 step 025: route the comment text through the markdown-ish renderer so
    // `[[term]]` inside an MLPL `# comment` becomes a clickable glossary link. Code (the
    // part before `#`) stays a plain string -- the lexer parses MLPL source there and
    // [[ would be ambiguous with array literals.
    let comment_html = match comment {
        Some(c) => html! {
            <span class="line-comment">
                {" # "}
                { crate::path_body::render_inline(c) }
            </span>
        },
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
        //
        // Saga 29 step 022 + 024: route the body through the
        // path_body markdown-ish renderer so paragraph splits,
        // bullets, **bold**, `code`, and `[[glossary-term]]`
        // links all work in narration bodies the same way
        // they do in learning-path Note bodies.
        return html! {
            <div class="narration">
                <div class="narration-heading">{ &entry.input }</div>
                <div class="narration-body">{ crate::path_body::render_body(&entry.output) }</div>
            </div>
        };
    }
    if entry.kind == EntryKind::Running {
        // Saga 29 step 018: "this line is currently
        // evaluating" placeholder. The CSS spinner keeps
        // animating during the JS-blocking WASM eval because
        // CSS animations run on the browser compositor, not
        // the JS thread. Replaced by a `Command` entry when
        // the eval returns.
        return html! {
            <div class="entry running">
                { render_input_line(&entry.input) }
                <div class="output-line running-line">
                    <span class="spinner" aria-hidden="true"></span>
                    <span class="running-text">{ &entry.output }</span>
                </div>
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
