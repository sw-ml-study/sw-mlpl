//! Rendering helpers for REPL/tutorial history entries.
//! Extracted from main.rs to keep that module under the
//! sw-checklist file-LOC budget.

use yew::prelude::*;

use mlpl_web_eval::state::{EntryKind, HistoryEntry};
use mlpl_web_eval::summary;

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
fn render_input_line(input: &str) -> Html {
    let (code, comment) = mlpl_web_tutorial::split_inline_comment(input);
    // Saga 29 step 025: route the comment text through the markdown-ish renderer so
    // `[[term]]` inside an MLPL `# comment` becomes a clickable glossary link. Code (the
    // part before `#`) stays a plain string -- the lexer parses MLPL source there and
    // [[ would be ambiguous with array literals.
    let comment_html = match comment {
        Some(c) => html! {
            <span class="line-comment">
                {" # "}
                { mlpl_web_path_body::render_inline(c) }
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

pub fn render_entry(entry: &HistoryEntry) -> Html {
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
                <div class="narration-body">{ mlpl_web_path_body::render_body(&entry.output) }</div>
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
                // Live backend CPU/GPU/RAM/VRAM sparklines while a
                // connect-mode eval runs; renders nothing in local mode.
                // Keyed by the eval generation so a LOADED DEMO (which
                // reuses one running-entry position across lines) remounts
                // the panel per line -- a propless component is otherwise
                // skipped on re-render, freezing it on the first line's
                // (browser-local) generation and never activating for a
                // later server-side `:ask`.
                <crate::telemetry_panel::TelemetryPanel
                    key={mlpl_web_eval::telemetry_trace::current_gen()} />
            </div>
        };
    }
    let body = if !entry.is_error
        && entry
            .output
            .trim_start()
            .starts_with(crate::plotly_panel::PLOTLY_MARKER)
    {
        // Saga 33 step 030: interactive Plotly 3D viz.
        html! { <crate::plotly_panel::PlotlyPanel payload={ entry.output.clone() } /> }
    } else if !entry.is_error && entry.output.trim_start().starts_with("<svg") {
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
