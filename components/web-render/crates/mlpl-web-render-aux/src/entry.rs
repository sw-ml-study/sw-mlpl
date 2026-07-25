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
    match entry.kind {
        EntryKind::Narration => render_narration(entry),
        EntryKind::Running => render_running(entry),
        _ => html! {
            <div class="entry">
                { render_input_line(&entry.input) }
                { render_result_body(entry) }
            </div>
        },
    }
}

/// Demo narration: prose framing with no `mlpl>` prompt. `input` is the
/// heading, `output` the body, routed through the path_body markdown-ish
/// renderer so bullets, **bold**, `code`, and `[[glossary-term]]` links work.
fn render_narration(entry: &HistoryEntry) -> Html {
    html! {
        <div class="narration">
            <div class="narration-heading">{ &entry.input }</div>
            <div class="narration-body">{ mlpl_web_path_body::render_body(&entry.output) }</div>
        </div>
    }
}

/// "Currently evaluating" placeholder. The CSS spinner keeps animating during
/// the JS-blocking WASM eval (CSS animations run on the compositor). The
/// telemetry panel is keyed by eval generation so a loaded demo (which reuses
/// one running-entry slot across lines) remounts it per line instead of
/// freezing a propless component on the first line's generation.
fn render_running(entry: &HistoryEntry) -> Html {
    html! {
        <div class="entry running">
            { render_input_line(&entry.input) }
            <div class="output-line running-line">
                <span class="spinner" aria-hidden="true"></span>
                <span class="running-text">{ &entry.output }</span>
            </div>
            <crate::telemetry_panel::TelemetryPanel
                key={mlpl_web_eval::telemetry_trace::current_gen()} />
            // Live loss curve for streamed connect-mode train blocks
            // (connect-telemetry step 002); renders nothing until the
            // eval actually streams metric frames.
            <mlpl_web_render_live::loss_panel::LiveLossPanel
                key={format!("loss-{}", mlpl_web_eval::telemetry_trace::current_gen())} />
        </div>
    }
}

/// The output region for a completed entry: an interactive Plotly panel, an
/// inline SVG, an error block, a collapsible stats summary, or plain text.
fn render_result_body(entry: &HistoryEntry) -> Html {
    let out = entry.output.trim_start();
    if !entry.is_error && out.starts_with(crate::plotly_panel::PLOTLY_MARKER) {
        return html! { <crate::plotly_panel::PlotlyPanel payload={ entry.output.clone() } /> };
    }
    if !entry.is_error && out.starts_with("<svg") {
        return render_svg_body(&entry.output);
    }
    if entry.is_error {
        return html! { <pre class={"output-line error"}>{ &entry.output }</pre> };
    }
    let Some(s) = summary::summarize(&entry.output) else {
        return html! { <pre class={"output-line"}>{ &entry.output }</pre> };
    };
    let line = format!(
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
            <summary>{ line }</summary>
            <pre class={"output-line"}>{ &entry.output }</pre>
        </details>
    }
}
