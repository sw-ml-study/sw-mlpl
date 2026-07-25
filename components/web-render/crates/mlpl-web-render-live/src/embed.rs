//! Persisting the final loss chart into the result entry
//! (connect-telemetry step 004).
//!
//! While a streamed train runs, the layout is hardware sparklines,
//! then the live 2D chart. When the running marker is replaced by the
//! result, the chart used to collapse into a text sparkline ABOVE
//! where the chart had been -- a disconcerting reversal. Instead, the
//! completion path now appends `final_report` (the final chart
//! rendered from the loss trace, THEN the one-line summary) to the
//! result text, and `render_entry` splits it back out via
//! `split_embedded_svg` + `render_composite`.

use yew::prelude::*;

use crate::loss_svg::loss_panel_svg;

/// Split a result output into `(pre, svg, post)` when one line is an
/// embedded single-line `<svg` chart. `None` when no chart is
/// embedded (the common plain-text result).
#[must_use]
pub fn split_embedded_svg(out: &str) -> Option<(String, String, String)> {
    let idx = out.lines().position(|l| l.starts_with("<svg"))?;
    let lines: Vec<&str> = out.lines().collect();
    let pre = lines[..idx].join("\n");
    let svg = lines[idx].to_string();
    let post = lines[idx + 1..].join("\n");
    Some((pre, svg, post))
}

/// The persistent record for a completed streamed train: the final
/// chart (rendered from the generation's loss trace) followed by the
/// one-line loss summary -- chart first, matching the during-training
/// visual order. Consumes the generation. Leading newline so callers
/// append it directly below the value text. `None` when nothing was
/// streamed.
#[must_use]
pub fn final_report(gen_id: u32) -> Option<String> {
    let (train, val) = mlpl_web_eval::loss_trace::series(gen_id);
    let chart = loss_panel_svg(&train, &val);
    let summary = mlpl_web_eval::loss_trace::summary(gen_id)?;
    match chart {
        Some(svg) => Some(format!("\n{svg}\n{summary}")),
        None => Some(format!("\n{summary}")),
    }
}

/// Render a split composite result: value/telemetry text, the
/// persisted chart (same `.loss-panel` frame as the live view), and
/// the trailing loss line.
pub fn render_composite(pre: &str, svg: String, post: &str) -> Html {
    let text = |s: &str| {
        if s.is_empty() {
            html! {}
        } else {
            html! { <pre class="output-line">{ s.to_string() }</pre> }
        }
    };
    html! {
        <>
            { text(pre) }
            <div class="loss-panel">{ Html::from_html_unchecked(AttrValue::from(svg)) }</div>
            { text(post) }
        </>
    }
}
