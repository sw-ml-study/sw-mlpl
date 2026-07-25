//! Connect-telemetry step 004: persist the final loss chart into the
//! result entry. `split_embedded_svg` finds a single-line `<svg` chart
//! embedded in an output string; `final_report` builds the persistent
//! record (chart first, THEN the one-line loss summary -- matching the
//! during-training visual order) from a generation's loss trace.

use mlpl_web_eval::loss_trace;
use mlpl_web_render_live::embed::{final_report, split_embedded_svg};

#[test]
fn split_finds_embedded_chart_between_text_blocks() {
    let out = "42\nbackend load during eval:\n  CPU ...\n<svg viewBox=\"0 0 1 1\"></svg>\nlive loss ... 3 steps";
    let (pre, svg, post) = split_embedded_svg(out).expect("embedded svg found");
    assert!(pre.starts_with("42"));
    assert!(pre.contains("backend load"));
    assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    assert_eq!(post, "live loss ... 3 steps");
}

#[test]
fn split_returns_none_without_an_embedded_chart() {
    assert!(split_embedded_svg("plain text\nmore text").is_none());
    assert!(split_embedded_svg("").is_none());
}

#[test]
fn split_handles_chart_with_no_trailing_text() {
    let out = "value\n<svg></svg>";
    let (pre, svg, post) = split_embedded_svg(out).expect("found");
    assert_eq!(pre, "value");
    assert_eq!(svg, "<svg></svg>");
    assert!(post.is_empty());
}

#[test]
fn final_report_is_chart_then_summary_and_consumes_the_gen() {
    let gen_id = 881_001;
    loss_trace::push(gen_id, "loss", 4.0);
    loss_trace::push(gen_id, "loss", 2.0);
    loss_trace::push(gen_id, "loss", 1.0);
    let report = final_report(gen_id).expect("report for populated gen");
    // Chart comes BEFORE the text sparkline, matching the live layout.
    let svg_at = report.find("<svg").expect("chart in report");
    let text_at = report.find("live loss").expect("summary in report");
    assert!(
        svg_at < text_at,
        "chart must precede the loss line: {report}"
    );
    assert!(report.starts_with('\n'), "appends below the value text");
    assert!(final_report(gen_id).is_none(), "report consumes the gen");
}

#[test]
fn final_report_with_one_point_still_records_the_summary() {
    let gen_id = 881_002;
    loss_trace::push(gen_id, "loss", 7.0);
    let report = final_report(gen_id).expect("summary-only report");
    assert!(!report.contains("<svg"), "no chart from a single point");
    assert!(report.contains("live loss"));
}

#[test]
fn final_report_is_none_for_unknown_gen() {
    assert!(final_report(881_999).is_none());
}
