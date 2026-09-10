//! Positioned rendering of one atom at a baseline: a text run as a
//! left-anchored `<text>` (keeping inline sub/superscripts), or a big
//! operator whose limits are STACKED -- centered above and below it, the
//! display convention. Each returns its horizontal advance so the caller
//! lays a line out left to right.

use crate::atoms::Atom;
use crate::spans::{push_span, spans};

pub(crate) const FILL: &str = "#cdd6f4"; // light glyphs
pub(crate) const BG: &str = "#1e1e2e"; // dark canvas
pub(crate) const FS: f64 = 22.0; // main glyph + operator size (px)
const LIM_FS: f64 = 12.0; // limit (bound) size
const CHAR_W: f64 = 12.0; // main advance estimate
const LIM_CW: f64 = 7.0; // limit advance estimate
// The operator's cap reaches ~0.7*FS above the baseline, so the upper
// limit must clear that plus a gap to stay readable and unclipped.
pub(crate) const SUP_RISE: f64 = 26.0; // upper-limit baseline, above main
pub(crate) const SUB_DROP: f64 = 21.0; // lower-limit baseline, below main
pub(crate) const LIM_CAP: f64 = 14.0; // room a limit line needs (cap + gap)
pub(crate) const PAD: f64 = 18.0;
pub(crate) const LINE_H: f64 = 64.0; // line advance (fits stacked limits)

const FONT: &str = "'Latin Modern Math','Cambria Math','STIX Two Math',serif";

/// Render `atom` at baseline `(x, y)`; return its horizontal advance.
pub(crate) fn push_atom(out: &mut String, atom: &Atom, x: f64, y: f64) -> f64 {
    match atom {
        Atom::Text(run) => push_text(out, run, x, y),
        Atom::BigOp { op, sub, sup } => push_bigop(out, op, sub.as_deref(), sup.as_deref(), x, y),
    }
}

/// A left-anchored text run (inline `_x`/`^x` become shifted tspans).
fn push_text(out: &mut String, run: &str, x: f64, y: f64) -> f64 {
    out.push_str(&format!(
        "<text x=\"{x:.0}\" y=\"{y:.0}\" fill=\"{FILL}\" font-size=\"{FS:.0}\" font-family=\"{FONT}\">"
    ));
    for span in spans(run) {
        push_span(out, &span);
    }
    out.push_str("</text>");
    run.chars()
        .filter(|c| !matches!(c, '_' | '^' | '{' | '}'))
        .count() as f64
        * CHAR_W
}

/// A big operator with limits stacked centered above (`sup`) and below
/// (`sub`). The advance is wide enough for the wider of the two limits.
fn push_bigop(
    out: &mut String,
    op: &str,
    sub: Option<&str>,
    sup: Option<&str>,
    x: f64,
    y: f64,
) -> f64 {
    let lim_w = |s: Option<&str>| s.map_or(0.0, |t| t.chars().count() as f64 * LIM_CW);
    let w = (1.4 * CHAR_W).max(lim_w(sub)).max(lim_w(sup)) + 4.0;
    let cx = x + w / 2.0;
    let mut emit = |s: &str, ty: f64, size: f64| {
        let esc = s
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        out.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ty:.1}\" fill=\"{FILL}\" font-size=\"{size:.0}\" \
             font-family=\"{FONT}\" text-anchor=\"middle\">{esc}</text>"
        ));
    };
    emit(op, y, FS);
    if let Some(s) = sup {
        emit(s, y - SUP_RISE, LIM_FS);
    }
    if let Some(s) = sub {
        emit(s, y + SUB_DROP, LIM_FS);
    }
    w
}
