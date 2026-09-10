//! Layout: split the equation into lines and stack them as `<text>`
//! rows on a high-contrast dark canvas. Inline sub/superscript parsing
//! lives in `spans`.

use crate::spans::{push_span, spans};

const FILL: &str = "#cdd6f4"; // light glyphs
const BG: &str = "#1e1e2e"; // dark canvas
const FS: f64 = 20.0; // main glyph size (px)
const LINE_H: f64 = 30.0; // line advance (px)
const CHAR_W: f64 = 11.0; // advance estimate, for the canvas width only
const PAD: f64 = 16.0;

/// Render `text` (Unicode math, newline-separated) as a self-contained,
/// high-contrast SVG document. `_`/`^` runs become sub/superscripts.
pub fn render_equation(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let cols = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let w = 2.0 * PAD + cols as f64 * CHAR_W;
    let h = 2.0 * PAD + lines.len().max(1) as f64 * LINE_H;
    let mut out = open(w, h);
    for (i, line) in lines.iter().enumerate() {
        push_line(&mut out, line, PAD, PAD + LINE_H * (i as f64 + 0.75));
    }
    out.push_str("</svg>");
    out
}

/// Open an SVG document of the given size with the dark background rect.
fn open(w: f64, h: f64) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" \
         width=\"{w:.0}\" height=\"{h:.0}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{BG}\"/>"
    )
}

/// Emit one equation line as a `<text>`, delegating each inline span.
fn push_line(out: &mut String, line: &str, x: f64, y: f64) {
    out.push_str(&format!(
        "<text x=\"{x:.0}\" y=\"{y:.0}\" fill=\"{FILL}\" font-size=\"{FS:.0}\" \
         font-family=\"'Latin Modern Math','Cambria Math','STIX Two Math',serif\">"
    ));
    for span in spans(line) {
        push_span(out, &span);
    }
    out.push_str("</text>");
}
