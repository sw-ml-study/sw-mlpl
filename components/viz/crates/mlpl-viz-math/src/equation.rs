//! Layout: stack the equation's lines and lay each out left-to-right as
//! positioned atoms (text runs + big operators with stacked limits). The
//! canvas leaves room above the first baseline for upper limits and below
//! the last for lower limits.

use crate::atoms::atoms;
use crate::render::{BG, LINE_H, PAD, SUB_DROP, SUP_RISE, push_atom};

/// Render `text` (Unicode math, newline-separated) as a self-contained,
/// high-contrast SVG document. Big operators (sum, product, integral)
/// followed by `_{...}`/`^{...}` show their limits stacked above and
/// below; ordinary `_x`/`^x` stay inline.
pub fn render_equation(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let base0 = PAD + SUP_RISE + 14.0;
    let mut body = String::new();
    let mut max_w = 0.0f64;
    for (i, line) in lines.iter().enumerate() {
        let y = base0 + LINE_H * i as f64;
        let mut x = PAD;
        for atom in atoms(line) {
            x += push_atom(&mut body, &atom, x, y);
        }
        max_w = max_w.max(x);
    }
    let w = max_w + PAD;
    let h = base0 + LINE_H * (lines.len().max(1) as f64 - 1.0) + SUB_DROP + PAD;
    format!("{}{body}</svg>", open(w, h))
}

/// Open an SVG document of the given size with the dark background rect.
fn open(w: f64, h: f64) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" \
         width=\"{w:.0}\" height=\"{h:.0}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{BG}\"/>"
    )
}
