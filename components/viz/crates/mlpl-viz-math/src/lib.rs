//! `mlpl-viz-math`: render a Unicode math STRING as a high-contrast SVG
//! visual (the `"equation"` render type). A sibling to the chart-mark
//! and dataflow renderers, used to show the math a demo derives its
//! code from. Self-contained: no external fonts, scripts, or crates, so
//! it renders identically in the web playground and the native REPL's
//! cached-SVG output.

mod equation;
mod spans;

pub use equation::render_equation;
