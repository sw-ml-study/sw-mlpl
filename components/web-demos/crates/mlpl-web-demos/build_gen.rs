//! Codegen for the `DEMOS` const: parse `demos.toml` and emit the same
//! `const DEMOS: &[Demo]` the crate used to aggregate by hand, into
//! `OUT_DIR`. Demo PROSE + MLPL `lines` live in data (`demos.toml`), not Rust
//! source, so they no longer count against the file/module-LOC budgets. See
//! docs/code_metrics.md section 15a (data lists) + section 14 (build.rs).

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Doc {
    demos: Vec<DemoRow>,
}

#[derive(Deserialize)]
struct DemoRow {
    category: String,
    name: String,
    intro: String,
    takeaway: String,
    lines: Vec<String>,
}

/// Read `demos.toml`, generate the `DEMOS` const, write it to `OUT_DIR/demos.rs`.
pub fn run() {
    println!("cargo:rerun-if-changed=demos.toml");
    let toml = std::fs::read_to_string("demos.toml").expect("read demos.toml");
    let doc: Doc = toml::from_str(&toml).expect("parse demos.toml");
    let body: String = doc.demos.iter().map(render_demo).collect();
    let src = format!("pub const DEMOS: &[Demo] = &[{body}];\n");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("demos.rs");
    std::fs::write(out, src).expect("write generated demos.rs");
}

/// One `Demo { .. }` literal, comma-terminated, with its `lines` slice.
fn render_demo(d: &DemoRow) -> String {
    let lines: String = d.lines.iter().map(|l| format!("{l:?},")).collect();
    format!(
        "Demo{{name:{},category:{},intro:{},takeaway:{},lines:&[{lines}]}},",
        lit(&d.name),
        lit(&d.category),
        lit(&d.intro),
        lit(&d.takeaway)
    )
}

/// A `&str` as an escaped Rust string literal -- `{:?}` does exactly that.
fn lit(s: &str) -> String {
    format!("{s:?}")
}
