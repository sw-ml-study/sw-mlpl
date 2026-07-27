//! Codegen for the demo consts: parse `demos.toml` and emit `DEMOS` plus the
//! per-demo metadata consts (`DEMO_CAPABILITIES`, `LITERATE_DOCS`,
//! `PROGRESS_NOTES`) the cluster used to hand-maintain in Rust source, into
//! `OUT_DIR`. All demo PROSE, MLPL `lines`, gating tiers, and heads-up notes
//! live in data (`demos.toml`), not Rust source, so they no longer count
//! against the file/module-LOC budgets. See docs/code_metrics.md section 15a
//! (data lists) + section 14 (build.rs).

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Doc {
    demos: Vec<DemoRow>,
    #[serde(default)]
    capabilities: Vec<CapRow>,
    #[serde(default)]
    literate: Vec<LitRow>,
    #[serde(default)]
    progress_notes: Vec<NoteRow>,
}

#[derive(Deserialize)]
struct DemoRow {
    category: String,
    name: String,
    intro: String,
    takeaway: String,
    lines: Vec<String>,
}

#[derive(Deserialize)]
struct CapRow {
    name: String,
    requires_connect: bool,
    #[serde(default)]
    prefers_connect: bool,
    device: String,
}

#[derive(Deserialize)]
struct LitRow {
    name: String,
    html: String,
}

#[derive(Deserialize)]
struct NoteRow {
    demo: String,
    line_idx: usize,
    heading: String,
    body: String,
}

/// Read `demos.toml`, generate the `DEMOS` + metadata consts, write to
/// `OUT_DIR/demos.rs`.
pub fn run() {
    println!("cargo:rerun-if-changed=demos.toml");
    let toml = std::fs::read_to_string("demos.toml").expect("read demos.toml");
    let doc: Doc = toml::from_str(&toml).expect("parse demos.toml");
    let body: String = doc.demos.iter().map(render_demo).collect();
    let src = format!(
        "pub const DEMOS: &[Demo] = &[{body}];\n{}",
        render_meta(&doc)
    );
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("demos.rs");
    std::fs::write(out, src).expect("write generated demos.rs");
}

/// One `Demo { .. }` literal, comma-terminated, with its `lines` slice.
fn render_demo(d: &DemoRow) -> String {
    let lines: String = d.lines.iter().map(|l| format!("{l:?},")).collect();
    format!(
        "Demo{{name:{:?},category:{:?},intro:{:?},takeaway:{:?},lines:&[{lines}]}},",
        d.name, d.category, d.intro, d.takeaway
    )
}

/// The three metadata const definitions, generated from the toml tables.
fn render_meta(doc: &Doc) -> String {
    let device = |d: &str| match d {
        "mlx" => "Device::Mlx",
        "cuda" => "Device::Cuda",
        "ollama" => "Device::Ollama",
        _ => "Device::Cpu",
    };
    let caps: String = doc
        .capabilities
        .iter()
        .map(|c| {
            format!(
                "({:?},Capability{{requires_connect:{},prefers_connect:{},device:{}}}),",
                c.name,
                c.requires_connect,
                c.prefers_connect,
                device(&c.device)
            )
        })
        .collect();
    let lits: String = doc
        .literate
        .iter()
        .map(|l| format!("({:?},{:?}),", l.name, l.html))
        .collect();
    let notes: String = doc
        .progress_notes
        .iter()
        .map(|n| {
            format!(
                "ProgressNote{{demo:{:?},line_idx:{},heading:{:?},body:{:?}}},",
                n.demo, n.line_idx, n.heading, n.body
            )
        })
        .collect();
    format!(
        "pub const DEMO_CAPABILITIES: &[(&str, Capability)] = &[{caps}];\n\
         pub const LITERATE_DOCS: &[(&str, &str)] = &[{lits}];\n\
         pub const PROGRESS_NOTES: &[ProgressNote] = &[{notes}];\n"
    )
}
