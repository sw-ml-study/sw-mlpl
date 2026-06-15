//! Codegen for the `PATHS` const: parse `paths.toml` and emit the same
//! `const PATHS: &[LearningPath]` the crate used to hand-write, into
//! `OUT_DIR`. The path PROSE lives in data (`paths.toml`), not Rust source,
//! so it no longer counts against the file/module-LOC budgets. See
//! docs/code_metrics.md section 15a (data lists) + section 14 (build.rs).

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Doc {
    paths: Vec<PathRow>,
}

#[derive(Deserialize)]
struct PathRow {
    title: String,
    blurb: String,
    steps: Vec<StepRow>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StepRow {
    kind: String,
    title: String,
    body: String,
    why: String,
    name: String,
    slug: String,
    term: String,
}

/// Read `paths.toml`, generate the `PATHS` const, write it to `OUT_DIR/paths.rs`.
pub fn run() {
    println!("cargo:rerun-if-changed=paths.toml");
    let toml = std::fs::read_to_string("paths.toml").expect("read paths.toml");
    let doc: Doc = toml::from_str(&toml).expect("parse paths.toml");
    let body: String = doc.paths.iter().map(render_path).collect();
    let src = format!("pub const PATHS: &[LearningPath] = &[{body}];\n");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("paths.rs");
    std::fs::write(out, src).expect("write generated paths.rs");
}

/// One `LearningPath { .. }` literal, comma-terminated.
fn render_path(p: &PathRow) -> String {
    let steps: String = p.steps.iter().map(render_step).collect();
    format!(
        "LearningPath {{ title: {}, blurb: {}, steps: &[{steps}] }},",
        lit(&p.title),
        lit(&p.blurb)
    )
}

/// One `Step::Variant { .. }` literal, comma-terminated, chosen by `kind`.
fn render_step(s: &StepRow) -> String {
    match s.kind.as_str() {
        "note" => format!(
            "Step::Note{{title:{},body:{}}},",
            lit(&s.title),
            lit(&s.body)
        ),
        "lesson" => format!(
            "Step::Lesson{{title:{},why:{}}},",
            lit(&s.title),
            lit(&s.why)
        ),
        "demo" => format!("Step::Demo{{name:{},why:{}}},", lit(&s.name), lit(&s.why)),
        "diagram" => format!(
            "Step::Diagram{{slug:{},why:{}}},",
            lit(&s.slug),
            lit(&s.why)
        ),
        "glossary" => format!(
            "Step::Glossary{{term:{},why:{}}},",
            lit(&s.term),
            lit(&s.why)
        ),
        other => panic!("paths.toml: unknown step kind {other:?}"),
    }
}

/// A `&str` as an escaped Rust string literal -- `{:?}` does exactly that.
fn lit(s: &str) -> String {
    format!("{s:?}")
}
