//! Codegen for the `LESSONS` const: parse `lessons.toml` and emit the same
//! `const LESSONS: &[Lesson]` the crate used to hand-write, into `OUT_DIR`.
//! Tutorial PROSE + example code live in data (`lessons.toml`), not Rust
//! source. See docs/code_metrics.md section 15a + section 14.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Doc {
    lessons: Vec<LessonRow>,
}

#[derive(Deserialize)]
struct LessonRow {
    title: String,
    intro: String,
    examples: Vec<String>,
    try_it: String,
}

/// Read `lessons.toml`, generate the `LESSONS` const, write it to OUT_DIR.
pub fn run() {
    println!("cargo:rerun-if-changed=lessons.toml");
    let toml = std::fs::read_to_string("lessons.toml").expect("read lessons.toml");
    let doc: Doc = toml::from_str(&toml).expect("parse lessons.toml");
    let body: String = doc.lessons.iter().map(render_lesson).collect();
    let src = format!("pub const LESSONS: &[Lesson] = &[{body}];\n");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("lessons_generated.rs");
    std::fs::write(out, src).expect("write generated lessons_generated.rs");
}

/// One `Lesson { .. }` literal, comma-terminated, with its `examples` slice.
fn render_lesson(l: &LessonRow) -> String {
    let examples: String = l.examples.iter().map(|e| format!("{e:?},")).collect();
    format!(
        "Lesson{{title:{},intro:{},examples:&[{examples}],try_it:{}}},",
        lit(&l.title),
        lit(&l.intro),
        lit(&l.try_it)
    )
}

/// A `&str` as an escaped Rust string literal -- `{:?}` does exactly that.
fn lit(s: &str) -> String {
    format!("{s:?}")
}
