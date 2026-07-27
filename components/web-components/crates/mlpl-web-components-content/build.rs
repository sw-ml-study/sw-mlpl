//! Build entrypoint: converts the two doc-tab markdown files into
//! HTML at compile time (naming: user asked for include_str of
//! HTML with inline CSS -- generating it from the canonical .md
//! keeps a single source of truth). Logic lives in
//! `build_md_html.rs`; this file stays a thin coordinator.

mod build_md_html;

fn main() {
    build_md_html::emit("../../../../docs/lang-reference.md", "lang_reference.html");
    build_md_html::emit("../../../../docs/usage.md", "usage.html");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_md_html.rs");
    println!("cargo:rerun-if-changed=../../../../docs/lang-reference.md");
    println!("cargo:rerun-if-changed=../../../../docs/usage.md");
}
