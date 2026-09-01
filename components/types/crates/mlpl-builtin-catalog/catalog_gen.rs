//! Read each domain's catalog TOML and write its generated
//! `<domain>.rs` (holding `const GROUPS`) into `OUT_DIR`, where
//! `groups_lang.rs` / `groups_ml.rs` `include!` it.

use std::path::Path;

use crate::catalog_emit::render;
use crate::catalog_parse::Doc;

/// Generate `catalog_lang.rs` and `catalog_ml.rs` into `OUT_DIR`.
pub fn run() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    for name in ["catalog_lang", "catalog_ml"] {
        let toml_path = format!("{name}.toml");
        println!("cargo:rerun-if-changed={toml_path}");
        let text =
            std::fs::read_to_string(&toml_path).unwrap_or_else(|e| panic!("read {toml_path}: {e}"));
        let doc: Doc = toml::from_str(&text).unwrap_or_else(|e| panic!("parse {toml_path}: {e}"));
        let out = Path::new(&out_dir).join(format!("{name}.rs"));
        std::fs::write(&out, render(&doc)).expect("write generated catalog");
    }
}
