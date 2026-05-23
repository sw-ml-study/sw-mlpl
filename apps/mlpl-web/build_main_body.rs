//! Saga 33 step 007: emit `$OUT_DIR/main_body.inc.rs` from the
//! target-appropriate source file so `src/main.rs` can be a
//! one-statement entrypoint that `include!()`s the right body.
//!
//! - `CARGO_CFG_TARGET_ARCH == "wasm32"` -> Yew mount body
//!   (`src/main_wasm_body.rs`).
//! - anything else -> native stub body
//!   (`src/main_not_wasm_body.rs`).

use std::env;
use std::fs;
use std::path::Path;

pub fn emit_main_body_include() {
    let target = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let src = pick_body_source(&target);
    write_body_include(src);
    println!("cargo:rerun-if-changed={src}");
}

fn pick_body_source(target: &str) -> &'static str {
    if target == "wasm32" {
        "src/main_wasm_body.rs"
    } else {
        "src/main_not_wasm_body.rs"
    }
}

fn write_body_include(src: &str) {
    let body = fs::read_to_string(src).expect("read main body source");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_path = Path::new(&out_dir).join("main_body.inc.rs");
    fs::write(out_path, body).expect("write main_body.inc.rs");
}
