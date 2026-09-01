//! Build-script facade. Generates the per-domain `GROUPS` consts from
//! `catalog_lang.toml` / `catalog_ml.toml` into `OUT_DIR`; the behavior
//! lives in named sibling files (docs/code_metrics.md section 14:
//! build.rs is a facade, logic in named files):
//!
//! - `catalog_parse` -- the serde row/group types.
//! - `catalog_emit`  -- a parsed doc -> the `const GROUPS` literal.
//! - `catalog_gen`   -- read each TOML, emit its generated `.rs`.

mod catalog_emit;
mod catalog_gen;
mod catalog_parse;

fn main() {
    catalog_gen::run();
}
