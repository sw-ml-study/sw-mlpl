//! Build-script facade. Generates the `PATHS` const from `paths.toml` into
//! `OUT_DIR`; the logic lives in `build_gen.rs` (docs/code_metrics.md
//! section 14: build.rs is a facade, behavior in named files).

mod build_gen;

fn main() {
    build_gen::run();
}
