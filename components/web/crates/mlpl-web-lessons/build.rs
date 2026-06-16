//! Build-script facade. Generates the `LESSONS` const from `lessons.toml`
//! into `OUT_DIR`; logic lives in `build_gen.rs` (docs/code_metrics.md
//! section 14: build.rs is a facade, behavior in named files).

mod build_gen;

fn main() {
    build_gen::run();
}
