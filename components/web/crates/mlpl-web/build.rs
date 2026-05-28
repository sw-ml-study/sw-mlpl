//! mlpl-web build script entrypoint. Saga 33 step 007: the
//! actual work is delegated to per-task helpers so this file
//! stays a thin coordinator. Add a new task by writing a
//! `build_<task>.rs` sibling, declaring `mod build_<task>;`
//! below, and calling its emit fn from `main`.

mod build_env;
mod build_main_body;

fn main() {
    build_env::emit_build_env_vars();
    build_main_body::emit_main_body_include();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_env.rs");
    println!("cargo:rerun-if-changed=build_main_body.rs");
}
