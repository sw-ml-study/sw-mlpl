//! mlpl-serve build script: stamp the server's build commit + UTC
//! timestamp into env vars so `/v1/health` can report them and the
//! web UI can flag a UI-vs-server build skew. Delegated to a sibling
//! so this stays a thin entry point (mirrors apps/mlpl-web).

mod build_env;

fn main() {
    build_env::emit_build_env_vars();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_env.rs");
}
