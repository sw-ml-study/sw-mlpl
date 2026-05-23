//! mlpl-web binary entrypoint. Saga 33 step 007: the body is
//! selected by `build.rs` per target arch and `include!()`d
//! here so this file is a 3-line stub.

fn main() {
    include!(concat!(env!("OUT_DIR"), "/main_body.inc.rs"));
}
