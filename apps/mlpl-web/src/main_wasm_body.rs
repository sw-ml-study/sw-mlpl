// Saga 33 step 007: body of `fn main()` when target_arch =
// wasm32. Included into `src/main.rs` via build.rs codegen.
// Mounting lives in the library crate so this is one call.
//
// `include!()` parses as a single expression, so this file is
// the expression itself (no trailing semicolon, no `//!`
// inner-doc comments that would bind to `fn main`).
mlpl_web::app::start()
