// Saga 33 step 007: body of `fn main()` when target_arch is
// NOT wasm32. Included into `src/main.rs` via build.rs codegen.
// mlpl-web is a Yew/WASM bundle, not a native CLI; when
// invoked as a native binary we print the build/serve hints
// and exit non-zero.
//
// `include!()` parses as a single expression, so this file is
// a block expression `{ ... }` (no `//!` inner-doc comments
// that would bind to `fn main`).
{
    eprintln!(
        "mlpl-web is a WASM/Yew bundle, not a native CLI.\n\
         \n\
         To run it locally:\n\
           cd apps/mlpl-web && trunk serve\n\
         \n\
         To rebuild the deployed pages/ bundle:\n\
           ./scripts/build-pages.sh\n\
         \n\
         Or visit the live demo: https://sw-ml-study.github.io/sw-mlpl/"
    );
    std::process::exit(1);
}
