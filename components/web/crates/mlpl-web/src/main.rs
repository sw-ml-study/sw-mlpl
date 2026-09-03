//! mlpl-web binary entrypoint. On wasm32 it mounts the Yew app;
//! as a native binary it prints the build/serve hints and exits
//! non-zero (mlpl-web is a WASM bundle, not a native CLI).
//! Spike step 014: the per-target bodies moved back inline as
//! `cfg` blocks -- one file instead of build.rs codegen plus two
//! include fragments, which also retires two crate modules.

fn main() {
    #[cfg(target_arch = "wasm32")]
    mlpl_web::app::start();
    #[cfg(not(target_arch = "wasm32"))]
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
             Or visit the live demo:\n\
               Stable:  https://mlpl.softwarewrighter.com/\n\
               Latest:  https://sw-ml-study.github.io/sw-mlpl/"
        );
        std::process::exit(1);
    }
}
