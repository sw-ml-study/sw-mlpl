//! Saga 33 step 007: bottom-of-page chrome (Footer +
//! DocDialog) extracted from `render_shell`. Pure
//! presentational helper; no Yew hooks.

use yew::prelude::*;

use crate::components::{DocDialog, Footer};

const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

pub fn render_shell_footer(dialog_open: bool, on_close: Callback<MouseEvent>) -> Html {
    // Saga 82: Footer moved to mlpl-web-components-chrome; only this
    // crate's build.rs emits the BUILD_* env vars, so format the
    // build-info chip here and hand it in as a prop.
    let build_info = format!(
        "v{}.{} \u{00b7} {} \u{00b7} {} \u{00b7} {}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT"),
        env!("BUILD_HOST"),
        env!("BUILD_SHA"),
        env!("BUILD_TIMESTAMP"),
    );
    html! {
        <>
            <Footer url={REPO_URL} build_info={build_info} />
            <DocDialog open={dialog_open} {on_close} />
        </>
    }
}
