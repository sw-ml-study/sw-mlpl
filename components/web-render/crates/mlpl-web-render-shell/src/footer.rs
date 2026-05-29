//! Bottom-of-page chrome (Footer + DocDialog) extracted from
//! `render_shell`. Pure presentational helper; no Yew hooks.
//! Saga 82 step 8: build_info is now threaded from
//! mlpl-web (where build.rs sets BUILD_*) via RenderArgs
//! since this crate has no build script.

use yew::prelude::*;

use mlpl_web_components_chrome::footer::Footer;
use mlpl_web_components_content::doc_dialog::DocDialog;

const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

pub fn render_shell_footer(
    dialog_open: bool,
    on_close: Callback<MouseEvent>,
    build_info: AttrValue,
) -> Html {
    html! {
        <>
            <Footer url={REPO_URL} build_info={build_info} />
            <DocDialog open={dialog_open} {on_close} />
        </>
    }
}
