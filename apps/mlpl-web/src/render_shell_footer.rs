//! Saga 33 step 007: bottom-of-page chrome (Footer +
//! DocDialog) extracted from `render_shell`. Pure
//! presentational helper; no Yew hooks.

use yew::prelude::*;

use crate::components::{DocDialog, Footer};

const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

pub fn render_shell_footer(dialog_open: bool, on_close: Callback<MouseEvent>) -> Html {
    html! {
        <>
            <Footer url={REPO_URL} />
            <DocDialog open={dialog_open} {on_close} />
        </>
    }
}
