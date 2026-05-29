//! Saga 33 step 007: top-of-page chrome composer. Spreads the
//! header + mode-bar templates across two sibling files
//! (`render_shell_header`, `render_shell_modebar`) so each
//! html! invocation stays under the 25-LOC budget.

use yew::prelude::*;

use crate::header::render_shell_header;
use crate::modebar::render_shell_modebar;
use mlpl_web_glossary::popup::GlossaryPopupHost;
use mlpl_web_mode::callbacks::ModeCallbacks;
use mlpl_web_render_types::args::InputCallbacks;
use mlpl_web_render_types::args::Modes;
use mlpl_web_render_types::args::RenderArgs;

pub const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

pub fn render_shell_chrome(
    a: &RenderArgs,
    inputs: &InputCallbacks,
    modes: &Modes,
    cb: &ModeCallbacks,
    on_tour: Callback<MouseEvent>,
) -> Html {
    html! {
        <>
            <mlpl_web_components_chrome::github_corner::GithubCorner url={REPO_URL} />
            <GlossaryPopupHost />
            { render_shell_header(inputs, modes, cb, on_tour) }
            { render_shell_modebar(a, modes) }
        </>
    }
}
