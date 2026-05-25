//! Saga 33 step 007: top-of-page chrome composer. Spreads the
//! header + mode-bar templates across two sibling files
//! (`render_shell_header`, `render_shell_modebar`) so each
//! html! invocation stays under the 25-LOC budget.

use yew::prelude::*;

use crate::glossary_popup::GlossaryPopupHost;
use crate::mode_callbacks::ModeCallbacks;
use crate::render::RenderArgs;
use crate::render_callbacks::InputCallbacks;
use crate::render_modes::Modes;
use crate::render_shell_header::render_shell_header;
use crate::render_shell_modebar::render_shell_modebar;

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
            <crate::components::GithubCorner url={REPO_URL} />
            <GlossaryPopupHost />
            { render_shell_header(inputs, modes, cb, on_tour) }
            { render_shell_modebar(a, modes) }
        </>
    }
}
