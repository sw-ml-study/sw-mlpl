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
) -> Html {
    html! {
        <>
            <mlpl_web_components_chrome::github_corner::GithubCorner url={REPO_URL} />
            <GlossaryPopupHost />
            { render_shell_header(inputs, modes, cb, make_tour_callback(a)) }
            { render_shell_modebar(a, modes) }
        </>
    }
}

/// The "Tour" button handler: close any lesson/path/dialog, rewind to
/// step 0, and show the onboarding tour.
fn make_tour_callback(a: &RenderArgs) -> Callback<MouseEvent> {
    let (th, sh) = (
        a.onboarding.show_tour.clone(),
        a.onboarding.tour_step.clone(),
    );
    let (lesson, path) = (a.ui.lesson_idx.clone(), a.ui.path_state.clone());
    let dialog = a.ui.dialog_open.clone();
    Callback::from(move |_: MouseEvent| {
        lesson.set(None);
        path.set(None);
        dialog.set(false);
        sh.set(0);
        th.set(true);
    })
}
