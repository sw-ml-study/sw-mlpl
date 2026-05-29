//! Saga 33 step 007: ModeBar component template. Just the html!
//! for `<ModeBar>` with props wired from the app's callbacks
//! + modes.

use yew::prelude::*;

use mlpl_web_components_content::mode_bar::ModeBar;
use mlpl_web_render_types::args::Modes;
use mlpl_web_render_types::args::RenderArgs;

pub fn render_shell_modebar(a: &RenderArgs, modes: &Modes) -> Html {
    html! {
        <ModeBar
            on_demo={a.callbacks.on_demo.clone()}
            on_upload={a.callbacks.on_upload.clone()}
            on_upload_cancel={a.callbacks.on_upload_cancel.clone()}
            upload_input_ref={a.upload.input_ref.clone()}
            tutorial_active={modes.tutorial_active}
        />
    }
}
