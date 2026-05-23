//! Saga 33 step 007: top Header component template + tab
//! callbacks. Just the html! for the `<Header>` tag with its
//! props wired from the mode callback bundle.

use yew::prelude::*;

use crate::components::Header;
use crate::mode_callbacks::ModeCallbacks;
use crate::render_callbacks::InputCallbacks;
use crate::render_modes::Modes;

pub fn render_shell_header(inputs: &InputCallbacks, modes: &Modes, cb: &ModeCallbacks) -> Html {
    html! {
        <Header
            on_help={inputs.open_dialog.clone()}
            on_select_repl={cb.repl.clone()}
            on_select_tutorial={cb.tutorial.clone()}
            on_select_paths={cb.paths.clone()}
            mode={modes.header_mode}
        />
    }
}
