//! Saga 33 step 007: top Header component template + tab
//! callbacks. Just the html! for the `<Header>` tag with its
//! props wired from the mode callback bundle.

use yew::prelude::*;

use mlpl_web_components_chrome::header::Header;
use mlpl_web_mode::callbacks::ModeCallbacks;
use mlpl_web_render_types::args::InputCallbacks;
use mlpl_web_render_types::args::Modes;

pub fn render_shell_header(
    inputs: &InputCallbacks,
    modes: &Modes,
    cb: &ModeCallbacks,
    on_tour: Callback<MouseEvent>,
) -> Html {
    html! {
        <Header
            on_help={inputs.open_dialog.clone()}
            {on_tour}
            on_select_repl={cb.repl.clone()}
            on_select_tutorial={cb.tutorial.clone()}
            on_select_paths={cb.paths.clone()}
            on_select_editor={cb.editor.clone()}
            mode={modes.header_mode}
        />
    }
}
