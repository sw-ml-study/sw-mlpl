//! Saga 33 step 007: outer-shell composer. The html! is split
//! across `render_shell_chrome` (header + mode bar) and
//! `render_shell_footer` (footer + dialog) so each `html!`
//! invocation lives in its own file under the 25-LOC budget.

use yew::prelude::*;

use mlpl_web_mode::callbacks as mode_callbacks;
use mlpl_web_render_core::callbacks::build_input_callbacks;
use mlpl_web_render_core::modes::derive_modes;
use mlpl_web_render_core::panel::{MainArgs, render_main};
use mlpl_web_render_types::args::{InputCallbacks, Modes, RenderArgs};
use mlpl_web_tutorial::run_example;

use crate::chrome::render_shell_chrome;
use crate::footer::render_shell_footer;
use crate::overlays::render_overlays;

/// Top-level render entry. Saga 82 step 8 merged the original
/// render-core/render.rs `render()` body (a 5-line dispatcher)
/// into shell so render-core stops depending on shell -- the
/// DAG goes shell -> core -> types, no cycles.
pub fn render(a: RenderArgs) -> Html {
    let inputs = build_input_callbacks(&a.callbacks, &a.ui);
    let modes = derive_modes(&a.ui);
    render_shell(a, inputs, modes)
}

pub fn render_shell(a: RenderArgs, inputs: InputCallbacks, modes: Modes) -> Html {
    let cb = mode_callbacks::bundle(
        a.ui.lesson_idx.clone(),
        a.ui.path_state.clone(),
        a.ui.tutorial_initial_view.clone(),
        a.callbacks.on_demo.clone(),
        a.ui.editor_open.clone(),
    );
    let main_args = build_main_args(&a, &cb, &inputs, &modes);
    let th = a.onboarding.show_tour.clone();
    let sh = a.onboarding.tour_step.clone();
    let lesson = a.ui.lesson_idx.clone();
    let path = a.ui.path_state.clone();
    let dialog = a.ui.dialog_open.clone();
    let on_tour = Callback::from(move |_: MouseEvent| {
        lesson.set(None);
        path.set(None);
        dialog.set(false);
        sh.set(0);
        th.set(true);
    });
    let chrome = render_shell_chrome(&a, &inputs, &modes, &cb, on_tour);
    let footer = render_shell_footer(*a.ui.dialog_open, inputs.close_dialog, a.build_info.clone());
    let overlays = render_overlays(&a);
    html! { <> { chrome } { render_main(main_args) } { footer } { overlays } </> }
}

fn build_main_args<'a>(
    a: &'a RenderArgs,
    cb: &'a mode_callbacks::ModeCallbacks,
    inputs: &InputCallbacks,
    modes: &Modes,
) -> MainArgs<'a> {
    MainArgs {
        tutorial_active: modes.tutorial_active,
        paths_active: modes.paths_active,
        cur_lesson: modes.cur_lesson,
        lesson_idx: a.ui.lesson_idx.clone(),
        initial_view: *a.ui.tutorial_initial_view,
        cur_path: modes.cur_path,
        cb,
        history: &a.active.history,
        input_value: &a.ui.input_value,
        on_input: inputs.on_input.clone(),
        on_keydown: inputs.on_keydown.clone(),
        on_run_example: run_example(a.callbacks.on_submit.clone(), a.ui.input_value.clone()),
        on_run_batch: a.callbacks.on_run_batch.clone(),
        on_clear: a.callbacks.on_clear.clone(),
        completion_candidates: (*a.ui.completion_candidates).clone(),
        on_pick_completion: make_pick_completion(
            a.ui.input_value.clone(),
            a.ui.completion_candidates.clone(),
            a.ui.completion_selected.clone(),
        ),
        completion_selected: *a.ui.completion_selected,
        show_3d: *a.ui.show_3d,
        on_toggle_3d: {
            let s = a.ui.show_3d.clone();
            Callback::from(move |_: MouseEvent| s.set(!*s))
        },
        editor_active: modes.editor_active,
        editor_open: a.ui.editor_open.clone(),
        editor_content: a.ui.editor_content.clone(),
    }
}

/// Saga 33 step 043: build the click handler that fires when
/// the user clicks a tab-completion chip. Inserts the chosen
/// completion at the cursor position and clears the popup.
/// Cursor lookup happens at the moment of click against the
/// current input value (we don't track cursor in app state).
fn make_pick_completion(
    input_value: UseStateHandle<String>,
    completion_candidates: UseStateHandle<Vec<String>>,
    completion_selected: UseStateHandle<usize>,
) -> Callback<String> {
    Callback::from(move |chosen: String| {
        let cur = input_value.len();
        let (out, _) = mlpl_web_completion::apply_completion(&input_value, cur, &chosen);
        input_value.set(out);
        completion_candidates.set(Vec::new());
        completion_selected.set(0);
    })
}
