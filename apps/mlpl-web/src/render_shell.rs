//! Saga 33 step 007: outer-shell composer. The html! is split
//! across `render_shell_chrome` (header + mode bar) and
//! `render_shell_footer` (footer + dialog) so each `html!`
//! invocation lives in its own file under the 25-LOC budget.

use yew::prelude::*;

use crate::mode_callbacks;
use crate::onboarding_splash::{SplashOverlay, make_splash_action};
use crate::onboarding_tour::TourTooltip;
use crate::render::RenderArgs;
use crate::render_callbacks::InputCallbacks;
use crate::render_main::{MainArgs, render_main};
use crate::render_modes::Modes;
use crate::render_shell_chrome::render_shell_chrome;
use crate::render_shell_footer::render_shell_footer;
use crate::tutorial::run_example;

pub fn render_shell(a: RenderArgs, inputs: InputCallbacks, modes: Modes) -> Html {
    let cb = mode_callbacks::bundle(
        a.ui.lesson_idx.clone(),
        a.ui.path_state.clone(),
        a.ui.tutorial_initial_view.clone(),
        a.callbacks.on_demo.clone(),
    );
    let main_args = build_main_args(&a, &cb, &inputs, &modes);
    let tour_h = a.onboarding.show_tour.clone();
    let step_h = a.onboarding.tour_step.clone();
    let on_tour = Callback::from(move |_: MouseEvent| {
        tour_h.set(true);
        step_h.set(0);
    });
    let chrome = render_shell_chrome(&a, &inputs, &modes, &cb, on_tour);
    let footer = render_shell_footer(*a.ui.dialog_open, inputs.close_dialog);
    let splash = render_splash(&a);
    let tour = render_tour(&a);
    html! { <> { chrome } { render_main(main_args) } { footer } { splash } { tour } </> }
}

fn render_splash(a: &RenderArgs) -> Html {
    if !*a.onboarding.show_splash {
        return html! {};
    }
    let on_action = make_splash_action(
        a.onboarding.show_splash.clone(),
        a.onboarding.show_tour.clone(),
        a.callbacks.on_demo.clone(),
        a.ui.input_value.clone(),
        a.ui.lesson_idx.clone(),
        a.ui.path_state.clone(),
    );
    html! { <SplashOverlay {on_action} /> }
}

fn render_tour(a: &RenderArgs) -> Html {
    if !*a.onboarding.show_tour {
        return html! {};
    }
    let step = *a.onboarding.tour_step;
    let sh = a.onboarding.tour_step.clone();
    let th = a.onboarding.show_tour.clone();
    let on_next = Callback::from(move |_: MouseEvent| {
        if *sh + 1 >= 6 {
            th.set(false);
        } else {
            sh.set(*sh + 1);
        }
    });
    let sh2 = a.onboarding.tour_step.clone();
    let on_prev = Callback::from(move |_: MouseEvent| {
        if *sh2 > 0 {
            sh2.set(*sh2 - 1);
        }
    });
    let ch = a.onboarding.show_tour.clone();
    let on_close = Callback::from(move |_: MouseEvent| ch.set(false));
    html! { <TourTooltip {step} {on_next} {on_prev} {on_close} /> }
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
        completion_candidates: (*a.ui.completion_candidates).clone(),
        on_pick_completion: make_pick_completion(
            a.ui.input_value.clone(),
            a.ui.completion_candidates.clone(),
            a.ui.completion_selected.clone(),
        ),
        completion_selected: *a.ui.completion_selected,
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
        let (out, _) = crate::completion::apply_completion(&input_value, cur, &chosen);
        input_value.set(out);
        completion_candidates.set(Vec::new());
        completion_selected.set(0);
    })
}
