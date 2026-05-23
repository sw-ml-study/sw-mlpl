//! Saga 33 step 006: tutorial-pane builder extracted from
//! `main.rs::render_tutorial`. Returns an empty `html!` when no
//! lesson is active; otherwise builds the `TutorialPanel` props
//! and mounts it.

use yew::prelude::*;

use crate::components::{TutorialPanel, TutorialPanelProps, TutorialView};
use crate::tutorial::{jump_lesson, step_lesson};

pub fn render_tutorial(
    cur: Option<usize>,
    lesson: UseStateHandle<Option<usize>>,
    initial_view: TutorialView,
    on_run_example: Callback<String>,
    on_run_batch: Callback<Vec<String>>,
) -> Html {
    let Some(idx) = cur else { return html! {} };
    let props = TutorialPanelProps {
        lesson_idx: idx,
        on_prev: step_lesson(lesson.clone(), -1),
        on_next: step_lesson(lesson.clone(), 1),
        on_jump: jump_lesson(lesson.clone()),
        on_close: Callback::from(move |_| lesson.set(None)),
        on_run_example,
        on_run_batch,
        initial_view,
    };
    html! { <TutorialPanel ..props /> }
}
