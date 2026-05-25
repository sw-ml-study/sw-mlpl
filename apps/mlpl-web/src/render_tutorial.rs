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
    has_active_path: bool,
) -> Html {
    let Some(idx) = cur else { return html! {} };
    // Saga 33 step 042: clearing `lesson_idx` flips
    // `paths_active` back on (see render_modes.rs), so the
    // walker re-renders at its saved position. Same target
    // as on_close, but presented as "Back to path" rather
    // than the close-X so the user understands they're
    // returning to the walker, not exiting the UI entirely.
    let on_back_to_path = has_active_path.then(|| {
        let lesson = lesson.clone();
        Callback::from(move |_| lesson.set(None))
    });
    let props = TutorialPanelProps {
        lesson_idx: idx,
        on_prev: step_lesson(lesson.clone(), -1),
        on_next: step_lesson(lesson.clone(), 1),
        on_jump: jump_lesson(lesson.clone()),
        on_close: Callback::from(move |_| lesson.set(None)),
        on_run_example,
        on_run_batch,
        initial_view,
        on_back_to_path,
    };
    html! { <TutorialPanel ..props /> }
}
