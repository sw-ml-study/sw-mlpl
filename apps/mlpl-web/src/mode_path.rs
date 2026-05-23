//! Saga 33 step 006: PathsView -> parent callback builders. The
//! PathsView component emits three events upward: state replace,
//! "open lesson N", and "run demo <name>"; each has its builder
//! here.

use yew::prelude::*;

use crate::components::TutorialView;
use crate::demos;

/// PathsView -> parent: replace the entire paths state. PathsView
/// owns its own picker/walker transitions; the parent just stores
/// whatever PathsView reports.
pub fn cb_path_change(
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
) -> Callback<Option<(Option<usize>, usize)>> {
    Callback::from(move |next| path_state.set(next))
}

/// PathsView -> parent: "open lesson N". Close paths, jump
/// TutorialPanel to the Lesson subview, set the lesson index.
pub fn cb_path_open_lesson(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    tutorial_view: UseStateHandle<TutorialView>,
) -> Callback<usize> {
    Callback::from(move |i: usize| {
        path_state.set(None);
        tutorial_view.set(TutorialView::Lesson);
        lesson_idx.set(Some(i));
    })
}

/// PathsView -> parent: "run demo <name>". Look the demo up by
/// name, close paths, and emit `on_demo` with its index. Unknown
/// names are silently dropped (PathsView only emits known demo
/// names today, but defensive).
pub fn cb_path_run_demo(
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    on_demo: Callback<usize>,
) -> Callback<String> {
    Callback::from(move |name: String| {
        if let Some(idx) = demos::DEMOS.iter().position(|d| d.name == name) {
            path_state.set(None);
            on_demo.emit(idx);
        }
    })
}
