//! Saga 33 step 006: PathsView -> parent callback builders. The
//! PathsView component emits three events upward: state replace,
//! "open lesson N", and "run demo <name>"; each has its builder
//! here.

use yew::prelude::*;

use mlpl_web_demos as demos;
use mlpl_web_tutorial::TutorialView;

/// PathsView -> parent: replace the entire paths state. PathsView
/// owns its own picker/walker transitions; the parent just stores
/// whatever PathsView reports.
pub fn cb_path_change(
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
) -> Callback<Option<(Option<usize>, usize)>> {
    Callback::from(move |next| path_state.set(next))
}

/// PathsView -> parent: "open lesson N". Jump TutorialPanel
/// to the Lesson subview and set the lesson index. Saga 33
/// step 042 (path-resume bug fix): do NOT clear `path_state`
/// -- the walker position must survive the navigation so the
/// user can click "Back to path" and resume where they were.
/// The `paths_active` derivation in render_modes.rs now
/// requires `cur_lesson.is_none()` so the walker pane and the
/// lesson pane don't fight over the screen.
pub fn cb_path_open_lesson(
    lesson_idx: UseStateHandle<Option<usize>>,
    tutorial_view: UseStateHandle<TutorialView>,
) -> Callback<usize> {
    Callback::from(move |i: usize| {
        tutorial_view.set(TutorialView::Lesson);
        lesson_idx.set(Some(i));
    })
}

/// PathsView -> parent: "run demo <name>". Look the demo up
/// by name and emit `on_demo` with its index. Unknown names
/// are silently dropped. Saga 33 step 042: `path_state` is
/// preserved so the user can resume after running the demo.
pub fn cb_path_run_demo(on_demo: Callback<usize>) -> Callback<String> {
    Callback::from(move |name: String| {
        if let Some(idx) = demos::DEMOS.iter().position(|d| d.name == name) {
            on_demo.emit(idx);
        }
    })
}
