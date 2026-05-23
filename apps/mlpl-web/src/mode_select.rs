//! Saga 33 step 006: header-tab mode-switch callback builders.
//! The Header has three tabs (REPL, Tutorial, Paths); each tab
//! has its own click-callback builder here.

use yew::prelude::*;

use crate::components::TutorialView;

/// "Switch to REPL": clear both lesson and path state so the main
/// pane shows the input + history.
pub fn cb_select_repl(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
) -> Callback<MouseEvent> {
    Callback::from(move |_| {
        lesson_idx.set(None);
        path_state.set(None);
    })
}

/// "Switch to Tutorial": clear path state and, if no lesson is
/// active, land on lesson 0 with the Toc subview. Idempotent: a
/// second click while a lesson is open is a no-op.
pub fn cb_select_tutorial(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    tutorial_view: UseStateHandle<TutorialView>,
) -> Callback<MouseEvent> {
    Callback::from(move |_| {
        path_state.set(None);
        if lesson_idx.is_none() {
            tutorial_view.set(TutorialView::Toc);
            lesson_idx.set(Some(0));
        }
    })
}

/// "Switch to Paths": clear lesson state and, if no path is
/// active, open the picker (None inner = picker, Some(p) = walk).
pub fn cb_select_paths(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
) -> Callback<MouseEvent> {
    Callback::from(move |_| {
        lesson_idx.set(None);
        if path_state.is_none() {
            path_state.set(Some((None, 0)));
        }
    })
}
