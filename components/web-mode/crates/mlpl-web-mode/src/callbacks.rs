//! Saga 33 step 006: bundle of all six mode/path callbacks. The
//! builder fns live in `mode_select` (3 header-tab callbacks) and
//! `mode_path` (3 PathsView->parent callbacks); this file holds
//! the `ModeCallbacks` struct and the `bundle` composer that
//! produces it.

use yew::prelude::*;

use crate::path::{cb_path_change, cb_path_open_lesson, cb_path_run_demo};
use crate::select::{cb_select_paths, cb_select_repl, cb_select_tutorial};
use mlpl_web_tutorial::TutorialView;

/// Flat record of the six callbacks consumed by the top-level
/// shell html!. Kept as a flat struct so call sites can pattern-
/// bind individual fields; the *production* of the fields is
/// decomposed into per-callback builders in the sibling files.
pub struct ModeCallbacks {
    pub repl: Callback<MouseEvent>,
    pub tutorial: Callback<MouseEvent>,
    pub paths: Callback<MouseEvent>,
    pub editor: Callback<MouseEvent>,
    pub path_change: Callback<Option<(Option<usize>, usize)>>,
    pub path_open_lesson: Callback<usize>,
    pub path_run_demo: Callback<String>,
}

/// Compose all six mode/path callbacks into one bundle. This is
/// the only place that knows the full set; individual `cb_*`
/// builders are independently usable for testing or future
/// reorganization.
pub fn bundle(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    tutorial_view: UseStateHandle<TutorialView>,
    on_demo: Callback<usize>,
    editor_open: UseStateHandle<bool>,
) -> ModeCallbacks {
    let ed = editor_open.clone();
    let ed_lesson = lesson_idx.clone();
    let ed_path = path_state.clone();
    ModeCallbacks {
        repl: {
            let ed = editor_open.clone();
            let r = cb_select_repl(lesson_idx.clone(), path_state.clone());
            Callback::from(move |e: MouseEvent| {
                ed.set(false);
                r.emit(e);
            })
        },
        tutorial: cb_select_tutorial(
            lesson_idx.clone(),
            path_state.clone(),
            tutorial_view.clone(),
        ),
        paths: cb_select_paths(lesson_idx.clone(), path_state.clone()),
        editor: Callback::from(move |_: MouseEvent| {
            ed_lesson.set(None);
            ed_path.set(None);
            ed.set(true);
        }),
        path_change: cb_path_change(path_state),
        path_open_lesson: cb_path_open_lesson(lesson_idx, tutorial_view),
        path_run_demo: cb_path_run_demo(on_demo),
    }
}
