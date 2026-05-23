//! Saga 33 step 007: per-render mode derivation. Reads
//! `ui.lesson_idx` + `ui.path_state` and produces the booleans
//! + `HeaderMode` enum the outer shell needs. Pure; no Yew hooks.

use crate::app_state::UiState;
use crate::components::HeaderMode;

/// Derived booleans + HeaderMode for the current render. Carved
/// out of the inline derivation that bloated `render()`.
pub struct Modes {
    pub cur_lesson: Option<usize>,
    pub cur_path: Option<(Option<usize>, usize)>,
    pub tutorial_active: bool,
    pub paths_active: bool,
    pub header_mode: HeaderMode,
}

/// Derive the per-render Modes bag from UiState. Pure: same
/// inputs -> same outputs.
pub fn derive_modes(ui: &UiState) -> Modes {
    let cur_lesson = *ui.lesson_idx;
    let cur_path = *ui.path_state;
    let tutorial_active = cur_lesson.is_some();
    let paths_active = cur_path.is_some();
    let header_mode = pick_header_mode(tutorial_active, paths_active);
    Modes {
        cur_lesson,
        cur_path,
        tutorial_active,
        paths_active,
        header_mode,
    }
}

fn pick_header_mode(tutorial_active: bool, paths_active: bool) -> HeaderMode {
    if paths_active {
        HeaderMode::Paths
    } else if tutorial_active {
        HeaderMode::Tutorial
    } else {
        HeaderMode::Repl
    }
}
