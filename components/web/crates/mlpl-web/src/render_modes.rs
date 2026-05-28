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
    pub editor_active: bool,
    pub header_mode: HeaderMode,
}

pub fn derive_modes(ui: &UiState) -> Modes {
    compute_modes(*ui.lesson_idx, *ui.path_state, *ui.editor_open)
}

pub fn compute_modes(
    cur_lesson: Option<usize>,
    cur_path: Option<(Option<usize>, usize)>,
    editor_open: bool,
) -> Modes {
    let tutorial_active = cur_lesson.is_some();
    let paths_active = cur_path.is_some() && !tutorial_active;
    let editor_active = editor_open && !tutorial_active && !paths_active;
    let header_mode = if editor_active {
        HeaderMode::Editor
    } else if paths_active {
        HeaderMode::Paths
    } else if tutorial_active {
        HeaderMode::Tutorial
    } else {
        HeaderMode::Repl
    };
    Modes {
        cur_lesson,
        cur_path,
        tutorial_active,
        paths_active,
        editor_active,
        header_mode,
    }
}

#[cfg(test)]
mod tests {
    //! Saga 33 step 042: path-resume bug fix. The key
    //! invariant: an in-progress `cur_path` survives across
    //! lesson navigation, but the WALKER pane only renders
    //! when no lesson is open -- so the lesson pane wins the
    //! screen while the walker position is held in state for
    //! the "Back to path" button to resume.
    use super::compute_modes;
    use crate::components::HeaderMode;

    #[test]
    fn cold_start_is_repl_mode() {
        let m = compute_modes(None, None, false);
        assert!(!m.tutorial_active);
        assert!(!m.paths_active);
        assert_eq!(m.header_mode, HeaderMode::Repl);
    }

    #[test]
    fn lesson_open_alone_is_tutorial_mode() {
        let m = compute_modes(Some(3), None, false);
        assert!(m.tutorial_active);
        assert!(!m.paths_active);
        assert_eq!(m.header_mode, HeaderMode::Tutorial);
    }

    #[test]
    fn path_open_alone_is_paths_mode() {
        let m = compute_modes(None, Some((Some(0), 2)), false);
        assert!(!m.tutorial_active);
        assert!(m.paths_active);
        assert_eq!(m.header_mode, HeaderMode::Paths);
    }

    #[test]
    fn lesson_open_with_path_in_progress_renders_lesson_not_walker() {
        // The path-resume bug fix: when both states are set
        // (user navigated to a lesson FROM a path walker),
        // the lesson pane wins. `cur_path` is preserved for
        // "Back to path" restoration.
        let m = compute_modes(Some(5), Some((Some(0), 2)), false);
        assert!(m.tutorial_active);
        assert!(
            !m.paths_active,
            "paths walker must NOT render while a lesson is open"
        );
        assert!(
            m.cur_path.is_some(),
            "path position must be preserved for resume"
        );
        assert_eq!(m.header_mode, HeaderMode::Tutorial);
    }
}
