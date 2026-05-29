//! `derive_modes` -- per-render derivation of the active
//! Tutorial/Paths/Editor/REPL pane + matching HeaderMode. Pure;
//! no Yew hooks. Saga 82 moved the `Modes` struct definition
//! into `mlpl-web-render-types::args` so the shell crate (which
//! also needs the struct) can import it without depending on
//! render-core.

use mlpl_web_components_chrome::header::HeaderMode;
use mlpl_web_render_types::args::Modes;
use mlpl_web_render_types::state::UiState;

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
