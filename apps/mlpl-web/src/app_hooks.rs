//! Saga 33 step 007: custom Yew hooks that allocate the App's
//! reactive state. Each `use_*` hook is small (under 25 LOC),
//! testable in isolation, and only touches the state it owns.
//! Together they replace 11+ inline `use_state` calls that
//! previously bloated `App::app`.

use yew::prelude::*;

use crate::app_state::{Sessions, UiState, UploadState};
use crate::components::TutorialView;
use crate::scroll::scroll_and_focus;
use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::HistoryEntry;

/// Allocate the two isolated session scratchpads (main + tutorial)
/// via `use_mut_ref`. Stable identity across renders.
#[hook]
pub fn use_sessions() -> Sessions {
    Sessions {
        main: use_mut_ref(WasmSession::new),
        tutorial: use_mut_ref(WasmSession::new),
    }
}

/// Allocate every reactive UI state slot in one `#[hook]`. Calling
/// this from `app()` is hooks-rules-safe because the inner
/// `use_state` calls happen in a fixed order on every render.
#[hook]
pub fn use_ui_state() -> UiState {
    UiState {
        history: use_state(Vec::<HistoryEntry>::new),
        tutorial_history: use_state(Vec::<HistoryEntry>::new),
        input_value: use_state(String::new),
        cmd_history: use_state(Vec::<String>::new),
        cmd_index: use_state(|| None::<usize>),
        dialog_open: use_state(|| false),
        lesson_idx: use_state(|| None::<usize>),
        tutorial_initial_view: use_state(|| TutorialView::Toc),
        path_state: use_state(|| None::<(Option<usize>, usize)>),
    }
}

/// Allocate the upload-flow NodeRef + pending-name slot used by
/// the `:upload <name>` slash command + the file picker.
#[hook]
pub fn use_upload_state() -> UploadState {
    UploadState {
        input_ref: use_node_ref(),
        pending_name: use_state(|| None::<String>),
    }
}

/// Mount the side-effect that scrolls the output pane to its
/// bottom and refocuses the REPL input after every history
/// append. Keyed on the active history handle so the effect
/// re-fires when the user switches between tutorial / main.
#[hook]
pub fn use_scroll_effect(active_history: UseStateHandle<Vec<HistoryEntry>>) {
    use_effect_with(active_history, |_| {
        scroll_and_focus();
        || ()
    });
}
