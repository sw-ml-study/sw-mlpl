//! Saga 33 step 007: custom Yew hooks that allocate the App's
//! reactive state. Each `use_*` hook is small (under 25 LOC),
//! testable in isolation, and only touches the state it owns.
//! Together they replace 11+ inline `use_state` calls that
//! previously bloated `App::app`.

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use yew::prelude::*;

use crate::app_state::{OnboardingState, Sessions, UiState, UploadState};
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
        completion_candidates: use_state(Vec::<String>::new),
        completion_selected: use_state(|| 0_usize),
    }
}

/// Allocate onboarding overlay state from localStorage.
#[hook]
pub fn use_onboarding_state() -> OnboardingState {
    let dismissed = crate::onboarding_storage::read_splash_dismissed();
    OnboardingState {
        show_splash: use_state(move || !dismissed),
        show_tour: use_state(|| false),
        tour_step: use_state(|| 0_usize),
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

/// Wire a window-level Escape-key handler that closes the
/// doc dialog and the tutorial panel when either is open.
/// Same pattern the glossary popup already uses
/// (`crate::glossary_popup`) -- single window listener,
/// fired once at mount, closures `forget`ed because the App
/// host is mounted for the session lifetime. Setting a state
/// handle to its current value is a no-op, so the unconditional
/// `.set` calls don't cause spurious renders.
#[hook]
pub fn use_escape_closes_dialogs(
    dialog_open: UseStateHandle<bool>,
    lesson_idx: UseStateHandle<Option<usize>>,
) {
    use_effect_with((), move |_| {
        if let Some(window) = web_sys::window() {
            let closure = Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
                if e.key() == "Escape" {
                    dialog_open.set(false);
                    lesson_idx.set(None);
                }
            }) as Box<dyn FnMut(_)>);
            let _ = window
                .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
            closure.forget();
        }
        || ()
    });
}
