//! Saga 33 step 007: state-bag struct definitions for the App
//! component. Each struct is constructed by a custom hook in
//! `app_hooks` and read by downstream phases (active_context,
//! build_callbacks, render). The struct-of-structs decomposition
//! preserves type safety: every consumer takes the *specific*
//! sub-state it needs, no `&AppState` god-object passing.

use std::cell::RefCell;
use std::rc::Rc;

use yew::prelude::*;

use crate::components::TutorialView;
use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::HistoryEntry;

/// Two isolated session scratchpads: the main REPL and the
/// tutorial pane. Each has its own `WasmSession` + history so
/// experimentation in the tutorial doesn't pollute the user's
/// main flow.
pub struct Sessions {
    pub main: Rc<RefCell<WasmSession>>,
    pub tutorial: Rc<RefCell<WasmSession>>,
}

/// Reactive UI state allocated via `use_state` hooks. All Yew
/// reactivity flows through these handles.
pub struct UiState {
    pub history: UseStateHandle<Vec<HistoryEntry>>,
    pub tutorial_history: UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: UseStateHandle<String>,
    pub cmd_history: UseStateHandle<Vec<String>>,
    pub cmd_index: UseStateHandle<Option<usize>>,
    pub dialog_open: UseStateHandle<bool>,
    pub lesson_idx: UseStateHandle<Option<usize>>,
    pub tutorial_initial_view: UseStateHandle<TutorialView>,
    pub path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    /// Saga 33 step 043: candidates for the Tab-completion
    /// popup below the REPL input. Empty Vec = popup hidden.
    pub completion_candidates: UseStateHandle<Vec<String>>,
    /// Saga 33 step 047: highlighted index in the completion
    /// popup. Resets to 0 when new candidates land.
    pub completion_selected: UseStateHandle<usize>,
}

/// Upload-flow state: the hidden `<input type=file>` NodeRef and
/// the pending-name slot that bridges the slash-command handler
/// (sync) and the file picker's async onchange. Saga 29 step 016.
pub struct UploadState {
    pub input_ref: NodeRef,
    pub pending_name: UseStateHandle<Option<String>>,
}

/// Per-render derived "which session and history are active".
/// Computed from `Sessions` + `UiState::lesson_idx`. Owned (cloned)
/// so callback builders can keep their own captures.
pub struct ActiveContext {
    pub session: Rc<RefCell<WasmSession>>,
    pub history: UseStateHandle<Vec<HistoryEntry>>,
    pub in_tutorial: bool,
}
