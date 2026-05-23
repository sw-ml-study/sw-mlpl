//! Saga 33 step 007: pure derivation of the per-render active
//! session + history. When a tutorial lesson is open, the
//! tutorial scratchpad is active; otherwise the main scratchpad.
//! Used by every callback builder so they don't re-derive.

use crate::app_state::{ActiveContext, Sessions, UiState};

/// Pick the active session + history pair. Pure: same inputs ->
/// same outputs, no Yew hooks, safe to call anywhere.
pub fn active_context(sessions: &Sessions, ui: &UiState) -> ActiveContext {
    let in_tutorial = ui.lesson_idx.is_some();
    let session = if in_tutorial {
        sessions.tutorial.clone()
    } else {
        sessions.main.clone()
    };
    let history = if in_tutorial {
        ui.tutorial_history.clone()
    } else {
        ui.history.clone()
    };
    ActiveContext {
        session,
        history,
        in_tutorial,
    }
}
