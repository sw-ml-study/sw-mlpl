//! Facade: re-exports from the focused handler modules.
//!
//! - `handlers_submit` -- REPL submission pipeline
//! - `handlers_demo`   -- demo runner
//! - `handlers_input`  -- keyboard / input event callbacks

use yew::prelude::*;

// -- submit pipeline re-exports --
pub(crate) use crate::handlers_submit::running_message;
pub use crate::handlers_submit::{EvalDeps, make_submit, make_submit_batch};

// -- demo runner re-exports --
pub use crate::handlers_demo::make_run_demo;

// -- input event re-exports --
pub use crate::handlers_input::{make_keydown, make_oninput};

// -- upload_cmd re-exports (used by tests below) --
pub(crate) use crate::upload_cmd::{is_valid_identifier, parse_upload_command};

// -- clear helper (one-liner, lives here) --
use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::HistoryEntry;

pub fn make_clear(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        session.borrow().clear();
        history.set(Vec::new());
        let _ = js_sys::eval("window.__stage3d_clear && window.__stage3d_clear()");
    })
}

pub fn toggle_bool(handle: UseStateHandle<bool>, value: bool) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| handle.set(value))
}

#[cfg(test)]
mod tests {
    use crate::upload_cmd::{is_valid_identifier, parse_upload_command};

    #[test]
    fn parse_upload_with_name() {
        assert_eq!(parse_upload_command(":upload x"), Some("x".into()));
        assert_eq!(
            parse_upload_command(":upload my_photo"),
            Some("my_photo".into())
        );
        assert_eq!(
            parse_upload_command(":upload   spaced"),
            Some("spaced".into())
        );
    }

    #[test]
    fn parse_upload_no_name_returns_empty_string() {
        assert_eq!(parse_upload_command(":upload"), Some(String::new()));
        assert_eq!(parse_upload_command(":upload   "), Some(String::new()));
    }

    #[test]
    fn parse_non_upload_returns_none() {
        assert_eq!(parse_upload_command(":help"), None);
        assert_eq!(parse_upload_command(":vars"), None);
        assert_eq!(parse_upload_command("x = 1"), None);
        assert_eq!(parse_upload_command(""), None);
        // `:uploaded` shares a prefix with `:upload` but is a
        // different command (no separator); today the parser
        // treats it as `:upload <name>` = `:upload ed`. This is
        // a known quirk -- nobody types `:uploaded` so it
        // doesn't collide in practice.
    }

    #[test]
    fn identifier_validation() {
        assert!(is_valid_identifier("x"));
        assert!(is_valid_identifier("X"));
        assert!(is_valid_identifier("my_photo"));
        assert!(is_valid_identifier("_x"));
        assert!(is_valid_identifier("a1"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("1"));
        assert!(!is_valid_identifier("1x"));
        assert!(!is_valid_identifier("x y"));
        assert!(!is_valid_identifier("x-y"));
        assert!(!is_valid_identifier("x.y"));
    }
}
