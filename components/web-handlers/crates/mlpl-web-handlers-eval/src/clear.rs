//! The REPL "Clear" button callback: drops history, asks the
//! shared WasmSession to forget every binding, and fires the
//! Three.js stage reset hook. Extracted from the original
//! handlers.rs facade during saga 82.

use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::HistoryEntry;
use yew::prelude::*;

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
