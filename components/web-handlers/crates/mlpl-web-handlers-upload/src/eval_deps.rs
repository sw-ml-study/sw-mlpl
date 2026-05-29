//! The `EvalDeps` bundle of Yew state handles used by every
//! REPL-eval pipeline (submit, upload, demo). Saga 82 extracted
//! it from handlers_submit.rs into this upload sub-crate so the
//! upload_cmd module can take an `&EvalDeps` without a circular
//! dep on handlers-eval.

use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use mlpl_web_eval::state::HistoryEntry;
use yew::prelude::*;

use crate::upload::PendingUploadName;

#[derive(Clone)]
pub struct EvalDeps {
    pub session: Rc<RefCell<WasmSession>>,
    pub history: UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: UseStateHandle<String>,
    pub cmd_history: UseStateHandle<Vec<String>>,
    pub cmd_index: UseStateHandle<Option<usize>>,
    /// Saga 29 step 016: noderef + pending-name handle for the
    /// `:upload <name>` REPL command. `upload_input_ref` points
    /// at the hidden `<input type=file>` rendered by the REPL
    /// controls; `pending_upload_name` is set to `Some(<name>)`
    /// before the file picker fires, so the existing on-change
    /// pipeline knows which session variable to bind.
    pub upload_input_ref: NodeRef,
    pub show_3d: UseStateHandle<bool>,
    pub pending_upload_name: PendingUploadName,
}
