//! Saga 33 step 007: assemble the six top-level evaluator
//! callbacks from active session + ui + upload state. Each
//! callback is built by a domain-specific factory
//! (`handlers::make_*` or `upload::make_*`); this file's job
//! is just to wire the right state into each one and bundle
//! the results.

use yew::prelude::*;

use crate::app_state::{ActiveContext, UiState, UploadState};
use crate::handlers::{EvalDeps, make_clear, make_run_demo, make_submit, make_submit_batch};
use crate::upload;

/// Bundle of every top-level evaluator callback consumed by the
/// render shell.
pub struct AppCallbacks {
    pub on_submit: Callback<String>,
    pub on_run_batch: Callback<Vec<String>>,
    pub on_clear: Callback<MouseEvent>,
    pub on_demo: Callback<usize>,
    pub on_upload: Callback<web_sys::Event>,
    pub on_upload_cancel: Callback<web_sys::Event>,
}

/// Wire `active + ui + upload` state into each callback factory
/// and return the bundle. Pure (no Yew hooks); safe to call from
/// the component body without affecting hook order.
pub fn build_callbacks(active: &ActiveContext, upload: &UploadState, ui: &UiState) -> AppCallbacks {
    let deps = build_eval_deps(active, upload, ui);
    AppCallbacks {
        on_submit: make_submit(deps.clone()),
        on_run_batch: make_submit_batch(deps),
        on_clear: make_clear(active.session.clone(), active.history.clone()),
        on_demo: make_run_demo(active.session.clone(), active.history.clone()),
        on_upload: upload::make_upload_image(
            active.session.clone(),
            active.history.clone(),
            upload.pending_name.clone(),
        ),
        on_upload_cancel: upload::make_upload_cancel(
            active.session.clone(),
            active.history.clone(),
            upload.pending_name.clone(),
        ),
    }
}

/// Compose the `EvalDeps` struct passed to `make_submit` /
/// `make_submit_batch`. Splitting this out keeps
/// `build_callbacks` short and makes the `EvalDeps` field-by-
/// field wiring auditable in one place.
fn build_eval_deps(active: &ActiveContext, upload: &UploadState, ui: &UiState) -> EvalDeps {
    EvalDeps {
        session: active.session.clone(),
        history: active.history.clone(),
        input_value: ui.input_value.clone(),
        cmd_history: ui.cmd_history.clone(),
        cmd_index: ui.cmd_index.clone(),
        upload_input_ref: upload.input_ref.clone(),
        pending_upload_name: upload.pending_name.clone(),
        show_3d: ui.show_3d.clone(),
    }
}
