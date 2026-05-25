//! mlpl-web top-level render. Saga 33 step 007: the outer
//! shell was a 67-LOC `render()` body; it now decomposes into
//! - `RenderArgs::from_parts(...)` (struct-of-structs assembly)
//! - `build_input_callbacks(...)` (oninput / onkeydown / dialog
//!   toggles)
//! - `derive_modes(...)` (which mode is active + HeaderMode)
//! - `render_shell(...)` (the outer html! template)

use yew::prelude::*;

use crate::app_callbacks::AppCallbacks;
use crate::app_state::{ActiveContext, OnboardingState, UiState, UploadState};
use crate::render_callbacks::build_input_callbacks;
use crate::render_modes::derive_modes;
use crate::render_shell::render_shell;

/// Top-level render input. Grouped into sub-structs so each
/// downstream helper takes only the slice it needs (Type-safe
/// "struct of structs" -- no god-object `&RenderArgs`
/// threading).
pub struct RenderArgs {
    pub callbacks: AppCallbacks,
    pub ui: UiState,
    pub upload: UploadState,
    pub active: ActiveContext,
    pub onboarding: OnboardingState,
}

impl RenderArgs {
    /// Pack the four sub-structures into one render input. Kept
    /// as a named constructor (vs struct literal) so `app.rs`
    /// stays a clean 1-line `render(RenderArgs::from_parts(...))`.
    #[must_use]
    pub fn from_parts(
        callbacks: AppCallbacks,
        ui: UiState,
        upload: UploadState,
        active: ActiveContext,
        onboarding: OnboardingState,
    ) -> Self {
        Self {
            callbacks,
            ui,
            upload,
            active,
            onboarding,
        }
    }
}

pub fn render(a: RenderArgs) -> Html {
    let inputs = build_input_callbacks(&a.callbacks, &a.ui);
    let modes = derive_modes(&a.ui);
    render_shell(a, inputs, modes)
}
