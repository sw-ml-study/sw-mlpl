//! Shared render-input types: `RenderArgs` (the root struct
//! threaded through every layer), `Modes` (derived booleans +
//! HeaderMode), and `InputCallbacks` (the input-row / dialog
//! event bundle).
//!
//! Saga 82 step 8 carved these out of the render-core +
//! render-shell sub-crates so each can carry the OTHER's
//! function bodies without a cycle: shell calls into core for
//! the helpers, core stops at structs.

use mlpl_web_components_chrome::header::HeaderMode;
use yew::prelude::*;

use crate::app_callbacks::AppCallbacks;
use crate::state::{ActiveContext, OnboardingState, UiState, UploadState};

/// Top-level render input. Grouped into sub-structs so each
/// downstream helper takes only the slice it needs.
pub struct RenderArgs {
    pub callbacks: AppCallbacks,
    pub ui: UiState,
    pub upload: UploadState,
    pub active: ActiveContext,
    pub onboarding: OnboardingState,
    /// Pre-formatted build-info chip ("vX.Y.Z.<commits> ·
    /// <host> · <sha> · <timestamp>") for the footer; passed
    /// in from mlpl-web because only mlpl-web's build.rs
    /// emits the BUILD_* env vars.
    pub build_info: AttrValue,
    /// Pre-formatted version label ("vX.Y.Z.<commits>") for
    /// the splash overlay, also sourced from mlpl-web's
    /// build env vars.
    pub version_label: AttrValue,
}

impl RenderArgs {
    /// Pack the four sub-structures into one render input.
    #[must_use]
    pub fn from_parts(
        callbacks: AppCallbacks,
        ui: UiState,
        upload: UploadState,
        active: ActiveContext,
        onboarding: OnboardingState,
        build_info: AttrValue,
        version_label: AttrValue,
    ) -> Self {
        Self {
            callbacks,
            ui,
            upload,
            active,
            onboarding,
            build_info,
            version_label,
        }
    }
}

/// Derived booleans + HeaderMode for the current render.
/// HeaderMode lives in mlpl-web-components-chrome; it's wired
/// in by the caller (modes::derive_modes) so this types crate
/// stays a pure data crate with no Yew-component dep.
pub struct Modes {
    pub cur_lesson: Option<usize>,
    pub cur_path: Option<(Option<usize>, usize)>,
    pub tutorial_active: bool,
    pub paths_active: bool,
    pub editor_active: bool,
    pub header_mode: HeaderMode,
}

/// Bundle of input-row event callbacks + dialog toggles needed
/// by the outer html! shell.
pub struct InputCallbacks {
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<web_sys::KeyboardEvent>,
    pub open_dialog: Callback<MouseEvent>,
    pub close_dialog: Callback<MouseEvent>,
}
