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
    /// Build-stamp labels (footer chip, splash version, splash
    /// UTC-Z build time), sourced from mlpl-web because only its
    /// build.rs emits the BUILD_* env vars.
    pub build: BuildLabels,
}

/// The pre-formatted build-stamp strings threaded from mlpl-web.
pub struct BuildLabels {
    /// Footer chip: "vX.Y.Z.<commits> · <host> · <sha> · <time>".
    pub info: AttrValue,
    /// Splash version label: "vX.Y.Z.<commits>".
    pub version: AttrValue,
    /// Splash build time in UTC Zulu ("YYYY-MM-DDTHH:MM:SSZ") so a
    /// viewer can identify the build regardless of local timezone.
    pub time: AttrValue,
}

impl RenderArgs {
    /// Pack the sub-structures into one render input.
    #[must_use]
    pub fn from_parts(
        callbacks: AppCallbacks,
        ui: UiState,
        upload: UploadState,
        active: ActiveContext,
        onboarding: OnboardingState,
        build: BuildLabels,
    ) -> Self {
        Self {
            callbacks,
            ui,
            upload,
            active,
            onboarding,
            build,
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
