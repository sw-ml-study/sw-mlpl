//! Shared render-input types: app/session state, the
//! AppCallbacks bundle, and the RenderArgs / Modes /
//! InputCallbacks struct triple that the render-core and
//! render-shell sub-crates both depend on. Saga 82 step 8
//! carved these into a leaf crate to break the core <-> shell
//! cycle.

pub mod app_callbacks;
pub mod args;
pub mod state;
