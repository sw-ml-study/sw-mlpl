//! 3D-stage helpers for the mlpl-web REPL. Extracted from
//! mlpl-web during saga 80. Three small modules:
//!
//! - `events` -- shape-info builders + the JS stage-event
//!   emitter consumed by the inline Three.js client.
//! - `panel` -- the Yew `Stage3dPanel` function component that
//!   hosts the `<canvas>` and the `__stage3d_*` extern hooks.
//! - `toggle` -- pure `:3d` REPL command parser and the
//!   Ctrl+3 hotkey predicate.
//!
//! mlpl-web re-exports each module under its original
//! `crate::viz3d_*` name so handlers + render call sites
//! keep working unchanged.

pub mod events;
pub mod panel;
pub mod toggle;
