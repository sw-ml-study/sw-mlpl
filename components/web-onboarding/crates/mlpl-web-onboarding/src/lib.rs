//! First-launch onboarding surface for the web playground.
//! Four small Yew components carved out of mlpl-web during
//! saga 82: a splash overlay, the guided tour tooltip, a
//! "what's new" overlay tied to the last-seen version, and a
//! tiny localStorage shim shared by the other three.
//!
//! mlpl-web re-exports each module under its original
//! `crate::onboarding_*` namespace so existing call sites stay
//! unchanged.

pub mod splash;
pub mod storage;
pub mod tour;
pub mod whats_new;
