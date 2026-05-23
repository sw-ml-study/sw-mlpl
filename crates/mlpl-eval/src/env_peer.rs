//! Saga 33 step 003: peer-dispatcher install / clear / borrow
//! methods extracted from `env.rs`. The `PeerDispatcher` trait
//! itself stays in `env.rs` (the `Environment` struct holds an
//! `Option<Arc<dyn PeerDispatcher>>` field that this module
//! mutates).

use std::sync::Arc;

use crate::env::{Environment, PeerDispatcher};

impl Environment {
    pub fn set_peer_dispatcher(&mut self, dispatcher: Arc<dyn PeerDispatcher>) {
        self.peer_dispatcher = Some(dispatcher);
    }

    pub fn clear_peer_dispatcher(&mut self) {
        self.peer_dispatcher = None;
    }

    #[must_use]
    pub fn peer_dispatcher(&self) -> Option<Arc<dyn PeerDispatcher>> {
        self.peer_dispatcher.clone()
    }
}
