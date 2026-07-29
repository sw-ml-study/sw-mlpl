//! Saga 33 step 003: peer-dispatcher install / clear / borrow
//! methods extracted from `env.rs`. The `PeerDispatcher` trait
//! itself stays in `env.rs` (the `Environment` struct holds an
//! `Option<Arc<dyn PeerDispatcher>>` field that this module
//! mutates).

use std::sync::Arc;

use mlpl_eval_env::env::{Environment, PeerDispatcher};

impl EnvPeer for Environment {
    fn set_peer_dispatcher(&mut self, dispatcher: Arc<dyn PeerDispatcher>) {
        self.peer_dispatcher = Some(dispatcher);
    }

    fn clear_peer_dispatcher(&mut self) {
        self.peer_dispatcher = None;
    }

    fn peer_dispatcher(&self) -> Option<Arc<dyn PeerDispatcher>> {
        self.peer_dispatcher.clone()
    }
}

/// Peer dispatcher installation for device-routed blocks.
pub trait EnvPeer {
    fn set_peer_dispatcher(&mut self, dispatcher: Arc<dyn PeerDispatcher>);
    fn clear_peer_dispatcher(&mut self);
    fn peer_dispatcher(&self) -> Option<Arc<dyn PeerDispatcher>>;
}
