//! Floor of the web eval stack (spike step 015): shared state,
//! eval wire types, connect-URL parsing/guarding, device gating,
//! and the loss_trace store.

pub mod connect_guard;
pub mod devices;
pub mod eval_url;
pub mod loss_trace;
pub mod narration;
pub mod state;
pub mod wire;
