//! Foundation of the mlpl-serve crate family: auth, device listing,
//! the session store, TLS plumbing, the viz byte store, and the
//! wire/response types every layer shares.

pub mod auth;
pub mod devices;
pub mod eval_viz;
pub mod router_layers;
pub mod sessions;
pub mod store;

/// Path-compat alias: TLS helpers live in `auth` (merged module).
pub use auth as tls;
