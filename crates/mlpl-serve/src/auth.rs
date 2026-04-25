//! Bearer-token authentication for the eval
//! endpoint. Saga 21 step 001.
//!
//! `AuthMode::Required` is the default; `Disabled`
//! is loopback-only and exists for ergonomic local
//! testing. The middleware attaches to routes that
//! need it (created in `server::build_app`); no
//! per-handler auth checks.

use axum::http::HeaderMap;
use subtle::ConstantTimeEq;

/// Whether the eval endpoint requires a bearer
/// token. `--bind 0.0.0.0` requires `Required`;
/// `Disabled` is rejected at startup for non-
/// loopback binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthMode {
    /// Eval requires a valid `Authorization: Bearer
    /// <token>` matching the session's token.
    Required,
    /// Eval skips the bearer check. Loopback only.
    Disabled,
}

/// Pull the token out of an `Authorization: Bearer
/// <token>` header. Returns `None` if the header is
/// missing, not parseable as ASCII, or doesn't start
/// with the case-sensitive prefix `Bearer `.
pub fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get("authorization")?.to_str().ok()?;
    value.strip_prefix("Bearer ")
}

/// Constant-time compare of a provided token against
/// the expected token. Same length is a precondition
/// of equality; `ct_eq` already short-circuits at
/// the type level via `subtle::Choice`.
#[must_use]
pub fn check_token(provided: &str, expected: &str) -> bool {
    provided.as_bytes().ct_eq(expected.as_bytes()).into()
}
