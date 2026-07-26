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

// ---- TLS plumbing: crypto provider, PEM loading, self-signed certs ----
// (merged from tls.rs; re-exported as `tls` from lib.rs so
// `mlpl_serve_core::tls::*` / `mlpl_serve::tls::*` paths still work).
pub type TlsConfig = Option<axum_server::tls_rustls::RustlsConfig>;

use axum_server::tls_rustls::RustlsConfig;
use rcgen::{CertifiedKey, generate_simple_self_signed};
use sha2::{Digest, Sha256};

/// Install the `ring` crypto provider as rustls's
/// process-wide default. Idempotent. Required since
/// rustls 0.23 dropped its built-in default and forces
/// an explicit `ring` vs `aws-lc-rs` choice.
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// `RustlsConfig` from PEM cert + key on disk.
pub async fn from_pem_files(
    cert: &std::path::Path,
    key: &std::path::Path,
) -> Result<RustlsConfig, String> {
    ensure_crypto_provider();
    RustlsConfig::from_pem_file(cert, key)
        .await
        .map_err(|e| format!("TLS: failed to load cert/key: {e}"))
}

/// Self-signed cert covering loopback hosts, returned
/// as `(RustlsConfig, fingerprint)`. Fingerprint is
/// SHA-256 of the DER-encoded cert in colon-separated
/// uppercase hex.
pub async fn self_signed_loopback() -> Result<(RustlsConfig, String), String> {
    ensure_crypto_provider();
    let alt = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let CertifiedKey { cert, key_pair } = generate_simple_self_signed(alt)
        .map_err(|e| format!("TLS: failed to generate self-signed cert: {e}"))?;
    let der = cert.der().to_vec();
    let mut fp = String::with_capacity(95);
    for (i, b) in Sha256::digest(&der).iter().enumerate() {
        if i > 0 {
            fp.push(':');
        }
        std::fmt::Write::write_fmt(&mut fp, format_args!("{b:02X}")).unwrap();
    }
    let config = RustlsConfig::from_pem(
        cert.pem().into_bytes(),
        key_pair.serialize_pem().into_bytes(),
    )
    .await
    .map_err(|e| format!("TLS: failed to assemble rustls config: {e}"))?;
    Ok((config, fp))
}
