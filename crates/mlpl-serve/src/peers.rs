//! Saga R1 step 003: peer registry. The orchestrator
//! routes `device("<name>") { ... }` blocks to a
//! registered peer that owns the named device.
//! Peers are configured at startup via `--peer
//! <name>=<url>` flags; for R1 the registry is
//! immutable after build (R3 will add dynamic
//! registration + mDNS auto-discovery).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

const PEER_TIMEOUT_SECS: u64 = 600;

/// One registered peer. The blocking reqwest client
/// is built once at startup so peer-routing
/// dispatch in the evaluator is a synchronous call.
#[derive(Clone, Debug)]
pub struct Peer {
    pub url: String,
    pub client: reqwest::blocking::Client,
}

/// Shared peer registry, keyed by device name. R1
/// loopback-only deployment treats this as
/// build-once-at-startup-immutable; R3 will swap to
/// `Arc<RwLock<...>>` for dynamic registration.
pub type PeerRegistry = Arc<HashMap<String, Peer>>;

/// Construct an empty registry.
#[must_use]
pub fn empty_registry() -> PeerRegistry {
    Arc::new(HashMap::new())
}

/// Parse a single `--peer <device>=<url>` argument.
/// Returns `(device_name, url)`. The orchestrator's
/// arg loop calls this once per `--peer` flag; the
/// results feed `build_registry`.
pub fn parse_peer_arg(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((device, url)) if !device.is_empty() && !url.is_empty() => {
            Ok((device.to_string(), url.to_string()))
        }
        _ => Err(format!(
            "--peer: expected <device>=<url>, got {s:?} \
             (e.g. --peer mlx=http://localhost:6465)"
        )),
    }
}

/// Build a `PeerRegistry` from a list of `(device,
/// url)` pairs. Each peer gets its own blocking
/// reqwest client with a 600s timeout (long enough
/// for a device-bound training block; the eval-on-
/// device endpoint blocks until the program
/// finishes).
///
/// # Errors
/// Returns the device name + reason for the first
/// duplicate or malformed URL the caller didn't
/// catch.
pub fn build_registry(
    pairs: Vec<(String, String)>,
    insecure_peers: bool,
) -> Result<PeerRegistry, String> {
    let mut map: HashMap<String, Peer> = HashMap::new();
    for (device, url) in pairs {
        if !insecure_peers && !is_loopback_url(&url) {
            return Err(format!(
                "--peer {device}={url}: non-loopback peer URLs require --insecure-peers \
                 (R1 deployment is loopback-only by default)"
            ));
        }
        if map.contains_key(&device) {
            return Err(format!(
                "--peer {device}={url}: duplicate device {device:?} \
                 (each device may have at most one peer in R1)"
            ));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(PEER_TIMEOUT_SECS))
            .build()
            .map_err(|e| format!("reqwest client for {device}: {e}"))?;
        map.insert(device, Peer { url, client });
    }
    Ok(Arc::new(map))
}

/// True if `url` parses as `http://localhost:...`,
/// `http://127.0.0.1:...`, or `http://[::1]:...`.
/// Anything else (DNS names, public IPs) requires
/// `--insecure-peers`.
fn is_loopback_url(url: &str) -> bool {
    let stripped = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host_part = stripped.split('/').next().unwrap_or("");
    let host = host_part.rsplit_once(':').map_or(host_part, |(h, _)| h);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "0:0:0:0:0:0:0:1")
}

