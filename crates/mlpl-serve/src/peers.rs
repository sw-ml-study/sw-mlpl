//! Saga R1 step 003: peer registry. The orchestrator
//! routes `device("<name>") { ... }` blocks to a
//! registered peer that owns the named device.
//! Peers are configured at startup via `--peer
//! <name>=<url>` flags; for R1 the registry is
//! immutable after build (R3 will add dynamic
//! registration + mDNS auto-discovery).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use mlpl_array::{DenseArray, Shape};
use mlpl_eval::EvalError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const PEER_TIMEOUT_SECS: u64 = 600;

/// One registered peer. The blocking reqwest client
/// is built once at startup so peer-routing
/// dispatch in the evaluator is a synchronous call.
#[derive(Clone, Debug)]
pub struct Peer {
    pub url: String,
    pub client: &'static reqwest::blocking::Client,
}

/// Shared peer registry, keyed by device name. R1
/// loopback-only deployment treats this as
/// build-once-at-startup-immutable; R3 will swap to
/// `Arc<RwLock<...>>` for dynamic registration.
pub type PeerRegistry = Arc<HashMap<String, Peer>>;

#[derive(Clone, Debug, Default)]
pub struct PeerSessionMap {
    inner: Arc<Mutex<HashMap<String, PeerSession>>>,
}

#[derive(Clone, Debug)]
pub struct PeerSession {
    pub id: Uuid,
    pub token: String,
}

#[derive(Deserialize)]
struct CreateSessionResponse {
    session_id: Uuid,
    token: String,
}

#[derive(Serialize, Deserialize)]
struct WireEnvelope {
    version: u32,
    dtype: u8,
    ndim: u8,
    shape: Vec<u64>,
    data: Vec<u8>,
}

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
        let client = std::thread::spawn(|| {
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(PEER_TIMEOUT_SECS))
                .build()
        })
        .join()
        .map_err(|_| format!("reqwest client for {device}: builder thread panicked"))?
        .map_err(|e| format!("reqwest client for {device}: {e}"))?;
        let client = Box::leak(Box::new(client));
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

impl PeerSessionMap {
    pub fn get_or_create(&self, peer: &Peer) -> Result<PeerSession, EvalError> {
        if let Some(session) = self.inner.lock().unwrap().get(&peer.url).cloned() {
            return Ok(session);
        }
        let url = format!("{}/v1/sessions", peer.url.trim_end_matches('/'));
        let client = peer.client;
        let resp = std::thread::spawn(move || {
            client
                .post(url)
                .send()
                .map_err(|e| EvalError::Unsupported(format!("remote peer request: {e}")))?
                .error_for_status()
                .map_err(|e| EvalError::Unsupported(format!("remote peer request: {e}")))?
                .json::<CreateSessionResponse>()
                .map_err(|e| EvalError::Unsupported(format!("remote peer request: {e}")))
        })
        .join()
        .map_err(|_| EvalError::Unsupported("remote peer thread panicked".into()))??;
        let session = PeerSession {
            id: resp.session_id,
            token: resp.token,
        };
        self.inner
            .lock()
            .unwrap()
            .insert(peer.url.clone(), session.clone());
        Ok(session)
    }
}

pub fn encode_bindings(
    bindings: HashMap<String, DenseArray>,
) -> Result<Vec<crate::server::EvalOnDeviceBinding>, EvalError> {
    bindings
        .into_iter()
        .map(|(name, arr)| {
            let shape = arr.shape().dims().iter().map(|d| *d as u64).collect();
            let data = arr.data().iter().flat_map(|v| v.to_le_bytes()).collect();
            let bytes = bincode::serialize(&WireEnvelope {
                version: 1,
                dtype: 0,
                ndim: arr.rank() as u8,
                shape,
                data,
            })
            .map_err(|e| EvalError::Unsupported(format!("wire encode: {e}")))?;
            Ok(crate::server::EvalOnDeviceBinding {
                name,
                tensor: BASE64.encode(bytes),
            })
        })
        .collect()
}

pub fn decode_from_json(s: &str) -> Result<DenseArray, EvalError> {
    let bytes = BASE64
        .decode(s)
        .map_err(|e| EvalError::Unsupported(format!("wire base64: {e}")))?;
    let env: WireEnvelope = bincode::deserialize(&bytes)
        .map_err(|e| EvalError::Unsupported(format!("wire decode: {e}")))?;
    if env.version != 1 || env.dtype != 0 || env.shape.len() != env.ndim as usize {
        return Err(EvalError::Unsupported("wire envelope mismatch".into()));
    }
    let dims: Vec<usize> = env.shape.iter().map(|d| *d as usize).collect();
    if env.data.len() != dims.iter().product::<usize>() * 8 {
        return Err(EvalError::Unsupported("wire data length mismatch".into()));
    }
    let data = env
        .data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    DenseArray::new(Shape::new(dims), data).map_err(EvalError::from)
}
