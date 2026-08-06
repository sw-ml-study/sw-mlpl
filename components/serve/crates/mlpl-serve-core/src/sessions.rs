//! In-memory session storage. Saga 21 step 001.
//!
//! `Session` owns a fresh `mlpl_eval::Environment`;
//! the eval handler holds a write lock on the map
//! for the duration of the eval call (because
//! `Environment` is mutated by side-effecting
//! builtins). A future saga can swap the storage
//! to disk-backed for persistence.
//!
//! Saga 21.5 step 003: an `InterruptMap` lives
//! alongside the `SessionMap` to keep `/cancel`
//! reachable while an eval holds the session write
//! lock (`/eval`) or has removed the session for the
//! duration of the call (`/eval_stream`). The entry
//! duplicates the bearer token so cancel-time auth
//! does not have to take the sessions lock.

use std::collections::HashMap;
use std::sync::Arc;

use mlpl_eval::{Environment, Interrupt};
use rand::Rng;
use rand::distributions::Alphanumeric;
use tokio::sync::RwLock;
use uuid::Uuid;

use axum::Json;
use axum::http::{HeaderMap, StatusCode};

use crate::auth::{AuthMode, check_token, extract_bearer};
use crate::eval_viz::{ErrorResponse, json_err};

const TOKEN_LEN: usize = 32;

/// One client session. Holds the bearer token and a
/// long-lived `Environment` that accumulates state
/// across eval calls.
///
/// Saga 21.5 step 009: `created_at` (Unix seconds) is set on
/// session creation; `last_eval_at` is `None` until the first
/// `/eval` (or `/eval_stream`) handler success, then updated to
/// the wall-clock time of the most recent eval. The session-
/// reattach client (`mlpl-repl --connect --session <id>`) reads
/// both via `GET /v1/sessions/<id>` to render a "you are
/// rejoining a session last touched N seconds ago" banner.
pub struct Session {
    pub token: String,
    pub env: Environment,
    pub created_at: u64,
    pub last_eval_at: Option<u64>,
}

/// Unix seconds since epoch -- saga 21.5 step 009 session
/// timestamps. Falls back to 0 if the system clock is before
/// 1970 (impossible on modern hardware but `SystemTime`'s
/// `duration_since(UNIX_EPOCH)` can return an error in
/// principle).
#[must_use]
pub fn now_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Shared state on the application: maps session ids
/// to sessions. `Arc<RwLock<...>>` so axum handlers
/// can clone cheaply while serializing writes.
pub type SessionMap = Arc<RwLock<HashMap<Uuid, Session>>>;

/// Saga 21.5 step 003: parallel map of cancel-time
/// state. Keyed by the same `Uuid` as `SessionMap`; the
/// entry stores the bearer token (so `/cancel` can
/// authenticate without the sessions lock) and the
/// shared `Interrupt` (so flipping it from the cancel
/// handler trips the in-flight eval).
pub type InterruptMap = Arc<RwLock<HashMap<Uuid, InterruptEntry>>>;

/// One row in the `InterruptMap`. `token` is a copy of
/// the session's bearer token, frozen at creation; the
/// session record remains the single source of truth
/// for auth in non-cancel paths.
#[derive(Clone)]
pub struct InterruptEntry {
    pub token: String,
    pub interrupt: Interrupt,
}

/// Construct a fresh empty session map.
#[must_use]
pub fn new_map() -> SessionMap {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Saga 21.5 step 003: construct a fresh empty
/// interrupt map. Sibling of `new_map`.
#[must_use]
pub fn new_interrupt_map() -> InterruptMap {
    Arc::new(RwLock::new(HashMap::new()))
}

/// A session Environment with the connect-mode test-event
/// buffer enabled (emit_test_event buffers JSON lines; the
/// eval handler drains them into the response).
fn fresh_session_env() -> Environment {
    let mut env = Environment::new();
    env.test_event_lines = Some(Vec::new());
    env
}

/// Create a new session, insert it into both maps,
/// and return the id + token. Caller is responsible
/// for surfacing both back to the client.
pub async fn create_session(sessions: &SessionMap, interrupts: &InterruptMap) -> (Uuid, String) {
    let id = Uuid::new_v4();
    let token = generate_token();
    let session = Session {
        token: token.clone(),
        env: fresh_session_env(),
        created_at: now_unix_seconds(),
        last_eval_at: None,
    };
    sessions.write().await.insert(id, session);
    interrupts.write().await.insert(
        id,
        InterruptEntry {
            token: token.clone(),
            interrupt: Interrupt::new(),
        },
    );
    (id, token)
}

/// 32 alphanumeric characters from the thread-local
/// CSPRNG. ~190 bits of entropy; enough for the
/// loopback / LAN threat model. A future saga can
/// swap in `OsRng` + a longer alphabet if the
/// threat model changes.
fn generate_token() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(TOKEN_LEN)
        .map(char::from)
        .collect()
}

/// Shared bearer gate for session-scoped handlers: `Ok(())` when
/// `auth_mode` is `Disabled` or the request carries a bearer that
/// matches `token`. Every session handler (eval, eval_stream,
/// cancel, inspect, session-meta) had its own copy of this block;
/// centralizing it here retires those duplicates.
pub fn require_bearer(
    auth_mode: AuthMode,
    token: &str,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let unauthorized = || {
        (
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        )
    };
    if auth_mode == AuthMode::Required {
        let provided = extract_bearer(headers).ok_or_else(unauthorized)?;
        if !check_token(provided, token) {
            return Err(unauthorized());
        }
    }
    Ok(())
}
