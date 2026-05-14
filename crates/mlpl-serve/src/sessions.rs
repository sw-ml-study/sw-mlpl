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

const TOKEN_LEN: usize = 32;

/// One client session. Holds the bearer token and a
/// long-lived `Environment` that accumulates state
/// across eval calls.
pub struct Session {
    pub token: String,
    pub env: Environment,
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

/// Create a new session, insert it into both maps,
/// and return the id + token. Caller is responsible
/// for surfacing both back to the client.
pub async fn create_session(sessions: &SessionMap, interrupts: &InterruptMap) -> (Uuid, String) {
    let id = Uuid::new_v4();
    let token = generate_token();
    let session = Session {
        token: token.clone(),
        env: Environment::new(),
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
