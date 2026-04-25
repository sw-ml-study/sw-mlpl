//! In-memory session storage. Saga 21 step 001.
//!
//! `Session` owns a fresh `mlpl_eval::Environment`;
//! the eval handler holds a write lock on the map
//! for the duration of the eval call (because
//! `Environment` is mutated by side-effecting
//! builtins). A future saga can swap the storage
//! to disk-backed for persistence.

use std::collections::HashMap;
use std::sync::Arc;

use mlpl_eval::Environment;
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

/// Construct a fresh empty session map.
#[must_use]
pub fn new_map() -> SessionMap {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Create a new session, insert it into the map,
/// return the id + token. Caller is responsible for
/// surfacing both back to the client.
pub async fn create_session(map: &SessionMap) -> (Uuid, String) {
    let id = Uuid::new_v4();
    let token = generate_token();
    let session = Session {
        token: token.clone(),
        env: Environment::new(),
    };
    map.write().await.insert(id, session);
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
