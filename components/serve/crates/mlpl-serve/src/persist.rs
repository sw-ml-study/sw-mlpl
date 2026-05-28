//! Saga 21.5 step 010: session persistence across server
//! restart.
//!
//! When `mlpl-serve --persist <path>` is set, every successful
//! `/eval` flushes the slim per-session state (`session_id`,
//! `token`, `created_at`, `last_eval_at`, variable bindings) to
//! the JSON file at `path`. Startup reloads the same file so a
//! fresh server process picks up the prior session map.
//!
//! Slim subset rationale: the canonical reattach scenario binds
//! a few plain numeric arrays (`x = iota(3)`, `last_losses = ...`)
//! and resumes against them. Full Environment serialization --
//! including models, tokenizers, experiments, the autograd tape --
//! is a follow-up that wants stable on-disk schemas across many
//! types. For now we persist what `GET /v1/sessions/<id>` already
//! surfaces and what scalar / array vars Environment::set
//! accepts, so the dev-loopback reattach use case works
//! end-to-end without a multi-saga rewrite.

use std::collections::HashMap;
use std::path::Path;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, Interrupt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sessions::{InterruptEntry, InterruptMap, Session, SessionMap};

/// On-disk row for one session.
#[derive(Serialize, Deserialize, Debug)]
pub struct PersistedSession {
    pub session_id: Uuid,
    pub token: String,
    pub created_at: u64,
    #[serde(default)]
    pub last_eval_at: Option<u64>,
    #[serde(default)]
    pub vars: Vec<PersistedVar>,
    #[serde(default)]
    pub params: Vec<String>,
}

/// One variable from the session's `Environment.vars` map,
/// serialized as the shape + flat data the `DenseArray`
/// constructor wants on the way back.
#[derive(Serialize, Deserialize, Debug)]
pub struct PersistedVar {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Top-level on-disk shape -- a versioned wrapper so future
/// schema changes can bump the version and migrate.
#[derive(Serialize, Deserialize, Debug)]
struct PersistFile {
    persist_version: u32,
    sessions: Vec<PersistedSession>,
}

const PERSIST_VERSION: u32 = 1;

/// Snapshot every session in `sessions` (+ matching entries in
/// `interrupts`) and write to `path` as JSON. Best-effort: I/O
/// errors are surfaced so the caller can log them, but a failed
/// flush should NOT take down the eval handler.
pub async fn save_sessions(
    path: &Path,
    sessions: &SessionMap,
    interrupts: &InterruptMap,
) -> std::io::Result<()> {
    let sessions_guard = sessions.read().await;
    // `interrupts` is parameter-only here -- the token lives on
    // `Session`, but the signature accepts both maps so callers
    // (eval_handler) can pass the state by reference.
    let _ = interrupts;
    let mut rows: Vec<PersistedSession> = Vec::with_capacity(sessions_guard.len());
    for (id, session) in sessions_guard.iter() {
        let vars = session
            .env
            .vars_iter()
            .map(|(name, arr)| PersistedVar {
                name: name.clone(),
                shape: arr.shape().dims().to_vec(),
                data: arr.data().to_vec(),
            })
            .collect();
        let params = session
            .env
            .vars_iter()
            .filter(|(name, _)| session.env.is_param(name))
            .map(|(name, _)| name.clone())
            .collect();
        rows.push(PersistedSession {
            session_id: *id,
            token: session.token.clone(),
            created_at: session.created_at,
            last_eval_at: session.last_eval_at,
            vars,
            params,
        });
    }
    let file = PersistFile {
        persist_version: PERSIST_VERSION,
        sessions: rows,
    };
    let body = serde_json::to_vec(&file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, body)
}

/// Read `path` and populate `sessions` + `interrupts`. Malformed
/// or missing files are NOT errors -- the server starts with an
/// empty session map in either case, matching the legacy
/// in-memory-only behavior.
pub async fn load_sessions(path: &Path, sessions: &SessionMap, interrupts: &InterruptMap) {
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let Ok(file): Result<PersistFile, _> = serde_json::from_slice(&bytes) else {
        return;
    };
    if file.persist_version != PERSIST_VERSION {
        // Unknown schema version: don't crash, just skip.
        // (A future saga can add migration paths here.)
        return;
    }
    let mut sessions_guard = sessions.write().await;
    let mut interrupts_guard = interrupts.write().await;
    for row in file.sessions {
        let mut env = Environment::default();
        let params_set: HashMap<String, ()> = row.params.iter().map(|p| (p.clone(), ())).collect();
        for v in row.vars {
            let arr = match DenseArray::new(Shape::new(v.shape), v.data) {
                Ok(a) => a,
                Err(_) => continue,
            };
            env.set(v.name.clone(), arr);
            if params_set.contains_key(&v.name) {
                env.mark_param(&v.name);
            }
        }
        let session = Session {
            token: row.token.clone(),
            env,
            created_at: row.created_at,
            last_eval_at: row.last_eval_at,
        };
        sessions_guard.insert(row.session_id, session);
        interrupts_guard.insert(
            row.session_id,
            InterruptEntry {
                token: row.token,
                interrupt: Interrupt::new(),
            },
        );
    }
}

/// Convenience wrapper for `eval_handler`: flush + log on
/// failure, no-op when no `--persist` path is configured.
/// Lives here so `handlers.rs::eval_handler` stays under its
/// sw-checklist 50-line budget after threading persistence.
pub async fn maybe_flush(state: &crate::server::AppState) {
    if let Some(path) = state.persist_path.as_deref()
        && let Err(e) = save_sessions(path, &state.sessions, &state.interrupts).await
    {
        eprintln!("mlpl-serve: persist flush failed: {e}");
    }
}

/// Convenience wrapper for `build_app_with_peers_cors`: hydrate
/// the session map from `path` (when set). Synchronous facade
/// over the async `load_sessions`; uses
/// `tokio::task::block_in_place` since `build_app_with_peers_cors`
/// itself is sync.
pub fn maybe_load(path: Option<&Path>, sessions: &SessionMap, interrupts: &InterruptMap) {
    let Some(path) = path else {
        return;
    };
    let s = sessions.clone();
    let i = interrupts.clone();
    let p = path.to_path_buf();
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(load_sessions(&p, &s, &i));
    });
}
