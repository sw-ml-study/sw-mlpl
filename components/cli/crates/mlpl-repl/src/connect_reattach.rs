//! Saga 21.5 step 009: `mlpl-repl --connect --session <id>
//! --token <tok>` reattach support.
//!
//! Lifted from `connect_stream.rs` + `connect_repl.rs` so the
//! parent modules stay under their sw-checklist 7-fn budgets
//! after threading the reattach pieces through. Owns:
//!
//!   - `Reattach` struct -- carries `(session_id, token)` from
//!     argv into `read_loop`.
//!   - `fetch_session_meta` -- `GET /v1/sessions/<id>` to
//!     validate cached credentials.
//!   - `read_persisted_session` / `write_persisted_session` --
//!     JSON file round-trip for `MLPL_REPL_SESSION_FILE`.
//!   - `resolve_session` -- the priority-order dispatcher
//!     (`--session`/`--token` > env-file > fresh
//!     `create_session`) used by `read_loop`.
//!   - `persisted_session_path` -- env-var accessor.
//!   - `flag_value` -- shared argv-flag helper.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::connect::{ClientError, create_session};

/// Saga 21.5 step 009: server-side session credentials carried
/// through from the command line (or `MLPL_REPL_SESSION_FILE`).
#[derive(Debug, Clone)]
pub struct Reattach {
    pub session_id: String,
    pub token: String,
}

/// Saga 21.5 step 009: parsed `SessionMetaResponse` from
/// `GET /v1/sessions/<id>`. Mirrors the server shape; the
/// REPL banner uses `session_id` + `created_at` today, the rest
/// is kept on the deserialized shape for future welcome-banner
/// work.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct SessionMeta {
    pub session_id: String,
    pub created_at: u64,
    #[serde(default)]
    pub last_eval_at: Option<u64>,
    #[serde(default)]
    pub vars: Vec<serde_json::Value>,
    #[serde(default)]
    pub models: Vec<String>,
}

/// Locate `--flag <value>` in argv. Returns the value when both
/// the flag and a following positional are present. Shared by
/// `--session` and `--token`.
pub(crate) fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|p| args.get(p + 1))
        .cloned()
}

/// `GET /v1/sessions/<id>` -- validate reattach credentials
/// before printing the welcome banner.
pub fn fetch_session_meta(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
) -> Result<SessionMeta, ClientError> {
    let url = format!(
        "{}/v1/sessions/{session_id}",
        base_url.trim_end_matches('/')
    );
    let resp = client
        .get(url)
        .bearer_auth(token)
        .send()
        .map_err(|e| ClientError::Network(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or(body);
        return Err(ClientError::Server { status, message });
    }
    resp.json()
        .map_err(|e| ClientError::Network(format!("decode: {e}")))
}

/// Read a persisted `{session_id, token}` from the JSON file
/// pointed at by `MLPL_REPL_SESSION_FILE`. Returns `None` when
/// the file is missing, unreadable, or malformed -- caller
/// falls back to `create_session`.
#[must_use]
pub fn read_persisted_session(path: &Path) -> Option<(String, String)> {
    let bytes = std::fs::read(path).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let id = v.get("session_id")?.as_str()?.to_string();
    let token = v.get("token")?.as_str()?.to_string();
    Some((id, token))
}

/// Persist a `{session_id, token}` pair to the JSON file so the
/// next REPL launch can reattach. Caller is expected to set the
/// file's permissions to user-only on systems where that
/// matters; this helper just writes the bytes.
pub fn write_persisted_session(path: &Path, session_id: &str, token: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::json!({
        "session_id": session_id,
        "token": token,
    });
    std::fs::write(path, payload.to_string())
}

/// Resolved path for `MLPL_REPL_SESSION_FILE` env var. `None`
/// when unset.
pub fn persisted_session_path() -> Option<PathBuf> {
    std::env::var("MLPL_REPL_SESSION_FILE")
        .ok()
        .map(PathBuf::from)
}

/// Print the welcome banner the REPL shows after a session is
/// resolved. Lives here so `read_loop` stays under its 50-line
/// LOC budget after threading reattach logic through.
pub fn print_welcome_banner(base_url: &str, session_id: &str, stream: bool, reattached: bool) {
    let mode = if stream { " [streaming]" } else { "" };
    let verb = if reattached {
        "Re-attached to"
    } else {
        "Connected to"
    };
    println!("{verb} {base_url} (session {session_id}){mode}");
    println!("Type :help for commands, exit or Ctrl-D to quit.");
    println!("Press Ctrl-C twice within 2s to cancel an in-flight eval.");
    println!();
}

/// Decide which `(session_id, token)` to use. Priority order:
/// 1. `--session <id> --token <tok>` flags on argv.
/// 2. `MLPL_REPL_SESSION_FILE` env var pointing at a JSON file.
/// 3. POST `/v1/sessions` to mint a fresh session.
///
/// Reattach paths validate credentials via `fetch_session_meta`
/// first; a 401/404 drops the saved pair and falls through to
/// `create_session`. The third return value is `true` when the
/// session was successfully re-attached.
pub fn resolve_session(
    client: &reqwest::blocking::Client,
    base_url: &str,
    cli_reattach: Option<Reattach>,
) -> Result<(String, String, bool), String> {
    let from_file = persisted_session_path()
        .as_deref()
        .and_then(read_persisted_session);
    let candidate = cli_reattach.map(|r| (r.session_id, r.token)).or(from_file);
    if let Some((session_id, token)) = candidate
        && fetch_session_meta(client, base_url, &session_id, &token).is_ok()
    {
        return Ok((session_id, token, true));
    }
    let (session_id, token) = create_session(client, base_url).map_err(|e| e.to_string())?;
    if let Some(path) = persisted_session_path() {
        let _ = write_persisted_session(&path, &session_id, &token);
    }
    Ok((session_id, token, false))
}
