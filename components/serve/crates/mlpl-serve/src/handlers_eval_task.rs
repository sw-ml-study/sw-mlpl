//! The eval task pipeline behind `POST /v1/sessions/{id}/eval`:
//! session checkout, detached spawn, and the blocking eval body.
//! Split from `handlers.rs` (connect-telemetry step 003) to bring
//! that module back inside the sw-checklist budgets.

use axum::Json;
use axum::http::{HeaderMap, StatusCode};
use mlpl_eval::env_api::*;
use mlpl_eval::{Interrupt, eval_program_value};
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::auth::{AuthMode, check_token, extract_bearer};
use crate::handlers::{ErrorResponse, EvalResponse, json_err, value_kind};
use crate::server::AppState;

/// Authenticate, take the session OUT of the map (so the eval does not
/// hold the global write lock), reset + install its interrupt, and wire
/// the peer dispatcher. Returns the owned session + a clone of its
/// interrupt for the `AbortGuard`.
pub(crate) async fn take_session_for_eval(
    state: &AppState,
    id: Uuid,
    headers: &HeaderMap,
) -> Result<(crate::sessions::Session, Interrupt), (StatusCode, Json<ErrorResponse>)> {
    let unauthorized = || {
        (
            StatusCode::UNAUTHORIZED,
            json_err("missing or invalid authorization"),
        )
    };
    let entry = state
        .interrupts
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    if state.auth_mode == AuthMode::Required {
        let provided = extract_bearer(headers).ok_or_else(unauthorized)?;
        if !check_token(provided, &entry.token) {
            return Err(unauthorized());
        }
    }
    entry.interrupt.reset();
    let mut session = state
        .sessions
        .write()
        .await
        .remove(&id)
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    session.env.set_interrupt(entry.interrupt.clone());
    session
        .env
        .set_peer_dispatcher(Arc::new(crate::server::RemoteMlxDispatcher::new(
            state.peers.clone(),
            state.peer_sessions.clone(),
        )));
    Ok((session, entry.interrupt))
}

/// Spawn the eval as a detached task (so it unwinds cleanly even if the
/// handler future is dropped on disconnect) and deliver the built
/// response over a oneshot.
pub(crate) fn spawn_eval(
    state: &AppState,
    id: Uuid,
    session: crate::sessions::Session,
    stmts: Vec<mlpl_parser::Expr>,
    program: String,
) -> oneshot::Receiver<Result<EvalResponse, String>> {
    let (tx, rx) = oneshot::channel();
    let state = state.clone();
    tokio::spawn(async move {
        let resp = run_eval(&state, id, session, stmts, program).await;
        let _ = tx.send(resp);
    });
    rx
}

/// Body of the eval task: evaluate on a blocking thread, clear the
/// env's per-eval hooks, build the response (with viz attach) on
/// success, ALWAYS reinsert the session, and flush persistence.
pub(crate) async fn run_eval(
    state: &AppState,
    id: Uuid,
    mut session: crate::sessions::Session,
    stmts: Vec<mlpl_parser::Expr>,
    program: String,
) -> Result<EvalResponse, String> {
    let join = tokio::task::spawn_blocking(move || {
        let trimmed = program.trim();
        if trimmed.starts_with(':') && !mlpl_eval::is_colon_call_expr(trimmed) {
            let value = match mlpl_eval::inspect(&mut session.env, trimmed) {
                Some(out) => Ok(mlpl_eval::Value::Str(out)),
                None => Err(mlpl_eval::EvalError::Unsupported(
                    mlpl_eval::colon_ref_hint(trimmed)
                        .unwrap_or_else(|| format!("unknown command: {trimmed}")),
                )),
            };
            return (session, value);
        }
        session.env.set_pending_source(Some(program));
        let value = eval_program_value(&stmts, &mut session.env);
        session.env.set_pending_source(None);
        (session, value)
    })
    .await;
    let (mut session, value) = join.map_err(|_| "eval task panicked".to_string())?;
    session.env.clear_peer_dispatcher();
    session.env.clear_interrupt();
    // Surface any user-visible notices (e.g. a silent GPU->CPU fallback)
    // by prepending them to the result -- but not to an SVG/viz payload,
    // whose leading `<svg` the client sniffs for.
    let notices = session.env.take_notices();
    let out = match value {
        Ok(v) => {
            session.last_eval_at = Some(crate::sessions::now_unix_seconds());
            let kind = value_kind(&v);
            let mut formatted = format!("{v}");
            if !notices.is_empty() && !formatted.trim_start().starts_with("<svg") {
                formatted = format!("{}\n{formatted}", notices.join("\n"));
            }
            let attached = crate::viz_storage::attach_viz(&state.viz, &formatted, kind).await;
            Ok(crate::eval_viz::build_eval_response(
                &v, kind, formatted, attached,
            ))
        }
        Err(e) => Err(format!("{e}")),
    };
    state.sessions.write().await.insert(id, session);
    crate::persist::maybe_flush(state).await;
    out
}
