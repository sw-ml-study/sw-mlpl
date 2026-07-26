//! SSE eval streaming: `POST /v1/sessions/:id/eval_stream`.

pub use mlpl_serve_core::eval_viz::SseEvent;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use mlpl_eval::{MetricSink, eval_program_value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use uuid::Uuid;

use crate::handlers::{ErrorResponse, EvalRequest, json_err};
use crate::server::AppState;

/// Adapter from the synchronous `MetricSink` trait into an
/// async tokio mpsc channel. The eval runs inside
/// `spawn_blocking`, so `blocking_send` is the right primitive
/// -- it applies channel-bounded backpressure if the consumer
/// stalls without forcing the caller into async. `tx` is
/// `pub(super)` rather than wrapped behind a `new` constructor
/// so callers can build the struct inline and keep lib.rs
/// under the sw-checklist 7-fn-per-module budget.
#[derive(Debug)]
pub struct ChannelMetricSink {
    pub(super) tx: mpsc::Sender<SseEvent>,
}

impl MetricSink for ChannelMetricSink {
    fn emit_frame(&self, name: &str, step: usize, shape: &[usize], values: &[f64]) {
        let _ = self.tx.blocking_send(SseEvent::Frame {
            name: name.to_string(),
            step,
            shape: shape.to_vec(),
            values: values.to_vec(),
        });
    }

    fn emit(&self, name: &str, step: usize, value: f64) {
        let _ = self.tx.blocking_send(SseEvent::Metric {
            name: name.to_string(),
            step,
            value,
        });
    }
}

type SseError = (StatusCode, Json<ErrorResponse>);

/// Spawn the eval as a `spawn_blocking` task, route the result
/// through the channel, and re-insert the session into the
/// shared map on completion. Decoupled from `eval_stream_handler`
/// so each function stays under the sw-checklist 50-line budget.
fn spawn_eval_task(
    id: Uuid,
    mut session: crate::sessions::Session,
    stmts: Vec<mlpl_parser::Expr>,
    sessions_map: crate::sessions::SessionMap,
    viz: crate::viz_storage::SharedVizStore,
    tx: mpsc::Sender<SseEvent>,
) {
    tokio::spawn(async move {
        let _ = tx.send(SseEvent::Ready).await;
        let join = tokio::task::spawn_blocking(move || {
            let value = eval_program_value(&stmts, &mut session.env);
            (session, value)
        })
        .await;
        finish_stream_session(id, join, sessions_map, viz, tx).await;
    });
}

/// After the blocking eval returns: detach the streaming plumbing,
/// send the terminal frame (attaching any viz), and put the session
/// back in the map. Extracted from `spawn_eval_task` for the LOC
/// budget.
type EvalJoin = Result<
    (
        crate::sessions::Session,
        Result<mlpl_eval::Value, mlpl_eval::EvalError>,
    ),
    tokio::task::JoinError,
>;

async fn finish_stream_session(
    id: Uuid,
    join: EvalJoin,
    sessions_map: crate::sessions::SessionMap,
    viz: crate::viz_storage::SharedVizStore,
    tx: mpsc::Sender<SseEvent>,
) {
    let Ok((mut session, value)) = join else {
        let _ = tx
            .send(SseEvent::Error {
                error: "eval task panicked".into(),
            })
            .await;
        return;
    };
    session.env.clear_metric_sink();
    session.env.clear_peer_dispatcher();
    session.env.clear_interrupt();
    let terminal = crate::viz_storage::result_to_sse(&viz, value).await;
    let _ = tx.send(terminal).await;
    sessions_map.write().await.insert(id, session);
}

/// `POST /v1/sessions/:id/eval_stream` -- requires bearer when
/// `auth_mode == Required`. Lex + parse synchronously (so a
/// 400 still surfaces as plain JSON); on success, take the
/// session out of the map, run eval on a `spawn_blocking`
/// task with a `ChannelMetricSink` installed, and return an
/// SSE stream whose frames are: one `ready`, zero-or-more
/// `metric`, one terminal `done` or `error`.
pub async fn eval_stream_handler(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(body): Json<EvalRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, SseError> {
    {
        let sessions = state.sessions.read().await;
        let session = sessions
            .get(&id)
            .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
        mlpl_serve_core::sessions::require_bearer(state.auth_mode, &session.token, &headers)?;
    }
    let stmts = crate::handlers::parse_program(&body.program)?;
    let (tx, rx) = mpsc::channel::<SseEvent>(64);
    let session = take_stream_session(&state, &id, tx.clone()).await?;
    let sessions = state.sessions.clone();
    let viz = state.viz.clone();
    spawn_eval_task(id, session, stmts, sessions, viz, tx);
    let stream = ReceiverStream::new(rx).map(|ev| Ok::<_, Infallible>(ev.to_axum_event()));
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Remove the session from the map for the duration of the streamed
/// eval and wire it for streaming: metric sink into the SSE channel,
/// the cancel interrupt, and the GPU peer dispatcher.
async fn take_stream_session(
    state: &AppState,
    id: &Uuid,
    tx: mpsc::Sender<SseEvent>,
) -> Result<mlpl_serve_core::sessions::Session, SseError> {
    let mut session = state
        .sessions
        .write()
        .await
        .remove(id)
        .ok_or((StatusCode::NOT_FOUND, json_err("unknown session")))?;
    let sink: Arc<dyn MetricSink> = Arc::new(ChannelMetricSink { tx });
    session.env.set_metric_sink(sink);
    crate::handlers::install_session_interrupt(state, id, &mut session).await;
    session
        .env
        .set_peer_dispatcher(Arc::new(crate::server::RemoteMlxDispatcher::new(
            state.peers.clone(),
            state.peer_sessions.clone(),
        )));
    Ok(session)
}
