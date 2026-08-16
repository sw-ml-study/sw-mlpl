//! The 'parked main' launch inversion for native-UI applets. Normally
//! the interpreter owns the calling thread (REPL/script). A native
//! window needs the OS main thread for its event loop, so this runs the
//! MLPL interpreter on a spawned WORKER and hands the calling (main)
//! thread to a UI host -- the browser model: UI loop on main, logic on
//! a worker, connected by the port's channels.
//!
//! The host is anything `FnOnce(cmd_rx, ev_tx)`: it feeds events into
//! the port's event channel and drains the commands the applet
//! submits. In production it is the provider's winit loop; in tests it
//! is a scripted headless double (no winit). Only owned `Value`s cross
//! the channels, so the two threads never race.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::env::Environment;
use mlpl_eval_types::{EvalError, Value};

/// Run an MLPL applet: the interpreter evaluates `source` on a worker
/// thread with a port bound to the name `port`, while `host` runs on
/// the calling thread driving that port. Returns the applet's final
/// value (what `run(port, ...)` folded to).
pub fn run_applet_with_host<H>(source: &str, host: H) -> Result<Value, EvalError>
where
    H: FnOnce(Receiver<Value>, Sender<Value>),
{
    let (cmd_tx, cmd_rx) = mpsc::channel::<Value>();
    let (ev_tx, ev_rx) = mpsc::channel::<Value>();
    let (result_tx, result_rx) = mpsc::channel::<Result<Value, EvalError>>();
    let src = source.to_string();
    let worker = thread::spawn(move || {
        let _ = result_tx.send(run_worker(&src, cmd_tx, ev_rx));
    });
    host(cmd_rx, ev_tx);
    let _ = worker.join();
    result_rx
        .recv()
        .unwrap_or_else(|_| Err(EvalError::Unsupported("applet worker died".into())))
}

/// The worker side: build a UI-capable env, bind the port as `port`,
/// and evaluate the applet source (its `run(port, ...)` blocks until a
/// close event).
fn run_worker(
    src: &str,
    cmd_tx: Sender<Value>,
    ev_rx: Receiver<Value>,
) -> Result<Value, EvalError> {
    let mut env = Environment::new();
    env.ui_host_thread = true;
    let handle = env.register_port(cmd_tx, ev_rx);
    env.ext_handles.insert("port".to_string(), handle);
    let tokens = mlpl_parser::lex(src).map_err(|e| EvalError::Unsupported(e.to_string()))?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| EvalError::Unsupported(e.to_string()))?;
    crate::eval_program_value(&stmts, &mut env)
}

/// Policy guard: opening a native window requires the local main-thread
/// UI-host launch path. On any other path (connect/serve worker evals)
/// this is a clear error rather than a silent hang.
///
/// # Errors
/// Returns `ExtensionError` when `env` is not the UI-host launch path.
pub fn require_ui_host_thread(env: &Environment) -> Result<(), EvalError> {
    if env.ui_host_thread {
        return Ok(());
    }
    Err(EvalError::ExtensionError {
        function: "native window".into(),
        message: "requires the local main-thread applet launch path \
                  (not available over connect/serve)"
            .into(),
    })
}
