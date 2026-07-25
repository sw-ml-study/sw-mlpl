//! Saga 21 step 002: connect-mode REPL loop +
//! slash-command dispatch + `--connect` argv parser.
//! Pure HTTP transport lives in `connect.rs`; the
//! streaming wire path + connect-mode argv parsing live
//! in `connect_stream.rs` (Saga 21.5 step 002).

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::connect::{ClientError, build_client, eval_remote, inspect_remote};
use crate::connect_reattach::{Reattach, print_welcome_banner, resolve_session};
use crate::connect_stream::{
    CANCEL_DOUBLE_WINDOW, ConnectArgsError, ConnectMode, eval_remote_stream, parse_connect_args,
    post_cancel, render_metric, should_double_cancel,
};

const CONNECT_HELP: &str = "connect-mode commands:\n  \
     :vars         -- list workspace variables (remote)\n  \
     :models       -- list models (remote)\n  \
     :tokenizers   -- list tokenizers (remote)\n  \
     :experiments  -- list experiment names (remote)\n  \
     :wsid         -- workspace counts (remote)\n  \
     :ask <q>      -- ask local Ollama (no remote workspace framing)\n  \
     :help         -- this message\n  \
     exit, Ctrl-D  -- disconnect";

/// Inspect argv for `--connect <url>` (+ optional
/// `--stream` / `MLPL_REPL_STREAM=1`). Returns `true` if
/// connect mode handled the session so the caller
/// (`main`) can exit; returns `false` for local mode.
/// Pure parsing lives in `connect_stream::parse_connect_args`
/// so the integration tests can assert on
/// `ConnectArgsError` without subprocess machinery.
pub fn try_dispatch_args(args: &[String]) -> bool {
    let stream_env = std::env::var("MLPL_REPL_STREAM").ok();
    match parse_connect_args(args, stream_env.as_deref()) {
        Ok(ConnectMode::Local) => false,
        Ok(ConnectMode::Remote {
            url,
            stream,
            reattach,
        }) => {
            read_loop(&url, stream, reattach);
            true
        }
        Err(ConnectArgsError::StreamWithoutConnect) => {
            eprintln!(
                "error: --stream requires --connect <url>\n  \
                 --stream routes eval through `/v1/sessions/<id>/eval_stream`, \
                 which only exists on a remote `mlpl-serve`."
            );
            std::process::exit(2);
        }
        Err(ConnectArgsError::LocalFlagWithConnect(bad)) => {
            eprintln!(
                "error: --connect cannot be combined with {bad}\n  \
                 --connect delegates evaluation to a remote server; \
                 -f, --data-dir, and --exp-dir are local-mode only."
            );
            std::process::exit(2);
        }
        Err(ConnectArgsError::ReattachIncomplete) => {
            eprintln!(
                "error: --session and --token must be passed together\n  \
                 reattach to an existing session requires BOTH the session id \
                 and its bearer token."
            );
            std::process::exit(2);
        }
    }
}

/// Interactive read-eval-print loop in connect
/// mode. Creates a session (or rebinds to one via `reattach`
/// / `MLPL_REPL_SESSION_FILE`), then for each line either
/// dispatches a slash command (locally OR against `/inspect`)
/// or POSTs to `/eval` / `/eval_stream`.
pub fn read_loop(base_url: &str, stream: bool, reattach: Option<Reattach>) {
    let client = build_client();
    let (session_id, token, reattached) = resolve_session(&client, base_url, reattach)
        .unwrap_or_else(|e| {
            eprintln!(
                "error: failed to resolve session: {e}\n  is mlpl-serve running at {base_url}?"
            );
            std::process::exit(1);
        });
    print_welcome_banner(base_url, &session_id, stream, reattached);
    install_sigint_cancel(
        client.clone(),
        base_url.to_string(),
        session_id.clone(),
        token.clone(),
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("mlpl> ");
        stdout.flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            println!();
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" {
            break;
        }
        if let Some(out) = dispatch_slash(&client, trimmed, base_url, &session_id, &token) {
            if !out.is_empty() {
                println!("{out}");
            }
            continue;
        }
        eval_and_print(&client, base_url, &session_id, &token, trimmed, stream);
    }
}

/// Route one evaluation through `/eval` or
/// `/eval_stream` depending on `stream`. Streaming-mode
/// metric frames render via `render_metric` into stdout;
/// a trailing newline flushes the in-place display
/// before the result lands.
fn eval_and_print(
    client: &reqwest::blocking::Client,
    base_url: &str,
    session_id: &str,
    token: &str,
    program: &str,
    stream: bool,
) {
    let result = if stream {
        let mut stdout = io::stdout();
        let mut any_metric = false;
        let mut on_metric = |m: &crate::connect_stream::SseMetric| {
            any_metric = true;
            let _ = render_metric(&mut stdout, m);
        };
        let r = eval_remote_stream(client, base_url, session_id, token, program, &mut on_metric);
        if any_metric {
            println!();
        }
        r
    } else {
        eval_remote(client, base_url, session_id, token, program)
    };
    match result {
        Ok(r) => {
            // Saga 21.5 step 004: surface the server-minted
            // viz_url + viz_local_path (when set) before the
            // formatted value; the value itself still passes
            // through the local viz_cache so a co-located SVG
            // also lands at `viz: <local-path>`.
            if let Some(url) = &r.viz_url {
                println!("viz: {url}");
            }
            if let Some(path) = &r.viz_local_path {
                println!("viz: {path}");
            }
            println!("{}", mlpl_cli::viz_cache::transform_value(&r.value, None));
        }
        Err(ClientError::Cancelled {
            step,
            partial_losses,
        }) => crate::render::render_cancellation(step, &partial_losses),
        Err(e) => eprintln!("  {program}\n  error: {e}"),
    }
}

/// Saga 21.5 step 003: render a `Cancelled` terminal back to the
/// user. Prints the step the cancel landed on plus a short
/// preview of the partial loss curve so the user can see how far
/// the train got before they tapped Ctrl-Ctrl-C. The full curve
/// is still available via `:vars` (`last_losses`).
/// Saga 21.5 step 003: install the SIGINT handler that turns a
/// double-Ctrl-C inside the cancel window into a `/cancel` POST.
/// `ctrlc::set_handler` is process-global (and can only be
/// installed once); a second install attempt is treated as a
/// no-op so re-invoking the REPL inside the same process during
/// tests doesn't panic.
fn install_sigint_cancel(
    client: reqwest::blocking::Client,
    base_url: String,
    session_id: String,
    token: String,
) {
    let last: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
    // A handler may already be installed (e.g. a prior connect session in the same process);
    // leave it in place if so -- the double-press path still works.
    let _ = ctrlc::set_handler(move || {
        let mut last_guard = last.lock().expect("sigint state lock");
        let now = Instant::now();
        if should_double_cancel(*last_guard, now, CANCEL_DOUBLE_WINDOW) {
            match post_cancel(&client, &base_url, &session_id, &token) {
                Ok(()) => eprintln!("\ncancel requested."),
                Err(e) => eprintln!("\ncancel failed: {e}"),
            }
            *last_guard = None;
        } else {
            eprintln!("\n(press Ctrl-C again within 2s to cancel; or type 'exit' to quit)");
            *last_guard = Some(now);
        }
    });
}

/// Returns `Some(rendered_output)` if the input is a
/// slash command we handle in connect mode. Empty
/// string means "handled, nothing to print" (e.g.,
/// `:ask` prints its own output). `None` means the
/// caller should fall through to remote eval -- but
/// in connect mode all `:`-prefixed lines are
/// considered slash commands; non-supported ones
/// return a "(not supported)" message rather than
/// being POSTed as MLPL source.
fn dispatch_slash(
    client: &reqwest::blocking::Client,
    input: &str,
    base_url: &str,
    session_id: &str,
    token: &str,
) -> Option<String> {
    if !input.starts_with(':') {
        return None;
    }
    match input {
        ":help" => Some(CONNECT_HELP.into()),
        ":vars" | ":models" | ":experiments" | ":tokenizers" | ":wsid" => {
            match inspect_remote(client, base_url, session_id, token) {
                Ok(snap) => Some(crate::render::format_inspect(input, &snap)),
                Err(e) => Some(format!("error: {e}")),
            }
        }
        _ if input == ":ask" || input.starts_with(":ask ") => {
            ask_cmd(input.strip_prefix(":ask").unwrap_or("").trim());
            Some(String::new())
        }
        _ if input == ":connect" || input.starts_with(":connect ") => {
            connect_cmd(input.strip_prefix(":connect").unwrap_or("").trim());
            Some(String::new())
        }
        _ => Some(format!(
            "{input}: not supported in --connect mode (try :vars, :models, :experiments, :tokenizers, :wsid, :ask, :connect, :help)"
        )),
    }
}

/// Connect-mode `:ask <question>` -- the argument is sent verbatim to
/// the model. Model resolution lives in [`crate::ask_model`]; list/select
/// with `:connect list` / `:connect set <model>`. Server workspace
/// context isn't threaded into the prompt yet; `OLLAMA_HOST` selects the
/// Ollama endpoint.
fn ask_cmd(arg: &str) {
    if arg.is_empty() {
        eprintln!("usage: :ask <question>   (see :connect list / :connect set <model>)");
        return;
    }
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = crate::ask_model::resolve(&host);
    match mlpl_runtime::call_ollama(&host, arg, &model) {
        Ok(answer) => println!("{}", answer.trim_end()),
        Err(e) => eprintln!("error: {e}"),
    }
}

/// Connect-mode `:connect list` / `:connect set <model>` -- Ollama model
/// management against `$OLLAMA_HOST`.
fn connect_cmd(arg: &str) {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    println!("{}", crate::ask_model::connect_cmd(arg, &host));
}
