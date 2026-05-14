//! Saga 21 step 002: connect-mode REPL loop +
//! slash-command dispatch + `--connect` argv parser.
//! Pure HTTP transport lives in `connect.rs`; the
//! streaming wire path + connect-mode argv parsing live
//! in `connect_stream.rs` (Saga 21.5 step 002).

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::connect::{ClientError, InspectResponse, build_client, eval_remote, inspect_remote};
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
        }) => render_cancellation(step, &partial_losses),
        Err(e) => eprintln!("  {program}\n  error: {e}"),
    }
}

/// Saga 21.5 step 003: render a `Cancelled` terminal back to the
/// user. Prints the step the cancel landed on plus a short
/// preview of the partial loss curve so the user can see how far
/// the train got before they tapped Ctrl-Ctrl-C. The full curve
/// is still available via `:vars` (`last_losses`).
fn render_cancellation(step: usize, partial_losses: &[f64]) {
    eprintln!(
        "  cancelled at step {step} ({} partial loss{} captured; see :vars last_losses)",
        partial_losses.len(),
        if partial_losses.len() == 1 { "" } else { "es" }
    );
    if !partial_losses.is_empty() {
        let preview: Vec<String> = partial_losses
            .iter()
            .take(5)
            .map(|v| format!("{v:.4}"))
            .collect();
        let ellipsis = if partial_losses.len() > 5 {
            ", ..."
        } else {
            ""
        };
        eprintln!("  losses: [{}{ellipsis}]", preview.join(", "));
    }
}

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
    let result = ctrlc::set_handler(move || {
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
    if result.is_err() {
        // A handler is already installed (e.g. a prior connect
        // session in the same process). Leave it in place; the
        // double-press path still works.
    }
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
                Ok(snap) => Some(format_inspect(input, &snap)),
                Err(e) => Some(format!("error: {e}")),
            }
        }
        _ if input == ":ask" || input.starts_with(":ask ") => {
            // Connect-mode `:ask` does NOT thread server
            // workspace context into the prompt -- the
            // server-side framing path would need the inspect
            // snapshot composed into ask.rs's helpers, which
            // is a follow-up. The local OLLAMA_HOST /
            // OLLAMA_MODEL env vars still apply.
            let question = input.strip_prefix(":ask").unwrap_or("").trim();
            if question.is_empty() {
                eprintln!("usage: :ask <question>");
            } else {
                let host = std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".into());
                let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into());
                match mlpl_runtime::call_ollama(&host, question, &model) {
                    Ok(answer) => println!("{}", answer.trim_end()),
                    Err(e) => eprintln!("error: {e}"),
                }
            }
            Some(String::new())
        }
        _ => Some(format!(
            "{input}: not supported in --connect mode (try :vars, :models, :experiments, :tokenizers, :wsid, :ask, :help)"
        )),
    }
}

fn format_inspect(command: &str, snap: &InspectResponse) -> String {
    let mut out = String::new();
    let render_names = |out: &mut String, label: &str, names: &[String]| {
        if names.is_empty() {
            out.push_str(&format!("(no {label})"));
        } else {
            for n in names {
                out.push_str(&format!("  {n}\n"));
            }
            out.truncate(out.trim_end().len());
        }
    };
    match command {
        ":vars" => {
            if snap.vars.is_empty() {
                out.push_str("(no variables)");
            } else {
                for v in &snap.vars {
                    let tag = if v.is_param { " [param]" } else { "" };
                    let dims: Vec<String> = v.shape.iter().map(|d| d.to_string()).collect();
                    out.push_str(&format!("  {}: [{}]{tag}\n", v.name, dims.join(", ")));
                }
                if snap.more > 0 {
                    out.push_str(&format!("  ... ({} more)\n", snap.more));
                }
                out.truncate(out.trim_end().len());
            }
        }
        ":models" => render_names(&mut out, "models", &snap.models),
        ":tokenizers" => render_names(&mut out, "tokenizers", &snap.tokenizers),
        ":experiments" => render_names(&mut out, "experiments", &snap.experiments),
        ":wsid" => {
            out.push_str(&format!(
                "workspace (remote):\n  variables: {}\n  models:    {}\n  tokenizers: {}\n  experiments: {}",
                snap.vars.len() + snap.more,
                snap.models.len(),
                snap.tokenizers.len(),
                snap.experiments.len()
            ));
        }
        _ => unreachable!("dispatch_slash filters before format_inspect"),
    }
    out
}
