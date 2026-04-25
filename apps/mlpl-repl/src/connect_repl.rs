//! Saga 21 step 002: connect-mode REPL loop +
//! slash-command dispatch + `--connect` argv parser.
//! Pure HTTP transport lives in `connect.rs`.

use std::io::{self, BufRead, Write};

use crate::connect::{InspectResponse, build_client, create_session, eval_remote, inspect_remote};

const CONNECT_HELP: &str = "connect-mode commands:\n  \
     :vars         -- list workspace variables (remote)\n  \
     :models       -- list models (remote)\n  \
     :tokenizers   -- list tokenizers (remote)\n  \
     :experiments  -- list experiment names (remote)\n  \
     :wsid         -- workspace counts (remote)\n  \
     :ask <q>      -- ask local Ollama (no remote workspace framing)\n  \
     :help         -- this message\n  \
     exit, Ctrl-D  -- disconnect";

/// Inspect argv for `--connect <url>`. If present,
/// validate that no local-mode-only flags are also
/// set (`-f`, `--file`, `--data-dir`, `--exp-dir`),
/// run the connect-mode loop, and return `true` to
/// signal the caller (`main`) to exit. Returns
/// `false` if `--connect` was not present so the
/// caller continues into local mode.
pub fn try_dispatch_args(args: &[String]) -> bool {
    let url = match args
        .iter()
        .position(|a| a == "--connect")
        .and_then(|p| args.get(p + 1))
    {
        Some(u) => u.clone(),
        None => return false,
    };
    let conflicts = ["-f", "--file", "--data-dir", "--exp-dir"];
    if let Some(bad) = args.iter().find(|a| conflicts.contains(&a.as_str())) {
        eprintln!(
            "error: --connect cannot be combined with {bad}\n  \
             --connect delegates evaluation to a remote server; \
             -f, --data-dir, and --exp-dir are local-mode only."
        );
        std::process::exit(2);
    }
    read_loop(&url);
    true
}

/// Interactive read-eval-print loop in connect
/// mode. Creates a session, then for each line
/// either dispatches a slash command (locally OR
/// against `/inspect`) or POSTs to `/eval`.
pub fn read_loop(base_url: &str) {
    let client = build_client();
    let (session_id, token) = match create_session(&client, base_url) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to create session: {e}");
            eprintln!("  is mlpl-serve running at {base_url}?");
            std::process::exit(1);
        }
    };
    println!("Connected to {base_url} (session {session_id})");
    println!("Type :help for commands, exit or Ctrl-D to quit.");
    println!();

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
        match eval_remote(&client, base_url, &session_id, &token, trimmed) {
            Ok(r) => println!("{}", mlpl_cli::viz_cache::transform_value(&r.value, None)),
            Err(e) => eprintln!("  {trimmed}\n  error: {e}"),
        }
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
