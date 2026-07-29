use mlpl_eval::env_api::*;
mod args;
mod babel_session;
mod run;
mod script_mode;
mod svg_out;
mod version;

use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;
use mlpl_trace::Trace;
use svg_out::SvgOut;

fn main() {
    let config = args::parse(std::env::args().collect());
    run::run(config);
}

pub(crate) fn run_interactive(env: &mut Environment, svg_out: &mut SvgOut) {
    println!("{}", version::banner());
    println!("Type :help for commands, exit or Ctrl-D to quit.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut tracing = false;
    let mut last_trace: Option<Trace> = None;

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

        if trimmed.starts_with(':') && handle_command(trimmed, &mut tracing, &last_trace, env) {
            continue;
        }

        eval_line(trimmed, env, tracing, &mut last_trace, svg_out);
    }
}

fn handle_command(
    input: &str,
    tracing: &mut bool,
    last_trace: &Option<Trace>,
    env: &mut Environment,
) -> bool {
    if handle_trace_command(input, tracing, last_trace) || handle_ask_connect(input, env) {
        return true;
    }
    match input {
        ":help" => print_help(),
        ":version" => println!("{}", version::banner()),
        ":clear" => {
            *env = Environment::new();
            println!("Environment cleared.");
        }
        _ => match mlpl_eval::inspect(env, input) {
            Some(out) => println!("{out}"),
            None => eprintln!("Unknown command: {input}. Type :help for available commands."),
        },
    }
    true
}

/// The LLM-facing commands (`:ask ...`, `:connect ...`), split from
/// `handle_command` to keep it inside the LOC budget.
fn handle_ask_connect(input: &str, env: &mut Environment) -> bool {
    if input == ":ask" || input.starts_with(":ask ") {
        mlpl_repl_connect::ask::dispatch(input.strip_prefix(":ask").unwrap_or("").trim(), env);
    } else if input == ":connect" || input.starts_with(":connect ") {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
        let arg = input.strip_prefix(":connect").unwrap_or("").trim();
        println!("{}", mlpl_repl_connect::ask_model::connect_cmd(arg, &host));
    } else {
        return false;
    }
    true
}

/// The `:trace ...` family (toggle, summary, json print, json-to-file),
/// split from `handle_command` so each stays inside the LOC budget.
/// Returns false for non-trace commands.
fn handle_trace_command(input: &str, tracing: &mut bool, last_trace: &Option<Trace>) -> bool {
    let no_trace = || eprintln!("No trace available. Use :trace on first.");
    match input {
        ":trace on" | ":trace off" => {
            *tracing = input == ":trace on";
            println!("Tracing {}.", if *tracing { "enabled" } else { "disabled" });
        }
        ":trace json" => match last_trace {
            Some(t) => println!("{}", t.to_json()),
            None => no_trace(),
        },
        ":trace" => match last_trace {
            Some(t) => print_trace_summary(t),
            None => no_trace(),
        },
        _ if input.starts_with(":trace json ") => {
            write_trace_json(
                input.strip_prefix(":trace json ").unwrap().trim(),
                last_trace,
            );
        }
        _ => return false,
    }
    true
}

/// `:trace json <path>`: dump the last trace to a file.
fn write_trace_json(path: &str, last_trace: &Option<Trace>) {
    match last_trace {
        Some(t) => match std::fs::write(path, t.to_json()) {
            Ok(()) => println!("Trace written to {path}"),
            Err(e) => eprintln!("error writing file: {e}"),
        },
        None => eprintln!("No trace available. Use :trace on first."),
    }
}

fn eval_line(
    input: &str,
    env: &mut Environment,
    tracing: bool,
    last_trace: &mut Option<Trace>,
    svg_out: &mut SvgOut,
) {
    let report_err = |e: &dyn std::fmt::Display| {
        eprintln!("  {input}");
        eprintln!("  error: {e}");
    };
    let tokens = match mlpl_parser::lex(input) {
        Ok(t) => t,
        Err(e) => return report_err(&e),
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return,
        Ok(s) => s,
        Err(e) => return report_err(&e),
    };
    if tracing {
        let mut trace = Trace::new(input.into());
        match mlpl_eval::eval_program_traced(&stmts, env, &mut trace) {
            Ok(arr) => {
                println!("{arr}");
                *last_trace = Some(trace);
            }
            Err(e) => report_err(&e),
        }
    } else {
        env.set_pending_source(Some(input.to_string()));
        let evaluated = mlpl_eval::eval_program_value(&stmts, env);
        env.set_pending_source(None);
        match evaluated {
            Ok(value) => {
                let formatted = format!("{value}");
                let display =
                    mlpl_cli::viz_cache::transform_value(&formatted, svg_out.dir.as_deref());
                println!("{display}");
            }
            Err(e) => report_err(&e),
        }
    }
}

/// Static body of `:help`. The version banner is printed
/// separately so it stays current at runtime; everything
/// else is a fixed string that prints in one go.
const HELP_BODY: &str = "
Syntax:
  42              scalar literal
  [1, 2, 3]       array literal
  x = expr        assignment
  a + b           arithmetic (+, -, *, /)
  func(args)      function call
  repeat N { }    loop N times

Built-in functions:
  iota(n)              integers 0..n
  shape(a)             dimension vector
  rank(a)              number of dimensions
  reshape(a, dims)     reshape array
  transpose(a)         reverse axis order
  reduce_add(a)        sum all elements
  reduce_add(a, axis)  sum along axis
  reduce_mul(a)        product of all elements
  reduce_mul(a, axis)  product along axis
  dot(a, b)            vector dot product
  matmul(a, b)         matrix multiplication
  exp(a) log(a)        element-wise exp / log
  sqrt(a) abs(a)       element-wise sqrt / abs
  sigmoid(a) tanh_fn(a) activations
  pow(a, b)            element-wise power
  gt(a, b) lt(a, b)    element-wise comparison
  eq(a, b)             element-wise equality
  mean(a)              mean of all elements
  zeros(s) ones(s)     array constructors
  fill(s, v)           fill array with value

Commands:
  :help                show this help
  :help <topic>        focused help: vars, models, fns, builtins,
                       describe, wsid
  :version             show the build banner (version + host + commit + timestamp)
  :vars                list bound variables with shape
  :models              list bound models with layer structure
  :fns                 list user-defined functions (none yet)
  :builtins            list built-in functions by category
  :describe <name>     describe a variable, model, string, or built-in
  :ask <question>      ask a local Ollama server about the session (arg is sent verbatim to the model)
  :connect list        list installed Ollama models (marks the current pick)
  :connect set <model> select the Ollama model for this session
                       (default model: $OLLAMA_MODEL, else a median-size installed model; OLLAMA_HOST overrides the host). See docs/using-ollama.md
  :wsid                workspace summary (var/param/model counts)
  :clear               reset all variables
  :trace on/off        toggle execution tracing
  :trace               show last trace summary
  :trace json          print last trace as JSON
  :trace json <file>   write trace JSON to file
  exit                 quit

File mode: cargo run -p mlpl-repl -- -f <script.mlpl>
Version:   mlpl-repl -V    or    mlpl-repl --version
";

fn print_help() {
    println!("{}", version::banner());
    print!("{HELP_BODY}");
}

pub(crate) fn print_trace_summary(trace: &Trace) {
    println!("Trace for: {}", trace.source());
    println!("Events: {}", trace.events().len());
    for event in trace.events() {
        println!(
            "  [{:>3}] {:<12} span={}..{}",
            event.seq, event.op, event.span.start, event.span.end
        );
    }
}
