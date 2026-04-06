use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;
use mlpl_trace::Trace;

fn main() {
    println!("MLPL v0.1 -- Array Programming Language");
    println!("Type :help for commands, exit or Ctrl-D to quit.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut env = Environment::new();
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

        if handle_command(trimmed, &mut tracing, &last_trace, &mut env) {
            continue;
        }

        eval_line(trimmed, &mut env, tracing, &mut last_trace);
    }
}

fn handle_command(
    input: &str,
    tracing: &mut bool,
    last_trace: &Option<Trace>,
    env: &mut Environment,
) -> bool {
    if !input.starts_with(':') && input != "exit" {
        return false;
    }
    match input {
        ":help" => print_help(),
        ":clear" => {
            *env = Environment::new();
            println!("Environment cleared.");
        }
        ":trace on" => {
            *tracing = true;
            println!("Tracing enabled.");
        }
        ":trace off" => {
            *tracing = false;
            println!("Tracing disabled.");
        }
        ":trace json" => with_trace(last_trace, |t| println!("{}", t.to_json())),
        ":trace" => with_trace(last_trace, print_trace_summary),
        _ if input.starts_with(":trace json ") => {
            let path = input.strip_prefix(":trace json ").unwrap().trim();
            with_trace(last_trace, |t| match std::fs::write(path, t.to_json()) {
                Ok(()) => println!("Trace written to {path}"),
                Err(e) => eprintln!("error writing file: {e}"),
            });
        }
        _ => eprintln!("Unknown command: {input}. Type :help for available commands."),
    }
    true
}

fn with_trace(trace: &Option<Trace>, f: impl FnOnce(&Trace)) {
    match trace {
        Some(t) => f(t),
        None => eprintln!("No trace available. Use :trace on first."),
    }
}

fn eval_line(input: &str, env: &mut Environment, tracing: bool, last_trace: &mut Option<Trace>) {
    let tokens = match mlpl_parser::lex(input) {
        Ok(t) => t,
        Err(e) => {
            print_error(input, &format!("{e}"));
            return;
        }
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return,
        Ok(s) => s,
        Err(e) => {
            print_error(input, &format!("{e}"));
            return;
        }
    };
    if tracing {
        let mut trace = Trace::new(input.into());
        match mlpl_eval::eval_program_traced(&stmts, env, &mut trace) {
            Ok(arr) => {
                println!("{arr}");
                *last_trace = Some(trace);
            }
            Err(e) => print_error(input, &format!("{e}")),
        }
    } else {
        match mlpl_eval::eval_program(&stmts, env) {
            Ok(arr) => println!("{arr}"),
            Err(e) => print_error(input, &format!("{e}")),
        }
    }
}

fn print_error(source: &str, msg: &str) {
    eprintln!("  {source}");
    eprintln!("  error: {msg}");
}

fn print_help() {
    println!("MLPL v0.1 -- Array Programming Language");
    println!();
    println!("Syntax:");
    println!("  42              scalar literal");
    println!("  [1, 2, 3]       array literal");
    println!("  x = expr        assignment");
    println!("  a + b           arithmetic (+, -, *, /)");
    println!("  func(args)      function call");
    println!();
    println!("Built-in functions:");
    println!("  iota(n)              integers 0..n");
    println!("  shape(a)             dimension vector");
    println!("  rank(a)              number of dimensions");
    println!("  reshape(a, dims)     reshape array");
    println!("  transpose(a)         reverse axis order");
    println!("  reduce_add(a)        sum all elements");
    println!("  reduce_add(a, axis)  sum along axis");
    println!("  reduce_mul(a)        product of all elements");
    println!("  reduce_mul(a, axis)  product along axis");
    println!();
    println!("Commands:");
    println!("  :help                show this help");
    println!("  :clear               reset all variables");
    println!("  :trace on/off        toggle execution tracing");
    println!("  :trace               show last trace summary");
    println!("  :trace json          print last trace as JSON");
    println!("  :trace json <file>   write trace JSON to file");
    println!("  exit                 quit");
}

fn print_trace_summary(trace: &Trace) {
    println!("Trace for: {}", trace.source());
    println!("Events: {}", trace.events().len());
    for event in trace.events() {
        println!(
            "  [{:>3}] {:<12} span={}..{}",
            event.seq, event.op, event.span.start, event.span.end
        );
    }
}
