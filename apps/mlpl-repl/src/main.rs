use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;
use mlpl_trace::Trace;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut env = Environment::new();
    let trace_flag = args.iter().any(|a| a == "--trace");

    if let Some(pos) = args.iter().position(|a| a == "-f" || a == "--file") {
        let path = args.get(pos + 1).unwrap_or_else(|| {
            eprintln!("error: -f requires a file path");
            std::process::exit(1);
        });
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        });
        run_script(&content, &mut env, trace_flag);
    } else {
        run_interactive(&mut env);
    }
}

fn run_interactive(env: &mut Environment) {
    println!("MLPL v0.1 -- Array Programming Language");
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

        if handle_command(trimmed, &mut tracing, &last_trace, env) {
            continue;
        }

        eval_line(trimmed, env, tracing, &mut last_trace);
    }
}

fn run_script(content: &str, env: &mut Environment, tracing: bool) {
    let mut last_trace: Option<Trace> = None;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        println!("> {trimmed}");
        eval_line(trimmed, env, tracing, &mut last_trace);
    }
    if let Some(trace) = &last_trace {
        println!();
        print_trace_summary(trace);
    }
}

fn handle_command(
    input: &str,
    tracing: &mut bool,
    last_trace: &Option<Trace>,
    env: &mut Environment,
) -> bool {
    if !input.starts_with(':') {
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
        ":trace json" => match last_trace {
            Some(t) => println!("{}", t.to_json()),
            None => eprintln!("No trace available. Use :trace on first."),
        },
        ":trace" => match last_trace {
            Some(t) => print_trace_summary(t),
            None => eprintln!("No trace available. Use :trace on first."),
        },
        _ if input.starts_with(":trace json ") => {
            let path = input.strip_prefix(":trace json ").unwrap().trim();
            match last_trace {
                Some(t) => match std::fs::write(path, t.to_json()) {
                    Ok(()) => println!("Trace written to {path}"),
                    Err(e) => eprintln!("error writing file: {e}"),
                },
                None => eprintln!("No trace available. Use :trace on first."),
            }
        }
        _ => eprintln!("Unknown command: {input}. Type :help for available commands."),
    }
    true
}

fn eval_line(input: &str, env: &mut Environment, tracing: bool, last_trace: &mut Option<Trace>) {
    let tokens = match mlpl_parser::lex(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  {input}");
            eprintln!("  error: {e}");
            return;
        }
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return,
        Ok(s) => s,
        Err(e) => {
            eprintln!("  {input}");
            eprintln!("  error: {e}");
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
            Err(e) => {
                eprintln!("  {input}");
                eprintln!("  error: {e}");
            }
        }
    } else {
        match mlpl_eval::eval_program(&stmts, env) {
            Ok(arr) => println!("{arr}"),
            Err(e) => {
                eprintln!("  {input}");
                eprintln!("  error: {e}");
            }
        }
    }
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
    println!();
    println!("File mode: cargo run -p mlpl-repl -- -f <script.mlpl>");
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
