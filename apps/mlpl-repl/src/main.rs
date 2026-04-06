use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;
use mlpl_trace::Trace;

fn main() {
    println!("MLPL v0.1 -- Array Programming Language");
    println!("Built-ins: iota, shape, rank, reshape, transpose, reduce_add, reduce_mul");
    println!("Commands: :trace on/off, :trace, :trace json, exit");
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

        if handle_command(trimmed, &mut tracing, &last_trace) {
            continue;
        }

        eval_line(trimmed, &mut env, tracing, &mut last_trace);
    }
}

fn handle_command(input: &str, tracing: &mut bool, last_trace: &Option<Trace>) -> bool {
    match input {
        ":trace on" => {
            *tracing = true;
            println!("Tracing enabled.");
            true
        }
        ":trace off" => {
            *tracing = false;
            println!("Tracing disabled.");
            true
        }
        ":trace json" => {
            match last_trace {
                Some(t) => println!("{}", t.to_json()),
                None => eprintln!("No trace available. Use :trace on first."),
            }
            true
        }
        ":trace" => {
            match last_trace {
                Some(t) => print_trace_summary(t),
                None => eprintln!("No trace available. Use :trace on first."),
            }
            true
        }
        _ if input.starts_with(":trace json ") => {
            let path = input.strip_prefix(":trace json ").unwrap().trim();
            match last_trace {
                Some(t) => match std::fs::write(path, t.to_json()) {
                    Ok(()) => println!("Trace written to {path}"),
                    Err(e) => eprintln!("error writing file: {e}"),
                },
                None => eprintln!("No trace available. Use :trace on first."),
            }
            true
        }
        _ if input.starts_with(':') => {
            eprintln!("Unknown command: {input}");
            true
        }
        _ => false,
    }
}

fn eval_line(input: &str, env: &mut Environment, tracing: bool, last_trace: &mut Option<Trace>) {
    let tokens = match mlpl_parser::lex(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return;
        }
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return,
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
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
            Err(e) => eprintln!("error: {e}"),
        }
    } else {
        match mlpl_eval::eval_program(&stmts, env) {
            Ok(arr) => println!("{arr}"),
            Err(e) => eprintln!("error: {e}"),
        }
    }
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
