mod ask;
mod connect;
mod connect_repl;
mod svg_out;
mod version;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use mlpl_eval::Environment;
use mlpl_trace::Trace;
use svg_out::SvgOut;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-V" || a == "--version") {
        version::print();
        return;
    }
    if connect_repl::try_dispatch_args(&args) {
        return;
    }
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .cloned()
    };
    let mut env = Environment::new();
    let trace_flag = args.iter().any(|a| a == "--trace");
    let mut svg_out = SvgOut::new(flag("--svg-out").map(PathBuf::from));
    if let Some(dir) = flag("--data-dir") {
        env.set_data_dir(PathBuf::from(dir));
    }
    if let Some(dir) = flag("--exp-dir") {
        env.set_exp_dir(PathBuf::from(dir));
    }
    if let Some(path) = flag("-f").or_else(|| flag("--file")) {
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        });
        run_script(&content, &mut env, trace_flag, &mut svg_out);
    } else {
        run_interactive(&mut env, &mut svg_out);
    }
}

fn run_interactive(env: &mut Environment, svg_out: &mut SvgOut) {
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

        if handle_command(trimmed, &mut tracing, &last_trace, env) {
            continue;
        }

        eval_line(trimmed, env, tracing, &mut last_trace, svg_out);
    }
}

fn run_script(content: &str, env: &mut Environment, tracing: bool, svg_out: &mut SvgOut) {
    // Strip comment-only lines but preserve structure for multi-line constructs
    let cleaned: Vec<&str> = content
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('#') { "" } else { line }
        })
        .collect();
    let source = cleaned.join("\n");
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return;
    }
    for line in content.lines() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('#') {
            println!("> {t}");
        }
    }
    let mut last_trace: Option<Trace> = None;
    eval_line(trimmed, env, tracing, &mut last_trace, svg_out);
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
        ":version" => println!("{}", version::banner()),
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
        _ if input == ":ask" || input.starts_with(":ask ") => {
            ask::dispatch(input.strip_prefix(":ask").unwrap_or("").trim(), env);
        }
        _ => match mlpl_eval::inspect(env, input) {
            Some(out) => println!("{out}"),
            None => eprintln!("Unknown command: {input}. Type :help for available commands."),
        },
    }
    true
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
        match mlpl_eval::eval_program_value(&stmts, env) {
            Ok(value) => {
                let formatted = format!("{value}");
                if formatted.trim_start().starts_with("<svg") {
                    svg_out.handle(&formatted);
                } else {
                    println!("{formatted}");
                }
            }
            Err(e) => report_err(&e),
        }
    }
}

fn print_help() {
    println!("{}", version::banner());
    println!();
    println!("Syntax:");
    println!("  42              scalar literal");
    println!("  [1, 2, 3]       array literal");
    println!("  x = expr        assignment");
    println!("  a + b           arithmetic (+, -, *, /)");
    println!("  func(args)      function call");
    println!("  repeat N {{ }}    loop N times");
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
    println!("  dot(a, b)            vector dot product");
    println!("  matmul(a, b)         matrix multiplication");
    println!("  exp(a) log(a)        element-wise exp / log");
    println!("  sqrt(a) abs(a)       element-wise sqrt / abs");
    println!("  sigmoid(a) tanh_fn(a) activations");
    println!("  pow(a, b)            element-wise power");
    println!("  gt(a, b) lt(a, b)    element-wise comparison");
    println!("  eq(a, b)             element-wise equality");
    println!("  mean(a)              mean of all elements");
    println!("  zeros(s) ones(s)     array constructors");
    println!("  fill(s, v)           fill array with value");
    println!();
    println!("Commands:");
    println!("  :help                show this help");
    println!("  :help <topic>        focused help: vars, models, fns, builtins,");
    println!("                       describe, wsid");
    println!("  :version             show the build banner (version + host + commit + timestamp)");
    println!("  :vars                list bound variables with shape");
    println!("  :models              list bound models with layer structure");
    println!("  :fns                 list user-defined functions (none yet)");
    println!("  :builtins            list built-in functions by category");
    println!("  :describe <name>     describe a variable, model, string, or built-in");
    println!(
        "  :ask <question>      ask a local Ollama server about the session -- set OLLAMA_HOST / OLLAMA_MODEL to override; Saga 19 preview, see docs/using-ollama.md"
    );
    println!("  :wsid                workspace summary (var/param/model counts)");
    println!("  :clear               reset all variables");
    println!("  :trace on/off        toggle execution tracing");
    println!("  :trace               show last trace summary");
    println!("  :trace json          print last trace as JSON");
    println!("  :trace json <file>   write trace JSON to file");
    println!("  exit                 quit");
    println!();
    println!("File mode: cargo run -p mlpl-repl -- -f <script.mlpl>");
    println!("Version:   mlpl-repl -V    or    mlpl-repl --version");
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
