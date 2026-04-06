use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;

fn main() {
    println!("MLPL v0.1 -- Array Programming Language");
    println!("Built-ins: iota, shape, rank, reshape, transpose, reduce_add, reduce_mul");
    println!("Type \"exit\" or Ctrl-D to quit.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut env = Environment::new();

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

        match mlpl_parser::lex(trimmed) {
            Ok(tokens) => match mlpl_parser::parse(&tokens) {
                Ok(stmts) => match mlpl_eval::eval_program(&stmts, &mut env) {
                    Ok(arr) => println!("{arr}"),
                    Err(e) => eprintln!("error: {e}"),
                },
                Err(e) => eprintln!("error: {e}"),
            },
            Err(e) => eprintln!("error: {e}"),
        }
    }
}
