use std::io::{self, BufRead, Write};

fn main() {
    println!("MLPL v0.1 (PoC)");
    println!("Type numbers to create arrays. Type \"exit\" or Ctrl-D to quit.");
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

        match mlpl_parser::lex(trimmed) {
            Ok(tokens) => match mlpl_eval::evaluate(&tokens) {
                Ok(arr) => println!("{arr}"),
                Err(e) => eprintln!("eval error: {e}"),
            },
            Err(e) => eprintln!("lex error: {e}"),
        }
    }
}
