Scripting saga step 006: add stdin reads, exit code handling, and Err propagation.

Three small pieces in one step because each one alone is too small for a saga step:

1. read_stdin() builtin: block until EOF, return Value::Str with the contents. read_stdin_lines() returns Value::StrList split on newlines (trailing empty line stripped if present). Both should refuse to hang if stdin is a TTY -- detect via std::io::IsTerminal and return Err('read_stdin: stdin is a terminal; pipe input or use args() instead').

2. exit(code) builtin: terminate the script process with the given integer exit code. code must be in [0, 255]. exit(0) is clean. Implementation: std::process::exit(code as i32).

3. mlpl-repl -f mode: if the script's FINAL expression evaluates to Value::Result with ok=false, exit non-zero (code 1) and print the err message to stderr. Otherwise exit 0. This makes MLPL scripts compose with Unix tooling (foo.sh && bar.sh).

TDD:
- RED: stdin tests use std::process::Command with a piped stdin and assert the script's output reflects the piped content. /tmp/test-stdin.mlpl writes 'print(read_stdin())'.
- RED: exit tests assert that 'mlpl-repl -f exit42.mlpl' exits with code 42 where exit42.mlpl is 'exit(42)'.
- RED: Err-propagation test asserts that 'mlpl-repl -f err.mlpl' exits non-zero where err.mlpl is 'Err("deliberate")'.
- GREEN: add the two read_stdin* builtins + exit() in mlpl-runtime; add the final-Err check in apps/mlpl-repl's run_script.

Quality gates: cargo test workspace; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md.

After this step ships MLPL scripts work in Unix pipes.