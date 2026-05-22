Scripting saga step 003: add args() builtin + CLI passthrough in mlpl-repl.

Two parts that ship together because each is useless without the other:

1. args() builtin returns a Value::StrList of the trailing CLI args passed to the script (after the -- separator). When run from the REPL with no script (interactive mode), returns an empty list. Lives in crates/mlpl-runtime/src/builtins.rs.

2. mlpl-repl: extend the CLI parser to accept trailing positional args after 'mlpl-repl -f script.mlpl -- foo bar'. Without the -- separator, the existing behavior is preserved (no args; everything after -f path is silently ignored as it is today). The args are stored in the session and exposed to args() via a side channel (env or session field).

TDD:
- RED: unit test in crates/mlpl-runtime/tests/ that constructs a session with pre-set args = ['foo', 'bar'] and asserts args() returns a StrList of those two strings. Without a way to pre-set on the session, this step needs a small extension to Environment to carry the args. Document the extension.
- RED: integration test in apps/mlpl-repl/tests/ that spawns the binary via std::process::Command with '-f /tmp/script.mlpl -- a b c' and asserts the script's output (via print()) reflects the args. /tmp/script.mlpl writes args() to stdout via the print() builtin from step 001.
- GREEN: implement the Environment::args() carrier, the args() builtin, and the -- CLI parsing.

Quality gates: cargo test -p mlpl-runtime -p mlpl-eval -p mlpl-repl; cargo clippy --workspace --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower. Update docs/lang-reference.md.

After this step ships the user can write 'mlpl-repl -f train.mlpl -- 100 0.001' and the script can read those as strings via args(), then parse them with the to_number() from step 002.