Phase 1 step 001: Add file IO to MLPL. Implement load("path.csv") and load("path.txt") builtins.

CSV: auto-detect a header row (first row containing any non-numeric token). Return a DenseArray of the numeric cells; if a header is present, attach the header names as axis-1 labels so labels(X) comma-joins them. Numeric-only cells; error clearly on non-numeric data outside the optional header. Delimiters: comma.

TXT: return a Value::Str with the whole file contents. This feeds the Phase 2 BPE trainer.

Sandbox: terminal REPL (mlpl-repl + mlpl-build) accepts a --data-dir <path> flag (default ./data) that scopes relative reads; absolute paths and paths that traverse outside the sandbox via .. must error. Web REPL (mlpl-web via mlpl-wasm) does not expose filesystem load; instead add load_preloaded("<name>") that looks up a small in-memory corpus map compiled into the wasm binary (start with one entry like "tiny_corpus" returning a short string). The load("...") call errors cleanly in the web REPL pointing users at load_preloaded.

TDD:
(1) load a tiny test CSV with and without headers; verify shape, data, and labels when headers are present.
(2) load a tiny .txt returns Value::Str with the exact bytes.
(3) attempting load("/etc/passwd") errors with a sandbox message (terminal).
(4) attempting load("../outside.csv") errors.
(5) in the web path, load("x") errors with pointer to load_preloaded; load_preloaded("tiny_corpus") returns Value::Str.
(6) existing demos still run unchanged.
