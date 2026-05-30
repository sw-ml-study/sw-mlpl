# Example MLPL CLI scripts

Runnable `.mlpl` scripts that use a `#!/usr/bin/env mlpl-repl`
shebang, read positional arguments via `args()`, and set a Unix
exit code. They are the CLI counterpart to the in-browser demos.

## Running

`mlpl-repl` must be on your `PATH` (install with `sw-install`).
Script arguments come after a `--` separator -- everything
before `--` is parsed as mlpl-repl's own flags, everything after
becomes the script's `args()`:

```sh
mlpl-repl scripts/examples/sum.mlpl -- 3 4 5      # -> 12
mlpl-repl scripts/examples/stats.mlpl -- 10 20 30 # -> 3 / 60 / 20
```

To run a script directly via its shebang, make it executable
first; the `--` is still required:

```sh
chmod +x scripts/examples/*.mlpl
./scripts/examples/sum.mlpl -- 3 4 5
```

A leading `#!` line is a comment to the MLPL lexer, so the
shebang does not affect evaluation.

## Exit codes

A script's exit code follows its final value: a non-`Err`
result (or `exit(code)`) exits 0 (or the chosen code); a final
`Err(...)` exits 1 with the message on stderr. Both scripts
here demonstrate this:

| Script | Args | Output | Exit |
|--------|------|--------|------|
| `sum.mlpl` | `-- 3 4 5` | `12` | 0 |
| `sum.mlpl` | `-- 2 banana` | parse error on stderr | 1 |
| `stats.mlpl` | `-- 10 20 30` | `3` / `60` / `20` | 0 |
| `stats.mlpl` | (none) | usage line on stderr | 2 |

## See also

- `demos/classify.mlpl` -- a larger arg-parsing example
  (thresholds, `while`/`break`, verbose flag).
- `docs/usage.md` and `docs/lang-reference.md` -- the `args()`
  / `list_get` / `list_len` / `--` contract.
