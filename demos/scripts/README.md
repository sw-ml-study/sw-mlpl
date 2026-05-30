# Example MLPL CLI scripts

Runnable `.mlpl` scripts that use a `#!/usr/bin/env mlpl-repl`
shebang, read positional arguments via `args()`, and set a Unix
exit code. They are the CLI counterpart to the in-browser demos.

Two are plain numeric utilities (`sum`, `stats`); two are small
ML use-cases (`predict` = logistic-regression inference,
`zscore` = feature standardization).

## Running

`mlpl-repl` must be on your `PATH` (install with `sw-install`).
Script arguments come after a `--` separator -- everything
before `--` is parsed as mlpl-repl's own flags, everything after
becomes the script's `args()`:

```sh
mlpl-repl demos/scripts/sum.mlpl -- 3 4 5      # -> 12
mlpl-repl demos/scripts/stats.mlpl -- 10 20 30 # -> 3 / 60 / 20
```

To run a script directly via its shebang, make it executable
first; the `--` is still required:

```sh
chmod +x demos/scripts/*.mlpl
./demos/scripts/sum.mlpl -- 3 4 5
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
| `stats.mlpl` | `-- 10 20 30` | `count: 3` / `sum: 60` / `mean: 20` | 0 |
| `stats.mlpl` | (none) | usage line on stderr | 2 |
| `predict.mlpl` | `-- 2.5` | `prob: 0.73` / `class: 1` | 0 |
| `zscore.mlpl` | `-- 10 12 23 ...` | `mean:` / `std:` / one `z:` per input | 0 |

The ML scripts: `predict.mlpl` applies a (pretend pre-trained)
1-feature logistic-regression model -- `sigmoid(w*x + b)` then a
0.5 threshold. `zscore.mlpl` standardizes its inputs with a
`u:nth` user-defined helper and two accumulation passes.

## See also

- `demos/classify.mlpl` -- a larger arg-parsing example
  (thresholds, `while`/`break`, verbose flag).
- `docs/usage.md` and `docs/lang-reference.md` -- the `args()`
  / `list_get` / `list_len` / `--` contract.
