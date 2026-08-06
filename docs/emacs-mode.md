# Emacs mode for MLPL

`editors/emacs/mlpl-mode.el` provides editing support for
`.mlpl` files.

## Setup

```elisp
(add-to-list 'load-path "/path/to/sw-mlpl/editors/emacs")
(require 'mlpl-mode)
;; If mlpl-repl is not on PATH:
(setq mlpl-repl-program "/path/to/mlpl-repl")
```

Files ending in `.mlpl` open in `mlpl-mode` automatically.

## What it gives you

- Font-lock that keeps the language's THREE NAME KINDS
  visually distinct: a call `name(...)`, a quoted reference
  `:name` / `:u:name`, and a user function `u:name` -- plus
  `@word` annotation lines, `#` comments, strings, numbers.
- Brace-depth indentation (customize `mlpl-indent-offset`).
- imenu over `def u:` definitions.
- `C-c C-c` (`mlpl-run-buffer`): save and run the file with
  `mlpl-repl` in a compilation buffer.
- `C-c C-t` (`mlpl-run-tests`): run with `--test-events` and
  render each emitted test event as one
  `file:line: kind name status` line -- `RET` jumps to the
  test's source, courtesy of the typed JSONL event transport.

## Org-mode

`mlpl-repl --babel-session` provides a persistent stdin block
loop for `ob-mlpl`-style literate evaluation.
