# Appendix: Emacs Integration

MLPL is designed to be editor-agnostic, but for those who live in Emacs, we provide a basic `mlpl-mode` for syntax highlighting and REPL integration.

## Installation

Add the following to your `init.el`:

```elisp
(add-to-list 'load-path "/path/to/sw-mlpl/tools/emacs")
(require 'mlpl-mode)
```

## Keybindings

- `C-c C-c`: Send the current line to the MLPL REPL.
- `C-c C-b`: Send the entire buffer to the MLPL REPL.
- `C-c C-z`: Switch to the MLPL REPL buffer.

## Why Emacs?

The buffer-driven workflow of Emacs perfectly matches the iterative nature of MLPL development. You can maintain your model code in one window and the trace output in another, creating a tight feedback loop.
