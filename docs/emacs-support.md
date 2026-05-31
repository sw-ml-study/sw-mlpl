# MLPL Emacs Support

## Overview

MLPL includes a full Emacs integration in `elisp/`, providing editing, REPL interaction, SVG visualization, and interactive menus -- all using only built-in Emacs libraries (zero external package dependencies).

## File Structure

```
elisp/
  mlpl-all.el      One-shot loader: loads everything, in order (START HERE)
  mlpl.el          Entry point / loader for the four core modules
  mlpl-mode.el     Major mode for .mlpl files
  mlpl-repl.el     Comint-based REPL integration
  mlpl-svg.el      SVG display (inline + gallery)
  mlpl-menu.el     Interactive SVG menus (tutorial, demos, help)
  mlpl-fold.el     Result folding for long output
  ob-mlpl.el       Org-babel backend (#+begin_src mlpl)
  mlpl-org.el      Org helpers (C-c C-c, table import)
  mlpl-bootstrap.el  All-in-one self-contained bundle (alternative)
```

### Module Dependency Graph

```
mlpl-all.el (one-shot loader)
  +-- mlpl.el (loader)
  |     +-- mlpl-mode.el     (no deps beyond Emacs)
  |     +-- mlpl-repl.el     (requires: comint, mlpl-mode)
  |     +-- mlpl-svg.el      (no deps beyond Emacs)
  |     +-- mlpl-menu.el     (requires: mlpl-svg)
  +-- mlpl-fold.el           (no deps beyond Emacs)
  +-- ob-mlpl.el             (requires: ob, soft mlpl-repl)   [if Org present]
  +-- mlpl-org.el            (requires: ob-mlpl, mlpl-fold)   [if Org present]
```

A regression gate (`scripts/test-elisp.sh`) checks that every
`elisp/*.el` has balanced parens and that `mlpl-all.el` loads under
`emacs -Q` with all modules `featurep`. Run it after editing any
`.el` file.

## Installation

### One-shot loader (recommended)

The simplest entry point is `mlpl-all.el`. Load that ONE file and
everything works -- it locates its own directory, puts it on
`load-path`, and `require`s every module in dependency order (the four
core modules, result folding, and -- when Org is present -- org-babel
plus the Org helpers). No manual `load-path` setup, no eval'ing the
`mlpl-*.el` files one at a time.

```elisp
;; In *scratch* (C-x C-e after the form) or your init file:
(load-file "/path/to/sw-mlpl/elisp/mlpl-all.el")
```

`M-x load-file RET /path/to/sw-mlpl/elisp/mlpl-all.el RET` works too.
After it loads, `M-x mlpl-all-version` echoes which modules came up.
The REPL/org-babel binary is auto-resolved (see "Finding the
`mlpl-repl` binary" below), so a GUI Emacs with a minimal PATH still
finds `mlpl-repl`.

### Org-babel (`#+begin_src mlpl` blocks)

When Org is available, `mlpl-all.el` loads `ob-mlpl.el`, so MLPL source
blocks evaluate with `C-c C-c`:

```org
#+begin_src mlpl :results output
x = reshape(range(6), [2,3])
x
#+end_src
```

`C-c C-c` on the block runs the code through `mlpl-repl -f` (the binary
resolved the same way as the interactive REPL -- `exec-path` first,
then the `sw-install` location) and inserts the result:

```org
#+RESULTS:
: 0 1 2
: 3 4 5
```

SVG-producing blocks emit raw `<svg>` (render with `org-toggle-inline-images`
when exported to HTML). Point at a non-default build with
`(setq org-babel-mlpl-command "/absolute/path/to/mlpl-repl")`. Editing
a block with `C-c '` opens it in `mlpl-mode` (registered via
`org-src-lang-modes`). Sessions are not yet supported.

### Manual

If you'd rather wire it up by hand, add the `elisp/` dir to
`load-path`, then `(require 'mlpl)`. This one
`require` pulls in the four core modules (`mlpl-mode`, `mlpl-repl`,
`mlpl-svg`, `mlpl-menu`) so editing, the REPL, SVG display, and the
menu/demos all work. **Do NOT eval the individual `mlpl-*.el` files
one at a time** -- they have load-order dependencies, so a partial
load leaves the REPL/menu/demos broken. (The all-in-one bundle
`mlpl-bootstrap.el` is the alternative: a single self-contained file
you can load with no `load-path` setup -- but use *either* `(require
'mlpl)` *or* the bundle, not both.)

```elisp
(add-to-list 'load-path "/path/to/sw-mlpl/elisp")  ; full path to the elisp dir
(require 'mlpl)
```

Optional extras (org-babel, folding) are separate `require`s -- or
just use the one-shot loader above, which pulls them in automatically
when Org is present:

```elisp
(require 'mlpl-fold)  ; result folding
(when (require 'org nil t)
  (require 'ob-mlpl)  ; Org-babel: #+begin_src mlpl blocks
  (require 'mlpl-org) ; C-c C-c on a block, table import, result folding
  (add-to-list 'org-src-lang-modes '("mlpl" . mlpl)))
```

### Finding the `mlpl-repl` binary

`mlpl-repl-start` runs the program named by `mlpl-repl-command`
(default `"mlpl-repl"`). GUI Emacs -- especially on macOS -- starts
with a minimal PATH that usually omits `~/.local/softwarewrighter/bin`
(the `sw-install` location), so a bare `mlpl-repl` fails with
"Searching for program: No such file or directory". The REPL resolver
falls back to `~/.local/softwarewrighter/bin/mlpl-repl` automatically;
if your binary lives elsewhere, set the full path:

```elisp
(setq mlpl-repl-command "/absolute/path/to/mlpl-repl")
;; or, for an MLX-enabled / connect-mode build, see "Connect mode" below.
```

### Use-package

```elisp
(use-package mlpl
  :load-path "/path/to/sw-mlpl/elisp"
  :commands (mlpl-mode mlpl-menu mlpl-switch-to-repl)
  :mode "\\.mlpl\\'"
  :config
  (setq mlpl-repl-command "cargo run -p mlpl-repl --quiet --"))
```

### Byte-compile (optional)

```bash
cd elisp && emacs -batch -f batch-byte-compile *.el
```

## Design

### mlpl-mode (Major Mode)

A full major mode derived from `prog-mode` for editing `.mlpl` files.

**Features:**
- Syntax highlighting via `font-lock` with custom faces
- Indentation based on bracket nesting (`{`, `[`, `(`)
- Electric indentation on `)`, `]`, `}`, and newline
- Outline mode support (sections: `repeat`, `train`, `param`, `tensor`)
- Completion at point for keywords, builtins, and REPL commands
- Auto-mode-alist registration for `.mlpl` files

**Font-lock categories:**

| Category | Face | Examples |
|----------|------|---------|
| Keywords | `mlpl-keyword-face` (bold) | `repeat`, `train` |
| Context keywords | `mlpl-context-keyword-face` (type) | `param[`, `tensor[` |
| Builtins | `mlpl-builtin-face` | `iota`, `shape`, `svg`, `matmul` |
| Numbers | `mlpl-number-face` (yellow) | `42`, `1.5`, `-3` |
| Operators | `mlpl-operator-face` (orange) | `+`, `-`, `*`, `/`, `=` |
| Strings | `font-lock-string-face` | `"scatter"`, `"hello"` |
| Comments | `font-lock-comment-face` | `# this is a comment` |
| REPL commands | `font-lock-preprocessor-face` | `:help`, `:trace on` |

**Keybindings (in .mlpl buffers):**

| Key | Command |
|-----|---------|
| `C-c C-z` | Switch to REPL |
| `C-c C-c` | Send current line to REPL |
| `C-c C-r` | Send region to REPL |
| `C-c C-b` | Send entire buffer to REPL |
| `C-c C-l` | Load a .mlpl file |
| `C-c m` | Open MLPL menu |

### mlpl-repl-mode (Comint REPL)

A comint-derived major mode for interacting with the MLPL REPL.

**Features:**
- Full comint integration (input history with M-p/M-n, etc.)
- REPL prompt detection and read-only prompts
- SVG output capture from REPL results
- Inline SVG display for small visualizations (< 20KB)
- Automatic opening of larger SVGs in the gallery buffer
- Completion at point in REPL input
- Input ring with 1000 entries

**Keybindings (in REPL buffer):**

| Key | Command |
|-----|---------|
| `TAB` | Complete at point |
| `C-c C-o` | Show last SVG in gallery |
| `C-c C-k` | Clear SVG gallery buffer |
| `M-p` / `M-n` | History navigation (comint) |
| `C-c C-u` | Kill input line (comint) |

**SVG Output Flow:**

```
REPL evaluates svg(data, "scatter")
  -> eval returns <svg>...</svg> string
  -> comint output filter detects <svg in output
  -> if < 20KB: render inline in REPL buffer
  -> if >= 20KB: open in *MLPL Graphics* gallery
  -> always stored in gallery history
```

### mlpl-svg-gallery-mode (SVG Display)

A `special-mode`-derived mode for viewing MLPL SVG visualizations.

**Features:**
- Dedicated `*MLPL Graphics*` buffer
- SVG metadata info panel (size, dimensions)
- Gallery navigation (n/p) through all captured SVGs
- Save individual SVGs to files
- Header-line showing gallery position

**Keybindings (in gallery buffer):**

| Key | Command |
|-----|---------|
| `n` | Next gallery item |
| `p` | Previous gallery item |
| `g` | Redraw current item |
| `s` | Save SVG to file |
| `C` | Clear gallery |
| `q` | Quit window |

### mlpl-menu (Interactive SVG Menus)

Graphical interactive menus built with `svg.el`, inspired by the
`graphical-experiments` project patterns.

**Features:**
- SVG-rendered logo header
- Interactive card-based menu with selection highlighting
- Mouse click and keyboard navigation support
- Submenus for demos (auto-populated from `demos/` directory)
- Built-in tutorial with 8 sections covering the full language
- Help reference with categorized built-in functions

**Main Menu (M-x mlpl-menu):**

| Item | Action |
|------|--------|
| Tutorial | Open interactive 8-section tutorial |
| Demos | Browse and open demo files from `demos/` |
| REPL | Start the MLPL REPL |
| Help | Built-in function reference |
| Graphics | Open SVG gallery |
| Run File | Load and execute a .mlpl file |

**Keybindings (in menu buffer):**

| Key | Command |
|-----|---------|
| `n` | Next item |
| `p` | Previous item |
| `RET` | Select item |
| `q` | Quit menu |
| `g` | Redraw |

**Color Palette (Catppuccin Mocha):**

| Name | Hex | Usage |
|------|-----|-------|
| bg | `#1e1e2e` | Backgrounds, logo card |
| surface | `#313244` | Tutorial section headers, help sections |
| overlay | `#45475a` | Borders, inactive card strokes |
| text | `#cdd6f4` | Primary text |
| subtext | `#a6adc8` | Secondary text, descriptions |
| blue | `#89b4fa` | Tutorial section titles |
| green | `#a6e3a1` | Demo items |
| peach | `#fab387` | REPL menu item |
| mauve | `#cba6f7` | Help section titles |
| yellow | `#f9e2af` | Selection highlight |
| teal | `#94e2d5` | Graphics item |

## Customization

All customizable variables are in the `mlpl`, `mlpl-repl`, and `mlpl-svg` groups.

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `mlpl-indent-level` | `2` | Spaces per indent level |
| `mlpl-repl-command` | `"cargo run -p mlpl-repl --quiet --"` | REPL startup command |
| `mlpl-repl-args` | `'()` | Extra REPL arguments |
| `mlpl-repl-prompt-regexp` | `"^mlpl> "` | Prompt detection regex |
| `mlpl-svg-inline-max-bytes` | `20000` | Inline SVG size threshold |
| `mlpl-svg-external-width` | `600` | Gallery display width |
| `mlpl-svg-color-bg` | `"#1e1e2e"` | SVG canvas background |

### Customize via Emacs

```
M-x customize-group RET mlpl RET
M-x customize-group RET mlpl-repl RET
M-x customize-group RET mlpl-svg RET
```

## Usage Examples

### Editing .mlpl Files

Open any `.mlpl` file -- `mlpl-mode` activates automatically. Syntax highlighting, indentation, and REPL integration are available.

### Running Code in the REPL

```
C-c C-z          ;; open/switch to REPL
C-c C-c          ;; in .mlpl buffer, send current line
C-c C-r          ;; send selected region
C-c C-b          ;; send entire buffer
```

### Viewing SVG Output

```mlpl
# In the REPL:
x = random[100, 2]
svg(x, "scatter")    ;; small SVG -> inline in REPL
m = random[50, 50]
svg(m, "heatmap")    ;; large SVG -> opens in gallery
```

After SVG output appears:
- `C-c C-o` in REPL to re-open last SVG in gallery
- `n`/`p` in gallery to browse history
- `s` in gallery to save to file

### Using the Menu

```
M-x mlpl-menu      ;; or C-c C-h in a .mlpl buffer
```

Navigate with `n`/`p`, select with `RET`. The Tutorial walks through all language features. The Demos submenu lists all scripts in `demos/`.

### Connect mode, `:ask`, and MLX training

The Emacs REPL runs the `mlpl-repl` binary as a Comint inferior
process, so EVERYTHING the CLI REPL supports is available by typing
into the `*MLPL REPL*` buffer -- including the connect/server
workflows (`:ask`, `--connect`, `device("mlx")`). The browser is not
required for these.

Pick the REPL binary + mode with `mlpl-repl-command`:

```elisp
;; Local (default): runtime runs in-process.
(setq mlpl-repl-command "mlpl-repl")
;; Local GPU: an MLX-enabled build runs device("mlx") on the Mac GPU.
(setq mlpl-repl-command "/path/to/mlx-build/mlpl-repl")
;; Client/server: route eval to a running mlpl-serve (its GPU/peer).
(setq mlpl-repl-command "mlpl-repl --connect http://127.0.0.1:6464")
```

**`:ask` (LLM with REPL context).** Works in any mode. The CLI
`:ask` reads `OLLAMA_HOST` (default `http://localhost:11434`) and
`OLLAMA_MODEL` (default `llama3.2`) from the environment, so launch
Emacs pointed at your Ollama box:

```bash
OLLAMA_HOST=http://large12.local:11434 OLLAMA_MODEL=qwen2.5-coder:14b emacs
```

Then in the REPL: `:ask what did I just train?` -- the question is
sent with recent REPL history + `:models` + any demo narration as
context.

**MLX training.** Two routes:

- Local GPU: point `mlpl-repl-command` at an MLX-enabled build, then
  `device("mlx") { train ... }` runs on the Mac GPU in the inferior
  process.
- Offload (client/server): run `mlpl-serve --features mlx` and set
  `mlpl-repl-command` to `mlpl-repl --connect <url>`. Eval (including
  the `device("mlx")` block) runs server-side on the GPU, exactly
  like the web connect mode; connect mode adds slash-commands
  (`:inspect`, `:vars`) and cancel.

**Models.** `:models` lists the MLPL models bound in the session (in
any mode). The web-only `:models ollama` host listing is not a CLI
command; in Emacs use `OLLAMA_MODEL` / a shell `ollama list`, and
`:ask` uses the configured model.

See `docs/using-cli-server.md` and `docs/using-ollama.md` for the
server + Ollama setup these reuse.

## Architecture Decisions

### Major Mode (not Minor)

`mlpl-mode` is a major mode because MLPL has distinct syntax, indentation rules, and REPL integration needs that don't layer well onto another mode. It derives from `prog-mode` to inherit standard programming mode behavior.

### Comint-based REPL

Using `comint-mode` as the base for the REPL provides mature input handling: history, completion, prompt detection, and process management. This is the same approach used by `inferior-python-mode`, `cider`, and `geiser`.

### Dual SVG Display Strategy

Small SVGs (under 20KB, configurable) are displayed inline in the REPL output for immediate feedback. Larger SVGs open in the dedicated `*MLPL Graphics*` gallery buffer to avoid cluttering the REPL. All SVGs are stored in a gallery list for browsing regardless of display location.

### Zero External Dependencies

All modules use only built-in Emacs libraries (`svg.el`, `comint.el`, `subr-x.el`). This ensures the integration works on any Emacs 26.1+ installation without package manager setup.

### SVG Rendering Pipeline

Follows the pattern from `graphical-experiments`:

```
svg-create -> svg-primitives -> svg-image -> insert-image
```

Pure builder functions produce SVG objects; rendering is always a separate step. This makes SVG construction testable and composable.

## Future Work

- [ ] Treesitter grammar for more precise syntax highlighting
- [ ] REPL completion from actual environment (query REPL for vars)
- [ ] SVG thumbnails in the demos menu
- [ ] Trace visualization (from `:trace json` output)
- [ ] `:trace json` integration with Emacs outline/org-mode
- [ ] Integration with `org-babel` for literate MLPL programming
- [ ] Flycheck/diagnostics via `mlpl-parser` error messages
- [ ] Eldoc integration for function signatures
- [ ] Xwidget-based interactive SVG (zoom, pan)
