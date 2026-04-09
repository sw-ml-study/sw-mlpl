# MLPL Emacs Support

## Overview

MLPL includes a full Emacs integration in `elisp/`, providing editing, REPL interaction, SVG visualization, and interactive menus -- all using only built-in Emacs libraries (zero external package dependencies).

## File Structure

```
elisp/
  mlpl.el          Entry point / loader
  mlpl-mode.el     Major mode for .mlpl files
  mlpl-repl.el     Comint-based REPL integration
  mlpl-svg.el      SVG display (inline + gallery)
  mlpl-menu.el     Interactive SVG menus (tutorial, demos, help)
```

### Module Dependency Graph

```
mlpl.el (loader)
  +-- mlpl-mode.el       (no deps beyond Emacs)
  +-- mlpl-repl.el       (requires: comint, mlpl-mode)
  +-- mlpl-svg.el        (no deps beyond Emacs)
  +-- mlpl-menu.el       (requires: mlpl-svg)
```

## Installation

### Manual

```elisp
(add-to-list 'load-path "/path/to/sw-mlpl/elisp")
(require 'mlpl)
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
