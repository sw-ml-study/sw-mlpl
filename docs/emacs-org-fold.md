# MLPL Emacs: Org-babel & Output Folding

## Overview

MLPL provides three integrations for working with Org mode:

| Module | File | Purpose |
|--------|------|---------|
| `ob-mlpl` | `elisp/ob-mlpl.el` | Org-babel source block evaluation |
| `mlpl-fold` | `elisp/mlpl-fold.el` | Fold/expand large array outputs |
| `mlpl-org` | `elisp/mlpl-org.el` | Org-mode convenience commands |

All are loaded by `mlpl-bootstrap.el` with no additional setup.

## Org-babel: `#+begin_src mlpl`

### Setup

Add to your init (after loading mlpl-bootstrap):

```elisp
;; Register mlpl as an org-babel language
(with-eval-after-load 'org
  (add-to-list 'org-babel-load-languages '(mlpl . t)))
```

Or if you prefer to load `ob-mlpl` separately:

```elisp
(add-to-list 'org-babel-load-languages '(mlpl . t))
(require 'ob-mlpl)
```

### Basic Usage

```org
* Matrix Operations

#+begin_src mlpl
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
matmul(a, b)
#+end_src

#+RESULTS:
: 19 22
: 43 50
```

Execute with `C-c C-c` with point inside the source block.

### Numeric Results as Org Tables

When output is multi-line numeric data, it is automatically converted to an Org table:

```org
#+begin_src mlpl
m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transpose(m)
#+end_src

#+RESULTS:
| 1 | 4 | 7 |
| 2 | 5 | 8 |
| 3 | 6 | 9 |
```

### SVG Output

SVG visualizations are passed through inline:

```org
#+begin_src mlpl
x = random[50, 2]
svg(x, "scatter")
#+end_src
```

### Variable Passing

Pass elisp variables into MLPL blocks:

```org
#+begin_src mlpl :var n=10
iota(n)
#+end_src

#+RESULTS:
: 0 1 2 3 4 5 6 7 8 9
```

### Header Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `:results` | `output replace` | How results are handled |
| `:var` | nil | Pass variables from elisp |
| `:fold-threshold` | `8` | Lines before folding (overrides default) |

## Output Folding

### How It Works

Large numeric outputs are automatically folded to keep buffers readable. This mirrors the web UI's `<details>` element behavior.

**Folded appearance:**

```
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1
0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2
... 7 more line(s)  [10x10 (100 values)  min=0.10  max=1.20  mean=0.65  median=0.65  std=0.29]
```

**Expanded appearance:** click or press `RET` on the summary line to reveal the full output.

### Thresholds

Folding activates when output exceeds **both** thresholds:

| Variable | Default | Description |
|----------|---------|-------------|
| `mlpl-fold-line-threshold` | `8` | More than N lines |
| `mlpl-fold-char-threshold` | `200` | More than N characters |
| `mlpl-fold-preview-lines` | `3` | Lines shown before fold |

These match the web UI thresholds (`LINE_THRESHOLD=8`, `CHAR_THRESHOLD=200`).

### Numeric Summary

When output is a numeric grid (all values parse as numbers), the fold summary includes statistics computed in Elisp:

- **Shape** (rows x cols, total values)
- **min, max** -- range
- **mean** -- arithmetic average
- **median** -- 50th percentile
- **std** -- standard deviation

### Where Folding Applies

1. **Org-babel results** -- large numeric outputs from `#+begin_src mlpl` blocks are automatically folded after execution
2. **REPL output** -- `C-c C-f` in an org buffer folds all large results
3. **Manual** -- select a region and run `mlpl-fold-region`

### Interactive Commands

| Command | Key | Description |
|---------|-----|-------------|
| `mlpl-fold-toggle` | `RET` on fold | Expand/collapse folded output |
| `mlpl-fold-region` | none | Fold a selected region |
| `mlpl-org-fold-results` | `C-c C-f` | Fold all large results in buffer |
| `mlpl-org-unfold-all` | `C-c C-u` | Expand all folds in buffer |

Folded regions are clickable (mouse-1) and show a hover tooltip.

## Org-mode Convenience Commands (`mlpl-org`)

These commands work in any Org buffer containing MLPL source blocks.

| Command | Key | Description |
|---------|-----|-------------|
| `mlpl-org-execute-src-block` | `C-c C-c` | Execute block at point (delegates to org-babel) |
| `mlpl-org-send-block-to-repl` | `C-c C-v` | Send block to REPL for interactive exploration |
| `mlpl-org-execute-buffer` | `C-c C-b` | Execute all MLPL blocks in buffer |
| `mlpl-org-fold-results` | `C-c C-f` | Fold all large results |
| `mlpl-org-unfold-all` | `C-c C-u` | Expand all folds |
| `mlpl-org-table-to-mlpl` | none | Convert Org table at point to MLPL array literal |

## Architecture

### File Structure

```
elisp/
  ob-mlpl.el      Org-babel language definition
  mlpl-fold.el    Fold/expand engine (pure functions)
  mlpl-org.el     Org-mode convenience layer (uses mlpl-fold)
  mlpl-bootstrap.el  All of the above concatenated
```

### Dependencies

```
ob-mlpl.el  -->  ob.el (org-babel API)
mlpl-fold.el     (no deps beyond cl-lib)
mlpl-org.el  -->  ob-mlpl.el, mlpl-fold.el
```

### Execution Pipeline (ob-mlpl)

```
org-babel-execute:mlpl(body, params)
  |
  |--> write body to temp .mlpl file
  |--> call-process: mlpl-repl -f <tmp> --svg-out <dir>
  |--> read output
  |
  +--> if starts with <svg:  pass through (SVG result)
  +--> if numeric multiline: convert to org table + fold
  +--> else: plain text result
```

### Fold Pipeline

```
mlpl-fold--should-fold-p(output)
  |-- check line-count > threshold
  |-- check char-count > threshold
  |
  mlpl-fold--numeric-summary(output)
    |-- parse as numeric grid (all tokens must be f64)
    |-- compute min, max, mean, median, std
    |-- return summary string
    |
  mlpl-fold--insert(output)
    |-- insert preview lines (first 3)
    |-- insert summary line with stats
    |-- create overlay with:
    |     - hidden text (remaining lines)
    |     - keymap: RET and mouse-1 to expand
    |     - face: mlpl-fold-summary-face (blue, bold, boxed)
    |     - help-echo tooltip
```

### Comparison: Emacs vs Web UI

| Feature | Web UI | Emacs |
|---------|--------|-------|
| Fold trigger | lines > 8 AND chars > 200 | Same thresholds |
| Fold UI | `<details>` / `<summary>` HTML | Emacs overlay + text properties |
| Expand | Click triangle | RET or mouse-1 on summary |
| Summary | shape + min/max/mean/median/std | Same statistics |
| Arrow indicator | `▸` / `▾` via CSS `::before` | Blue bold boxed text |
| Hover | `color: var(--blue)` | `mouse-face: highlight` + tooltip |
| Non-numeric | Not collapsed | Not collapsed |
| SVG output | Inline SVG rendering | Passed through as result |

## Customization

### Global Settings

```elisp
;; Fold thresholds
(setq mlpl-fold-line-threshold 10)
(setq mlpl-fold-char-threshold 300)
(setq mlpl-fold-preview-lines 5)

;; Org-babel
(setq org-babel-mlpl-command "mlpl-repl")
```

### Per-Block Override

```org
#+begin_src mlpl :fold-threshold 20
m = random[20, 20]
m
#+end_src
```

### Customize via Emacs

```
M-x customize-group RET mlpl-fold RET
```

## Example: Literate MLPL Document

```org
* Logistic Regression from Scratch

#+begin_src mlpl
# Generate training data
x = random[100, 2]
y_label = random[100]
#+end_src

#+begin_src mlpl
# Initialize weights
w = param[2, 1]
b = param[1, 1]

# Sigmoid
sigmoid = fn(z) { 1 / (1 + exp(-z)) }

# Forward pass
z = matmul(x, w) + b
pred = sigmoid(z)

# Loss
loss = -mean(y_label * log(pred) + (1 - y_label) * log(1 - pred))
loss
#+end_src

#+RESULTS:
: 0.6931471805599453
```

The loss value is short and displayed directly. If you were to display a large matrix, it would be folded with a summary line showing shape and statistics.
