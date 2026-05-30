# UI / CLI Fixes Checklist

Running checklist of UI and CLI feedback raised in review, with
status. `[x]` done + shipped, `[~]` coded (awaiting deploy/
install), `[ ]` not started.

## Web playground (3D viz + REPL)

- [x] **MathJax subscript bracing (#4).** `d_model` rendered as
  `d`-sub-`m` + full-size "odel". `mlplToLatex` now braces
  multi-char subscripts (`d_model` -> `d_{model}`), leaving `_`
  inside `\text{...}` literal. (`braceSubscripts` in stage3d.js.)
  Coded; needs pages rebuild to go live.
- [x] **REPL output collapses in 3D (#3).** FIXED: When 3D is open the
  output pane shrinks to ~2 lines (only the prompt visible).
  Want: at least the last 3 output lines visible, plus a
  scroll-up affordance (up arrow / hint) signalling more above.
- [x] **Attention hover-to-trace (#5).** Working after a
  hard-refresh (was a deploy-cache lag). Hovering a cell dims
  the grid and lights its row + column + token labels.
- [x] **Reset 3D camera on toggle.** FIXED: Toggling 3D off/on does not
  re-orient; multi-head attention shows edge-on (a vertical row)
  with no way back to the normal horizontal view. Want: toggling
  out of 3D and back in restores the default camera orientation.
- [x] **Escape-close parity.** FIXED: Closing the inspector with Escape
  removes the `(i)` affordance, but clicking the dialog X or
  outside keeps it. All three exit paths should behave the same
  (stay in the mode where you can scroll left/right after close).
- [x] **Dialog prev/next carousel.** FIXED (Left/Right arrows): While a dialog is open,
  jump to the previous/next sculpture's dialog via arrow keys
  and/or an on-dialog nav control -- a carousel, instead of
  close -> move -> click `(i)` to reopen.

### Already shipped earlier this review

- [x] REPL auto-focus after splash dismiss.
- [x] `2D | 3D` toggle on both REPL input lines (active bold,
  other clickable; drives `:3d`/`:2d`).
- [x] Taller REPL output default in 3D (30vh -> 45vh) -- partial;
  superseded by the `#3` collapse fix above.

## CLI (`mlpl-repl` script mode)

- [x] **Stale installed binary (#7).** Installed `mlpl-repl` was
  v0.7.0 (April), predating script mode. Rebuilt + `sw-install`d
  the current v0.20.0; `args()` / `--` / exit codes now work.
- [x] **Quiet script mode + flags.** FIXED: Script mode echoes every
  source line with `> `, burying the real output. Want:
  - quiet by default (no per-line echo);
  - `-v` / `--verbose` to opt into the echo (and trace);
  - `-h` / `--help` usage output;
  - `-V` / `--version` (confirm it works in all paths).
- [x] **CLI demo scripts.** `demos/scripts/{sum,stats}.mlpl`
  + README: shebang + `args()` + exit codes. Created, tested
  against the installed binary; pending commit.

## Notes

- The `--` separator is required to pass args to a script
  (`mlpl-repl script.mlpl -- ARG`); bare positionals after the
  script path are not captured. Documented in
  `demos/classify.mlpl` and `demos/scripts/README.md`.
- Web fixes are batched into a single `pages/` rebuild + deploy
  to avoid repeated multi-GB WASM builds.
