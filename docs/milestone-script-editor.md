# Script Editor Milestone

Saga 36, proposed.

## Why this exists

The web REPL is ephemeral and single-line. Users can't
save work, load scripts, or edit multi-line programs.
The desktop `mlpl-repl -f script.mlpl` supports file
execution but the web UI has no equivalent.

## Deliverables

1. **Upload .mlpl** -- file picker button loads a text
   file and runs it line by line (like a demo).
2. **Inline editor** -- a multi-line text area where
   users type or paste scripts. Run button executes all
   lines. Stays in the browser.
3. **Download .mlpl** -- saves the current REPL history
   (input lines only) as a downloadable .mlpl file.
4. **Edit + re-run** -- after running a script, the
   editor retains the source so the user can edit and
   re-run without re-uploading.

## Architecture

### Editor panel

A new tab in the header: "Editor" alongside REPL,
Tutorial, Paths. The editor panel shows a `<textarea>`
(or a lightweight code editor like CodeMirror/Monaco
if we want syntax highlighting later -- textarea for
MVP).

Buttons: Run (execute all lines), Clear, Save (.mlpl
download), Load (file picker).

### Execution

Lines execute through the existing `on_run_batch`
callback (same path as tutorial Run All). Each line
becomes a history entry. The :3d integration works
automatically -- if 3D is on, each line emits a
sculpture.

### File format

Plain text, one MLPL expression per line. Comments
start with `#`. Empty lines are skipped. Extension
`.mlpl`. This matches the desktop `mlpl-repl -f`
format exactly.

### Download

Collects all input lines from the REPL history
(filtering out narration/system entries), joins with
newlines, triggers a browser download via
`Blob` + `URL.createObjectURL` + click-on-anchor
pattern.

## Steps

### Step 001 -- Editor tab + textarea

New HeaderMode::Editor. Clicking the Editor tab shows
a full-height textarea with monospace font. Run button
at the top. No execution yet -- just the UI shell.

### Step 002 -- Run + Load

Run button submits all non-empty non-comment lines
via on_run_batch. Load button opens a file picker
that reads a .mlpl text file into the editor textarea.

### Step 003 -- Save/download

Save button collects REPL input history, creates a
Blob, triggers download as `session.mlpl`. Also a
"Copy to editor" button that loads history into the
editor for editing.

### Step 004 -- Polish + integration

Keyboard shortcut Ctrl+Enter to run from editor.
Line numbers in the textarea (CSS counter or
pre-line numbering). Editor retains content across
tab switches. Tour stop for the editor.

## Quality requirements

Same as saga 35. Warning-target design. Pages rebuild.
