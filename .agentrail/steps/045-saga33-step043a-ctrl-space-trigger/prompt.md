Bug fix + UX change: the Tab trigger from step 043 does not work in practice -- pressing Tab navigates focus out of the REPL input to the footer (browser default focus traversal wins over the Yew keydown handler's preventDefault, despite the same mechanism working fine for ArrowUp/ArrowDown). User feedback: switch the completion trigger to Ctrl+Space (the IDE standard -- VS Code, IntelliJ, Emacs) and REMOVE the Tab arm entirely to avoid further user confusion.

(1) In apps/mlpl-web/src/handlers.rs make_keydown: replace the "Tab" arm with a guarded arm matching ctrl+space. The web_sys::KeyboardEvent exposes ctrl_key() and code() (returns physical key code like "Space"). The cleanest pattern is a leading guard in the match: `k if k == " " && e.ctrl_key() => { ... }` -- but Rust's match guards on a key string + ctrl_key() side condition don't compose cleanly with the other arms, so a simpler approach is to check ctrl_key + key code BEFORE the match: if e.ctrl_key() && e.code() == "Space" { e.prevent_default(); handle_tab(&e, &input_value, &completion_candidates); return; }.

(2) Call prevent_default() so the browser does not open any built-in Ctrl+Space action.

(3) Rename `handle_tab` to `handle_completion` (it's no longer Tab-specific).

(4) Update the 'Tab completion (REPL)' glossary entry: change the heading to 'Completion popup (REPL)' (alphabetical position shifts), replace every 'Tab' reference with 'Ctrl+Space', explain the IDE-standard choice. Cross-reference the old name in a one-line redirect.

(5) Update apps/mlpl-web/src/completion.rs module doc to reflect the trigger change.

(6) Add a unit test for the keypress predicate. Extract a tiny pure fn `is_completion_trigger(ctrl_key: bool, code: &str) -> bool` returning ctrl_key && code == "Space"; test the four combinations.

(7) Pages rebuild required since apps/mlpl-web changed.

(8) The static completion logic (REPL_COMMANDS, KEYWORDS, runtime_builtin_names) and the popup chips component are unchanged -- this is a trigger swap only.