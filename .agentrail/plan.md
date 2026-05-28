# Function LOC FAILs paydown -- resume (saga 72)

Saga 66 was archived as Active after step 001 (autograd propagate)
shipped. Resume the remaining 6 single-function refactors:

- mlpl-web:    editor_panel (69), render_main (76), mode_bar (75), make_submit_batch (54)
- mlpl-wasm:   eval_input_with_values (61)
- mlpl-serve:  parse_args (73)

Each step retires -1 FAIL via the "extract helpers" pattern.
