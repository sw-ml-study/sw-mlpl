# Retire Function LOC FAILs in our code (saga 66)

7 single-function FAILs (>50 LOC) across 4 components. Each is a
mechanical "extract helpers" refactor. High ROI: ~30 minutes per
function for -1 FAIL each.

## Targets

| Component | File | Function | Lines |
|-----------|------|----------|-------|
| autograd | backward.rs | propagate | 77 |
| web | editor_panel.rs | editor_panel | 69 |
| web | render_main.rs | render_main | 76 |
| web | component_mode_bar.rs | mode_bar | 75 |
| web | handlers_submit.rs | make_submit_batch | 54 |
| wasm | lib.rs | eval_input_with_values | 61 |
| serve | main.rs | parse_args | 73 |

## Step plan

1. autograd-propagate
2. web-fails (4 functions, one step)
3. wasm-eval-input
4. serve-parse-args
5. close
