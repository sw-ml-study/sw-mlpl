# Phase 1.5: extract eval-types (saga 73)

Move the kitchen-sink types (Value, EvalError, TokenizerSpec) +
their From impls into a small mlpl-eval-types crate. Unlocks the
Phase 1 extraction backlog (fetch, experiment, tag, singletons).

## Files to move

- value.rs (Value enum)
- error.rs (EvalError enum)
- error_fmt.rs (Display for EvalError)
- error_from_models.rs (From impls for mlpl-models-* errors)
- error_from_tools.rs (From impls for tool errors)
- tokenizer.rs (TokenizerSpec)

= 6 files into components/eval-types/crates/mlpl-eval-types/

## Cross-component deps the new crate gains

mlpl-array, mlpl-core, mlpl-eval-core (already external to eval) +
8 models-* crates (for the From impls -- carried due to Rust orphan
rule on `From<ForeignError> for EvalError`).

## Step plan

1. scaffold + move files
2. update mlpl-eval's ~42 internal callers to use mlpl_eval_types::*
3. close
