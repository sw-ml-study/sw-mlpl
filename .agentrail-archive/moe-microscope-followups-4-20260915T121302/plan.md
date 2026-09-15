# moe-microscope-followups-4

Address the optimizer-state + grad-surface findings F19-F22 from
../moe-microscope (docs/sw-mlpl-findings.md "Follow-up batch 4"). Lead with the
two process panics (F19, F20), then the optimizer-state footguns (F22, F21).

## Steps

1. f19-matmul-shape-panic -- a matmul inner-dim mismatch inside grad panics
   ("compatible matmul shapes") instead of the structured shape error F18 gave
   elementwise ops. Pre-validate matmul shapes in the grad path -> clean error.
   TDD.
2. f20-take-index-in-grad -- take's index parameter is unbound inside an inlined
   user function (resolves to a same-named global instead), and out-of-range
   panics. Resolve the index against the traced scope (F11/F15-class) and make
   out-of-range a clean error, not a panic. TDD.
3. f22-adam-state-rebind -- adam per-parameter state is keyed by name and
   survives rebinding the name to a new model. Clear optimizer state when a name
   is rebound to a new model; add a reset_optimizer() builtin. TDD.
4. f21-adam-in-user-fn -- adam inside a user function trains function-local
   copies with no persistence/error. Resolve the optimizer param list against
   the caller's bindings, or error/document loudly. TDD.
5. relay-and-close -- mark F19-F22 resolved/documented, refresh CHANGES + wiki,
   mark the saga shipped, rebuild binaries. --done.
