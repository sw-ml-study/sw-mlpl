# Extract pets_tiny from mlpl-eval (saga 74)

First post-types Phase 1 extraction. pets_tiny.rs has:
- No Environment dependency
- Only mlpl-array + mlpl-eval-types imports
- 1 pub function: load() -> Result<Value, EvalError>
- 1 caller (loader.rs)

Cleanest possible extraction case after eval-types unlocked Value.
