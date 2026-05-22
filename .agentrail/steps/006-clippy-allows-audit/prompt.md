Tech-debt saga step 006: Clippy Allows audit.

12 #[allow(clippy::...)] attributes outside mlx-rs (vendored is exempt). Each one is a deferred refactor. Pass through every one:

1. List them: 'sw-checklist -v 2>&1 | grep "FAIL.*Clippy Allows" | grep -v mlx-rs'.
2. For each:
   - Identify what clippy lint is being suppressed.
   - Identify why it was suppressed (read the surrounding code + git blame).
   - Default action: refactor the code to satisfy the lint, drop the allow.
   - Exception: if the lint is genuinely incorrect for this case (very rare), add a one-line comment justifying it before the allow.
3. Re-run sw-checklist; each retirement is -1 FAIL.

Target retirement: -10 to -12 Clippy-Allows FAILs.

Strict gate: sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1.