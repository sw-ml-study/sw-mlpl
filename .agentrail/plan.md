# Unblocking mlplunit: prioritized sw-MLPL changes

Source contract: `../mlplunit/docs/sw-MLPL-changes-needed.md`
(seven capability areas; its own analysis says the first four
are independently implementable and items 5-7 build on 3+4).
This plan is the sw-MLPL-side prioritization and saga mapping.
Written 2026-08-05; the gen-state-kv-cache saga is paused
(archived Active) until this program lands its P0/P1 items.

## Priority order and rationale

P0-a. **Structural equality + stable rendering** (`equal(a, b)`,
`repr(value)`) -- contract item 2. FIRST because it is the
smallest surface (two eval-layer builtins recursing over the
existing Value kinds) with immediate payoff: every mlplunit
assertion becomes honest (no more escaping hard errors on type
mismatch) and diagnostic (bounded expected/actual rendering).
No dependencies.

P0-b. **Static include** (`include "path"`) -- contract item 1,
mlplunit's own top priority: it removes host-side source
concatenation entirely. Medium effort with sharp edges the
contract spells out (literal-path only, `--source-dir` sandbox,
traversal rejection, load-once + cycle chains, diagnostics that
keep file/line/column, a source-provider boundary so WASM stays
filesystem-free). Reference design lives in mlplunit's
`docs/static-include-design.md`.

P1-a. **First-class callable user functions** (`:u:name` +
`call(f, args...)`) -- contract item 3. The largest language
change (a function-reference Value kind + uniform invocation),
and the gate for items 4-6. Double payoff: the APL2
higher-order saga (Track 8) lists first-class user functions as
ITS prerequisite too -- `each`/`scan`/`outer` and the vectorized
`evaluate_rows` all wait on the same capability.

P1-b. **Test metadata + reflection** (`@test`, enumeration
without execution) -- contract item 4. Parser attributes plus a
reflection builtin over the user-fn table; pairs naturally with
P1-a and completes native discovery.

P2. **Parameterized cases, fixture lifecycle, structured
events/process controls** -- contract items 5-7. All build on
P1; the lifecycle item additionally needs a guaranteed-finally
mechanism (a real language design task). Scheduled as a
follow-up saga once mlplunit has adopted P0/P1.

## Saga: mlplunit-unblock

1. plan (this document)
2. equal-repr -- `equal`/`repr` builtins, TDD against the
   semantics in contract item 2; run mlplunit's
   `scripts/check-capabilities` expecting the structural-
   equality fixture to flip AVAILABLE.
3. include-design -- read mlplunit's static-include-design.md;
   design the parse-time loader, source-provider seam, sandbox
   rules; pause for review.
4. include-impl -- the loader per design; acceptance =
   mlplunit's `scripts/verify-native-include` passes without the
   host `--include`.
5. callables-design -- `:u:name` reference value + `call`;
   interaction with the BuiltinRef story and the Track 8 HOF
   saga; pause for review.
6. callables-impl, then metadata -- sized after the design
   review.
7. close -- capabilities re-run, queue update, wiki, handoff
   notes for mlplunit adoption.

## Standing check

After each shipped step: build `mlpl-repl` release and run
mlplunit's `scripts/check-capabilities`; a capability that flips
AVAILABLE is reported to mlplunit for same-change adoption per
its definition-of-ready.
