# Retrofitting Result<val, err> onto prior demos and tutorials

Status: draft, Saga 29 step 012 follow-up
Last updated: 2026-05-19

## Why retrofit?

Saga 29 step 012 shipped a first-class `Value::Result { ok, payload }`
with constructors `ok(v)` / `err(v)` and the five accessors
`is_ok`, `is_err`, `unwrap`, `unwrap_or`, `err_message`. Today
the type is built but unused -- every existing fallible builtin
still raises `EvalError` and crashes the program.

A retrofit pass would let users write defensive demos that
demonstrate failure modes, recover from bad input, and chain
fallible ops without an early-exit. Without it, the type's only
in-tree consumer is the upcoming `:upload x` REPL command.

## Goals (in priority order)

1. **Make at least one tutorial lesson read like a real error-
   handling example** -- "user types something bad, the program
   degrades gracefully instead of crashing." This is the
   educational case.
2. **Remove the worst footguns in the current demos** --
   `load_preloaded` typos, malformed string args, etc., that
   today print a stack-trace-equivalent and stop the REPL.
3. **Set the precedent for future fallible builtins** --
   `:upload`, `load_image(path)`, `fetch(url)`, parse helpers --
   so they consistently return `Result<...>` rather than
   adding new bespoke error paths.

Non-goal: rewriting every existing builtin to return Result.
The opt-in pattern below preserves backward compatibility
(important because hundreds of demos and tests rely on the
"raises EvalError" behavior of `load_preloaded`, `apply_tokenizer`,
etc.).

## The retrofit pattern: `try_<op>` siblings

Rather than changing existing builtins' return types (massive
breakage), add a `try_` sibling for each fallible op:

| Existing (raises EvalError) | Result-returning sibling |
|-----|-----|
| `load_preloaded(name)` | `try_load_preloaded(name)` |
| `load(path)` | `try_load(path)` |
| `load_images(dir, [H, W])` | `try_load_images(dir, [H, W])` |
| `fetch_dataset(name)` | `try_fetch_dataset(name)` |
| `apply_tokenizer(tok, text)` | `try_apply_tokenizer(tok, text)` |
| `train_bpe(corpus, vocab, seed)` | `try_train_bpe(corpus, vocab, seed)` |
| `parse_json(s)` (future) | `try_parse_json(s)` |
| `connect(url)` (future) | `try_connect(url)` |

The `try_` form catches the inner `EvalError`, converts it to a
`Value::Str` (via `format!("{e}")`), and wraps as
`Err("...")`. On success, it wraps the original return value as
`Ok(_)`.

Implementation cost: one shared helper in `mlpl-eval`,
something like

```rust
fn wrap_try(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    inner: &str,  // the non-try builtin name
) -> Result<Value, EvalError> {
    match eval_fncall_by_name(inner, args, env, trace) {
        Ok(v) => Ok(Value::Result { ok: true, payload: Box::new(v) }),
        Err(e) => Ok(Value::Result {
            ok: false,
            payload: Box::new(Value::Str(format!("{e}"))),
        }),
    }
}
```

Then each `try_xxx` dispatch in eval.rs's FnCall arm is one
line: `return wrap_try(name, args, env, trace, "xxx");`. No
behavior change for existing builtins; users opt in by typing
the `try_` prefix.

## Staged rollout

### Stage 1 -- Wire the pattern (one PR / one saga step)

Ship `wrap_try` helper + `try_load_preloaded` + 5 tests
(Ok-on-success, Err-on-typo, Err-on-unsupported-name,
chaining via `unwrap_or`, display Ok/Err). New lang-reference
row for `try_<op>` family.

This is small and self-contained -- pure additive change with
no risk to existing demos.

### Stage 2 -- Cover the common fallibility surface

Add `try_` siblings for the five busiest fallible ops:

- `try_load_preloaded`
- `try_load`
- `try_apply_tokenizer`
- `try_load_images` (gated on `image-io`)
- `try_fetch_dataset` (native-only, like the original)

Each one is ~5 lines + ~3 tests. Shippable as one saga step.

### Stage 3 -- One tutorial lesson uses it

New lesson in `apps/mlpl-web/src/lessons.rs`:

> "Bad input doesn't have to crash" -- the user types a typo
> into `try_load_preloaded`, sees `Err("unknown corpus: ...")`,
> uses `unwrap_or` to substitute a default, and proceeds.

This is the educational anchor for the type. Two or three
REPL turns; the gallery / training demos remain unchanged.

### Stage 4 -- One demo retrofitted as a defensive example

Pick the lightest one (`tiny_lm.mlpl` is the cleanest
candidate) and add a `try_` variant `demos/tiny_lm_defensive.mlpl`
that:

- Tries `try_load_preloaded` on a name list, falling back to
  the first one that succeeds
- Tries `try_apply_tokenizer` on a user-supplied prompt,
  defaulting to a stock prompt on failure

The non-defensive `tiny_lm.mlpl` stays in place. The defensive
version is a sibling demo that shows the pattern. It also gets
its own dropdown entry "Tiny LM (defensive load + tokenize)".

### Stage 5 -- New fallible builtins ship Result-native

After stages 1-4, the precedent is set:

- `:upload x` (Saga 29 step 013) returns `Ok(image)` /
  `Err("cancelled")` directly -- no `try_` prefix needed,
  because there is no non-Result baseline to preserve.
- Future `load_image(path)`, `fetch(url)`, `parse_csv(s)`,
  `parse_json(s)` all return Result by default.

The asymmetry is intentional: pre-existing builtins keep
their EvalError contract; new fallible builtins ship Result-
native. The `try_` prefix is the bridge.

## Tutoring messages

`unwrap()` already records the `Err` payload's display form
in `EvalError::UnwrapOnErr { message }`. That message is what
the user sees when an `unwrap` fails -- so the upstream
EvalError display string is what surfaces. The Tier-1 fallible
builtins (load_preloaded etc.) already produce readable error
messages, so `try_<op>` + `unwrap` will produce the same
tutoring quality as today's bare raises.

The one place where this needs attention is `try_load_images`
and `try_fetch_dataset` -- on the WASM build they raise
`EvalError::Unsupported("...")` with a long pointer at
`load_preloaded`. That message is fine to show as the `Err`
payload string.

## Out of scope (deliberately)

- A "try-block" macro / syntactic sugar (`try { ... }` is
  parser work, not value-type work).
- Auto-converting `EvalError` -> `Value::Result` at every
  builtin call site. The `try_` opt-in keeps backward
  compatibility.
- Multi-error chaining (`?`-style). Today's surface is
  `unwrap` + `unwrap_or` + manual `if is_ok(r) { ... } else
  { ... }`; that covers the demo use cases without
  introducing control-flow primitives.
- Touching `EvalError` itself -- the type stays exactly as
  it is; we just wrap it.

## Estimated saga steps

| Step | What | Effort |
|---|---|---|
| Stage 1 | `wrap_try` + first `try_load_preloaded` + tests + docs | small |
| Stage 2 | Four more `try_` siblings + tests | small-medium |
| Stage 3 | Tutorial lesson | small |
| Stage 4 | Defensive demo sibling for tiny_lm | small |
| Stage 5 | (no new step -- precedent set, executed inline by future fallible builtins) | n/a |

Five small steps, plus the precedent. Could be a Saga 29.5
follow-up after Saga 29 ships, or folded into Saga 30 if a
new fallible builtin is on the near horizon.

## Order-of-operations advice

If the user wants the educational payoff fastest, Stages 1
and 3 are enough -- one builtin + one lesson. Stages 2 and 4
add breadth but do not unlock new capabilities. Stage 5 just
documents the rule that future builtins follow.

Recommended sequence: 1 -> 3 (educational anchor) -> 5
(future builtins) -> 2 (breadth) -> 4 (defensive demo).
