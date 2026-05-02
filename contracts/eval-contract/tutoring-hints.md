# Tutoring Hints Catalog (Saga 23 step 008)

## Purpose

The educational keystone of Saga 23. Every type-related error
raises `EvalError::TypeMismatch { op, expected, actual, hint }`
where `hint` is a 3-5 line tutor message that names the most
likely cause and one or two concrete fixes in copy-pasteable
MLPL. A student who trips a type predicate should treat the
error as a learning moment, not an obstacle.

This doc catalogs every hint shipped through Saga 23, locks
the canonical fix syntax, and documents the four-part shape
each hint follows. Tests in
`crates/mlpl-eval/tests/tutoring_hints_tests.rs` assert
specific phrasings so hints cannot drift away from valid MLPL
syntax.

## The four-part shape

Every hint follows the same structure:

1. **What the op expected and why.** Open with the expected
   tag and the reason the consumer needs it. "expects Logit
   (raw scores) because cross_entropy assumes an unnormalized
   score per class".
2. **What it got.** Repeat the actual tag in prose so the
   user can scan the hint without re-reading the
   `expected: ... actual: ...` line above it.
3. **Likely cause.** Name the common bug pattern that
   produces this combination ("you applied softmax already",
   "you forgot the loss function", "arg order confusion").
4. **One or two concrete fixes.** Copy-pasteable MLPL syntax
   that resolves the error. Use real builtins; reference
   future-only ops with a saga number ("once Saga 24 ships
   nll").

## Canonical fix vocabulary

The hints reference only this MLPL surface:

- `softmax(x, axis)` -- bridge Logit -> Probability over a
  given axis. Always pass an axis; there is no
  `softmax(x)` form.
- `log_softmax` -- not yet a language-surface builtin; do
  not reference as a fix.
- `cross_entropy(logits, y)` -- canonical loss producer.
  `logits` shape `[N, V]` or `[B, T, V]`; `y` shape `[N]` or
  `[B, T]` integer-valued.
- `mse(predictions, target)` -- referenced as a future fix
  pattern for non-classification regression. Not yet a
  builtin.
- `kl_divergence(p, q)` -- referenced as a future fix.
  Saga 24 ships it.
- `nll(probs, y)` and `nll(log_probs, y)` -- referenced as
  future fixes for log-prob inputs. Saga 24 ships nll.
- `adam(loss, params, lr, b1, b2, eps)` -- the canonical
  optimizer step. The full arg list is shown in fixes
  because users often forget the `b1 / b2 / eps` triple.
- `exp(log_probs)` -- Probability recovery from
  LogProbability for viz / sampling.

## Hint table

Each row: `(op, got_tag) -> hint_constant`. The constant lives
in `crates/mlpl-eval/src/type_errors.rs` (consumer hints) or
`crates/mlpl-eval/src/tag_propagate.rs` (propagation hints).

### Consumer-side hints (type_errors.rs)

| op             | got tag         | hint constant                      |
|----------------|-----------------|------------------------------------|
| cross_entropy  | Probability     | `HINT_LOGIT_GOT_PROBABILITY`       |
| cross_entropy  | LogProbability  | `HINT_LOGIT_GOT_LOG_PROBABILITY`   |
| cross_entropy  | Labels          | `HINT_LOGIT_GOT_LABELS`            |
| sample         | Probability     | `HINT_LOGIT_GOT_PROBABILITY`       |
| sample         | LogProbability  | `HINT_LOGIT_GOT_LOG_PROBABILITY`   |
| top_k          | Probability     | `HINT_LOGIT_GOT_PROBABILITY`       |
| top_k          | LogProbability  | `HINT_LOGIT_GOT_LOG_PROBABILITY`   |
| adam           | Probability     | `HINT_LOSS_GOT_PROBABILITY`        |
| adam           | Logit           | `HINT_LOSS_GOT_LOGIT`              |
| adam           | (anything else) | `HINT_LOSS_GOT_OTHER`              |
| momentum_sgd   | Probability     | `HINT_LOSS_GOT_PROBABILITY`        |
| momentum_sgd   | Logit           | `HINT_LOSS_GOT_LOGIT`              |
| momentum_sgd   | (anything else) | `HINT_LOSS_GOT_OTHER`              |

### Propagation-side hints (tag_propagate.rs)

| context             | tags involved             | hint constant                |
|---------------------|---------------------------|------------------------------|
| binop (+, -, *, /)  | mismatched domains        | `HINT_DOMAIN_MISMATCH`       |

## Notes on specific hints

### `HINT_LOGIT_GOT_LABELS`

cross_entropy-only. The other Logit consumers (sample, top_k)
do not produce this hint because passing labels to those is a
nonsensical workflow rather than a common bug. cross_entropy
is the only op where users routinely confuse arg order
(predicting vs labeling).

### `HINT_LOSS_GOT_OTHER`

A catch-all for the "not Probability and not Logit" leftover
case (a Labels-tagged scalar, an AttentionMap, a
LearningRate-tagged scalar, etc.). The hint stays generic on
purpose: enumerating every case would not improve the user's
experience -- they're already in unusual territory and the
generic "produce a Loss with cross_entropy / mse /
kl_divergence" advice covers all of them.

### `HINT_DOMAIN_MISMATCH`

The propagation-side hint covers any tag combination that
would conflate domains across an arithmetic op, including
Logit + Probability, Loss + Probability, and Loss + Logit.
The hint is uniform because the fix vocabulary -- softmax /
log / cross_entropy / mse -- is the same in every case; only
the choice depends on which side the user wants to convert.

## Future hints (deferred)

When the named builtins ship, this catalog grows:

- `nll` (Saga 24): producer + the corresponding consumer
  hints adjust to mention "use nll(probs, y) directly" rather
  than "once Saga 24 ships nll".
- `kl_divergence` (Saga 24): consumer rules for Distribution
  inputs.
- `mse` (Saga 24 or follow-up): currently referenced as a
  fix pattern; once it lands as a builtin, the hint catalog
  grows entries for `(mse, ...)` predicates of its own.
- Distribution tags (Saga 24): new consumer rules for
  `sample(distribution, seed)` etc.
- LayerRole tags (Saga 27): new consumer rules for layer
  composition with mismatched input / output kinds.
- User-defined tags (Saga 28): user-defined hints attached
  to user-defined tag definitions; predicate failures show
  the user's own hint string rather than a curated one.

## Out of scope

- Per-call-site hint rewriting based on local context (e.g.
  "you defined `probs = softmax(L, 1)` on line 3"). The
  hint catalog stays static for now; richer call-site context
  is a follow-up step.
- Localization. All hints are English. Future
  internationalization can wrap the catalog in a translation
  layer without changing call sites.
- Color / formatting in terminal output. Hints are plain
  text; the REPL renders them however it likes. The web REPL
  trace panel may bold key terms in a future polish step.
