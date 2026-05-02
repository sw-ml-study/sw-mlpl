# Tag Propagation Contract (Saga 23 step 005)

## Purpose

`ValueTag` propagation rules for arithmetic, transpose,
reshape, reductions, unary negation, and identifier aliases.
The runtime applies these rules at the assignment site so that
a tagged input flows through derived expressions to the bound
result without the user re-stating the tag.

The producer rules from Saga 23 steps 002 and 003 take
precedence: `softmax`, `cross_entropy`, `linear`, `apply` and
the rest of the curated producers attach the canonical tag for
their op family. Propagation only fires when the right-hand
side is not a recognized producer and at least one operand
carries a tag in the side table.

## Where it runs

The `Expr::Assign { name, value }` evaluator first calls
`auto_tag::for_assign(value, env)` (the producer dispatcher).
On `None`, it calls `tag_propagate::propagate(value, env)`.
The propagated tag (if any) is attached to `name` via
`Environment::set_tag`. Domain-mixing arithmetic raises
`EvalError::TypeMismatch` and the assignment fails.

Only top-level binop / fncall / unary / ident expressions on
the rhs are inspected. A binop nested inside a builtin call
does not raise this error in step 005 -- the `infer` walk
recursively resolves the tag but swallows propagation errors.
A future step can lift the check earlier in evaluation if the
deferred case proves load-bearing.

## Rules

### Identifier alias

```
y = x   ; y inherits x's tag (or no tag)
```

A bare-identifier rhs copies the side-table entry verbatim.

### Unary negation

```
y = -x  ; y inherits x's tag
```

Sign flip does not change the value's domain. Logit stays
Logit; Loss stays Loss; Probability stays Probability (a
negated probability has odd semantics, but the language layer
does not police that here -- the user can clear with a
follow-up `:untag`).

### Binary arithmetic (+, -, *, /)

| lhs tag       | rhs tag       | result                                  |
|---------------|---------------|-----------------------------------------|
| `None`        | `None`        | `None`                                  |
| `T`           | `None`        | `T`                                     |
| `None`        | `T`           | `T`                                     |
| `Logit`       | `Logit`       | `Logit`                                 |
| `Loss{ka}`    | `Loss{kb}`    | `Loss{ka}` (lhs kind wins; see note)    |
| `T`           | `T` (same)    | `T`                                     |
| anything else (different families) | `TypeMismatch` |             |

Note: combining two losses keeps the lhs's `LossKind`.
A future step may upgrade differing kinds (CrossEntropy + Mse)
to `LossKind::Custom`; for now the lhs kind is preserved.

`TypeMismatch` carries the hint:

> operands live in different typed-value domains. fix:
> convert one side first -- softmax(logits, axis) bridges
> Logit -> Probability, log lifts Probability ->
> LogProbability, and cross_entropy / mse / kl_divergence
> bridge predictions to Loss.

### transpose / reshape_labeled / label / relabel

Preserve the tag. Axis permutation and explicit re-labeling
keep semantic identity.

### reshape

Clears the tag. Shape reflow loses semantic identity --
a reshape from `[batch, vocab]` to `[batch * vocab]` no
longer represents per-row probabilities or per-token logits.

### Reductions: mean / reduce_add / reduce_mul / argmax

| input tag      | output tag    |
|----------------|---------------|
| `Loss{kind}`   | `Loss{kind}`  |
| `Probability`  | `None`        |
| `Logit`        | `None`        |
| anything else  | `None`        |

Loss survives because a partial sum / mean of losses is still
a loss (training-loop convention). Probability does not
survive because a partial sum of probabilities is not a
probability. Logit does not survive because the reduced axis
loses the per-row semantics (the per-position score becomes a
batch-summary scalar with no `Logit` meaning).

### Function calls (other than recognized producers)

If `auto_tag::for_assign` did not match a producer rule, the
function is unknown to the typing layer. Propagation does not
fire and the result is untagged.

Exceptions: the structural ops above (`transpose`, `reshape`,
`reshape_labeled`, `label`, `relabel`) and reductions (`mean`,
`reduce_add`, `reduce_mul`, `argmax`) are special-cased
because they only manipulate shape, not domain.

## Gradual-typing additivity

Untagged values pass every propagation rule and never raise
`TypeMismatch`. A user who has not adopted typed values keeps
the existing untyped semantics with no surprises.

A binding that previously had a tag and is re-assigned to an
untagged-producing expression keeps the OLD tag (the
side-table entry follows the binding name and is not
auto-cleared by re-assignment to a non-producer rhs). To
clear deliberately, use `:untag <name>` (Saga 26).

## Examples

```
W = param[2, 3]                         ; Weight (auto-tagged via Saga 23 step 003)
X = randn(0, [4, 2])
Y = matmul(X, W)                        ; untagged (matmul not in producer set)
loss = mse(Y, target)                   ; ... mse not yet a builtin, hypothetical
loss_total = loss + reg                 ; Loss + None -> Loss (reg untagged)
loss_n = mean(loss_per_sample)          ; Loss survives reduction
y = -y                                  ; tag preserved
T = transpose(W)                        ; Weight survives transpose (in step 003 W is auto-tagged)
F = reshape(W, [6])                     ; Weight cleared (shape reflow)
mix = logits + probs                    ; TypeMismatch with hint
```

## Out of scope

- Inline binop checking inside builtin args: `cross_entropy(L
  + P, Y)` does not raise the `Logit + Probability` mismatch
  in step 005. The predicate consumer in `cross_entropy` sees
  the binop as untagged (the `infer` walk swallows the
  propagation error). A future step can lift the check.
- Differing `LossKind` upgrade to `Custom`. Currently the lhs
  kind wins; a follow-up step can refine.
- Tag propagation through `apply` / `chain` / `residual`
  composition (these are model-level concerns handled in
  Saga 23 step 003 and Saga 27).
- Custom user-defined tag propagation. User tags from Saga 28
  do not have arithmetic propagation rules.
