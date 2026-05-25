# Feature Design Process

Every new feature is an opportunity to produce better-
architected code. Begin with the end in mind: design the
module structure BEFORE writing code, so the result meets
all budgets by construction -- not by post-hoc trimming.

## The problem this solves

The anti-pattern: write code, run sw-checklist, discover
violations, scramble to trim lines or retire warnings
elsewhere. This produces code that barely fits, accumulates
structural debt, and wastes time on mechanical adjustments
that should never have been necessary.

## Process (follow in order)

### 1. Budget inventory (before writing any code)

Check the current state of every module and crate you plan
to touch:

```bash
sw-checklist -v 2>&1 | grep "module_name\|crate_name"
```

Record the current fn count, line count, and module count
for each target. These are your starting constraints.

### 2. Design the decomposition (on paper, not in code)

For each new concern, answer:

- **What is the single responsibility?** Name it in 3 words.
  If you can't, it's two concerns -- split them.
- **How many functions will it need?** Count them. If >4,
  you'll hit the module-fn-count warning. Plan sub-modules.
- **How many lines?** Estimate. If >25 per function, plan
  helper extraction. If >350 per file, plan file splits.
- **Where does it go?** Which crate, which module? Will
  adding it push an existing module over budget? If yes,
  refactor the existing module FIRST.
- **What struct fields does it need?** If >5, compose from
  sub-structs. Each sub-struct should represent one concept.

### 3. Design for the budgets (not against them)

Warnings and FAILs are both problems to fix -- different
severity, same obligation. Never trade a FAIL retirement
for warning growth. Fix warnings; FAILs shrink naturally.

| Metric | Design target | Warning (=problem) | FAIL (=worse problem) |
|--------|--------------|--------------------|-----------------------|
| Function LOC | <=20 | >25 | >50 |
| Module fn count | <=4 | >4 | >7 |
| File LOC | <=250 | >350 | >500 |
| Crate module count | <=4 | >4 | >7 |
| Struct fields | 5 +/- 2 | n/a | n/a |

**Every commit must reduce the total problem count
(warnings + FAILs).** If a feature adds warnings, the
design is wrong -- redesign before committing.

Practical implications:

- A new module with 3 functions + 2 tests = 5. That's at
  the warning line. Either put tests in a separate test
  file, or design for <=2 public functions + 1 private
  helper = 3 total.
- A Yew function_component with its Props struct, the
  component fn, and 2 render helpers = 4 items. That's at
  the warning line. Keep the component to 1 fn + 1 Props
  struct (2 items); extract helpers to a sibling.
- A callback-builder module with 6 builders is over budget
  by design. Split by concern: 3 callbacks per module.

### 4. Refactor first, then add

If the target module is already at or near budget:

1. Refactor it to create headroom FIRST.
2. Commit the refactor.
3. THEN add the new feature into the newly-created space.

Never add to a full module and then try to fix it after.
The refactor-first approach means the feature commit is
clean and the budget is met by construction.

### 5. Verify BEFORE committing

Run sw-checklist BEFORE `git add`. If it's worse than the
baseline, your design was wrong -- go back to step 2 and
redesign, don't patch.

## Anti-patterns to avoid

- **Add then trim.** Writing code that violates budgets
  and then micro-trimming other functions to compensate.
  This is treating symptoms, not causes.
- **Compress to fit.** Removing whitespace, merging lines,
  or using terse variable names to squeeze under a line
  count. The budget assumes idiomatic formatting.
- **Exception creep.** Documenting "structural exception"
  for every new module that hits a warning. If every
  module is an exception, the architecture is wrong.
- **Facade bloat.** Creating facade modules that re-export
  from many sub-modules. Facades should have 0 functions;
  they're `pub use` only.
- **God structs.** Structs with 10+ fields that mix
  unrelated concerns. Compose from sub-structs of 3-5
  fields each.

## General vs domain-specific separation

Every module, crate, and component should be clearly either
general-purpose (UI framework, localStorage, state machine,
utility) or domain-specific (ML algorithms, MLPL syntax,
demo content, tutorial lessons).

- **General-purpose code** lives in crates/modules named
  for what they DO, not what they serve. Example:
  `onboarding_storage` (localStorage access) is general.
  It should not contain MLPL-specific predicates.
- **Domain-specific code** lives in crates/modules named
  for the domain concept. Example: `onboarding_splash`
  (MLPL welcome overlay) is domain-specific. It knows about
  demo indices, tutorial lessons, and learning paths.
- **Functional helpers and design-pattern utilities** are
  general-purpose: dispatchers, state machines, observers,
  builder patterns, iterator adapters, validation
  pipelines. These work abstractly over any domain and
  should live in domain-independent modules.
- **The test:** could this module be extracted to a
  separate repo and used by a non-MLPL project? If yes,
  it's general. If no, it's domain-specific.
- **Physical design reflects the distinction.** General
  modules and crates should have no `use crate::` imports
  from domain-specific modules. Domain-specific modules
  depend on general ones, not the reverse.

This separation enables future extraction of general-
purpose crates into supporting repos for reuse.

## How to count (sw-checklist rules)

sw-checklist counts ALL `fn` items in a module, including:
- `#[test]` functions
- Private helpers
- Trait impls

Plan accordingly. If a module needs 4 public functions,
it can afford 0 tests inline (move tests to a test file
or a `#[cfg(test)]` sibling). If it needs 3 public
functions, it can afford 1 inline test.
