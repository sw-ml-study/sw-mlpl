# components/autograd/ migration (saga 56)

Move mlpl-autograd + mlpl-trace into a new components/autograd/
workspace. Both observe runtime execution; trace consumes
autograd-shaped data.

## Crowded crates to check

- mlpl-autograd: 7 modules at limit + reduction_ops 11 fns FAIL +
  backward 11 fns FAIL + propagate 77 LOC FAIL. Heavy splitting
  needed.
- mlpl-trace: small (4 modules).

## Step plan

1. scaffold + move both crates.
2. split-autograd: decompose reduction_ops and backward modules
   (each has 11 fns, FAIL). Probably 2-3 sibling crates inside the
   component (e.g. autograd-forward, autograd-backward,
   autograd-reduction).
3. close.
