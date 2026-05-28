# components/runtime/ migration (saga 55)

Move the runtime family (11 crates) into a new
`components/runtime/` workspace. Most crates are already sparse
from saga 50's earlier extractions; the big work here is the
bulk move + path updates.

## Crates to move

From crates/ to components/runtime/crates/:
- mlpl-runtime-core
- mlpl-runtime-math
- mlpl-runtime-conv
- mlpl-runtime-rnn
- mlpl-runtime-array
- mlpl-runtime-ml
- mlpl-runtime-data
- mlpl-runtime-dim-reduction
- mlpl-runtime-umap
- mlpl-runtime-mds-rp
- mlpl-runtime

## Step plan

1. **scaffold**: create components/runtime/ workspace.
2. **bulk-move**: git mv all 11 crates, update workspace members
   (remove from root, add to runtime), update inter-crate paths
   inside the family (now siblings = `../mlpl-runtime-X`) and
   external consumer paths (now point into components/runtime/).
3. **review-for-splits**: identify any remaining over-budget crates
   (e.g. mlpl-runtime-dim-reduction at 7 modules) and decompose
   if it would retire FAILs/WARNs.
4. **close**: sw-checklist delta + language-status update.
