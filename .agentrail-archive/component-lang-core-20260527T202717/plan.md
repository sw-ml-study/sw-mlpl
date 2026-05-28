# components/lang-core/ migration (saga 52)

First component-migration saga. Move the three foundational crates
(mlpl-core, mlpl-array, mlpl-eval-core) into a new
`components/lang-core/` nested workspace. Bottom of the dep stack,
so every other crate's path references must be updated.

## Why these three together

- mlpl-core: spans, identifiers, base types -- depended on by ~15
- mlpl-array: DenseArray, Shape -- depended on by ~30
- mlpl-eval-core: shared eval types -- depended on by ~7

All Layer 0-1, no intra-workspace deps among them except
mlpl-array -> mlpl-core. Move them together so referrers update
once, not three times.

## Step plan

1. **scaffold-component**: create `components/lang-core/Cargo.toml`
   (workspace manifest), `components/lang-core/crates/` empty
   directory. Verify the empty component workspace builds.
2. **move-mlpl-core**: `git mv crates/mlpl-core
   components/lang-core/crates/mlpl-core`. Update root Cargo.toml
   members. Update every Cargo.toml that has
   `mlpl-core = { path = "../mlpl-core" }` to point to the new
   location. Verify full workspace build.
3. **move-mlpl-array**: `git mv` then update all referrers (the
   biggest batch -- ~30 crates).
4. **move-mlpl-eval-core**: `git mv` then update all referrers
   (~7 crates).
5. **close**: language-status update, --done.

Each step ends with a green `cargo check --workspace` and pushed
commits, so no step leaves the tree broken.

## Path-reference rules

Crates currently in `crates/<foo>/` referring to a moved crate:
`mlpl-core = { path = "../../components/lang-core/crates/mlpl-core" }`

Crates in `components/<other>/crates/<foo>/` referring to a moved
crate: `mlpl-core = { path = "../../../lang-core/crates/mlpl-core" }`

Crates in `apps/<foo>/` referring to a moved crate:
`mlpl-core = { path = "../../components/lang-core/crates/mlpl-core" }`

Crates in `services/<foo>/` referring to a moved crate:
`mlpl-core = { path = "../../components/lang-core/crates/mlpl-core" }`

## Shared target

Already in place from saga 51. The new component workspace inherits
.cargo/config.toml automatically and writes to the same target/.
