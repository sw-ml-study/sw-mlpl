Saga 21.5 step 013: release-v020.

Goal: ship v0.20.0. Bump every Cargo.toml's package version
from 0.19.0 to 0.20.0 (workspace root + every crate / app /
service that participates in the workspace). Update CHANGELOG.md
with a v0.20.0 section summarizing Saga 21.5's shipped scope
(SSE streaming, cancellation, viz storage, persistence,
reattach, web connect mode, f32/u8 wire). Refresh CHANGES.md
via scripts/gen-changes.sh after the version-bump commit lands.
Tag v0.20.0 and push the tag. Add --done to agentrail complete.