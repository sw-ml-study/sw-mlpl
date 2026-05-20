Saga 29 step 001: load-images-and-pets-tiny.

Goal: ship a new `load_images(dir, [H, W])` builtin in
crates/mlpl-runtime (or wherever feels right per the existing
fixture-loading patterns) that decodes PNG / JPEG via image-rs,
resizes to [H, W], normalizes to f64 in [-1, 1], and returns a
DenseArray of shape [N, 3, H, W] with named axes [batch,
channel, y, x]. Native-only behind a Cargo feature flag so the
WASM target does not pull in image-rs; the WASM-side stub raises
a clean error pointing the user at load_preloaded.

In parallel, build the pets_tiny fixture: 100 cat + 100 dog
images from Oxford-IIIT Pet, resized to 64x64 (smaller than the
demo target to fit in the WASM bundle), normalized, serialized
as a single .bin blob shipped alongside tiny_shakespeare_snippet.
load_preloaded("pets_tiny") returns {X: [200, 3, 64, 64], Y:
[200], names: [str]}.

TDD: write tests first per the success criteria in
docs/milestone-vit.md:
- load_preloaded("pets_tiny") returns shape [200, 3, 64, 64]
  with [batch, channel, y, x] labels and 100 cats + 100 dogs.
- Decode a known PNG fixture and match a recorded byte hash
  (within fp tolerance).
- WASM build does NOT pull in image-rs (feature gate
  verified via cargo metadata or by attempting a WASM build
  in the gate).
- load_preloaded("pets_tiny") round-trip verifies one
  labelled cat and one labelled dog.

Quality gates: cargo test (main workspace), clippy -D warnings,
fmt, markdown-checker (if docs change), sw-checklist held or
lowered. Update contracts/runtime-contract/ for the new
builtin signature. Commit + push the source change before
agentrail complete.

Disk hygiene: PNG/JPEG decode via image-rs adds a sizeable dep
tree. Run cargo check -p mlpl-runtime first to gauge target/
growth before the full test gate. If target/ approaches 10 GB,
cargo clean and re-check.

Large-download discipline: pets_tiny fixture creation likely
needs the Oxford-IIIT Pet tarball (~750 MB). DO NOT download it
without explicit user permission -- the fixture can be
hand-built from a much smaller alternative source, or you can
ask the user for permission first. The recorded-PNG-hash test
should use a tiny fixture (one cat, one dog, ~10 KB each)
checked into the repo, not the upstream dataset.
