Saga 23 step 001: tag-enum-side-table.

Goal: ship the ValueTag enum and Environment::tags side table as the foundation
for every later step in this saga. Zero behavior change in any existing demo.

TDD (Red/Green/Refactor):

1. RED: write failing tests in crates/mlpl-core/tests/value_tag_tests.rs for:
   - ValueTag variants construct cleanly (Logit, Probability, LogProbability,
     Loss { kind: LossKind }, Gradient { wrt: String },
     Weight { layer: String, name: String }, Bias { layer: String },
     Activation { layer: String, kind: ActivationKind }, LearningRate,
     Labels { num_classes: usize }, AttentionMap)
   - ValueTag implements Clone + Debug + PartialEq + Serialize + Deserialize
     (serde derives so trace JSON can carry tags later in step 006)
   - ValueTag::display_name() returns a stable short string for each variant
     ("Logit", "Probability", etc.) used by :describe and trace output

   Plus failing tests in crates/mlpl-eval/tests/environment_tags_tests.rs for:
   - Environment::set_tag(name, tag) attaches a tag to a binding
   - Environment::get_tag(name) returns Option<ValueTag>
   - re-binding via set_tag overwrites the previous tag (per-name semantics)
   - clear_tag(name) removes a tag
   - tags() returns an iterator over (name, tag) pairs for :tags listing

2. GREEN: implement
   - new file crates/mlpl-core/src/value_tag.rs with the ValueTag enum + the
     LossKind / ActivationKind helper enums it references
   - register the module in crates/mlpl-core/src/lib.rs
   - new HashMap<String, ValueTag> field on Environment in
     crates/mlpl-eval/src/environment.rs (or wherever Environment is defined)
   - set_tag / get_tag / clear_tag / tags helpers on the Environment impl

3. REFACTOR: keep modules under the sw-checklist 7-fn budget. mlpl-eval is at
   the module/fn limit per CLAUDE.md, so the helpers go on the existing
   Environment impl as small methods, not in a new module.

Constraints:
- ValueTag lives in mlpl-core (not mlpl-eval) because step 006 will surface it
  in trace JSON which is downstream of mlpl-core, and because mlpl-eval is at
  its module budget.
- Variant metadata follows docs/optional-typing-design.md "Tier A vocabulary"
  table exactly. If a variant in the doc lacks structured metadata (e.g.
  AttentionMap), it stays a unit variant.
- Tags are per-binding-name; set_tag on a name that already has a tag overwrites.
- This step adds NO language surface (no :tag command, no annotation parsing,
  no auto-tagging from producers). Those land in steps 002-005.
- Every existing demo must still run unchanged. Run cargo test workspace-wide
  before committing.

Quality gates per /mw-cp (CLAUDE.md):
- cargo test (full workspace)
- cargo clippy --all-targets --all-features -- -D warnings (zero warnings)
- cargo fmt --all && cargo fmt --all -- --check
- markdown-checker -f "**/*.md" only if docs touched (probably not this step)
- sw-checklist (baseline 102/139 must hold; not regress)

Commit before agentrail complete. Push after commit.