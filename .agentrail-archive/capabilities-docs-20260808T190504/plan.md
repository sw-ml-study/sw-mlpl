# Saga: capabilities-docs

Give the repo a clear, honest statement of capabilities and
current status that distinguishes production-usable/built
features from proof-of-concept / partial ones (user direction).
Main surfaces: README (a maturity section, present-tense per the
docs policy -- no saga names/dates/plans) and docs/status.md (a
current "what exists today" maturity view). Fix stale/overselling
docs the maturity survey flagged.

Survey verdicts: BUILT core (interpreter/parser/arrays/autograd/
Model DSL/optimizers/tokenizers/tiny-LM/typed-values/serialization
+fs/viz/REPLs/web); PARTIAL (MLX -- production at scale, overhead
below crossover; connect/server MVP; LLM native-only); POC (CUDA
single-GPU in-process LoRA vertical slice; compile-to-Rust
numerical-subset only).

## Steps
1. readme-status -- README maturity section + fix the compile
   bullet overselling + drop the banned version stamp;
   docs/status.md current "what exists today" section.
2. doc-caveats -- caveat/refresh the stale docs (using-mlx slow
   numbers, using-cuda PoC, serialization-variant-encoding now
   shipped, compile milestone subset).
3. close -- markdown gate, readme_counts, --done (no pages deploy
   -- README/docs are not web sources).
