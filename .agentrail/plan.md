# Saga E2: engram-dsl (the engram(...) model layer + training)

Per docs/engram-sagas-plan.md (E2) and decisions D3/D4. The
primitive layer (E1: ngram_hash/gather_rows + engram-core) exists;
this saga makes Engram a trainable Model DSL citizen with the
"concat" gate, culminating in the Learnable Phrase Memory demo.

Steps (draft):
1. modelspec-engram -- ModelSpec::Engram variant (mlpl-eval-core,
   serde-compatible) + the engram(hidden, ngrams, heads, slots,
   head_dim, seed) constructor builtin (mlpl-eval-models family):
   creates the flattened memory table + value projection + concat
   gate params in the env (near-identity init: small W_v, negative
   gate bias), :describe shows the accounting.
2. apply-engram-forward -- apply_engram(e, h, ids) forward builtin
   (hash -> gather -> flatten -> project -> concat gate ->
   residual), inference-path tests incl. near-identity at init.
3. engram-grad -- tape differentiation: gather_rows scatter-ADD
   gradient (duplicate indices accumulate), apply_engram lowering
   in grad_calls so train/adam updates the memory table; gradcheck
   + only-addressed-rows-move tests.
4. phrase-memory-demo -- Learnable Phrase Memory demo (train the
   table to complete known bigrams/trigrams, show before/after +
   that unaddressed rows stay zero), docs + pins, deploy.
