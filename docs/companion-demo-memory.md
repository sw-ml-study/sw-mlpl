# Companion repo: demo-memory

Planning doc (forward-looking). Overview: `companion-repos.md`.
Source brief: `docs/sw-mlpl-demo-memory.txt`.

## Purpose

A collection of memory-organization demos that bridges
classical computer science to modern machine learning. The
same questions recur at both ends -- how do I find the right
item quickly, how much metadata does it cost, how does cache
locality and pointer/index compression affect throughput, how
full can a structure get before performance collapses -- and
demo-memory answers them with runnable MLPL, starting from a
Robin Hood hash table and arriving at KV-cache acceleration
and sparse-attention indexing, all in one language.

This framing is why the brief recommends prioritizing a
`demo-memory/` collection over a generic `demo-systems/`: it
makes classical data structures and ML memory (engrams, KV
caches, sparse attention) legible as the SAME subject.

## Proposed shape

```text
demo-memory/
    hash/         # Robin Hood, Swiss, cuckoo, tiny-pointer, iceberg
    cache/        # cache-aware layouts, locality experiments
    bloom/        # Bloom + counting filters
    lru/          # eviction policies
    kv/           # key-value stores; the KV-cache bridge
    attention/    # sparse-attention indexing (BinaryPC-style)
    retrieval/    # nearest-neighbor / demonstration selection
```

## The distinctive idea: demos that measure themselves

Per the brief, each demo reports its own operating
characteristics -- inserts/sec, lookups/sec, memory, bytes per
key, load factor, collisions, probe lengths, latency
distribution -- so the collection is a BENCHMARK SUITE, not
sample code. A reader can put Swiss vs Robin Hood vs
tiny-pointer vs iceberg on the same workload and see the
tradeoffs. The "implementation as a parameter" idea (choosing
a hashing strategy or load factor by name) is a natural demo
spine, and could later suggest a core feature.

## Why this is possible now, and what it may pull upstream

Expressible today: hashing, probing, and accounting are array
and record operations; `run_script` + the fs API let the suite
run and compare cases. The brief flags language conveniences
these algorithms want, which would be upstream requests to
`sw-mlpl` if the demos prove them out:

- fixed-width integer / bit views (`u8`/`u12`/`u14`, `bits16`)
  where every byte matters;
- cache-aware / packed array layouts;
- explicit first-class randomness (`seed` / `rng` / `sample` /
  `shuffle`) rather than ad-hoc seeding;
- handle/offset/index objects instead of pointer arithmetic
  (which stays out, by philosophy).

Each is a "prove it in a demo first, then request the minimal
core surface" candidate -- the mlplunit model applied to
data-structure research.

## Relationship to sw-mlpl

Deeply synergistic with the ML roadmap: the engram memory,
KV-cache (`gen_state` / `gen_logits` / `gen_append`), and any
future sparse-attention work are memory-organization problems,
and demo-memory is where their classical foundations get
taught. It starts as a pure consumer; its upstream requests
(fixed-width ints, packed layouts, first-class randomness) are
tracked here until a demo makes the case.
