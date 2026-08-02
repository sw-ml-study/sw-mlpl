# Saga: thinking-in-arrays-doc

User direction 2026-08-02: (1) write docs/thinking-in-arrays.md
-- the essay version of the data-loop vs time-loop discussion --
and link it from the "Thinking in Arrays" playground demo;
(2) the user dislikes the name `cumprod` (negative connotations)
and asked for alternatives -- a rename/deprecation step follows
once they pick a name (candidates presented: running_product
[recommended], prefix_product, product_scan,
cumulative_product). Sweep size if renamed: ~13 repo files
(reduce.rs dispatch + NAMES, demos.toml, demos/diffusion_2d.mlpl,
lang-reference, glossary, apl2 docs, tests, CHANGES historical
mentions stay). Follow the iota precedent: new canonical name,
cumprod deprecated to works-but-undocumented, examples swept,
literate parity check.

## Steps

1. write-doc -- docs/thinking-in-arrays.md (data vs time loops,
   associative borderline/scan, ML mapping, APL history +
   parallel-scan note); link from the demo takeaway; pages
   rebuild + deploy; wiki Resources page pointer.
2. cumprod-rename -- BLOCKED on user naming decision; then the
   iota-style deprecation sweep.
