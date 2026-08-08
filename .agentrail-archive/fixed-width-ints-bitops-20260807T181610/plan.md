# Saga: fixed-width-ints-bitops
demo-memory request #2 (docs/demo-memory-upstream.md): fixed-
width unsigned integer bit operations -- masks, shifts,
popcount, conversions -- unlocking Swiss control bytes,
compact Bloom filters, Hamming indexes, binary sparse
retrieval. PURE array ops -> mlpl-runtime-array (universal:
browser + native + connect), unlike the native-only clock.
No new value kind: integers are f64 values in the exact range;
ops validate non-negative-integer domain and use widths for
fixed-width wrap.
## Steps
1. bit-design -- docs/bit-ops-design.md: domain (0..2^53 exact,
   width cap), LSB-first bit order, element-wise + scalar
   broadcast, width-aware shl, error policy. Op set.
2. bit-logic -- band/bor/bxor/bnot/popcount in bit_logic.rs;
   dispatch + NAMES; catalog + lang-ref + glossary; TDD.
3. bit-shift-views -- shl(x,k,width)/shr(x,k)/bmask(x,width)/
   bits(x,width)/from_bits(v) in bit_views.rs; Hamming =
   popcount(bxor); TDD; docs.
4. close -- upstream doc status, wiki, pages rebuild+deploy
   (pure builtins reach the browser), --done.
