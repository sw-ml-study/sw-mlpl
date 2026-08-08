# Fixed-width integer bit operations: design

Status: committed design (saga fixed-width-ints-bitops, step
001). Source: demo-memory request #2
(`docs/demo-memory-upstream.md`). These unlock Swiss-table
control bytes, compact Bloom filters, Hamming-distance
indexes, and binary sparse retrieval.

## No new value kind

MLPL values are `f64` dense arrays. Bit operations do not add
an integer type; they operate on `f64` values that hold exact
non-negative integers, pervading arrays like every other
element-wise op. This keeps the numeric core numeric (the
callable design's principle) -- a "byte" is just an `f64` that
happens to be in `0..256`.

## Integer domain

- **Non-negative integers only**, exact: `0 .. 2^53`. `f64`
  represents every integer up to `2^53` exactly, which covers
  `u8` / `u16` / `u32` and well beyond -- the widths these
  algorithms need.
- Any operand that is negative, non-integer, NaN, infinite, or
  `>= 2^53` is a LOUD error naming the builtin and the
  offending value. No silent truncation of the INPUT.
- **Width parameters** are integers `1..=53`. A width outside
  that range errors.

## Bit order: LSB-first

Bit `i` has value `2^i`. `bits(x, width)` returns a `[width]`
vector with index `i` holding bit `i` (least significant
first), and `from_bits(v)` is its exact inverse
(`sum(v[i] * 2^i)`). LSB-first is the composable arithmetic
convention; it is documented so downstream code (Hamming,
Bloom) can rely on it.

## Element-wise + scalar broadcast

`band` / `bor` / `bxor` are element-wise over equal-shaped
arrays, with scalar broadcast (a scalar applies against every
element), matching MLPL's pervasion. Width/shift parameters
are scalars.

## The op set

| Builtin | Meaning |
|---|---|
| `band(a, b)` | bitwise AND (element-wise, broadcast) |
| `bor(a, b)` | bitwise OR |
| `bxor(a, b)` | bitwise XOR |
| `bnot(x, width)` | complement within `width` bits: `~x & mask(width)` |
| `popcount(x)` | number of set bits in each element |
| `shl(x, k, width)` | fixed-width left shift: `(x << k) & mask(width)` |
| `shr(x, k)` | logical right shift by `k` (never grows) |
| `bmask(x, width)` | keep the low `width` bits (explicit truncation / width conversion) |
| `bits(x, width)` | expand a scalar `x` to a `[width]` 0/1 vector, LSB-first |
| `from_bits(v)` | pack a 0/1 vector back to an integer scalar |

### Why `shl` is width-aware

A plain `x << k` can exceed `2^53` and lose `f64` exactness.
Making `shl(x, k, width)` mask to `width` in the same step
keeps every result exact (for `width <= 53`) and matches the
fixed-width intent -- a `u8` shift stays a `u8`. `shr` cannot
grow a value, so it needs no width.

### Worked idioms

```text
hamming = popcount(bxor(a, b))          # distance between codes
byte    = bor(shl(hi, 4, 8), lo)        # pack two nibbles into a u8
isset   = band(shr(flags, i), 1)        # test bit i
code    = from_bits(mask_vector)        # bit vector -> integer key
```

## Placement

These are PURE (no state, no platform dependency), so they
live in `mlpl-runtime-array` and are UNIVERSAL -- browser,
native, and connect -- unlike the native-only `clock_ms`.
Split across two modules (logic vs shift/views) to respect the
per-module function-count budget.

## Not in scope

No signed integers, no `u64`-exact arithmetic (beyond `2^53`),
no packed/bit-level STORAGE (that is demo-memory request #3,
`packed-layouts`; these ops give the bit SEMANTICS, not a
denser layout). `bmask` is the width "conversion"; there is no
separate typed-width value to convert between.
