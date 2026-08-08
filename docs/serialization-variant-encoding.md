# Serialization: variant / tagged-value encoding -- design note

Status: design decision (2026-08-08), FORWARD-LOOKING (planning
doc). No code yet -- it records the direction the JSON/TOML
codecs should take when lossless variant serialization is built.
Companions: `docs/option-result-design.md` (the VALUE model --
Result is a tagged sum, Option is zilde), and the "typed native
formats" serialization item tracked for demo-algorithms
(`docs/q-and-a.md`).

## The problem

JSON and TOML have no variant/sum type. sw-MLPL's `Result`
(`ok(x)`/`err(e)`) is a tagged sum, and any future general
variant would be too. The shipped text convention encodes a
Result as an ordinary object:

```json
{"ok": true, "value": 42}
{"ok": false, "error": "boom"}
```

This COLLIDES with a genuine record that happens to have those
keys, which is why `parse_json`/`parse_toml` reconstruction is
OPT-IN (`{results: 1}`) rather than automatic: the decoder cannot
tell a serialized `ok(42)` from a data record `{ok: 1, value:
42}`. The compact form is serviceable, but it must NOT become the
universal variant encoding.

## Options considered

| Encoding | Compact | Self-describing | Unambiguous | Extends to N variants |
|---|---|---|---|---|
| Boolean discriminator (`{ok, value}`) -- SHIPPED | yes | no | **no** | no |
| String variant tag (`{type, variant, ...}`) | mid | yes | no (bare) | yes |
| Reserved tagged envelope (`{"$mlpl": {...}}`) | no | yes | **yes** | yes |
| None/Some literal | mid | partial | no | needs a distinct Option type |

## Decision: reserved tagged envelope (long-term canonical)

```json
{"$mlpl": {"v": 1, "type": "result", "variant": "ok", "value": 42}}
{"$mlpl": {"v": 1, "type": "result", "variant": "err", "error": "boom"}}
{"$mlpl": {"v": 1, "type": "option", "variant": "some", "value": 42}}
{"$mlpl": {"v": 1, "type": "option", "variant": "none"}}
{"$mlpl": {"v": 1, "type": "array", "shape": [2, 2], "data": [1, 2, 3, 4]}}
```

The same envelope carries the HIGHER-RANK round-trip problem: an
`array` entry with an explicit `shape` + flat `data` round-trips
a rank-`>=2` array losslessly, which the bare JSON encoding
cannot (it nests, and the flat-array parser rejects nested
arrays). So one mechanism answers three of the open serialization
items at once -- variant reconstruction, a shape/type envelope
for higher-rank data, and the text projection of a typed native
format. (An alternative to a shape envelope is genuine
nested-array support in the language itself; that is a value-
model decision, not a codec one -- see `docs/option-result-design.md`
and the Stage 6 nested-arrays story.)

The decisive advantage is NOT verbosity or self-description --
it is **disambiguation**, which changes the decode contract:

- A reserved top-key (`$mlpl`) is, by convention, never
  application data, so a decoder can reconstruct these
  **unconditionally** -- no `{results: 1}` opt-in. The opt-in
  today exists ONLY because the compact shape is ambiguous;
  the envelope removes the reason for it.
- `v` versions the envelope, so the format can evolve without
  breaking old documents (forward/backward compatibility).
- `type` + `variant` + payload is uniform across `result`,
  `option`, and any future user-defined variant -- one decoder
  path, not one per type.
- It is the same shape a **typed native** serialization would
  use; the envelope is that format's text projection, so this
  decision and the "typed native formats" item converge.

### Collision / escape policy

Application data that genuinely has a top-level `"$mlpl"` key is
the one hazard. Policy: reserve the `$mlpl` key; on ENCODE, a
data record whose key set includes `$mlpl` is wrapped
(`{"$mlpl":{"v":1,"type":"record","fields":{...}}}`) so the raw
key never appears un-escaped; on DECODE, `$mlpl` is always the
tag. This keeps the reserved key a true namespace, not a
heuristic.

## What stays

- `{ok, value|error}` + `parse_*(..., {results: 1})` remains a
  small, DOCUMENTED, opt-in **plain-JSON interop convention** --
  for exchanging Results with non-MLPL tools that want a readable
  object. It is not deprecated and not promoted to universal.
- The value model is unchanged: Result is the tagged sum,
  Option is zilde (`docs/option-result-design.md`). The envelope
  is a WIRE format, not a new value kind.

## Sequencing (a future saga, not now)

1. Depends on the language's variant story. Result can adopt the
   envelope immediately (as a `to_json(v, {tagged: 1})` mode +
   unconditional `$mlpl` reconstruction on decode). `option`/
   `some`/`none` envelopes only make sense once a distinct
   general variant type exists -- today "Option" is zilde, which
   serializes as an array, and a general sum type is a separate
   language-design decision.
2. Introduce the versioned envelope schema + reserve `$mlpl`
   + the escape policy.
3. Extend uniformly to each variant type as it lands.

Until then, the compact opt-in convention is the shipped answer,
and this note is the contract the envelope work will honor.
