# Saga: semantic-result-reconstruction

demo-algorithms serialization: close the Result round-trip gap.
to_json encodes ok(x)/err(e) as {"ok":true,"value":x} /
{"ok":false,"error":e}, but parse_json reads that back as a
plain RECORD -- the Result semantics are lost. This saga adds
OPT-IN reconstruction so ok/err survive a JSON round trip.

## Why opt-in

The encoded shape {ok, value} is a legitimate ordinary record;
turning every such record into a Result unconditionally would
mangle genuine data. So reconstruction is a parse OPTION, off by
default -- the caller enables it precisely when they serialized
Results and want them back.

## Design

- Extend the parse-options record (2nd arg of parse_json /
  parse_toml, already carrying max_depth / max_bytes) with a
  `results` flag: `parse_json(text, {results: 1})`.
- When set, after decoding, walk the value and convert every
  Result-SHAPED record into a Result:
  - exactly {ok, value} with ok == 1  -> ok(reconstruct(value))
  - exactly {ok, error} with ok == 0  -> err(reconstruct(error))
  - any other record -> recurse into its fields
  - non-records unchanged (JSON arrays are flat, no nested
    records).
  Recursion rebuilds nested Results (e.g. ok({a: ok(1)})).
- Default (no flag) keeps today's behavior exactly. The
  ambiguity (a genuine {ok:1, value:x} data record becomes ok(x)
  under the flag) is documented -- it is the intended inverse of
  to_json's encoding.
- Applies to parse_toml too (the walker is generic), though
  Results in TOML are niche.

## Modules

- result_reconstruct.rs -- pure `reconstruct(Value) -> Value`
  walker.
- decode_limits.rs -- `text_and_options` also returns the
  `results` flag (read inline in the option parser); no new
  function.
- fncall_json / fncall_toml -- apply reconstruct to the decoded
  value when the flag is set.

## Steps
1. result-reconstruct -- result_reconstruct.rs walker; thread the
   `results` flag through the options; apply in eval_parse_json /
   eval_parse_toml; catalog signature notes + lang-ref +
   glossary. TDD (ok/err round trip, nested Results, opt-in
   default off, non-result records untouched, err payloads).
2. close -- rebuild serve+pages+repl, deploy, connect smoke of a
   full ok/err JSON round trip, wiki row, q-and-a (Result
   reconstruction done; remaining: typed native formats,
   streaming, higher-rank text round-trips), --done.
