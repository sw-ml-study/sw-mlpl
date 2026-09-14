# moe-microscope-followups-3

Address the eval_stream server-surface gaps from ../moe-microscope's host-handoff
step (docs/sw-mlpl-findings.md "Follow-up batch 3"): F16 (include, filesystem
sandbox, args) and the remainder of S1. Lead with include -- it unblocks
submitting a module-split lesson without a client-side bundler.

## Steps

1. f16a-include-over-wire -- accept an optional `includes` map (virtual path ->
   source text) on the eval request; when present (or the program uses
   `include`), resolve the include tree server-side with the existing
   mlpl_source_loader MemoryProvider + expand (which already enforces the
   no-absolute / no-escape sandbox), and evaluate the expanded chunks in the
   session environment. Backward-compatible: no `includes` -> current behavior.
   TDD: a program that `include`s a module submitted with its map evaluates.

2. f16b-fs-sandbox -- give server-run programs a filesystem sandbox root for the
   fs builtins (read_bytes/write_bytes/...), configured on the server; reads and
   writes outside the root are refused. TDD.

3. f16c-args -- let the eval request pass `args` visible to the program. TDD.

4. relay-and-close -- mark F16/S1 resolved/documented, refresh CHANGES + wiki,
   mark the saga shipped, rebuild binaries (incl mlpl-serve). --done.
