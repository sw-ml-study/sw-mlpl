# Saga: file-metadata-timestamp

Add a confined `file_metadata(path)` primitive to the MLPL filesystem
surface so applications can read a file's last-MODIFIED time. This
unblocks ../demo-extensions' real-file Model Atlas picker (sort/show by
date) and ../demo-file-processing's date-scanning demos, which today
must label modification time unavailable.

Exact contract (from demo-extensions/docs/sw-mlpl-blockers.md +
upstream-contract.md):

- `file_metadata(path) -> ok({kind, size, modified_unix_ms}) / err`.
  A record is preferred over a bare scalar (Records + Results already
  ship); `kind` is "file"/"dir"/"other", `size` is the byte length,
  `modified_unix_ms` is the last-modified time.
- `modified_unix_ms` is an exact UTC **Unix-millisecond integer**
  (well within f64's exact-integer range). It is the MODIFICATION
  time only -- NEVER access time, creation/birth time, local time, or
  the current clock.
- When the platform has no modification time, return an `err` Result
  (a descriptive message), NEVER a silent 0 / sentinel.
- Same sandbox as `file_size` / `read_bytes` / `fs_walk`: paths
  resolve against `env.fs_root`, symlink-escape is refused
  (`contained`), and an unconfigured surface (no `fs_root`) is an
  `err` ("no filesystem sandbox on this surface").
- Consistent on macOS and Linux.

Interpreter reference to mirror: `fs_range.rs::eval_file_size` (same
sandbox + Result shape), helpers `contained` / `fs_ok` / `fs_err` in
`fncall_fs.rs`, dispatch in `fncall_fs.rs::try_dispatch`. Records are
`Value::Record { fields: BTreeMap<String, Value> }`.

This saga ships the INTERPRETER primitive (the interpreted downstream
demo + model picker consume it). The compile-to-Rust lowering is a
later compiler rung (queued), not part of this saga.

Each step is TDD (RED failing test -> GREEN minimal code -> refactor).
Hold or lower sw-checklist.

## Steps

1. interpreter-primitive -- implement `file_metadata(path)` in a new
   `fs_meta.rs` module (pure `read_metadata` helper + a thin eval
   shell), wired into `fncall_fs.rs` dispatch and `lib.rs`. TDD: a
   sandboxed temp file whose mtime is set to a KNOWN value returns
   `ok({kind:"file", size, modified_unix_ms})` with that exact ms
   (wall-clock-independent); a directory returns kind "dir"; a missing
   file, a symlink-escape, and an unconfigured surface each return a
   descriptive `err`; the value is the modification time, not the
   current clock (proven by the known-mtime pin).

2. docs-close -- user-facing docs for `file_metadata` (lang-reference
   fs table, WHAT/HOW only), update docs/future-sagas-queue.md
   (`file-metadata-timestamp` interpreter-side SHIPPED; compiled-path
   lowering queued as a follow-on), refresh
   docs/companion-demo-extensions.md +
   docs/companion-demo-file-processing.md (primitive shipped; the
   downstream repos now demonstrate/consume it), and the wiki
   (Documentation-Errata + a Capability-Matrix row). Hold
   sw-checklist. `--done`.
