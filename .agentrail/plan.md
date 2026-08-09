# Saga: byte-stream-contract

demo-file-processing / demo-ml-utils: bounded INCREMENTAL output
(the sink half of the ByteSource/ByteSink contract). MLPL has no
mutable handles, and the source half already exists (read_bytes
range = read, file_size = size/position). The missing verb is the
incremental sink -> append_bytes(path, bytes): append a byte-
array chunk to a file (creating if absent), returning ok(count).
Position = file_size; flush is implicit per append. Libraries
build framing / stream-folds / buffering on top.

append_bytes mirrors write_bytes: byte array (rank-<=1, 0..=255),
Result-based (invalid input -> err Result), sandboxed. Text
appends via tokenize_bytes. New fs_append.rs module (keeps
fs_bytes/fs_range at budget). Compiler parity is the separate
compiler-io-parity saga (item 3).

## Steps
1. append-bytes -- fs_append.rs (append_bytes) + dispatch;
   catalog/lang-ref/glossary; TDD (new file, incremental grow,
   count, bounded read->transform->append, invalid/sandbox/no-
   sandbox errs).
2. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a, --done.
