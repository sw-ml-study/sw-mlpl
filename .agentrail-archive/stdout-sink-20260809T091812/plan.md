# Saga: stdout-sink

demo-file-processing: the non-seekable binary sink. append_bytes
covers file-path output; the remaining gap is writing raw bytes
to process stdout (pipes, non-seekable). Add write_stdout(bytes)
-> ok(count)/err: write a rank-<=1 byte array (0..=255) to
process stdout and flush. The ByteSink counterpart to the
existing read_stdin (non-seekable source). Text via
tokenize_bytes. Lives in eval_script.rs (process-stdio effects:
read_stdin/exit), wired through eval_intercepts. Not sandboxed
(stdout is the process own); Result-based.

## Steps
1. write-stdout -- eval_write_stdout in eval_script.rs + intercept
   dispatch; catalog/lang-ref/glossary; TDD (byte write via -f
   script to captured stdout; invalid input err; count).
2. close -- rebuild serve+pages+repl, deploy, wiki row, q-and-a,
   --done.
