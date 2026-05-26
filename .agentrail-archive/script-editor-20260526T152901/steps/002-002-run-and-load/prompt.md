Step 002: Run + Load.

Run button submits all non-empty non-comment lines from the editor textarea via on_run_batch. Lines starting with # are skipped. Clears REPL history before running (like a demo).

Load button opens a file picker (accept=.mlpl,.txt) that reads the file into the editor textarea. Uses the existing FileReader pattern from upload.rs.

Pages rebuild required.