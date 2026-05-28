Step 001: Split paths.rs (1162 lines) into smaller files.

Extract each path definition into a separate file (paths_zoo.rs, paths_history.rs, etc.) with paths.rs as a facade that re-exports. Target: paths.rs under 100 lines, each path file under 300 lines.