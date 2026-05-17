Saga 21.5 step 005: viz-format-table.

Goal: grow mlpl_cli::viz_cache::is_svg_string into a small format-detection table that covers SVG (<svg), HTML (<!DOCTYPE html, <html), PNG (magic bytes 89 50 4E 47), JPEG (FF D8 FF), and an explicit application/json opt-in. Both the MLPL_CACHE_DIR path and the /v1/viz storage endpoint use this table, so a loss_curve(last_losses) returning a PNG (when mlpl-viz emits one) gets the right Content-Type AND the right file extension in the cache dir. The /v1/viz store now persists arbitrary content-types from the table.

TDD (Red/Green/Refactor):

1. RED tests:
   - crates/mlpl-cli/tests/viz_format_tests.rs: detector returns Svg for <svg payload; Html for <!DOCTYPE html; Png for magic-byte PNG fixture; Jpeg for FF D8 FF; Json only when payload starts with explicit '{' AND opt-in flag.
   - crates/mlpl-serve/tests/viz_storage_tests.rs: extend with a PNG round-trip test (POST PNG, GET with Content-Type image/png).
   - crates/mlpl-cli/tests/viz_cache_tests.rs (extend): write_to_cache writes .png/.html/.svg by detected format.

2. GREEN:
   - new enum VizFormat { Svg, Html, Png, Jpeg, Json } + detect(bytes: &[u8]) -> Option<VizFormat>.
   - is_svg_string stays as a wrapper for back-compat: detect(s.as_bytes()) == Some(Svg).
   - write_to_cache grows a format-aware filename: <hash>.{svg|html|png|jpg|json}.
   - mlpl-serve's attach_viz handles non-SVG returns when (some future) builtin emits them; the SVG path is unchanged.
   - Content-Type lookup table: Svg -> image/svg+xml, Html -> text/html, Png -> image/png, etc.

3. REFACTOR: keep mlpl-cli viz_cache.rs under sw-checklist budgets; add detect to a small inline  sub-mod if necessary.

Quality gates per /mw-cp: cargo test (workspace), cargo clippy, cargo fmt, markdown-checker on contract touch, sw-checklist (held). Commit before agentrail complete; push after.

Out of scope: web REPL connect mode (step 006); web REPL streaming (007); web REPL viz storage fetch (008).