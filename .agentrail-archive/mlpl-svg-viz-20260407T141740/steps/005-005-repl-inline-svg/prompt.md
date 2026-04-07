Make both REPLs render SVG output specially.

1. Browser REPL (apps/mlpl-web):
   - In render_entry, detect if entry.output starts with "<svg"
   - If yes: use Yew's Html::from_html_unchecked or
     gloo dangerously_set_inner_html equivalent
     to render the SVG inline (not as text)
   - Add CSS so the SVG sits in a bordered card with padding
   - If the output is plain text, render as before in <pre>

2. CLI REPL (apps/mlpl-repl):
   - If output starts with "<svg", offer two behaviors:
     a. Print "[svg: N bytes]" placeholder
     b. With --svg-out flag, write each SVG to a numbered file
        in cwd (svg-001.svg, svg-002.svg, ...)
   - Default: placeholder

3. Tests:
   - Browser: manual verification (build + serve)
   - CLI: integration test that writes a file with --svg-out

4. Update apps/mlpl-web pages/ via build-pages.sh

Allowed: apps/mlpl-web, apps/mlpl-repl, crates/mlpl-eval