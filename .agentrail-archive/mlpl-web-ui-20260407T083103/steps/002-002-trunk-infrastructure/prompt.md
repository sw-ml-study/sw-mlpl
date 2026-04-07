Set up Trunk build infrastructure for the web app.

1. Add dependencies to apps/mlpl-web/Cargo.toml:
   - yew = { version = "0.21", features = ["csr"] }
   - wasm-bindgen = "0.2"
   - web-sys (with features: HtmlInputElement, HtmlTextAreaElement,
     KeyboardEvent, console, Window, Document)
   - gloo = "0.11"
   - mlpl-wasm = { path = "../../crates/mlpl-wasm" }

2. Create apps/mlpl-web/index.html as the Trunk entry point:
   - Minimal HTML5 boilerplate
   - Catppuccin Mocha theme CSS (inline in <style>)
   - Monospace font (JetBrains Mono or similar from CDN)
   - <link data-trunk ... > for CSS if separate file

3. Create Trunk.toml at apps/mlpl-web/ (or repo root):
   - Configure dist directory
   - Set public-url to /sw-mlpl/

4. Create scripts/build-pages.sh:
   - Runs trunk build --release from apps/mlpl-web/
   - Copies output to pages/ directory at repo root
   - Preserves .nojekyll file

5. Create scripts/serve.sh for local dev:
   - Runs trunk serve from apps/mlpl-web/

6. Create pages/.nojekyll (empty file)

7. Verify trunk build works (or at minimum cargo build
   compiles without errors)

Allowed: apps/mlpl-web/, scripts/, pages/, Trunk.toml