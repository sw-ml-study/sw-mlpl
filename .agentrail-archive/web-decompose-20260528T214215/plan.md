Multi-step decomposition of mlpl-web (god-crate, 73 modules) into themed component workspaces.

Step plan (leaves first, hub last):
  1. extract-demos      -> components/web-demos/      (12 modules: demos.rs + demos_*)
  2. extract-glossary   -> components/web-glossary/   (2: glossary_*)
  3. extract-onboarding -> components/web-onboarding/ (4: onboarding_*)
  4. extract-paths      -> components/web-paths/      (6: paths*, paths_*)
  5. extract-components -> components/web-components/ (7: components.rs + component_*)
  6. extract-handlers   -> components/web-handlers/   (6: handlers*, upload_cmd)
  7. extract-mode       -> components/web-mode/       (3: mode_*)
  8. extract-render     -> components/web-render/     (11: render*, render_*, resize_handle)
  9. extract-misc       -> components/web-misc/       (plotly_panel, scroll, upload, readme_counts, tutorial)
 10. close              -> sw-checklist sweep + saga close

After all steps, mlpl-web should be ~6 modules (lib.rs facade + main + main_wasm_body + main_not_wasm_body + app + app cluster), all PASS-sized; crate-module-count FAIL retires.