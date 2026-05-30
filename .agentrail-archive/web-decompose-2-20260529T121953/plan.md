Resume mlpl-web god-crate decomposition (paused from saga 81 after Tour hotfix). Same step plan, leaves-first carve order. Each step extracts one themed cluster into a new components/web-* workspace with the 4-crate sub-component pattern where the cluster overruns the 7-modules-per-crate limit. Goal: reduce mlpl-web from 73 modules toward the 7-module crate-module-count limit.

Step plan:
  1. extract-demos      -> components/web-demos/      (12 modules, needs 4 sub-crates: types + basic + vision + facade)
  2. extract-glossary   -> components/web-glossary/   (2 modules: glossary_*)
  3. extract-onboarding -> components/web-onboarding/ (4 modules: onboarding_*)
  4. extract-paths      -> components/web-paths/      (6 modules: paths*, paths_*)
  5. extract-components -> components/web-components/ (7 modules: components.rs + component_*)
  6. extract-handlers   -> components/web-handlers/   (6 modules: handlers*, upload_cmd)
  7. extract-mode       -> components/web-mode/       (3 modules: mode_*)
  8. extract-render     -> components/web-render/     (11 modules: render*, render_*, resize_handle)
  9. extract-misc       -> components/web-misc/       (plotly_panel, scroll, upload, readme_counts, tutorial)
 10. close              -> sw-checklist sweep + saga close