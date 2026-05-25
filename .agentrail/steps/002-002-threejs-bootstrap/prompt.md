Phase 1b: Three.js bootstrap.

Create apps/mlpl-web/js/stage3d.js (~200 LOC JS module): Three.js scene, PerspectiveCamera, OrbitControls, white PlaneGeometry ground with grid lines, ambient + directional light, render loop. Export window.__stage3d_init(canvas) and window.__stage3d_destroy().

Add Three.js + OrbitControls as vendored JS or CDN <script> tags in index.html (loaded only when :3d is active -- lazy load). Wire Yew to call init/destroy via wasm_bindgen js_sys when show_3d state changes (use_effect_with).

New viz3d_panel.rs (2 fns max): the Yew component that renders the canvas and manages the JS lifecycle. Pages rebuild required.