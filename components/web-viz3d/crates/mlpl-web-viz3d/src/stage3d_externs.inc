// JS bridge to the THREE.js stage in stage3d.js: a flat list of body-less
// FFI declarations, no logic. Kept in a `.inc` (not a `.rs`) and pulled into
// panel.rs with `include!`, so this simple list does not inflate panel.rs's
// function count and is never seen by the complexity checker (which scans
// `.rs` only). This is the sanctioned "simple list via include!" exception --
// see docs/code_metrics.md. No control flow or logic belongs in this file.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_init(canvas: &web_sys::HtmlCanvasElement);
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_destroy();
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_prev();
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_next();
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_home();
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_end();
}
