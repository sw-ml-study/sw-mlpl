use wasm_bindgen::prelude::*;
use yew::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_init(canvas: &web_sys::HtmlCanvasElement);
    #[wasm_bindgen(js_namespace = window)]
    fn __stage3d_destroy();
}

#[function_component(Stage3dPanel)]
pub fn stage3d_panel() -> Html {
    let node = use_node_ref();
    let node_c = node.clone();
    use_effect_with((), move |_| {
        if let Some(canvas) = node_c.cast::<web_sys::HtmlCanvasElement>() {
            __stage3d_init(&canvas);
        }
        || {
            __stage3d_destroy();
        }
    });
    html! { <canvas ref={node} id="stage3d" /> }
}
