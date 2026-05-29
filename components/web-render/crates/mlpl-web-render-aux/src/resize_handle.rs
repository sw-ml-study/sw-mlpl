use wasm_bindgen::prelude::*;
use yew::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn __initResize(el: &web_sys::HtmlElement);
}

#[function_component(ResizeHandle)]
pub fn resize_handle() -> Html {
    let node = use_node_ref();
    let node_c = node.clone();
    use_effect_with((), move |_| {
        if let Some(el) = node_c.cast::<web_sys::HtmlElement>() {
            __initResize(&el);
        }
        || ()
    });
    html! { <div ref={node} class="resize-handle" id="viz3d-resize" /> }
}
