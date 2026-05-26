use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use yew::prelude::*;

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

#[derive(Properties, PartialEq)]
pub struct Stage3dProps {
    pub on_inspect: Callback<String>,
}

#[function_component(Stage3dPanel)]
pub fn stage3d_panel(props: &Stage3dProps) -> Html {
    let node = use_node_ref();
    let node_c = node.clone();
    let on_inspect = props.on_inspect.clone();
    use_effect_with((), move |_| {
        if let Some(canvas) = node_c.cast::<web_sys::HtmlCanvasElement>() {
            __stage3d_init(&canvas);
        }
        let cb = Closure::wrap(Box::new(move |e: web_sys::CustomEvent| {
            let name = e.detail().as_string().or_else(|| {
                js_sys::Reflect::get(&e.detail(), &"name".into())
                    .ok()?
                    .as_string()
            });
            if let Some(n) = name {
                on_inspect.emit(n);
            }
        }) as Box<dyn FnMut(_)>);
        if let Some(w) = web_sys::window() {
            let _ =
                w.add_event_listener_with_callback("stage3d-inspect", cb.as_ref().unchecked_ref());
        }
        cb.forget();
        || {
            __stage3d_destroy();
        }
    });
    let prev = Callback::from(|_: MouseEvent| __stage3d_prev());
    let next = Callback::from(|_: MouseEvent| __stage3d_next());
    let home = Callback::from(|_: MouseEvent| __stage3d_home());
    let end = Callback::from(|_: MouseEvent| __stage3d_end());
    html! {
        <div class="stage3d-wrap">
            <canvas ref={node} id="stage3d" />
            <div class="stage3d-nav">
                <button class="stage3d-btn" onclick={home} title="First step">{"\u{23ee}"}</button>
                <button class="stage3d-btn" onclick={prev} title="Previous step">{"\u{25c0}"}</button>
                <button class="stage3d-btn" onclick={next} title="Next step">{"\u{25b6}"}</button>
                <button class="stage3d-btn" onclick={end} title="Latest step">{"\u{23ed}"}</button>
            </div>
        </div>
    }
}
