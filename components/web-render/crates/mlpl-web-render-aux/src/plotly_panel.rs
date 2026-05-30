//! `<PlotlyPanel>` -- renders a `plotly3d` HTML/JS payload
//! from `mlpl-viz` and re-executes its inline `<script>` tag
//! after insertion. `Html::from_html_unchecked` sets
//! `innerHTML`, which the browser does NOT execute for script
//! tags inserted that way (per the WHATWG spec). The
//! `use_effect_with` hook scans the mounted node for any
//! `<script>` children and replaces each with a freshly-created
//! `<script>` element carrying the same text content -- which
//! the browser DOES execute. Saga 33 step 030.

use web_sys::HtmlDivElement;
use yew::prelude::*;

pub const PLOTLY_MARKER: &str = "<!-- mlpl-plotly3d -->";

#[derive(Properties, PartialEq)]
pub struct PlotlyPanelProps {
    pub payload: String,
}

#[function_component(PlotlyPanel)]
pub fn plotly_panel(props: &PlotlyPanelProps) -> Html {
    let node_ref = use_node_ref();
    let payload_key = props.payload.clone();
    {
        let nr = node_ref.clone();
        use_effect_with(payload_key, move |_| {
            if let Some(div) = nr.cast::<HtmlDivElement>() {
                rerun_inline_scripts(&div);
            }
            || ()
        });
    }
    html! {
        <div class="plotly-output" ref={node_ref}>
            { Html::from_html_unchecked(AttrValue::from(props.payload.clone())) }
        </div>
    }
}

fn rerun_inline_scripts(container: &HtmlDivElement) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let scripts = container.get_elements_by_tag_name("script");
    let len = scripts.length();
    let mut originals = Vec::with_capacity(len as usize);
    for i in 0..len {
        if let Some(s) = scripts.item(i) {
            originals.push(s);
        }
    }
    for old in originals {
        let Ok(new_script) = document.create_element("script") else {
            continue;
        };
        let text = old.text_content().unwrap_or_default();
        new_script.set_text_content(Some(&text));
        if let Some(parent) = old.parent_node() {
            let _ = parent.replace_child(&new_script, &old);
        }
    }
}
