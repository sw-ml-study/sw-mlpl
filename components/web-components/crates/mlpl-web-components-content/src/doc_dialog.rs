use yew::prelude::*;

use mlpl_web_eval::state::DocTab;
use mlpl_web_glossary::view::GlossaryView;

const LANG_REFERENCE: &str = include_str!("../../../../../docs/lang-reference.md");
const USAGE_GUIDE: &str = include_str!("../../../../../docs/usage.md");

#[derive(Properties, PartialEq)]
pub struct DocDialogProps {
    pub open: bool,
    pub on_close: Callback<MouseEvent>,
}

#[function_component(DocDialog)]
pub fn doc_dialog(props: &DocDialogProps) -> Html {
    let active_tab = use_state(|| DocTab::LangReference);

    if !props.open {
        return html! {};
    }

    let body = match *active_tab {
        DocTab::LangReference => html! { <pre class="doc-content">{ LANG_REFERENCE }</pre> },
        DocTab::Usage => html! { <pre class="doc-content">{ USAGE_GUIDE }</pre> },
        DocTab::Glossary => html! { <GlossaryView /> },
        DocTab::Diagrams => html! { <mlpl_web_paths::diagrams::DiagramsView /> },
    };
    let cls = |t: DocTab| {
        if *active_tab == t {
            "tab active"
        } else {
            "tab"
        }
    };
    let setter = |t: DocTab| {
        let h = active_tab.clone();
        Callback::from(move |_| h.set(t))
    };
    let stop = Callback::from(|e: MouseEvent| e.stop_propagation());
    html! {
        <div class="modal-backdrop" onclick={props.on_close.clone()}>
            <div class="modal" onclick={stop}>
                <div class="modal-header">
                    <div class="tabs">
                        <button class={cls(DocTab::LangReference)} onclick={setter(DocTab::LangReference)} data-tour-target="doc-tab-reference">{"Language Reference"}</button>
                        <button class={cls(DocTab::Usage)} onclick={setter(DocTab::Usage)} data-tour-target="doc-tab-usage">{"Usage Guide"}</button>
                        <button class={cls(DocTab::Glossary)} onclick={setter(DocTab::Glossary)} data-tour-target="doc-tab-glossary">{"Glossary"}</button>
                        <button class={cls(DocTab::Diagrams)} onclick={setter(DocTab::Diagrams)} data-tour-target="doc-tab-diagrams">{"Diagrams"}</button>
                    </div>
                    <button class="close-btn" onclick={props.on_close.clone()} aria-label="Close">{"×"}</button>
                </div>
                <div class="modal-body">{ body }</div>
            </div>
        </div>
    }
}
