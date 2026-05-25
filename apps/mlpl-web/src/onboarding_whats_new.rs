use yew::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const ITEMS: &[(&str, &str)] = &[
    (
        "Guided tour",
        "Click the Tour button next to ? to walk through every feature.",
    ),
    (
        "Splash screen",
        "First-time visitors see an interactive welcome overlay with quick-start cards.",
    ),
    (
        "REPL autocomplete",
        "Press Ctrl+Space for context-aware completions. Arrow keys to navigate, Enter to accept.",
    ),
    (
        "Dimensionality reduction",
        "UMAP, MDS, random projection, PCA loadings, and critical-dimensions heatmaps.",
    ),
];

#[derive(Properties, PartialEq)]
pub struct WhatsNewProps {
    pub on_dismiss: Callback<MouseEvent>,
}

#[function_component(WhatsNewOverlay)]
pub fn whats_new_overlay(props: &WhatsNewProps) -> Html {
    html! {
        <div class="splash-backdrop" role="dialog" aria-label="What is new">
            <div class="splash-panel">
                <h2>{format!("What's new in v{VERSION}")}</h2>
                <ul class="whats-new-list">
                    { for ITEMS.iter().map(|(h, b)| html! {
                        <li><strong>{ *h }</strong>{" -- "}{ *b }</li>
                    }) }
                </ul>
                <div class="splash-actions">
                    <button class="splash-btn splash-btn-primary" onclick={props.on_dismiss.clone()}>{"Got it"}</button>
                </div>
            </div>
        </div>
    }
}
