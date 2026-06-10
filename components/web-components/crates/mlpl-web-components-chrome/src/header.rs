use yew::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeaderMode {
    Repl,
    Tutorial,
    Paths,
    Editor,
}

#[derive(Properties, PartialEq)]
pub struct HeaderProps {
    pub on_help: Callback<MouseEvent>,
    pub on_tour: Callback<MouseEvent>,
    pub on_select_repl: Callback<MouseEvent>,
    pub on_select_tutorial: Callback<MouseEvent>,
    pub on_select_paths: Callback<MouseEvent>,
    pub on_select_editor: Callback<MouseEvent>,
    pub mode: HeaderMode,
}

#[function_component(Header)]
pub fn header(props: &HeaderProps) -> Html {
    let cls = |m: HeaderMode| {
        if props.mode == m { "tab active" } else { "tab" }
    };
    html! {
        <header>
            <h1><img src="mlpl-badge.webp" alt="" class="title-badge" />{"sw-MLPL"}</h1>
            <div class="title-text">
                <span class="title-line">{"Software Wrighter's Machine Learning Programming Language Playground"}</span>
                <span class="title-subtitle">{"Array Programming for Machine Learning"}</span>
            </div>
            <div class="header-tabs">
                <button class={cls(HeaderMode::Repl)} onclick={props.on_select_repl.clone()} data-tour-target="tab-repl">{"REPL"}</button>
                <button class={cls(HeaderMode::Tutorial)} onclick={props.on_select_tutorial.clone()} data-tour-target="tab-tutorial">{"Tutorial"}</button>
                <button class={cls(HeaderMode::Paths)} onclick={props.on_select_paths.clone()} data-tour-target="tab-paths">{"Paths"}</button>
                <button class={cls(HeaderMode::Editor)} onclick={props.on_select_editor.clone()} data-tour-target="tab-editor">{"Editor"}</button>
            </div>
            <crate::connect_button::ConnectButton />
            <button class="tour-btn-header" onclick={props.on_tour.clone()} aria-label="Guided tour" title="Guided tour" data-tour-target="tour-btn">{"Tour"}</button>
            <button class="help-btn" onclick={props.on_help.clone()} aria-label="Show documentation" title="Documentation" data-tour-target="help-btn">{"?"}</button>
        </header>
    }
}
