use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    html! {
        <div id="repl-container">
            <header>
                <h1>{"MLPL"}</h1>
                <span>{"v0.2 — Array Programming Language for ML"}</span>
            </header>
            <main>
                <p>{"REPL loading..."}</p>
            </main>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
