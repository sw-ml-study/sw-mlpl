use yew::prelude::*;

#[function_component(Welcome)]
pub fn welcome() -> Html {
    html! {
        <div class="welcome">
            <p>{"Welcome to MLPL. Type expressions and press Enter."}</p>
            <p>{"Try: "}<code>{"1 + 2"}</code>{", "}<code>{"range(5)"}</code>{", "}<code>{"reshape(range(6), [2, 3])"}</code></p>
            <p>{"Type "}<code>{":help"}</code>{" for the function list, "}<code>{":clear"}</code>{" to reset, or click "}<code>{"?"}</code>{" for full docs."}</p>
        </div>
    }
}
