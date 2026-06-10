//! `<ConnectButton>` -- pick a backend to connect to.
//!
//! Forward-looking: when this app is served by a connect PROXY that
//! lists running mlpl-serve instances (e.g. several multi-GPU hosts,
//! some CUDA, some MLX), the button fetches `GET /v1/backends` from the
//! serving origin and shows a picker; choosing one reloads the page
//! with `?connect=<url>`. On the static GitHub Pages demo there is no
//! proxy, so it warns (n/a) and points at the manual `?connect=`
//! mechanism. Self-contained -- no props, no parent plumbing.

use yew::prelude::*;

const NO_PROXY: &str = "Backend selection needs a connect proxy that lists running \
    mlpl-serve instances -- this static demo has none. To use your own server, append \
    ?connect=<server-url> to this page's URL (e.g. ?connect=http://host:6464).";

#[derive(Clone, PartialEq)]
struct Backend {
    name: String,
    url: String,
}

#[derive(Clone, PartialEq)]
enum Panel {
    Closed,
    Note(String),
    List(Vec<Backend>),
}

/// GET `<origin>/v1/backends` -> the proxy's backend list, or `None`
/// when there is no proxy / the response isn't the expected shape.
async fn fetch_backends() -> Option<Vec<Backend>> {
    let origin = web_sys::window()?.location().origin().ok()?;
    let url = format!("{}/v1/backends", origin.trim_end_matches('/'));
    let resp = gloo::net::http::Request::get(&url).send().await.ok()?;
    resp.ok().then_some(())?;
    let body: serde_json::Value = resp.json().await.ok()?;
    Some(
        body.get("backends")?
            .as_array()?
            .iter()
            .filter_map(|b| {
                Some(Backend {
                    name: b
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("backend")
                        .to_string(),
                    url: b
                        .get("url")
                        .and_then(serde_json::Value::as_str)?
                        .to_string(),
                })
            })
            .collect(),
    )
}

/// The current `?connect=` value, if the page is in connect mode.
fn current_connect() -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search
        .trim_start_matches('?')
        .split('&')
        .find_map(|p| p.strip_prefix("connect=").map(str::to_string))
        .filter(|s| !s.is_empty())
}

/// Reload the page pointed at `url` as the connect backend.
fn navigate_connect(url: &str) {
    if let Some(win) = web_sys::window() {
        let _ = win.location().set_search(&format!("connect={url}"));
    }
}

fn render_panel(panel: &Panel) -> Html {
    match panel {
        Panel::Closed => html! {},
        Panel::Note(msg) => html! {
            <div class="connect-panel"><div class="connect-note">{ msg }</div></div>
        },
        Panel::List(items) => html! {
            <div class="connect-panel">
                { for items.iter().map(|b| {
                    let url = b.url.clone();
                    let go = Callback::from(move |_: MouseEvent| navigate_connect(&url));
                    html! {
                        <button class="connect-item" onclick={go}>
                            { format!("{} ({})", b.name, b.url) }
                        </button>
                    }
                }) }
            </div>
        },
    }
}

#[function_component(ConnectButton)]
pub fn connect_button() -> Html {
    let panel = use_state(|| Panel::Closed);
    let onclick = {
        let panel = panel.clone();
        Callback::from(move |_: MouseEvent| {
            if !matches!(*panel, Panel::Closed) {
                panel.set(Panel::Closed);
                return;
            }
            let panel = panel.clone();
            wasm_bindgen_futures::spawn_local(async move {
                panel.set(match fetch_backends().await {
                    Some(list) if !list.is_empty() => Panel::List(list),
                    _ => Panel::Note(NO_PROXY.to_string()),
                });
            });
        })
    };
    let label = if current_connect().is_some() {
        "Connected \u{2713}"
    } else {
        "Connect"
    };
    html! {
        <div class="connect-wrap">
            <button class="connect-btn" {onclick} title="Connect to a backend">{ label }</button>
            { render_panel(&panel) }
        </div>
    }
}
