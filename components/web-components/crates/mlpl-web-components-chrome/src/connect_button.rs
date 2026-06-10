//! `<ConnectButton>` -- show/choose the connected backend.
//!
//! - Connected (`?connect=` set): clicking shows WHAT we're connected to
//!   (hostname + devices, fetched from the server) and a Disconnect
//!   action. Later, a serving proxy can also list other backends to
//!   switch to.
//! - Not connected: clicking asks the serving origin for a backend list
//!   (`GET /v1/backends`, the future proxy contract) and shows a picker;
//!   when there is no proxy -- the static GitHub Pages demo -- it warns
//!   (n/a) and points at the manual `?connect=` mechanism.
//!
//! Self-contained -- no props, no parent plumbing.

use yew::prelude::*;

const NO_PROXY: &str = "Not connected. Backend selection needs a connect proxy that lists \
    running mlpl-serve instances -- this static demo has none. To use your own server, append \
    ?connect=<server-url> to this page's URL (e.g. ?connect=http://host:6464).";

#[derive(Clone, PartialEq)]
struct Backend {
    name: String,
    url: String,
}

#[derive(Clone, PartialEq)]
enum Panel {
    Closed,
    Connected {
        url: String,
        host: Option<String>,
        devices: Option<String>,
    },
    Note(String),
    List(Vec<Backend>),
}

async fn get_json(url: &str) -> Option<serde_json::Value> {
    let resp = gloo::net::http::Request::get(url).send().await.ok()?;
    resp.ok().then_some(())?;
    resp.json().await.ok()
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

/// Reload with (`Some` -> connect to url) or without (`None` ->
/// disconnect) the `?connect=` query.
fn set_connect(url: Option<&str>) {
    if let Some(win) = web_sys::window() {
        let q = url.map_or_else(String::new, |u| format!("connect={u}"));
        let _ = win.location().set_search(&q);
    }
}

/// Compute the panel contents on click: when connected, what we're
/// connected to (+ devices) for a Disconnect; otherwise the proxy's
/// backend list, or the no-proxy note.
async fn discover(connected: Option<String>) -> Panel {
    if let Some(url) = connected {
        let info = get_json(&format!("{}/v1/devices", url.trim_end_matches('/'))).await;
        let field = |k: &str| info.as_ref().and_then(|b| b.get(k));
        let host = field("hostname")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let devices = field("devices").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        });
        return Panel::Connected { url, host, devices };
    }
    let Some(origin) = web_sys::window().and_then(|w| w.location().origin().ok()) else {
        return Panel::Note(NO_PROXY.to_string());
    };
    let backend = |b: &serde_json::Value| {
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
    };
    match get_json(&format!("{}/v1/backends", origin.trim_end_matches('/'))).await {
        Some(b) => match b.get("backends").and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => Panel::List(arr.iter().filter_map(backend).collect()),
            _ => Panel::Note(NO_PROXY.to_string()),
        },
        None => Panel::Note(NO_PROXY.to_string()),
    }
}

fn render_panel(panel: &Panel) -> Html {
    match panel {
        Panel::Closed => html! {},
        Panel::Connected { url, host, devices } => {
            let title = host.clone().unwrap_or_else(|| "server".to_string());
            html! {
                <div class="connect-panel">
                    <div class="connect-note">
                        { format!("Connected to {title}") }
                        <div class="connect-sub">{ url.clone() }</div>
                        { devices.as_ref().map_or_else(|| html!{}, |d| html!{ <div class="connect-sub">{ format!("devices: {d}") }</div> }) }
                    </div>
                    <button class="connect-item" onclick={Callback::from(|_: MouseEvent| set_connect(None))}>
                        { "Disconnect" }
                    </button>
                </div>
            }
        }
        Panel::Note(msg) => html! {
            <div class="connect-panel"><div class="connect-note">{ msg }</div></div>
        },
        Panel::List(items) => html! {
            <div class="connect-panel">
                { for items.iter().map(|b| {
                    let url = b.url.clone();
                    let go = Callback::from(move |_: MouseEvent| set_connect(Some(&url)));
                    html! { <button class="connect-item" onclick={go}>{ format!("{} ({})", b.name, b.url) }</button> }
                }) }
            </div>
        },
    }
}

#[function_component(ConnectButton)]
pub fn connect_button() -> Html {
    let panel = use_state(|| Panel::Closed);
    let connected = current_connect();
    let onclick = {
        let panel = panel.clone();
        let connected = connected.clone();
        Callback::from(move |_: MouseEvent| {
            if !matches!(*panel, Panel::Closed) {
                panel.set(Panel::Closed);
                return;
            }
            let (panel, connected) = (panel.clone(), connected.clone());
            wasm_bindgen_futures::spawn_local(async move {
                panel.set(discover(connected).await);
            });
        })
    };
    let label = if connected.is_some() {
        "Connected \u{2713}"
    } else {
        "Connect"
    };
    html! {
        <div class="connect-wrap">
            <button class="connect-btn" {onclick} title="Connection status / backends">{ label }</button>
            { render_panel(&panel) }
        </div>
    }
}
