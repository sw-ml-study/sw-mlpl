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

const UNREACHABLE_HINT: &str = "Check that mlpl-serve is running there and the URL is right \
    -- usually the page's own origin (e.g. ?connect=http://large12:6464). NOTE: \"localhost\" \
    here means the machine running this BROWSER, not the page's server.";

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

use crate::connect_probe::get_json;

/// The current `?connect=` value, if the page is in connect mode
/// (the `off` disconnect sentinel reads as not connected).
fn current_connect() -> Option<String> {
    mlpl_web_eval::eval_url::current_connect_url_from_window()
}

/// Reload with (`Some` -> connect to url) or (`None` -> disconnect)
/// the `?connect=` query. Disconnect writes the `off` sentinel
/// rather than clearing the param, so same-origin autoconnect does
/// not immediately re-connect on the reload.
fn set_connect(url: Option<&str>) {
    if let Some(win) = web_sys::window() {
        let q = format!("connect={}", url.unwrap_or("off"));
        let _ = win.location().set_search(&q);
    }
}

/// Compute the panel contents on click: when connected, what we're
/// connected to (+ devices) for a Disconnect; otherwise the proxy's
/// backend list, or the no-proxy note.
async fn discover(connected: Option<String>) -> Panel {
    // An https demo page can't reach an http connect server (mixed content);
    // show why instead of trying to fetch /v1/devices and failing.
    if let Some(reason) = mlpl_web_eval::connect_guard::connect_blocked_reason() {
        return Panel::Note(reason);
    }
    if let Some(url) = connected {
        let Some(body) = get_json(&format!("{}/v1/devices", url.trim_end_matches('/'))).await
        else {
            return Panel::Note(format!(
                "?connect={url} is set, but that server is NOT responding. {UNREACHABLE_HINT}"
            ));
        };
        let host = body["hostname"].as_str().map(str::to_string);
        return Panel::Connected {
            url,
            host,
            devices: Some(crate::connect_probe::backend_status(&body)),
        };
    }
    let Some(origin) = web_sys::window().and_then(|w| w.location().origin().ok()) else {
        return Panel::Note(NO_PROXY.to_string());
    };
    discover_unconnected(&origin).await
}

/// Backend choices when NOT connected: the serving origin itself
/// (when it answers `/v1/devices` -- the disconnect->reconnect
/// path), plus anything a connect proxy lists at `/v1/backends`.
async fn discover_unconnected(origin: &str) -> Panel {
    let mut items = Vec::new();
    let origin = origin.trim_end_matches('/');
    if let Some(body) = get_json(&format!("{origin}/v1/devices")).await {
        let host = body["hostname"].as_str().unwrap_or("this server");
        let name = format!("{host} (the server hosting this page)");
        let url = origin.to_string();
        items.push(Backend { name, url });
    }
    let backend = |b: &serde_json::Value| {
        let name = b["name"].as_str().unwrap_or("backend").to_string();
        let url = b["url"].as_str()?.to_string();
        Some(Backend { name, url })
    };
    if let Some(b) = get_json(&format!("{origin}/v1/backends")).await
        && let Some(arr) = b.get("backends").and_then(|v| v.as_array())
    {
        items.extend(arr.iter().filter_map(backend));
    }
    if items.is_empty() {
        Panel::Note(NO_PROXY.to_string())
    } else {
        Panel::List(items)
    }
}

/// The connected-state panel: host title, url, per-backend status
/// line, and the Disconnect action.
fn render_connected(url: &str, host: Option<&str>, devices: Option<&str>) -> Html {
    let title = host.unwrap_or("server").to_string();
    let status = devices.map_or_else(
        || html! {},
        |d| html! { <div class="connect-sub">{ format!("devices: {d}") }</div> },
    );
    html! {
        <div class="connect-panel">
            <div class="connect-note">
                { format!("Connected to {title}") }
                <div class="connect-sub">{ url.to_string() }</div>
                { status }
            </div>
            <button class="connect-item" onclick={Callback::from(|_: MouseEvent| set_connect(None))}>
                { "Disconnect" }
            </button>
        </div>
    }
}

fn render_panel(panel: &Panel) -> Html {
    match panel {
        Panel::Closed => html! {},
        Panel::Connected { url, host, devices } => {
            render_connected(url, host.as_deref(), devices.as_deref())
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
    // Only claim "Connected" when the server actually ANSWERED the
    // devices probe -- a ?connect= URL pointing at a dead port must
    // show a warning, not a lying check mark.
    let reachable = crate::connect_probe::use_reachable();
    let blocked = mlpl_web_eval::connect_guard::connect_blocked_reason().is_some();
    let label = if blocked {
        "Connect \u{26a0}"
    } else if connected.is_some() {
        match reachable {
            None => "Connecting\u{2026}",
            Some(true) => "Connected \u{2713}",
            Some(false) => "Connect \u{26a0}",
        }
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
