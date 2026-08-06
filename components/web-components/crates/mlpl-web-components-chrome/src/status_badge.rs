//! Header staleness badge: is the RUNNING bundle the one the
//! serving origin currently publishes? On mount the badge
//! fetches the page's OWN index.html (cache-busted, same
//! origin) and compares hashed bundle names -- so a browser
//! showing a cached page learns a newer build exists. The
//! default is UNKNOWN: offline, airplane mode, or any fetch
//! failure shows a neutral badge, never a false claim.

use yew::prelude::*;

/// The comparison verdict behind the badge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BundleStatus {
    /// The served index references the bundle we are running.
    Fresh,
    /// The origin serves a NEWER build -- reload to get it.
    Stale,
    /// Could not tell (offline / fetch failed / parse failed).
    Unknown,
}

/// The `mlpl-web-<hash>` bundle name inside an index.html (or a
/// script URL). None when no bundle reference is present.
#[must_use]
pub fn extract_bundle(html: &str) -> Option<String> {
    let start = html.find("mlpl-web-")?;
    let tail = &html[start + 9..];
    let hex: String = tail.chars().take_while(char::is_ascii_hexdigit).collect();
    if hex.len() < 8 {
        return None;
    }
    Some(format!("mlpl-web-{hex}"))
}

/// Compare the running bundle against the served one; any
/// missing side is UNKNOWN (the honest offline default).
#[must_use]
pub fn verdict(own: Option<&str>, served: Option<&str>) -> BundleStatus {
    match (own, served) {
        (Some(a), Some(b)) if a == b => BundleStatus::Fresh,
        (Some(_), Some(_)) => BundleStatus::Stale,
        _ => BundleStatus::Unknown,
    }
}

/// Mount-time check: own bundle name from the DOM script tag,
/// served bundle name from a cache-busted same-origin fetch.
#[hook]
pub fn use_bundle_status() -> BundleStatus {
    let status = use_state(|| BundleStatus::Unknown);
    {
        let status = status.clone();
        use_effect_with((), move |()| {
            wasm_bindgen_futures::spawn_local(async move {
                status.set(probe_bundle_status().await);
            });
            || ()
        });
    }
    *status
}

async fn probe_bundle_status() -> BundleStatus {
    let own = own_bundle();
    let ts = js_sys::Date::now() as u64;
    let served = match gloo::net::http::Request::get(&format!("index.html?ts={ts}"))
        .send()
        .await
    {
        Ok(resp) if resp.ok() => resp.text().await.ok().as_deref().and_then(extract_bundle),
        _ => None,
    };
    verdict(own.as_deref(), served.as_deref())
}

/// The running page's own hashed bundle name, read from the
/// script tag trunk injected.
fn own_bundle() -> Option<String> {
    let doc = web_sys::window()?.document()?;
    let scripts = doc.query_selector_all("script[src]").ok()?;
    for i in 0..scripts.length() {
        if let Some(el) = scripts.item(i)
            && let Some(el) = el.dyn_ref::<web_sys::Element>()
            && let Some(src) = el.get_attribute("src")
            && let Some(name) = extract_bundle(&src)
        {
            return Some(name);
        }
    }
    None
}

use wasm_bindgen::JsCast;

/// The badge itself: green "current", amber "update -- reload"
/// (clicking reloads), neutral "status unknown" otherwise.
#[function_component(StatusBadge)]
pub fn status_badge() -> Html {
    let status = use_bundle_status();
    let (class, label, title) = match status {
        BundleStatus::Fresh => (
            "status-badge status-badge-fresh",
            "current",
            "This page runs the build the site currently serves.",
        ),
        BundleStatus::Stale => (
            "status-badge status-badge-stale",
            "update available -- reload",
            "The site serves a newer build than this page. Click to reload.",
        ),
        BundleStatus::Unknown => (
            "status-badge status-badge-unknown",
            "status unknown",
            "Could not reach the server to compare builds (offline?).",
        ),
    };
    let onclick = Callback::from(move |_: MouseEvent| {
        if status == BundleStatus::Stale
            && let Some(win) = web_sys::window()
        {
            let _ = win.location().reload();
        }
    });
    html! { <span {class} {title} {onclick}>{ label }</span> }
}
