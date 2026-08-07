//! Header + splash staleness badge, provenance edition: the
//! RUNNING page carries its identity in the `mlpl-build` meta
//! tag build-pages.sh stamps; the SERVING origin's identity
//! comes from a cache-busted same-origin `build-info.json`;
//! and (best effort) the REPO's committed pages state comes
//! from raw.githubusercontent's main `pages/build-info.json`.
//! Verdicts: current / stale page (reload fixes it) / deploy
//! pending (the repo holds a newer build than the origin
//! serves -- reloading will not help yet) / UNKNOWN as the
//! default whenever a comparison cannot happen (offline,
//! airplane mode, fetch failure).

use yew::prelude::*;

const RAW_BUILD_INFO: &str =
    "https://raw.githubusercontent.com/sw-ml-study/sw-mlpl/main/pages/build-info.json";

/// The comparison verdict behind the badge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BundleStatus {
    /// The page runs what the origin serves (and, when the
    /// repo answered, nothing newer is committed).
    Fresh,
    /// The origin serves a NEWER build -- reload to get it.
    Stale,
    /// The repo's committed pages are newer than what the
    /// origin serves -- a deploy is pending; reload later.
    DeployPending,
    /// Could not tell (offline / fetch failed / parse failed).
    Unknown,
}

/// The build stamp (commit) inside a build-info.json body or a
/// `mlpl-build` meta content string.
#[must_use]
pub fn extract_commit(text: &str) -> Option<String> {
    if let Some(i) = text.find("\"commit\":\"") {
        let tail = &text[i + 10..];
        let end = tail.find('\"')?;
        return Some(tail[..end].to_string());
    }
    // Meta form: "<commit> <built_at>".
    let first = text.split_whitespace().next()?;
    (first.len() >= 7 && first.chars().all(|c| c.is_ascii_hexdigit())).then(|| first.to_string())
}

/// Three-way verdict; every missing side degrades honestly.
#[must_use]
pub fn verdict(running: Option<&str>, served: Option<&str>, repo: Option<&str>) -> BundleStatus {
    match (running, served) {
        (Some(r), Some(s)) if r != s => BundleStatus::Stale,
        (Some(r), Some(s)) => match repo {
            Some(g) if g != s && g != r => BundleStatus::DeployPending,
            _ => BundleStatus::Fresh,
        },
        _ => BundleStatus::Unknown,
    }
}

/// Mount-time probe feeding the badge.
#[hook]
pub fn use_bundle_status() -> BundleStatus {
    let status = use_state(|| BundleStatus::Unknown);
    {
        let status = status.clone();
        use_effect_with((), move |()| {
            wasm_bindgen_futures::spawn_local(async move {
                status.set(probe_status().await);
            });
            || ()
        });
    }
    *status
}

async fn probe_status() -> BundleStatus {
    let running = running_commit();
    let ts = js_sys::Date::now() as u64;
    let served = fetch_text(&format!("build-info.json?ts={ts}")).await;
    let repo = fetch_text(&format!("{RAW_BUILD_INFO}?ts={ts}")).await;
    verdict(
        running.as_deref(),
        served.as_deref().and_then(|t| extract_commit(t)).as_deref(),
        repo.as_deref().and_then(|t| extract_commit(t)).as_deref(),
    )
}

async fn fetch_text(url: &str) -> Option<String> {
    match gloo::net::http::Request::get(url).send().await {
        Ok(resp) if resp.ok() => resp.text().await.ok(),
        _ => None,
    }
}

/// The RUNNING page's commit, from the meta tag build-pages.sh
/// stamped into index.html.
fn running_commit() -> Option<String> {
    let doc = web_sys::window()?.document()?;
    let meta = doc.query_selector("meta[name='mlpl-build']").ok()??;
    let content = meta.get_attribute("content")?;
    extract_commit(&content)
}

/// The badge: green current, amber reload, blue deploy-pending,
/// neutral unknown.
#[function_component(StatusBadge)]
pub fn status_badge() -> Html {
    let status = use_bundle_status();
    let (class, label, title) = match status {
        BundleStatus::Fresh => (
            "status-badge status-badge-fresh",
            "current",
            "This page runs the newest deployed build.",
        ),
        BundleStatus::Stale => (
            "status-badge status-badge-stale",
            "update available -- reload",
            "The site serves a newer build than this page. Click to reload.",
        ),
        BundleStatus::DeployPending => (
            "status-badge status-badge-unknown",
            "deploy pending",
            "A newer build is committed; the site has not finished publishing it.",
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
