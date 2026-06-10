//! Connect-mode `:status` self-test: probe the connected server's
//! `GET /v1/devices` + `GET /v1/stats` and render a report -- the
//! backend is up (it responded), which devices it has (cpu/cuda/mlx),
//! live CPU/RAM/GPU/VRAM, and whether Ollama is wired for `:ask`.
//! WASM-only, like the other connect fetches.

#![cfg(target_arch = "wasm32")]

use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::render::status_lines;

use crate::eval::ResultCb;

/// Async `:status`: probe the server and fire `on_result` with a
/// human-readable report (or an `error:` message when it does not
/// respond -- which is itself the "backend is down" signal).
pub fn fetch_status(base_url: String, ollama_ready: bool, on_result: ResultCb) {
    wasm_bindgen_futures::spawn_local(async move {
        on_result(status_report(&base_url, ollama_ready).await);
    });
}

/// `GET <url>` returning the parsed JSON body, or `None` on any
/// transport / status / decode failure.
async fn get_json(url: &str) -> Option<serde_json::Value> {
    let resp = gloo::net::http::Request::get(url).send().await.ok()?;
    resp.ok().then_some(())?;
    resp.json::<serde_json::Value>().await.ok()
}

/// Comma-joined device list from a `/v1/devices` body, or `"unknown"`.
fn device_list(devices: Option<&serde_json::Value>) -> String {
    devices
        .and_then(|d| d.get("devices"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "unknown".to_string())
}

async fn status_report(base_url: &str, ollama_ready: bool) -> String {
    let base = base_url.trim_end_matches('/');
    let devices = get_json(&format!("{base}/v1/devices")).await;
    let Some(stats) = get_json(&format!("{base}/v1/stats")).await else {
        return format!("error: status: connect server at {base} did not respond -- is it up?");
    };
    let snap: Snapshot = serde_json::from_value(stats).unwrap_or_default();
    let ollama = if ollama_ready {
        "configured (:ask ready)"
    } else {
        "not configured"
    };
    let mut out = vec![
        format!("Status: connected to {base}"),
        format!("  devices : {}", device_list(devices.as_ref())),
    ];
    out.extend(status_lines(&snap));
    out.push(format!("  Ollama  : {ollama}"));
    out.join("\n")
}
