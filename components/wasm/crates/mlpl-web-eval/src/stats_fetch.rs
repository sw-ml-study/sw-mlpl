//! Connect-mode `:status` self-test: probe the connected server's
//! `GET /v1/devices` + `GET /v1/stats` and render a numbered BACKEND
//! LIST -- the backend is up (it responded), which devices + GPU types
//! it has, live CPU/RAM/GPU/VRAM, and whether Ollama is wired for
//! `:ask`. Today the list has length 1 (one connected mlpl-serve); a
//! future proxy aggregating several backends just lengthens it. Also
//! hosts the `:reset` POST. WASM-only, like the other connect fetches.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use mlpl_monitor_types::Snapshot;
use mlpl_monitor_types::render::snapshot_lines;

use crate::eval::ResultCb;

thread_local! {
    // Set by `:reset` (the prompt), consumed by the next submitted
    // line (the y/N answer). One-shot: a `:reset` that is not followed
    // by a confirmation never fires.
    static RESET_ARMED: RefCell<bool> = const { RefCell::new(false) };
}

/// Arm the `:reset` confirmation: the next submitted line is treated as
/// the y/N answer.
pub fn arm_reset() {
    RESET_ARMED.with(|c| *c.borrow_mut() = true);
}

/// Consume the armed flag, returning whether a `:reset` confirmation
/// was pending (and clearing it either way).
pub fn take_reset_armed() -> bool {
    RESET_ARMED.with(|c| c.replace(false))
}

/// Async `:status`: probe the server and fire `on_result` with a
/// human-readable backend-list report (or an `error:` message when it
/// does not respond -- itself the "backend is down" signal).
pub fn fetch_status(base_url: String, ollama_ready: bool, on_result: ResultCb) {
    wasm_bindgen_futures::spawn_local(async move {
        on_result(status_report(&base_url, ollama_ready).await);
    });
}

/// Async `:reset`: POST `/v1/reset` to cancel all in-flight evals on
/// the connected server (the UI-driven recovery from a hung/orphaned
/// demo). Fires `on_result` with how many sessions were signalled.
pub fn fetch_reset(base_url: String, on_result: ResultCb) {
    wasm_bindgen_futures::spawn_local(async move {
        let url = format!("{}/v1/reset", base_url.trim_end_matches('/'));
        let out = match gloo::net::http::Request::post(&url).send().await {
            Ok(r) => r
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|b| b.get("cancelled").and_then(serde_json::Value::as_u64))
                .map_or_else(
                    || "reset: server responded (count unknown)".to_string(),
                    |n| {
                        format!(
                            "reset: signalled {n} in-flight session(s) to cancel; backend freed."
                        )
                    },
                ),
            Err(e) => format!("error: reset: {e}"),
        };
        on_result(out);
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
    let host = devices
        .as_ref()
        .and_then(|d| d.get("hostname"))
        .and_then(serde_json::Value::as_str);
    let head = host.map_or_else(|| format!("[1] {base}"), |h| format!("[1] {h}  ({base})"));
    let mut out = vec![
        "Status: 1 backend connected".to_string(),
        head,
        format!("    devices : {}", device_list(devices.as_ref())),
    ];
    out.extend(snapshot_lines(&snap));
    out.push(format!("    Ollama  : {ollama}"));
    out.join("\n")
}
