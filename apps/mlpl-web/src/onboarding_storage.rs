pub fn should_show_splash(dismissed: bool) -> bool {
    !dismissed
}

pub fn should_show_whats_new(last_seen: Option<&str>, current: &str) -> bool {
    matches!(last_seen, Some(v) if v != current)
}

pub fn read_splash_dismissed() -> bool {
    storage()
        .and_then(|s| s.get_item("mlpl_splash_dismissed").ok()?)
        .is_some()
}

pub fn write_splash_dismissed() {
    if let Some(s) = storage() {
        let _ = s.set_item("mlpl_splash_dismissed", "1");
    }
}

pub fn read_last_seen_version() -> Option<String> {
    storage()?.get_item("mlpl_last_seen_version").ok()?
}

pub fn write_last_seen_version(version: &str) {
    if let Some(s) = storage() {
        let _ = s.set_item("mlpl_last_seen_version", version);
    }
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}
