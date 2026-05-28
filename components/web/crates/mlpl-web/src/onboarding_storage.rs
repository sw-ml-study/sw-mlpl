fn ls() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

pub fn read_splash_dismissed() -> bool {
    ls().and_then(|s| s.get_item("mlpl_splash_dismissed").ok()?)
        .is_some()
}

pub fn write_splash_dismissed() {
    if let Some(s) = ls() {
        let _ = s.set_item("mlpl_splash_dismissed", "1");
    }
}

pub fn read_last_seen_version() -> Option<String> {
    ls()?.get_item("mlpl_last_seen_version").ok()?
}

pub fn write_last_seen_version(version: &str) {
    if let Some(s) = ls() {
        let _ = s.set_item("mlpl_last_seen_version", version);
    }
}
