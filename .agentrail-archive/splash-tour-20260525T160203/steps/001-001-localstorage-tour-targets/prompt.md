Step 001: localStorage helper + data-tour-target IDs.

New module onboarding_storage.rs (~60 LOC, 3 functions) wrapping web_sys::Storage for two keys: mlpl_splash_dismissed (bool) and mlpl_last_seen_version (String). Pure predicates should_show_splash(dismissed) and should_show_whats_new(last_seen, current). Unit tests for both.

Add data-tour-target attributes to six target elements: repl-input, demo-select, tab-tutorial, tab-paths, help-btn, completion-popup in components.rs and render_shell_header.rs.

Add Storage and DomRect to web-sys features in apps/mlpl-web/Cargo.toml.

Pages rebuild required.