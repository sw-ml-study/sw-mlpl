//! Server configuration value types, grouped into nested composite
//! structs (`OllamaConfig` inside `ServeConfig` inside `RunConfig`)
//! so `run()` and `build_app_with_peers_cors()` take a single config
//! argument instead of a long positional list -- keeping them under
//! the sw-checklist argument + LOC budgets. Phase 0 of the
//! local-gpu-agentic saga.

use std::net::SocketAddr;
use std::path::PathBuf;

use crate::auth::AuthMode;
use crate::peers::PeerRegistry;
use crate::server::TlsConfig;

/// Default Ollama endpoint + model when nothing is configured.
const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL: &str = "qwen2.5:0.5b";

/// Server-owned Ollama settings: the default host + model the web
/// `:ask` falls back to, plus the allow-list of hosts the server is
/// permitted to reach (mirrors the connect-mode CORS allow-list --
/// no arbitrary outbound network from a browser-triggered call).
#[derive(Clone, Debug)]
pub struct OllamaConfig {
    pub default_host: String,
    pub default_model: String,
    pub allowed_hosts: Vec<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            default_host: DEFAULT_OLLAMA_HOST.to_string(),
            default_model: DEFAULT_OLLAMA_MODEL.to_string(),
            allowed_hosts: vec![DEFAULT_OLLAMA_HOST.to_string()],
        }
    }
}

impl OllamaConfig {
    /// Is the server permitted to reach `host`? Only the resolved
    /// default host plus explicitly allow-listed hosts qualify.
    /// Trailing slashes are ignored on both sides.
    pub fn is_allowed(&self, host: &str) -> bool {
        let h = host.trim_end_matches('/');
        self.allowed_hosts
            .iter()
            .any(|a| a.trim_end_matches('/') == h)
    }
}

/// Resolve the effective Ollama config, precedence highest-first:
/// the CLI `--ollama-host` flag, the `OLLAMA_HOST` env value, then
/// the built-in default. `env_host` is passed in (not read here) so
/// resolution stays pure + unit-testable without env races. The
/// resolved default host is always allow-listed, plus any
/// `--ollama-allow` hosts.
pub fn resolve_ollama(
    flag_host: Option<String>,
    env_host: Option<String>,
    flag_model: Option<String>,
    extra_allow: Vec<String>,
) -> OllamaConfig {
    let d = OllamaConfig::default();
    let host = flag_host
        .or(env_host)
        .filter(|s| !s.is_empty())
        .unwrap_or(d.default_host);
    let model = flag_model
        .filter(|s| !s.is_empty())
        .unwrap_or(d.default_model);
    let mut allowed = vec![host.clone()];
    allowed.extend(extra_allow.into_iter().filter(|s| !s.is_empty()));
    OllamaConfig {
        default_host: host,
        default_model: model,
        allowed_hosts: allowed,
    }
}

/// Router-construction config: the optional static dir, CORS
/// origin, persistence path, and the Ollama settings. Bundled so
/// the builder takes one struct instead of four positional args.
#[derive(Clone, Debug, Default)]
pub struct ServeConfig {
    pub static_dir: Option<PathBuf>,
    pub cors_origin: Option<String>,
    pub persist_path: Option<PathBuf>,
    pub ollama: OllamaConfig,
}

/// Everything `run()` needs: bind address, auth mode, peer
/// registry, optional TLS, and the router config -- one argument
/// instead of seven.
pub struct RunConfig {
    pub addr: SocketAddr,
    pub auth_mode: AuthMode,
    pub peers: PeerRegistry,
    pub tls: TlsConfig,
    pub serve: ServeConfig,
}
