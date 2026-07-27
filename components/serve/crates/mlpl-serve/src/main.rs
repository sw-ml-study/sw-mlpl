//! `mlpl-serve` binary entry. Saga 21 step 001 +
//! Saga R1 step 003 (--peer flag for routing
//! `device("mlx") { ... }` blocks to peer servers).
//!
//! Thin shell around `mlpl_serve::server::run`. CLI
//! parsing is hand-rolled because the workspace
//! avoids `clap` for small one-purpose binaries.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use mlpl_serve::auth::AuthMode;
use mlpl_serve::config::{RunConfig, ServeConfig, resolve_ollama};
use mlpl_serve::peers::{build_registry, parse_peer_arg};
use mlpl_serve::server::run;

pub(crate) const DEFAULT_BIND: &str = "127.0.0.1:6464";

pub(crate) struct Args {
    pub(crate) bind: SocketAddr,
    pub(crate) auth: AuthMode,
    pub(crate) peer_pairs: Vec<(String, String)>,
    pub(crate) insecure_peers: bool,
    pub(crate) static_dir: Option<PathBuf>,
    pub(crate) tls_cert: Option<PathBuf>,
    pub(crate) tls_key: Option<PathBuf>,
    self_signed: bool,
    /// Saga 21.5 step 006: when set, the router is wrapped in
    /// a `tower-http` CORS layer that allows browsers on this
    /// origin to reach `/v1/*`. Required for the connect-mode
    /// `apps/mlpl-web` running on a different origin (e.g.
    /// `https://sw-ml-study.github.io/sw-mlpl/`).
    pub(crate) cors_allow: Option<String>,
    /// Saga 21.5 step 010: when set, every successful `/eval`
    /// flushes the slim per-session state (token + timestamps +
    /// variable bindings) to this JSON file. Startup reads the
    /// same file so a restart picks up the prior session map.
    /// Absent means in-memory-only (legacy behavior).
    pub(crate) persist: Option<PathBuf>,
    /// Phase 0 (local-gpu-agentic): override the default Ollama
    /// host (`--ollama-host`), default model (`--ollama-model`),
    /// and add allow-listed hosts the server may reach
    /// (`--ollama-allow`, repeatable). The resolved host is always
    /// allow-listed. `OLLAMA_HOST` env is the host fallback below
    /// the flag.
    pub(crate) ollama_host: Option<String>,
    pub(crate) ollama_model: Option<String>,
    pub(crate) ollama_allow: Vec<String>,
}

mod args;

fn main() -> ExitCode {
    let args = match args::parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match run_main(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

/// Cap rayon's global thread pool at `cores - 2` (min 1) so the array
/// crate's parallel matmul can never saturate every core. Without this,
/// a CPU-heavy eval (e.g. an attention + vocab-projection pretrain)
/// starves the async runtime and concurrent requests -- notably the
/// telemetry panel's `GET /v1/stats` polls -- queue until the eval ends.
/// Reserving 2 cores keeps HTTP serving responsive during the eval.
/// Best-effort: a no-op if the global pool is already initialized.
fn reserve_cores_for_serving() {
    let total = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    let workers = total.saturating_sub(2).max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global();
}

/// GPU workspace split, S2 (cycle-break): register this build's GPU
/// optimizer step once at startup so `device("cuda"|"mlx") { }` blocks
/// (forwarded from the connect-mode web demos) run on the GPU in-process.
/// `mlpl-eval::Environment::new` reads the registered step instead of
/// constructing it -- which is what lets the compute move to sibling
/// crates in S3/S4. A no-op on a CPU-only build.
fn register_gpu_step() {
    // CUDA (S3): the step lives in the sibling mlpl-cuda-eval crate.
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    mlpl_eval::register_gpu_step(mlpl_cuda_eval::gpu_step());
    // MLX (S4): the step lives in the sibling mlpl-mlx-eval crate.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    mlpl_eval::register_gpu_step(mlpl_mlx_eval::gpu_step());
}

/// Build the tokio runtime, resolve TLS + peers, and dispatch to
/// `server::run`. Extracted from `main` so the body stays under
/// the sw-checklist 50-line LOC budget while still surfacing
/// every error to stderr.
fn run_main(args: Args) -> Result<(), String> {
    register_gpu_step();
    reserve_cores_for_serving();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to start tokio runtime: {e}"))?;
    let tls = runtime.block_on(build_tls(
        args.tls_cert.as_deref(),
        args.tls_key.as_deref(),
        args.self_signed,
    ))?;
    let peers = build_registry(args.peer_pairs.clone(), args.insecure_peers)?;
    print_banner(&args, &peers, tls.is_some());
    let (bind, auth, serve) = serve_config(args);
    let cfg = RunConfig {
        addr: bind,
        auth_mode: auth,
        peers,
        tls,
        serve,
    };
    runtime.block_on(run(cfg)).map_err(|e| format!("{e}"))
}

/// Resolve the three TLS-related flags into an
/// `Option<RustlsConfig>`. Mutually exclusive: pass either
/// `--tls-cert` + `--tls-key` together, or `--self-signed`,
/// or neither (HTTP). Logs the SHA-256 fingerprint when
/// generating a self-signed cert so the user can verify
/// what their browser is asked to trust.
async fn build_tls(
    cert: Option<&std::path::Path>,
    key: Option<&std::path::Path>,
    self_signed: bool,
) -> Result<mlpl_serve::server::TlsConfig, String> {
    match (cert, key, self_signed) {
        (None, None, false) => Ok(None),
        (Some(c), Some(k), false) => Ok(Some(mlpl_serve::tls::from_pem_files(c, k).await?)),
        (None, None, true) => {
            let (config, fingerprint) = mlpl_serve::tls::self_signed_loopback().await?;
            eprintln!("mlpl-serve self-signed cert SHA-256 fingerprint:\n  {fingerprint}");
            Ok(Some(config))
        }
        (Some(_), None, _) | (None, Some(_), _) => {
            Err("--tls-cert and --tls-key must be set together".into())
        }
        (Some(_), Some(_), true) => {
            Err("--self-signed is exclusive with --tls-cert/--tls-key; pick one path".into())
        }
    }
}

fn print_banner(args: &Args, peers: &mlpl_serve::peers::PeerRegistry, tls_set: bool) {
    let scheme = if tls_set { "https" } else { "http" };
    let peers_str = peers
        .iter()
        .map(|(d, p)| format!("{d}={}", p.url))
        .collect::<Vec<_>>()
        .join(", ");
    let web = match &args.static_dir {
        Some(p) => format!(", web={scheme}://{}/sw-mlpl/ ({})", args.bind, p.display()),
        None => String::new(),
    };
    eprintln!(
        "mlpl-serve listening on {scheme}://{} (auth={:?}, peers=[{peers_str}]){web}",
        args.bind, args.auth
    );
}

pub(crate) fn print_usage() {
    eprintln!(
        "usage: mlpl-serve [--bind <host:port>] [--auth <required|disabled>]\n\
         \x20            [--peer <device>=<url>]... [--insecure-peers]\n\
         \x20            [--static-dir <path>] [--cors-allow <origin>[,<origin>...]]\n\
         \x20            [--tls-cert <cert.pem> --tls-key <key.pem> | --self-signed]\n\
         \n\
         Defaults: --bind 127.0.0.1:6464  --auth required\n\
         Non-loopback binds (e.g. 0.0.0.0:...) require --auth required.\n\
         Non-loopback peer URLs require --insecure-peers (R1 deployment\n\
         is loopback-only by default).\n\
         \n\
         --static-dir <path> mounts a static-file tree at /sw-mlpl/ on\n\
         the same listener. The directory is expected to contain the\n\
         output of `./scripts/build-pages.sh`. With this flag set,\n\
         <scheme>://<bind>/sw-mlpl/ serves the web REPL on the same\n\
         origin as the /v1 API -- no CORS plumbing required for the\n\
         WASM client to call back.\n\
         \n\
         TLS modes (mutually exclusive):\n\
         \x20 --tls-cert <cert.pem> --tls-key <key.pem>\n\
         \x20     Production-style PEM cert + key. The pair must\n\
         \x20     be set together.\n\
         \x20 --self-signed\n\
         \x20     Generate an in-memory self-signed cert covering\n\
         \x20     `localhost`, `127.0.0.1`, `::1`. Browser shows a\n\
         \x20     warning the first time; click Advanced -> Proceed.\n\
         \x20     The SHA-256 fingerprint is printed at startup so\n\
         \x20     you can verify what you accepted.\n\
         \n\
         Examples:\n\
         \x20 --peer mlx=http://localhost:6465\n\
         \x20     routes device(\"mlx\") {{ ... }} blocks to a peer.\n\
         \x20 --static-dir ./pages --self-signed\n\
         \x20     serves the web UI at https://127.0.0.1:6464/sw-mlpl/\n\
         \x20     -- compatible with browsers that enforce HTTPS-First."
    );
}

/// Fold the parsed CLI flags (+ `OLLAMA_HOST`) into the bind
/// address, auth mode, and `ServeConfig`. Pure except for the
/// env read; extracted from `run_main` for the LOC budget.
fn serve_config(args: Args) -> (std::net::SocketAddr, AuthMode, ServeConfig) {
    let env_host = std::env::var("OLLAMA_HOST").ok();
    let Args {
        bind,
        auth,
        static_dir,
        cors_allow,
        persist,
        ollama_host,
        ollama_model,
        ollama_allow,
        ..
    } = args;
    let ollama = resolve_ollama(ollama_host, env_host, ollama_model, ollama_allow);
    let serve = ServeConfig {
        static_dir,
        cors_origin: cors_allow,
        persist_path: persist,
        ollama,
    };
    (bind, auth, serve)
}
