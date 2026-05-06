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
use mlpl_serve::peers::{build_registry, parse_peer_arg};
use mlpl_serve::server::run;

const DEFAULT_BIND: &str = "127.0.0.1:6464";

struct Args {
    bind: SocketAddr,
    auth: AuthMode,
    peer_pairs: Vec<(String, String)>,
    insecure_peers: bool,
    static_dir: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    let peers = match build_registry(args.peer_pairs, args.insecure_peers) {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to start tokio runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let peer_summary: Vec<String> = peers
        .iter()
        .map(|(d, p)| format!("{d}={}", p.url))
        .collect();
    let static_summary = match &args.static_dir {
        Some(p) => format!(", web=http://{}/sw-mlpl/ ({})", args.bind, p.display()),
        None => String::new(),
    };
    eprintln!(
        "mlpl-serve listening on http://{} (auth={:?}, peers=[{}]){static_summary}",
        args.bind,
        args.auth,
        peer_summary.join(", ")
    );
    match runtime.block_on(run(args.bind, args.auth, peers, args.static_dir)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args<I: IntoIterator<Item = String>>(iter: I) -> Result<Args, String> {
    let mut bind: SocketAddr = DEFAULT_BIND.parse().expect("default bind parses");
    let mut auth = AuthMode::Required;
    let mut peer_pairs: Vec<(String, String)> = Vec::new();
    let mut insecure_peers = false;
    let mut static_dir: Option<PathBuf> = None;
    let mut it = iter.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => {
                let v = it.next().ok_or("--bind requires a value")?;
                bind = v
                    .parse()
                    .map_err(|e| format!("--bind: invalid SocketAddr {v:?}: {e}"))?;
            }
            "--auth" => auth = parse_auth(&it.next().ok_or("--auth requires a value")?)?,
            "--peer" => {
                let v = it.next().ok_or("--peer requires a value")?;
                peer_pairs.push(parse_peer_arg(&v)?);
            }
            "--insecure-peers" => {
                insecure_peers = true;
            }
            "--static-dir" => {
                static_dir = Some(parse_static_dir(
                    it.next().ok_or("--static-dir requires a value")?,
                )?);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        bind,
        auth,
        peer_pairs,
        insecure_peers,
        static_dir,
    })
}

fn parse_auth(value: &str) -> Result<AuthMode, String> {
    match value {
        "required" => Ok(AuthMode::Required),
        "disabled" => Ok(AuthMode::Disabled),
        other => Err(format!("--auth: expected required|disabled, got {other:?}")),
    }
}

fn parse_static_dir(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(&value);
    if !path.is_dir() {
        return Err(format!("--static-dir: not a directory: {value}"));
    }
    Ok(path)
}

fn print_usage() {
    eprintln!(
        "usage: mlpl-serve [--bind <host:port>] [--auth <required|disabled>]\n\
         \x20            [--peer <device>=<url>]... [--insecure-peers]\n\
         \x20            [--static-dir <path>]\n\
         \n\
         Defaults: --bind 127.0.0.1:6464  --auth required\n\
         Non-loopback binds (e.g. 0.0.0.0:...) require --auth required.\n\
         Non-loopback peer URLs require --insecure-peers (R1 deployment\n\
         is loopback-only by default).\n\
         \n\
         --static-dir <path> mounts a static-file tree at /sw-mlpl/ on\n\
         the same listener. The directory is expected to contain the\n\
         output of `./scripts/build-pages.sh` (i.e. the `pages/`\n\
         build with `--public-url /sw-mlpl/`). With this flag set,\n\
         http://<bind>/sw-mlpl/ serves the web REPL on the same\n\
         origin as the /v1 API -- no CORS plumbing required for the\n\
         WASM client to call back.\n\
         \n\
         Examples:\n\
         \x20 --peer mlx=http://localhost:6465\n\
         \x20     routes device(\"mlx\") {{ ... }} blocks to a peer.\n\
         \x20 --static-dir ./pages\n\
         \x20     also serves the web UI at http://127.0.0.1:6464/sw-mlpl/."
    );
}
