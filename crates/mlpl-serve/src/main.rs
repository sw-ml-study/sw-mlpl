//! `mlpl-serve` binary entry. Saga 21 step 001 +
//! Saga R1 step 003 (--peer flag for routing
//! `device("mlx") { ... }` blocks to peer servers).
//!
//! Thin shell around `mlpl_serve::server::run`. CLI
//! parsing is hand-rolled because the workspace
//! avoids `clap` for small one-purpose binaries.

use std::net::SocketAddr;
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
    eprintln!(
        "mlpl-serve listening on http://{} (auth={:?}, peers=[{}])",
        args.bind,
        args.auth,
        peer_summary.join(", ")
    );
    match runtime.block_on(run(args.bind, args.auth, peers)) {
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
    let mut it = iter.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => {
                let v = it.next().ok_or("--bind requires a value")?;
                bind = v
                    .parse()
                    .map_err(|e| format!("--bind: invalid SocketAddr {v:?}: {e}"))?;
            }
            "--auth" => {
                let v = it.next().ok_or("--auth requires a value")?;
                auth = match v.as_str() {
                    "required" => AuthMode::Required,
                    "disabled" => AuthMode::Disabled,
                    other => {
                        return Err(format!("--auth: expected required|disabled, got {other:?}"));
                    }
                };
            }
            "--peer" => {
                let v = it.next().ok_or("--peer requires a value")?;
                peer_pairs.push(parse_peer_arg(&v)?);
            }
            "--insecure-peers" => {
                insecure_peers = true;
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
    })
}

fn print_usage() {
    eprintln!(
        "usage: mlpl-serve [--bind <host:port>] [--auth <required|disabled>]\n\
         \x20            [--peer <device>=<url>]... [--insecure-peers]\n\
         \n\
         Defaults: --bind 127.0.0.1:6464  --auth required\n\
         Non-loopback binds (e.g. 0.0.0.0:...) require --auth required.\n\
         Non-loopback peer URLs require --insecure-peers (R1 deployment\n\
         is loopback-only by default).\n\
         \n\
         Example: --peer mlx=http://localhost:6465 routes device(\"mlx\")\n\
         {{ ... }} blocks to a mlpl-mlx-serve peer running on the same host."
    );
}
