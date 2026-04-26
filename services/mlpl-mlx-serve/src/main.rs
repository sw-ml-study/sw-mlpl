//! `mlpl-mlx-serve` binary entry. Saga R1 step 001.
//!
//! Thin shell around `mlpl_mlx_serve::server::run`.
//! Mirrors `mlpl-serve`'s arg-parser shape; default
//! port is 6465 (orchestrator's mlpl-serve defaults
//! to 6464 -- shift one so the two can run on the
//! same host without conflict).

use std::net::SocketAddr;
use std::process::ExitCode;

use mlpl_mlx_serve::server::run;
use mlpl_serve::auth::AuthMode;

const DEFAULT_BIND: &str = "127.0.0.1:6465";

struct Args {
    bind: SocketAddr,
    auth: AuthMode,
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
    eprintln!(
        "mlpl-mlx-serve listening on http://{} (auth={:?}, device=mlx)",
        args.bind, args.auth
    );
    match runtime.block_on(run(args.bind, args.auth)) {
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
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args { bind, auth })
}

fn print_usage() {
    eprintln!(
        "usage: mlpl-mlx-serve [--bind <host:port>] [--auth <required|disabled>]\n\
         \n\
         Defaults: --bind 127.0.0.1:6465  --auth required\n\
         Non-loopback binds (e.g. 0.0.0.0:...) require --auth required.\n\
         \n\
         The orchestrator (mlpl-serve) defaults to :6464; this server\n\
         defaults to :6465 so both can run on the same host."
    );
}
