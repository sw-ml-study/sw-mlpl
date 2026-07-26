//! Saga 72 step 003: argument parsing extracted from main.rs
//! to keep its function count under the sw-checklist module-fn cap.

use std::path::PathBuf;

use crate::{Args, AuthMode, DEFAULT_BIND, parse_peer_arg, print_usage};

pub(crate) fn parse_args<I: IntoIterator<Item = String>>(iter: I) -> Result<Args, String> {
    let mut acc = Args {
        bind: DEFAULT_BIND.parse().expect("default bind parses"),
        auth: AuthMode::Required,
        peer_pairs: Vec::new(),
        insecure_peers: false,
        static_dir: None,
        tls_cert: None,
        tls_key: None,
        self_signed: false,
        cors_allow: None,
        persist: None,
        ollama_host: None,
        ollama_model: None,
        ollama_allow: Vec::new(),
    };
    let mut it = iter.into_iter();
    while let Some(arg) = it.next() {
        apply_one_arg(&arg, &mut it, &mut acc)?;
    }
    Ok(acc)
}

fn apply_one_arg<I: Iterator<Item = String>>(
    arg: &str,
    it: &mut I,
    acc: &mut Args,
) -> Result<(), String> {
    let need = |it: &mut I, flag: &str| it.next().ok_or_else(|| format!("{flag} requires a value"));
    match arg {
        "--bind" => {
            let v = need(it, "--bind")?;
            acc.bind = v
                .parse()
                .map_err(|e| format!("--bind: invalid SocketAddr {v:?}: {e}"))?;
        }
        "--auth" => {
            acc.auth = match need(it, "--auth")?.as_str() {
                "required" => AuthMode::Required,
                "disabled" => AuthMode::Disabled,
                o => return Err(format!("--auth: expected required|disabled, got {o:?}")),
            };
        }
        "--peer" => acc.peer_pairs.push(parse_peer_arg(&need(it, "--peer")?)?),
        other => return apply_extra_arg(other, it, acc),
    }
    Ok(())
}

/// Second half of the flag table: TLS, CORS, persistence, and
/// Ollama flags, plus help/unknown. Split from `apply_one_arg`
/// for the function-LOC budget.
fn apply_extra_arg<I: Iterator<Item = String>>(
    arg: &str,
    it: &mut I,
    acc: &mut Args,
) -> Result<(), String> {
    let need = |it: &mut I, flag: &str| it.next().ok_or_else(|| format!("{flag} requires a value"));
    match arg {
        "--insecure-peers" => acc.insecure_peers = true,
        "--static-dir" => acc.static_dir = Some(parse_static_dir(need(it, "--static-dir")?)?),
        "--tls-cert" => acc.tls_cert = Some(PathBuf::from(need(it, "--tls-cert")?)),
        "--tls-key" => acc.tls_key = Some(PathBuf::from(need(it, "--tls-key")?)),
        "--self-signed" => acc.self_signed = true,
        "--cors-allow" => acc.cors_allow = Some(need(it, "--cors-allow")?),
        "--persist" => acc.persist = Some(PathBuf::from(need(it, "--persist")?)),
        "--ollama-host" => acc.ollama_host = Some(need(it, "--ollama-host")?),
        "--ollama-model" => acc.ollama_model = Some(need(it, "--ollama-model")?),
        "--ollama-allow" => acc.ollama_allow.push(need(it, "--ollama-allow")?),
        "-h" | "--help" => {
            print_usage();
            std::process::exit(0);
        }
        other => return Err(format!("unknown argument {other:?}")),
    }
    Ok(())
}

fn parse_static_dir(value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(&value);
    if !path.is_dir() {
        return Err(format!("--static-dir: not a directory: {value}"));
    }
    Ok(path)
}
