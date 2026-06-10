//! Run phase for `mlpl-repl`: turn a parsed `Config` into action.
//! `run` is the entire body of `main` after `args::parse`, split
//! into init (build the session) and execute (script or REPL).

use mlpl_eval::Environment;

use crate::args::{Config, Info};
use crate::run_interactive;
use crate::script_mode;
use crate::svg_out::SvgOut;

/// Per-run state assembled from `Config` by `init`.
struct Session {
    env: Environment,
    svg_out: SvgOut,
    trace: bool,
    verbose: bool,
}

/// Execute a parsed config: honor an info-only flag, hand off to
/// a connect subcommand, or init + run (script or interactive).
pub(crate) fn run(config: Config) {
    match config.info {
        Info::Version => return crate::version::print(),
        Info::HelpLong => return print!("{}", crate::version::help_long()),
        Info::HelpShort => return print!("{}", crate::version::help_short()),
        Info::None => {}
    }
    if crate::connect_repl::try_dispatch_args(&config.connect_args) {
        return;
    }
    let mut session = init(&config);
    execute(&config, &mut session);
}

/// Init phase: build the session (env + output sink) from config.
fn init(config: &Config) -> Session {
    let mut env = Environment::new();
    env.set_cli_args(config.script_args.clone());
    if let Some(dir) = &config.data_dir {
        env.set_data_dir(dir.clone());
    }
    if let Some(dir) = &config.exp_dir {
        env.set_exp_dir(dir.clone());
    }
    Session {
        svg_out: SvgOut::new(config.svg_out.clone()),
        trace: config.trace,
        verbose: config.verbose,
        env,
    }
}

/// Execute phase: run the script (then exit with its code) or
/// drop into the interactive REPL.
fn execute(config: &Config, s: &mut Session) {
    if config.babel_session {
        crate::babel_session::run_session(&mut s.env, &mut s.svg_out, s.trace, s.verbose);
        return;
    }
    if let Some(path) = &config.script {
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        });
        let code =
            script_mode::run_script(&content, &mut s.env, s.trace, s.verbose, &mut s.svg_out);
        std::process::exit(code);
    }
    run_interactive(&mut s.env, &mut s.svg_out);
}
