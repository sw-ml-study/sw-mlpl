//! `mlpl-repl --babel-session`: a persistent session for the Org-babel
//! backend (`ob-mlpl`). Reads block after block from stdin -- each
//! terminated by a sentinel line -- and runs each through the SAME path
//! as `-f` ([`crate::script_mode::run_script`]) against ONE long-lived
//! [`Environment`]. After each block's output it prints a sentinel line
//! so the caller can frame the result.
//!
//! Why: ob-mlpl used to model a `:session` as the whole accumulated
//! program re-run via `mlpl-repl -f` for every block -- O(n^2) over a
//! literate document, which made publishing a heavy page (e.g. a base
//! pretrain) painfully slow. Persisting the env makes each block O(1)
//! in prior blocks, and reusing `run_script` keeps the per-block output
//! byte-identical to the old path.

use std::io::{self, BufRead, Write};

use mlpl_eval::Environment;

use crate::script_mode::run_script;
use crate::svg_out::SvgOut;

/// Sentinel the caller writes after a block to request evaluation.
const BLOCK_EOF: &str = "__MLPL_BABEL_EOF__";
/// Sentinel this mode writes after a block's output to delimit it.
const BLOCK_DONE: &str = "__MLPL_BABEL_DONE__";

/// Run the persistent block loop until stdin closes. Each block is the
/// lines received since the previous `BLOCK_EOF`; it is evaluated in
/// `env` (state persists across blocks) and its output is followed by a
/// flushed `BLOCK_DONE` line.
pub(crate) fn run_session(env: &mut Environment, svg_out: &mut SvgOut, trace: bool, verbose: bool) {
    let stdin = io::stdin();
    let mut block = String::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line == BLOCK_EOF {
            run_script(&block, env, trace, verbose, svg_out);
            block.clear();
            println!("{BLOCK_DONE}");
            io::stdout().flush().ok();
        } else {
            block.push_str(&line);
            block.push('\n');
        }
    }
}
