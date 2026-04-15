//! `mlpl!` procedural macro (compile-to-rust step 005).
//!
//! Wraps `mlpl-lower-rs` so a block of MLPL source can sit inside a
//! Rust program:
//!
//! ```ignore
//! use mlpl::mlpl;
//!
//! let r = mlpl! {
//!     x : [seq] = iota(5);
//!     reduce_add(x, "seq")
//! };
//! println!("{}", r.data()[0]);  // 10
//! ```
//!
//! Paths emitted by the lowering go through the `mlpl` facade crate
//! (`::mlpl::__rt::...`), not `::mlpl_rt::` directly, so end users
//! only need `mlpl` in their `[dependencies]`. The facade re-exports
//! the runtime under a hidden `__rt` alias.
//!
//! Error handling: parse and lower errors surface as
//! `syn::Error::to_compile_error()` so rustc shows `error[E0605]`
//! at the macro's invocation site with the MLPL error string in the
//! message. Span-preservation to individual MLPL tokens is a
//! follow-up -- the current pass uses `Span::call_site()` for every
//! error.

use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream;
use quote::quote;

/// `mlpl!` -- embed MLPL source inside Rust code, compiled to
/// calls into `mlpl-rt` at Rust compile time.
///
/// Multi-statement programs use semicolons as separators:
/// `mlpl! { x = iota(5); reduce_add(x) }`. Statements that would
/// normally end at a newline in the REPL must be semicolon-
/// terminated inside the macro because `proc_macro` strips
/// newlines.
#[proc_macro]
pub fn mlpl(input: ProcTokenStream) -> ProcTokenStream {
    let input: TokenStream = input.into();
    match expand(&input) {
        Ok(ts) => ts.into(),
        Err(msg) => {
            // compile_error!(msg) at the macro call site.
            let lit = proc_macro2::Literal::string(&msg);
            quote! { ::core::compile_error!(#lit) }.into()
        }
    }
}

/// Shared expansion entry point. Stringifies the input tokens,
/// runs them through the MLPL lexer + parser + lowering, and
/// returns either the lowered TokenStream or an error message
/// suitable for `compile_error!`.
fn expand(input: &TokenStream) -> Result<TokenStream, String> {
    // `TokenStream::to_string` may insert `\n` for display-formatting
    // reasons (bracketed groups, long-line wrapping). Our parser uses
    // `\n` as a top-level statement separator, so any stray ones
    // inside an expression are misread as unexpected tokens. Users
    // must use `;` as the statement separator inside the macro; that
    // lets us rewrite every `\n` to a space here safely.
    let src: String = input
        .to_string()
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    let tokens = mlpl_parser::lex(&src).map_err(|e| format!("mlpl: lex error: {e}"))?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| format!("mlpl: parse error: {e}"))?;
    let cfg = mlpl_lower_rs::LowerConfig {
        rt_path: quote! { ::mlpl::__rt },
    };
    mlpl_lower_rs::lower_with_config(&stmts, &cfg).map_err(|e| format!("mlpl: {e}"))
}
