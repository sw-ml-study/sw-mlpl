//! FnCall lowering: a builtin registry + dispatcher (chain of
//! responsibility). `REGISTRY` DESCRIBES each builtin (names, arity,
//! and an `Emit` tag for its lowering shape); `lower_fncall`
//! DISPATCHES a name+arity to the first matching spec; `emit`
//! interprets the tag -- inlining the simple shapes and delegating
//! the few complex ones (`bit`/`check`/`label`/`matmul`). Adding a
//! builtin is adding a `Spec` row (plus an `Emit` variant only if its
//! shape is genuinely new) -- no growing dispatch match. Static label
//! inference (`labels_of`, `extract_label_list`) lives here too.

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{Ctx, LowerError, lower_expr};

/// How many arguments a spec accepts. `Any` defers the arity check to
/// the runtime (the bit ops validate arity inside their dispatch).
enum Arity {
    Exactly(usize),
    Any,
}

/// The lowering shape of a builtin -- the "config" the dispatcher
/// interprets. Simple shapes are emitted inline by `emit`; the last
/// four delegate to a dedicated handler.
enum Emit {
    /// `rt::<name>(&a0)` -- the runtime fn shares the builtin's name.
    UnaryRt,
    /// `rt::iota((a0).data()[0] as usize)`.
    Iota,
    /// `reshape(a0, dims)` with a runtime dim vector.
    Reshape,
    /// `reduce_add(a0, axis)` -- the per-axis reduce primitive.
    ReduceAxis,
    /// `rt::<name>(&<cval a0>)` -- a CVal-position arg (write_stdout/arg).
    CvalIo,
    /// `rt::cli_args()`.
    Args,
    /// `rt::read_stdin()` -- reads all of stdin to a `CVal::Str`.
    ReadStdin,
    /// `rt::exit(&<cval a0>)` -- ends the process (returns `!`); the
    /// arg is a scalar exit code in CVal position.
    Exit,
    /// `ok`/`err` -> `CVal::result(name == "ok", <cval a0>)`.
    Result,
    /// `read_bytes(path, offset, length)` -- path in CVal position,
    /// offset + length as `DenseArray` scalars.
    ReadRange,
    /// `write_bytes`/`append_bytes` -- `rt::<name>(&cval path, &cval
    /// bytes)`, both args in CVal position.
    WriteBytes,
    Label,
    Matmul,
    Check,
    /// Pure name/args runtime dispatch to `rt::<disp>_try_call`: bit
    /// ops / `take` / `floor` (all `DenseArray -> DenseArray`).
    BitCall,
    ArrayCall,
    MathCall,
    /// `gt`/`lt`/`eq` -- elementwise comparison, `rt::ApplyBinopExt::
    /// apply_binop(&a0, &a1, |x, y| ...)` yielding a `DenseArray` of
    /// 1.0/0.0, mirroring the arithmetic binop lowering.
    Cmp,
    /// `type_of(v)` -- the value's kind as a `CVal::Str` (value_kind
    /// parity), and `equal(a, b)` -- structural CVal equality as a
    /// scalar `DenseArray` 1.0/0.0 (CVal derives `PartialEq`).
    TypeOf,
    Equal,
    /// The `str_*` family on `CVal::Str` (`str_len`/`str_concat`/
    /// `str_find`/`str_slice`/`str_split`), dispatched by name.
    StrOp,
}

/// One registry row: which builtin names at which arity lower with
/// which emission shape.
struct Spec {
    names: &'static [&'static str],
    arity: Arity,
    emit: Emit,
}

/// Build the registry from compact `[names] @ arity => Emit` rows.
/// One line per builtin; `@ any` is `Arity::Any`, `@ N` is
/// `Arity::Exactly(N)`. Keeps the config dense as the surface grows.
macro_rules! builtins {
    ( $( [ $($n:literal),+ ] @ $arity:tt => $emit:ident );+ $(;)? ) => {
        &[ $( Spec {
            names: &[$($n),+],
            arity: builtins!(@arity $arity),
            emit: Emit::$emit,
        } ),+ ]
    };
    (@arity any) => { Arity::Any };
    (@arity $n:literal) => { Arity::Exactly($n) };
}

/// The builtin registry -- the single source of truth for what the
/// compiler lowers. Ordered; the dispatcher takes the first match.
const REGISTRY: &[Spec] = builtins! {
    ["shape", "rank", "transpose", "reduce_add", "tally"] @ 1 => UnaryRt;
    ["floor"] @ 1 => MathCall;
    ["iota", "range"] @ 1 => Iota;
    ["reshape"] @ 2 => Reshape;
    ["reduce_add"] @ 2 => ReduceAxis;
    ["label", "relabel"] @ 2 => Label;
    ["reshape_labeled"] @ 3 => Label;
    ["matmul"] @ 2 => Matmul;
    ["write_stdout", "arg", "read_bytes", "file_size",
     "tokenize_bytes", "decode_bytes", "to_int", "disp",
     "print", "eprint", "read_stdin_chunk"] @ 1 => CvalIo;
    ["read_bytes"] @ 3 => ReadRange;
    ["write_bytes", "append_bytes"] @ 2 => WriteBytes;
    ["args"] @ 0 => Args;
    ["read_stdin"] @ 0 => ReadStdin;
    ["exit"] @ 1 => Exit;
    ["ok", "err"] @ 1 => Result;
    ["check"] @ 1 => Check;
    ["take"] @ 3 => ArrayCall;
    ["at"] @ 2 => ArrayCall;
    ["gt", "lt", "eq", "ge", "le", "ne"] @ 2 => Cmp;
    ["type_of"] @ 1 => TypeOf;
    ["equal"] @ 2 => Equal;
    ["str_len"] @ 1 => StrOp;
    ["str_concat", "str_find", "str_split"] @ 2 => StrOp;
    ["str_slice"] @ 3 => StrOp;
    ["band", "bor", "bxor", "bnot", "popcount", "shl", "shr", "bmask", "bits", "from_bits"] @ any => BitCall;
};

/// The builtin names the compiler lowers -- the registry's name set,
/// sorted + deduplicated. Exposed so tooling (and the dispatch
/// coverage tests) can see the supported surface without reaching
/// into the private registry.
pub fn supported_builtin_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = REGISTRY
        .iter()
        .flat_map(|s| s.names.iter().copied())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Lower a call to a user function (`u:name` -> `user_name(args)`) or
/// to a registered builtin. Unknown name/arity is `Unsupported`.
pub(crate) fn lower_fncall(
    ctx: &Ctx,
    name: &str,
    args: &[Expr],
) -> Result<TokenStream, LowerError> {
    if let Some(bare) = name.strip_prefix("u:") {
        let id = format_ident!("user_{bare}");
        let lowered: Vec<TokenStream> = args
            .iter()
            .map(|a| lower_expr(ctx, a))
            .collect::<Result<_, _>>()?;
        return Ok(quote! { #id(#(#lowered),*) });
    }
    let arity = args.len();
    let spec = REGISTRY.iter().find(|s| {
        let arity_ok = match s.arity {
            Arity::Exactly(n) => n == arity,
            Arity::Any => true,
        };
        arity_ok && s.names.contains(&name)
    });
    let Some(spec) = spec else {
        return Err(LowerError::Unsupported(format!("fncall {name}/{arity}")));
    };
    // The emission match lives in this dispatcher (not a private
    // helper): simple shapes inline, the four complex ones delegate.
    let rt = &ctx.rt;
    match &spec.emit {
        Emit::UnaryRt => {
            let (a, f) = (crate::lower_darr(ctx, &args[0])?, format_ident!("{name}"));
            Ok(quote! { #rt::#f(&(#a)) })
        }
        Emit::Iota => {
            let a = crate::lower_darr(ctx, &args[0])?;
            Ok(quote! { #rt::iota((#a).data()[0] as usize) })
        }
        Emit::Reshape => {
            let a = crate::lower_darr(ctx, &args[0])?;
            let shape = crate::lower_darr(ctx, &args[1])?;
            Ok(quote! {{
                let __shape = #shape;
                let __dims: Vec<usize> = __shape.data().iter().map(|&d| d as usize).collect();
                #rt::reshape(&(#a), &__dims).unwrap()
            }})
        }
        Emit::ReduceAxis => {
            let a = crate::lower_darr(ctx, &args[0])?;
            let axis = crate::lower_darr(ctx, &args[1])?;
            Ok(quote! { #rt::reduce_add_axis(&(#a), (#axis).data()[0] as usize).unwrap() })
        }
        Emit::CvalIo => {
            let (a, f) = (crate::lower_cval(ctx, &args[0])?, format_ident!("{name}"));
            Ok(quote! { #rt::#f(&(#a)) })
        }
        Emit::Args => Ok(quote! { #rt::cli_args() }),
        Emit::ReadStdin => Ok(quote! { #rt::read_stdin() }),
        Emit::Exit => {
            let a = crate::lower_cval(ctx, &args[0])?;
            Ok(quote! { #rt::exit(&(#a)) })
        }
        Emit::Result => {
            let (payload, ok) = (crate::lower_cval(ctx, &args[0])?, name == "ok");
            Ok(quote! { #rt::CVal::result(#ok, #payload) })
        }
        Emit::ReadRange => {
            let p = crate::lower_cval(ctx, &args[0])?;
            let off = crate::lower_darr(ctx, &args[1])?;
            let len = crate::lower_darr(ctx, &args[2])?;
            Ok(quote! { #rt::read_bytes_range(&(#p), &(#off), &(#len)) })
        }
        Emit::WriteBytes => {
            let (p, b) = (
                crate::lower_cval(ctx, &args[0])?,
                crate::lower_cval(ctx, &args[1])?,
            );
            let f = format_ident!("{name}");
            Ok(quote! { #rt::#f(&(#p), &(#b)) })
        }
        Emit::Label => lower_label_attach(ctx, name, args),
        Emit::Matmul => lower_matmul(ctx, args),
        // `check(expr)` -- the `?` operator. Valid only inside a
        // CVal-returning function: on `ok` yield the payload, on
        // `err` early-return the whole Result. Top-level `?` has
        // nowhere to propagate, so it is Unsupported.
        Emit::Check if !ctx.in_cval_fn.get() => Err(LowerError::Unsupported(
            "check (the `?` operator) outside a Result-returning function \
             (define the caller with `def u:` and have it return ok/err)"
                .into(),
        )),
        Emit::Check => {
            let call = lower_expr(ctx, &args[0])?;
            Ok(quote! {
                match #call {
                    #rt::CVal::Result { ok: true, payload } => *payload,
                    __err => return __err,
                }
            })
        }
        Emit::BitCall => lower_rt_call(ctx, "bit", name, args),
        Emit::ArrayCall => lower_rt_call(ctx, "array", name, args),
        Emit::MathCall => lower_rt_call(ctx, "math", name, args),
        // gt/lt/eq -- elementwise comparison to 1.0/0.0 with scalar
        // broadcasting, mirroring the arithmetic binop lowering. `eq`
        // compares within `f64::EPSILON` (mlpl-runtime-math parity).
        // Inlined (not a helper fn) to keep fncall.rs at its module
        // function-count ceiling; see the queued fncall-split debt.
        Emit::Cmp => {
            let (l, r) = (
                crate::lower_darr(ctx, &args[0])?,
                crate::lower_darr(ctx, &args[1])?,
            );
            let closure = match name {
                "gt" => quote! { |__a, __b| if __a > __b { 1.0 } else { 0.0 } },
                "lt" => quote! { |__a, __b| if __a < __b { 1.0 } else { 0.0 } },
                "ge" => quote! { |__a, __b| if __a >= __b { 1.0 } else { 0.0 } },
                "le" => quote! { |__a, __b| if __a <= __b { 1.0 } else { 0.0 } },
                "ne" => quote! {
                    |__a: f64, __b: f64| if (__a - __b).abs() >= f64::EPSILON { 1.0 } else { 0.0 }
                },
                _ => quote! {
                    |__a: f64, __b: f64| if (__a - __b).abs() < f64::EPSILON { 1.0 } else { 0.0 }
                },
            };
            Ok(quote! { #rt::ApplyBinopExt::apply_binop(&(#l), &(#r), #closure).unwrap() })
        }
        // type_of(v) -- value kind as a CVal::Str (value_kind parity).
        Emit::TypeOf => {
            let v = crate::lower_cval(ctx, &args[0])?;
            Ok(quote! {
                #rt::CVal::Str(match &(#v) {
                    #rt::CVal::Arr(_) => "array",
                    #rt::CVal::Str(_) => "string",
                    #rt::CVal::StrList(_) => "string-list",
                    #rt::CVal::Record(_) => "record",
                    #rt::CVal::Result { .. } => "result",
                }.to_string())
            })
        }
        // equal(a, b) -- structural CVal equality -> scalar 1.0/0.0.
        Emit::Equal => {
            let (a, b) = (
                crate::lower_cval(ctx, &args[0])?,
                crate::lower_cval(ctx, &args[1])?,
            );
            Ok(quote! { #rt::DenseArray::from_scalar(if (#a) == (#b) { 1.0 } else { 0.0 }) })
        }
        // The str_* family on CVal::Str (interpreter parity). `.str()`
        // pulls the &str; char-based indexing matches the interpreter.
        Emit::StrOp => {
            let s = crate::lower_cval(ctx, &args[0])?;
            match name {
                "str_len" => Ok(quote! {
                    #rt::DenseArray::from_scalar((#s).str().chars().count() as f64)
                }),
                "str_concat" => {
                    let b = crate::lower_cval(ctx, &args[1])?;
                    Ok(quote! { #rt::CVal::Str(format!("{}{}", (#s).str(), (#b).str())) })
                }
                "str_find" => {
                    let n = crate::lower_cval(ctx, &args[1])?;
                    Ok(quote! { #rt::DenseArray::from_scalar({
                        let (__sv, __nv) = (#s, #n);
                        let (__s, __n) = (__sv.str(), __nv.str());
                        match __s.find(__n) {
                            Some(__b) => __s[..__b].chars().count() as f64,
                            None => -1.0,
                        }
                    }) })
                }
                "str_slice" => {
                    let (start, len) = (
                        crate::lower_darr(ctx, &args[1])?,
                        crate::lower_darr(ctx, &args[2])?,
                    );
                    Ok(quote! { #rt::CVal::Str(
                        (#s).str().chars()
                            .skip((#start).data()[0] as usize)
                            .take((#len).data()[0] as usize)
                            .collect::<String>()
                    ) })
                }
                _ => {
                    let sep = crate::lower_cval(ctx, &args[1])?;
                    Ok(quote! { #rt::CVal::StrList({
                        let (__sv, __sepv) = (#s, #sep);
                        let (__s, __sep) = (__sv.str(), __sepv.str());
                        if __sep.is_empty() {
                            __s.chars().map(|__c| __c.to_string()).collect()
                        } else {
                            __s.split(__sep).map(|__x| __x.to_string()).collect()
                        }
                    }) })
                }
            }
        }
    }
}

// -- Complex handlers (multi-line / stateful emission) --

/// Lower a pure name/args runtime-dispatch call to
/// `rt::<disp>_try_call(name, vec![args])` -- one shape for every pure
/// `DenseArray -> DenseArray` builtin. Args lower via `lower_darr`;
/// arity/domain errors are validated inside the dispatch (a domain
/// violation panics -- interpreter `RuntimeError` parity).
fn lower_rt_call(
    ctx: &Ctx,
    disp: &str,
    name: &str,
    args: &[Expr],
) -> Result<TokenStream, LowerError> {
    let rt = &ctx.rt;
    let call = format_ident!("{disp}_try_call");
    let lowered: Vec<TokenStream> = args
        .iter()
        .map(|a| crate::lower_darr(ctx, a))
        .collect::<Result<_, _>>()?;
    Ok(quote! {
        #rt::#call(#name, vec![#(#lowered),*]).expect("builtin").unwrap()
    })
}

/// Lower `label`, `relabel`, and `reshape_labeled` -- the builtins
/// that attach axis labels to an array at runtime.
fn lower_label_attach(ctx: &Ctx, name: &str, args: &[Expr]) -> Result<TokenStream, LowerError> {
    let rt = &ctx.rt;
    let labels_arg_idx = if name == "reshape_labeled" { 2 } else { 1 };
    let labels = extract_label_list(&args[labels_arg_idx])
        .ok_or_else(|| LowerError::LabelsMustBeStringLiterals(name.into()))?;
    let lits: Vec<TokenStream> = labels
        .into_iter()
        .map(|s| quote! { Some(#s.into()) })
        .collect();
    let a = crate::lower_darr(ctx, &args[0])?;
    if name == "reshape_labeled" {
        let shape = crate::lower_darr(ctx, &args[1])?;
        Ok(quote! {{
            let __shape = #shape;
            let __dims: Vec<usize> = __shape.data().iter().map(|&d| d as usize).collect();
            #rt::reshape(&(#a), &__dims).unwrap().with_labels(vec![#(#lits),*]).unwrap()
        }})
    } else {
        Ok(quote! { (#a).with_labels(vec![#(#lits),*]).unwrap() })
    }
}

/// Lower `matmul`, performing the static contraction-axis check when
/// both operands' labels are known at lower time. A disagreement
/// raises `LowerError::StaticShapeMismatch`.
fn lower_matmul(ctx: &Ctx, args: &[Expr]) -> Result<TokenStream, LowerError> {
    let a_labels = labels_of(ctx, &args[0]);
    let b_labels = labels_of(ctx, &args[1]);
    if let (Some(al), Some(bl)) = (&a_labels, &b_labels)
        && al.len() == 2
        && !bl.is_empty()
        && let (Some(ac), Some(bc)) = (&al[1], &bl[0])
        && ac != bc
    {
        return Err(LowerError::StaticShapeMismatch {
            op: "matmul".into(),
            expected: al.clone(),
            actual: bl.clone(),
        });
    }
    let a = crate::lower_darr(ctx, &args[0])?;
    let b = crate::lower_darr(ctx, &args[1])?;
    let rt = &ctx.rt;
    Ok(quote! { #rt::MatmulExt::matmul(&(#a), &(#b)).unwrap() })
}

/// Extract a list of string literals from an `ArrayLit` -- used for
/// the label-attaching builtins. `None` if not an all-StrLit ArrayLit.
pub(crate) fn extract_label_list(expr: &Expr) -> Option<Vec<String>> {
    let Expr::ArrayLit(elems, _) = expr else {
        return None;
    };
    let mut out = Vec::with_capacity(elems.len());
    for e in elems {
        let Expr::StrLit(s, _) = e else {
            return None;
        };
        out.push(s.clone());
    }
    Some(out)
}

/// Statically infer the labels of an expression where possible.
/// `None` when they cannot be known without running the program.
pub(crate) fn labels_of(ctx: &Ctx, expr: &Expr) -> Option<Vec<Option<String>>> {
    match expr {
        Expr::Ident(name, _) => ctx.known_labels.get(name).cloned(),
        Expr::FnCall { name, args, .. } => match (name.as_str(), args.len()) {
            ("label" | "relabel", 2) => {
                extract_label_list(&args[1]).map(|v| v.into_iter().map(Some).collect())
            }
            ("reshape_labeled", 3) => {
                extract_label_list(&args[2]).map(|v| v.into_iter().map(Some).collect())
            }
            ("transpose", 1) => labels_of(ctx, &args[0]).map(|mut v| {
                v.reverse();
                v
            }),
            _ => None,
        },
        _ => None,
    }
}
