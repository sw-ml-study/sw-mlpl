//! Dispatch coverage guards. A data-driven registry loses the
//! compile-time exhaustiveness a `match` gives, so these tests
//! recover it three ways:
//!
//! 1. An exhaustive `Builtin` enum drives a sample call per builtin
//!    (`sample`/`name` are `match`es -- adding a variant without an
//!    arm fails to COMPILE, the analog of a missing match arm).
//! 2. A reverse check: every name the registry dispatches must have a
//!    `Builtin` variant, so adding to the registry forces adding a
//!    covered sample here.
//! 3. A cross-check binding the bit-op family to its authoritative
//!    source, `mlpl-runtime-bits::NAMES`.

use mlpl_lower_rs::{lower, supported_builtin_names};
use mlpl_parser::{lex, parse};

fn lowers_ok(src: &str) -> bool {
    let Ok(toks) = lex(src) else { return false };
    let Ok(stmts) = parse(&toks) else {
        return false;
    };
    lower(&stmts).is_ok()
}

/// Every builtin the compiler dispatches. Adding one means adding a
/// variant -- which the `match`es below then force you to give a
/// sample + name for (or the crate does not compile).
#[derive(Clone, Copy)]
enum Builtin {
    Shape,
    Rank,
    Tally,
    Floor,
    Take,
    Transpose,
    ReduceAdd,
    ReduceAddAxis,
    Iota,
    Range,
    Reshape,
    Label,
    Relabel,
    ReshapeLabeled,
    Matmul,
    WriteStdout,
    Arg,
    Args,
    Exit,
    Ok,
    Err,
    Check,
    Band,
    Bor,
    Bxor,
    Bnot,
    Popcount,
    Shl,
    Shr,
    Bmask,
    Bits,
    FromBits,
    ReadBytes,
    ReadBytesRange,
    FileSize,
    WriteBytes,
    AppendBytes,
    TokenizeBytes,
    DecodeBytes,
    ToInt,
    Disp,
    Print,
    Eprint,
    ReadStdin,
    Gt,
    Lt,
    Eq,
    TypeOf,
    Equal,
    StrLen,
    StrConcat,
    StrFind,
    StrSlice,
    StrSplit,
}

const ALL: &[Builtin] = &[
    Builtin::Shape,
    Builtin::Rank,
    Builtin::Tally,
    Builtin::Floor,
    Builtin::Take,
    Builtin::Transpose,
    Builtin::ReduceAdd,
    Builtin::ReduceAddAxis,
    Builtin::Iota,
    Builtin::Range,
    Builtin::Reshape,
    Builtin::Label,
    Builtin::Relabel,
    Builtin::ReshapeLabeled,
    Builtin::Matmul,
    Builtin::WriteStdout,
    Builtin::Arg,
    Builtin::Args,
    Builtin::Exit,
    Builtin::Ok,
    Builtin::Err,
    Builtin::Check,
    Builtin::Band,
    Builtin::Bor,
    Builtin::Bxor,
    Builtin::Bnot,
    Builtin::Popcount,
    Builtin::Shl,
    Builtin::Shr,
    Builtin::Bmask,
    Builtin::Bits,
    Builtin::FromBits,
    Builtin::ReadBytes,
    Builtin::ReadBytesRange,
    Builtin::FileSize,
    Builtin::WriteBytes,
    Builtin::AppendBytes,
    Builtin::TokenizeBytes,
    Builtin::DecodeBytes,
    Builtin::ToInt,
    Builtin::Disp,
    Builtin::Print,
    Builtin::Eprint,
    Builtin::ReadStdin,
    Builtin::Gt,
    Builtin::Lt,
    Builtin::Eq,
    Builtin::TypeOf,
    Builtin::Equal,
    Builtin::StrLen,
    Builtin::StrConcat,
    Builtin::StrFind,
    Builtin::StrSlice,
    Builtin::StrSplit,
];

impl Builtin {
    /// A valid source snippet exercising this builtin. EXHAUSTIVE
    /// match: a new variant with no sample is a compile error.
    fn sample(self) -> &'static str {
        match self {
            Builtin::Shape => "shape(iota(3))",
            Builtin::Rank => "rank(iota(3))",
            Builtin::Tally => "tally(iota(3))",
            Builtin::Floor => "floor(iota(3))",
            Builtin::Take => "take(iota(3), 0, 1)",
            Builtin::Transpose => "transpose(iota(3))",
            Builtin::ReduceAdd => "reduce_add(iota(3))",
            Builtin::ReduceAddAxis => "reduce_add(reshape(iota(6), [2, 3]), 0)",
            Builtin::Iota => "iota(3)",
            Builtin::Range => "range(3)",
            Builtin::Reshape => "reshape(iota(6), [2, 3])",
            Builtin::Label => "label(iota(3), [\"s\"])",
            Builtin::Relabel => "relabel(iota(3), [\"s\"])",
            Builtin::ReshapeLabeled => "reshape_labeled(iota(6), [2, 3], [\"r\", \"c\"])",
            Builtin::Matmul => "matmul(reshape(iota(6), [2, 3]), reshape(iota(6), [3, 2]))",
            Builtin::WriteStdout => "write_stdout(\"hi\")",
            Builtin::Arg => "arg(0)",
            Builtin::Args => "args()",
            Builtin::Exit => "exit(0)",
            Builtin::Ok => "ok(1)",
            Builtin::Err => "err(1)",
            // `?` is valid only inside a Result-returning function.
            Builtin::Check => "def u:f(n) { ok(n) }\ndef u:g(n) { u:f(n)? }\nu:g(1)",
            Builtin::Band => "band(1, 2)",
            Builtin::Bor => "bor(1, 2)",
            Builtin::Bxor => "bxor(1, 2)",
            Builtin::Bnot => "bnot(1, 8)",
            Builtin::Popcount => "popcount(1)",
            Builtin::Shl => "shl(1, 2, 8)",
            Builtin::Shr => "shr(1, 2)",
            Builtin::Bmask => "bmask(1, 4)",
            Builtin::Bits => "bits(1, 8)",
            Builtin::FromBits => "from_bits([1, 0])",
            Builtin::ReadBytes => "read_bytes(\"f\")",
            Builtin::ReadBytesRange => "read_bytes(\"f\", 0, 4)",
            Builtin::FileSize => "file_size(\"f\")",
            Builtin::WriteBytes => "write_bytes(\"f\", [65, 66])",
            Builtin::AppendBytes => "append_bytes(\"f\", [67])",
            Builtin::TokenizeBytes => "tokenize_bytes(\"hi\")",
            Builtin::DecodeBytes => "decode_bytes([72, 105])",
            Builtin::ToInt => "to_int(\"42\")",
            Builtin::Disp => "disp(\"hi\")",
            Builtin::Print => "print(\"hi\")",
            Builtin::Eprint => "eprint(\"hi\")",
            Builtin::ReadStdin => "read_stdin()",
            Builtin::Gt => "gt(5, 3)",
            Builtin::Lt => "lt(5, 3)",
            Builtin::Eq => "eq(4, 4)",
            Builtin::TypeOf => "type_of([1, 2])",
            Builtin::Equal => "equal([1, 2], [1, 2])",
            Builtin::StrLen => "str_len(\"hi\")",
            Builtin::StrConcat => "str_concat(\"a\", \"b\")",
            Builtin::StrFind => "str_find(\"hi\", \"i\")",
            Builtin::StrSlice => "str_slice(\"hi\", 0, 1)",
            Builtin::StrSplit => "str_split(\"a,b\", \",\")",
        }
    }

    /// The builtin's dispatch name. EXHAUSTIVE match.
    fn name(self) -> &'static str {
        match self {
            Builtin::Shape => "shape",
            Builtin::Rank => "rank",
            Builtin::Tally => "tally",
            Builtin::Floor => "floor",
            Builtin::Take => "take",
            Builtin::Transpose => "transpose",
            Builtin::ReduceAdd | Builtin::ReduceAddAxis => "reduce_add",
            Builtin::Iota => "iota",
            Builtin::Range => "range",
            Builtin::Reshape => "reshape",
            Builtin::Label => "label",
            Builtin::Relabel => "relabel",
            Builtin::ReshapeLabeled => "reshape_labeled",
            Builtin::Matmul => "matmul",
            Builtin::WriteStdout => "write_stdout",
            Builtin::Arg => "arg",
            Builtin::Args => "args",
            Builtin::Exit => "exit",
            Builtin::Ok => "ok",
            Builtin::Err => "err",
            Builtin::Check => "check",
            Builtin::Band => "band",
            Builtin::Bor => "bor",
            Builtin::Bxor => "bxor",
            Builtin::Bnot => "bnot",
            Builtin::Popcount => "popcount",
            Builtin::Shl => "shl",
            Builtin::Shr => "shr",
            Builtin::Bmask => "bmask",
            Builtin::Bits => "bits",
            Builtin::FromBits => "from_bits",
            Builtin::ReadBytes | Builtin::ReadBytesRange => "read_bytes",
            Builtin::FileSize => "file_size",
            Builtin::WriteBytes => "write_bytes",
            Builtin::AppendBytes => "append_bytes",
            Builtin::TokenizeBytes => "tokenize_bytes",
            Builtin::DecodeBytes => "decode_bytes",
            Builtin::ToInt => "to_int",
            Builtin::Disp => "disp",
            Builtin::Print => "print",
            Builtin::Eprint => "eprint",
            Builtin::ReadStdin => "read_stdin",
            Builtin::Gt => "gt",
            Builtin::Lt => "lt",
            Builtin::Eq => "eq",
            Builtin::TypeOf => "type_of",
            Builtin::Equal => "equal",
            Builtin::StrLen => "str_len",
            Builtin::StrConcat => "str_concat",
            Builtin::StrFind => "str_find",
            Builtin::StrSlice => "str_slice",
            Builtin::StrSplit => "str_split",
        }
    }
}

#[test]
fn every_builtin_sample_lowers() {
    for &b in ALL {
        assert!(
            lowers_ok(b.sample()),
            "`{}` sample failed to lower",
            b.name()
        );
    }
}

/// Reverse coverage: every name the registry dispatches has a
/// `Builtin` variant here. Adding a builtin to the registry without a
/// covered sample fails this test -- the missing-handler check the
/// user asked for.
#[test]
fn every_registered_name_has_a_builtin_variant() {
    let covered: Vec<&str> = ALL.iter().map(|b| b.name()).collect();
    for name in supported_builtin_names() {
        assert!(
            covered.contains(&name),
            "registered builtin `{name}` has no Builtin coverage variant -- add one so dispatch stays exhaustively tested"
        );
    }
}

/// The bit-op family is bound to its source of truth: a bit op added
/// to `mlpl-runtime-bits` but not registered in the compiler fails
/// here (the runtime and compiler cannot silently drift).
#[test]
fn every_runtime_bit_op_is_registered() {
    let supported = supported_builtin_names();
    for name in mlpl_runtime_bits::NAMES {
        assert!(
            supported.contains(name),
            "bit op `{name}` is exported by mlpl-runtime-bits but not registered in the compiler"
        );
    }
}
