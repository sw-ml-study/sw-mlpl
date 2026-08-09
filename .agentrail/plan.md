# Saga: compiled-value-model-strings

Saga A of the phased compiler-io-parity program (user decision
2026-08-09). The compile-to-Rust path is numerical-expression-
only (StrLit/If/While/For/FnDef all Unsupported; result assumed
DenseArray). This saga adds a COMPILED VALUE MODEL beyond
DenseArray -- strings -- plus write_stdout + args, yielding the
first compiled binary that does I/O.

Design (minimize churn to the working numerical path):
- mlpl-rt gains CVal { Arr(DenseArray), Str(String),
  StrList(Vec<String>) } + Display + arr() accessor + IO:
  write_stdout(&CVal)->CVal(count), cli_args()->CVal(StrList),
  arg(idx)->CVal(Str).
- lowering: numerical subexpressions stay DenseArray (apply_binop
  / matmul untouched); a new lower_cval wraps the top-level
  result and write_stdout args -- StrLit -> CVal::Str,
  write_stdout/args -> their CVal lowering, else
  CVal::Arr(lower_expr(..)). lower() now returns a CVal-producing
  block.
- callers: mlpl-build main template prints CVal via Display;
  existing numerical macro/e2e tests read result.arr().data()[0].
- No string VARIABLES yet (a compiled binary that prints a
  literal / echoes an arg / writes a byte array); string vars +
  read_bytes + control flow + functions are later rungs.

## Steps
1. cval-runtime -- mlpl-rt CVal enum + Display + arr() +
   write_stdout + cli_args + arg; unit TDD.
2. cval-lowering -- lower_cval + StrLit/write_stdout/args
   lowering; lower() returns CVal; update main template + macro
   test call sites; token TDD.
3. cval-e2e -- gated compile test: a compiled binary that prints
   a string, echoes an arg, and writes a byte array to stdout.
4. close -- docs (maturity note: compiler now handles strings +
   stdout/args), queue saga B (read_bytes/bytes); --done.
