# Saga: extension-registry-static-provider

First slice of the demo-extensions native-extension contract
(docs/companion-demo-extensions.md), REGISTRY-FIRST per user
decision 2026-08-09: prove the hard architectural core with the
EXISTING colon spelling (hello:answer()); defer the `use hello` +
dotted `hello.answer()` grammar + module.mlpl facade to a saga-2
built jointly with modules-namespaces. Compiler integration is a
follow-up contract (depends on compiler-io-parity), not this
saga. Interpreter + REPL only; static provider; scalar V1 only.

Acceptance: registered static `hello` provider -> hello:answer()
returns typed i64 42; a failing extension call -> err Result; a
panicking extension is CONTAINED (EvalError::ExtensionError, no
unwind); help/:describe shows the declared signature.

New crates in components/runtime-core/crates/ (one shared
OnceLock registry across eval + provider + binaries):
- mlpl-extension-abi: ExtValue (nil/bool/i64/f64/string/bytes),
  ExtError, ExtFn, ExtFnDesc, ExtensionDescriptorV1,
  call_contained (catch_unwind).
- mlpl-extension-registry: OnceLock<RwLock<Registry>>,
  register/lookup/signatures; fail-closed dup; host-owned copies.
- mlpl-ext-hello-static: in-repo static hello provider + facade
  string (facade unused until saga-2).

## Steps
1. abi -- mlpl-extension-abi crate + call_contained; TDD.
2. registry -- mlpl-extension-registry crate; TDD.
3. dispatch -- mlpl-ext-hello-static provider; fncall_ext.rs
   Value-tier dispatcher wired into eval_fncalls; Value<->ExtValue
   marshal; EvalError::ExtensionError; TDD (hello:answer()->42,
   failing->err Result, panic contained).
4. help -- :describe/help shows the registered signature from the
   registry; TDD; catalog surface reused.
5. close -- register provider in repl+serve binaries; acceptance
   test (script+REPL); rebuild+deploy; wiki row; q-and-a; queue
   the compiler follow-up contract + saga-2 (use/dotted with
   modules-namespaces); --done.
