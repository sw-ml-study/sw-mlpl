# Decompose mlpl-eval (saga 65)

96 modules, single largest FAIL. Extract groups with no impl
constraint into sibling crates within components/eval/.

## File groups and impl status

- env_*    (26 files): `impl Environment` — MUST stay with env.rs
- model_*  (18 files): free functions — extractable
- eval_*   (11 files): mostly free functions — extractable
- fetch_*  ( 6 files): free functions — extractable
- inspect_*( 5 files): free functions — extractable
- grad_*   ( 4 files): free functions — extractable
- fncall_* ( 4 files): free functions — extractable
- error_*  ( 4 files): impls on error type — stays with error.rs
- tag_*    ( 2 files): free functions — extractable
- image_*  ( 2 files): free functions — extractable
- experiment_* (2 files): free functions — extractable

Singletons (12 files): lib, env, error, value, type, device,
result, tokenizer, pets_tiny, loader, llm_call, interrupt,
auto_tag, bpe.

## Target after extraction

mlpl-eval keeps: lib + env (27) + error (4) + value + type + device
+ result + tokenizer + pets_tiny + loader + llm + interrupt +
auto_tag + bpe = ~42 modules. Still FAIL (>7) but massive
reduction from 96. Env-decompose deferred to future saga.

Extracted siblings:
- mlpl-eval-model (18)
- mlpl-eval-dispatch (eval_* 11)
- mlpl-eval-fetch (6)
- mlpl-eval-inspect (5)
- mlpl-eval-grad (4)
- mlpl-eval-fncall (4)
- mlpl-eval-tagprop (tag_* 2 + auto_tag 1 = 3)
- mlpl-eval-image (2)
- mlpl-eval-experiment (2)

= 9 new sibling crates. Most sparse (under 7 modules each).

## Step plan

1. extract-model
2. extract-eval-dispatch  
3. extract-fetch
4. extract-inspect
5. extract-grad-fncall
6. extract-smalls (tag + image + experiment)
7. close
