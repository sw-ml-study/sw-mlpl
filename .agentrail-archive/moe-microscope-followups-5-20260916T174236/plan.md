# moe-microscope-followups-5

The moe-microscope rerun after followups-4 found F22 not fully fixed and two new
grad-surface findings (docs/sw-mlpl-findings.md). Lead with the F22 redo.

## Steps

1. f22-redo-per-param-step -- reset_optimizer + rebind-clearing were necessary
   but not sufficient: Adam's step counter is keyed per-optimizer ("adam"),
   shared across models, so two variants trained in sequence cross-contaminate
   via bias correction. Key the step counter per parameter (or per-param-set)
   so each model's bias correction is independent; clear it on rebind/reset.
   TDD: reused-name and fresh-name first steps match without a shared-counter
   artifact.
2. f23-reshape-dims-from-fn-params -- reshape dims bound to function parameters
   drop the gradient silently inside grad. Resolve/handle so the gradient
   flows (or errors loudly). TDD.
3. f24-apply-engram-ids-from-fn-param -- apply_engram with ids bound to a
   function parameter is not seen inside grad (F11/F15-class scope). Resolve
   ids against the traced scope. TDD.
4. relay-and-close -- mark F22/F23/F24 resolved, refresh CHANGES + wiki, mark
   the saga shipped, rebuild binaries. --done.
