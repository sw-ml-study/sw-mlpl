# moe-microscope-followups

Address the three follow-up findings from the ../moe-microscope dogfooding
rerun (see docs/sw-mlpl-findings.md "Open follow-ups"). All have clean
workarounds downstream; none is a blocker. Lead with F5 (highest value).

## Steps

1. F5 -- index/mask builtins on the tape (one_hot, argmax). Let a top-1 router
   mask be computed inside a traced loss by treating index/mask builtins as
   stop-gradient (constant) nodes on the tape: the forward value is the eager
   result, no gradient flows through the indices, gradient flows through the
   selected values where applicable. TDD: grad(sum(one_hot(argmax(W), k) * W), W)
   succeeds and matches the eager mask; verify one_hot/argmax are constant wrt
   their own input. Keep backward-compatible (additive tape arms).

2. F6 -- repeat inside a traced function. Decide and implement: either make a
   bounded repeat expression tape-expressible (unroll onto the tape) OR
   document nested apply / per-depth user functions as the recurrence spelling
   with a loud, specific error. TDD/doc as chosen.

3. D1 -- loud failure for grad wrt an untracked tape constant. When the wrt
   leaf is a constant (not a param/tensor leaf), grad currently can return
   silent zeros; make it a clear error (or a documented, tested boundary).
   TDD: grad(<const>, <const>) errors loudly.

4. relay-and-close -- update docs/sw-mlpl-findings.md marking F5/F6/D1
   resolved/documented, refresh CHANGES + wiki, mark the saga shipped in
   docs/future-sagas-queue.md, rebuild binaries. --done.
