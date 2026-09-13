# moe-microscope-followups-2

Address the second downstream dogfooding batch (F9-F15) from ../moe-microscope
(see docs/sw-mlpl-findings.md "Follow-up batch 2"). Lead with F10 -- a Rust
panic in the autograd tape, the only correctness/safety issue in the batch.

## Steps (priority order)

1. F10 (BUG) -- a labeled sinusoidal_encoding in a residual block panics the
   autograd tape. Turn the panic into a clean MLPL error at minimum; support
   it on the tape if tractable. TDD: the reproducer errors cleanly (no panic).
2. F14 -- constant constructors (fill, zeros, ones, ...) rejected inside grad;
   treat as constant leaves on the tape (like literals). TDD gradcheck.
3. F9 -- embed rejects batched [B, T] token input; accept it (or document).
4. F15 -- repeat count bound to a function parameter fails inside a traced fn;
   resolve the count against the traced local scope (F6 edge). TDD.
5. F12 -- shape-derived size arithmetic rejected inside grad; support or
   document.
6. F11 -- F2 inliner drops a param used only in index arithmetic; fix or
   document the boundary.
7. F13 (low) -- attention_weights cannot see inside residual(chain(...));
   document (the downstream keeps the hand-written residual regardless).
8. relay-and-close -- mark F9-F15 resolved/documented, refresh CHANGES + wiki,
   mark the saga shipped, rebuild binaries. --done.
