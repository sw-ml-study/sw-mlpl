# Saga: demo-pedagogy-and-queue-reorder

Two user-directed short-term items (2026-08-02) before saga E5:

1. Loop-avoidance pedagogy in the APL-related demos. The APL2
   category already tells the "zero cell loops" story (Life);
   what is missing is the DATA-loop vs TIME-loop (recurrence)
   distinction:
   - Life demos: add takeaway text naming the loop that
     legitimately survives (repeat over generations =
     grid(t+1) = step(grid(t))) vs the cell loops that vanished.
   - New small APL2-category demo "Thinking in Arrays": one task
     solved twice (explicit for-loop vs one array expression),
     then a train/repeat block explaining why the time loop
     survives, plus the scan/associativity borderline (cumprod
     today, general scan in the G1 parity saga).
   - Attention demo intros: one sentence each ("every loop over
     position pairs is one matmul") tying the APL thesis to ML.
   - apl2-parity-gap.md G1: note that the scan saga should carry
     the associative-recurrence pedagogy.
   - demos.toml changes -> pages rebuild + deploy (build-pages,
     commit pages/, push). Wiki errata discipline applies.

2. Reorder docs/future-sagas-queue.md per user direction
   (2026-08-02, overriding the mHC-before-MTP placement):
   - E5 engram-mlx stays next.
   - MTP program promoted directly after E5:
     generation-state-kv-cache -> mtp-training ->
     mtp-self-speculation (CPU + MLX).
   - mHC-CPU sagas (p1 constrained-transforms, p2 cpu-layer)
     PARKED BEHIND the MTP program (user-confirmed: parked, not
     cancelled); mhc-p3-mlx-resident and mhc-cuda DEFERRED.
   - engram-cuda stays deferred to the Linux box (Track 5).
   - Agent track (episodes -> ICL -> ICRL) after the first MTP
     speed result, per docs/project-direction.txt.

## Steps

1. queue-reorder -- rewrite docs/future-sagas-queue.md to the
   order above (docs-only; markdown + sw-checklist gates; update
   wiki errata if any wiki page states the old order).
2. loop-avoidance-demos -- the pedagogy work above: demos.toml
   takeaway/intro edits + the new "Thinking in Arrays" demo +
   apl2-parity-gap.md G1 note + pages rebuild/deploy + demo smoke
   test coverage (the new demo must pass every_quick_demo_runs).
