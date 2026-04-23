Phase 2 step 004: `demos/lora_finetune.mlpl` end-to-end on
CPU. First runnable LoRA fine-tune demo.

1. Author `demos/lora_finetune.mlpl`:
   a. Build a Saga 13-shape Tiny LM base
      (V=280 via `train_bpe` on the preloaded
      Shakespeare snippet; d=32, h=1, context=32).
   b. Pre-train the base for ~100 Adam steps (same budget
      as the neural_thicket demo; this is the "base"
      phase, not the point of the demo).
   c. Build a tiny synthetic instruction corpus as a
      single string -- ~10 short Q/A pairs, e.g.
      `"Q: hi A: hello\nQ: name A: alice\n..."`.
      Tokenize it with the SAME BPE (reuse `tok`) and
      build `instr_X` / `instr_Y` via `shift_pairs_x/y`.
   d. `freeze(base)` to mark all base params frozen.
   e. `student = lora(base, 8, 16.0, 0)` -- rank-8
      adapters, alpha=16 (standard LoRA ratio).
   f. Fine-tune `student` on the instruction corpus for
      ~50 Adam steps via `train 50 { adam(cross_entropy(
      apply(student, instr_X), instr_Y), student, ...);
      loss_metric = cross_entropy(...) }`.
   g. `loss_curve(last_losses)` visualizes the fine-tune
      loss.
   h. Demonstrate that the base is unchanged: re-run
      `apply(base, instr_X)` vs the pre-fine-tune value;
      they should be identical. (Not strictly necessary in
      the demo output; the integration test pins this.)
   i. Optional: sample a short generation from the
      fine-tuned student to show the specialization.

2. Integration test
   `crates/mlpl-eval/tests/lora_finetune_tests.rs`:
   - Cut-down (V=32 byte-level, d=8, ctx=4, 3 base-train
     + 3 lora-finetune steps).
   - Assert `env.get(base_W)` before fine-tune equals
     `env.get(base_W)` after fine-tune, for every base
     param (bit-identical).
   - Assert adapter A has moved from its randn init AND
     adapter B has moved from zero init.
   - Assert fine-tune loss at step 3 <= loss at step 1
     (allow wiggle for tiny models; set a conservative
     bound).

3. Add a one-line entry to `docs/demos-scripts.md`
   pointing at the new demo. Name it Demo 8 or similar,
   following the Saga 20 Demo 7 pattern.

4. Run `./target/release/mlpl-repl -f
   demos/lora_finetune.mlpl` manually once and paste the
   final loss / ens metric into the commit message.

5. Quality gates + `/mw-cp`. Commit demo + test + doc
   entry together. Commit message references Saga 15
   step 004.
