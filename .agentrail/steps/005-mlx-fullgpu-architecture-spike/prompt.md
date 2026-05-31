MLX full-GPU training: ARCHITECTURE DECISION + spike (Phase 1b cont.).

Context: step 004 mapped the reality + recovered groundwork. Today
device("mlx") accelerates only forward; backward (mlpl-autograd, CPU
f64) and adam (mlpl-eval/src/grad_optim.rs, Vec<f64>) run on CPU. To
run the LoRA fine-tune fully on the GPU there are TWO approaches --
decide BEFORE implementing (this fork shapes all later steps):

  A) MLX built-in autodiff (RECOMMENDED). Express the MLX fine-tune
     step's forward as an mlx_rs-traceable function and use
     mlx_rs value_and_grad / grad to get gradients on-device; keep
     moment buffers as mlx_rs::Array and do the adam update with MLX
     elementwise ops. Bypasses mlpl-autograd's 19 hand-written
     backward formulas for the MLX path -- far less code + no parity
     drift, at the cost of restructuring the MLX fine-tune loop to
     build an MLX graph instead of the CPU tape.
  B) Hand-port backward to MLX. Add MLX backward ops to mlpl-mlx-rt
     (matmul/elementwise/softmax/cross_entropy backward) and make
     mlpl-autograd's backward dispatch to MLX when device==mlx, plus
     an on-device adam. Keeps the existing tape architecture but is
     ~19 backward formulas to port + parity-maintain.

Work: (1) confirm the mlx_rs version's autodiff surface
(value_and_grad/grad availability) with a tiny spike build
(Mac/aarch64, --features mlx; DISK: target/ was cleaned, an MLX build
is multi-GB -- check df -h / first, scope the build to the mlx crates,
clean after). (2) Write the decision into docs/saga-local-gpu-agentic.md
(replace/extend the 'CRITICAL reality' section with the chosen design).
(3) Define the follow-on steps via agentrail complete --next-* (e.g. for
A: 'mlx-traceable-finetune-forward' then 'mlx-value-and-grad-adam'; for
B: 'mlx-rt-backward-ops' then 'autograd-mlx-dispatch' then
'mlx-on-device-adam'). Then implement the first slice. Goal end-state
(across the follow-ons): a device("mlx") LoRA fine-tune where forward,
backward, AND optimizer run on GPU, parity-tested vs CPU within fp32
tolerance, and the 'MLX LoRA fine-tune' demo relabeled from hybrid to
true-GPU. Keep sw-checklist non-regressing; commit .agentrail with source.