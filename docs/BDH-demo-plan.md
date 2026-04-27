# BDH demo plan

## Goal

Use "Baby Dragon Hatchlings" as an MLPL-branded teaching demo, not as
an existing public model architecture. The demo should be playful but
technically real: tiny agents hatch, learn simple behaviors, and gain
small task adapters through LoRA.

The strongest version is a compact multitask model:

- base "hatchling brain" trained from scratch on simple grid-world
  behaviors;
- LoRA adapters for personalities or skills;
- MLX acceleration for adapter training;
- REPL visualizations for behavior changes.

## Why this is better than a notebook

- One source file can define the world, policy, reward, adapter, and
  visualization.
- Users can train a new hatchling personality live in the REPL without
  re-running notebook cells in the right order.
- LoRA adapters are small enough to save, compare, and swap.
- The compiled app can become a tiny interactive toy: load a base
  model, choose an adapter, and watch the agent act.

## Repository strategy

Use the existing Baby Dragon Hatchlings repo as the home for art
assets, saved adapters, interactive app code, and artifacts. This plan
is for adding an sw-MLPL track to that repo, not for creating a new
repo.

Suggested layout inside the existing repo:

- `mlpl/hatchling_base.mlpl`
- `mlpl/hatchling_lora_fire.mlpl`
- `mlpl/hatchling_lora_guard.mlpl`
- `assets/` for SVG sprites or generated bitmaps
- `src/` for compiled app wrapper
- `adapters/` ignored or Git LFS-managed if published

## MLPL support needed

1. Adapter persistence.
   - `save_lora(model, "path")`
   - `load_lora(model, "path")`
2. Tiny RL or imitation helpers.
   - `gridworld(seed, n, h, w)`
   - `policy_loss(logits, actions, rewards)`
   - Optional: `self_play` later.
3. Better demo asset handling.
   - SVG sprite composition or simple raster export.
4. Model comparison helpers.
   - `compare_policy(base, adapted, world)`
   - behavior heatmaps.
5. Compiled app wrapper.
   - Interactive mode can embed the interpreter first.

## Demo shape

### REPL/interpreter flow

```mlpl
world = hatchling_world(0, 256, 8, 8)
brain = hatchling_policy(obs_dim(world), 32, 4, 0)

device("mlx") {
  train 100 {
    logits = apply(brain, world.obs)
    loss = policy_loss(logits, world.actions, world.rewards)
    adam(loss, brain, 0.003, 0.9, 0.999, 0.00000001)
    reward_metric = mean(world.rewards)
    loss
  }
}

guard = lora(brain, 4, 8.0, 1)
device("mlx") {
  train 40 {
    logits = apply(guard, world.obs)
    loss = policy_loss(logits, world.guard_actions, world.guard_rewards)
    adam(loss, guard, 0.01, 0.9, 0.999, 0.00000001)
    loss
  }
}

svg(policy_rollout(guard, world), "gridworld")
```

### Compiled-app flow

```sh
mlpl build mlpl/hatchling_demo.mlpl -o target/hatchling-demo
target/hatchling-demo --adapter guard --seed 7
```

## Phases

1. CPU imitation-learning policy on a generated grid world.
2. MLX training for the base policy.
3. LoRA adapters for distinct behaviors.
4. Adapter save/load.
5. Visual rollout viewer.
6. Compiled interactive app.

## Acceptance tests

- A tiny policy improves reward over a random baseline.
- A LoRA adapter changes behavior without mutating the base model.
- The rollout visualization is deterministic for a fixed seed.
- The same demo has REPL and compiled-app commands.

## References

No public ML architecture named "Baby Dragon Hatchlings" was found in
the initial scan. Treat this as an original demo concept hosted in the
existing BDH repo.
