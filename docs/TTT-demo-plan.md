# Tic-Tac-Toe and tiny networks demo plan

## Goal

Build a family of from-scratch neural-network demos, centered on a
Tic-Tac-Toe agent that learns to play better. This should become the
approachable "MLPL can train models, not just call models" showcase.

The broader tiny-network set should include:

- MLP classifier from scratch.
- CNN on tiny image-like grids.
- RNN on toy sequence prediction.
- Tic-Tac-Toe supervised imitation from minimax.
- Tic-Tac-Toe self-play fine-tuning.

## Why this is better than a notebook

- The whole experiment is a program with stable execution order.
- Users can inspect board tensors, logits, illegal-move masks, loss,
  and win-rate metrics in the REPL.
- Visualizations are part of the language, not side-channel plotting
  code.
- A compiled app can expose `play`, `train`, and `evaluate` commands
  from the same source-level model.
- MLX acceleration is optional: tiny CPU demos stay fast, and larger
  self-play batches can move to MLX.

## Repository strategy

Use the existing Tic-Tac-Toe repo as the home for the interactive game
app, trained policies, screenshots, and demo scripts. Keep only
reusable tiny-network/game-helper language support in `sw-mlpl`. This
plan is for adding an sw-MLPL track to the existing repo, not for
creating a new repo.

Suggested layout inside the existing repo:

- `mlpl/ttt_supervised.mlpl`
- `mlpl/ttt_self_play.mlpl`
- `mlpl/cnn_tiny.mlpl`
- `mlpl/rnn_tiny.mlpl`
- `src/main.rs` for playable CLI or small GUI wrapper
- `artifacts/` ignored by default

## MLPL support needed

1. CNN primitives.
   - `conv2d`, `max_pool2d`, `flatten`, `conv_layer`.
2. RNN primitives.
   - `rnn_cell`, `gru_cell`, `scan`.
3. Game helpers.
   - `ttt_all_boards()`
   - `ttt_legal_mask(board)`
   - `ttt_minimax_policy(board)`
   - `ttt_step(board, action)`
   - `ttt_outcome(board)`
4. Policy losses.
   - masked softmax/cross entropy.
   - value loss for win/draw/loss prediction.
5. Self-play.
   - deterministic opponent sampling.
   - replay buffer or generated-on-demand batches.
6. Visualization.
   - board SVG.
   - policy heatmap over nine squares.
   - win-rate curve.
7. Compiled app path.
   - `play` command against a trained policy.
   - `train` command if train lowering is ready; otherwise embed the
     interpreter for training and compile inference first.

## Demo shape

### Supervised REPL/interpreter flow

```mlpl
data = ttt_all_boards()
X = data.boards
Y = ttt_minimax_policy(X)

policy = chain(linear(9, 32, 0), tanh_layer(), linear(32, 9, 1))

train 200 {
  logits = apply(policy, X)
  masked = mask_logits(logits, ttt_legal_mask(X))
  loss = cross_entropy(masked, Y)
  adam(loss, policy, 0.01, 0.9, 0.999, 0.00000001)
  acc_metric = policy_accuracy(masked, Y)
  loss
}

svg(ttt_board(X[0]), "ttt")
svg(policy_heatmap(policy, X[0]), "heatmap")
loss_curve(last_losses)
```

### Self-play extension

```mlpl
device("mlx") {
  train 100 {
    games = ttt_self_play(policy, 128, step)
    loss = policy_value_loss(policy, games)
    adam(loss, policy, 0.003, 0.9, 0.999, 0.00000001)
    win_metric = ttt_win_rate(policy, "random", 128, step)
    loss
  }
}
```

### Compiled-app flow

```sh
mlpl build mlpl/ttt_play.mlpl -o target/ttt-demo
target/ttt-demo play --model artifacts/policy.json
target/ttt-demo eval --opponent minimax --games 100
```

## Phases

1. Board representation and minimax labels.
2. CPU supervised MLP policy.
3. Board/policy visualizations.
4. Illegal-move masking and quality metrics.
5. Compiled inference app.
6. Self-play training.
7. CNN/RNN sibling demos using the same tiny-network pattern.

## Acceptance tests

- `ttt_legal_mask` rejects occupied squares.
- `ttt_minimax_policy` never chooses illegal moves.
- Supervised policy beats random after a short training run.
- Policy never emits an illegal move after masking.
- Compiled app can play a legal full game.
- CNN and RNN tiny demos each have one deterministic training test.
