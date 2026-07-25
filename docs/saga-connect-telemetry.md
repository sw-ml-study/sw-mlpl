# Saga: Connect Telemetry (live training signals in the web UI)

**Status: COMPLETE (2026-07-25).** All steps shipped the same day it
was kicked off, plus two pulled-forward inserts (implicit `loss`
metric; persisted final chart) and a workspace-lock sweep. See the
retrospective in `docs/saga.md`.

Kicked off 2026-07-25 on branch `feature/connect-telemetry`. Follow-on to
the beginner-ML comprehension work (`docs/beginner-ml-comprehension-plan.md`,
items 1-4 shipped June-July): item 2 closed with the note that "a true
in-place live-loss panel over the SSE stream remains a connect-path
follow-up." This saga is that follow-up.

## Vision

When a browser is connected to an `mlpl-serve` peer, training should feel
live: the loss curve grows point by point while the `train` block runs, and
the machine's vital signs (CPU/GPU/RAM/VRAM) scroll alongside it, so a
learner sees BOTH what the model is learning and what the hardware is doing
to learn it. Today those signals exist but are disconnected: per-step
`event: metric` SSE frames already stream on the connect path
(`components/wasm/crates/mlpl-web-eval/src/eval_sse.rs`), resource
sparklines already poll `/v1/stats` every 400ms (`TelemetryPanel`,
`components/web-render/crates/mlpl-web-render-aux/src/telemetry_panel.rs`),
and `train_val_curve` renders a static SVG only at checkpoints.

## What this enables, and where it surfaces in the UI

- **UI change (connect mode)**: a live loss panel -- an in-place chart that
  appends a point per `metric` frame during `train`, replacing the
  "Evaluating..." dead air. Train + validation series on shared axes so
  overfitting appears as a widening gap while it happens.
- **UI change (connect mode)**: the existing `TelemetryPanel` sparklines
  rendered in the same visual frame as the live loss, time-aligned, so
  "GPU busy" and "loss falling" read as one story.
- **Demo upgrades, not new demos**: "Watch a Model Learn (overfitting)" and
  "How Gradient Descent Works" gain live behavior when connected; the
  chunked-repaint fallback keeps them working on the public CPU/browser
  demo.
- **Glossary/tour wiring**: pointers from the live panel to the relevant
  terms (Loss, Overfitting, Epoch) per the beginner-spine conventions.

## Steps

1. **reconcile-and-kickoff** (meta) -- reconcile agentrail with the
   June-July off-rail history (audit -> retroactive saga -> archives), park
   stable-diffusion, capture planned work in docs, kick off this saga.
2. **live-loss-sse-panel** -- the in-place live loss chart fed by SSE
   `metric` frames on the connect path, with train/val series and the
   chunked local-path fallback. TDD: frame-to-point reducer and panel
   state tested in `mlpl-web-eval` / `mlpl-web-render-aux`; a connect-mode
   test drives a scripted metric stream.
3. **telemetry-metric-overlay** -- time-align `TelemetryPanel` resource
   sparklines with the live loss panel (shared clock/window), so hardware
   effort and learning progress are read together during a connect train.
4. **docs-pages-retrospective** -- glossary/tour/lesson pointers, literate
   notes, `pages/` rebuild (web source changed), CHANGES refresh, and the
   saga retrospective in `docs/saga.md`.

## Constraints and risks

- Single-threaded WASM blocks during local eval (`docs/worker-threads.md`):
  live-per-step animation is a connect-path feature; the local path stays
  chunked. Do not promise per-step animation in the browser-only demo.
- SSE frame cadence under fast small models can outpace repaint; the panel
  must coalesce frames (append in batches per animation frame).
- Keep the sw-checklist ratchet: new panel code lands as small named
  modules per docs/code_metrics.md gates (25 LOC / 5 fns / 5 modules).

## Relationship to other planned work

- `docs/future-saga-stable-diffusion.md` -- parked 2026-07-25 after its
  step 001; resumes after this saga (conv2d is unaffected by this work).
- `docs/apl2-staging-plan.md` -- APL2 stages 2+ queue behind this saga.
- `docs/plan-for-speculative-decoding.md` -- MLX spec-decoding demo,
  future saga.
- `docs/saga-tech-debt-paydown.md` -- ratchet spikes interleave as usual.
