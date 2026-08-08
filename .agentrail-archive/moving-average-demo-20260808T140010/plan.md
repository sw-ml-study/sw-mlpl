# Saga: moving-average-demo

Add a J-inspired moving-average demo to the APL2 group,
from docs/sw_mlpl_moving_average_concise.mlpl, with detailed
comments. Frames the J windowed adverb vs MLPL prefix-sum
identity; highlights running_sum (scan), compress (APL
mask-select), concat, whole-array arithmetic, no loop.

## Steps
1. add-demo -- new [[demos]] entry in demos.toml (APL2 group);
   README demo count 85->86; smoke test runs it; gates.
2. close -- rebuild pages, deploy, verify live, --done.
