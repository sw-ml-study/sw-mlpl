# Split mlpl-session/ (10 crates) into sparse sub-components (saga 70)

- session-infra (3):  mlpl-bpe-core, mlpl-env-traits, mlpl-loader-helpers
- models-read (3):    mlpl-models-feasibility, mlpl-models-freeze, mlpl-models-inspect
- models-write (3):   mlpl-models-mutate, mlpl-models-tape, mlpl-models-tune
- models-llm (1):     mlpl-models-llm
