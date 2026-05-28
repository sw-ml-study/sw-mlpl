# Split components/runtime/ into 6 sub-components (saga 67)

components/runtime/ has 12 crates — way over the 4-crate cap.
Split into themed sub-components:

- components/runtime-core/      mlpl-runtime + mlpl-runtime-core (2)
- components/runtime-element/   mlpl-runtime-math + mlpl-runtime-array (2)
- components/runtime-layers/    mlpl-runtime-conv + mlpl-runtime-rnn + mlpl-runtime-ml (3)
- components/runtime-data/      mlpl-runtime-data (1)
- components/runtime-dr/        mlpl-runtime-dim-reduction + mlpl-runtime-umap + mlpl-runtime-mds-rp (3)
- components/ml-helpers/        mlpl-ml (1)

Each sub-component has 1-3 crates. All under the warning line.
