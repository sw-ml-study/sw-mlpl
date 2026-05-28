# mlpl-web

## Purpose
Yew/WASM visual environment for stepping through MLPL traces.

## Deploying to the live demo

The live demo at <https://sw-ml-study.github.io/sw-mlpl/> is served
from the committed `./pages/` directory at the repo root. The GitHub
Pages workflow (`.github/workflows/pages.yml`) does NOT build from
source -- it only uploads `pages/`.

After any change in this crate, rebuild and commit the output:

```bash
./scripts/build-pages.sh
git add pages/
git commit -m "chore(pages): rebuild for <what changed>"
git push
```
