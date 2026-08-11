# Saga: footer-repo-dialog

The footer "GitHub" link navigates straight to sw-mlpl. Change it
to open a DIALOG listing sw-mlpl PLUS every sw-ml-study/demo* repo,
each linking to its GitHub page. Authoritative repo names + one-line
descriptions come from the org (baked in as a const, since the WASM
page cannot query GitHub at runtime; the local `demo-funtional-*`
dir is a typo -- use the org name `demo-functional-pipelines`).

Repos (org names): sw-mlpl, demo-algorithms, demo-combinators,
demo-extensions, demo-file-processing, demo-functional-pipelines,
demo-memory, demo-ml-utils.

Implementation: a self-contained RepoLinks component (own
use_state open/close) placed IN footer.rs (chrome crate is at 7
modules -- a new module would hit the crate-module-count FAIL).
Reuse the existing .modal / .modal-backdrop / .modal-header /
.close-btn / .modal-body CSS (as DocDialog does) + a compact
variant and repo-row styling. The footer "GitHub" link becomes the
dialog trigger (keeps an href to sw-mlpl as a no-JS fallback);
drop the now-unused FooterProps.url + REPO_URL.

## Steps
1. repo-dialog -- RepoLinks in footer.rs (trigger + modal + repo
   rows, const REPO list), footer wiring, index.html CSS; drop
   url prop/REPO_URL; clippy/fmt/tests green.
2. deploy -- build-pages.sh, commit pages/, deploy-pages.sh, verify
   live + trigger fresh Pages build if stalled, --done.
