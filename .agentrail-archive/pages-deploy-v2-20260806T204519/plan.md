# Saga: pages-deploy-v2
User direction 2026-08-07: NO rust builds on GitHub at all.
The live demo already deploys the preferred way (pre-built
./pages committed; gh-pages branch publish -- even lighter
than the reference repo's upload-only workflow since no runner
is involved). Remaining deltas: (1) delete the rustup CI
workflow; (2) badges change meaning -- build-pages.sh writes
pages/build-info.json (commit, built_at, bundle, ledger) and
stamps a build meta tag into pages/index.html; README badges
become the runnerless pages-build-deployment badge + shields
dynamic-JSON badges reading the LIVE build-info.json (gates
run locally, recorded at build time); (3) splash badge v2
compares the RUNNING page's baked stamp against BOTH the
serving origin (stale page -> reload) and raw.github main
pages/build-info.json (deploy behind repo -> pending), with
unknown/offline as the default.
## Steps
1. no-rust-on-github -- ci.yml deleted; build-info.json + meta
   stamp in build-pages.sh; README badges reworked; docs.
2. splash-badge-v2 -- three-way verdict (current / reload /
   deploy-pending / unknown), pure logic TDD.
3. close -- deploy, wiki, CLAUDE.md + memory updates, --done.
