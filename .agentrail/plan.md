# Saga: status-badges-emacs
User direction 2026-08-06: (1) splash/banner staleness badge --
compare the RUNNING bundle against the serving origin's own
index.html (fresh / stale-reload / unknown-offline default per
user: airplane mode, stale page), plus best-effort deploying
badge from the Pages build API on the live origin; (2) README
build/deploy/tests badges (pages-build-deployment badge + a
fast CI test workflow); (3) editors/emacs/mlpl-mode.el --
font-lock (three name kinds, @annotations), indentation, run
buffer, run tests via --test-events JSONL, imenu.
## Steps
1. banner-badge -- staleness/deploy badge in the web banner
   with offline default. TDD where pure.
2. readme-badges -- ci.yml (fast representative suite) +
   badges in README.
3. emacs-mode -- mlpl-mode.el + docs note.
4. close -- rebuild/deploy, wiki, q-and-a, queue advance.
