# Git history rewrite + force-push -- 2026-08-11

**Action required for anyone with an existing clone: re-clone or hard-reset.**

## What happened

On 2026-08-11 the `main` branch history was rewritten with
`git filter-branch` to purge the `.agentrail/**/sessions/*.jsonl`
raw session transcripts from ALL history, and `main` (plus all
release tags) was force-pushed.

Why: those session logs are append-only Claude Code event
transcripts (one reached 72 MB) that were committed 100+ times,
bloating `.git` to 2.2 GB and tripping GitHub's large-file warning
(with a 100 MB hard-block looming). They are not part of the
durable saga handoff (`steps/`, `trajectories/`, `plan.md`,
`saga.toml` stay tracked), so they were dropped from tracking
(gitignored) and purged from history.

Result: `.git` shrank from 2.2 GB to 614 MB. Every commit SHA
changed; all 24 tags were re-pointed and force-pushed.

## What you must do

An existing local clone now has divergent history. Easiest fix is a
fresh clone. To keep an existing checkout instead:

```bash
git fetch origin
git reset --hard origin/main
git fetch --tags --force
```

If you have local branches or unpushed work, rebase them onto the
new `main` (their old base commits no longer exist):

```bash
git rebase --onto origin/main <old-main-sha> <your-branch>
```

## Going forward

`.agentrail/sessions/` and `.agentrail-archive/*/sessions/` are
gitignored. Session transcripts stay on disk locally for replay but
are never committed. The structured saga record remains tracked and
committed as before.
