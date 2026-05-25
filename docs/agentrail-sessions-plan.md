# `.agentrail/sessions/` migration & compaction plan

`agentrail complete` snapshots the Claude Code session transcript into
`.agentrail/sessions/<session-uuid>.jsonl`. The file grows ~400-500
lines (~700 KB-1 MB) per completed step. Current session
(862332c6...) is at 68 MB / 35,559 lines as of saga 33 step 035.

GitHub warns at 50 MB and rejects at 100 MB. We need a plan
before we get there.

## Triggers

Watch the live session file with:

```bash
ls -lh .agentrail/sessions/*.jsonl | sort -k5 -h | tail -1
```

- **Yellow (>= 60 MB)**: skim the plan; pick the archive destination.
- **Red (>= 95 MB)**: execute the migration immediately. Do not
  push past 100 MB.

## Archive destination (no git-lfs)

Avoid git-lfs (bandwidth fees + workflow complexity). Local-disk
+ off-machine backup is enough.

**Default target:** `~/agentrail-archive/<repo>/<session-uuid>.jsonl`

- Same machine, outside the repo tree so git never sees it.
- Already covered by the user's existing Time Machine / iCloud
  backup. (If TM is not configured: rsync nightly to an
  external SSD or `~/Documents/`.)

**Future option:** Backblaze B2 or S3 bucket via `aws s3 cp`.
~$0.005/GB/month at B2; sessions are ~70 MB/year-per-saga so
cents/year. Defer until disk pressure is real.

## Compaction (derive concise records before archiving)

The full jsonl is noisy: every tool call, every tool result, every
file diff, every system reminder. The useful signals to keep:

1. **Per-step rewards** -- already captured in
   `.agentrail/trajectories/feature/run_*.json`. These ARE what
   `agentrail next` reads as "Past Successes". No work needed here.

2. **Step-level summaries** -- each step's `summary.md` (already
   committed under `.agentrail/steps/<step>/summary.md`). The
   pithy 1-2 line distillation.

3. **Failure modes / dead-ends** -- the tool errors, the
   "we tried X, it didn't work because Y" moments. The full
   jsonl has these buried in tool-result content. A simple
   extractor:

   ```bash
   jq -r 'select(.type=="user" and (.message.content[0].is_error // false))
          | .message.content[0].content' < session.jsonl > errors.txt
   ```

   Yields the failure list. Append to the session's `summary.md`
   under a `## Lessons learned` heading before archiving.

4. **User-injected requirements** -- user messages mid-session
   ("file it as a new saga step", "add a perplexity builtin",
   "consider a new path") that drove unplanned work. Extract:

   ```bash
   jq -r 'select(.type=="user" and (.message.content | type == "string"))
          | .message.content' < session.jsonl > user-prompts.txt
   ```

   The trajectory record currently captures the formal step
   prompt but not these mid-session course corrections.

Once 1-4 are derived, the raw jsonl can be archived.

## Migration procedure

Sequenced commands. Run when a session file approaches 95 MB.

```bash
# 1. Snapshot current .agentrail/ to a refs/agentrail/snapshots/
#    ref so the in-progress state survives.
agentrail snapshot

# 2. Derive concise records from the raw session (see above).
#    Write outputs into each step's summary.md.
./scripts/compact-session.sh \
  .agentrail/sessions/<session-uuid>.jsonl   # TBD: write this script

# 3. Copy the raw jsonl to the archive destination.
mkdir -p ~/agentrail-archive/sw-mlpl
cp .agentrail/sessions/<session-uuid>.jsonl \
   ~/agentrail-archive/sw-mlpl/

# 4. Verify the copy survived (size + line count match).
ls -lh ~/agentrail-archive/sw-mlpl/<session-uuid>.jsonl
wc -l ~/agentrail-archive/sw-mlpl/<session-uuid>.jsonl

# 5. Add .agentrail/sessions/ to .gitignore.
echo ".agentrail/sessions/" >> .gitignore
git add .gitignore
git commit -m "chore(agentrail): gitignore sessions/, archive raw transcripts off-repo"

# 6. Remove the file from tracked state (still on disk).
git rm --cached .agentrail/sessions/<session-uuid>.jsonl
git commit -m "chore(agentrail): untrack large session jsonl (now archived)"

# 7. Strip the large blob from history with git-filter-repo.
#    git-filter-repo is preferred over BFG; install via:
#      brew install git-filter-repo
git filter-repo --path .agentrail/sessions/ --invert-paths

# 8. Force-push the rewritten history. Coordinate with
#    collaborators (this is a destructive rewrite of main).
#    Solo: just push.
git push --force-with-lease origin main
```

## Open questions for future iteration

- **Whether the trajectory data is enough**, or whether we
  actually want richer per-step retrospectives (chain-of-thought
  snippets, key file diffs) before discarding the raw jsonl.
  Step 1-4 of compaction should answer this empirically: if the
  derived summaries are sufficient for postmortems, archiving
  the raw is safe; if not, refine the extractor first.
- **One archive per session or one consolidated** (e.g. a yearly
  rollup tarball). Default: one-per-session for now; switch if
  the archive directory gets unwieldy.
- **Compaction automation**. The `compact-session.sh` script
  referenced above doesn't exist yet -- write it when we
  actually need to compact, not preemptively.

## Cross-references

- `.agentrail/trajectories/feature/run_*.json` -- the structured
  signal that `agentrail next` reads. Already concise.
- `.agentrail/steps/<step>/summary.md` -- per-step pithy
  retrospective committed at `agentrail complete` time.
- `CLAUDE.md` agentrail discipline sections -- the workflow this
  archive plan supports.
