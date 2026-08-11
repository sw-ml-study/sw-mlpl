# Saga: demo-next-steps-block

Follow-up: the "Next Steps?" epilogue currently renders INSIDE the
"What just happened" (takeaway) narration panel. The user wants it
as a SEPARATE narration block, for every demo.

Fix: in demo.rs schedule_demo_line's end-of-demo block, keep the
takeaway as its own panel (input "What just happened") and add a
SEPARATE Next-Steps narration panel via the existing
running::push_narration (NEXT_STEPS text, its first line is the
"Next steps?" header). No new function (demo.rs + running.rs are
both at their 7-fn max); push_narration already renders a
standalone narration entry.

## Steps
1. separate-block -- revert the takeaway output to demo.takeaway;
   add push_narration(NEXT_STEPS) as a second panel; keep
   schedule_demo_line <= 50 LOC; clippy/fmt.
2. deploy -- build-pages, commit pages/, deploy, verify, --done.
