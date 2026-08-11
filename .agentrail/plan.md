# Saga: fix-next-steps-list-desc

The demo Next-Steps epilogue mis-describes `:list` as "everything
in the workspace at once". `:list <fn>` actually prints a single
user-defined function's DEFINITION (env.list_fn); it takes a name.
Correct the NEXT_STEPS const in demo.rs to `:list <fn> -- print a
user function's definition`. Verify the other lines are accurate.
Web-visible -> rebuild + deploy.

## Steps
1. fix-and-deploy -- correct NEXT_STEPS; clippy/fmt; build-pages,
   deploy, verify live; --done.
