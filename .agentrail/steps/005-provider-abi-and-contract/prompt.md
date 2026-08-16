Fifth step of extensions-event-loop. Define the C-ABI + contract a native provider (demo-extensions wgpu/winit) implements so a real window drives the port, and prove the whole path headlessly. Provide: (1) the thread-safe provider entry shape -- an open that (on the local main-thread launch path only, gated by require_ui_host_thread) starts the provider's UI host and returns a Port handle wired to its internal channels, a poll(port, limit) that returns up to cputime         unlimited
filesize        unlimited
datasize        unlimited
stacksize       7MB
coredumpsize    0kB
addressspace    unlimited
memorylocked    unlimited
maxproc         10666
descriptors     1048576 ordered event records (bounded delivery -- key/pointer/resize/close with a 'kind' field), and a submit(port, command) the UI host applies; document the main-thread / non-blocking-pump / lifecycle rules (winit on main, provider owns the loop, deterministic close/finalize). (2) A #[repr(C)] HEADLESS provider in tests (no winit) that implements the contract over in-process channels and exercises open -> on-handlers -> run -> submit/poll -> close end to end, including bounded delivery (poll returns at most cputime         unlimited
filesize        unlimited
datasize        unlimited
stacksize       7MB
coredumpsize    0kB
addressspace    unlimited
memorylocked    unlimited
maxproc         10666
descriptors     1048576) and the local-only guard (open over a non-UI env errors). Reuse the shipped extension C-ABI (mlpl-extension-cabi) + the Port/dispatch builtins; do not add winit to sw-mlpl. Keep functions <=25 lines; keep sw-checklist fails/warns decreasing (extension-cabi + mlpl-eval-env crate splits are queued if module ceilings block clean placement). Gate, commit, complete queuing docs-demo-close.