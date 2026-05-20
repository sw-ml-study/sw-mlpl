Saga 29 step inserted (bugfix after step 016): wire image-decode and file-read error handlers in apps/mlpl-web/src/upload.rs so a non-image file with a .jpg extension (or any file FileReader/Image cannot decode) binds 'name = Err("decode failed: not a valid image")' instead of silently failing or being clobbered by a stray cancel event.

Concrete changes:

1. apps/mlpl-web/src/upload.rs:
   - Wire img.onerror in start_image_load: when set, bind Err("decode failed: not a valid image") under the passed-through name and push the error history entry. The pending_upload_name handle has already been cleared at this point, so the name must come from the closure capture.
   - Wire reader.onerror in start_read similarly: bind Err("read failed") on FileReader error.
   - Add an 'already bound' guard so the cancel event handler does NOT clobber an existing Ok/Err binding: track per-session whether the current pending upload has produced a binding yet (e.g., via the same pending_name handle -- if it was already cleared by the change handler, do not re-bind on a follow-up cancel event). The simplest pattern: if pending_upload_name is None when the cancel handler fires, skip the binding entirely.

2. Tests:
   - Add a mlpl-wasm test that calls bind_upload_result_err('img', 'decode failed: not a valid image') and confirms err_message(img) round-trips.
   - The browser-side onerror path can't be exercised in unit tests; document in the file header.

3. Docs: brief note in the :upload glossary entry that Err can carry 'cancelled', 'decode failed', or 'read failed' messages.

Quality gates: cargo test/clippy/fmt/markdown-checker/sw-checklist (157). Pages rebuild + push (the WASM has the fix). Pre-bug commit was d2c7af7 (saga 29 step 016).