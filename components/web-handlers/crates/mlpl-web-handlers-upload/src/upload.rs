//! Client-side image upload via the browser's File / Image /
//! Canvas APIs. Saga 29 step 011 follow-up; step 016 rework.
//!
//! Two entry points share the same Canvas decode + resize
//! pipeline:
//!
//! 1. **Button-bound upload** (`make_upload_image`). Wired to
//!    a file `<input type="file">`'s `onchange` event. The
//!    chosen image binds under the name resolved from
//!    `pending_name`: `Some(name)` if a `:upload <name>` REPL
//!    command triggered the picker programmatically, else the
//!    default `"uploaded"` (the legacy button click).
//!
//! 2. **`:upload x` REPL command** (Saga 29 step 016). Routed
//!    in `handlers.rs::make_submit_batch`: setting
//!    `pending_name` to `Some("x")` and programmatically
//!    clicking the file input. The handler stays in the
//!    synchronous chain of the user's Enter keypress, so the
//!    browser accepts the `click()` as a user gesture.
//!
//! Either way, the uploaded image lands as a `Value::Result`
//! in the WASM session: `Ok({pixels: [1, 3, 64, 64], h: 64,
//! w: 64})` on success, `Err("cancelled")` if the picker is
//! dismissed, `Err("...")` for any decode/resize failure.
//! `:upload x` becomes the first in-tree consumer of the
//! `Value::Result` type shipped in Saga 29 step 012.

use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    CanvasRenderingContext2d, Event, FileReader, HtmlCanvasElement, HtmlImageElement,
    HtmlInputElement, ImageData,
};
use yew::prelude::*;

use mlpl_web_eval::state::{EntryKind, HistoryEntry};

const TARGET_H: u32 = 64;
const TARGET_W: u32 = 64;
const DEFAULT_NAME: &str = "uploaded";

/// Caller-chosen variable name for the next upload binding.
/// `None` = the legacy button-click path (binds `"uploaded"`).
/// `Some(name)` = the `:upload <name>` REPL path.
pub type PendingUploadName = UseStateHandle<Option<String>>;

pub fn make_upload_image(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    pending_name: PendingUploadName,
) -> Callback<Event> {
    Callback::from(move |e: Event| {
        let target: HtmlInputElement = match e.target_dyn_into() {
            Some(t) => t,
            None => return,
        };
        let files = match target.files() {
            Some(f) => f,
            None => return,
        };
        if files.length() == 0 {
            return;
        }
        let file = match files.get(0) {
            Some(f) => f,
            None => return,
        };
        let name = pending_name
            .as_ref()
            .map_or_else(|| DEFAULT_NAME.to_string(), Clone::clone);
        pending_name.set(None);
        if let Err(err) = start_read(&file, &session, &history, &name) {
            session
                .borrow()
                .bind_upload_result_err(&name, "decode-init failed");
            push_error(&history, &name, &format!("{err:?}"));
        }
        // Reset so re-selecting the same file fires `change` again.
        target.set_value("");
    })
}

/// Saga 29 step 016: cancel handler for the `<input type=file>`'s
/// `cancel` event. Wires `Err("cancelled")` under the pending
/// variable name (if any) so a `:upload x` command leaves
/// `x = Err("cancelled")` instead of an undefined variable.
///
/// Saga 29 step 017 guard: only fires when `pending_name` is
/// still `Some(_)`. The change-event handler clears
/// `pending_name` the moment it starts processing a real file
/// selection -- so if a stray `cancel` event arrives AFTER a
/// failed decode (some browsers fire both), it sees `None` and
/// skips, avoiding the "decoder failed but message says
/// cancelled" bug where a bad-format file appeared to bind
/// `Err("cancelled")`.
pub fn make_upload_cancel(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    pending_name: PendingUploadName,
) -> Callback<Event> {
    Callback::from(move |_: Event| {
        let Some(name) = pending_name.as_ref().cloned() else {
            return;
        };
        pending_name.set(None);
        session.borrow().bind_upload_result_err(&name, "cancelled");
        push_cancel(&history, &name);
    })
}

fn start_read(
    file: &web_sys::File,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
    name: &str,
) -> Result<(), JsValue> {
    let reader = FileReader::new()?;
    handlers::install_reader_onload(&reader, session, history, name);
    // Saga 29 step 017: FileReader.onerror fires when the file
    // is unreadable (zero bytes, permission issue, abort).
    handlers::install_reader_onerror(&reader, session, history, name);
    reader.read_as_data_url(file)?;
    Ok(())
}

fn start_image_load(
    url: &str,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
    name: &str,
) -> Result<(), JsValue> {
    let img = HtmlImageElement::new()?;
    handlers::install_image_onload(&img, session, history, name, decode_image_to_session);
    // Saga 29 step 017: fires when the browser cannot decode
    // the data URL (e.g. a non-image file renamed to .jpg).
    // Without this handler the upload silently fails and a
    // stray cancel event later binds Err("cancelled"), which
    // misrepresents the actual failure.
    handlers::install_image_onerror(&img, session, history, name);
    img.set_src(url);
    Ok(())
}

fn decode_image_to_session(
    img: &HtmlImageElement,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
    name: &str,
) -> Result<(), JsValue> {
    let (src_w, src_h) = (img.natural_width(), img.natural_height());
    let document = web_sys::window()
        .ok_or_else(|| JsValue::from_str("no window"))?
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")?
        .dyn_into()
        .map_err(|_| JsValue::from_str("not a canvas"))?;
    canvas.set_width(TARGET_W);
    canvas.set_height(TARGET_H);
    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("no 2d context"))?
        .dyn_into()
        .map_err(|_| JsValue::from_str("not 2d context"))?;
    let (tw, th) = (f64::from(TARGET_W), f64::from(TARGET_H));
    ctx.draw_image_with_html_image_element_and_dw_and_dh(img, 0.0, 0.0, tw, th)?;
    let imgdata: ImageData = ctx.get_image_data(0.0, 0.0, tw, th)?;
    let bytes = imgdata.data();
    let (h, w) = (TARGET_H as usize, TARGET_W as usize);
    let mut chw = vec![0.0_f64; 3 * h * w];
    // bytes is RGBA in row-major HWC. Re-arrange to CHW + normalize u8 [0,255] -> f64 [-1,1].
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let v = f64::from(bytes[(y * w + x) * 4 + c]);
                chw[c * h * w + y * w + x] = v / 127.5 - 1.0;
            }
        }
    }
    session
        .borrow()
        .bind_upload_result_ok(name, &chw, TARGET_H, TARGET_W)?;
    push_success(history, name, src_w, src_h);
    Ok(())
}

// History-entry pushers + async-handler installers live in
// sub-modules so the success/cancel/error trio and the four
// install_*_on{load,error} helpers don't blow upload.rs past
// the sw-checklist per-module function-count budget.
use messages::{push_cancel, push_error, push_success};

mod messages {
    use super::{EntryKind, HistoryEntry, TARGET_H, TARGET_W, UseStateHandle};

    pub(super) fn push_success(
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
        src_w: u32,
        src_h: u32,
    ) {
        let body = format!(
            "Your {src_w}x{src_h} photo was resized to {target_w}x{target_h} by the \
             browser's Canvas API and bound as \n\n\
             `{name} = Ok({{pixels: [1, 3, {target_h}, {target_w}], h: {target_h}, w: {target_w}}})`\n\n\
             Inspect: `is_ok({name})` (returns 1), `unwrap({name}).pixels` (the tensor), \
             `unwrap({name}).h` (height), or `svg(unwrap({name}).pixels, \"gallery\")` to view it.\n\n\
             Classify (after running \"Pets: cat vs dog (quick)\" or \"Pets: multi-head ViT (quick + viz)\"):\n\
             `img = unwrap({name}).pixels`\n\
             `predict_batch(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(img, 16), [16, 768])), [1, 16, 128])), 1, 0))` (0 = cat, 1 = dog)",
            target_w = TARGET_W,
            target_h = TARGET_H,
        );
        push(history, "Image upload OK", name, body, false);
    }

    pub(super) fn push_cancel(history: &UseStateHandle<Vec<HistoryEntry>>, name: &str) {
        let body = format!(
            "You closed the file picker without choosing a photo. \
             `{name}` is now bound to `Err(\"cancelled\")`. \
             Check with `is_err({name})` or read the message with `err_message({name})`."
        );
        push(history, "Image upload cancelled", name, body, false);
    }

    pub(super) fn push_error(history: &UseStateHandle<Vec<HistoryEntry>>, name: &str, msg: &str) {
        let body = format!(
            "{msg}\n\n`{name}` is bound to `Err(\"decode failed\")` -- inspect via `err_message({name})`."
        );
        push(history, "Image upload failed", name, body, true);
    }

    fn push(
        history: &UseStateHandle<Vec<HistoryEntry>>,
        title: &str,
        name: &str,
        body: String,
        is_error: bool,
    ) {
        let mut entries: Vec<HistoryEntry> = (**history).clone();
        entries.push(HistoryEntry {
            input: format!("{title} ({name})"),
            output: body,
            is_error,
            kind: EntryKind::Narration,
        });
        history.set(entries);
    }
}

mod handlers {
    use std::cell::RefCell;
    use std::rc::Rc;

    use mlpl_wasm::WasmSession;
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};
    use web_sys::{Event, FileReader, HtmlImageElement};
    use yew::UseStateHandle;

    use super::{HistoryEntry, push_error, start_image_load};

    /// Saga 29 step 016: wire FileReader.onload to call
    /// `start_image_load` with the read-back data URL. Any
    /// read-side failure (non-string result, error event)
    /// binds Err("read failed") under the captured name.
    pub(super) fn install_reader_onload(
        reader: &FileReader,
        session: &Rc<RefCell<WasmSession>>,
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
    ) {
        let reader_clone = reader.clone();
        let session_clone = session.clone();
        let history_clone = history.clone();
        let name_clone = name.to_string();
        let onload = Closure::wrap(Box::new(move |_: Event| match reader_clone.result() {
            Ok(v) => handle_reader_payload(v, &session_clone, &history_clone, &name_clone),
            Err(err) => {
                session_clone
                    .borrow()
                    .bind_upload_result_err(&name_clone, "read failed");
                push_error(&history_clone, &name_clone, &format!("{err:?}"));
            }
        }) as Box<dyn FnMut(Event)>);
        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();
    }

    fn handle_reader_payload(
        v: JsValue,
        session: &Rc<RefCell<WasmSession>>,
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
    ) {
        match v.as_string() {
            Some(url) => {
                if let Err(err) = start_image_load(&url, session, history, name) {
                    session
                        .borrow()
                        .bind_upload_result_err(name, "image-load failed");
                    push_error(history, name, &format!("{err:?}"));
                }
            }
            None => {
                session.borrow().bind_upload_result_err(name, "read failed");
                push_error(history, name, "FileReader result was not a string");
            }
        }
    }

    /// Saga 29 step 017: wire FileReader.onerror to bind
    /// Err("read failed") so an unreadable file doesn't fall
    /// through to a stray cancel event.
    pub(super) fn install_reader_onerror(
        reader: &FileReader,
        session: &Rc<RefCell<WasmSession>>,
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
    ) {
        let session_clone = session.clone();
        let history_clone = history.clone();
        let name_clone = name.to_string();
        let onerror = Closure::wrap(Box::new(move |_: Event| {
            session_clone
                .borrow()
                .bind_upload_result_err(&name_clone, "read failed");
            push_error(
                &history_clone,
                &name_clone,
                "FileReader emitted an error event",
            );
        }) as Box<dyn FnMut(Event)>);
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    }

    /// Saga 29 step 016: wire `<img>` onload to call the
    /// caller-supplied decode pipeline (Canvas resize +
    /// bind_upload_result_ok).
    pub(super) fn install_image_onload<F>(
        img: &HtmlImageElement,
        session: &Rc<RefCell<WasmSession>>,
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
        decode: F,
    ) where
        F: Fn(
                &HtmlImageElement,
                &Rc<RefCell<WasmSession>>,
                &UseStateHandle<Vec<HistoryEntry>>,
                &str,
            ) -> Result<(), JsValue>
            + 'static,
    {
        let img_clone = img.clone();
        let session_clone = session.clone();
        let history_clone = history.clone();
        let name_clone = name.to_string();
        let onload = Closure::wrap(Box::new(move |_: Event| {
            if let Err(err) = decode(&img_clone, &session_clone, &history_clone, &name_clone) {
                session_clone
                    .borrow()
                    .bind_upload_result_err(&name_clone, "decode failed");
                push_error(&history_clone, &name_clone, &format!("{err:?}"));
            }
        }) as Box<dyn FnMut(Event)>);
        img.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();
    }

    /// Saga 29 step 017: wire `<img>` onerror to bind
    /// Err("decode failed: not a valid image") so a
    /// renamed-binary upload doesn't fall through silently.
    pub(super) fn install_image_onerror(
        img: &HtmlImageElement,
        session: &Rc<RefCell<WasmSession>>,
        history: &UseStateHandle<Vec<HistoryEntry>>,
        name: &str,
    ) {
        let session_clone = session.clone();
        let history_clone = history.clone();
        let name_clone = name.to_string();
        let onerror = Closure::wrap(Box::new(move |_: Event| {
            session_clone
                .borrow()
                .bind_upload_result_err(&name_clone, "decode failed: not a valid image");
            push_error(
                &history_clone,
                &name_clone,
                "the browser could not decode this file as an image \
                 (e.g. a non-image file renamed to .jpg)",
            );
        }) as Box<dyn FnMut(Event)>);
        img.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    }
}
