//! Client-side image upload via the browser's File / Image /
//! Canvas APIs. Saga 29 step 011 follow-up.
//!
//! The user picks a photo of any size and any browser-
//! supported format (PNG / JPEG / WebP). The Canvas API
//! decodes + resizes it to the model's input dimensions
//! (`TARGET_W` x `TARGET_H`); the resulting RGB pixels get
//! normalized to f64 in `[-1, 1]` (matching pets_tiny) and
//! bound under the name `uploaded` in the WASM session.
//! Once bound, the user can `predict_batch(classifier,
//! uploaded)` against any trained model in the same
//! session.
//!
//! No server is involved: the JPEG decoder + bilinear
//! resize happen entirely in the browser, so this works on
//! the static GitHub Pages live demo.

use std::cell::RefCell;
use std::rc::Rc;

use mlpl_wasm::WasmSession;
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{
    CanvasRenderingContext2d, Event, FileReader, HtmlCanvasElement, HtmlImageElement,
    HtmlInputElement, ImageData,
};
use yew::prelude::*;

use crate::state::{EntryKind, HistoryEntry};

const TARGET_H: u32 = 64;
const TARGET_W: u32 = 64;

pub fn make_upload_image(
    session: Rc<RefCell<WasmSession>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
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
        if let Err(err) = start_read(&file, &session, &history) {
            push_error(&history, &format!("{err:?}"));
        }
        // Reset so re-selecting the same file fires `change` again.
        target.set_value("");
    })
}

fn start_read(
    file: &web_sys::File,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
) -> Result<(), JsValue> {
    let reader = FileReader::new()?;
    let reader_clone = reader.clone();
    let session_clone = session.clone();
    let history_clone = history.clone();
    let onload = Closure::wrap(Box::new(move |_: Event| match reader_clone.result() {
        Ok(v) => match v.as_string() {
            Some(url) => {
                if let Err(err) = start_image_load(&url, &session_clone, &history_clone) {
                    push_error(&history_clone, &format!("{err:?}"));
                }
            }
            None => push_error(&history_clone, "FileReader result was not a string"),
        },
        Err(err) => push_error(&history_clone, &format!("{err:?}")),
    }) as Box<dyn FnMut(Event)>);
    reader.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget();
    reader.read_as_data_url(file)?;
    Ok(())
}

fn start_image_load(
    url: &str,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
) -> Result<(), JsValue> {
    let img = HtmlImageElement::new()?;
    let img_clone = img.clone();
    let session_clone = session.clone();
    let history_clone = history.clone();
    let onload = Closure::wrap(Box::new(move |_: Event| {
        if let Err(err) = decode_image_to_session(&img_clone, &session_clone, &history_clone) {
            push_error(&history_clone, &format!("{err:?}"));
        }
    }) as Box<dyn FnMut(Event)>);
    img.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget();
    img.set_src(url);
    Ok(())
}

fn decode_image_to_session(
    img: &HtmlImageElement,
    session: &Rc<RefCell<WasmSession>>,
    history: &UseStateHandle<Vec<HistoryEntry>>,
) -> Result<(), JsValue> {
    let src_w = img.natural_width();
    let src_h = img.natural_height();
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
    ctx.draw_image_with_html_image_element_and_dw_and_dh(
        img,
        0.0,
        0.0,
        f64::from(TARGET_W),
        f64::from(TARGET_H),
    )?;
    let imgdata: ImageData =
        ctx.get_image_data(0.0, 0.0, f64::from(TARGET_W), f64::from(TARGET_H))?;
    let bytes = imgdata.data();
    let h = TARGET_H as usize;
    let w = TARGET_W as usize;
    let mut chw = vec![0.0_f64; 3 * h * w];
    // bytes is RGBA in row-major HWC. Re-arrange to CHW and
    // normalize from u8 [0, 255] -> f64 [-1, 1].
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
        .bind_image("uploaded", &chw, TARGET_H, TARGET_W)?;
    push_success(history, src_w, src_h);
    Ok(())
}

fn push_success(history: &UseStateHandle<Vec<HistoryEntry>>, src_w: u32, src_h: u32) {
    let mut entries: Vec<HistoryEntry> = (**history).clone();
    entries.push(HistoryEntry {
        input: "Image uploaded".to_string(),
        output: format!(
            "Your {src_w}x{src_h} photo was resized to {target_w}x{target_h} by the \
             browser's Canvas API and bound under `uploaded` as a `[1, 3, \
             {target_h}, {target_w}]` tensor (RGB pixels normalized to [-1, 1]). \n\n\
             Try: `uploaded` to inspect; if you have already run the \"Pets: cat vs \
             dog (quick)\" demo, you can classify your photo with `predict_batch(classifier, \
             take(apply(attn, reshape(apply(linear_p, reshape(patchify(uploaded, 16), [16, \
             768])), [1, 16, 128])), 1, 0))` (0 = cat, 1 = dog).",
            target_w = TARGET_W,
            target_h = TARGET_H,
        ),
        is_error: false,
        kind: EntryKind::Narration,
    });
    history.set(entries);
}

fn push_error(history: &UseStateHandle<Vec<HistoryEntry>>, msg: &str) {
    let mut entries: Vec<HistoryEntry> = (**history).clone();
    entries.push(HistoryEntry {
        input: "Image upload failed".to_string(),
        output: msg.to_string(),
        is_error: true,
        kind: EntryKind::Narration,
    });
    history.set(entries);
}
