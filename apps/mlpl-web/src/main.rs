// On a native (non-wasm) target the entire Yew/wasm-bindgen frontend
// is dead code -- only the stub `main` below runs. Suppress the
// resulting dead_code / unused-import warnings so `cargo build` and
// `cargo clippy -D warnings` stay clean for both targets.
#![cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, unused_imports, unused_variables)
)]

mod components;
mod demos;
mod diagrams_view;
mod entry_render;
mod eval;
mod eval_sse;
mod eval_url;
#[cfg(target_arch = "wasm32")]
mod eval_wasm;
mod glossary_view;
mod handlers;
mod help;
mod intro_md;
mod lessons;
mod lessons_advanced;
mod paths;
mod paths_view;
#[cfg(test)]
mod readme_counts;
mod state;
mod summary;
mod tutorial;
mod upload;
mod upload_cmd;

use components::{
    DocDialog, Footer, GithubCorner, Header, HeaderMode, InputRow, ModeBar, TutorialPanel,
    TutorialPanelProps, TutorialView, Welcome,
};
use entry_render::render_entry;
use handlers::{
    EvalDeps, make_clear, make_keydown, make_oninput, make_run_demo, make_submit,
    make_submit_batch, toggle_bool,
};
use mlpl_wasm::WasmSession;
use state::HistoryEntry;
use tutorial::{jump_lesson, run_example, step_lesson};
use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;
use yew::prelude::*;

const REPO_URL: &str = "https://github.com/sw-ml-study/sw-mlpl";

#[function_component(App)]
fn app() -> Html {
    // Isolated scratchpads: main (session+history) vs tutorial.
    let session = use_mut_ref(WasmSession::new);
    let tutorial_session = use_mut_ref(WasmSession::new);
    // Saga 21.5 step 006: parse the connect-mode URL from
    // `window.location.search` and log it for visibility. The
    // actual evaluator swap into the REPL flow lands in step
    // 007 alongside the streaming SSE plumbing. Until then,
    // the eval path runs entirely in WASM regardless of the
    // query parameter -- but the trait + native tests in
    // `apps/mlpl-web/tests/connect_mode_tests.rs` already
    // exercise the REST path end-to-end against an in-process
    // `mlpl-serve`.
    #[cfg(target_arch = "wasm32")]
    if let Some(url) = eval::current_connect_url_from_window() {
        web_sys::console::log_1(
            &format!("[mlpl-web] ?connect={url} parsed (wiring in step 007)").into(),
        );
    }
    let history = use_state(Vec::<HistoryEntry>::new);
    let tutorial_history = use_state(Vec::<HistoryEntry>::new);
    let input_value = use_state(String::new);
    let cmd_history = use_state(Vec::<String>::new);
    let cmd_index = use_state(|| None::<usize>);
    let dialog_open = use_state(|| false);
    let lesson_idx = use_state(|| None::<usize>);
    // Which sub-tab the TutorialPanel should land on the
    // next time it mounts. Header "Tutorial" click sets
    // this to Toc; Paths "Open lesson X" sets it to Lesson
    // (because the user already chose a specific lesson
    // and we want the content view, not the index list).
    let tutorial_initial_view = use_state(|| TutorialView::Toc);
    // Paths-mode state: outer None = not in paths mode; inner
    // (None, _) = picker; inner (Some(p), s) = walking path p
    // at step s. See paths_view::PathsView.
    let path_state = use_state(|| None::<(Option<usize>, usize)>);
    let in_tutorial = lesson_idx.is_some();
    let active_session = if in_tutorial {
        tutorial_session
    } else {
        session.clone()
    };
    let active_history: UseStateHandle<Vec<HistoryEntry>> = if in_tutorial {
        tutorial_history.clone()
    } else {
        history.clone()
    };

    // Saga 29 step 016: state for the `:upload <name>` REPL
    // command. The NodeRef points at the hidden <input
    // type=file> rendered in components.rs; the pending name
    // bridges the slash-command handler (which is sync in the
    // keydown gesture) and the file picker's async onchange
    // (which fires after the user picks a file).
    let upload_input_ref = use_node_ref();
    let pending_upload_name = use_state(|| None::<String>);

    let deps = EvalDeps {
        session: active_session.clone(),
        history: active_history.clone(),
        input_value: input_value.clone(),
        cmd_history: cmd_history.clone(),
        cmd_index: cmd_index.clone(),
        upload_input_ref: upload_input_ref.clone(),
        pending_upload_name: pending_upload_name.clone(),
    };
    let on_submit = make_submit(deps.clone());
    let on_run_batch = make_submit_batch(deps);
    let on_clear = make_clear(active_session.clone(), active_history.clone());
    let on_upload = upload::make_upload_image(
        active_session.clone(),
        active_history.clone(),
        pending_upload_name.clone(),
    );
    let on_upload_cancel = upload::make_upload_cancel(
        active_session.clone(),
        active_history.clone(),
        pending_upload_name,
    );
    let on_demo = make_run_demo(active_session, active_history.clone());

    use_effect_with(active_history.clone(), |_| {
        scroll_and_focus();
        || ()
    });

    render(RenderArgs {
        on_submit,
        on_run_batch,
        on_clear,
        on_demo,
        on_upload,
        on_upload_cancel,
        upload_input_ref,
        input_value,
        cmd_history,
        cmd_index,
        history: active_history,
        dialog_open,
        lesson_idx,
        tutorial_initial_view,
        path_state,
    })
}

struct RenderArgs {
    on_submit: Callback<String>,
    on_run_batch: Callback<Vec<String>>,
    on_clear: Callback<MouseEvent>,
    on_demo: Callback<usize>,
    on_upload: Callback<web_sys::Event>,
    on_upload_cancel: Callback<web_sys::Event>,
    upload_input_ref: NodeRef,
    input_value: UseStateHandle<String>,
    cmd_history: UseStateHandle<Vec<String>>,
    cmd_index: UseStateHandle<Option<usize>>,
    history: UseStateHandle<Vec<HistoryEntry>>,
    dialog_open: UseStateHandle<bool>,
    lesson_idx: UseStateHandle<Option<usize>>,
    tutorial_initial_view: UseStateHandle<TutorialView>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
}

fn render(a: RenderArgs) -> Html {
    let on_input = make_oninput(a.input_value.clone());
    let on_keydown = make_keydown(
        a.on_submit.clone(),
        a.input_value.clone(),
        a.cmd_history,
        a.cmd_index,
    );
    let open_dialog = toggle_bool(a.dialog_open.clone(), true);
    let close_dialog = toggle_bool(a.dialog_open.clone(), false);
    let cur_lesson = *a.lesson_idx;
    let cur_path = *a.path_state;
    let tutorial_active = cur_lesson.is_some();
    let paths_active = cur_path.is_some();
    let header_mode = if paths_active {
        HeaderMode::Paths
    } else if tutorial_active {
        HeaderMode::Tutorial
    } else {
        HeaderMode::Repl
    };
    let cb = mode_callbacks(
        a.lesson_idx.clone(),
        a.path_state.clone(),
        a.tutorial_initial_view.clone(),
        a.on_demo.clone(),
    );
    let on_run_example = run_example(a.on_submit.clone(), a.input_value.clone());

    html! {
        <>
            <GithubCorner url={REPO_URL} />
            <Header
                on_help={open_dialog}
                on_select_repl={cb.repl.clone()}
                on_select_tutorial={cb.tutorial.clone()}
                on_select_paths={cb.paths.clone()}
                mode={header_mode}
            />
            <ModeBar
                on_clear={a.on_clear}
                on_demo={a.on_demo}
                on_upload={a.on_upload}
                on_upload_cancel={a.on_upload_cancel}
                upload_input_ref={a.upload_input_ref}
                {tutorial_active}
            />
            { render_main(MainArgs {
                tutorial_active,
                paths_active,
                cur_lesson,
                lesson_idx: a.lesson_idx.clone(),
                initial_view: *a.tutorial_initial_view,
                cur_path,
                cb: &cb,
                history: &a.history,
                input_value: &a.input_value,
                on_input,
                on_keydown,
                on_run_example,
                on_run_batch: a.on_run_batch,
            }) }
            <Footer url={REPO_URL} />
            <DocDialog open={*a.dialog_open} on_close={close_dialog} />
        </>
    }
}

/// Bundle of mode-switch + paths-callback handlers built
/// from the shared lesson/path state. Extracted so the
/// `render` body stays under the function-LOC budget; the
/// callback closures are otherwise just plumbing.
struct ModeCallbacks {
    repl: Callback<MouseEvent>,
    tutorial: Callback<MouseEvent>,
    paths: Callback<MouseEvent>,
    path_change: Callback<Option<(Option<usize>, usize)>>,
    path_open_lesson: Callback<usize>,
    path_run_demo: Callback<String>,
}

fn mode_callbacks(
    lesson_idx: UseStateHandle<Option<usize>>,
    path_state: UseStateHandle<Option<(Option<usize>, usize)>>,
    tutorial_view: UseStateHandle<TutorialView>,
    on_demo: Callback<usize>,
) -> ModeCallbacks {
    let repl = {
        let l = lesson_idx.clone();
        let p = path_state.clone();
        Callback::from(move |_| {
            l.set(None);
            p.set(None);
        })
    };
    let tutorial = {
        let l = lesson_idx.clone();
        let p = path_state.clone();
        let v = tutorial_view.clone();
        Callback::from(move |_| {
            p.set(None);
            if l.is_none() {
                v.set(TutorialView::Toc);
                l.set(Some(0));
            }
        })
    };
    let paths = {
        let l = lesson_idx.clone();
        let p = path_state.clone();
        Callback::from(move |_| {
            l.set(None);
            if p.is_none() {
                p.set(Some((None, 0)));
            }
        })
    };
    let path_change = {
        let p = path_state.clone();
        Callback::from(move |next| p.set(next))
    };
    let path_open_lesson = {
        let l = lesson_idx.clone();
        let p = path_state.clone();
        let v = tutorial_view;
        Callback::from(move |i: usize| {
            p.set(None);
            v.set(TutorialView::Lesson);
            l.set(Some(i));
        })
    };
    let path_run_demo = {
        let p = path_state;
        Callback::from(move |name: String| {
            if let Some(idx) = demos::DEMOS.iter().position(|d| d.name == name) {
                p.set(None);
                on_demo.emit(idx);
            }
        })
    };
    ModeCallbacks {
        repl,
        tutorial,
        paths,
        path_change,
        path_open_lesson,
        path_run_demo,
    }
}

struct MainArgs<'a> {
    tutorial_active: bool,
    paths_active: bool,
    cur_lesson: Option<usize>,
    lesson_idx: UseStateHandle<Option<usize>>,
    initial_view: TutorialView,
    cur_path: Option<(Option<usize>, usize)>,
    cb: &'a ModeCallbacks,
    history: &'a UseStateHandle<Vec<HistoryEntry>>,
    input_value: &'a UseStateHandle<String>,
    on_input: Callback<InputEvent>,
    on_keydown: Callback<web_sys::KeyboardEvent>,
    on_run_example: Callback<String>,
    on_run_batch: Callback<Vec<String>>,
}

fn render_main(a: MainArgs) -> Html {
    let tutorial_pane = render_tutorial(
        a.cur_lesson,
        a.lesson_idx,
        a.initial_view,
        a.on_run_example,
        a.on_run_batch,
    );
    let paths_pane = html! {
        <paths_view::PathsView
            state={a.cur_path}
            on_change={a.cb.path_change.clone()}
            on_open_lesson={a.cb.path_open_lesson.clone()}
            on_run_demo={a.cb.path_run_demo.clone()}
        />
    };
    let repl_pane = html! {
        <>
            <div id="output" class="output">
                { if a.tutorial_active || a.paths_active { html!{} } else { html!{ <Welcome /> } } }
                { for a.history.iter().map(render_entry) }
            </div>
            <InputRow value={(**a.input_value).clone()} on_input={a.on_input} on_keydown={a.on_keydown} in_tutorial={a.tutorial_active} />
        </>
    };
    if a.tutorial_active {
        html! {
            <main class="tutorial-split">
                <section class="tutorial-pane">{ tutorial_pane }</section>
                <section class="repl-pane">{ repl_pane }</section>
            </main>
        }
    } else {
        html! {
            <main>
                { tutorial_pane }
                { paths_pane }
                { repl_pane }
            </main>
        }
    }
}

fn render_tutorial(
    cur: Option<usize>,
    lesson: UseStateHandle<Option<usize>>,
    initial_view: TutorialView,
    on_run_example: Callback<String>,
    on_run_batch: Callback<Vec<String>>,
) -> Html {
    let Some(idx) = cur else {
        return html! {};
    };
    let on_prev = step_lesson(lesson.clone(), -1);
    let on_next = step_lesson(lesson.clone(), 1);
    let on_jump = jump_lesson(lesson.clone());
    let on_close = Callback::from(move |_| lesson.set(None));
    let props = TutorialPanelProps {
        lesson_idx: idx,
        on_prev,
        on_next,
        on_jump,
        on_run_example,
        on_run_batch,
        on_close,
        initial_view,
    };
    html! { <TutorialPanel ..props /> }
}

fn scroll_and_focus() {
    if let Some(window) = web_sys::window()
        && let Some(document) = window.document()
    {
        if let Some(el) = document.get_element_by_id("output") {
            el.set_scroll_top(el.scroll_height());
        }
        if let Some(el) = document.get_element_by_id("repl-input")
            && let Ok(input) = el.dyn_into::<HtmlInputElement>()
        {
            let _ = input.focus();
        }
    }
}

// mlpl-web is a Yew/WASM bundle served via `trunk serve` (or built
// into ./pages/ via scripts/build-pages.sh). It is *not* a native
// CLI -- linking succeeds for the host target but the resulting
// binary panics on first call into js-sys because the imported
// statics only resolve inside a wasm runtime. Provide a friendly
// stub when invoked as a native binary so users get a clear error
// instead of a stack trace, and skip mlpl-web from sw-install via
// `sw-install -p . --bin mlpl-repl --bin mlpl-lab`.

#[cfg(target_arch = "wasm32")]
fn main() {
    yew::Renderer::<App>::new().render();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "mlpl-web is a WASM/Yew bundle, not a native CLI.\n\
         \n\
         To run it locally:\n\
           cd apps/mlpl-web && trunk serve\n\
         \n\
         To rebuild the deployed pages/ bundle:\n\
           ./scripts/build-pages.sh\n\
         \n\
         Or visit the live demo: https://sw-ml-study.github.io/sw-mlpl/"
    );
    std::process::exit(1);
}
