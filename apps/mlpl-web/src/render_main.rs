//! Saga 33 step 006: main-pane composer extracted from
//! `main.rs::render_main`. The original 42-LOC body inlined
//! three panes + a shell-layout match; now each pane is its
//! own helper and `render_main` is a 7-line composer.

use yew::prelude::*;

use crate::components::{InputRow, Welcome};
use crate::editor_panel::EditorPanel;
use crate::entry_render::render_entry;
use crate::mode_callbacks::ModeCallbacks;
use crate::paths_view;
use crate::render_tutorial::render_tutorial;
use crate::viz3d_panel::Stage3dPanel;
use mlpl_web_eval::state::HistoryEntry;

pub struct MainArgs<'a> {
    pub tutorial_active: bool,
    pub paths_active: bool,
    pub cur_lesson: Option<usize>,
    pub lesson_idx: UseStateHandle<Option<usize>>,
    pub initial_view: crate::components::TutorialView,
    pub cur_path: Option<(Option<usize>, usize)>,
    pub cb: &'a ModeCallbacks,
    pub history: &'a UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: &'a UseStateHandle<String>,
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<web_sys::KeyboardEvent>,
    pub on_run_example: Callback<String>,
    pub on_run_batch: Callback<Vec<String>>,
    pub completion_candidates: Vec<String>,
    pub on_pick_completion: Callback<String>,
    pub completion_selected: usize,
    pub show_3d: bool,
    pub editor_active: bool,
    pub editor_open: UseStateHandle<bool>,
    pub editor_content: UseStateHandle<String>,
}

pub fn render_main(a: MainArgs) -> Html {
    let editor_pane = if a.editor_active {
        let content = (*a.editor_content).clone();
        let on_change = {
            let h = a.editor_content.clone();
            Callback::from(move |s: String| h.set(s))
        };
        let on_run = {
            let batch = a.on_run_batch.clone();
            let ec = a.editor_content.clone();
            let eo = a.editor_open.clone();
            Callback::from(move |_: MouseEvent| {
                let lines: Vec<String> = (*ec)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .collect();
                if !lines.is_empty() {
                    eo.set(false);
                    batch.emit(lines);
                }
            })
        };
        let on_clear = {
            let h = a.editor_content.clone();
            Callback::from(move |_: MouseEvent| h.set(String::new()))
        };
        let on_save = {
            let ec = a.editor_content.clone();
            Callback::from(move |_: MouseEvent| {
                let text = (*ec).clone();
                if text.is_empty() {
                    return;
                }
                let _ = js_sys::eval(&format!(
                    "{{const b=new Blob([decodeURIComponent('{}')],{{type:'text/plain'}});\
                     const a=document.createElement('a');a.href=URL.createObjectURL(b);\
                     a.download='session.mlpl';a.click();URL.revokeObjectURL(a.href);}}",
                    js_sys::encode_uri_component(&text)
                ));
            })
        };
        html! { <EditorPanel {content} {on_change} {on_run} {on_save} {on_clear} /> }
    } else {
        html! {}
    };
    let tutorial_pane = render_tutorial(
        a.cur_lesson,
        a.lesson_idx,
        a.initial_view,
        a.on_run_example,
        a.on_run_batch,
        a.cur_path.is_some(),
    );
    let paths_pane = render_paths_pane(a.cur_path, a.cb);
    let repl_pane = render_repl_pane(ReplPaneArgs {
        history: a.history,
        input_value: a.input_value,
        on_input: a.on_input,
        on_keydown: a.on_keydown,
        tutorial_active: a.tutorial_active,
        paths_active: a.paths_active,
        completion_candidates: a.completion_candidates,
        on_pick_completion: a.on_pick_completion,
        completion_selected: a.completion_selected,
    });
    render_main_shell(
        a.tutorial_active,
        a.show_3d,
        a.editor_active,
        editor_pane,
        tutorial_pane,
        paths_pane,
        repl_pane,
    )
}

/// PathsView render: thin wrapper so the parent doesn't need to
/// know about `paths_view::PathsView`'s prop shape.
fn render_paths_pane(cur_path: Option<(Option<usize>, usize)>, cb: &ModeCallbacks) -> Html {
    html! {
        <paths_view::PathsView
            state={cur_path}
            on_change={cb.path_change.clone()}
            on_open_lesson={cb.path_open_lesson.clone()}
            on_run_demo={cb.path_run_demo.clone()}
        />
    }
}

struct ReplPaneArgs<'a> {
    history: &'a UseStateHandle<Vec<HistoryEntry>>,
    input_value: &'a UseStateHandle<String>,
    on_input: Callback<InputEvent>,
    on_keydown: Callback<web_sys::KeyboardEvent>,
    tutorial_active: bool,
    paths_active: bool,
    completion_candidates: Vec<String>,
    on_pick_completion: Callback<String>,
    completion_selected: usize,
}

fn render_repl_pane(a: ReplPaneArgs) -> Html {
    let welcome = if a.tutorial_active || a.paths_active {
        html! {}
    } else {
        html! { <Welcome /> }
    };
    let value = (**a.input_value).clone();
    html! {
        <>
            <div id="output" class="output">
                { welcome }
                { for a.history.iter().map(render_entry) }
            </div>
            <InputRow
                {value}
                on_input={a.on_input}
                on_keydown={a.on_keydown}
                in_tutorial={a.tutorial_active}
                completion_candidates={a.completion_candidates}
                on_pick_completion={a.on_pick_completion}
                completion_selected={a.completion_selected}
            />
        </>
    }
}

/// Layout shell: when the tutorial is active, render a 2-pane
/// split (tutorial + repl); otherwise stack the panes in a
/// single `<main>` container.
fn render_main_shell(
    tutorial_active: bool,
    show_3d: bool,
    editor_active: bool,
    editor_pane: Html,
    tutorial_pane: Html,
    paths_pane: Html,
    repl_pane: Html,
) -> Html {
    let stage = if show_3d {
        html! { <>
            <div class="resize-handle" id="viz3d-resize" />
            <section class="stage3d-pane"><Stage3dPanel /></section>
        </> }
    } else {
        html! {}
    };
    if tutorial_active {
        html! {
            <main class="tutorial-split">
                <section class="tutorial-pane">{ tutorial_pane }</section>
                <section class="repl-pane">{ repl_pane }{ stage }</section>
            </main>
        }
    } else {
        let cls = if show_3d { "viz3d-split" } else { "" };
        html! {
            <main class={cls}>
                { editor_pane }
                { tutorial_pane }
                { paths_pane }
                { repl_pane }
                { stage }
            </main>
        }
    }
    // Note: in viz3d-split mode, CSS flex-direction:column
    // stacks output, input-wrap, and stage3d-pane vertically.
    // The .output gets max-height:40vh; stage3d-pane fills the rest.
}
