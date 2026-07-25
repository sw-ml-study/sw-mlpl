//! Saga 33 step 006: main-pane composer extracted from
//! `main.rs::render_main`. The original 42-LOC body inlined
//! three panes + a shell-layout match; now each pane is its
//! own helper and `render_main` is a 7-line composer.

use yew::prelude::*;

use mlpl_web_components_content::input_row::InputRow;
use mlpl_web_components_content::welcome::Welcome;
use mlpl_web_eval::state::HistoryEntry;
use mlpl_web_mode::callbacks::ModeCallbacks;
use mlpl_web_paths::view as paths_view;
use mlpl_web_render_aux::editor_panel::EditorPanel;
use mlpl_web_render_aux::entry::render_entry;
use mlpl_web_render_aux::resize_handle::ResizeHandle;
use mlpl_web_render_aux::tutorial::render_tutorial;
use mlpl_web_viz3d::panel::Stage3dPanel;

pub struct MainArgs<'a> {
    pub tutorial_active: bool,
    pub paths_active: bool,
    pub cur_lesson: Option<usize>,
    pub lesson_idx: UseStateHandle<Option<usize>>,
    pub initial_view: mlpl_web_tutorial::TutorialView,
    pub cur_path: Option<(Option<usize>, usize)>,
    pub cb: &'a ModeCallbacks,
    pub history: &'a UseStateHandle<Vec<HistoryEntry>>,
    pub input_value: &'a UseStateHandle<String>,
    pub on_input: Callback<InputEvent>,
    pub on_keydown: Callback<web_sys::KeyboardEvent>,
    pub on_run_example: Callback<String>,
    pub on_run_batch: Callback<Vec<String>>,
    /// Saga 83: Reset callback moved from the top mode bar
    /// into the input row. Clears the active session's history
    /// (tutorial or main, depending on `tutorial_active`).
    pub on_clear: Callback<MouseEvent>,
    pub completion_candidates: Vec<String>,
    pub on_pick_completion: Callback<String>,
    pub completion_selected: usize,
    pub show_3d: bool,
    /// Flip 2D <-> 3D from the input-row toggle. Threaded down
    /// to the shared InputRow so both REPLs (main + tutorial)
    /// carry the control.
    pub on_toggle_3d: Callback<MouseEvent>,
    pub editor_active: bool,
    pub editor_open: UseStateHandle<bool>,
    pub editor_content: UseStateHandle<String>,
}

pub fn render_main(a: MainArgs) -> Html {
    let editor_pane = if a.editor_active {
        render_editor_pane(&a)
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
        on_clear: a.on_clear,
        tutorial_active: a.tutorial_active,
        paths_active: a.paths_active,
        completion_candidates: a.completion_candidates,
        on_pick_completion: a.on_pick_completion,
        completion_selected: a.completion_selected,
        show_3d: a.show_3d,
        on_toggle_3d: a.on_toggle_3d,
    });
    render_main_shell(
        a.tutorial_active,
        a.show_3d,
        editor_pane,
        tutorial_pane,
        paths_pane,
        repl_pane,
    )
}

/// Build the EditorPanel with its 4 callbacks (change, run, clear, save).
fn render_editor_pane(a: &MainArgs) -> Html {
    let content = (*a.editor_content).clone();
    let on_change = {
        let h = a.editor_content.clone();
        Callback::from(move |s: String| h.set(s))
    };
    let on_run = make_editor_run_callback(a);
    let on_clear = {
        let h = a.editor_content.clone();
        Callback::from(move |_: MouseEvent| h.set(String::new()))
    };
    let on_save = make_editor_save_callback(a.editor_content.clone());
    html! { <EditorPanel {content} {on_change} {on_run} {on_save} {on_clear} /> }
}

fn make_editor_run_callback(a: &MainArgs) -> Callback<MouseEvent> {
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
}

fn make_editor_save_callback(ec: UseStateHandle<String>) -> Callback<MouseEvent> {
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
    on_clear: Callback<MouseEvent>,
    tutorial_active: bool,
    paths_active: bool,
    completion_candidates: Vec<String>,
    on_pick_completion: Callback<String>,
    completion_selected: usize,
    show_3d: bool,
    on_toggle_3d: Callback<MouseEvent>,
}

fn render_repl_pane(a: ReplPaneArgs) -> Html {
    // Option<Html> renders as nothing when None.
    let welcome = (!a.tutorial_active && !a.paths_active).then(|| html! { <Welcome /> });
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
                on_clear={a.on_clear}
                in_tutorial={a.tutorial_active}
                completion_candidates={a.completion_candidates}
                on_pick_completion={a.on_pick_completion}
                completion_selected={a.completion_selected}
                show_3d={a.show_3d}
                on_toggle_3d={a.on_toggle_3d}
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
    editor_pane: Html,
    tutorial_pane: Html,
    paths_pane: Html,
    repl_pane: Html,
) -> Html {
    let stage = if show_3d {
        html! { <>
            <ResizeHandle />
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
    // The .output gets height:45vh; stage3d-pane fills the rest
    // (the drag handle lets the user rebalance).
}
