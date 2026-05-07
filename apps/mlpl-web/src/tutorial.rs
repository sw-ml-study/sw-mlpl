use yew::prelude::*;

pub use crate::lessons::LESSONS;

pub fn step_lesson(
    lesson: UseStateHandle<Option<usize>>,
    delta: i32,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if let Some(cur) = *lesson {
            let next = i32::try_from(cur).unwrap_or(0) + delta;
            if next >= 0 {
                let next_usize = next as usize;
                if next_usize < LESSONS.len() {
                    lesson.set(Some(next_usize));
                }
            }
        }
    })
}

pub fn jump_lesson(lesson: UseStateHandle<Option<usize>>) -> Callback<usize> {
    Callback::from(move |idx: usize| {
        if idx < LESSONS.len() {
            lesson.set(Some(idx));
        }
    })
}

pub fn run_example(
    on_submit: Callback<String>,
    input_value: UseStateHandle<String>,
) -> Callback<String> {
    Callback::from(move |line: String| {
        input_value.set(line.clone());
        on_submit.emit(line);
    })
}

#[derive(Clone, Copy, PartialEq)]
pub enum TutorialView {
    Toc,
    Lesson,
}

#[derive(Properties, PartialEq)]
pub struct TutorialPanelProps {
    pub lesson_idx: usize,
    pub on_prev: Callback<MouseEvent>,
    pub on_next: Callback<MouseEvent>,
    pub on_jump: Callback<usize>,
    pub on_run_example: Callback<String>,
    /// Run-all path: emits the whole lesson example list in one
    /// shot. Distinct from `on_run_example` because per-line
    /// emission would race the Yew state batcher and lose all
    /// but the last entry (see `handlers::make_submit_batch`).
    pub on_run_batch: Callback<Vec<String>>,
    pub on_close: Callback<MouseEvent>,
    /// Which sub-tab to land on when this panel mounts.
    /// Header "Tutorial" click -> `Toc` (browsable index).
    /// Paths "Open lesson X" click -> `Lesson` (jumps
    /// straight to the selected lesson's content, since
    /// the user already chose a specific lesson upstream).
    pub initial_view: TutorialView,
}

#[function_component(TutorialPanel)]
pub fn tutorial_panel(props: &TutorialPanelProps) -> Html {
    let view = use_state(|| props.initial_view);
    let on_select_toc = {
        let view = view.clone();
        Callback::from(move |_| view.set(TutorialView::Toc))
    };
    let on_select_lesson = {
        let view = view.clone();
        Callback::from(move |_| view.set(TutorialView::Lesson))
    };
    let on_jump_then_show = {
        let on_jump = props.on_jump.clone();
        let view = view.clone();
        Callback::from(move |idx: usize| {
            on_jump.emit(idx);
            view.set(TutorialView::Lesson);
        })
    };
    let body = match *view {
        TutorialView::Toc => render_toc(props.lesson_idx, on_jump_then_show),
        TutorialView::Lesson => render_lesson(props),
    };
    let cls = |t: TutorialView| if *view == t { "tab active" } else { "tab" };
    html! {
        <div class="tutorial-panel">
            <div class="tutorial-header">
                <div class="tabs">
                    <button class={cls(TutorialView::Toc)} onclick={on_select_toc}>{"Index"}</button>
                    <button class={cls(TutorialView::Lesson)} onclick={on_select_lesson}>{"Current Lesson"}</button>
                </div>
                <button class="close-btn" onclick={props.on_close.clone()} aria-label="Exit tutorial">{"\u{00d7}"}</button>
            </div>
            { body }
        </div>
    }
}

fn render_toc(current: usize, on_jump_then_show: Callback<usize>) -> Html {
    let items = LESSONS.iter().enumerate().map(|(idx, lesson)| {
        let on_click = {
            let cb = on_jump_then_show.clone();
            Callback::from(move |_| cb.emit(idx))
        };
        let class = if idx == current {
            "toc-item current"
        } else {
            "toc-item"
        };
        html! {
            <button class={class} onclick={on_click} title="Open lesson">
                <span class="toc-num">{ format!("{:>2}.", idx + 1) }</span>
                <span class="toc-title">{ lesson.title }</span>
            </button>
        }
    });
    html! {
        <div class="tutorial-toc">
            <p class="tutorial-toc-intro">{ format!("{} lessons. Click to open. The current lesson is highlighted.", LESSONS.len()) }</p>
            <div class="toc-list">{ for items }</div>
        </div>
    }
}

fn render_lesson(props: &TutorialPanelProps) -> Html {
    let Some(lesson) = LESSONS.get(props.lesson_idx) else {
        return html! {};
    };
    let total = LESSONS.len();
    let is_first = props.lesson_idx == 0;
    let is_last = props.lesson_idx + 1 == total;
    let on_run_all = {
        let on_run_batch = props.on_run_batch.clone();
        let lines = lesson.examples;
        Callback::from(move |_: MouseEvent| {
            on_run_batch.emit(lines.iter().map(|s| (*s).to_string()).collect());
        })
    };
    html! {
        <>
            <div class="tutorial-subnav">
                <div class="tutorial-nav">
                    <button class="ctrl-btn" disabled={is_first} onclick={props.on_prev.clone()} aria-label="Previous lesson" title="Previous lesson">{"\u{2190}"}</button>
                    <button class="ctrl-btn" disabled={is_last} onclick={props.on_next.clone()} aria-label="Next lesson" title="Next lesson">{"\u{2192}"}</button>
                </div>
                <span class="tutorial-progress">{ format!("Lesson {} of {}", props.lesson_idx + 1, total) }</span>
                <h2>{ lesson.title }</h2>
                <button class="run-all-btn" onclick={on_run_all} title="Run every example in this lesson, in order">{"Run all"}</button>
            </div>
            <div class="tutorial-intro">{ crate::intro_md::render(lesson.intro) }</div>
            <div class="lesson-examples">
                { for lesson.examples.iter().map(|line| render_example(line, &props.on_run_example)) }
            </div>
            <p class="tutorial-tryit"><strong>{"Try it: "}</strong>{ lesson.try_it }</p>
        </>
    }
}

fn render_example(line: &'static str, on_run: &Callback<String>) -> Html {
    let on_click = {
        let on_run = on_run.clone();
        let line_str = line.to_string();
        Callback::from(move |_| on_run.emit(line_str.clone()))
    };
    let (code, tip) = crate::entry_render::split_inline_comment(line);
    let title = tip
        .map(|t| format!("{t}\n\nClick to run."))
        .unwrap_or_else(|| "Click to run".to_string());
    html! {
        <button class="lesson-example" onclick={on_click} title={title}>
            <span class="example-prompt">{"mlpl> "}</span>{ code }
        </button>
    }
}
