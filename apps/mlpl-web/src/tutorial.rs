use yew::prelude::*;

pub use crate::lessons::{LESSONS, Lesson};

pub fn toggle_tutorial(lesson: UseStateHandle<Option<usize>>) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if lesson.is_some() {
            lesson.set(None);
        } else {
            lesson.set(Some(0));
        }
    })
}

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
enum TutorialView {
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
    pub on_close: Callback<MouseEvent>,
}

#[function_component(TutorialPanel)]
pub fn tutorial_panel(props: &TutorialPanelProps) -> Html {
    let view = use_state(|| TutorialView::Toc);
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
    let lesson = match LESSONS.get(props.lesson_idx) {
        Some(l) => l,
        None => return html! {},
    };
    let total = LESSONS.len();
    let is_first = props.lesson_idx == 0;
    let is_last = props.lesson_idx + 1 == total;
    let examples_html = lesson.examples.iter().map(|line| {
        let line_str = (*line).to_string();
        let on_click = {
            let on_run = props.on_run_example.clone();
            let line_str = line_str.clone();
            Callback::from(move |_| on_run.emit(line_str.clone()))
        };
        html! {
            <button class="lesson-example" onclick={on_click} title="Click to run">
                <span class="example-prompt">{"mlpl> "}</span>{ line }
            </button>
        }
    });
    html! {
        <>
            <div class="tutorial-subnav">
                <div class="tutorial-nav">
                    <button class="ctrl-btn" disabled={is_first} onclick={props.on_prev.clone()} aria-label="Previous lesson" title="Previous lesson">{"\u{2190}"}</button>
                    <button class="ctrl-btn" disabled={is_last} onclick={props.on_next.clone()} aria-label="Next lesson" title="Next lesson">{"\u{2192}"}</button>
                </div>
                <span class="tutorial-progress">{ format!("Lesson {} of {}", props.lesson_idx + 1, total) }</span>
                <h2>{ lesson.title }</h2>
            </div>
            <p class="tutorial-intro">{ lesson.intro }</p>
            <div class="lesson-examples">{ for examples_html }</div>
            <p class="tutorial-tryit"><strong>{"Try it: "}</strong>{ lesson.try_it }</p>
        </>
    }
}
