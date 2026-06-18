//! Paths picker + walker view.
//!
//! State (passed in from main.rs):
//! - `None`           -> not in paths mode, render nothing.
//! - `Some((None, _))` -> picker (paths landing).
//! - `Some((Some(p), s))` -> walking path `p` at step `s`.
//!
//! Cross-tab actions on lesson / demo steps fire the
//! corresponding callbacks (`on_open_lesson`, `on_run_demo`)
//! which the parent uses to switch into Tutorial / REPL mode
//! and exit Paths mode.

use yew::prelude::*;

use crate::diagrams::DIAGRAMS;
use mlpl_web_demos::DEMOS;
use mlpl_web_glossary::view::find_by_term;
use mlpl_web_lessons::lessons::LESSONS;
use mlpl_web_paths_data::{PATHS, Step};

#[derive(Properties, PartialEq)]
pub struct PathsViewProps {
    pub state: Option<(Option<usize>, usize)>,
    pub on_change: Callback<Option<(Option<usize>, usize)>>,
    pub on_open_lesson: Callback<usize>,
    pub on_run_demo: Callback<String>,
}

#[function_component(PathsView)]
pub fn paths_view(props: &PathsViewProps) -> Html {
    let Some((path_opt, step_idx)) = props.state else {
        return html! {};
    };
    match path_opt {
        None => render_picker(&props.on_change),
        Some(p) => render_walker(props, p, step_idx),
    }
}

fn render_picker(on_change: &Callback<Option<(Option<usize>, usize)>>) -> Html {
    let cards = PATHS.iter().enumerate().map(|(i, path)| {
        let on_click = {
            let on_change = on_change.clone();
            Callback::from(move |_| on_change.emit(Some((Some(i), 0))))
        };
        // The Tour narration ("Let's open the Dimensionality Reduction
        // path") and the highlighted card must agree, so the
        // `path-card-dim-reduction` marker is anchored to the
        // Dimensionality Reduction card by title (case-insensitive --
        // the const uses "Dimensionality reduction" with a lowercase
        // 'r') -- not to position 0, which has shifted as PATHS gets
        // reordered.
        let target = if path.title.eq_ignore_ascii_case("Dimensionality reduction") {
            Some("path-card-dim-reduction")
        } else {
            None
        };
        html! {
            <button class="path-card" onclick={on_click} data-tour-target={target}>
                <span class="path-card-title">{ path.title }</span>
                <span class="path-card-blurb">{ path.blurb }</span>
                <span class="path-card-meta">{ format!("{} steps", path.steps.len()) }</span>
            </button>
        }
    });
    let intro = "A curated walk through MLPL's lessons, demos, diagrams, and glossary entries. \
                 Pick a path that matches your goal -- you can change paths at any time, and \
                 individual surfaces (REPL, Tutorial, Help -> Diagrams) are still browsable on their own.";
    html! {
        <div class="paths-panel" data-tour-target="paths-panel">
            <h2 class="paths-heading">{"Learning paths"}</h2>
            <p class="paths-intro">{ intro }</p>
            <div class="path-cards">{ for cards }</div>
        </div>
    }
}

fn render_walker(props: &PathsViewProps, path_idx: usize, step_idx: usize) -> Html {
    let path = &PATHS[path_idx];
    let total = path.steps.len();
    let step = &path.steps[step_idx];
    let on_back = {
        let on_change = props.on_change.clone();
        Callback::from(move |_| on_change.emit(Some((None, 0))))
    };
    let on_close = {
        let on_change = props.on_change.clone();
        Callback::from(move |_| on_change.emit(None))
    };
    let on_prev = step_callback(&props.on_change, path_idx, step_idx, -1, total);
    let on_next = step_callback(&props.on_change, path_idx, step_idx, 1, total);
    let is_first = step_idx == 0;
    let is_last = step_idx + 1 == total;
    html! {
        <div class="paths-panel">
            <div class="paths-walker-header">
                <button class="ctrl-btn" onclick={on_back} title="Back to picker">{"\u{2190} Paths"}</button>
                <span class="paths-walker-title">{ path.title }</span>
                <div class="paths-walker-nav">
                    <button class="ctrl-btn" disabled={is_first} onclick={on_prev} title="Previous step">{"\u{2190}"}</button>
                    <span class="paths-walker-progress">{ format!("Step {} of {}", step_idx + 1, total) }</span>
                    <button class="ctrl-btn" disabled={is_last} onclick={on_next} title="Next step">{"\u{2192}"}</button>
                </div>
                <button class="ctrl-btn" onclick={on_close} title="Exit paths mode">{"\u{2715}"}</button>
            </div>
            <div class="paths-walker-body" data-tour-target="paths-walker-body">{ render_step(props, step) }</div>
        </div>
    }
}

fn step_callback(
    on_change: &Callback<Option<(Option<usize>, usize)>>,
    path_idx: usize,
    step_idx: usize,
    delta: i32,
    total: usize,
) -> Callback<MouseEvent> {
    let on_change = on_change.clone();
    Callback::from(move |_| {
        let next = step_idx as i32 + delta;
        if next < 0 || next as usize >= total {
            return;
        }
        on_change.emit(Some((Some(path_idx), next as usize)));
    })
}

fn render_step(props: &PathsViewProps, step: &Step) -> Html {
    let (kind, title, why_opt, body) = match step {
        Step::Lesson { title, why } => ("Lesson", *title, Some(*why), lesson_body(props, title)),
        Step::Demo { name, why } => ("Demo", *name, Some(*why), demo_body(props, name)),
        Step::Diagram { slug, why } => {
            let title = DIAGRAMS
                .iter()
                .find(|(s, _, _)| *s == *slug)
                .map(|(_, t, _)| *t)
                .unwrap_or(*slug);
            let src = format!("diagrams/{slug}.svg");
            let body = html! {
                <div class="diagram-card">
                    <img class="diagram-svg" src={src} alt={title.to_string()} loading="lazy"/>
                </div>
            };
            ("Diagram", title, Some(*why), body)
        }
        Step::Glossary { term, why } => {
            let body = match find_by_term(term) {
                Some(entry) => {
                    html! { <div class="path-step-preview">{ mlpl_web_path_body::render_body(&entry.body) }</div> }
                }
                None => html! {
                    <p class="path-step-error">{ format!("(glossary entry {term:?} not found)") }</p>
                },
            };
            ("Concept", *term, Some(*why), body)
        }
        Step::Note { title, body } => (
            "Note",
            *title,
            None,
            html! { <div class="path-step-preview">{ mlpl_web_path_body::render_body(body) }</div> },
        ),
    };
    let why_html = match why_opt {
        Some(s) => html! { <p class="path-step-why">{ s }</p> },
        None => html! {},
    };
    html! {
        <div class="path-step">
            <span class="path-step-kind">{ kind }</span>
            <h3 class="path-step-title">{ title }</h3>
            { why_html }
            { body }
        </div>
    }
}

fn lesson_body(props: &PathsViewProps, title: &'static str) -> Html {
    let Some(i) = LESSONS.iter().position(|l| l.title == title) else {
        return html! { <p class="path-step-error">{ format!("(lesson {title:?} not found)") }</p> };
    };
    let on_open = {
        let cb = props.on_open_lesson.clone();
        Callback::from(move |_| cb.emit(i))
    };
    let label = format!("Open lesson \"{title}\" \u{2192}");
    html! {
        <>
            <div class="path-step-preview">{ mlpl_web_lessons::intro_md::render(LESSONS[i].intro) }</div>
            <button class="path-jump-btn" onclick={on_open} data-tour-target="path-open-lesson">{ label }</button>
        </>
    }
}

fn demo_body(props: &PathsViewProps, name: &'static str) -> Html {
    let Some(demo) = DEMOS.iter().find(|d| d.name == name) else {
        return html! { <p class="path-step-error">{ format!("(demo {name:?} not found)") }</p> };
    };
    let on_run = {
        let cb = props.on_run_demo.clone();
        let n = name.to_string();
        Callback::from(move |_| cb.emit(n.clone()))
    };
    let label = format!("Run demo \"{name}\" in REPL \u{2192}");
    html! {
        <>
            <div class="path-step-preview">{ mlpl_web_path_body::render_body(demo.intro) }</div>
            <button class="path-jump-btn" onclick={on_run}>{ label }</button>
        </>
    }
}
