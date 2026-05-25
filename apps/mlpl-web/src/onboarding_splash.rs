use yew::prelude::*;

#[derive(Clone, PartialEq)]
pub enum SplashAction {
    Dismiss,
    StartTour,
    RunDemo(usize),
    TypeExpr(&'static str),
    OpenTutorial,
    OpenPaths,
}

#[derive(Properties, PartialEq)]
pub struct SplashProps {
    pub on_action: Callback<SplashAction>,
}

#[function_component(SplashOverlay)]
pub fn splash_overlay(props: &SplashProps) -> Html {
    let emit = |action: SplashAction| {
        let cb = props.on_action.clone();
        Callback::from(move |_: MouseEvent| cb.emit(action.clone()))
    };
    html! {
        <div class="splash-backdrop" role="dialog" aria-label="Welcome">
            <div class="splash-panel">
                <h2><img src="mlpl-badge.webp" alt="" />{"sw-MLPL"}</h2>
                <p class="splash-version">{format!("v{} \u{00b7} {}", env!("CARGO_PKG_VERSION"), env!("BUILD_SHA"))}</p>
                <p class="splash-subtitle">
                    {"An array programming language for learning machine learning, from scalars to transformers."}
                </p>
                <div class="splash-cards">
                    { card(&emit, SplashAction::RunDemo(7), "Run the Basics demo", "See arithmetic, arrays, and broadcasting in action.") }
                    { card(&emit, SplashAction::TypeExpr("1 + 2"), "Try 1 + 2 in the REPL", "Type an expression and press Enter.") }
                    { card(&emit, SplashAction::OpenTutorial, "Open a tutorial lesson", "Guided lessons from scalars to attention.") }
                    { card(&emit, SplashAction::OpenPaths, "Explore learning paths", "Structured sequences of lessons and demos.") }
                </div>
                <div class="splash-actions">
                    <button class="splash-btn splash-btn-primary" onclick={emit(SplashAction::StartTour)}>{"Take a guided tour"}</button>
                    <button class="splash-btn splash-btn-secondary" onclick={emit(SplashAction::Dismiss)}>{"Dismiss"}</button>
                </div>
            </div>
        </div>
    }
}

pub fn make_splash_action(
    splash: UseStateHandle<bool>,
    tour: UseStateHandle<bool>,
    demo: Callback<usize>,
    input: UseStateHandle<String>,
    lesson: UseStateHandle<Option<usize>>,
    path: UseStateHandle<Option<(Option<usize>, usize)>>,
) -> Callback<SplashAction> {
    Callback::from(move |action: SplashAction| {
        splash.set(false);
        match action {
            SplashAction::Dismiss => {}
            SplashAction::StartTour => tour.set(true),
            SplashAction::RunDemo(idx) => demo.emit(idx),
            SplashAction::TypeExpr(s) => input.set(s.to_string()),
            SplashAction::OpenTutorial => lesson.set(Some(0)),
            SplashAction::OpenPaths => path.set(Some((None, 0))),
        }
    })
}

fn card(
    emit: &dyn Fn(SplashAction) -> Callback<MouseEvent>,
    action: SplashAction,
    title: &str,
    body: &str,
) -> Html {
    html! {
        <div class="splash-card" onclick={emit(action)}>
            <strong>{ title }</strong>
            { body }
        </div>
    }
}
