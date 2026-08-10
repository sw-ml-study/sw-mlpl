use mlpl_web_components_chrome::connect_probe::ServerBuild;
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
    /// Pre-formatted version label ("vX.Y.Z.<commits>"). Saga 82
    /// extracted splash.rs out of mlpl-web; only mlpl-web's
    /// build.rs emits BUILD_COMMIT_COUNT, so the host crate
    /// formats the string with its own `env!()` macros and
    /// hands it in.
    pub version_label: AttrValue,
    /// Build time in UTC Zulu ("YYYY-MM-DDTHH:MM:SSZ"). Shown so a
    /// viewer can identify which build they are running regardless
    /// of local timezone (the commit hash is not on the splash).
    pub build_time: AttrValue,
}

#[function_component(SplashOverlay)]
pub fn splash_overlay(props: &SplashProps) -> Html {
    let emit = |action: SplashAction| {
        let cb = props.on_action.clone();
        Callback::from(move |_: MouseEvent| cb.emit(action.clone()))
    };
    let dismiss = emit(SplashAction::Dismiss);
    let stop = Callback::from(|e: MouseEvent| e.stop_propagation());
    // When connected, show the SERVER's build beside the UI build and
    // flag a skew (rendering split out to keep this fn under budget).
    let server_build = mlpl_web_components_chrome::connect_probe::use_server_build();
    let server_line = render_server_line(server_build.as_ref(), &props.build_time);
    html! {
        <div class="splash-backdrop" role="dialog" aria-label="Welcome" onclick={dismiss.clone()}>
            <div class="splash-panel" onclick={stop}>
                <button class="splash-close" onclick={dismiss} aria-label="Close">{"\u{00d7}"}</button>
                <div class="splash-header-shim">
                    <h2><img src="mlpl-badge.webp" alt="" />{"sw-MLPL"}</h2>
                    <p class="splash-version">{&props.version_label}</p>
                    <p class="splash-buildtime">{"UI built "}{&props.build_time}</p>
                    { server_line }
                    <mlpl_web_components_chrome::status_badge::StatusBadge />
                    <p class="splash-subtitle">
                        {"An array programming language for learning machine learning, from scalars to transformers."}
                    </p>
                </div>
                <div class="splash-cards">
                    { card(&emit, SplashAction::RunDemo(basics_demo_index()), "Run the Basics demo", "See arithmetic, arrays, and broadcasting in action.") }
                    { card(&emit, SplashAction::TypeExpr("1 + 2"), "Try 1 + 2 in the REPL", "Type an expression and press Enter, or :help for the function list.") }
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

/// Index of the demo NAMED "Basics" in the registry. Resolved by
/// name because raw indexes rot as demos.toml gains and reorders
/// entries (a hardcoded 7 once ran Game of Life from this card).
#[must_use]
pub fn basics_demo_index() -> usize {
    mlpl_web_demos::DEMOS
        .iter()
        .position(|d| d.name == "Basics")
        .unwrap_or(0)
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

/// True when the UI and server build times are far enough apart to
/// signal a STALE build rather than the normal minutes between
/// building the server and the UI in one deploy. Both are
/// `YYYY-MM-DDTHH:MM:SSZ` (UTC); an unparseable value -> not skewed
/// (never a false alarm). Threshold: 30 minutes.
#[must_use]
pub fn build_skew(ui_ts: &str, server_ts: &str) -> bool {
    let epoch = |s: &str| -> Option<i64> {
        let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
        let (y, mo, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
        let (h, mi, se) = (n(11..13)?, n(14..16)?, n(17..19)?);
        let a = (14 - mo) / 12;
        let yy = y + 4800 - a;
        let mm = mo + 12 * a - 3;
        let jdn = d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045;
        Some((jdn - 2_440_588) * 86_400 + h * 3_600 + mi * 60 + se)
    };
    match (epoch(ui_ts), epoch(server_ts)) {
        (Some(a), Some(b)) => (a - b).abs() > 1_800,
        _ => false,
    }
}

/// Render the "server built ..." splash line (connect mode only),
/// flagging a build skew vs the UI build. Empty when not connected.
fn render_server_line(build: Option<&ServerBuild>, ui_time: &str) -> Html {
    let Some(sb) = build else {
        return html! {};
    };
    let skew = build_skew(ui_time, &sb.built_at);
    let cls = if skew {
        "splash-buildtime splash-build-skew"
    } else {
        "splash-buildtime"
    };
    let note = if skew {
        " (stale build? rebuild/restart)"
    } else {
        ""
    };
    html! { <p class={cls}>{"server built "}{&sb.built_at}{note}</p> }
}
