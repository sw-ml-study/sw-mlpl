use yew::prelude::*;

use mlpl_web_onboarding::splash::{SplashOverlay, make_splash_action};
use mlpl_web_onboarding::tour::TourTooltip;
use mlpl_web_onboarding::whats_new::WhatsNewOverlay;
use mlpl_web_render_types::args::RenderArgs;

pub fn render_overlays(a: &RenderArgs) -> Html {
    let splash = render_splash(a);
    let tour = render_tour(a);
    let wn_h = a.onboarding.show_whats_new.clone();
    let last_seen_version = a.onboarding.last_seen_version.clone();
    let whats_new = if *a.onboarding.show_whats_new {
        let on_dismiss = Callback::from(move |_: MouseEvent| {
            mlpl_web_onboarding::storage::write_last_seen_version(&last_seen_version);
            wn_h.set(false);
        });
        html! { <WhatsNewOverlay {on_dismiss} /> }
    } else {
        html! {}
    };
    html! { <> { splash } { whats_new } { tour } </> }
}

fn render_splash(a: &RenderArgs) -> Html {
    if !*a.onboarding.show_splash {
        return html! {};
    }
    // Saga 82: SplashOverlay moved to mlpl-web-onboarding; only
    // mlpl-web's build.rs emits BUILD_COMMIT_COUNT, so the
    // host crate hands the formatted version label in via
    // RenderArgs.
    html! { <SplashOverlay
        on_action={make_splash_action(
            a.onboarding.show_splash.clone(),
            a.onboarding.show_tour.clone(),
            a.callbacks.on_demo.clone(),
            a.ui.input_value.clone(),
            a.ui.lesson_idx.clone(),
            a.ui.path_state.clone(),
        )}
        version_label={a.build.version.clone()}
        build_time={a.build.time.clone()}
    /> }
}

fn render_tour(a: &RenderArgs) -> Html {
    if !*a.onboarding.show_tour {
        return html! {};
    }
    let step = *a.onboarding.tour_step;
    let sh = a.onboarding.tour_step.clone();
    let th = a.onboarding.show_tour.clone();
    let on_next = Callback::from(move |_: MouseEvent| {
        if *sh + 1 >= mlpl_web_onboarding::tour::STEP_COUNT {
            th.set(false);
            let _ = js_sys::eval("window.__stage3d_reset_view && window.__stage3d_reset_view()");
        } else {
            sh.set(*sh + 1);
        }
    });
    let sh2 = a.onboarding.tour_step.clone();
    let on_prev = Callback::from(move |_: MouseEvent| sh2.set((*sh2).saturating_sub(1)));
    let ch = a.onboarding.show_tour.clone();
    let on_close = Callback::from(move |_: MouseEvent| {
        ch.set(false);
        let _ = js_sys::eval("window.__stage3d_reset_view && window.__stage3d_reset_view()");
    });
    html! { <TourTooltip {step} {on_next} {on_prev} {on_close} /> }
}
