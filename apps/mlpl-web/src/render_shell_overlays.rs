use yew::prelude::*;

use crate::onboarding_splash::{SplashOverlay, make_splash_action};
use crate::onboarding_tour::TourTooltip;
use crate::onboarding_whats_new::WhatsNewOverlay;
use crate::render::RenderArgs;

pub fn render_overlays(a: &RenderArgs) -> Html {
    let splash = render_splash(a);
    let tour = render_tour(a);
    let wn_h = a.onboarding.show_whats_new.clone();
    let whats_new = if *a.onboarding.show_whats_new {
        let on_dismiss = Callback::from(move |_: MouseEvent| {
            crate::onboarding_storage::write_last_seen_version(env!("CARGO_PKG_VERSION"));
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
    html! { <SplashOverlay on_action={make_splash_action(
        a.onboarding.show_splash.clone(),
        a.onboarding.show_tour.clone(),
        a.callbacks.on_demo.clone(),
        a.ui.input_value.clone(),
        a.ui.lesson_idx.clone(),
        a.ui.path_state.clone(),
    )} /> }
}

fn render_tour(a: &RenderArgs) -> Html {
    if !*a.onboarding.show_tour {
        return html! {};
    }
    let step = *a.onboarding.tour_step;
    let sh = a.onboarding.tour_step.clone();
    let th = a.onboarding.show_tour.clone();
    let on_next = Callback::from(move |_: MouseEvent| {
        if *sh + 1 >= crate::onboarding_tour::STEP_COUNT {
            th.set(false);
        } else {
            sh.set(*sh + 1);
        }
    });
    let sh2 = a.onboarding.tour_step.clone();
    let on_prev = Callback::from(move |_: MouseEvent| {
        if *sh2 > 0 {
            sh2.set(*sh2 - 1);
        }
    });
    let ch = a.onboarding.show_tour.clone();
    let on_close = Callback::from(move |_: MouseEvent| ch.set(false));
    html! { <TourTooltip {step} {on_next} {on_prev} {on_close} /> }
}
