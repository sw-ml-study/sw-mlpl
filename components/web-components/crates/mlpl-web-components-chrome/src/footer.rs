//! Page footer with legal blurb, links, and a build-info
//! summary. Saga 82 moved this out of mlpl-web; the
//! `BUILD_HOST` / `BUILD_SHA` / `BUILD_TIMESTAMP` /
//! `BUILD_COMMIT_COUNT` env vars are emitted only by
//! mlpl-web's build.rs, so the host crate formats the
//! `build_info` string with its own `env!()` macros and
//! hands it in via FooterProps.

use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct FooterProps {
    /// URL the GitHub link points to.
    pub url: &'static str,
    /// Pre-formatted build-info chip rendered at the right
    /// edge: version + host + sha + timestamp, joined with
    /// the same middot separator the rest of the footer uses.
    pub build_info: AttrValue,
}

#[function_component(Footer)]
pub fn footer(props: &FooterProps) -> Html {
    html! {
        <footer>
            <span>{"MIT License"}</span>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <span>{"\u{00a9} 2026 Michael A Wright"}</span>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href={props.url} target="_blank" rel="noopener">{"GitHub"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://github.com/sw-ml-study/sw-mlpl/blob/main/CHANGES.md" target="_blank" rel="noopener">{"Changes"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="literate/" target="_blank" rel="noopener">{"Literate"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://software-wrighter-lab.github.io/" target="_blank" rel="noopener">{"Blog"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://discord.com/invite/Ctzk5uHggZ" target="_blank" rel="noopener">{"Discord"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://www.youtube.com/@SoftwareWrighter" target="_blank" rel="noopener">{"YouTube"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://github.com/sw-ml-study/sw-mlpl/wiki" target="_blank" rel="noopener">{"Wiki"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <span>{ &props.build_info }</span>
        </footer>
    }
}
