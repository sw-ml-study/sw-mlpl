use yew::prelude::*;

use crate::components::UrlProps;

#[function_component(Footer)]
pub fn footer(props: &UrlProps) -> Html {
    let version = format!(
        "v{}.{}",
        env!("CARGO_PKG_VERSION"),
        env!("BUILD_COMMIT_COUNT")
    );
    let build_info = format!(
        "{} \u{00b7} {} \u{00b7} {} \u{00b7} {}",
        version,
        env!("BUILD_HOST"),
        env!("BUILD_SHA"),
        env!("BUILD_TIMESTAMP")
    );
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
            <a href="https://software-wrighter-lab.github.io/" target="_blank" rel="noopener">{"Blog"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://discord.com/invite/Ctzk5uHggZ" target="_blank" rel="noopener">{"Discord"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <a href="https://www.youtube.com/@SoftwareWrighter" target="_blank" rel="noopener">{"YouTube"}</a>
            <span class="footer-sep">{"\u{00b7}"}</span>
            <span>{ build_info }</span>
        </footer>
    }
}
