//! Page footer with legal blurb, links, and a build-info
//! summary. Saga 82 moved this out of mlpl-web; the
//! `BUILD_HOST` / `BUILD_SHA` / `BUILD_TIMESTAMP` /
//! `BUILD_COMMIT_COUNT` env vars are emitted only by
//! mlpl-web's build.rs, so the host crate formats the
//! `build_info` string with its own `env!()` macros and
//! hands it in via FooterProps.
//!
//! The "GitHub" link opens a `RepoLinks` dialog listing this
//! project plus its companion demo repositories, rather than
//! navigating straight to one repo.

use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct FooterProps {
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
            <RepoLinks />
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

/// This project plus its companion demo repositories
/// (github.com/sw-ml-study/<name>), shown in the footer's GitHub
/// dialog, grouped by subject. The WASM page cannot query GitHub at
/// runtime, so the set is baked in -- keep it in sync with the org's
/// `sw-mlpl` + `demo*` repositories (names are the GitHub repo names).
const REPO_SECTIONS: &[(&str, &[(&str, &str)])] = &[
    (
        "Core",
        &[(
            "sw-mlpl",
            "The MLPL language, browser playground, and native tools.",
        )],
    ),
    (
        "Machine learning",
        &[(
            "demo-ml-utils",
            "Machine-learning utility demos built with MLPL.",
        )],
    ),
    (
        "Programming",
        &[
            ("demo-algorithms", "General-purpose algorithms in MLPL."),
            (
                "demo-data-structures",
                "General-purpose data structures in MLPL.",
            ),
            (
                "demo-extensions",
                "Authoring native MLPL language extensions in Rust.",
            ),
            (
                "demo-file-processing",
                "Bounded byte and file processing (hexdump, WAV, MP3/ID3, Ogg).",
            ),
            (
                "demo-functional-pipelines",
                "A functional pipeline library for MLPL.",
            ),
            (
                "demo-memory",
                "Companion demos for hashmaps, memory, and retrieval.",
            ),
        ],
    ),
    (
        "Mathematics",
        &[
            (
                "demo-abstract-algebra",
                "Groups, rings, and fields explored and visualized in MLPL.",
            ),
            (
                "demo-category-theory",
                "Category-theory constructions (functors, products, ...) in MLPL.",
            ),
            (
                "demo-combinators",
                "\"To Mock a Mockingbird\" combinator birds, in MLPL.",
            ),
            (
                "demo-linear-algebra",
                "Vectors, matrices, and linear-algebra operations in MLPL.",
            ),
        ],
    ),
];

/// The footer "GitHub" link. Opens a dialog of the project +
/// companion repos instead of navigating straight to one repo
/// (the href is kept as a no-JavaScript fallback).
#[function_component(RepoLinks)]
pub fn repo_links() -> Html {
    let open = use_state(|| false);
    let show = {
        let open = open.clone();
        Callback::from(move |e: MouseEvent| {
            e.prevent_default();
            open.set(true);
        })
    };
    let close = {
        let open = open.clone();
        Callback::from(move |_: MouseEvent| open.set(false))
    };
    html! {
        <>
            <a href="https://github.com/sw-ml-study/sw-mlpl" onclick={show} aria-haspopup="dialog">{"GitHub"}</a>
            { (*open).then(|| repo_dialog(&close)) }
        </>
    }
}

/// The modal listing every project repository. Clicking the
/// backdrop or the close button dismisses it; each row opens that
/// repo on GitHub in a new tab.
fn repo_dialog(close: &Callback<MouseEvent>) -> Html {
    let stop = Callback::from(|e: MouseEvent| e.stop_propagation());
    html! {
        <div class="modal-backdrop" onclick={close.clone()}>
            <div class="modal compact" onclick={stop} role="dialog" aria-label="Project repositories">
                <div class="modal-header">
                    <span class="repo-modal-title">{"sw-ml-study repositories"}</span>
                    <button class="close-btn" onclick={close.clone()} aria-label="Close">{"\u{00d7}"}</button>
                </div>
                <div class="modal-body">
                    { for REPO_SECTIONS.iter().map(|(label, repos)| repo_section(label, repos)) }
                </div>
            </div>
        </div>
    }
}

/// One subject section of the repo dialog: a heading plus its repo
/// rows (each opens that repo on GitHub in a new tab).
fn repo_section(label: &str, repos: &[(&str, &str)]) -> Html {
    html! {
        <div class="repo-section">
            <div class="repo-section-label">{ label.to_string() }</div>
            { for repos.iter().map(|(name, desc)| html! {
                <a class="repo-row" href={format!("https://github.com/sw-ml-study/{name}")} target="_blank" rel="noopener">
                    <span class="repo-name">{ *name }</span>
                    <span class="repo-desc">{ *desc }</span>
                </a>
            }) }
        </div>
    }
}
