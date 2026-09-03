# Stable release hosting (mlpl.softwarewrighter.com)

Two independent channels serve the live demo:

| Channel | URL | Base path | Deploys when |
| --- | --- | --- | --- |
| **Dev** (rolling) | `https://sw-ml-study.github.io/sw-mlpl/` | `/sw-mlpl/` | every push, via `build-pages.sh` + `deploy-pages.sh` (mirrors into `../sw-mlpl.pages`, branch `gh-pages`) |
| **Stable** (public) | `https://mlpl.softwarewrighter.com/` | `/` (root) | ONLY when you run `release-stable.sh` and push `../mlpl-live` |

The stable channel exists so that pushing work-in-progress to `sw-mlpl`
(which auto-deploys the dev demo) never breaks the URL you hand to a
large audience. It moves only when you deliberately cut a release.

## Why a separate repo (not the same one)

A GitHub repo has exactly ONE Pages site, and a custom domain attaches to
exactly ONE Pages site. So the stable copy lives in its own repo,
`sw-ml-study/mlpl-live`, holding ONLY the built site plus a `CNAME` file.
`release-stable.sh` builds it and mirrors it into `../mlpl-live`.

## The base-path gotcha

The dev build uses `--public-url /sw-mlpl/` (project-pages subpath), so
`index.html` references `/sw-mlpl/...wasm`. At a root custom domain those
absolute paths 404. The stable build therefore uses `--public-url /`.
`release-stable.sh` handles this; do not copy the dev `pages/` directly.

## Cutting a release

```bash
./scripts/release-stable.sh     # builds root-base site, mirrors + commits ../mlpl-live
git -C ../mlpl-live push        # publish (after first-time setup below)
```

## One-time setup

1. `./scripts/release-stable.sh` (creates + commits `../mlpl-live`).
2. Create an EMPTY GitHub repo `sw-ml-study/mlpl-live` (no README/license).
3. `git -C ../mlpl-live remote add origin git@github.com:sw-ml-study/mlpl-live.git`
4. `git -C ../mlpl-live push -u origin main`
5. DNS at Squarespace (softwarewrighter.com): add a `CNAME` record
   `mlpl` -> `sw-ml-study.github.io`. (Same as `blog.hardwarewrighter.com`;
   GitHub routes to the right repo via the repo's `CNAME` file, not the
   DNS target.)
6. Repo Settings -> Pages: Source = `main` branch / root. It reads the
   `CNAME` file; confirm the custom domain `mlpl.softwarewrighter.com`,
   then enable "Enforce HTTPS" once the Let's Encrypt cert provisions
   (usually a few minutes).

## Notes

- Served by GitHub Pages / Fastly CDN: free, handles HN-scale spikes, gzip,
  correct `application/wasm` MIME. The bundle is ~7.6 MB (~2.5 MB gzipped).
- GPU / `?connect=` demos will NOT work on the public domain (no server
  behind it, and CORS). The default in-browser playground demos all work.
- The stable page's `build-info.json` carries `"channel":"stable"` and the
  source commit it was cut from.
- The `mlpl-live` repo carries its own `README.md`, `LICENSE`, and
  `COPYRIGHT` (MIT, matching `sw-mlpl`). `release-stable.sh` excludes them
  from the `rsync --delete`, so a rebuild never wipes them.
