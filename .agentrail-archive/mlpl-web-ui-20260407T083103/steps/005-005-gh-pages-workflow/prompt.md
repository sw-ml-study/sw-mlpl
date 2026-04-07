Set up GitHub Actions workflow for GitHub Pages deployment.

1. Create .github/workflows/pages.yml:
   - Name: "Deploy to GitHub Pages"
   - Trigger: push to main, workflow_dispatch
   - Permissions: contents read, pages write, id-token write
   - Concurrency: single deployment, cancel in-progress
   - Steps:
     a. Checkout (actions/checkout@v4)
     b. Upload pages artifact from ./pages (actions/upload-pages-artifact@v3)
     c. Deploy to pages (actions/deploy-pages@v4)

2. Run scripts/build-pages.sh to generate pages/ content

3. Commit pages/ directory with built artifacts

4. Push and verify the workflow runs successfully

5. Verify the site loads at https://sw-ml-study.github.io/sw-mlpl/
   (may need to enable Pages in repo settings first)

Allowed: .github/, scripts/, pages/