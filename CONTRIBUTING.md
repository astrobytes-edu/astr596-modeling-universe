# Contributing to ASTR 596: Modeling the Universe

This repository is primarily maintained by `@drannarosen` (Anna Rosen — alrosen@sdsu.edu). It documents the minimal workflow and expectations for contributors and TAs.

## Purpose

- Capture workflow, quick commands, and review expectations so contributors can get started quickly and consistently.

## Quick Rules

- **Owner:** `@drannarosen` (alrosen@sdsu.edu)
- **Branching:** Create feature branches using `feature/<short-name>` or fixes with `fix/<short-desc>`.
- **Target branch:** Open PRs to `main` unless instructed otherwise.
- **Commits:** Use short imperative messages (e.g., "Add syllabus updates"). Optionally adopt Conventional Commits like `feat:`, `fix:`, `docs:`.
- **Exclude from site:** If you add files that should not be published, add patterns to `myst.yml` `exclude:` (for example `TODO.md` or `CONTRIBUTING.md`).

## Local setup & build

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# Contributing to ASTR 596: Modeling the Universe

This repository is primarily maintained by `@drannarosen` (Anna Rosen — `alrosen@sdsu.edu`). It documents the minimal workflow and expectations for contributors and TAs.

## Purpose

- Capture workflow, quick commands, and review expectations so contributors can get started quickly and consistently.

## Quick Rules

- **Owner:** `@drannarosen` (`alrosen@sdsu.edu`)
- **Branching:** Create feature branches using `feature/<short-name>` or fixes with `fix/<short-desc>`.
- **Target branch:** Open PRs to `main` unless instructed otherwise.
- **Commits:** Use short imperative messages (e.g., "Add syllabus updates"). Optionally adopt Conventional Commits like `feat:`, `fix:`, `docs:`.
- **Exclude from site:** If you add files that should not be published, add patterns to `myst.yml` `exclude:` (for example `TODO.md` or `CONTRIBUTING.md`).

## Local setup & build

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```bash

1. Install dependencies and build the site:

```bash
pip install -r requirements.txt
jupyter-book build .
```bash

If you prefer a Markdown-first workflow, install and use the MyST CLI:

```bash
# (optional) install myst CLI
pip install mystmd
# serve a local dev server (auto-rebuild)
myst start
# build statically
myst build
```bash

## Deploying to GitHub Pages (recommended options)

Choose one of the following:

- Local manual push using `ghp-import`:

```bash
# Build the site
jupyter-book build .

# Install ghp-import if needed
pip install ghp-import

# Push to gh-pages branch
ghp-import -n -p -f _build/html
```bash

- Automated deploy with GitHub Actions (recommended for CI): create `.github/workflows/deploy.yml` that installs dependencies, runs `jupyter-book build .`, and uses `peaceiris/actions-gh-pages` to publish `_build/html` to `gh-pages`.

Follow the official `--gh-pages` init flow to generate a GitHub Action automatically:

- From the repo root, run one of the following and answer prompts:

```bash
# MyST CLI (recommended for Markdown-first projects)
myst init --gh-pages

# Or Jupyter Book CLI
jupyter book init --gh-pages
```bash

This scaffolds a `.github/workflows/*.yml` GitHub Action that will build and deploy your site on push to the configured branch. Commit and push the generated files to trigger the Action. If your repo is not `username.github.io`, ensure `BASE_URL` is set appropriately (the `--gh-pages` init handles this for you in most cases).

See `README.md` for an example workflow and notes about `BASE_URL`.

1. (Optional) If you add Python code, run tests:

```bash
pytest -q
```bash

## Testing & Linting

- Format Python code with `black .` before committing.
- Lint with `flake8` or `pylint` as needed. These are included in `requirements.txt` if used.

## Pull Requests

- Open an issue for larger changes to discuss scope before starting.
- Create a pull request from a feature branch and include:
  - A short description of the change
  - Related issue number (if any)
  - Build or test steps you ran locally

## PR Checklist (suggested)

- [ ] Site builds locally (no errors)
- [ ] No broken internal links
- [ ] Tests pass (if applicable)
- [ ] `myst.yml` updated to exclude any new non-published files

## Security and Sensitive Data

- Do not commit credentials, secrets, or large datasets. Add such files to `.gitignore` and use external storage (private cloud storage, GitHub secrets, or data repositories).

## Contact

- Owner: `@drannarosen` (`alrosen@sdsu.edu`)

This is intentionally concise — expand if you onboard external collaborators or TAs.
