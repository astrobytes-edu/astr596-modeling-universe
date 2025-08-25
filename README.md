
# ASTR 596: Modeling the Universe

Course materials and site source for ASTR 596 (Fall 2025).

## Quick start

Set up a virtual environment, install dependencies, and build the site locally (Jupyter Book 2.x only):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Build with Jupyter Book 2.x
jupyter book start #Following https://next.jupyterbook.org/
```

If you prefer the MyST CLI (Markdown-first workflow), install and use `mystmd`:

```bash
# (optional) install myst CLI
pip install mystmd
# serve a local dev server (auto-rebuild)
myst start
# build statically
myst build
```

Serve the built site for local preview (simple HTTP server):

```bash
python -m http.server -d _build/html 8000
# then open http://localhost:8000
```

Notes on `jupyter-book start`:

- `jupyter-book build .` is the canonical build command for Jupyter Book 2.x and is the command we pin in `requirements.txt` (`jupyter-book==2.*`).
- A `jupyter-book start` dev server may be available depending on the installed Jupyter Book release, but it is not required — `myst start` or the simple `http.server` approach are reliable alternatives.

## Prerequisites

- Tested on Python 3.11+. See `requirements.txt` for package versions.

## Repository layout

- `01-course-info/` — syllabus, schedule, instructor & TA contact
- `02-getting-started/` — setup and environment notes
- `03-scientific-computing-with-python/` — course modules and notebooks
- `manim-media/` — animation scenes and assets (manim is experimental; media not yet published)
- `drafts/` — unpublished drafts and files moved out of the published site
- `tests/` — unit and content checks

## Contributing

See `CONTRIBUTING.md` for branching, commit style, local build, and PR checklist.

## Testing & formatting

- Run tests: `pytest -q`
- Format code: `black .`
- Lint: `flake8` or `pylint` (optional)

## CI & Deployment

Recommended deploy workflow (follow `mystmd` / Jupyter Book docs):

Use the built-in initializer to create a GitHub Actions workflow that deploys to GitHub Pages. This follows the `--gh-pages` flow in the official docs (preferred):

- Option A (MyST CLI, Markdown-first):

```bash
# Run from the repository root; this interactively scaffolds a Pages action and config
myst init --gh-pages

# After answering prompts, commit the generated files and push to your repo
git add .github/workflows && git commit -m "Add GitHub Pages workflow (myst init --gh-pages)" && git push
```

- Option B (Jupyter Book CLI):

```bash
# Run from the repository root; this interactively scaffolds a Pages action and config
jupyter book init --gh-pages

# After answering prompts, commit and push the generated workflow
git add .github/workflows && git commit -m "Add GitHub Pages workflow (jupyter book init --gh-pages)" && git push
```

After you push the generated workflow, enable GitHub Pages in the repository Settings -> Pages and set the source to *GitHub Actions*. Pushing to the branch you selected (e.g., `main`) will trigger the action and publish your site to `https://<org-or-user>.github.io/<repo>/`.

BASE_URL / repository subpath note:

- If your repo is not a user/organization-level repo (i.e., `username.github.io`), your site will usually be served at `/repo-name/`. The MyST CLI's `--gh-pages` init will configure `BASE_URL` automatically; if not, set `BASE_URL` as an environment variable in the generated GitHub Action or in the action's `with:` configuration.

Build commands (local testing):

```bash
# Build with Jupyter Book 2.x (canonical)
jupyter-book build .

# Or with MyST CLI (if using mystmd)
myst build

# Serve the built site locally
python -m http.server -d _build/html 8000
```

Resources:

- Jupyter Book: <https://next.jupyterbook.org/>
- MyST Markdown: <https://mystmd.org/>

## Optional JAX installation

The JAX ecosystem is optional and requires platform-specific wheels (GPU/CPU). To install it on a capable machine, run:

```bash
python -m pip install -r requirements-jax.txt
```

Do not install `requirements-jax.txt` on CI or binder unless you control the runner environment.

## Excluded files

Files listed in `myst.yml` `exclude:` (for example `TODO.md` and `CONTRIBUTING.md`) are intentionally not published to the public site. Do not remove these entries unless you want them visible.

## License & contact

- Content: CC-BY-4.0
- Code: MIT
- Contact: `@drannarosen` — <alrosen@sdsu.edu>

