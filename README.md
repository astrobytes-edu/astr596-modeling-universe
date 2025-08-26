
# ASTR 596: Modeling the Universe

Course materials and site source for ASTR 596 (Fall 2025).

## Quick start

Set up a lightweight environment and build the site locally using the MyST-first workflow (recommended):

```bash
# create and activate a virtualenv (macOS / zsh)
python -m venv .venv
source .venv/bin/activate

# install Python dependencies (core site build)
pip install -r requirements.txt

# Install MyST CLI (recommended):
# Option A (npm, same as CI):
npm install -g mystmd
# Option B (pip): install myst-cli if you prefer the Python packaging
pip install myst-cli

# Serve a live dev server (auto-rebuild)
myst start

# Build a static site (HTML)
myst build --html

# Serve built output for local preview
python -m http.server -d _build/html 8000
# then open http://localhost:8000
```

Notes:

- This repository uses the Markdown-first MyST workflow as the canonical developer experience. Avoid running `jupyter book build` unless you know you need Jupyter Book 2.x compatibility.

- The repository's GitHub Actions workflow (`.github/workflows/deploy.yml`) runs `myst build --html` during CI and provides the canonical `BASE_URL` environment variable for Pages deployments.

Important: PDF/Typst export is disabled

- Automated PDF/Typst exports have been intentionally disabled for this repository. The CI workflow and site navigation do not build or link PDFs by default.
- To re-enable: restore the `exports:` block in `myst.yml`, add a Typst/LaTeX setup step to CI that installs a pinned Typst binary (or LaTeX toolchain), and test locally with the same CLI versions.

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

This repository includes a ready GitHub Actions workflow at `.github/workflows/deploy.yml` that builds the site with the MyST CLI and deploys to GitHub Pages. The workflow sets `BASE_URL` for Pages automatically.

To test the same steps locally:

```bash
# Build the site (HTML only — PDFs are disabled in CI)
myst build --html

# Run the local dev server (auto-rebuild)
myst start
```

PDF exports and Typst: currently, automated Typst/PDF exports are disabled in CI and the site nav (the PDF download action has been commented out). To re-enable PDF exports you must:

1. Restore the Typst export block in `myst.yml`.
2. Ensure a pinned Typst binary is installed in CI (or vendored into the repo) and re-enable the Typst setup step in `.github/workflows/deploy.yml`.
3. Test `myst build --typst` locally with a matching Typst CLI version.

See `.github/copilot-instructions.md` for guidelines on re-enabling PDFs safely.

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

