
# ASTR 596: Modeling the Universe

Course materials and site source for ASTR 596 (Fall 2025).

## Quick start

Set up a virtual environment, install dependencies, and build the site locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Build with Jupyter Book (v2+)
jupyter-book build .
```

Alternative: using the MyST CLI (`myst`) which provides similar commands:

```bash
# Install mystmd (if you prefer the myst CLI)
pip install mystmd
# Initialize (if needed) and build
myst init  # optional: creates myst.yml and basic structure
myst build
```

Serve the built site for local preview:

```bash
python -m http.server -d _build/html 8000
# then open http://localhost:8000
```

Alternative live preview with the CLI (auto-rebuilds and serves):

```bash
# Jupyter Book (v2+) local dev server
jupyter-book start

# Or with the MyST CLI
myst start
```

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

We recommend adding a GitHub Actions workflow that builds the Jupyter Book on PRs and pushes the `_build/html` to GitHub Pages or another hosting provider. PRs should include a successful build check before merging.

Resources:
- Jupyter Book 2 docs: https://next.jupyterbook.org/
- MyST Markdown docs: https://mystmd.org/

## Excluded files

Files listed in `myst.yml` `exclude:` (for example `TODO.md` and `CONTRIBUTING.md`) are intentionally not published to the public site. Do not remove these entries unless you want them visible.

## License & contact

- Content: CC-BY-4.0
- Code: MIT
- Contact: `@drannarosen` — alrosen@sdsu.edu

