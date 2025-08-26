# MyST Build Guide for ASTR 596 (MyST-only)

This file documents the recommended local development and GitHub Pages deployment workflow for the ASTR 596 site using the MyST CLI. This guide intentionally omits PDF/Typst export steps — PDFs are disabled for this repository.

Checklist

- **Python**: Use a virtual environment and install `requirements.txt`.
- **MyST CLI**: Install via `npm` or `pip` (one of the options below).
- **Local preview**: Use `myst start` and `myst build --html`.
- **Deployment**: Build HTML and upload `_build/site` with GitHub Actions.

---

## Prerequisites

- Python 3.10+ (recommended)
- git
- Node.js (optional, required only if you prefer the npm installer for MyST)

### Create and activate a virtual environment (recommended)

```bash
# macOS / Linux (zsh):
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies from repo
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer a conda environment:

```bash
conda env create -f environment.yml
conda activate astro
```

## Installing the MyST CLI

Choose one method:

- npm (CLI via Node):

```bash
# optional: install globally
npm install -g mystmd

# or run with npx (no global install):
npx mystmd start
```

- pip (Python install):

```bash
pip install mystmd
```

Verify the CLI:

```bash
myst --version
# or
mystmd --version
```

## Local development

Start the live-reload server (recommended for editing content):

```bash
myst start
```

Notes:

- MyST will serve the site locally (default port 3001). The terminal shows the server URL.
- If port 3001 is occupied, pick another port with `myst start --port 8080`.

Build the site to static HTML (`_build/site`):

```bash
myst build --html
```

Preview the built site locally:

```bash
python -m http.server --directory _build/site 8000
# then visit http://localhost:8000
```

Clean build artifacts:

```bash
myst clean         # prompts before removing
myst clean --all   # remove templates, cache, and all build artifacts
```

## GitHub Actions deployment (recommended)

The GitHub Action should build the HTML and upload the generated `_build/site` directory to GitHub Pages. Example minimal deploy job:

```yaml
name: Build and deploy MyST site

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pages: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 18.x
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install MyST CLI
        run: npm install -g mystmd
      - name: Install Python deps
        run: pip install -r requirements.txt
      - name: Build HTML site
        run: myst build --html
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./_build/site
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

Notes:

- Do not enable Typst/LaTeX workflows unless you intentionally add those toolchains to CI.
- Use the `BASE_URL` env var only when deploying to a project subpath. CI can set it if needed.

## Troubleshooting

- Port already in use:

```bash
myst start --port 8080
# or kill process
lsof -ti:3001 | xargs kill -9
```

- `myst` command not found:

```bash
# If installed via npm globally, ensure npm global bin is on PATH
export PATH="$HOME/.npm-global/bin:$PATH"
# If installed via pip, ensure local bin is on PATH
export PATH="$HOME/.local/bin:$PATH"
```

- Build errors:

```bash
myst clean --all
myst build --html --verbose
```

## Quick reference

- Start dev server: `myst start`
- Build HTML: `myst build --html`
- Clean: `myst clean --all`
- Preview built site: `python -m http.server --directory _build/site 8000`

---

If you'd like, I can also add a short note to `README.md` stating that PDF/Typst export is disabled and how to re-enable it (CI and toolchain requirements). Do you want that added?