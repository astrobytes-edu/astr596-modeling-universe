## Repository AI / Copilot Instructions

Purpose: provide concise, repo-specific guidance to GitHub Copilot-style coding agents and contributors so automated changes and PRs are predictable, safe, and aligned with maintainers' expectations.

- **Minimal, surgical edits:** Prefer the smallest change that fixes the issue. Avoid broad refactors or style-only changes unless asked.
- **Document every change:** For non-trivial edits, include a short description in the PR and update `CHANGELOG.md` or `README.md` when appropriate.
- **CI-first approach:** Run or simulate the relevant build steps before opening PRs. For docs and site changes, prefer `myst build --html` and `myst build --typst` locally to validate HTML and PDF outputs.
- **Do not modify deploy workflows without approval:** Changes to `.github/workflows/deploy.yml` require a project maintainer review and CI green checks. If you must update the workflow, explain why in the PR description and include a rollback plan.
- **Dependency policy:** Keep heavy packages (e.g., `jax`, `jaxlib`) out of default `requirements.txt` and `environment.yml`. Use `requirements-jax.txt` for opt-in installations and document optional installs in `README.md`.
- **MyST/Jupyter Book usage:** Use `myst` CLI commands as canonical operations. For PDF builds prefer `myst build --typst` (Typst template) and include any local template downloads in the repo or reference exact release URLs. Avoid `--execute` in CI unless notebooks are fully pinned and deterministic.
- **Templates and assets:** If using an external Typst or LaTeX template, vendor a copy under `ci-templates/` and reference it explicitly in `.github/workflows/deploy.yml` to avoid 404s in CI.
- **Testing changes:** When code changes include runnable examples or notebooks, add a minimal test or a short CI job that runs `myst build --html` and checks for non-empty `_build/html/index.html` and any exported PDFs under `_build/exports/`.
- **Sensitive data:** Never add secrets or credentials to the repository. Use GitHub Secrets for CI tokens and `pages` deployments.
- **Clear commit messages:** Use imperative tense and reference the issue or TODO when present (e.g., "Fix myst PDF export by vendoring lapreprint-typst template (#123)").

Quick local validation steps (macOS / zsh):

```bash
# create a lightweight environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install myst-cli nbclient nbconvert jupytext
# build HTML
myst build --html
# build Typst PDF (requires typst installed or myst will use remote template)
myst build --typst
```

When creating PRs, include these checks in your description:
- Files changed are minimal and scoped
- `myst build --html` succeeds locally
- `myst build --typst` succeeds locally and any templates are vendored or explicitly referenced
- No heavy optional packages were added to `requirements.txt` or `environment.yml`

Contact: maintainers listed in `README.md`.
