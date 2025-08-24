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
```

1. Install dependencies and build the site:

```bash
pip install -r requirements.txt
jupyter-book build .
```

1. (Optional) If you add Python code, run tests:

```bash
pytest -q
```

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
