# Contributing to ASTR 596: Modeling the Universe

Thank you for taking an interest in this repository. This project is primarily maintained by @drannarosen. The notes below capture the contribution conventions and quick commands to keep the site buildable and consistent.

Purpose
- Capture workflow and quick commands so contributors (including future TAs) can get up and running quickly.

Quick rules
- Owner: `@drannarosen`
- Work in feature branches named `feature/<short-name>` or `fix/<short-desc>`.
- Keep commit messages short and in imperative tense (e.g., "Add syllabus updates").
- Update `myst.yml` `exclude:` if you add files that should not be published to the site (example: `TODO.md`).

Build & test (local)

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build the site locally with Jupyter Book:

```bash
jupyter-book build .
```

3. Run the test suite (if any):

```bash
pytest -q
```

Formatting & quality
- Use `black .` to format Python code before committing if you use Black.
- Optionally run `pylint` or `flake8` to lint code noted in `requirements.txt`.

Pull requests & issues
- Open an issue to discuss non-trivial changes before starting work.
- Create PRs from feature branches; link the issue and add a short description of the change.

Minimal PR checklist
- [ ] Built the site locally and verified no build errors
- [ ] Checked for broken internal links
- [ ] Added or updated `myst.yml` `exclude:` entries if necessary

Contact
- Repo owner: `@drannarosen` (Anna Rosen)

This file is intentionally short — expand it later if you invite collaborators or TAs.
