# Growth Memo Guidelines

This document describes the required Growth Memo for each short project. The Growth Memo is a concise reflection (recommended 1–2 pages) submitted alongside your project implementation that documents what you built, why you made technical choices, how you verified results, and how you used (and verified) AI tools.

## Purpose

- **Reflect** on your learning progress, small wins, and decisions — this is a low-key, metacognitive exercise.
- **Optional evidence**: short notes, minimal repro, or a single figure are helpful but not required.
- **Record** AI/tool usage briefly if used (see course AI policy: `../01-course-info/03-astr596-ai-policy.md`).

## Submission Instructions

- **Filename**: `growth_memo.md` (preferred) or `growth_memo.pdf` placed in the project repository root.
- **Due**: by **11:59 PM on the Monday** the project is due (see schedule: `../01-course-info/02-astr596-schedule.md`).
- **Length**: 1–2 pages (approx. 400–800 words) or equivalent PDF.
-- **Where to include evidence (optional)**: a short note in your repo (e.g., `notes/effort.md`) or a minimal script is fine. New figures, extensive commit histories, or tests are not required for the Growth Memo.
-- **Submission**: place `growth_memo.md` (preferred) or `growth_memo.pdf` in your project repo root. A short note in the PR description is also fine.

## Student Template (use these headings)

1. **Title, Date, Project Number**
2. **Short summary** (2–3 sentences): one-line scientific goal, one-line technical goal
3. **What I built (brief)**: 1–3 bullets describing the main idea or component you worked on.
4. **Key challenge(s)** and how you addressed them (short paragraph or bullets — 2–4 lines).
5. **Evidence & verification (optional)**: a single sentence if you ran a quick check, test, or figure — otherwise leave blank.
6. **AI Usage Reflection** (required if AI was used): include tool name, short prompt(s), what AI suggested, how you verified the suggestion, and the final decision
7. **What I learned & next steps**: 3–5 concise bullets
8. **Optional artifacts**: links to notebooks, plots, and example outputs

## AI Documentation Requirements (Course Policy)

- If you used AI for any code-related task, include a 3-line in-code note directly above the edited function/block in your code:

```python
# AI: [Tool] suggested [very short what]
# Verified: [document/source or quick check: test/unit/plot]
# Because: [one short reason you kept it]
```

- In the Growth Memo `AI Usage Reflection` section, include the prompt(s) you used (or a short paraphrase), describe the verification you performed, and state whether you accepted the suggestion fully, partially, or rejected it.
- Follow the three-phase scaffold in `../01-course-info/03-astr596-ai-policy.md` for allowed AI activity by week.

## Quick Suggested Structure (very low effort)

- Title, Date, Project Number (1 line)
- Short summary (2–3 sentences) — what did you try?
- One key challenge and what you tried (2–4 lines)
- One small win or insight (1–2 lines)
- AI usage reflection (one short paragraph) if applicable
- Optional: one-line note pointing to a script or figure if you added one

## Instructor Notes (grading intent)

- The Growth Memo is primarily **metacognitive**. I read these to see growth over time, not to audit extensive evidence.
- Short, honest reflections that show thinking and next steps are valued over long technical appendices.
- If a student includes evidence (scripts/figures), that's great — but it's optional.

## Grading Notes & Expectations

-- You should be able to explain any code you submit for the project; the Growth Memo can note gaps and a plan to address them.
-- Small, honest commits are fine but not required for the memo itself.
-- AI use should still be documented briefly when used.

## Tips for a Useful Growth Memo

- Be concise and honest — 5–10 bullets or a short 1-page reflection is perfect.
- Focus on what you learned and one small next step.
- Use the AI section to show how you verified suggestions, not just that you used them.

## Example (very short)

Title: Project 1 — Stellar Luminosity Model

Short summary: Implemented a vectorized luminosity model and tests for main-sequence scaling.

What I built: `src/luminosity.py` (vectorized function), `tests/test_luminosity.py` (limiting-case checks)

Key challenge: initially had indexing mismatch; fixed by reshaping arrays in `src/luminosity.py:calc_luminosity` (see commit `abc123`). Minimal repro: `scripts/repro_lum.py`.

AI usage: Asked ChatGPT for debugging hints after 45 minutes of effort; AI suggested checking broadcasting shapes. Verified by adding assertions and unit test `tests/test_luminosity.py`.

What I learned: importance of shapes and vectorized thinking; next step: add error handling and docstrings.

