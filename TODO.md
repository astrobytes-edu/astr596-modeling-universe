<!-- TODO.md: Master task list for the `astr596-modeling-universe` course website -->
# ASTR 596 Course Website TODO

This file is the living task list for the `astr596-modeling-universe` course website. Use it to track work, prioritize items, and assign owners/estimates.

**How to use**
- Add tasks under the most relevant section.
- Use the checkbox format (`- [ ]`) to mark progress.
- Optional metadata: `@owner`, `!priority`, `~estimate` (hours).
- Keep items brief and move completed items under the **Done** section or check them off.

**Legend**
- `@owner` — person responsible
- `!priority` — high/med/low
- `~estimate` — rough hours

**Sections**

## Website Content

- [ ] Update `01-course-info/01-astr596-syllabus-fall25.md` with final schedule @anna !high ~2
- [ ] Finalize course overview and learning objectives @anna !high ~1
- [ ] Add instructor/TA contact and office hours to `01-course-info` @anna !high ~0.5
- [ ] Review and publish `02-getting-started` pages @TA !med ~2
- [ ] Audit images in `assets/` and optimize for web @anna !med ~3

## Course Materials (Lectures, Labs, Notes)

- [ ] Convert lecture notebooks to MyST / Jupyter Book 2.x pages @TA !high ~6
- [ ] Add downloadable slides for each week @instructor !med ~4
- [ ] Link readings and external resources with DOIs @anna !low ~3

## Assignments & Projects

- [ ] Create assignment templates in `08-short-projects` @TA !high ~4
- [ ] Finalize project descriptions and grading rubric @instructor !high ~3
- [ ] Add auto-grading notes and expected deliverables @TA !med ~2

## Manim Media & Animations

- [ ] Add Manim example scenes to `manim-media/` and link from site @anna !med ~4
- [ ] Create a short tutorial page: "Using Manim for assignments" @anna !low ~2
- [ ] Validate that `manim-media/requirements.txt` is current @anna !low ~0.5

## Drafts Cleanup

- [ ] Review files moved into `drafts/` and decide publish/archive action @anna !med ~3
- [ ] Restore or incorporate useful content from old drafts into course pages @TA !med ~4

## Site Build, CI & Deployment

- [ ] Add GitHub Actions to build the site on push and check for broken links @anna !high ~3
- [ ] Add `requirements.txt` and docs build instructions in `README.md` @anna !med ~1
- [ ] Configure automatic deploy to GitHub Pages or Netlify @anna !med ~2

## Accessibility & SEO

- [ ] Run accessibility audit and fix major issues (alt text, headings) @TA !med ~3
- [ ] Add metadata and improve page titles/descriptions for SEO @anna !low ~2

## Tests & Quality

- [ ] Add basic content validation script (check for broken internal links) @TA !low ~2
- [ ] Add unit tests for any interactive components or scripts @dev !low ~3

## Administration

- [ ] Update `LICENSE` and contributor guidelines if needed @anna !low ~1
- [ ] Add contributor quickstart to `CONTRIBUTING.md` @anna !low ~1

## Done

- [ ] (example) Moved old drafts into `drafts/` — @anna

---

If you'd like, I can:

- convert each high-level item into smaller actionable subtasks,
- scaffold GitHub Actions for site building,
- or open issues in this repo for each `!high` item. Tell me which next step you prefer.
