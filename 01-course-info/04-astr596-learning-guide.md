# ASTR 596: Course Learning Guide

## Quick Links

- [Course Philosophy](why-astr596-is-different)
- [AI Usage Policy](astr596-ai-policy)
- [Project Submission Guide](short-projects/0_project_submission_guide)
- [Software Setup](../02-getting-started/02-software-setup)

\:::{admonition} Quick Start (read this first) {#sec-quick-start} **Before class (≤30 min):** skim the chapter → attempt the project for **30 min** (capture **effort evidence** per the AI Policy) → note two questions.\
**In class:** ask; pair program; debug one issue to closure.\
**After class (≤45 min):** implement one increment; add one figure; write one takeaway.\
[Workflow](#sec-learning-workflow) · [Debug & Test](#sec-debug-test) · [Getting Help](#sec-getting-help)\
**Note:** This course has **no exams** — progress is demonstrated through projects, growth memos, and participation. :::

\:::{admonition} What you can expect from me / from you **From me:** clear milestones, fast feedback, chances to re-try, and honest coaching.\
**From you:** start early, verify claims, ask for help with context, and explain your code in your own words. :::

\:::{dropdown} Quick Reference Card (toggle) **Daily Checklist**

-

**When Stuck**

1. Re‑read error
2. Minimal repro (≤20 lines)
3. Check docs
4. Rubber‑duck
5. Search *specific* error
6. Ask with context

**Git Workflow**

```bash
git pull origin main
git checkout -b feat/short-name
# ... work ...
git add -A && git commit -m "message"
git push -u origin feat/short-name
```

\:::

---

## Effective Learning Workflow {#sec-learning-workflow}

### Before Class

1. **Read actively** — skim headings, figures, equations.
2. **Try examples** — type code yourself (no paste).
3. **Attempt project start (30 min)** — capture **effort evidence** (see AI Policy): link a relevant doc section; a ≤20‑line minimal repro; one failed approach + why it failed.
4. **Note questions** — two precise questions for class/Slack.

### During Class

1. **Ask high‑leverage questions.**
2. **Pair program** — switch driver/navigator every 20–25 min.\
   **Driver/Navigator (20–25 min):** Driver talks while typing; Navigator asks “what/why” and tracks TODOs. Switch on a timer; summarize decisions in 2 bullets.
3. **Debug together** — close one issue to completion.
4. **Record decisions** — brief bullets in README or notebook.

### After Class

1. **Implement incrementally** — one small, verified feature.
2. **Visualize** — plot and annotate with units.
3. **Reflect** — one paragraph: what worked, next step.
4. **Commit/push** — treat your repo as a lab notebook.

## Algorithm Planning & Pseudocode {#sec-algo-planning}

**Why plan first:** coding without a plan leads to brittle code and wasted time. A 5‑minute sketch saves hours.

**5‑minute pseudocode habit (before you open the editor)**

1. Define **inputs/outputs** (with units).
2. Write **3–10 steps** in plain language.
3. Note **assumptions** and **edge cases**.
4. Mark **checks** (units, limits, shape).
5. Only then start coding the smallest step.

**Template**

```
# Function: what does it compute?
# Inputs: ... (units)
# Outputs: ... (units)
# Steps:
# 1) ...
# 2) ...
# 3) ...
# Checks: units/limits/shape
```

**Example (luminosity distance ****\(D_L(z)\)****)**

```
# compute_DL(z, params)
# Inputs: z (unitless), params = {H0 [km/s/Mpc], Om0, Ode0}
# Output: D_L [Mpc]
# Steps:
# 1) define H(z) from params
# 2) integrate 1/H(z) from 0→z (numerical)
# 3) Dc = c * integral
# 4) DL = (1+z) * Dc
# Checks: DL(z) monotonic; Om0→0 limit; units via your **Units** module
```

**Test plan (before code)**

- Compare small‑z series against analytical approx.
- Check \(D_L(z=0)=0\); verify units using your **Units** module.

---

## Evidence‑Based Learning Frameworks {#sec-evidence-frameworks}

*A transparent, honest map of what actually improves learning and transfer.*

### Retrieval Practice

- **Why it works:** strengthens memory traces; outperforms re‑reading.
- **Use here:** forecast quizzes; “blank‑sheet” code (rebuild a function from memory); oral minute on algorithm steps.

**Forecast Quiz (5 minutes, before you read)**

1. Without notes, write two equations or invariants you expect to need.
2. Sketch the steps you’d take to compute the main quantity (e.g., \(D_L(z)\)).
3. After reading, check what you got right/wrong; correct in place.

### Spacing & Interleaving

- **Why:** spacing boosts retention; interleaving improves discrimination/transfer.
- **Use:** revisit old content weekly; mix tasks (Friedmann distances ↔ likelihoods ↔ JAX vectorization) in problem sets.

### Worked Examples → Example‑Problem Pairs → Faded Guidance

- **Why:** reduces cognitive load then scaffolds independence.
- **Use:** each section: one worked example, one isomorphic problem, then a “faded” version with steps removed.

**Faded version checklist**

- Remove numeric scaffolds; keep symbols.
- Hide one intermediate identity (name it).
- Ask for a unit/limit check instead of providing it.

### Self‑Explanation

- **Why:** explaining steps reveals gaps and builds schema.
- **Use:** include two lines under key solutions: *principle used* and *why it applies*.

**Two lines to add under key solutions**

- *Principle used:* e.g., “chain rule on \(D_L(z)\); small-z limit; log-sum-exp for stability.”
- *Why applicable:* e.g., “expansion is monotonic in z; avoiding overflow in the likelihood.”

### Productive Failure (Guided Struggle)

- **Why:** early struggle increases post‑instruction learning.
- **Use:** 10–15 min “invent the method” prompts before the canonical approach.

### Multiple Representations (Dual Coding)

- **Why:** linking equations, graphs, and words builds robust understanding.
- **Use:** require graph ↔ equation ↔ verbal mapping for core ideas (e.g., H(z), D\_L(z)).

### Error Analysis & the Hypercorrection Effect

- **Why:** correcting confident errors produces large gains.
- **Use:** annotate flawed solutions; submit fixes in growth memos.

### ICAP Framework (Interactive > Constructive > Active > Passive)

- **Why:** learning tracks engagement mode.
- **Use:** prioritize pair explanations, whiteboarding, student‑generated derivations.

### Deliberate Practice

- **Why:** targeted practice on weak subskills accelerates growth.
- **Use:** short, focused drills (units checks; limit cases; vectorization patterns).

### Metacognition (Assignment Wrappers)

- **Why:** planning/monitoring improves regulation and transfer.
- **Use:** one‑page wrapper with strategies used, errors made, next adjustments.

\:::{admonition} Evidence at a Glance Retrieval > re‑reading; active learning > lecture; spacing & interleaving improve transfer; worked‑example → faded guidance builds independence; self‑explanation and error analysis are high‑yield. (See course site references.) :::

---

## Debugging & Testing Playbook {#sec-debug-test}

### The Systematic Approach

1. **Read the error** — slowly.
2. **Pin the line** — where exactly?
3. **Check assumptions** — inputs, shapes, units, limits.
4. **Simplify** — minimal repro (≤20 lines).
5. **Instrument** — print key values; plot intermediates.
6. **Use the debugger** — step, inspect, continue.
7. **Take a break** — reset attention, then retry.

### Using the Debugger (`breakpoint()` / pdb)

```python
def problematic_function(x):
    result = x * 2
    breakpoint()  # Execution stops here (Python 3.7+)
    return result / 0  # Obviously wrong
```

**pdb cheatsheet:** `n` next · `s` step into · `c` continue · `l` list · `p x` print · `q` quit\
**IPython:** after a crash, run `%debug` to drop into the last exception.

### Common Python Pitfalls

- **Division semantics:** Python 3: `/` is float; use `//` for integer division.
- **Mutable defaults:** avoid `[]`/`{}` as defaults; use `None` + set inside.
- **Broadcasting:** check `.shape` carefully.
- **Off‑by‑one:** remember 0‑indexed slices.
- **Tabs vs spaces:** don’t mix.

### Testing Strategies

- **Known solutions** — reproduce textbook cases.
- **Limiting cases** — e.g., \(\Omega_\Lambda \to 0\).
- **Conservation laws** — where applicable.
- **Units** — use your **Units** module (from Project 1) and verify conversions.
- **Visualization** — plots often reveal bugs.

**Units sanity snippet (Course Units module)**

````python
# Example API; adapt to your Units module
from units import U, Q  # or your names

H0 = Q(70, U.km/U.s/U.Mpc)   # 70 km/s/Mpc
H0_s = H0.to(U.s**-1)        # convert to 1/s
assert H0_s.value > 0


:::{admonition} Minimal Repro (≤20 lines)
Include this repro in your PR/issue so others can run it in isolation.

- Expected vs actual behavior
- Short code to reproduce (≤20 lines)
- Version info: python, numpy, jax
  :::

---


### Course Units Module (reference) {#sec-units-module}
- You will **build a small Units/Quantity library in Project 1** and extend it across the term.
- **Design goals:**
  - explicit base units (e.g., km, s, Mpc) and **dimension system** (L, T, …)
  - derived units via `*`, `/`, and exponentiation; unit simplification
  - `Quantity(value, unit)` type with `.to(target_unit)` conversion
  - **dimension checking** that raises on incompatible conversions
  - immutable quantities; pure conversions (no hidden globals)
- **Recommended invariants/tests:** identity `Q(1,u).to(u)==1`, round‑trip `Q(v,a).to(b).to(a)≈v`, dimensional errors on `km.to(s)`.
- **Starter snippet (adapt to your API):**
```python
from units import U, Q
v = Q(300_000, U.km/U.s)
lam = Q(500, U.nm)
nu = (U.c / lam).to(U.Hz)     # if you represent constants as quantities
````

| Resource                                                                                                                 | Purpose                     |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| [https://jax.readthedocs.io/](https://jax.readthedocs.io/)                                                               | Core JAX API and tutorials  |
| [https://jax.readthedocs.io/en/latest/thinking\_in\_jax.html](https://jax.readthedocs.io/en/latest/thinking_in_jax.html) | Mental model for JAX        |
| [https://flax.readthedocs.io/](https://flax.readthedocs.io/)                                                             | Flax neural‑network library |

\:::{admonition} Docs‑First Rule When stuck: consult the docs → attempt minimal repro → then (and only then) use AI for clarification. Place the **AI / Verified / Because** 3‑line note above any code you accept.

**When you do use AI, ask like this:**\
“Given the following course excerpt [paste small, relevant passage], what is a numerically stable way to compute \(D_L(z)\) at high z? Return a short rationale and one limit test. If unsupported by the excerpt, say so.” :::

---

## Getting Help {#sec-getting-help}

**Immediate help:** blocked >1 hour; can’t install; don’t understand the prompt even after reading.\
**Office hours / class:** approach choice; after genuine attempt; debugging with context.

**Ask like a pro (template)**

```
Context: I’m working on [specific part] of [project].
Attempt: I tried [approach] (code snippet/minimal repro).
Expected: I expected [outcome]; Actual: I got [error/behavior].
Hypothesis: I think the issue is [your guess].
Question: Can you help me understand [specific aspect]?
```

**Red flags (ask now):** “It works but I can’t explain why”; random changes; solution is 10× longer than expected; overwhelm >3 days.

---

## Project Workflow (milestones, not micromanagement) {#sec-project-workflow}

**Principle:** start early, iterate in small steps, scaffold your work. Use milestones as guardrails—not a daily schedule.

### Core milestones (suggested windows)

- **Kickoff (days 1–2):** clarify the question; clone template; run a sanity script; list risks.
- **Baseline (first third):** get a minimal model running (one correct figure/metric).
- **Deepening (second third):** improve correctness/performance; add one new capability (e.g., likelihood, refactor, vectorization).
- **Polish (final third):** clean figures; write short results; tighten repo; prepare demo.
- **Repro pass (24h before submit):** fresh clone → single command regenerates the key result.

### Choose‑your‑own cadence (pick a few per work session)

**Foundation**

- read one section + extract equations
- set up config/env; pin versions
- create a minimal repro for the core calculation

**Build**

- implement one function
- replace a loop with vectorization
- add one plot with labeled axes + units

**Validate**

- check one limit case
- compare to a known solution/textbook value
- add/run a simple test or assert

**Communicate**

- write a 3–5 sentence log or figure caption
- file a TODO/issue; link a doc section
- record a 60‑sec demo clip (optional)

### Minimum weekly evidence (to keep you honest)

- A visible commit trail (small, descriptive commits)
- One new figure or metric with units/labels
- A brief progress note (what worked, next step)

\:::{dropdown} Example lightweight timelines (optional)

- **Fast‑start:** kickoff Day 1 → baseline by Day 3 → deepen Day 6–9 → polish Day 12–13 → repro Day 13–14.
- **Research‑heavy:** kickoff Day 1–2 → baseline by Day 5 → literature/analysis Day 6–10 → polish/repro Day 11–14. :::

**Project structure (suggested)**

```
project_name/
├── README.md                     # how to run + purpose
├── environment.yml               # or requirements.txt
├── src/
│   ├── __init__.py
│   ├── physics.py                # physics calculations
│   ├── numerics.py               # numerical methods
│   └── plotting.py               # visualization
├── tests/                        # provided or optional (start simple)
├── data/
│   └── input_files/
├── outputs/
│   └── figures/
└── main.py
```

---

## Work Habits & Mindset (Appendix) {#sec-habits}

### Focus & Distraction Tools

- 25→45 min focus blocks; notifications off; phone in another room.
- Optional site blockers during work sessions.

### Time Management & Energy

- 90‑min daily minimum: plan → implement → test → document.
- Respect 90–120 min ultradian cycles; real breaks between cycles.

### Growth Mindset

- Reframe: “I’m not there *yet*.”
- Normalize struggle; track progress; celebrate small wins.

### Imposter Syndrome (Reality Check)

- Nearly everyone reports it at some point; share struggles; document wins.

---

\:::{dropdown} Quick Reference Card (printable) **Daily Checklist**

-

**When Stuck**\
Re‑read error → minimal repro → docs → rubber‑duck → search → ask

**AI Note Template (place above the edited block)**

```
# AI: [Tool] suggested [very short what]
# Verified: [doc / limit test / unit / plot]
# Because: [1 short reason kept]
```

\:::

