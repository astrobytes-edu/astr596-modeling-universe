---
title: "ASTR 596: Course Learning Guide"
subtitle: "Modeling the Universe | **Instructor:** Anna Rosen"
exports:
  - format: pdf
---

%# ASTR 596: Course Learning Guide
:::{tip} Quick Start
**Before class (≤30 min):** skim the chapter → attempt the project for **30 min** (capture effort evidence) → note two questions.

**In class:** ask questions; pair program; close **one small bug** to completion.

**After class (≤45 min):** implement one increment; add one figure; write one takeaway.

**Note:** This course has **no exams** – progress is demonstrated through projects, growth memos, and participation.
:::

:::{important} What you can expect from me / from you
**From me:** clear milestones, fast feedback, and honest coaching on your progress.

**From you:** start early, verify claims, ask for help with context, and explain your code in your own words.
:::

:::{important} Quick Reference Card
:class: dropdown

## Daily Checklist

- [ ] Pull & create a short-name branch
- [ ] 30-min attempt (capture effort evidence)
- [ ] One small verified increment
- [ ] Plot something (axes + **units**)
- [ ] Commit with a descriptive message; push
- [ ] Write one takeaway line

## When Stuck

1. **Re-read error** – note exception, file:line, and first frame in your code
2. **Minimal repro** – smallest standalone script (≤20 lines) that reproduces the bug
3. **Check docs** – verify signature, types/shapes, units match your use
4. **Rubber-duck** – explain each step; predict → run → compare
5. **Search specific error** – exact exception + library/version
6. **Ask with context** – branch link, file:line, minimal repro, expected vs actual

## Git Workflow

```bash
git pull origin main
git checkout -b feat/short-name
# ... work ...
git add -A && git commit -m "message"
git push -u origin feat/short-name
```

:::

---

## Learning Workflow

 
### Before Class

1. **Read actively** – skim headings, figures, equations
2. **Try examples** – type code yourself (no paste)
3. **Attempt project start (30 min)** – capture effort evidence
4. **Note questions** – two specific questions for class

### During Class

1. **Ask high-leverage questions**
2. **Pair program** – switch driver/navigator every 20-25 min
3. **Debug together** – close one issue to completion
4. **Record decisions** – brief notes in README

### After Class

1. **Implement incrementally** – one small, verified feature
2. **Visualize** – plot with **units in axis labels**
3. **Reflect** – one paragraph: what worked, next step
4. **Commit/push** – treat your repo as a lab notebook

---

## Evidence-Based Learning Principles

### Core Techniques

**Spacing:** Spread study sessions over time with gaps between them. Three 30-minute sessions over different days beats one 90-minute session.

**Interleaving:** Mix different topics within study sessions rather than focusing on one at a time. Alternate between different problem types to improve recognition of which tool to use.

**Retrieval Practice:** Test yourself before reading (forecast quiz), rebuild functions from memory, explain algorithms without notes. Strengthens memory more than re-reading.

**Metacognition:** Think about your own thinking. Monitor what you know (and don't know), recognize when you don't understand, and adjust your approach. Growth memos develop this skill.

### Applied in This Course

- **Forecast quizzes:** Before reading, predict key equations/steps you'll need
- **Worked examples → faded guidance:** Each section progresses from complete examples to partial scaffolding
- **Self-explanation:** Under solutions, note the principle used and why it applies
- **Productive failure:** 10-15 min struggle before seeking help builds deeper understanding
- **Multiple representations:** Connect equations ↔ graphs ↔ physical meaning

---

## Algorithm Planning

**Before coding, spend 5 minutes on pseudocode:**

1. Define inputs/outputs (with units)
2. Write 3-10 steps in plain language
3. Note assumptions and edge cases
4. Mark where to add checks (units, limits, shapes)
5. Only then code the smallest step

**Template:**
```text
# Function: what does it compute?
# Inputs: ... (units)
# Outputs: ... (units)
# Steps:
# 1) ...
# 2) ...
# Checks: units/limits/shape
```

**Computational thinking for astrophysics:**
-
**Computational thinking for astrophysics:**
- Break continuous equations into discrete steps
- Identify what varies (loop/vectorize) vs what's constant
- Consider numerical stability (overflow, underflow, cancellation)
- Think about scales: when is log-space better?

---

## Debugging Playbook

### Systematic Approach
1. **Read the error** – slowly, completely
1. **Read the error** – slowly, completely
2. **Pin the line** – where exactly?
3. **Check assumptions** – inputs, shapes, units
4. **Simplify** – minimal repro (≤20 lines)
5. **Instrument** – print key values; plot intermediates
6. **Use the debugger** – `breakpoint()` or `%debug` in IPython
7. **Take a break** – reset attention, then retry

### Minimal Reproducible Example
A standalone script that anyone can run:

- ≤20 lines of code
A standalone script that anyone can run:
- ≤20 lines of code
- Fixed inputs (no file dependencies)
- Clear expected vs actual behavior
- Version info if relevant

### Testing Strategies
- **Known solutions** – reproduce textbook cases
- **Known solutions** – reproduce textbook cases
- **Limiting cases** – check behavior at extremes
- **Conservation laws** – verify invariants
- **Units** – dimensional analysis
- **Visualization** – plots often reveal bugs

---

## Getting Help

### When to Ask

- Blocked >1 hour after genuine attempt
- Can't understand the prompt after reading
- Installation/environment issues

### How to Ask (Template)

```text
Context: I'm working on [specific part] of [project].
Attempt: I tried [approach] (minimal repro attached).
Expected: [outcome]
Actual: [error/behavior]
Hypothesis: I think the issue is [your guess].
Question: Can you help me understand [specific aspect]?
```
```text
Context: I'm working on [specific part] of [project].
Attempt: I tried [approach] (minimal repro attached).
Expected: [outcome]
Actual: [error/behavior]
Hypothesis: I think the issue is [your guess].
Question: Can you help me understand [specific aspect]?
```

Include: branch/commit link, file:line number, minimal repro

---

## Project Workflow

### Milestones (not daily schedule)
- **Kickoff (days 1-2):** Clarify question, run sanity check, list risks
- **Kickoff (days 1-2):** Clarify question, run sanity check, list risks
- **Baseline (first third):** Get minimal model running with one correct output
- **Deepening (second third):** Improve correctness/performance, add capability
- **Polish (final third):** Clean figures, document results
- **Repro check (24h before):** Fresh clone → regenerate key result

### Per Session (choose a few)
- Read one section, extract key equations
- Implement one function
- Add one plot with labeled axes + units
- Check one limit case
- Write 3-5 sentence progress note

### Every Plot Must Have
- Labeled axes with units
- Title or caption explaining what it shows
- Legend if multiple datasets

---

## Effort Evidence

**For each 30-min work session, capture:**
- Link to relevant documentation or textbook section
- Minimal repro of what you tried (≤20 lines)
- One failed approach and what you learned

**Optional additions:**
- One figure/metric with units
- Brief note on next steps

Store in `notes/effort.md` or similar. Not graded, but helps you track progress and supports growth memo writing.

---

## Growth Memos

Periodic reflections on your learning process and code evolution. These develop metacognition—your awareness of how you learn and solve problems. Details provided separately.

---

## Work Habits

### Focus Sessions
- 25-45 min focused blocks
- Notifications off
- Clear goal for each session

### Time Management
- 90 min daily minimum: plan → implement → test → document
- Respect natural energy cycles
- Take real breaks between sessions

### Growth Mindset
- Reframe: "I'm not there *yet*"
- Normalize struggle
- Track progress, celebrate small wins

---

## Key Reminders

1. **Every function needs a check** – limit case, known solution, or unit verification
2. **Commit early and often** – your repo is a lab notebook
3. **When stuck, simplify** – minimal repro reveals most bugs
4. **Plots need context** – axes, units, and explanations
5. **Growth over perfection** – document what you learned, not just what worked