---
title: Course Schedule & Important Dates
subtitle: "ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

*Course Reading Schedule will be updated weekly, please check often!*

## Schedule Structure

**Fall 2025 Semester:** August 25 - December 18, 2025

**Mondays:** New project assigned via GitHub Classroom; previous project & growth memo due at 11:59 PM

**Fridays:** Class meeting (11:00 AM - 1:40 PM)

- 11:00-11:40: Interactive review, Q&A, clarify requirements
- 11:40-1:40: Hands-on pair programming (rotate partners every 20-25 min)

The projects follow a deliberate progression from **deterministic simulation** → **stochastic methods** → **statistical inference** → **machine learning**, mirroring how computational astrophysics evolved as a field. Each project builds on previous code and concepts, with scaffolding levels that decrease over time.

(due-dates)=
## Master Project Timeline & Due Dates

*Short Projects are due on Wednesdays by 11:59 pm, see due dates below.*

| Project | Title | Duration | Assigned | Due | Core Concept | AI Phase |
|---------|-------|----------|----------|-----|--------------|----------|
| 1 | Stellar Populations | 1.5 weeks | Aug 29 | Sept 10 | Object-oriented design & vectorization | Phase 1 |
| 2 | N-Body Dynamics | 2 weeks | Sept 10 | Sept 24 | ODE integration, stability, sampling methods (IMF, positions, velocities) | Phase 1 |
| 3 | Monte Carlo Radiative Transfer | 3 weeks | Sept 24 | Oct 15 | Stochastic methods & photon transport | Phase 1 |
| 4 | Bayesian Cosmological Inference | 3 weeks | Oct 15 | Nov 5 | Bayesian statistics & MCMC | Phase 2 |
| 5 | Gaussian Process Emulation | 3 weeks | Nov 5 | Nov 26 | Non-parametric models, emulation & JAX refactoring | Phase 2 |
| Final | Neural Networks with JAX | 4.5 weeks | Nov 19 | Dec 18 | Deep learning & autodiff | Phase 3 |

(ai-phases)=
## AI Scaffolding Phase Transitions

:::{important} Phase Boundaries are Fixed
Phase transitions occur at project submission deadlines. You remain in your current phase until you submit the final project of that phase.
:::

:::{list-table} AI Scaffolding Phases
:header-rows: 1
:widths: 20 30 35

* - Phase
  - Period
  - Coverage
* - **Phase 1: Foundation Building**
  - Projects 1, 2, and 3
  - Struggle first, AI for debugging only after 30 minutes. Ends when you submit Project 3.
* - **Phase 2: Strategic Integration**
  - Projects 4 and 5
  - Strategic AI use after baseline implementation. Ends when you submit Project 5.
* - **Phase 3: Professional Practice**
  - Final Project only
  - Professional AI integration as productivity multiplier. Begins only AFTER Project 5 submission.
:::

### Phase Transition Self-Check

**Entering Phase 2 (Oct 14):** Can you debug most issues without AI assistance? Can you identify when your code violates physical principles?

**Entering Phase 3 (Nov 25):** Can you verify AI suggestions against documentation? Do you understand why code works, not just that it works?

## Weekly Topics & Course Readings

| Week | Date | Topic | AI Phase | Reading |
|------|------|-------|----------|---------|
| 1 | Aug 29 | Python Fundamentals & <br> Object-Oriented Programming | Phase 1 | **Primary:** Python Ch. 5-6 (Functions, OOP);<br> **Reference:** Ch. 1-4 (Python Fundamentals) |
| 2 | Sept 5 | NumPy, Vectorization, Linear Algebra Review | Phase 1 | **Primary:** Python Ch. 7-8 (NumPy, Matplotlib) |
| 3 | Sept 12 | Intro to Numerical Methods, Statistical Foundations, & <br> Random Sampling | Phase 1 | [Linear Algebra Module 1](../04-mathematical-foundations/00a-linear-algebra-core-module.md), <br> Intro to Numerical Methods *(coming soon...)*,<br> [Statistical Thinking Module 1](../05-statistical-thinking/module-1-how-nature-computes/) |
| 4 | Sept 19 | Numerical Methods (cont.), <br> N-body dynamics & Conservation Laws, The Virial Theorem | Phase 1 | Statistical Thinking [Module 2](../05-statistical-thinking/module-2-from-particles-to-stars/) & [Module 3](../05-statistical-thinking/module-3-when-stars-become-particles/) |
| 5 | Sept 26 | Intro to Radiative Transfer, Numerical Integration | Phase 1 | TBD |
| 6 | Oct 3 | MCRT Implementation, Solving Linear Systems | Phase 1 | [Linear Algebra Module 2](../04-mathematical-foundations/00b-linear-algebra-stats-module.md); TBD |
| 7 | Oct 10 | Intro to Bayesian Inference & Cosmology | Phase 1 | TBD |
| 8 | Oct 17 | Bayesian Inference (cont.), <br> Intro to MCMC | Phase 2 | TBD |
| 9 | Oct 24 | MCMC & HMC | Phase 2 | TBD |
| 10 | Oct 31 | Intro to ML & Advanced Optimization | Phase 2 | TBD |
| 11 | Nov 7 | Gaussian Processes | Phase 2 | TBD |
| 12 | Nov 14 | Gaussian Processes & Intro to JAX | Phase 2 | TBD |
| 13 | Nov 21 | Neural Networks & JAX | Phase 2 | TBD |
| 14 | Nov 28 | **THANKSGIVING BREAK** (Optional lab session earlier in week if students want) | Phase 3 | TBD |
| 15 | Dec 5 | Neural Networks & JAX | Phase 3 | TBD |
| Finals | Dec 18 | Final Presentations (Time TBD) | Phase 3 | — |

## Important Dates (Fall 2025)

:::{list-table} Project Deadlines (all 11:59 PM PT)
:header-rows: 1
:widths: 50 50

* - Assignment
  - Due Date
* - Project 1: Stellar Populations + Growth Memo 1
  - Wednesday, Sept 10
* - Project 2: N-Body Dynamics + Growth Memo 2
  - Wednesday, Sept 24
* - Project 3: MCRT + Growth Memo 3
  - Wednesday Oct 15 → **Phase 1 ends**
* - **Begin Phase 2**
  - Thursday, Oct 16
* - Project 4: Bayesian/MCMC + Growth Memo 4
  - Wednesday, Nov 5
* - Project 5: GP + JAX N-body + Growth Memo 5
  - Wednesday, Nov 26 → **Phase 2 ends**
* - **Begin Phase 3**
  - Thursday, Nov 27 (after Project 5 submission)
* - Final Project: Neural Networks + Technical Growth Synthesis
  - Thursday, Dec 18
:::

## Pair Programming Schedule

Partners are randomly assigned each project to maximize diverse collaboration:

| Project | Strategy | Note |
|---------|----------|------|
| 1-5 | Random pairs each project | Different perspectives accelerate learning |
| Final | Individual | Show your independent capability |

**Collaboration Philosophy**: Your assigned partner is your "home base" - primary development happens together. Other classmates are "consultants" - discuss ideas, debug together, share insights. Just ensure core implementation happens with your partner.
