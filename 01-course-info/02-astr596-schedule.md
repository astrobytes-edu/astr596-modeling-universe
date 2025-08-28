---
title: Course Schedule & Important Dates
subtitle: "ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

:::{note} Living Document & Python Course Textbook
This schedule is a living document and subject to change as the course progresses. This is the first offering of ASTR 596, and reading materials will be developed and posted to the course website throughout the semester. Weekly readings for Weeks 3+ are tentative and marked with asterisks (*) to indicate they may be adjusted based on course progression and student needs.

**About the Online Course Python Textbook:** The *"Scientific Computing with Python"* chapters serve primarily as reference material rather than required reading. Students should skim to identify gaps in their knowledge and use these chapters as a resource throughout the course when implementing projects. Focus on understanding concepts rather than memorizing syntax.
:::

## Schedule Structure

**Fall 2025 Semester:** August 25 - December 18, 2025

**Mondays:** New project assigned via GitHub Classroom; previous project & growth memo due at 11:59 PM

**Fridays:** Class meeting (11:00 AM - 1:40 PM)

- 11:00-11:40: Interactive review, Q&A, clarify requirements
- 11:40-1:40: Hands-on pair programming (rotate partners every 20-25 min)

The projects follow a deliberate progression from **deterministic simulation** → **stochastic methods** → **statistical inference** → **machine learning**, mirroring how computational astrophysics evolved as a field. Each project builds on previous code and concepts, with scaffolding levels that decrease over time.

## Master Project Timeline

| Project | Title | Duration | Assigned | Due | Core Concept | AI Phase |
|---------|-------|----------|----------|-----|--------------|----------|
| 1 | Stellar Populations | 1.5 weeks | Aug 29 | Sept 8 | Object-oriented design & vectorization | Phase 1 |
| 2 | N-Body Dynamics | 2 weeks | Sept 8 | Sept 22 | ODE integration, stability, sampling methods (IMF, positions, velocities) | Phase 1 |
| 3 | Monte Carlo Radiative Transfer | 3 weeks | Sept 22 | Oct 13 | Stochastic methods & photon transport | Phase 1 |
| 4 | Bayesian Cosmological Inference | 3 weeks | Oct 13 | Nov 3 | Bayesian statistics & MCMC | Phase 2 |
| 5 | Gaussian Process Emulation | 3 weeks | Nov 3 | Nov 24 | Non-parametric models, emulation & JAX refactoring | Phase 2 |
| Final | Neural Networks with JAX | 4.5 weeks | Nov 17 | Dec 18 | Deep learning & autodiff | Phase 3 |

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

## Weekly Topics & Learning Objectives

| Week | Date | Topic | AI Phase | Reading |
|------|------|-------|----------|---------|
| 1 | Aug 29 | Python Fundamentals & <br> Object-Oriented Programming | Phase 1 | **Primary:** Python Ch. 5-6 (Functions, OOP);<br> **Reference:** Ch. 1-4 (as needed) |
| 2 | Sept 5 | NumPy, Vectorization, <br> Stellar Physics, & <br> Numerical Methods | Phase 1 | **Primary:** JIT Astro Ch. 1 (Stellar Structure), JIT Math Ch. 1 (Error Analysis); <br> **Reference:** Python Ch. 7-8 (NumPy, Matplotlib) |
| 3 | Sept 12 | Advanced Integration & <br> Monte Carlo Sampling | Phase 1 | JIT Numerical Methods Ch. 1 (ODE Integration)\*; JIT Probability Ch. 1-2 (Sampling, Distributions)\*; JIT Astro Ch. 2 (Gravitational Dynamics)\* |
| 4 | Sept 19 | Linear Algebra & <br> Radiative Transfer Fundamentals | Phase 1 | JIT Math Ch. 2-3 (Linear Algebra, Vector Calculus)\*; JIT Astro Ch. 3 (Radiative Transfer)\* |
| 5 | Sept 26 | Monte Carlo Radiative Transfer | Phase 1 | JIT Numerical Methods Ch. 3 (MC Integration)\*; JIT Astro Ch. 3 (cont.)\* |
| 6 | Oct 3 | Monte Carlo Radiative Transfer | Phase 1 | JIT Probability Ch. 3 (Statistical Testing & Convergence)\* |
| 7 | Oct 10 | Bayesian Foundations | Phase 1 | JIT Bayesian Ch. 1 (Bayesian Framework)* |
| 8 | Oct 17 | Bayesian Inference & <br> MCMC Implementation | Phase 2 | JIT Bayesian Ch. 2-4 (Why MCMC?, Practical MCMC, HMC)\*; JIT Astro Ch. 4 (Cosmology)\* |
| 9 | Oct 24 | MCMC Implementation | Phase 2 | JIT Bayesian Ch. 5 (Putting It All Together)\* |
| 10 | Oct 31 | Advanced Optimization | Phase 2 | JIT Numerical Methods Ch. 2 (Optimization)\* |
| 11 | Nov 7 | Gaussian Processes | Phase 2 | JIT ML Ch. 1 (GP Fundamentals)\*; JIT Probability Ch. 4 (Multivariate Statistics)\* |
| 12 | Nov 14 | Gaussian Processes & Intro to JAX | Phase 2 | JIT ML Ch. 2 (GP Implementation)*; JIT ML Ch. 5 (JAX for Scientific Computing)*; JIT Math Ch. 4 (Multivariate Calculus)\*; JIT Numerical Methods Ch. 4 (Linear Solvers)\* |
| 13 | Nov 21 | Neural Networks & JAX | Phase 2 | JIT ML Ch. 3-5 (Neural Networks, Automatic Differentiation, JAX for Scientific Computing)\* |
| 14 | Nov 28 | **THANKSGIVING BREAK** (Optional lab session earlier in week if students want) | Phase 3 | — |
| 15 | Dec 5 | Neural Networks & JAX | Phase 3 | JIT ML Ch. 3-5 (Neural Networks, Automatic Differentiation, JAX for Scientific Computing)\* |
| Finals | Dec 18 | Final Presentations (Time TBD) | Phase 3 | — |

*Readings marked with asterisk are subject to change as course materials are developed

## Important Dates (Fall 2025)

:::{list-table} Project Deadlines (all 11:59 PM PT)
:header-rows: 1
:widths: 50 50

* - Assignment
  - Due Date
* - Project 1: Stellar Populations + Growth Memo 1
  - Monday, Sept 8
* - Project 2: N-Body Dynamics + Growth Memo 2
  - Monday, Sept 22
* - Project 3: MCRT + Growth Memo 3
  - Monday, Oct 13 → **Phase 1 ends**
* - **Begin Phase 2**
  - Tuesday, Oct 14
* - Project 4: Bayesian/MCMC + Growth Memo 4
  - Monday, Nov 3
* - Project 5: GP + JAX N-body + Growth Memo 5
  - Monday, Nov 24 → **Phase 2 ends**
* - **Begin Phase 3**
  - Tuesday, Nov 25 (after Project 5 submission)
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