# ASTR 596 Course Materials - Quick Reference Guide

## Key Statistics
- **Total Files:** 6,462 (entire repository)
- **Main Content Files:** 121 markdown/notebook files
- **Modules:** 10 major topic areas
- **Projects:** 6 (4 completed, 2 in development)
- **Repository:** GitHub - astrobytes-edu/astr596-modeling-universe

## Module Overview at a Glance

| # | Module | Files | Focus | Key Topics |
|---|--------|-------|-------|-----------|
| 01 | Course Info | 6 | Syllabus, policies, pedagogy | Glass box, growth mindset, AI scaffolding |
| 02 | Getting Started | 4 | Setup & onboarding | Git, Python, environment |
| 03 | Scientific Computing | 25 | Python fundamentals to JAX | NumPy, matplotlib, JAX ecosystem |
| 04 | Math Foundations | 3 | Prerequisites | Linear algebra, probability |
| 05 | Statistical Universe | 19 | Core astrophysical physics | Stats, statistical mechanics, dynamics, radiative transfer |
| 06 | Learnable Universe | 8 | Inference & ML (active dev) | Bayesian statistics, MCMC, JAX/automatic differentiation |
| 07 | Numerical Methods | 42 | Computational techniques | Finite differences, ODE integration, symplectic methods |
| 08 | Short Projects | 17 | Six progressive projects | Stellar pop., N-body, MCRT, Bayesian inference |
| 09 | Final Project | 3 | Capstone (in development) | Neural networks with JAX |
| 10 | Reference | 5 | Support materials | CLI guides, workflows, troubleshooting |

## Project Progression (The Three Acts)

### Act I: Building Universes (Weeks 1-6.5)
- **Project 1 (1.5w):** Stellar Populations - OOP & vectorization
- **Project 2 (2w):** N-Body Dynamics - Numerical integration & leapfrog
- **Project 3 (3w):** Monte Carlo Radiative Transfer - Stochastic methods

### Act II: Observing Universes (Weeks 8-13)
- **Project 4 (3w):** Bayesian Cosmological Inference - MCMC/HMC
- **Project 5 (3w):** Gaussian Process Emulation - JAX refactoring

### Act III: Learning from Universes (Weeks 14-18)
- **Final Project (4.5w):** Neural Networks with JAX - Deep learning

## Special Highlights

### Module 6: Computing the Universe (Most Recent/Active)
- **Status:** Under active development
- **Files:** 2 main files (48 KB outline + 116 KB tutorial)
- **Purpose:** Bridge from statistical inference to machine learning
- **Key Content:**
  - Part 1: Why JAX Exists (motivation from Project 4's manual gradients)
  - Parts 2-4: (Planned) JAX scientific stack, differentiable simulators, production code

### Scaffolding Architecture (Key Design Feature)
```
Project 1 → Star class
           ↓ (reused in)
Project 2 → N-Body simulator + Leapfrog integrator
           ↓ (integrator reused in)
Project 4 → HMC sampler (leapfrog in parameter space!)
           ↓ (manual gradients motivate)
Module 6 → JAX automatic differentiation
           ↓ (refactor into)
Project 5 → Production JAX code
           ↓ (foundation for)
Final Project → Neural networks + ML
```

## File Organization Patterns

### Each Major Content Module (05, 06, 07) Follows:
- **Part 0:** Overview & learning outcomes
- **Parts 1-4:** Progressive content buildup
- **Synthesis:** Integration & application
- **OLD/ or drafts/:** Multiple versions showing development

### Project Structure:
- Description file (pedagogical overview + scientific context)
- Starter code template
- Data files (where needed)
- Grading notes and rubrics

## Unique Pedagogical Features

1. **Glass Box Methodology:** Every algorithm from first principles
2. **Three-Phase AI Scaffolding:** Phase 1 (limited), Phase 2 (strategic), Phase 3 (productive)
3. **Growth Mindset:** Productive struggle as learning design
4. **Code Reuse:** No throwaway projects (portfolio effect)
5. **Authentic Science:** Real datasets, Nobel Prize discoveries included

## Key Figures

- **Python Fundamentals Module:** 8 files covering syntax → OOP
- **JAX Ecosystem Module:** 6 files from basics to advanced
- **Numerical Methods:** 42 files covering finite differences through ODE methods
- **Statistical Universe:** 19 files + legacy versions (extensive physics content)
- **Linear Algebra:** 86 KB core + 73 KB stats modules = comprehensive math foundation

## Critical Dependencies & Connections

```
Numerical Methods (Ch. 3) → ODE integration needed for Project 2
                 ↓
Leapfrog integrator → Used in Project 2 AND Project 4 (HMC)
                 ↓
Manual gradients (Project 4) → Motivates JAX (Module 6)
                 ↓
Statistical Universe (Modules 1-4) → All project physics foundations
                 ↓
Project data generation → Training material for final project ML
```

## Build & Access

**Website:** https://astrobytes-edu.github.io/astr596-modeling-universe/
**Repository:** https://github.com/astrobytes-edu/astr596-modeling-universe
**Build Tool:** MyST (Markdown-first, Jupyter Book compatible)
**Hosting:** GitHub Pages with automatic CI/CD

## For Review/Analysis

**Comprehensive Catalog:** `/ASTR596_COMPREHENSIVE_CATALOG.md` (721 lines)
- Complete breakdown of all modules
- All file listings with descriptions
- Learning outcomes
- Pedagogical patterns
- Development philosophy

**Quick Files to Review:**
1. `index-home.md` - Course philosophy & structure
2. `01-course-info/03-astr596-ai-policy.md` - AI scaffolding framework
3. `06-the-learnable-universe/module-2-computing-the-universe/outline-00-computing-universe-REVISED.md` - Module 6 framework
4. `08-short-projects/` - All project descriptions and guidelines

