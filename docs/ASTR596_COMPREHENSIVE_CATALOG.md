# ASTR 596: Modeling the Universe — Comprehensive Course Materials Catalog

**Institution:** San Diego State University (SDSU)  
**Course:** ASTR 596 - Modeling the Universe  
**Semester:** Fall 2025  
**Instructor:** Dr. Anna Rosen  
**Meeting:** Fridays 11:00 AM - 1:40 PM | PS 256  
**Repository:** GitHub (astrobytes-edu/astr596-modeling-universe)

---

## CATALOG OVERVIEW

This course is a comprehensive, carefully scaffolded computational astrophysics course built around six progressive projects that advance from deterministic simulation through machine learning. The materials total approximately **6,462 files** across integrated course modules, including **121 markdown/notebook files** in the main content areas.

**Course Arc:** Acts I-III
- **Act I:** Deterministic Simulation (Projects 1-3)
- **Act II:** Inverse Problems & Inference (Projects 4-5)
- **Act III:** Machine Learning (Final Project)

---

## DIRECTORY STRUCTURE & MODULE ORGANIZATION

### Root Level Content

```
/astr596-modeling-universe/
├── README.md (Quick start & tech setup)
├── index-home.md (Homepage with course overview)
├── myst.yml (MyST CLI configuration)
├── requirements.txt (Core dependencies)
├── requirements-jax.txt (Optional JAX ecosystem)
├── environment.yml (Conda environment)
│
├── 01-course-info/ (6 files)
├── 02-getting-started/ (4 files)
├── 03-scientific-computing-with-python/ (25 files)
├── 04-mathematical-foundations/ (3 files)
├── 05-the-statistical-universe/ (19 files)
├── 06-the-learnable-universe/ (2 files)
├── 07-numerical-methods/ (42 files)
├── 08-short-projects/ (17 files + 4 project subdirs)
├── 09-final-project/ (3 files)
├── 10-reference/ (Reference materials)
│
├── drafts/ (48 archived/development files)
├── papers-for-projects/ (Research papers for projects)
├── plotting-scripts/ (Visualization utilities)
└── node_modules/ (Build dependencies)
```

---

## MODULE BREAKDOWN & CONTENT MAPPING

### 1. COURSE INFO & ADMINISTRATION (01-course-info/)
**Purpose:** Syllabus, schedule, policies, course philosophy

**Files (6):**
- `01-astr596-syllabus-fall25.md` - Complete course syllabus with grading rubrics
- `02-astr596-schedule.md` - Detailed week-by-week schedule with reading assignments
- `03-astr596-ai-policy.md` - Three-phase AI scaffolding framework (23 KB)
- `04-astr596-course-overview.md` - Pedagogical philosophy & course design (30 KB)
- `05-astr596-learning-guide.md` - Student learning strategies
- `06-why-astr596-is-different.md` - Course philosophy & differentiators (17 KB)
- `growth-mindset.jpeg` - Visual resource

**Key Concepts Covered:**
- Course structure and philosophies
- "Glass box" modeling methodology
- Growth mindset pedagogy
- Scaffolding architecture
- AI usage guidelines (Phase 1, 2, 3)

---

### 2. GETTING STARTED & SETUP (02-getting-started/)
**Purpose:** Onboarding and environment configuration

**Files (4):**
- `index.md` - Setup guide index
- `01-cli-intro.md` - Command-line interface introduction (15 KB)
- `02-software-setup.md` - Python environment, Git, conda setup (15 KB)
- `03-git-intro.md` - Version control fundamentals (16 KB)
- `images/` - Setup screenshots and diagrams

**Topics:**
- Development environment setup
- Git version control workflow
- Python virtual environments
- Testing and code quality tools

---

### 3. SCIENTIFIC COMPUTING WITH PYTHON (03-scientific-computing-with-python/)
**Purpose:** Progressive Python literacy from fundamentals to modern frameworks

**Structure:** 5 submodules with hierarchical progression

#### Submodule 1: Python Fundamentals (01-python-fundamentals/)
**Files (8):**
- `01-python-environment.md` - Python setup and IDEs
- `02-python-calculator.md` - Basic arithmetic and operations
- `03-python-control-flow.md` - If/loops/logic structures
- `04-python-data-structures.md` - Lists, dicts, tuples, sets
- `05-python-functions-modules.md` - Functions, imports, namespaces
- `06-oop-fundamentals.md` - Classes, inheritance, polymorphism
- `draft-exercises.md` - Practice problems
- `index.md` - Module index

**Topics:** Python syntax, data structures, object-oriented programming

#### Submodule 2: Scientific Computing Core (02-scientific-computing-core/)
**Files (6):**
- `07-numpy-fundamentals.md` - Arrays, broadcasting, vectorization (key for performance)
- `08-matplotlib-fundamentals.md` - 2D/3D plotting
- `09-robust-computing.md` - Error handling, debugging, testing
- `10-python-advanced-oop-ORIG.md` - Advanced OOP patterns
- `11-python-pandas-fundamentals.md` - DataFrames and data manipulation
- `index.md`

**Topics:** Numerical computing, data visualization, code robustness

#### Submodule 3: Advanced Scientific Computing (03-advanced-scientific-computing/)
**Files (5):**
- `12-python-scipy.md` - Optimization, integration, interpolation
- `13-performance-optimization.md` - Profiling, numba, cython
- `14-parallel-computing-v1.md` - Multiprocessing, GPU basics
- `15-python-sympy.md` - Symbolic mathematics
- `16-python-scikit-learn.md` - Machine learning fundamentals

**Topics:** Advanced numerical methods, performance optimization

#### Submodule 4: Parallel & High-Performance Computing (04-parallel-and-high-performance-computing/)
**Note:** Minimal content currently

#### Submodule 5: Modern Approaches - JAX Ecosystem (05-modern-approaches-jax-ecosystem/)
**Files (6):**
- `01-jax_fundamentals_chapter.md` - JAX basics (functional programming, JIT)
- `02-jax_scientific_stack.md` - JAX with Diffrax, Equinox, Lineax
- `03-jax_deep_learning_stack.md` - JAX for neural networks
- `04-jax_specialized_libraries.md` - Specialized JAX packages
- `05-jax_advanced_chapter.md` - Advanced JAX topics
- `05-jax_advanced_continued.md` - Continued advanced topics

**Topics:** Automatic differentiation, JIT compilation, functional programming, modern frameworks

**Total Files:** 25 markdown documents + supporting materials

---

### 4. MATHEMATICAL FOUNDATIONS (04-mathematical-foundations/)
**Purpose:** Mathematical prerequisites and tools

**Files (3):**
- `00a-linear-algebra-core-module.md` - Vectors, matrices, decompositions (86 KB)
- `00b-linear-algebra-stats-module.md` - Linear algebra for statistics (73 KB)
- `00c-linear-algebra-formula-sheet.md` - Quick reference (8 KB)
- `images/` - Diagrams and visualizations

**Topics:** Linear algebra, eigenvectors, matrix operations, Bayesian probability

---

### 5. THE STATISTICAL UNIVERSE (05-the-statistical-universe/)
**Purpose:** Core statistical and astrophysical physics modules

**Structure:** 4 major modules, each with 5 parts (overview + 4 content parts + synthesis)

#### Module 1: How Nature Computes (module-1-how-nature-computes/)
**Files (6):**
- `00-mod1-part0-overview.md`
- `01-mod1-part1-foundations.md` - Probability distributions, randomness
- `02-mod1-part2-statistical-tools.md` - Hypothesis testing, estimators
- `03-mod1-part3-moments.md` - Mean, variance, covariance, correlation
- `04-mod1-part4-sampling.md` - Rejection sampling, importance sampling
- `05-mod1-part5-synthesis.md` - Integration and application

**Topics:** Probability theory, statistics, sampling methods

#### Module 2: From Particles to Stars (module-2-from-particles-to-stars/)
**Files (9 current + multiple OLD versions):**
- `00-mod2-part0-overview.md`
- `01-mod2-part1-scale.md` - Length/time/energy scales in astrophysics
- `02-mod2-part2-boltzmann.md` - Statistical mechanics, thermal physics
- `03-mod2-part3-stellar.md` - Stellar structure and evolution
- `04-mod2-part4-synthesis.md` - Integration synthesis
- `OLD/` - Multiple versions (v1, v2, v3) showing iterative development

**Topics:** Statistical mechanics, stellar physics, scaling laws

#### Module 3: When Stars Become Particles (module-3-when-stars-become-particles/)
**Files (9 current + multiple OLD versions):**
- `00-mod3-part0-overview.md`
- `01-mod3-part1-phase.md` - Phase space, ergodicity, Liouville theorem
- `02-mod3-part2-dynamics.md` - N-body dynamics, collisions
- `03-mod3-part3-virial.md` - Virial theorem, relaxation, stability
- `04-mod3-part4-synthesis.md` - Integration synthesis
- `OLD/` - Multiple versions showing development

**Topics:** Dynamics, phase space, gravitational systems

#### Module 4: From Photons to Information (module-4-from-photons-to-information/)
**Files (8 with OLD versions):**
- `01-mod4-part1-light.md` - Radiative transfer, stellar spectra
- `02-mod4-part2-rte.md` - Radiative transfer equation
- `03-mod4-part3-mcrt.md` - Monte Carlo radiative transfer
- `OLD_versions/` - Multiple development versions

**Topics:** Radiative transfer, photon transport, spectroscopy

**Pedagogical Pattern:** Each module follows a consistent structure (overview, 4 content parts, synthesis) enabling students to understand how concepts build systematically.

**Total Files:** 19 markdown files + extensive legacy versions

---

### 6. THE LEARNABLE UNIVERSE (06-the-learnable-universe/)
**Purpose:** Advanced topics in statistical inference and machine learning (Active Development)

**Current State:** Actively growing; 2 files currently, expanding with new modules

#### Module 1: Statistical Inference (module-1-statistical-inference/)
**Substructure:** 01-bayesian-statistics-inference/

**Files (6):**
- `01-mod5-the-fundamental-problem.md` - Bayesian framework foundations
- `02-mod5-inferential-thinking.md` - How to think about inverse problems
- `03-mod5-part3-MCMC.md` - Markov Chain Monte Carlo methods
- `04-mod5-part4-advanced-mcmc.md` - Advanced MCMC (HMC, etc.) with comprehensive comparison table
- `figures/draft-04-mod5-part4-HMC.md` - Draft HMC materials
- `figures/draft_README_HMC_FIGURES.md` - Figure generation notes

**Topics:** Bayesian inference, MCMC, HMC, parameter estimation

#### Module 2: Computing the Universe (module-2-computing-the-universe/) **[SPECIAL FOCUS]**
**Current Files (2, expanding):**
- `outline-00-computing-universe-REVISED.md` (48 KB) - Complete module outline
  - **Purpose:** Bridge from statistical inference to machine learning
  - **Timeline:** 2.5 weeks (5-6 lecture hours)
  - **Content Structure:**
    - Part 1: Why JAX Exists (1 week)
    - Parts 2-4: (To be implemented)
  - **Learning Outcomes:** 10 specific JAX-related competencies
  - **Key Concept:** Transform from script writers to scientific software engineers
  
- `part-01-why-jax.md` (116 KB) - Comprehensive JAX introduction
  - Motivation: From numerical gradients (Project 4) to automatic differentiation
  - Pure functional programming in JAX
  - Core transformations: `jit`, `grad`, `vmap`, `pmap`
  - JAX PRNG system
  - Performance optimization and profiling

**Planned Extensions:**
- Part 2: JAX Scientific Stack (Diffrax, Equinox, Optax)
- Part 3: Building Differentiable Simulators
- Part 4: Performance & Production Code

#### Module 2b: Machine Learning (module-2-machine-learning/)
**Current State:** Minimal (placeholder for future expansion)

**Total Files:** 8 markdown files

**Strategic Role:** Serves as bridge between statistical inference (Modules 1-5) and final project (neural networks + JAX). Emphasizes practical software engineering and modern computational paradigms.

---

### 7. NUMERICAL METHODS (07-numerical-methods/)
**Purpose:** Computational techniques for solving mathematical problems

**Structure:** 3 main modules + extensive drafts showing iterative development

#### Module 1: Foundations of Discrete Computing (module-1-foundations-of-discrete-computing/)
**Files (8):**
- `00-mod1-overview.md` - Module objectives
- `01-mod1-part1-finite-differences.md` - Approximating derivatives numerically
- `02-mod1-part2-numerical-errors.md` - Error analysis, stability
- `03-mod1-part3-taylor-series.md` - Taylor series and error bounds
- `04-mod1-synthesis.md` - Integration and synthesis
- `draft-*.md` - Multiple development versions

**Topics:** Finite differences, error propagation, Taylor series, discretization

#### Module 2: Static Problems and Quadrature (module-2-static-problems-and-quadrature/)
**Files (7):**
- `00-mod2-overview.md`
- `01-mod2-part1-root-finding.md` - Newton's method, bisection, etc.
- `02-mod2-part2-quadrature.md` - Numerical integration (trapezoidal, Simpson's, Gaussian)
- `03-mod2-synthesis.md`
- `draft-*.md`

**Topics:** Root finding, numerical integration, quadrature rules

#### Module 3: ODE Methods and Conservation (module-3-ODE-methods-and-conservation/)
**Files (10):**
- `00-mod3-overview.md`
- `01-module3-part1-euler.md` - Euler, improved Euler methods
- `02-module3-part2-runge-kutta.md` - RK2, RK4 family
- `03-module3-part3-symplectic.md` - **Leapfrog integration, energy conservation**
- `04-module3-part4-stability.md` - Stability analysis, CFL conditions
- `05-mod3-synthesis.md`
- `draft-*.md` + `draft-module3-part5-performance.md`

**Topics:** ODE integration, stability, symplectic methods (critical for Projects 2 & 4)

#### Drafts Directory
**Files (21):** Multiple versions showing module development:
- Submodule versions (v0, v1, v2, v3)
- Alternative module structures
- Comprehensive review documents
- Synthesis materials

**Total Files:** 42 markdown files + 21 draft versions = extensive reference library

---

### 8. SHORT PROJECTS (08-short-projects/)
**Purpose:** Six progressive projects (4 completed, 2 planned)

#### Project Infrastructure (Root level files)
**Files (8):**
- `index.md` - Projects overview
- `00-pair-programming-guidelines.md` - Collaboration framework
- `01-project-submission-guide.md` - Submission and grading procedures
- `02-growth-memo-student-guide.md` - Reflective learning guide
- `03-growth-memo-template.md` - Template for growth memos
- `04-astr596-markdown-cheatsheet.md` - Formatting reference (24 KB)
- `instructor-project-notes.md` - Instructor guidance

#### Project 1: Stellar Populations (project-1/)
**Duration:** 1.5 weeks | **Due:** Sept 10

**Files:**
- `astr596-project1.md` - Full project description
- `astr596-project-1-starter-code.md` - Starter code template

**Overview:** Build Star and StellarPopulation classes
- Implement ZAMS (Zero-Age Main Sequence) functions for luminosity and radius
- Implement Star class with 4 physics methods (temperature, mass, density, age)
- Implement StellarPopulation class with dual initialization (grid or sampling)
- Demonstrate vectorization (5-100× speedup)
- Create 5 required visualizations

**Scaffolding:** Foundation for all subsequent projects

**AI Phase:** Phase 1 (Limited - 30-min rule)

#### Project 2: N-Body Star Clusters (project-2/)
**Duration:** 2 weeks | **Due:** Sept 24

**Files:**
- `project2-description-nbody.md` - Full description
- `project2-science-background.md` - Astrophysical context

**Overview:** Simulate gravitational dynamics
- Generate initial conditions using Project 1's StellarPopulation
- Implement leapfrog integration (symplectic, energy-conserving)
- Sample initial conditions (IMF, positions, velocities)
- Track conservation laws and cluster evolution
- Visualize dynamics and energy conservation

**Scaffolding:** Reuses Project 1 code; builds leapfrog integrator used later in MCMC/HMC

**AI Phase:** Phase 1

#### Project 3: Monte Carlo Radiative Transfer (project-3/)
**Duration:** 3 weeks | **Due:** Oct 15

**Files:**
- `project3_description_mcrt.md` - Full description
- `project3_starter_code.md` - Starter template
- `kext_albedo_*.txt` - Dust extinction data files (2 datasets)

**Overview:** Photon transport and radiation
- Implement Monte Carlo radiative transfer algorithm
- Sample photon paths through astrophysical media
- Handle absorption, scattering, thermal radiation
- Compute observables (spectra, images, polarization)
- Applications: dust scattering, stellar atmospheres

**Scaffolding:** Introduces stochastic methods; foundation for Bayesian inference (Project 4)

**AI Phase:** Phase 1

#### Project 4: Bayesian Cosmological Inference (project-4/)
**Duration:** 3 weeks | **Due:** Nov 5

**Files:**
- `project4-description-FINAL.md` - Comprehensive description (100+ KB)
- `project4-scientific-background-FINAL.md` - Scientific context
- `OLD_project4-description.md` - Previous version
- `OLD_project4-scientific-background.md` - Previous context
- `jla_mub.txt` - Supernova data (Pantheon/JLA catalog)
- `jla_mub_covmatrix.txt` - Covariance matrix for data

**Overview:** Parameter estimation using Bayesian inference
- Implement forward model: luminosity distance for cosmology
- Implement likelihood function for Type Ia supernova data
- Build Metropolis-Hastings sampler (learns basic MCMC)
- Implement Hamiltonian Monte Carlo (uses leapfrog from Project 2!)
- Infer (Ωm, h) parameters from real supernova data
- Compute diagnostics: convergence, autocorrelation, Gelman-Rubin

**Scientific Context:** Replicates 2011 Nobel Prize discovery of cosmic acceleration (dark energy)

**Scaffolding:** 
- Reuses physics understanding from Projects 1-3
- Uses leapfrog integrator from Project 2
- Manual gradient computation (finite differences) motivates JAX in Module 6
- Data in format compatible with Project 5 emulation

**AI Phase:** Phase 2 (Strategic use after baseline implementation)

#### Projects 5 & 6 (Planned/In Development)
**Status:** Project 5 and Final Project frameworks being developed

**Project 5:** Gaussian Process Emulation
- Duration: 3 weeks | Due: Nov 26
- Concepts: Surrogate modeling, kernel methods, uncertainty quantification
- Reuses N-body data from Project 2 and inference machinery from Project 4

**Final Project:** Neural Networks with JAX
- Duration: 4.5 weeks | Due: Dec 18
- Concepts: Deep learning, automatic differentiation, GPU acceleration
- Integrates all previous projects into comprehensive machine learning pipeline

**Total Project Files:** 17 markdown files + data files

---

### 9. FINAL PROJECT (09-final-project/)
**Purpose:** Capstone project (planned/in development)

**Files (3):**
- `index.md` - Final project overview
- `draft-01-final-project-guide.md` - Project framework (14 KB)
- `draft-astr596-final-project-outline.md` - Detailed outline (28 KB)

**Planned Scope:**
- Neural networks with JAX
- Physics-informed machine learning
- Integration of all course concepts
- Individual or group research projects

---

### 10. REFERENCE MATERIALS (10-reference/)
**Purpose:** Supplementary guides and resources

**Files (5):**
- `index.md` - Reference index
- `04-cli-advanced-guide.md` - Advanced command-line usage
- `classroom-workflow.md` - In-class collaboration procedures
- `external-resources/index.md` - Links to textbooks, tutorials, etc.
- `quick-references/index.md` - Cheat sheets and quick lookups
- `troubleshooting/index.md` - Common issues and solutions

---

## KEY PATTERNS IN COURSE ORGANIZATION

### 1. **Scaffolding Architecture**
Each project builds on previous work:
```
Project 1 (Star/StellarPopulation classes)
    ↓ [Reuse stellar populations]
Project 2 (N-Body + Leapfrog integrator)
    ↓ [Reuse stellar physics]
Project 3 (Monte Carlo methods)
    ↓ [Learn stochastic approaches]
Project 4 (MCMC/HMC using leapfrog!)
    ↓ [Manual gradients motivate automation]
Module 6 (JAX + automatic differentiation)
    ↓ [Refactor with JAX]
Project 5 (GP Emulation with JAX)
    ↓ [All optimization tools]
Final (Neural Networks with JAX)
```

### 2. **Modular Content Structure**
Each major pedagogical module follows consistent pattern:
- Part 0: Overview & motivation
- Parts 1-4: Content progression
- Synthesis: Integration and application
- Shows deliberate instructional design

### 3. **Iterative Development**
Multiple versions preserved in:
- `OLD/` directories (statistical universe modules)
- `drafts/` subdirectories (numerical methods)
- Show evolution of course materials and instructor refinement

### 4. **Theory-Practice Balance**
- Mathematical foundations (Module 4) support applications
- Statistical theory (Module 5) informs projects
- Numerical methods (Module 7) underpin implementations
- JAX (Module 6) bridges to modern computational paradigms

### 5. **Data-Driven Pedagogy**
- Real datasets included (supernovae, dust properties)
- Scientific authenticity throughout
- Nobel Prize-winning science (dark energy discovery)

---

## PROJECT TIMELINE & PROGRESSION

| Phase | Projects | Duration | Concepts | Computational Paradigm |
|-------|----------|----------|----------|------------------------|
| **Act I: Building Universes** | 1-3 | 6.5 weeks | OOP, vectorization, numerical methods, stochastic methods | Deterministic simulation + Monte Carlo |
| **Act II: Observing Universes** | 4-5 | 6 weeks | Bayesian inference, MCMC/HMC, ML foundations | Statistical inference + optimization |
| **Act III: Learning from Universes** | Final | 4.5 weeks | Deep learning, autodiff, GPU acceleration | Neural networks + JAX |

---

## SPECIALIZED CONTENT HIGHLIGHTS

### Module 6: Computing the Universe (Active Development)
- **Current Focus:** JAX foundations and automatic differentiation
- **Outline Size:** 48 KB (comprehensive framework)
- **Part 1 Content:** 116 KB detailed tutorial
- **Unique Feature:** Explicitly teaches transformation from script writing to scientific software engineering
- **Connection to Projects:** Directly motivated by Project 4's manual gradients; feeds into Project 5's JAX refactoring

### Mathematical Foundations
- **Linear Algebra Core:** 86 KB covering vectors, matrices, decompositions
- **Linear Algebra Stats:** 73 KB for probabilistic applications
- **Coverage:** Eigenvalues, SVD, covariance, Bayesian foundations

### Numerical Methods
- **42 files:** Extensive coverage of computational techniques
- **Leapfrog Integration:** Central to both Project 2 (N-body) and Project 4 (HMC)
- **Emphasis:** Symplectic methods and energy conservation
- **Development:** 21 draft versions showing iterative refinement

---

## COURSE PHILOSOPHY & SPECIAL FEATURES

### "Glass Box" Methodology
- Every algorithm implemented from first principles
- Black boxes prohibited until foundations understood
- Creates transferable, deep understanding

### Three-Phase AI Scaffolding
1. **Phase 1 (Projects 1-3):** Foundation building with limited AI (30-min rule)
2. **Phase 2 (Projects 4-5):** Strategic integration (after baseline works)
3. **Phase 3 (Final Project):** Professional practice (full productivity tools)

### Growth Mindset Pedagogy
- Productive struggle as learning design
- Iteration and refinement as normal
- Professional code rarely right first time
- Cognitive challenge optimized for learning

### Real Astrophysics
- All projects solve genuine scientific problems
- Uses published datasets and methods
- Nobel Prize-winning discoveries included
- Research-grade code and analysis

---

## ADMINISTRATIVE & SUPPORT MATERIALS

### Course Information
- Complete syllabus with grading rubrics
- Week-by-week schedule with reading assignments
- Detailed learning outcomes (10+ per module)
- AI policy framework (23 KB)

### Pedagogical Resources
- Growth mindset guidance
- Pair programming guidelines
- Reflection and learning strategies
- Markdown formatting guide

### Development Setup
- Software installation guides
- Git/GitHub workflow
- Python environment configuration
- Testing and quality assurance

---

## FILE STATISTICS

### By Module
- Course Info: 6 files
- Getting Started: 4 files
- Scientific Computing (Python): 25 files
- Mathematical Foundations: 3 files
- Statistical Universe: 19 files
- Learnable Universe: 8 files (expanding)
- Numerical Methods: 42 files (+ 21 drafts)
- Short Projects: 17 files
- Final Project: 3 files
- Reference: 5 files

**Total Documented:** 132 main files + 48 draft files = 180 content files
**Overall Repository:** 6,462 files (including build artifacts, node_modules, .git)

### Content Types
- Markdown (.md): Primary format, ~121 files
- Jupyter Notebooks (.ipynb): (to be cataloged)
- Python (.py): Starter code templates and examples
- Data files: CSV, TXT (project datasets)
- Configuration: YAML, JSON (MyST, environment, CI/CD)

---

## BUILD & DEPLOYMENT

### Technology Stack
- **Site Generator:** MyST (Markdown-first, Jupyter Book compatible)
- **Hosting:** GitHub Pages (automatic CI/CD via GitHub Actions)
- **Repository:** GitHub (astrobytes-edu/astr596-modeling-universe)
- **Development:** Local MyST CLI with hot reload
- **Format:** HTML (PDF/Typst export currently disabled)

### Key Configuration Files
- `myst.yml` - MyST build configuration
- `.github/workflows/deploy.yml` - CI/CD pipeline
- `requirements.txt` - Core Python dependencies
- `environment.yml` - Conda environment specification

---

## COURSE VISION & IMPACT

### Learning Outcomes
By course completion, students will have:
- Built every algorithm from scratch
- Discovered algorithm generality and transferability
- Developed computational intuition and debugging skills
- Created a portfolio of 6+ working projects
- Mastered modern JAX ecosystem tools
- Practiced professional research skills (Git, testing, documentation)

### Pedagogical Innovation
- **Scaffolding:** Each project builds on previous code (reduces cognitive load)
- **Integration:** No throwaway code (Portfolio effect)
- **Authenticity:** Real datasets, genuine scientific problems
- **Modern Tools:** Bridges NumPy → JAX ecosystem progression
- **Transparency:** All AI usage in course development documented

### Career Preparation
- Research-grade coding practices
- Portfolio demonstrating progression
- Modern computational astrophysics tools
- Scientific communication and documentation
- Professional software engineering standards

---

## NOTES ON COURSE DEVELOPMENT

### Iterative Refinement
- Multiple versions of modules preserved (v1, v2, v3)
- Extensive draft materials show thinking process
- Continuous improvement visible in file history

### AI Integration in Course Development
- Developed with Claude AI as thought partner
- All content rigorously fact-checked
- Pedagogical decisions informed by AI limitations
- Emphasizes verification and critical evaluation

### Active Development
- Module 6 (Computing the Universe) currently expanding
- Projects 5 and 6 frameworks in development
- Continuous community contribution (CONTRIBUTING.md)
- GitHub issues for tracking improvements

---

## ACCESSING & USING MATERIALS

### Online
- **Public Website:** https://astrobytes-edu.github.io/astr596-modeling-universe/
- **Repository:** https://github.com/astrobytes-edu/astr596-modeling-universe

### Local Development
```bash
# Clone repository
git clone https://github.com/astrobytes-edu/astr596-modeling-universe.git
cd astr596-modeling-universe

# Install dependencies
pip install -r requirements.txt
npm install -g mystmd

# Serve locally with auto-rebuild
myst start

# Build static HTML
myst build --html
```

### Customization & Contribution
- See `CONTRIBUTING.md` for branching and PR workflow
- Licensed: Content (CC-BY-4.0) | Code (MIT)
- Contact: Dr. Anna Rosen (alrosen@sdsu.edu)

---

## SUMMARY & HIGHLIGHTS

**ASTR 596: Modeling the Universe** is a carefully engineered computational astrophysics course featuring:

1. **Comprehensive Scaffolding:** 6 progressive projects with intentional code reuse
2. **Modern Pedagogy:** Growth mindset, productive struggle, authentic AI integration
3. **Cutting-Edge Tools:** NumPy → JAX ecosystem bridge
4. **Real Science:** Nobel Prize-winning discoveries and authentic datasets
5. **Professional Standards:** Testing, documentation, version control, software engineering
6. **Extensive Materials:** 130+ content files covering 10 major topics
7. **Active Development:** Continuously refined with visible iteration history

**Unique Position:** Bridges fundamental computational science (numerical methods, statistics) with modern machine learning, preparing students for contemporary computational astrophysics research.

