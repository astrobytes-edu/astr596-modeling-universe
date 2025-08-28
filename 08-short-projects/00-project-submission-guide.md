---
title: "Project Submission Guide"
subtitle: "ASTR 596: Modeling the Universe"
exports:
    - format: pdf
---

## Course Philosophy on Collaboration

This course thrives on collaboration! You are **strongly encouraged** to:

- Discuss ideas and concepts with classmates
- Help each other debug and troubleshoot
- Share insights and "aha!" moments
- Work through challenging concepts together
- Learn from each other's approaches

**Important:** While collaboration is encouraged, all submitted code, memos, and documentation must be your own work. Think of it like studying together for an exam—you can discuss and learn together, but when it's time to write, you do so independently.

**This is not a competition!** We're all here to learn, grow, and support each other in mastering computational astrophysics.

## Project Schedule & Deadlines

Projects are assigned on Mondays (posted to GitHub Classroom) with varying completion periods based on complexity. This schedule allows you to review requirements before Friday's class, where we'll work on implementation together.

| Project | Assigned | Due Date | Duration | Topic | Key Concepts |
|---------|----------|----------|----------|-------|--------------|
| **Project 1** | Aug 29 (Fri) | Sept 8 (Mon) | ~1.5 weeks | Python/OOP/Stellar Physics Basics | OOP & Classes, HR diagrams |
| **Project 2** | Sept 8 (Mon) | Sept 22 (Mon) | 2 weeks | ODE Integration & N-Body Dynamics | Euler, RK4, Leapfrog, Planetary + Star Cluster Dynamics, IMF Sampling |
| **Project 3** | Sept 22 (Mon) | Oct 13 (Mon) | 3 weeks | Monte Carlo Radiative Transfer (MCRT) | Photon packets, scattering, absorption |
| **Project 4** | Oct 13 (Mon) | Nov 3 (Mon) | 3 weeks | Bayesian Inference/MCMC | Priors, Likelihood, Metropolis-Hastings, Gradient descent |
| **Project 5** | Nov 3 (Mon) | Nov 24 (Mon) | 3 weeks | Gaussian Processes | Kernels, Hyperparameters, Regression |
| **Final Project** | Nov 17 (Mon) | Dec 18 (Thu) | 4.5 weeks | Neural Networks (From Scratch + JAX) | Backprop, autodiff, JAX Ecosystem |

### General Project Timeline

**2-week projects:** Foundation (days 1-5) → Implementation & Testing (days 6-10) → Polish & Submit (days 11-14)

**3-week projects:** Theory & Planning (week 1) → Core Implementation (week 2) → Extensions & Polish (week 3)

**Pro tip:** Start early, commit often, and use Friday lab sessions for debugging with peers!

## Project Extensions: Your Creative Playground

### Extension Requirements

**Graduate Students:** Must complete at least one substantial extension beyond base requirements.

**Undergraduate Students:** Extensions are optional but **highly recommended**—they're where the real fun and deeper learning happen!

### The Spirit of Extensions

Extensions are YOUR opportunity to explore what interests you most. The goal is to promote curiosity and experimentation. This is about what YOU want to explore, not what you think I want to see!

### Extension Ideas

**Scientific Investigation:**
- Compare with analytical solutions
- Error analysis and convergence studies
- Connection to real astronomical observations

**Computational Exploration:**
- Implement alternative algorithms and compare
- Performance optimization and profiling
- Adaptive methods (timestep, resolution)

**Creative Visualization:**
- Animations showing time evolution
- Interactive plots for parameter exploration
- Novel ways to display multi-dimensional data

**Physics Extensions:**
- Add additional physical processes
- Extend to more realistic scenarios
- Explore extreme parameter regimes

Not sure about your extension idea? Talk to me or classmates! The best extensions come from genuine curiosity.

## Submission Requirements

### GitHub Classroom

Accept the assignment link → Clone your repo → Work locally → Push regularly → Submit by deadline

For detailed GitHub Classroom instructions, see the [Getting Started Guide](../02-getting-started/03-git-intro.md).

### Required Project Structure

```bash
project_N/
├── src/                  # Your source code (modular design)
├── tests/                # Basic tests for key functions
├── outputs/
│   ├── figures/          # All generated plots
│   └── data/             # Output data files
├── notes/                # Optional: ongoing project notes
├── README.md             # Installation, usage, results summary
├── requirements.txt      # Dependencies with versions
├── research_memo.md      # Analysis (2-3 pages)
├── growth_memo.md        # Reflection (1-2 pages)
└── .gitignore            # Use provided template
```

### 1. Code Requirements

**Standards:**
- Modular design with clear separation of concerns
- No magic numbers (use named constants)
- Meaningful variable names
- Error handling for edge cases
- Proper function documentation (see docstring example below)

**Required Docstring Format (NumPy style recommended):**
```python
def integrate_orbit(initial_conditions, time_span, method='RK4', dt=0.01):
    """
    Integrate orbital dynamics using specified numerical method.
    
    Parameters
    ----------
    initial_conditions : np.ndarray
        Shape (6,) array of [x, y, z, vx, vy, vz]
    time_span : tuple
        (t_start, t_end) for integration
    method : str, optional
        Integration method: 'Euler', 'RK4', or 'Leapfrog'
    dt : float, optional
        Time step size
    
    Returns
    -------
    trajectory : np.ndarray
        Shape (n_steps, 6) array of positions and velocities
    times : np.ndarray
        Shape (n_steps,) array of time points
    
    Raises
    ------
    ValueError
        If method is not recognized or dt <= 0
    
    Examples
    --------
    >>> ic = np.array([1, 0, 0, 0, 1, 0])
    >>> traj, t = integrate_orbit(ic, (0, 10))
    """
    # Implementation here
```

### 2. Research Memo (Markdown or PDF, 2-3 pages of text)

Required sections (2-3 pages of text, not counting figures/references):
- **Executive Summary** - What you did and key findings
- **Methodology** - Approach, algorithms, numerical methods
- **Results** - Key findings with figures, quantitative analysis (figures required but don't count toward page limit)
- **Validation** - How you verified correctness
- **Extensions** - What you explored beyond requirements
- **Conclusions** - What you learned about physics and methods
- **References** - If applicable (doesn't count toward page limit)

Submit as `research_memo.md` or `research_memo.pdf` in your project repository.

### 3. Growth Memo (Markdown, 1-2 pages)

Informal reflection about your learning journey. See the **Growth Memo Guide** for detailed instructions and use the provided **growth_memo_template.md**.

Key elements to address:
- Technical skills developed
- Challenges and solutions
- Conceptual insights
- What excited or surprised you
- AI usage reflection (per phase guidelines)
- What you'd tell your past self

**Pro tip:** Keep notes throughout (`notes/notes.md`) for authentic reflection!

### 4. README Requirements

Must include:
- Project description
- Installation instructions
- Usage examples
- Key results summary
- Acknowledgment of collaborators

### 5. Git Practices

- Minimum 5-10 meaningful commits
- Descriptive commit messages
- Regular pushes to GitHub

## Grading Components

| Component | Focus |
|-----------|-------|
| **Core Implementation** | Correctness, completeness |
| **Extensions** | Required for grads, valued for undergrads |
| **Code Quality** | Structure, documentation, Git practices |
| **Research Memo** | Analysis quality, scientific writing |
| **Growth Memo** | Reflection depth, learning insights |

**Note:** Weight varies by project to align with learning objectives.

## Submission Checklist

Before submitting:

- [ ] Code runs on clean environment
- [ ] All required files present and named correctly
- [ ] README has clear instructions
- [ ] 5+ meaningful Git commits
- [ ] Research memo complete (2-3 pages)
- [ ] Growth memo complete (1-2 pages)
- [ ] Plots saved in outputs/figures/
- [ ] Extensions documented (if applicable)
- [ ] AI usage documented per phase
- [ ] Collaborators acknowledged
- [ ] Pushed before Monday 11:59 PM

## Late Policy

- **One free extension:** Request ≥24h before deadline → 2-day grace
- **Late penalty:** 10% per day, max 3 days
- **After 3 days:** Not accepted without documented emergency and advance notice

## Getting Help

**When to seek help:**
- After 30 minutes genuine effort on a bug
- Confused about physics/math after reading
- Unsure about requirements
- Want to discuss extension ideas

**How to ask effectively:**
1. What you're trying to do
2. What you've tried (code/errors)
3. Expected vs. actual behavior
4. Minimal reproducible example

**Resources:**
- **Classmates** - Collaborate and learn together!
- **Friday Labs** - Pair programming and debugging
- **Student Hacking Hours** - Wednesdays 1-2 PM
- **Course Slack** - Quick questions and peer support

**Remember:** Struggling is learning, but struggling alone too long is inefficient. This is a collaborative environment—use it!