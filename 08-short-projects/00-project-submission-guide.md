# ASTR 596: Project Submission Guide

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

### Project Timeline

Projects are assigned on Mondays (posted to GitHub Classroom) with varying completion periods based on complexity. This schedule allows you to review requirements before Friday's class, where we'll work on implementation together.

| Project | Assigned | Due Date | Duration | Topic | Key Concepts |
|---------|----------|----------|----------|-------|--------------|
| **Project 1** | Aug 29 (Fri) | Sept 8 (Mon) | ~1.5 weeks | Python/OOP/Stellar Physics Basics | OOP & Classes, HR diagrams |
| **Project 2** | Sept 8 (Mon) | Sept 22 (Mon) | 2 weeks | ODE Integration & N-Body Dynamics | Euler, RK4, Leapfrog, Planetary + Star Cluster Dynamics, IMF Sampling |
| **Project 3** | Sept 22 (Mon) | Oct 13 (Mon) | 3 weeks | Monte Carlo Radiative Transfer (MCRT) | Photon packets, scattering, absorption |
| **Project 4** | Oct 13 (Mon) | Nov 3 (Mon) | 3 weeks | Bayesian Inference/MCMC | Priors, Likelihood, Metropolis-Hastings, Gradient descent (integrated) |
| **Project 5** | Nov 3 (Mon) | Nov 24 (Mon) | 3 weeks | Gaussian Processes | Kernels, Hyperparameters, Regression |
| **Final Project** | Nov 24 (Mon) | Dec 18 (Thu) | 3.5 weeks | Neural Networks (From Scratch + `JAX`) | Backprop, `autodiff`, `JAX` Ecosystem |

### Project Workflows

**Note:** These timelines serve as example workflows. How you manage your time outside of class will likely vary week-to-week, but staying on track is crucial for producing quality work. If you're struggling with time management on your projects, please reach out early—I'm here to help you find strategies that work.

#### Two-Week Project Workflow (Project 2)

**Week 1: Understanding & Initial Implementation**

- **Day 1-2 (Mon-Tue):** Read assignment thoroughly, understand requirements, review relevant Course Jupyter Book chapters
- **Day 3-4 (Wed-Thu):** Begin implementation, focus on core functionality  
- **Day 5 (Fri):** Class session - ask questions, pair programming, debug with peers
- *Optional* **Day 6-7 (Sat-Sun):** Continue implementation based on class insights

**Week 2: Refinement & Completion**

- **Day 8-9 (Mon-Tue):** Complete base requirements, begin extensions
- **Day 10-11 (Wed-Thu):** Test edge cases, optimize performance
- **Day 12 (Fri):** Class session - final debugging, optimization discussions
- *Optional* **Day 13-14 (Sat-Mon):** Polish code, write documentation, complete memos, submit by Monday 11:59 PM

#### Three-Week Project Workflow (Projects 3-5)

**Week 1: Foundation & Exploration**

- **Day 1-3 (Mon-Wed):** Deep dive into theory, understand physics/math, plan approach
- **Day 4 (Thu):** Begin initial implementation of core algorithms
- **Day 5 (Fri):** Class session - clarify concepts, discuss approaches with peers
- *Optional* **Day 6-7 (Sat-Sun):** Continue core implementation

**Week 2: Core Implementation & Testing**

- **Day 8-10 (Mon-Wed):** Complete base implementation, begin testing
- **Day 11 (Thu):** Debug, validate against known solutions
- **Day 12 (Fri):** Class session - share progress, get feedback, debug challenging issues
- *Optional* **Day 13-14 (Sat-Sun):** Refine implementation, start extensions

**Week 3: Extensions & Polish**

- **Day 15-17 (Mon-Wed):** Complete extensions, optimize performance
- **Day 18 (Thu):** Write memos, finalize documentation
- **Day 19 (Fri):** Class session - final optimizations, peer code review
- *Optional* **Day 20-21 (Sat-Mon):** Final polish, submit by Monday 11:59 PM

## Project Extensions: Your Creative Playground

### Extension Requirements

**Graduate Students:** Must complete at least one substantial extension beyond base requirements.

**Undergraduate Students:** Extensions are optional but **highly recommended**—they're where the real fun and deeper learning happen!

### The Spirit of Extensions

Extensions are meant to be **freeing and engaging** — this is *YOUR* opportunity to explore what interests you most about the project. The goal is to promote:

- **Curiosity:** What happens if...?
- **Experimentation:** Let me try this approach...
- **Creativity:** I wonder if I could visualize this differently...
- **Deep Diving:** I want to understand this aspect better...

**This is about what YOU want to explore, not what you think I want to see!**

### Example Extension Ideas (Jump-Off Points)

**Parameter Exploration:**

- How does the system behave under extreme conditions?
- What parameter ranges produce interesting phenomena?
- Can you identify phase transitions or critical points?

**Additional Methods:**

- Implement an alternative algorithm and compare
- Try a different numerical scheme
- Add adaptive methods (timestep, resolution, etc.)

**Enhanced Visualizations:**

- Create animations showing time evolution
- Interactive plots for parameter exploration
- 3D visualizations where appropriate
- Novel ways to display multi-dimensional data

**Scientific Investigation:**

- Compare with analytical solutions
- Error analysis and convergence studies
- Stability analysis
- Connection to real astronomical observations

**Performance Optimization:**

- Vectorization and algorithmic improvements
- Memory optimization for large-scale problems
- Parallel processing implementation
- Profiling and bottleneck analysis

**Physics Extensions:**

- Add additional physical processes
- Extend to more realistic scenarios
- Include effects initially neglected

### Not Sure About Your Extension Idea?

**Talk to me or your classmates!** Some of the best extensions come from conversations like:

- "I noticed something weird when..."
- "I'm curious what would happen if..."
- "Could we make this work for..."
- "What if we tried..."

**Remember:** The best extension is one that genuinely interests you. If you're excited about it, you're on the right track!

## Submission Requirements

### GitHub Classroom Submission

:::{important} GitHub Classroom Submission
:class: dropdown
Follow these steps to accept, submit, and verify assignments via GitHub Classroom.

1. Accept the assignment

    - Use the GitHub Classroom link posted in the LMS (or the assignment URL emailed to you). After accepting, you will receive a private repository named something like `astr596-project-<N>-<your-github-user>` under your GitHub account or the course organization.

2. Clone the repository to your machine

    - SSH (recommended):

      ```bash
      git clone git@github.com:<org>/<assignment-repo>.git
      cd <assignment-repo>
      ```

    - HTTPS alternative:

      ```bash
      git clone https://github.com/<org>/<assignment-repo>.git
      cd <assignment-repo>
      ```

3. Work locally and run checks

    - Create and activate a virtual environment, install dependencies, and run tests:

      ```bash
      python -m venv .venv
      source .venv/bin/activate       # macOS / Linux
      .venv\Scripts\activate          # Windows (PowerShell)
      pip install -r requirements.txt
      pytest -q                       # run available tests (if provided)
      ```

    - Recommended workflow: work on a feature branch and open PRs if you want instructor feedback before final submission:

      ```bash
      git checkout -b feature/my-solution
      # work, commit, run tests locally
      ```

4. Finalize and push your submission

    - Add required files, commit with a clear message, and push to the repository's `main` branch (or the branch specified by the assignment):

      ```bash
      git add -A
      git commit -m "Complete project: core requirements"
      git push origin main
      # or push your feature branch:
      git push -u origin feature/my-solution
      ```

5. Verify submission (important)

    - Classroom UI: Open the GitHub Classroom assignment page and confirm your repository appears and that the latest commit timestamp is correct. Many Classroom assignments show a "Submitted" indicator or list the latest commit.
    - GitHub: Check your repository in the browser for the final commit and for files required by the rubric (`research_memo.pdf`, `growth_memo.md`, `README.md`, `requirements.txt`, and `outputs/` contents).
    - CI: If the assignment includes Actions, confirm the GitHub Actions checks pass (look under the repository's "Actions" tab or on the commit status). Passing CI is often required for full credit.

6. Resubmitting or fixing issues

    - To resubmit, push additional commits to the same repository/branch; Classroom will pick up the latest commit as your submission.
    - If you accidentally committed large files, remove them from history (or contact the instructor/TA for help). Example to remove a file from the latest commit:

      ```bash
      git rm --cached large_file.dat
      git commit -m "Remove large binary"
      git push
      ```

7. Quick verification checklist

    - [ ] I accepted the assignment in GitHub Classroom
    - [ ] I cloned the correct assignment repository
    - [ ] Virtual environment created and `requirements.txt` installed
    - [ ] Tests (if provided) pass locally (`pytest -q`)
    - [ ] `research_memo.pdf`, `growth_memo.md`, `README.md`, and required code are present
    - [ ] Final commit pushed to the assignment repository/branch
    - [ ] Classroom UI shows the repository and final commit timestamp
    - [ ] CI (GitHub Actions) passed, if applicable

If anything is unclear, or if Classroom shows an unexpected status, see the *Getting Started: Introduction to Git and GitHub* chapter (../02-getting-started/03-git-intro.md) or contact the instructor via Slack/email.
:::

### Each Project Must Include

**Note:** The project code and research memo structures below are provided as starting points. Feel free to adapt or restructure them based on what makes sense for your project—developing your own organizational approach is part of the learning process.

#### 1. Code Components (recommended structure)

```bash
project_N/
├── src/
│   ├── __init__.py
│   ├── main.py           # Entry point with clear argument parsing
│   ├── physics.py        # Physics calculations
│   ├── numerics.py       # Numerical methods
│   ├── utils.py          # Helper functions
│   └── visualization.py  # Plotting functions
├── tests/
│   └── test_core.py      # At least basic tests
├── outputs/
│   ├── figures/          # All generated plots
│   └── data/             # Any output data files
├── README.md             # Installation and usage instructions
├── requirements.txt      # All dependencies with versions
├── research_memo.pdf     # Your analysis (2-3 pages)
├── growth_memo.md        # Your reflection (1-2 pages)
└── .gitignore            # Properly configured
```

**Code Standards:**

- Modular design with clear separation of concerns
- No God functions (functions should do one thing well)
- No magic numbers (use named constants and configuration variables)
- Meaningful variable names (avoid single letters except for indices)
- Type hints encouraged for function signatures
- No global variables unless absolutely necessary
- Error handling for edge cases

#### 2. Research Memo (PDF Format)

Your research memo should be **2-3 pages of text** (not counting figures/references) and include:

```markdown
# Project N: [Title] - Research Memo
Author: [Your Name]
Date: [Submission Date]

## Executive Summary
[1-2 paragraphs summarizing what you did and key findings]

## Methodology
[How you approached the problem, key algorithmic choices, numerical methods used]

## Results
[Key findings with embedded figures, quantitative analysis]
![Description](outputs/figures/plot1.png)

## Computational Performance
[Runtime analysis, bottlenecks identified, optimizations made]

## Validation
[How you verified correctness - comparison to analytical solutions, convergence tests, etc.]

## Extensions Implemented
[Description of extensions completed and their impact]

## Conclusions
[What you learned about the physics and computational methods]

## References
[Literature citations if applicable]
```

#### 3. Growth Memo (Markdown or PDF Format)

Your growth memo should be **1-2 pages** of informal reflection and include:

- Technical skills developed
- Challenges encountered and solutions found
- Connection to course concepts
- AI usage and verification process (following course phase guidelines)
- Next learning goals
- Any insights, surprises, or moments that shaped your understanding
- How collaboration with peers enhanced your learning

#### 4. Documentation Requirements

**README.md must include:**

```markdown
# Project N: [Descriptive Title]

## Description

[Brief description of what this project does]

## Installation

```bash
conda create -n proj_n python=3.10
conda activate proj_n
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py --input data.txt --output results.png
```

## Project Structure

[Brief description of each file's purpose]

## Key Results

[Summary of main findings]

## Acknowledgments

[List classmates you discussed ideas with, acknowledging the nature of collaboration]

## Dependencies
- Python 3.10+
- NumPy, SciPy, Matplotlib
- [Any other specific packages]
```

**Function Docstrings Example:**
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

#### 5. GitHub Best Practices

**Commit Practices:**

- Commit early and often (minimum 5-10 meaningful commits per project)
- Each commit should represent a logical unit of work
- Commit messages should be descriptive:
  - ✅ Good: "Add RK4 integration method with adaptive timestep"
  - ❌ Bad: "Update code" or "Fix stuff"

**.gitignore must include:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Project specific
outputs/figures/*.png
outputs/data/*.txt
*.log

# But track
!outputs/figures/.gitkeep
!outputs/data/.gitkeep
```

## Grading Rubric

### Project Grading Breakdown

Each project is worth 100 points, distributed across the following components:

| Component | Description |
|-----------|-------------|
| **Core Implementation** | Correctness, completeness, follows specifications |
| **Extensions** | Required for grad students, optional but valued for undergrads |
| **Code Quality** | Structure, readability, documentation, Git practices |
| **Research Memo** | Analysis quality, scientific writing, visualizations |
| **Growth Memo** | Reflection depth, learning insights, metacognitive development |

**Note:** The relative weight between components may vary by project to align with learning objectives and skill development progression. Early projects may emphasize foundational skills and reflection, while later projects may weight technical implementation and scientific analysis more heavily.

## Common Issues & Solutions

### Git/GitHub Issues

**Problem:** "I accidentally committed large files"

```bash
git rm --cached large_file.dat
git commit -m "Remove large file"
git push
```

**Problem:** "I forgot to commit regularly"

- Start committing now! Better late than never
- Break your current code into logical pieces and commit each

**Problem:** "I want to undo my last commit"
```bash
git reset --soft HEAD~1  # Keeps changes
git reset --hard HEAD~1  # Discards changes (careful!)
```

### Python Issues

**Problem:** "ImportError: No module named..."
- Check your virtual environment is activated
- Verify package is in requirements.txt
- Install with: `pip install package_name`

**Problem:** "My code is slow"

- Profile first: `python -m cProfile -s time your_script.py`
- Vectorize NumPy operations
- Avoid loops where possible
- Consider `numba` for critical sections

**Problem:** "Memory error with large arrays"

- Use generators instead of lists where possible
- Process data in chunks
- Use `np.float32` instead of `np.float64` if precision allows

## Submission Checklist

Before submitting, verify:

- [ ] Code runs without errors on a clean environment
- [ ] All required files are present and properly named
- [ ] README includes clear installation and usage instructions
- [ ] At least 5 meaningful commits in Git history
- [ ] Research memo includes all required sections (2-3 pages text)
- [ ] Growth memo includes reflection (1-2 pages)
- [ ] All plots are generated and saved in outputs/figures/
- [ ] Extensions are complete and documented (required for grad students)
- [ ] Code follows style guidelines
- [ ] AI usage follows current phase guidelines
- [ ] Acknowledged any classmates you collaborated with
- [ ] Final push completed before Monday 11:59 PM deadline

## Late Policy Reminder

- **One free extension per semester:** Request ≥24h before deadline → 2-day grace, no penalty
- **Late submission penalty:** 10% per day (24 hours), maximum 3 days late (30% deduction)
- **After 3 days:** Not accepted without documented emergency
- **Note:** Late policy does not apply to Project 0 (Initial Course Reflection & Setup)

## Getting Help

**When to seek help:**

- After 20-30 minutes of genuine effort on a bug
- When you don't understand the physics/math despite reading
- If you're unsure about project requirements
- When you want to bounce extension ideas off someone

**How to ask for help effectively:**

1. Describe what you're trying to do
2. Show what you've tried (code snippets, error messages)
3. Explain what you expected vs. what happened
4. Include minimal reproducible example if possible

**Resources:**

- **Classmates** (collaborate, discuss, learn together!)
%- [**ASTR 596 Virtual Tutor**](https://chatgpt.com/g/g-68aabb9278d08191926feb3f5512686c-astr-596-modeling-the-universe-tutor) (custom GPT for concept and debugging help)
- **Course Slack** (fastest response for quick questions and peer support)
- **Office Hours** (Wednesdays 1-2 PM for complex issues)
- **Friday Lab Sessions** (pair programming and peer support)

**Remember:** Struggling is part of learning. But struggling alone for too long is inefficient. This is a collaborative learning environment—use it!
