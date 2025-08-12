# ASTR 596: Project Submission Guide

## Project Schedule & Deadlines

### Project Timeline
Projects are assigned on Mondays (posted to GitHub Classroom) and due the following Monday at 11:59 PM. This schedule allows you to review requirements before Friday's class, where we'll work on implementation together.

| Project | Assigned | Due Date | Topic | Key Concepts |
|---------|----------|----------|-------|--------------|
| **Project 1** | Aug 25 (Mon) | Sept 8 (Mon) | Python/OOP/Stellar Physics Basics | Classes, inheritance, HR diagrams |
| **Project 2** | Sept 8 (Mon) | Sept 22 (Mon) | ODE Integration + N-Body Dynamics + Monte Carlo Sampling | Euler, RK4, Leapfrog, IMF sampling |
| **Project 3** | Sept 22 (Mon) | Oct 6 (Mon) | Regression/ML Fundamentals | Gradient descent, loss functions, optimization |
| **Project 4** | Oct 6 (Mon) | Oct 20 (Mon) | Monte Carlo Radiative Transfer | Photon packets, scattering, absorption |
| **Project 5** | Oct 20 (Mon) | Nov 3 (Mon) | Bayesian/MCMC | Priors, likelihood, Metropolis-Hastings |
| **Project 6** | Nov 3 (Mon) | Nov 17 (Mon) | Gaussian Processes | Kernels, hyperparameters, regression |
| **Final Project** | Nov 17 (Mon) | Dec 18 (Thu) | Neural Networks (From Scratch + JAX) | Backprop, autodiff, JAX ecosystem |

### Two-Week Project Workflow

**Week 1: Understanding & Initial Implementation**
- **Day 1-2 (Mon-Tue):** Read assignment thoroughly, understand requirements, review relevant JupyterBook chapter
- **Day 3-4 (Wed-Thu):** Begin implementation, focus on core functionality
- **Day 5 (Fri):** Class session - ask questions, pair programming, debug with peers
- **Day 6-7 (Sat-Sun):** Continue implementation based on class insights

**Week 2: Refinement & Completion**
- **Day 8-9 (Mon-Tue):** Complete base requirements, begin mandatory extensions
- **Day 10-11 (Wed-Thu):** Test edge cases, optimize performance
- **Day 12 (Fri):** Class session - final debugging, optimization discussions
- **Day 13 (Sat-Sun):** Polish code, write documentation, complete project memo
- **Day 14 (Mon):** Final review, submit by 11:59 PM

## Submission Requirements

### Each Project Must Include

#### 1. Code Components
```
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
├── project_memo.md       # Your analysis and reflection
└── .gitignore           # Properly configured
```

**Code Standards:**
- Modular design with clear separation of concerns
- No God functions (functions should do one thing well)
- Meaningful variable names (no single letters except for indices)
- Type hints encouraged for function signatures
- No global variables unless absolutely necessary
- Error handling for edge cases

#### 2. Project Memo (Markdown Format)

Your memo should be **2-5 pages** and include:

```markdown
# Project N: [Title] - Memo
Author: [Your Name]
Date: [Submission Date]

## Executive Summary
[1-2 paragraphs summarizing what you did and key findings]

## Methodology
[How you approached the problem, key algorithmic choices]

## Results
[Key findings with embedded plots using relative paths]
![Description](outputs/figures/plot1.png)

## Computational Performance
[Runtime analysis, bottlenecks identified, optimizations made]

## Challenges & Solutions
[What was hard, how you solved it, what you learned]

## Extensions Implemented
[Description of mandatory extensions completed]

## Reflection
[What you learned about computational physics and programming]
```

#### 3. Documentation Requirements

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

#### 4. GitHub Classroom Requirements

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

### Project Grading Breakdown (100 points total)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Core Implementation** | 40 | Correctness, completeness, follows specifications |
| **Mandatory Extensions** | 30 | All required extensions implemented and working |
| **Code Quality** | 15 | Structure, readability, documentation, style |
| **Project Memo** | 15 | Analysis quality, reflection depth, visualization |

### Detailed Rubric

#### Core Implementation (40 points)
- **Excellent (36-40):** All requirements met, code runs without errors, produces correct results, handles edge cases
- **Good (32-35):** Most requirements met, minor bugs, generally correct results
- **Satisfactory (28-31):** Core functionality works, some requirements missing, several bugs
- **Needs Improvement (0-27):** Major functionality missing, significant bugs, incorrect results

#### Mandatory Extensions (30 points)
- **Excellent (27-30):** All extensions complete, creative implementation, goes beyond minimum
- **Good (24-26):** All extensions complete, solid implementation
- **Satisfactory (21-23):** Most extensions complete, basic implementation
- **Needs Improvement (0-20):** Extensions missing or non-functional

#### Code Quality (15 points)
- **Structure (5 pts):** Modular design, appropriate file organization
- **Documentation (5 pts):** Clear docstrings, helpful comments, complete README
- **Style (5 pts):** Consistent formatting, meaningful names, follows Python conventions

#### Project Memo (15 points)
- **Analysis (7 pts):** Demonstrates understanding, interprets results correctly
- **Reflection (4 pts):** Thoughtful discussion of challenges and learning
- **Presentation (4 pts):** Clear writing, effective visualizations, proper formatting

## Mandatory Extensions

Each project includes required extensions that push you beyond the base implementation. These are NOT optional and constitute 30% of your project grade.

### Types of Extensions

**Performance Extensions:**
- Optimize algorithms for speed (vectorization, better algorithms)
- Memory optimization for large-scale problems
- Parallel processing implementation

**Scientific Extensions:**
- Parameter studies and sensitivity analysis
- Comparison with analytical solutions where available
- Error analysis and convergence studies

**Methodological Extensions:**
- Implement alternative algorithms and compare
- Add adaptive methods (timestep, resolution, etc.)
- Extend to more complex physics

**Visualization Extensions:**
- Interactive plots
- Animations of time evolution
- 3D visualizations where appropriate

*Specific extensions for each project will be detailed in the individual project assignments.*

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
- Consider numba for critical sections

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
- [ ] Project memo includes all required sections
- [ ] All plots are generated and saved in outputs/figures/
- [ ] Mandatory extensions are complete and documented
- [ ] Code follows style guidelines (no IDE AI assistance used)
- [ ] Final push completed before Monday 11:59 PM deadline

## Late Policy Reminder

- One no-questions-asked 2-day extension per semester
- Must be requested at least 24 hours before deadline
- 10% penalty per day after grace period
- Submit early if complete—no bonus, but peace of mind!

## Getting Help

**When to seek help:**
- After 20-30 minutes of genuine effort on a bug
- When you don't understand the physics/math despite reading
- If you're unsure about project requirements

**How to ask for help effectively:**
1. Describe what you're trying to do
2. Show what you've tried (code snippets, error messages)
3. Explain what you expected vs. what happened
4. Include minimal reproducible example if possible

**Resources:**
- Course Slack (fastest response)
- Office hours (for complex issues)
- Pair programming sessions (learn from peers)
- AI tutors (for concepts, not code generation)

**Remember:** Struggling is part of learning. But struggling alone for too long is inefficient. Ask for help!