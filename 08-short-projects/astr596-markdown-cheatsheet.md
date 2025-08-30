# Markdown Cheatsheet for ASTR 596 Memos
*Made by Claude.ai*  
*Quick reference for writing research and growth memos*

## Table of Contents
- [Quick Learning Resources](#quick-learning-resources)
- [Basic Text Formatting](#basic-text-formatting)
- [Lists](#lists)
- [Code Formatting](#code-formatting)
- [Mathematics (LaTeX)](#mathematics-latex-support)
- [Figures with Captions](#figures-with-captions)
- [Tables](#tables)
- [Two-Column Layouts](#two-column-layouts-not-supported)
- [Links and References](#links-and-references)
- [Directory Structure](#directory-structure-display)
- [Research Memo Template](#research-memo-template-structure)
- [Growth Memo Template](#growth-memo-style-informal)
- [Special Characters](#special-characters--symbols)
- [VS Code Tips](#vs-code-tips-for-markdown)
- [Git Best Practices](#git-best-practices-for-projects)
- [Final Checklist](#final-checklist-for-your-memos)

## Important Note on GitHub Compatibility
This guide prioritizes **GitHub-compatible Markdown** since your memos will be viewed directly on GitHub. All examples here will render correctly in GitHub, VS Code preview, and most Markdown viewers.

## Quick Learning Resources

### Learn Markdown
- **[CommonMark 10-minute interactive tutorial](https://commonmark.org/help/tutorial/)** - Best place to start!
- **[GitHub's Markdown Guide](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)** - Official GitHub flavored markdown
- **[Markdown Guide](https://www.markdownguide.org/)** - Comprehensive reference
- **[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)** - Popular quick reference

### Learn Git & GitHub
- **[GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/)** - 10 minute read
- **[Interactive Git Tutorial](https://learngitbranching.js.org/)** - Visual and interactive
- **[Pro Git Book](https://git-scm.com/book/en/v2)** - Free comprehensive guide (just read Ch 1-3 for this course)
- **[GitHub Desktop](https://desktop.github.com/)** - If you prefer GUI over command line
- **[Oh My Git!](https://ohmygit.org/)** - Git learning game

### Quick Git Commands for This Course
```bash
# Clone your assignment repo
git clone https://github.com/YOUR_USERNAME/project-02-YOUR_USERNAME.git

# ALWAYS test before committing (see best practices below)
python star_cluster_sim.py  # Run your code
python -m pytest tests/      # Run tests if you have them

# Check what files changed
git status

# Add specific files (recommended) or all files
git add src/nbody.py  # Add specific file
git add .             # Add all changed files

# Commit with descriptive message
git commit -m "Implement Leapfrog integrator with energy conservation check"

# Push to GitHub
git push

# View commit history
git log --oneline
```

## Basic Text Formatting

```markdown
# Heading 1 (Use for memo title)
## Heading 2 (Main sections like "Methodology")
### Heading 3 (Subsections)
#### Heading 4 (Rarely needed)

Regular paragraph text. No special formatting needed.

Text with **bold emphasis** for important points.
Text in *italics* for emphasis or figure captions.
You can also use ***bold and italic*** together.

Line break within paragraph: end line with two spaces  
This continues on the next line.

New paragraph: leave a blank line between paragraphs.

> Blockquote for highlighting important findings or quotes.
> Continues on multiple lines.

Horizontal rule (section separator):
---
```

## Lists

```markdown
**Bullet points (unordered):**
- First point
- Second point
  - Nested point (indent 2 spaces)
  - Another nested point
- Back to main level

**Numbered lists:**
1. First step
2. Second step
3. Third step
   1. Sub-step (indent 3 spaces)
   2. Another sub-step

**Mixed lists:**
1. Main point
   - Supporting detail
   - Another detail
2. Next main point
```

## Code Formatting

```markdown
Inline code: Use `np.array()` or `dt = 0.01` in sentences.

Code blocks with syntax highlighting (NumPy docstring style per requirements):

```python
def integrate_orbit(positions, velocities, masses, time_span, method='RK4', dt=0.01):
    """
    Integrate star cluster dynamics using specified numerical method.
    
    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) array of initial positions [x, y, z] in parsecs
    velocities : np.ndarray
        Shape (N, 3) array of initial velocities [vx, vy, vz] in km/s
    masses : np.ndarray
        Shape (N,) array of stellar masses in solar masses
    time_span : tuple
        (t_start, t_end) in Myr for integration
    method : str, optional
        Integration method: 'Euler', 'RK4', or 'Leapfrog'
    dt : float, optional
        Time step size in Myr
    
    Returns
    -------
    trajectory : np.ndarray
        Shape (n_steps, N, 3) array of positions over time
    times : np.ndarray
        Shape (n_steps,) array of time points
    
    Raises
    ------
    ValueError
        If method is not recognized or dt <= 0
    
    Examples
    --------
    >>> pos = plummer_model(N=1000, a=1.0)  # 1 pc scale length
    >>> vel = sample_velocities(N=1000, sigma=5.0)  # 5 km/s dispersion
    >>> masses = sample_kroupa_imf(N=1000)  # Kroupa IMF for full mass range
    >>> traj, t = integrate_orbit(pos, vel, masses, (0, 100))
    """
    # Implementation here
    return trajectory, times
```

Shell commands:
```bash
python star_cluster_sim.py --n_stars 1000 --imf salpeter --integrator leapfrog
```

Plain text output:
```
t=0.00 Myr: KE=2.456e+46 erg, PE=-4.912e+46 erg, E_tot=-2.456e+46 erg
t=10.0 Myr: KE=2.456e+46 erg, PE=-4.912e+46 erg, E_tot=-2.456e+46 erg
Energy drift: 0.0001%
```
```

## Mathematics (LaTeX Support)

GitHub supports LaTeX math notation using dollar signs:

```markdown
Inline math: The force follows $F = ma$ where $a = GM/r^2$.

Our fit gives $\alpha = 2.3 \pm 0.1$ with $\chi^2 = 1.2$.

Display equations (centered on their own line):

$$
F = G \frac{m_1 m_2}{r^2}
$$

$$
E = \frac{1}{2}mv^2 - \frac{GMm}{r}
$$

$$
\chi^2 = \sum_{i=1}^{N} \frac{(O_i - E_i)^2}{\sigma_i^2}
$$

Common symbols:
- Subscripts: $x_i$, $t_0$
- Superscripts: $r^2$, $10^{-5}$
- Greek letters: $\alpha$, $\beta$, $\gamma$, $\Omega$
- Fractions: $\frac{a}{b}$
- Square root: $\sqrt{2}$
- Summation: $\sum_{i=1}^{N}$
- Integral: $\int_0^{\infty}$
- Partial derivatives: $\frac{\partial f}{\partial x}$
```

## Figures with Captions

**Critical for your memos:** Figures must be in your `outputs/figures/` directory and committed to Git!

```markdown
## Single Figure

![HR diagram](outputs/figures/hr_diagram.png)
*Figure 1: Hertzsprung-Russell diagram showing the main sequence and giant branch. 
Stars are colored by age, with blue representing young stars (< 1 Myr) and red 
representing evolved stars (> 10 Myr).*

## Referencing Figures in Text

As shown in Figure 1, the main sequence is clearly visible...

The orbital decay (Figure 2) demonstrates numerical energy dissipation...

## Multiple Related Figures

![Initial conditions](outputs/figures/initial_state.png)
*Figure 2a: Initial particle distribution showing uniform density.*

![Final state](outputs/figures/final_state.png)
*Figure 2b: Final particle distribution after 10 Gyr showing core collapse.*
```

### Important Figure Guidelines
1. **Always use relative paths** starting with `outputs/figures/`
2. **Include descriptive alt text** in the square brackets
3. **Write detailed captions** in italics immediately below
4. **Number your figures** for easy reference
5. **Save figures before committing** - broken image links lose points!

## Tables

Tables work well for parameters, results comparison, and data summary:

```markdown
## Simple Table

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| N_stars | 1000 | - | Number of stars |
| $M_{total}$ | 650 | $M_\odot$ | Total cluster mass |
| $R_{half}$ | 1.0 | pc | Half-mass radius |
| $t_{cross}$ | 2.3 | Myr | Crossing time |

## Method Comparison Table (great for research memos!)

| Method | CPU Time (s) | Energy Error | Angular Mom. Error | Stability |
|--------|--------------|--------------|-------------------|-----------|
| Euler | 0.5 | $10^{-2}$ | $10^{-1}$ | Unstable |
| RK4 | 2.1 | $10^{-8}$ | $10^{-9}$ | Stable |
| Leapfrog | 1.2 | $10^{-10}$ | $10^{-12}$ | Stable |

## IMF Sampling Results

| Mass Range | Kroupa | Chabrier | Salpeter (M>0.5) | Our Sample |
|------------|--------|----------|------------------|------------|
| 0.08-0.5 $M_\odot$ | 55% | 58% | N/A | 56% |
| 0.5-1.0 $M_\odot$ | 25% | 23% | 35% | 24% |
| 1.0-8.0 $M_\odot$ | 18% | 17% | 58% | 18% |
| >8.0 $M_\odot$ | 2% | 2% | 7% | 2% |

*Note: Salpeter IMF only valid for M ≥ 0.5 M☉; we used Kroupa for full mass range*

## Table Alignment (optional, doesn't affect GitHub rendering)

| Left Aligned | Center | Right Aligned |
|:-------------|:------:|--------------:|
| Text | Text | Numbers |
| More | More | 123.45 |
```

## Two-Column Layouts (Not Supported)

**Important:** GitHub Markdown does **not** support two-column layouts. However, here are workarounds:

### For Side-by-Side Figures (Use HTML)
```markdown
<div align="center">
<img src="outputs/figures/before.png" width="45%" />
<img src="outputs/figures/after.png" width="45%" />
</div>

*Figure 3: (Left) System before collision. (Right) System after collision.*
```

### For Side-by-Side Content (Use Tables)
```markdown
| Initial Conditions | Final State |
|--------------------|-------------|
| Uniform distribution | Core collapse |
| Total energy: -0.5 | Total energy: -0.48 |
| Virial ratio: 0.5 | Virial ratio: 0.45 |
```

**Note:** For true two-column layouts, you'd need to export to PDF using MyST or LaTeX. For this class, single-column Markdown on GitHub is perfectly fine!

## Links and References

```markdown
## External Links
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)

## Internal Links (to other files in your repo)
See [`src/nbody.py`](src/nbody.py) for implementation details.

Details in the [research memo](research_memo.md).

## Acknowledging Collaborators (REQUIRED if you discussed with others)
### Acknowledgments

I discussed debugging strategies with Jane Doe and the concept of symplectic 
integration with John Smith. All code implementation is my own work.

## Footnote-style Links (cleaner for long URLs)
The data comes from Gaia DR3[^1].

[^1]: https://www.cosmos.esa.int/web/gaia/dr3

## References Section
## References

1. Aarseth, S. (2003). *Gravitational N-Body Simulations*
2. Binney & Tremaine (2008). *Galactic Dynamics*, 2nd Edition
3. Heggie & Hut (2003). *The Gravitational Million-Body Problem*
4. [Astropy Documentation](https://docs.astropy.org)
```

## Directory Structure Display

```markdown
Show your project organization (per submission requirements):

```bash
project_2/
├── src/                    # Your source code (modular design)
│   ├── nbody.py           # Main N-body integrator
│   ├── forces.py          # Force calculations  
│   ├── integrators.py     # Euler, RK4, Leapfrog implementations
│   └── imf.py             # IMF sampling functions
├── tests/                  # Basic tests for key functions
│   └── test_energy.py     # Energy conservation tests
├── outputs/
│   ├── figures/           # All generated plots
│   │   ├── cluster_evolution.png
│   │   └── energy_conservation.png
│   └── data/              # Output data files
│       └── final_state.npy
├── notes/                  # Optional: ongoing project notes
│   └── notes.md           # Your thoughts while working
├── README.md              # Installation, usage, results summary
├── requirements.txt       # Dependencies with versions
├── research_memo.md       # Analysis (2-3 pages)
├── growth_memo.md         # Reflection (1-2 pages)  
└── .gitignore            # Use provided template
```
```

## Research Memo Template Structure

```markdown
# Research Memo - Project 2: Star Cluster Dynamics

**Author:** Your Name  
**Date:** September 22, 2024

## Executive Summary

This project implemented N-body simulation methods to model star cluster dynamics. 
Using RK4 and Leapfrog integrators, we simulated a 1000-star cluster sampled from 
a Salpeter IMF. Key findings include core collapse after 50 crossing times and 
energy conservation to 0.01% over 100 Myr...

## Methodology

### Initial Mass Function Sampling

We sampled stellar masses from the Salpeter IMF:
$\xi(m) \propto m^{-2.35}$
with masses ranging from 0.1 to 100 $M_\odot$.

### Integration Methods

Implemented three numerical integrators:
- Euler method (first-order, unstable)
- RK4 (fourth-order, general purpose)
- Leapfrog (second-order, symplectic)

### Force Calculation

Direct N-body summation with softening parameter $\epsilon = 0.01$ pc:
$F_i = -G \sum_{j \neq i} \frac{m_j (r_i - r_j)}{(|r_i - r_j|^2 + \epsilon^2)^{3/2}}$

## Results

### Cluster Evolution

Figure 1 shows the cluster evolution over 100 Myr...

![Cluster evolution](outputs/figures/cluster_evolution.png)
*Figure 1: Star cluster evolution showing core collapse. Panels show 
t = 0, 25, 50, and 100 Myr. Colors indicate stellar mass following the IMF.*

### Energy Conservation

The Leapfrog integrator conserved energy to machine precision...

![Energy conservation](outputs/figures/energy_conservation.png)
*Figure 2: Relative energy error over time for three integration methods. 
Leapfrog maintains $\Delta E/E < 10^{-10}$ while RK4 shows secular drift.*

## Validation

Verified implementation through:
1. Two-body Kepler orbits (analytical comparison)
2. Virial equilibrium for self-gravitating systems
3. Energy and angular momentum conservation

## Extensions

### Adaptive Timestep
Implemented KDK Leapfrog with individual timesteps based on local dynamical time...

### Mass Segregation Analysis
Measured mass segregation using minimum spanning tree method...

## Conclusions

This project demonstrated that symplectic integrators are essential for long-term 
cluster evolution. The Leapfrog method preserved energy over 1000+ crossing times 
while maintaining computational efficiency...

## References

1. Aarseth, S. (2003). *Gravitational N-Body Simulations*
2. Heggie & Hut (2003). *The Gravitational Million-Body Problem*
3. Kroupa, P. (2001). "On the variation of the initial mass function"
```

## Growth Memo Style (Informal)

```markdown
## Growth Memo - Project 2

**Name:** Your Name  
**Date:** September 22, 2024  
**Project:** Star Cluster Dynamics

### Summary

We built an N-body simulator to model star clusters, implementing different 
integrators to see how 1000 stars evolve under gravity. The technical challenge 
was balancing accuracy with computational efficiency while conserving energy.

### Technical Skills Developed

I can now implement symplectic integrators like Leapfrog, sample from an IMF 
using inverse transform method, and diagnose energy drift in N-body simulations. 
Also learned to vectorize force calculations with NumPy for 10x speedup.

### Key Challenges & Solutions

**The Problem:**
My star cluster kept exploding - all stars flying apart even with small timesteps.

**My Solution:**
After checking units three times, I realized I was using parsecs for distance 
but solar masses for mass without proper G. Added a proper units module and 
everything clicked. Then discovered I needed softening to prevent close encounters.

**What This Taught Me:**
Always establish a consistent unit system from the start. Astrophysics bugs are 
often unit bugs. Also, physical intuition matters - stars shouldn't fly apart 
at 1000 km/s in a cluster!

### Lessons in Algorithm Design

**The Problem:**
Particles kept flying off to infinity no matter what timestep I used, even 
though my force calculation was correct.

**The Journey:**
- Hour 1: Checked force calculation 10 times
- Hour 2: Printed every intermediate value
- Hour 3: Drew the algorithm on paper... wait... OH NO

**The Solution:**
I was updating particle positions inside the force loop! Classic mistake - you 
need ALL forces calculated before updating ANY positions. Drawing the algorithm 
flow on paper saved me.

**Key Learning:**
Sometimes the bug isn't in the physics or the math - it's in the algorithm flow. 
Now I always diagram the update sequence before coding. Planning on paper first 
would have saved me 3 hours of debugging!

### AI Usage Reflection

**Most Significant AI Interaction:**
Asked Claude about why Leapfrog conserves energy better than RK4. It explained 
symplectic integration and phase space volume preservation.

**Critical Thinking Check:**
Claude initially said RK4 is "always better" - but that's wrong for long-term 
integration! Tested both and found Leapfrog superior for energy conservation.

### What Got Me Excited

When I zoomed in on the cluster core and saw binary stars forming spontaneously 
through three-body interactions - that wasn't programmed, it just emerged from 
gravity! That's when it hit me: we're literally watching stellar dynamics that 
take millions of years, compressed into seconds on my laptop.

### Reflection

Seeing mass segregation happen naturally (massive stars sinking to center) without 
explicitly coding it was mind-blowing. Physics is just there in the equations. 
I'm starting to appreciate how computational astrophysics lets us be "experimental 
astronomers" in a way.
```

## Special Characters & Symbols

```markdown
Common symbols (copy and paste as needed):
- Degree: ° (as in 45°)
- Plus/minus: ± 
- Approximately: ≈
- Not equal: ≠
- Less/greater than or equal: ≤ ≥
- Multiplication: × 
- Arrow: →
- Infinity: ∞
- Pi: π
- Solar mass: M☉ or $M_\odot$
- Solar radius: R☉ or $R_\odot$

Escape special Markdown characters with backslash:
\* \_ \# \[ \] \( \) \! \` \> \{ \}
```

## VS Code Tips for Markdown

### Essential Setup
1. Install "Markdown Preview Enhanced" extension
2. Use `Cmd+Shift+V` to preview your memo
3. Use `Cmd+K V` to open preview side-by-side

### Helpful Extensions
- **Markdown All in One**: Table formatting, shortcuts
- **markdownlint**: Catches formatting issues
- **Paste Image**: Paste screenshots directly (saves to folder)

### Quick Commands
- Create table: Type `|Column 1|Column 2|` then Tab
- Toggle bold: `Cmd+B`
- Toggle italic: `Cmd+I`
- Format document: `Shift+Option+F`

## Git Best Practices for Projects

### The Golden Rule: Test Before You Commit!

**Before EVERY commit:**
```bash
# 1. Test your code runs without errors
python src/nbody.py  # Or whatever your main script is

# 2. Quick validation check
python -c "from src.nbody import integrate_cluster; print('Import successful!')"

# 3. If you have tests, run them
python -m pytest tests/  # If using pytest
# or
python tests/test_energy.py  # Direct test execution

# 4. Check your figures were generated
ls outputs/figures/  # Should see your plots

# 5. NOW you're ready to commit!
git add .
git commit -m "Add Leapfrog integrator with working energy conservation"
```

### Commit Early and Often

**Good commit rhythm:**
- Commit when you get something working (even small things!)
- Commit before trying something risky
- Commit before lunch/end of day
- Commit when you finish a function/class
- Commit before refactoring

**Good commit messages:**
```bash
# ✅ GOOD - Specific and informative
git commit -m "Add IMF sampling using inverse transform method"
git commit -m "Fix unit conversion bug in force calculation"
git commit -m "Implement Plummer model for initial conditions"
git commit -m "Add energy conservation plot to outputs"

# ❌ BAD - Too vague
git commit -m "Update"
git commit -m "Fix bug"
git commit -m "Changes"
git commit -m "Done"
```

### Pro Workflow: Test → Stage → Review → Commit

```bash
# 1. Run your code to make sure it works
python src/star_cluster_sim.py --test_mode

# 2. Check what you're about to commit
git status  # See what files changed
git diff    # See actual changes (press 'q' to exit)

# 3. Stage files intentionally (not everything!)
git add src/integrators.py  # Add the file you actually worked on
git add outputs/figures/energy_plot.png  # Add new output

# 4. Double-check what's staged
git status  # Files in green will be committed

# 5. Commit with meaningful message
git commit -m "Implement KDK Leapfrog variant for better energy conservation"

# 6. Push to GitHub
git push
```

### When Things Go Wrong

```bash
# Accidentally committed without testing? Fix it:
# Edit the broken code, then:
git add .
git commit --amend -m "Fixed: Add Leapfrog integrator with working energy conservation"

# Want to undo last commit (before pushing)?
git reset --soft HEAD~1  # Keeps your changes
# Fix the issue, then commit again

# Already pushed broken code? Just fix and commit again:
# Fix the code
git add .
git commit -m "Fix energy conservation bug in Leapfrog implementation"
git push
```

### Tips for Success

1. **Never commit broken code** - Your future self will thank you
2. **Use .gitignore** - Don't commit `__pycache__/`, `.DS_Store`, or huge data files
3. **Commit related changes together** - One feature/fix per commit
4. **Write messages for your future self** - "What did I change and why?"
5. **Push regularly** - GitHub is your backup!

### Sample .gitignore for Projects
```
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Large data files (generate these, don't commit)
*.npy
*.hdf5
large_simulation_output.txt

# But DO commit your figure outputs!
# outputs/figures/ should be included
```

## Final Checklist for Your Memos

### Before Submitting Research Memo
- [ ] All required sections present (Executive Summary, Methodology, Results, Validation, Extensions, Conclusions, References)
- [ ] 2-3 pages of text (not counting figures/references)
- [ ] Figures saved in `outputs/figures/` and displaying correctly
- [ ] Figure captions in italics with numbers
- [ ] Math equations using $ symbols render properly
- [ ] Extensions documented (required for grad students)
- [ ] Collaborators acknowledged if you discussed with classmates
- [ ] File named `research_memo.md` (or .pdf if exported)
- [ ] Preview looks good in VS Code
- [ ] Pushed to GitHub and images display correctly online

### Before Submitting Growth Memo  
- [ ] Used the provided template as starting point
- [ ] ~400-800 words of reflection (1-2 pages)
- [ ] Included AI usage reflection per phase guidelines
- [ ] Honest about challenges and victories
- [ ] Specific examples rather than general statements
- [ ] Reflected on technical skills developed
- [ ] File named `growth_memo.md` in project root
- [ ] Mentioned what you'd tell your past self

### Git Requirements
- [ ] Committed regularly throughout the project
- [ ] Descriptive commit messages (not just "update" or "fix")
- [ ] All required files in correct locations
- [ ] Tested code before final commit
- [ ] Pushed to GitHub and verified everything displays correctly

## Need More Features?

If you need advanced features not supported in GitHub Markdown (like true two-column layouts, complex equations, or citations), you have two options:

1. **For final submission**: Export to PDF using MyST or Pandoc
2. **For most cases**: Keep it simple with GitHub Markdown - it's sufficient for this course!

Remember: **Content > Formatting**. Focus on clear scientific writing rather than perfect typography.

---

*Last tip: Keep this cheatsheet open while writing your memos. Copy-paste the templates and modify as needed!*