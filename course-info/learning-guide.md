# ASTR 596: Course Learning Guide

## Quick Links
- [Course Philosophy](course-info/why-astr596-is-different)
- [AI Usage Guidelines](course-info/ai-guidelines)
- [Project Requirements](short-projects/0_project_submission_guide)
- [Software Setup](reference/software-setup)

## Table of Contents
1. [Learning Strategies](#learning-strategies)
2. [Debugging Strategies](#debugging-strategies)
3. [Resources & Documentation](#resources--documentation)
4. [Study Tips & Best Practices](#study-tips--best-practices)
5. [Getting Help](#getting-help)

## Learning Strategies

### The Three Pillars of Computational Learning

1. **Conceptual Understanding** - Know the theory behind what you're implementing
2. **Practical Implementation** - Transform theory into working code
3. **Critical Evaluation** - Understand when methods work, fail, and why

### Effective Learning Workflow

#### Before Class
1. **Read actively** - Don't just skim the JupyterBook chapters
2. **Try examples** - Type out code examples yourself (no copy-paste!)
3. **Note questions** - Write down confusion points to ask in class
4. **Attempt project start** - Even 30 minutes of trying helps frame questions

#### During Class
1. **Ask "dumb" questions** - They're usually the most important
2. **Engage in pair programming** - Explain your thinking out loud
3. **Take implementation notes** - Document approaches that work
4. **Debug together** - Learn from others' errors too

#### After Class
1. **Review immediately** - Solidify concepts while fresh
2. **Implement incrementally** - Small, tested pieces beat large untested code
3. **Document learnings** - Your future self will thank you
4. **Help peers** - Teaching solidifies understanding

## Debugging Strategies

### The Systematic Approach

1. **Read the error message** - Really read it, don't just panic
2. **Identify the line** - Where exactly is the problem?
3. **Check your assumptions** - What do you think should happen?
4. **Simplify the problem** - Can you reproduce with minimal code?
5. **Print debugging** - Sometimes `print()` beats fancy debuggers
6. **Rubber duck debugging** - Explain to an imaginary listener
7. **Take a break** - Fresh eyes catch obvious errors

### Common Python Pitfalls

```python
# Mutable default arguments - WRONG
def bad_function(lst=[]):  
    lst.append(1)
    return lst

# Correct approach
def good_function(lst=None):
    if lst is None:
        lst = []
    lst.append(1)
    return lst
```

Other common issues:
- **Integer division**: Be aware of Python 2 vs 3 differences
- **Indentation errors**: Never mix tabs and spaces
- **Off-by-one errors**: Remember Python is 0-indexed
- **Scope confusion**: Understand local vs global variables
- **NumPy broadcasting**: Check array shapes with `.shape`

### Using Python Debugger (pdb)

```python
import pdb

def problematic_function(x):
    result = x * 2
    pdb.set_trace()  # Execution stops here
    return result / 0  # Obviously wrong

# Commands in pdb:
# n - next line
# s - step into function
# c - continue
# l - list code
# p variable - print variable
# pp variable - pretty print
# h - help
```

## Resources & Documentation

### Essential Python References

| Resource | Best For | Link |
|----------|----------|------|
| Official Python Docs | Language features | [docs.python.org](https://docs.python.org/3/) |
| NumPy Documentation | Array operations | [numpy.org/doc](https://numpy.org/doc/stable/) |
| Matplotlib Gallery | Plot examples | [matplotlib.org/gallery](https://matplotlib.org/stable/gallery/index.html) |
| SciPy Documentation | Scientific functions | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) |
| Real Python | Tutorials | [realpython.com](https://realpython.com/) |
| Python Tutor | Visualize execution | [pythontutor.com](http://pythontutor.com/) |

### Machine Learning & JAX

| Resource | Purpose |
|----------|---------|
| [JAX Documentation](https://jax.readthedocs.io/) | Core JAX features |
| [Equinox Docs](https://docs.kidger.site/equinox/) | Neural network library |
| [Flax Documentation](https://flax.readthedocs.io/) | Alternative NN library |
| [Ting's ML Review](https://arxiv.org/abs/2506.12230) | Astronomy-specific ML |

### Recommended Video Resources

- **3Blue1Brown** - Visual mathematical intuition
- **StatQuest** - Statistics with clear explanations
- **Computerphile** - Computer science concepts
- **Two Minute Papers** - Latest research explained

## Study Tips & Best Practices

### Project Management Timeline

**Two-Week Project Workflow:**

| Days | Focus | Actions |
|------|-------|---------|
| 1-2 | Understanding | Read requirements, review relevant chapters |
| 3-4 | Initial attempt | Start coding, identify challenges |
| 5-6 | Debug & test | Fix errors, test edge cases |
| 7 (Fri) | Class session | Get help, collaborate |
| 8-9 | Refinement | Incorporate class insights |
| 10-11 | Extensions | Complete required extensions |
| 12-13 | Polish | Document, create figures, write memo |
| 14 | Submit | Final review and push to GitHub |

### Code Organization Best Practices

```python
# Project structure
project_name/
├── README.md           # Installation and usage instructions
├── requirements.txt    # Package dependencies
├── src/
│   ├── __init__.py
│   ├── physics.py     # Physics calculations
│   ├── numerics.py    # Numerical methods
│   └── plotting.py    # Visualization functions
├── tests/
│   └── test_physics.py
├── data/
│   └── input_files/
├── outputs/
│   └── figures/
└── main.py            # Entry point
```

### Writing Good Documentation

```python
def integrate_orbit(initial_conditions, time_span, method='RK4'):
    """
    Integrate orbital dynamics using specified method.
    
    Parameters
    ----------
    initial_conditions : array-like
        [x, y, z, vx, vy, vz] initial position and velocity
    time_span : tuple
        (t_start, t_end) integration time bounds
    method : str, optional
        Integration method: 'Euler', 'RK4', or 'Leapfrog'
    
    Returns
    -------
    trajectory : ndarray
        Shape (n_steps, 6) array of positions and velocities
    
    Examples
    --------
    >>> ic = [1, 0, 0, 0, 1, 0]  # Circular orbit
    >>> t_span = (0, 10)
    >>> orbit = integrate_orbit(ic, t_span)
    """
    # Implementation here
```

### Testing Strategies

Always validate your code with:

1. **Known solutions** - Reproduce textbook examples
2. **Limiting cases** - Check behavior at extremes
3. **Conservation laws** - Verify energy/momentum when applicable
4. **Unit analysis** - Ensure dimensional consistency
5. **Visualization** - Plot everything; patterns reveal bugs

Example test:
```python
def test_circular_orbit():
    """Test that circular orbit maintains constant radius."""
    ic = [1, 0, 0, 0, 1, 0]
    orbit = integrate_orbit(ic, (0, 2*np.pi))
    radii = np.sqrt(orbit[:, 0]**2 + orbit[:, 1]**2)
    assert np.allclose(radii, 1.0, rtol=1e-3)
```

## Getting Help

### When to Seek Help

**Immediate help needed:**
- Stuck on same error for >1 hour
- Don't understand project requirements
- Technical issues (can't install software, GitHub problems)

**Office hours appropriate:**
- Conceptual confusion after reading
- Want to discuss approach before coding
- Need debugging help after genuine attempt

### How to Ask Good Questions

**Good question format:**
```
"I'm trying to [goal]. 
I've attempted [what you tried].
I expected [expected result] but got [actual result].
I've checked [what you've verified].
Could you help me understand [specific confusion]?"
```

**Include:**
- Minimal reproducible example
- Full error message
- What you've already tried
- Relevant code snippet (not entire file)

### Red Flags: You Need Help NOW

**Technical Warning Signs:**
- Your code "works" but you can't explain why
- Changing things randomly hoping for success
- Solution is 10x longer than expected
- Avoiding entire project sections

**Emotional Warning Signs:**
- Consistent overwhelm
- Comparing yourself negatively to peers
- Physical stress symptoms
- Considering dropping the course

**What to do:** Email instructor immediately or visit office hours. Don't wait!

### Study Group Guidelines

With only 4 students, you're all in this together:

1. **Form partnerships early** - Don't work in isolation
2. **Rotate pairs weekly** - Learn from everyone
3. **Share struggles** - Everyone gets stuck
4. **Teach each other** - Best way to solidify understanding
5. **Respect boundaries** - Individual work stays individual

## Time Management

### Daily Practice Schedule

**Recommended daily workflow:**
- **15 min**: Review previous day's work
- **45 min**: New implementation
- **15 min**: Test and debug
- **15 min**: Document and commit

This 90-minute daily practice is more effective than marathon sessions.

### Pomodoro Technique for Coding

```
25 min: Focused coding (no distractions)
5 min: Break (stretch, water, look away from screen)
25 min: Continue coding
5 min: Break
25 min: Debug and test
5 min: Break
25 min: Document and refactor
15 min: Longer break, review progress
```

## Final Reminders

- **The struggle is the learning** - Don't avoid it
- **Mistakes are data** - Learn from every bug
- **Progress over perfection** - Working code beats perfect theory
- **You're not alone** - Use office hours and peers
- **Celebrate small wins** - Every function that works is progress

Remember: Everyone feels lost sometimes. The difference between success and failure isn't ability—it's persistence and willingness to seek help.

**You've got this!** 🚀