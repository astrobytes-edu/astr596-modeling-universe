# ASTR 596 Learning Guide

## Failure Recovery Protocols

### When Things Go Wrong (And They Will)

**Your code is a disaster? Good, now you're learning:**
- **Git broke everything?** `git reflog` shows all commits, even "lost" ones
- **Accidentally deleted files?** Check your IDE's local history (VS Code: Timeline view) OR if you've been committing regularly, `git checkout -- filename` recovers the last committed version. This is why you should commit every time you get something working, even partially.
- **Algorithm completely wrong?** Keep it in `failed_attempts/` folder—documenting what doesn't work is valuable
- **Can't understand your own code from last week?** You forgot to comment. Fix it now, learn for next time.

**Recovery is a skill:** Industry developers break things daily. The difference is they know how to recover quickly. Every disaster teaches you a new git command or IDE feature.

**Git saves you from yourself:** Commit early, commit often, push regularly. Your future self will thank you when you need to recover that working version from 3 days ago.

## Building Research Intuition

**Debugging IS hypothesis testing:**
1. Hypothesis: "The error is in the boundary conditions"
2. Test: Add print statement at boundaries
3. Result: Boundaries are fine
4. New hypothesis: "Check the indexing in the main loop"
5. Test: Print indices at each iteration
6. Result: Off-by-one error found

This IS the scientific method. You're already doing research, just with code instead of lab equipment.

**Read error messages like papers:** Both require parsing dense technical text for the one crucial piece of information buried in paragraph 3.# ASTR 596: Course Learning Guide

This document contains practical strategies for succeeding in this course. It's not about course policies (see syllabus) or philosophy (see "Why ASTR 596 is Different")—it's your technical reference when you're stuck, confused, or need to level up your skills. Everything here is based on cognitive science research and industry best practices. Use this guide to build strong computational habits that will serve you throughout your career.

## Learning Strategies

### Effective Learning Workflow

**Before Class:**
- Read actively - type out examples yourself
- Note specific confusion points
- Attempt project for 30 min (primes your brain for learning)

**During Class:**
- Ask your confusion points immediately
- Debug with partners
- Take implementation notes

**After Class:**
- Review within 24 hours (critical for retention)
- Implement incrementally
- Test each piece before moving on

## Debugging Strategies

### The Systematic Approach

1. **Read the error message** - Really read it, don't just panic
2. **Identify the line** - Where exactly is the problem?
3. **Check your assumptions** - What do you think should happen?
4. **Simplify the problem** - Can you reproduce with minimal code?
5. **Print debugging** - Sometimes `print()` beats fancy debuggers
6. **Use Python debugger (pdb)** - Set breakpoints and step through code (see example below)
7. **Rubber duck debugging** - Explain to an imaginary listener
8. **Take a break** - Fresh eyes catch obvious errors

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

### Common Error Messages Decoded

`IndexError: list index out of range`  
→ You're trying to access element N in a list with <N elements. Check loop bounds and off-by-one errors.

`TypeError: 'NoneType' object is not subscriptable`  
→ A function returned None when you expected a list/array. Check your return statements.

`ValueError: too many values to unpack`  
→ Mismatch between variables and returned values. Print the shape/length of what you're unpacking.

`KeyError: 'key_name'`  
→ Dictionary doesn't have that key. Print dict.keys() to see what's actually there.

`NameError: name 'variable' is not defined`  
→ You're using a variable before defining it, or it's out of scope. Check spelling and indentation.

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
| Matplotlib Gallery | Plot examples | [matplotlib.org/stable/gallery](https://matplotlib.org/stable/gallery/index.html) |
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

## Study Tips & Best Practices

### Why Projects Take Time (It's Not You, It's Neuroscience)

**Distributed Practice > Massed Practice**  
Your brain needs time between sessions to consolidate learning (Cepeda et al., 2006). Complex debugging and algorithm design rarely happen in single marathon sessions—they require "diffuse mode" processing, where your brain works on problems subconsciously between active work periods.

**Practical reality:** Yes, learning to code is time-intensive. A "simple" implementation might take 8+ hours when you're learning. But those hours spread across a week with sleep cycles between them yield better understanding than 8 straight hours of increasingly frustrated debugging.

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

**Why this matters:** In research and industry, undocumented code is dead code. Your future self (and collaborators) need to understand what you wrote and why. Good documentation is expected in any professional setting.

```python
def integrate_orbit(initial_conditions, time_span, method='RK4'):
    """
    Integrate orbital dynamics using specified method.
    
    This is a docstring - it appears when someone types help(integrate_orbit).
    Use triple quotes and follow NumPy/SciPy style (industry standard).
    
    Parameters
    ----------
    initial_conditions : array-like
        [x, y, z, vx, vy, vz] initial position and velocity
        Describe type and what it represents
    time_span : tuple
        (t_start, t_end) integration time bounds
        Always specify units in docs (assumed: seconds)
    method : str, optional
        Integration method: 'Euler', 'RK4', or 'Leapfrog'
        List all valid options explicitly
    
    Returns
    -------
    trajectory : ndarray
        Shape (n_steps, 6) array of positions and velocities
        Always specify output shape/structure
    
    Examples
    --------
    >>> ic = [1, 0, 0, 0, 1, 0]  # Circular orbit
    >>> t_span = (0, 10)
    >>> orbit = integrate_orbit(ic, t_span)
    
    Notes
    -----
    The RK4 method is 4th-order accurate but not symplectic.
    For long-term stability, use 'Leapfrog' despite lower order.
    """
    # Implementation here
```

**Key Documentation Principles:**
1. **Docstrings are contracts** - They promise what your function does
2. **Parameters section** - Type, shape, units, and valid ranges
3. **Returns section** - Exactly what comes back and in what form
4. **Examples section** - Copy-pasteable code showing usage
5. **Notes section** - Gotchas, algorithm choices, or citations

**Industry expectation:** Every public function needs a docstring. In research, include citations to papers/equations you're implementing.

### Testing Strategies

Always validate your code with:

1. **Known solutions** - Reproduce textbook examples
2. **Limiting cases** - Check behavior at extremes
3. **Conservation laws** - Verify energy/momentum when applicable
4. **Unit analysis** - Ensure dimensional consistency
5. **Visualization** - Plot everything; patterns reveal bugs

Example of a test that teaches:
```python
def test_energy_conservation():
    """This test SHOULD fail for Euler method—that teaches us about numerical stability."""
    energy_initial = calculate_total_energy(state_0)
    state_final = integrate_euler(state_0, dt=0.1, steps=1000)
    energy_final = calculate_total_energy(state_final)
    # This assertion will fail, teaching you Euler doesn't conserve energy
    assert np.isclose(energy_initial, energy_final)  # FAILS - that's the lesson!
```

## Getting Help

### When to Seek Help

**Immediate help needed:**
- Stuck on same error for >1 hour
- Don't understand project requirements
- Technical issues (can't install software, GitHub problems)

**Hacking Hours (Thursdays 11 AM) ideal for:**
- Conceptual confusion after genuine attempt
- Discussing different approaches to problems
- Debugging help after you've tried the systematic approach
- "Is my thinking on track?" questions

**Friday class questions valuable for everyone:**
- Your confusion probably helps 3 other students
- Real-time problem solving benefits the whole class
- No question is too basic if you've attempted it first

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

**Green Flags You're Growing:**
- Your questions are becoming more specific
- You're catching bugs faster
- You can predict what will break before running code
- You're helping classmates debug

**What to do:** Visit Hacking Hours or ask in class. Don't wait!

## Time Management

### Evidence-Based Learning Strategies

**These techniques are proven by cognitive science to enhance retention and understanding:**

#### Active Recall (Most Powerful)
**What it is:** Testing yourself WITHOUT looking at notes/code first  
**Why it works:** Retrieval strengthens memory more than re-reading (Karpicke & Blunt, 2011)  
**How to use it:**
- Before checking documentation, try to write the function signature from memory
- Close your code and explain what each function does
- Weekly: Write down everything you remember about a topic, THEN check notes

#### Spaced Repetition
**What it is:** Review material at increasing intervals (1 day, 3 days, 1 week, 2 weeks)  
**Why it works:** Forgetting and re-learning strengthens long-term memory (Cepeda et al., 2006)  
**How to use it:**
- Day after class: Review notes (5 min)
- Three days later: Try to recreate key code (10 min)
- Week later: Implement similar problem without looking
- Two weeks: Explain concept to someone else

#### Interleaving (Mix It Up)
**What it is:** Switch between different topics/projects instead of focusing on one  
**Why it works:** Forces your brain to actively retrieve and apply the right method (Rohrer & Taylor, 2007)  
**How to use it:**
- Work on current project for 1-2 hours, then switch
- Review old projects before starting new ones
- Mix conceptual learning with implementation

#### The Testing Effect
**What it is:** Taking practice tests improves learning more than studying  
**Why it works:** Identifies gaps and strengthens retrieval pathways (Roediger & Karpicke, 2006)  
**How to use it:**
- Write code without any auto-complete (remember: AI/Copilot should be disabled in your IDE)
- Predict output before running code
- Create minimal examples to test your understanding

#### Elaborative Interrogation
**What it is:** Asking "why" and "how" questions while learning  
**Why it works:** Connecting new information to existing knowledge (Dunlosky et al., 2013)  
**How to use it:**
- Don't just learn WHAT broadcasting does, understand WHY NumPy designed it that way
- Ask: "Why does this algorithm fail for this case?"
- Connect: "How is this similar to what we did last week?"

Remember: Everyone feels lost sometimes. The difference between success and failure isn't ability—it's persistence and willingness to seek help.