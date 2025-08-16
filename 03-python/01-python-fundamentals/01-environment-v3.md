# Chapter 1: Computational Environments & Scientific Workflows

## Learning Objectives

By the end of this chapter, you will be able to:
- Use IPython effectively as your primary interactive computing environment
- Diagnose why identical code produces different results on different machines
- Explain how Python's import system locates and loads packages
- Recognize and avoid the hidden dangers of Jupyter notebooks that corrupt scientific computing
- Create reproducible computational environments using conda
- Debug common environment problems systematically using diagnostic tools
- Convert notebook-based explorations into reproducible scripts
- Test quick ideas and algorithms efficiently at the terminal

## Chapter Overview

You download code from a recent astronomy paper, run it, and get completely different results than published. Or worse, it doesn't run at all. This isn't unusual — it's the norm in computational science. The problem isn't bad code; it's that scientific computing happens in complex environments where tiny differences cascade into major failures.

This chapter explains why these problems occur and how to prevent them. We'll explore how Python finds and loads code, master IPython as your computational laboratory, understand why we're abandoning Jupyter notebooks after Project 1 despite their popularity, and build your skills in creating reproducible computational environments. These foundations will support every line of code you write in this course and throughout your research career.

## Quick Setup Recap

If you've completed the setup, you should have Miniforge installed with a conda environment named `astr596`. The essential commands:

```bash
conda activate astr596        # Start here every time
ipython                      # Launch IPython (better than python)
python script.py             # Run Python scripts
jupyter lab                  # Start Jupyter (Project 1 only)
```

If something's not working, check you're in the right environment first — it's the cause of 90% of "module not found" errors.

## 1.1 IPython: Your Computational Laboratory

While you could use the basic Python interpreter, IPython (Interactive Python) transforms the terminal into a powerful environment for scientific computing. IPython is what professional computational scientists use for exploratory work, quick calculations, and algorithm testing. Let's see why.

### Basic IPython Advantages

Launch IPython by typing `ipython` in your terminal (after activating your environment):

```
$ ipython
Python 3.11.5 | packaged by conda-forge
Type 'copyright', 'credits' or 'license' for more information
IPython 8.14.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 
```

Notice the prompt is `In [1]:` instead of `>>>`. This numbering helps you reference previous inputs and outputs:

```python
In [1]: import numpy as np

In [2]: data = np.random.randn(1000)

In [3]: data.mean()
Out[3]: -0.0234567

In [4]: Out[3] * 100  # Reference previous output
Out[4]: -2.34567

In [5]: In[2]  # See what command you ran
Out[5]: 'data = np.random.randn(1000)'
```

IPython provides tab completion for exploration:

```python
In [6]: data.<TAB>
# Shows all available methods for the array
data.all       data.argmax    data.astype    data.clip      
data.compress  data.copy      data.mean      data.std
# ... and many more

In [7]: np.lin<TAB>
# Shows all NumPy functions starting with 'lin'
np.linalg      np.linspace
```

### Magic Commands: IPython's Superpowers

IPython's "magic" commands (prefixed with `%`) provide functionality beyond standard Python. These are essential tools for scientific computing:

```python
# Timing code performance - crucial for algorithm comparison
In [8]: %timeit sum(range(1000))
21.3 µs ± 267 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [9]: %timeit np.arange(1000).sum()
3.47 µs ± 42.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
# NumPy is 6x faster!

# Running scripts while keeping variables
In [10]: %run my_analysis.py
# Now all variables from the script are available in your session

# Quick debugging when something crashes
In [11]: def buggy_function(x):
    ...:     return x / (x - 5)

In [12]: buggy_function(5)
# ZeroDivisionError!

In [13]: %debug
# Opens debugger at the error point
ipdb> x
5
ipdb> x - 5
0
ipdb> quit
```

### Workspace Management

IPython helps you manage your computational workspace effectively:

```python
# See what variables you have
In [14]: %who
data     np

In [15]: %whos
Variable   Type       Data/Info
--------------------------------
data       ndarray    1000: 1000 elems, type `float64`
np         module     <module 'numpy' from '...'>

# Clear everything for a fresh start
In [16]: %reset
Once deleted, variables cannot be recovered. Proceed (y/[n])? y

# Save variables for later
In [17]: important_result = 42
In [18]: %store important_result
Stored 'important_result' (int)

# In a new IPython session:
In [1]: %store -r important_result
In [2]: important_result
Out[2]: 42
```

### Quick Documentation Access

IPython makes learning and exploration seamless:

```python
# Quick help with ?
In [19]: np.linspace?
# Shows function signature and documentation

In [20]: np.linspace??
# Shows the actual source code!

# General help
In [21]: %quickref
# Shows IPython quick reference
```

**🔍 Check Your Understanding**: Launch IPython and try these tasks:
1. Time how long it takes to sum the squares of numbers from 1 to 10000
2. Use tab completion to find all NumPy functions that contain "rand"
3. Use `?` to read the documentation for `np.random.randn`

<details>
<summary>Solutions</summary>

```python
# 1. Timing sum of squares
%timeit sum(i**2 for i in range(1, 10001))

# 2. Finding 'rand' functions
np.rand<TAB>
# Shows: np.rand, np.randint, np.randn, np.random

# 3. Documentation
np.random.randn?
```

</details>

## 1.2 The Hidden Complexity of "Simple" Python

Consider this innocent-looking code:

```python
import numpy as np
data = np.loadtxt('observations.dat')
result = np.mean(data)
```

This depends on finding the right NumPy (among possibly several installed versions), locating the data file (relative to where?), and numerous hidden assumptions. When you type `import numpy`, Python searches through a list of directories in a specific order, takes the first match it finds, and loads it.

Here's how to see what's actually happening:

```python
import sys
import numpy as np

print(f"Python executable: {sys.executable}")
print(f"NumPy location: {np.__file__}")
print(f"NumPy version: {np.__version__}")
print("\nPython searches these paths (in order):")
for i, path in enumerate(sys.path, 1):
    print(f"  {i}. {path}")
```

Let's visualize how Python finds NumPy when you import:

```
Visual: How Python Finds NumPy When You Import

Your Code:                 Python's Search Process:
import numpy as np         
                          1. Check sys.path[0]: /current/directory
                             ❌ No numpy here
                          
                          2. Check sys.path[1]: /home/user/.../site-packages
                             ✓ Found numpy/__init__.py!
                          
                          3. Load numpy and create 'np' reference
                             np ──→ [numpy module object in memory]

Multiple Python Installations Problem:
System Python:    /usr/lib/python3.9/site-packages/numpy (v1.19)
Conda Python:     /home/user/miniforge3/envs/astr596/lib/python3.11/site-packages/numpy (v1.24)
                  ↑
Which one?        Your choice depends on which python executable runs!
```

Run this diagnostic with and without your conda environment activated. Different Pythons, different NumPys, different results. This is why `conda activate astr596` is so crucial.

## 1.3 Jupyter Notebooks: Power and Peril

Jupyter notebooks are excellent for initial exploration, but they have hidden dangers that can corrupt your scientific computing. Understanding these pitfalls is crucial for using notebooks effectively in Project 1 and knowing when to abandon them.

### Starting and Using Jupyter

```bash
conda activate astr596
jupyter lab                  # Opens in browser
```

In Jupyter, you write code in cells and run them with Shift+Enter. The power comes from keeping variables in memory between cells:

```python
# Cell 1
import numpy as np
data = np.random.randn(1000)

# Cell 2 (run separately)
print(f"Mean: {data.mean():.3f}")

# Cell 3 (can go back and change Cell 1, but data persists)
print(f"Still have data: {len(data)} points")
```

### The Hidden State Problem

Here's why notebooks can be treacherous for scientific computing:

```python
# Cell 1: Define a constant
GRAVITY = 9.8

# Cell 2: Use it in a function
def calculate_fall_time(height):
    return np.sqrt(2 * height / GRAVITY)

# Cell 3: Oops, accidentally change it
GRAVITY = 9.81  # More precise value

# Cell 4: Function now uses new value!
time = calculate_fall_time(100)  # Which GRAVITY is this using?
```

### Memory Leaks and Accumulation

A critical problem students don't realize: repeatedly running cells can exhaust your system memory.

```python
# Cell 1: Looks innocent enough
results = []

# Cell 2: Accumulate results
for i in range(1000):
    results.append(np.random.randn(10000))
    
# Run Cell 2 multiple times by accident...
# Each run ADDS another 1000 arrays!
# Memory usage grows: 80MB, 160MB, 240MB...
```

This is particularly dangerous when developing iterative algorithms:

```python
# Cell 1: Initialize
convergence_history = []
current_value = 100

# Cell 2: Iterate (you run this cell repeatedly while debugging)
for iteration in range(100):
    current_value = current_value * 0.99
    convergence_history.append(current_value)

# After running Cell 2 five times:
len(convergence_history)  # 500! Not 100!
```

### The Psychological Trap: Fear of Deletion

Students become afraid to delete cells, thinking they might need that code later. This leads to notebooks with 200+ cells where only 20 do actual work. This isn't just messy — it's actively harmful:

```python
# Cell 43: First attempt at algorithm (doesn't work)
def integrate_orbit_v1(state, dt):
    # ... buggy code ...

# Cell 67: Second attempt (also doesn't work)
def integrate_orbit_v2(state, dt):
    # ... different bugs ...

# Cell 134: Working version (but which one are you actually using?)
def integrate_orbit(state, dt):
    # ... maybe works ...

# Cell 178: Wait, which function does this call?
result = integrate_orbit(initial_state, 0.01)
```

### Out-of-Order Execution Chaos

The most insidious problem: notebooks that only work when run in a specific order that you've forgotten:

```python
# You actually ran these in order: Cell 5, Cell 2, Cell 8, Cell 1, Cell 4
# But reading top-to-bottom suggests: Cell 1, Cell 2, Cell 4, Cell 5, Cell 8
# Someone else (or you next week) will get completely different results!
```

**🔍 Check Your Understanding: Notebook State**

Before continuing, predict what will print in this notebook scenario:

```
Cell 1: x = 10
Cell 2: y = x * 2
Cell 3: x = 5
Cell 4: print(y)
```

What value prints if you run cells in order 1, 3, 2, 4?

<details>
<summary>Click for answer</summary>

Answer: 10 (not 20!)

Execution order:
- Cell 1: x = 10
- Cell 3: x = 5 (overwrites x)
- Cell 2: y = x * 2 = 5 * 2 = 10
- Cell 4: prints 10

This demonstrates why out-of-order execution corrupts results!

</details>

### Why We're Abandoning Notebooks After Project 1

Beyond the state and memory problems, notebooks have fundamental limitations for serious computational work:

**Version Control Nightmare**: Notebooks are JSON files that include output, making git diffs unreadable and merges nearly impossible. You can't effectively collaborate or track changes.

**Testing Impossibility**: You can't write unit tests for notebook code or set up continuous integration. How do you know your algorithm still works after changes?

**Performance Issues**: Notebooks add overhead and make debugging harder. When your simulation crashes after 6 hours, you need real debugging tools, not cell-by-cell execution.

**No Modularity**: You can't import functions from notebooks easily. Code reuse requires copy-paste, which leads to divergent versions and bugs.

## 1.4 Scripts and the Terminal: Reproducible Science

Python scripts are text files containing Python code, run from top to bottom, exactly the same way every time. No hidden state, no out-of-order execution, no ambiguity.

### Building a Script Step by Step

Let's progress from IPython experimentation to a proper script:

**Step 1: Experiment in IPython**
```python
In [1]: import numpy as np

In [2]: data = np.array([1.2, 2.3, 3.4, 4.5])

In [3]: data.mean()
Out[3]: 2.85

In [4]: data.std()
Out[4]: 1.2747548783981961
```

**Step 2: Collect working code**
```python
# analysis.py - first version
import numpy as np

data = np.array([1.2, 2.3, 3.4, 4.5])
print(f"Mean: {data.mean()}")
print(f"Std: {data.std()}")
```

**Step 3: Make it reusable**
```python
# analysis.py - improved version
import numpy as np

def analyze_data(values):
    """Calculate statistics for data."""
    data = np.array(values)
    return {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max()
    }

if __name__ == '__main__':
    test_data = [1.2, 2.3, 3.4, 4.5]
    results = analyze_data(test_data)
    print(f"Mean: {results['mean']:.3f}")
    print(f"Std: {results['std']:.3f}")
```

### The if __name__ == '__main__' Pattern

This pattern makes your script both runnable and importable:

```python
# When you run: python analysis.py
# __name__ is set to '__main__', so the test code runs

# When you import: import analysis
# __name__ is set to 'analysis', so the test code doesn't run
# You can use: results = analysis.analyze_data(my_data)
```

## 1.5 How Python Finds Your Code

When you write `import numpy` or `from mymodule import function`, Python searches `sys.path` in order:

```python
import sys
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
```

Common import problems and solutions:

**Problem**: `ModuleNotFoundError: No module named 'myanalysis'`

```python
# Diagnosis script
import sys
from pathlib import Path

def diagnose_import():
    print(f"Running Python: {sys.executable}")
    print(f"Current directory: {Path.cwd()}")
    print("\nPython will search:")
    for p in sys.path[:5]:  # First 5 paths
        print(f"  {p}")
    
    # Check if you're in the right environment
    if 'astr596' not in sys.executable:
        print("\n⚠️  Not in astr596 environment!")
        print("Fix: conda activate astr596")

diagnose_import()
```

**🔧 Real Debugging Scenario**

You run your script and get:
```
ModuleNotFoundError: No module named 'astropy'
```

Your debugging workflow:
1. Check environment: `which python`
2. If wrong: `conda activate astr596`
3. If right but missing: `conda install astropy`

## 1.6 Making Your Code Reproducible

Reproducibility means someone else (or future you) can run your code and get the same results. This requires controlling your environment, paths, and random seeds.

### Control Randomness

```python
import numpy as np

def setup_reproducibility(seed=42):
    """Ensure reproducible results."""
    np.random.seed(seed)  # Legacy but still common
    
    # Modern NumPy way (preferred)
    rng = np.random.default_rng(seed)
    return rng

# Use throughout your code
rng = setup_reproducibility(42)
data = rng.standard_normal(1000)  # Always same "random" numbers
```

### Document Your Environment

Create `environment.yml`:
```yaml
name: astr596
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=1.24
  - scipy=1.11
  - matplotlib=3.7
  - astropy=5.3
```

Others recreate with:
```bash
conda env create -f environment.yml
```

### Use Paths Properly

```python
from pathlib import Path

# Bad - breaks on different systems
data = np.loadtxt('/Users/yourname/project/data.txt')

# Good - relative to script location
data_file = Path(__file__).parent / 'data' / 'observations.txt'
data = np.loadtxt(data_file)
```

## 1.7 Essential Debugging Strategies

When things break (they will), systematic debugging saves hours.

### The Universal First Check

```python
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

if 'astr596' not in sys.executable:
    print("⚠️  Wrong environment! Run: conda activate astr596")
```

### Quick Testing Workflow

Use IPython for rapid testing before writing scripts:

```python
In [1]: # Test your algorithm idea quickly
In [2]: def orbital_period(a, M):
   ...:     import math
   ...:     G = 6.67e-8
   ...:     return 2 * math.pi * math.sqrt(a**3 / (G * M))

In [3]: # Test with Earth
In [4]: P = orbital_period(1.496e13, 1.989e33)
In [5]: P / 86400  # Convert to days
Out[5]: 365.25  # Looks right!

In [6]: %save orbital_calc.py 1-5  # Save successful exploration
```

## Practice Exercises

### Exercise 1.1: IPython Proficiency

Complete these tasks in IPython:

```python
# 1. Time these two approaches to sum squares:
#    a) sum(i**2 for i in range(10000))
#    b) (np.arange(10000)**2).sum()

# 2. Use %run to execute a script while keeping variables

# 3. Use %debug to investigate why this fails:
def divide_list(numbers):
    return [10/n for n in numbers]
    
test = [1, 2, 0, 4]
result = divide_list(test)
```

### Exercise 1.2: Notebook to Script

Convert this problematic notebook pattern to a clean script:

```python
# Notebook cells (out of order execution):
# Cell 3: data = process_data(raw_data)
# Cell 1: import numpy as np
# Cell 5: results = analyze(data)
# Cell 2: def process_data(x): return x * 2
# Cell 4: def analyze(x): return x.mean()

# Your task: Create a proper script that always works
```

### Exercise 1.3: Environment Detective

Write a diagnostic function that reports:
- Current Python version and location
- Whether you're in the correct conda environment
- List of installed scientific packages with versions
- Current working directory
- First 5 paths Python searches for imports

## Key Takeaways

IPython transforms the terminal into a powerful computational laboratory. Use it for exploration, testing, and quick calculations. The magic commands like `%timeit` and `%debug` are essential tools for scientific computing.

Jupyter notebooks are powerful but dangerous. Hidden state, memory accumulation, and out-of-order execution can corrupt your scientific results. After Project 1, we move to scripts for reproducibility.

Scripts enforce linear execution and reproducibility. The `if __name__ == '__main__'` pattern makes code both runnable and importable. This is how professional scientific software is written.

Understanding how Python finds code prevents most import errors. When debugging, always check your environment first — it's usually the problem.

## Quick Reference: New Functions and Commands

| Command/Function | Purpose | Example |
|-----------------|---------|---------|
| `ipython` | Launch IPython | `$ ipython` |
| `%timeit` | Time code execution | `%timeit sum(range(1000))` |
| `%run` | Run script in IPython | `%run analysis.py` |
| `%debug` | Enter debugger after error | `%debug` |
| `%who` | List variables | `%who` |
| `%whos` | Detailed variable info | `%whos` |
| `%reset` | Clear all variables | `%reset -f` |
| `%store` | Save variables between sessions | `%store my_var` |
| `?` | Quick help | `np.array?` |
| `??` | Show source code | `np.array??` |
| `sys.executable` | Python interpreter path | `print(sys.executable)` |
| `sys.path` | Module search paths | `print(sys.path)` |
| `__file__` | Current script path | `Path(__file__)` |
| `Path.cwd()` | Current working directory | `Path.cwd()` |
| `id()` | Object memory address | `id(variable)` |

## Next Chapter Preview

Now that you understand your computational environment and can work effectively in IPython, Chapter 2 will explore Python as a scientific calculator. You'll learn how computers represent numbers, why 0.1 + 0.2 ≠ 0.3, and how to handle the numerical precision issues that appear in every computational physics calculation. These fundamentals are crucial — small numerical errors can compound into completely wrong results in your simulations.