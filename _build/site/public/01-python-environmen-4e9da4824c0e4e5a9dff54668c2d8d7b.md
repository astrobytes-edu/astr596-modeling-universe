---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Chapter 1: Computational Environments & Scientific Workflows

## Learning Objectives

By the end of this chapter, you will be able to:

- Configure and navigate `IPython` as your primary interactive computing environment.
- Diagnose why identical code produces different results on different machines.
- Explain how Python locates and loads code when you type `import`.
- Identify the hidden dangers of Jupyter notebooks that corrupt scientific results.
- Create reproducible computational environments using `conda`.
- Debug environment problems systematically using diagnostic tools.
- Transform notebook explorations into reproducible Python scripts.
- Execute Python code effectively from the terminal.

## Prerequisites Check

```{note} ✅ Before Starting This Chapter
- [ ] You have completed the {doc}`../getting-started/index` module (`basic setup`, `CLI`, `git`).
- [ ] You can navigate directories using `cd`, `ls`, and `pwd`.
- [ ] You have Miniforge installed with the `astr596` environment created.
- [ ] You can activate your conda environment: `conda activate astr596`.
- [ ] You understand file paths (absolute vs. relative).

If any boxes are unchecked, review the {doc}`../getting-started/index` module first.
```

## Chapter Overview

```{margin}
**environment** (Interactive Python) ADD DEF
```

Picture this: You download code from a groundbreaking astronomy paper, eager to reproduce their results. You run it exactly as instructed. Instead of the published results, you get error messages, or worse — completely different numbers with no indication why. This frustrating scenario happens to nearly every computational scientist, from undergraduates to professors. The problem isn't bad code or user error; it's that scientific computing happens in complex **environments** where tiny differences cascade into complete failures.

This chapter reveals the hidden machinery that makes Python work (or not work) on your computer. You'll discover why the same code produces different results on different machines, master IPython as your computational laboratory, understand the dangers of Jupyter notebooks, and learn to create truly reproducible computational environments. These aren't just technical skills — they're the foundation of trustworthy computational science.

By chapter's end, you'll transform from someone who hopes code works to someone who knows exactly why it works (or doesn't). You'll diagnose "`module not found`" errors in seconds, create environments that work identically on any machine, and understand the critical difference between exploration and reproducible science. Let's begin by exploring the tool that will become your new best friend: IPython.

## 1.1 IPython: Your Computational Laboratory

```{margin}
**IPython** (Interactive Python) is an enhanced version of the basic Python interpreter, designed specifically for scientific computing and data analysis.
```

While you could use the basic Python interpreter by typing `python`, **IPython** (just type `ipython` instead) transforms your terminal into a powerful environment for scientific exploration. Think of it as the difference between a basic calculator and a graphing calculator — both do math, but one is designed for serious work and is nicer to look at. Let's see why every professional computational scientist prefers `ipython` over `python`.

### Launching Your Laboratory

First, ensure you're in the right environment, then launch IPython:

```{code-cell} ipython3
# In your terminal (not in Python):
# First: conda activate astr596
# Then: ipython

# You'll see something like:
# Python 3.11.5 | packaged by conda-forge
# IPython 8.14.0 -- An enhanced Interactive Python
# In [1]: 

print("Note: This textbook simulates IPython features.")
print("In real IPython, you'll see 'In [1]:' prompts")
```

Notice the prompt says `In [1]:` instead of `>>>`. This numbering system is your first hint that IPython is different — it remembers everything. Just hit the up arrow to see previous `ipython` prompts.

### The Power of Memory

```{margin}
**Input/Output History**: IPython stores all inputs and outputs in special variables `In` and `Out`, making it easy to reference previous work.
```

```{note}
The following examples simulate IPython's behavior. In actual IPython, you would type these commands interactively and see immediate results.
```

IPython maintains a complete history of your session, accessible through special variables:

```ipython3
# Type these commands one at a time in IPython
import math # Python's built-in math module

radius = 6371  # Earth's radius in km

volume = (4/3) * math.pi * radius**3
print(f"Earth's volume: {volume:.2e} km³")
```

Now you can reference previous inputs and outputs:

```ipython3
# In real IPython, you could reference previous outputs:
# Out[3]  # Shows the volume calculation result
# In[2]   # Shows 'radius = 6371'
# _       # References the last output

# Here we simulate this behavior:
print("In IPython, Out[n] and In[n] store your history")
print("Example: Out[3] would contain the volume result")
print("Example: In[2] would contain 'radius = 6371'")
```

:::{hint} 🤔 Check Your Understanding
What's the difference between `In[5]` and `Out[5]` in IPython?
:::
:::{tip} Click for Answer
:class: dropdown

- `In[5]` contains the actual text/code you typed in cell 5 (as a string)
- `Out[5]` contains the result/value that cell 5 produced (if any)

For example:
- `In[5]` might be `"2 + 2"`
- `Out[5]` would be `4`

This history system lets you reference and reuse previous computations without retyping.
:::

### Tab Completion: Your Exploration Tool

Tab completion helps you discover what's available without memorizing everything. In IPython, try these examples:

```{code-cell} ipython3
# This demonstrates what happens with tab completion in IPython
import math

# In real IPython, you would type: math.<TAB>
# It shows all available functions like:
available_functions = [item for item in dir(math) if not item.startswith('_')]
print("math module contains:", available_functions[:10], "...")

# To see functions containing 'sin' (math.*sin*?<TAB> in IPython):
sin_functions = [item for item in dir(math) if 'sin' in item]
print("\nFunctions with 'sin':", sin_functions)
```

This feature is invaluable when exploring new libraries or trying to remember function names. It turns IPython into a self-documenting system.

### Magic Commands: IPython's Superpowers

:::{margin}
**Magic Commands**: Commands prefixed with `%` (line magics) or `%%` (cell magics) that provide functionality beyond standard Python.
:::

:::{note}
The following simulates IPython's `%timeit` magic command. In real IPython, you would simply type `%timeit` followed by your code for automatic statistical timing analysis.
:::

IPython's **"magic" commands** give you capabilities far beyond standard Python. Here's how timing works:

```{code-cell} ipython3
# In IPython, you would use: %timeit sum(range(1000))
# Here we simulate the comparison using Python's timeit module
import timeit

# Time list comprehension
time1 = timeit.timeit('[i**2 for i in range(100)]', number=10000)
print(f"List comprehension: {time1*100:.4f} µs per loop")

# Time map/lambda approach
time2 = timeit.timeit('list(map(lambda x: x**2, range(100)))', number=10000)
print(f"Map with lambda: {time2*100:.4f} µs per loop")

# Show which is faster
print(f"\nList comprehension is {time2/time1:.1f}x faster")
print("\nIn IPython, %timeit provides mean ± std dev automatically")
```
:::{margin} **optimization**
Define optimization
:::

The actual IPython `%timeit` magic provides statistical analysis with standard deviation - crucial for performance testing (i.e., **optimization**).

:::{warning} 🚨 Common Bug Alert: Platform-Specific Timing

Timing results vary significantly between machines due to:
- CPU speed and architecture
- System load and background processes  
- Python version and compilation options

Never assume timing results from one machine apply to another. Always benchmark on your target system.
:::

### Getting Help Instantly

`IPython` makes documentation accessible without leaving your workflow:

```ipython3
import math

# In real IPython, you'd use: math.sqrt?
# This shows the documentation instantly
print("In IPython, use ? for quick help:")
print("  math.sqrt?  - shows documentation")
print("  math.sqrt?? - shows source code (if available)")
print("\nExample documentation for math.sqrt:")
print("  Return the square root of x.")
print("  Domain: x ≥ 0, Range: result ≥ 0")
```

:::{margin} **Application Protocal Interface (API)**
Define API.
:::

:::{important} 💡 Computational Thinking: Interactive Exploration

The ability to quickly test ideas and explore **APIs** interactively is fundamental to computational science. `IPython`'s environment encourages experimentation. You can test a hypothesis, examine results, and refine your approach in seconds rather than minutes. This rapid iteration cycle is how algorithms are born and bugs are discovered.

This pattern appears everywhere: interactive debuggers, REPLs ([`Read–eval–print` loop](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)) in other languages, and even computational notebooks all follow this `explore-test-refine` cycle.
:::

### Managing Your Workspace

As you work, IPython helps you track what you've created:

```{code-cell} ipython3
# Simulating IPython's %who and %whos commands
import sys

# Create some variables for demonstration
data = [1, 2, 3, 4, 5]
result = sum(data)
name = "Earth"

# Show variables (simulating %who in IPython)
current_vars = [var for var in dir() 
                if not var.startswith('_') and var not in ['sys', 'inspect', 'timeit', 'math']]
print("Variables in workspace (%who in IPython):", current_vars)

# Detailed info (simulating %whos in IPython)
print("\nDetailed variable info (%whos in IPython):")
for var in current_vars[:3]:  # Show first 3
    obj = eval(var)
    print(f"  {var:10} {type(obj).__name__:10} {str(obj)[:30]}")

print("\nIn IPython, use %reset to clear all variables")
```

```{admonition} 🔬 Why This Matters: Research Reproducibility
:class: tip

In 2016, a study by Baker in *Nature* found that more than 70% of researchers failed to reproduce another scientist's experiments, and more than 50% failed to reproduce their own experiments [1]. Tools like IPython's `%history` and `%save` commands let you save entire sessions, ensuring you can always trace back exactly what you did to get a result.

[1] Baker, M. (2016). "1,500 scientists lift the lid on reproducibility." *Nature*, 533(7604), 452-454.
```

## 1.2 Understanding Python's Hidden Machinery

When you type a simple line like `import math`, a complex process unfolds behind the scenes. Understanding this machinery is the difference between guessing why code fails and knowing exactly how to fix it.

### The Import System Exposed

```{margin}
**Importing**: Loading Python code from external files (modules) into your current program, making their functions available for use.
```

Let's peek behind the curtain:

```{code-cell} ipython3
import sys
from pathlib import Path

# Where is Python running from?
print(f"Python executable: {sys.executable}")

# What version are we using?
print(f"Python version: {sys.version.split()[0]}")

# Where will Python look for code?
print("\nPython searches these locations (in order):")
for i, path in enumerate(sys.path[:5], 1):
    # Shorten paths for readability
    display_path = str(path).replace(str(Path.home()), "~")
    print(f"  {i}. {display_path}")

print("  ... and more")
```

This search path determines everything. When you `import something`, Python checks each directory in order and uses the first match it finds.

### Debugging Import Problems

Here's a diagnostic function you'll use throughout your career:

```{code-cell} ipython3
def diagnose_import(module_name):
    """Diagnose why a module can't be imported.
    
    Args:
        module_name: Name of the module to diagnose
        
    Returns:
        bool: True if module imports successfully, False otherwise
    """
    import sys
    from pathlib import Path
    
    print(f"Diagnosing import for: {module_name}")
    print(f"Python: {sys.executable}")
    
    # Extract environment name from path (if in conda environment)
    path_parts = sys.executable.split('/')
    if 'envs' in path_parts:
        env_idx = path_parts.index('envs')
        env_name = path_parts[env_idx + 1] if env_idx + 1 < len(path_parts) else "unknown"
        print(f"Environment: {env_name}")
    
    # Try to import the module
    try:
        module = __import__(module_name)
        # Check if it's a built-in or has a file location
        if hasattr(module, '__file__'):
            print(f"✓ Found at: {module.__file__}")
        else:
            print(f"✓ Found (built-in module)")
        return True
    except ImportError as e:
        print(f"✗ Not found: {e}")
        print("\nPython searched these locations:")
        # Show first 3 search paths for debugging
        for p in sys.path[:3]:
            display_p = str(p).replace(str(Path.home()), "~")
            print(f"  - {display_p}")
        return False

# Test with standard library module (should work)
diagnose_import('math')
print("\n" + "="*50 + "\n")
# Test with scientific package (might not be installed)
diagnose_import('numpy')
```

```{hint} 🤔 Check Your Understanding
You get `ModuleNotFoundError: No module named 'astropy'`. What are three possible causes?

```{tip} Click for Answer
:class: dropdown

1. **Wrong environment**: You're not in the conda environment where astropy is installed
2. **Not installed**: Astropy isn't installed in the current environment  
3. **Path issues**: Python's sys.path doesn't include the directory containing astropy

To diagnose, check:
```bash
which python          # Are you using the right Python?
conda list astropy    # Is it installed?
python -c "import sys; print(sys.path)"  # Where is Python looking?
```

The most common cause is forgetting to activate your conda environment!
```
```

### Multiple Pythons: A Common Disaster

Most systems have multiple Python installations, leading to confusion:

```{code-cell} ipython3
from pathlib import Path

# Common Python locations on Unix-like systems
possible_pythons = [
    '/usr/bin/python3',          # System Python
    '/usr/local/bin/python3',    # Homebrew Python (Mac)
    '~/miniforge3/bin/python',   # Conda Python
    '~/.pyenv/shims/python',     # Pyenv Python
    '/opt/python/bin/python',    # Custom installation
]

print("Potential Python locations on Unix-like systems:")
for path in possible_pythons:
    # Expand ~ to home directory and check existence
    expanded_path = Path(path).expanduser()
    exists = "✓" if expanded_path.exists() else "✗"
    print(f"  {exists} {path}")

print("\nThis is why 'conda activate' is crucial!")
print("It ensures you're using the right Python with the right packages.")
```

```{admonition} 🚨 Common Bug Alert: The Wrong Python
:class: warning

**Symptom**: Code works in terminal but fails in IDE, or vice versa

**Cause**: Different tools using different Python installations

**Fix**: Always verify with:
```bash
which python       # Unix/Mac
where python       # Windows
```

**Prevention**: Always activate your conda environment first:
```bash
conda activate astr596
```
```

## 1.3 Jupyter Notebooks: Beautiful Disasters Waiting to Happen

Jupyter notebooks seem perfect for scientific computing—you can mix code, results, and explanations in one document. They're widely used and seemingly convenient. However, they harbor dangerous flaws that can corrupt your scientific results. You'll use them for Project 1 to understand their appeal, then abandon them for more robust approaches.

### The Seductive Power of Notebooks

```{margin}
**Jupyter**: A web-based interactive computing platform that runs code in "cells" while maintaining results between executions.
```

To start Jupyter (after activating your environment):

```{code-cell} ipython3
# In terminal:
# conda activate astr596
# jupyter lab

# This opens a browser with the Jupyter interface
print("Jupyter Lab would open at: http://localhost:8888")
print("You can create notebooks, write code in cells, and see results inline")
```

In a notebook, you write code in cells and run them individually. This seems wonderful—immediate feedback, ability to modify and re-run. But this flexibility is exactly what makes notebooks dangerous.

### The Hidden State Monster

The most insidious problem: notebooks maintain hidden state between cell executions. Consider this experiment:

```{code-cell} ipython3
# Simulating notebook cells with their execution order problems

# Cell 1 (first execution)
gravity = 9.8
print(f"Cell 1: Set gravity = {gravity}")

# Cell 2 (depends on gravity)
import math
def calculate_fall_time(height):
    """Calculate fall time using gravity defined when function was created."""
    return math.sqrt(2 * height / gravity)

print(f"Cell 2: Defined function with gravity = {gravity}")

# Cell 3 (changes gravity)
gravity = 3.71  # Mars gravity
print(f"Cell 3: Changed gravity = {gravity}")

# Cell 4 (which gravity does this use?)
time = calculate_fall_time(100)
print(f"Cell 4: Fall time = {time:.2f} seconds")
print(f"  But function still uses gravity = 9.8 from when it was defined!")
print(f"  This hidden state causes wrong results!")
```

```{attention} 💥 Debug This!
A student's notebook has these cells:

```python
Cell 1: data = [1, 2, 3]
Cell 2: result = sum(data) / len(data)  
Cell 3: data.append(4)
Cell 4: print(f"Average: {result}")
```

They run cells in order: 1, 2, 3, 4, 2, 4. What prints the second time?

**Think before clicking!**

```{tip} Solution
:class: dropdown

The second execution of Cell 4 prints: `Average: 2.5`

Here's the execution trace:
1. Cell 1: `data = [1, 2, 3]`
2. Cell 2: `result = 2.0` (sum=6, len=3)
3. Cell 3: `data = [1, 2, 3, 4]`
4. Cell 4: Prints `"Average: 2.0"`
5. Cell 2 again: `result = 2.5` (sum=10, len=4)
6. Cell 4 again: Prints `"Average: 2.5"`

This demonstrates how re-running cells creates different states than sequential execution—a recipe for irreproducible results!
```
```

### Memory Accumulation Disasters

Notebooks can secretly consume gigabytes of memory:

```{code-cell} ipython3
import sys

# Simulating repeated cell execution
big_data = []

print("Initial memory state")

# First run of the cell
for i in range(100):
    big_data.append([0] * 1000)
    
# Calculate approximate memory usage
size_mb = sys.getsizeof(big_data) / (1024 * 1024)
print(f"After 1st run: ~{size_mb:.1f} MB")

# Second run (accumulates!)
for i in range(100):
    big_data.append([0] * 1000)
    
size_mb = sys.getsizeof(big_data) / (1024 * 1024)
print(f"After 2nd run: ~{size_mb:.1f} MB")

print("\nEach run ADDS to memory usage - notebooks don't reset!")
print("After 10 runs, you could be using 10× the memory!")
```

### The Out-of-Order Execution Trap

The deadliest notebook sin: cells that only work when run in a specific order that you've forgotten:

```{code-cell} ipython3
# Demonstrating execution order confusion
print("Notebook shows cells in order: 1, 2, 3, 4, 5")
print("You actually ran them: 5, 2, 1, 4, 3")
print("New user runs them: 1, 2, 3, 4, 5")
print("Result: Complete failure with cryptic errors!")
print("\nWorse: The 'correct' order isn't documented anywhere")
print("The notebook looks clean but hides a chaotic execution history")
```

```{admonition} 🔬 Why This Matters: The Reinhart-Rogoff Excel Error
:class: tip

In 2013, graduate student Thomas Herndon discovered a critical error in an influential economics paper by Reinhart and Rogoff that had been used to justify austerity policies worldwide [2]. The authors had accidentally excluded several countries from their Excel calculations. Like notebook state problems, the error was invisible in the final spreadsheet. This coding error influenced global economic policy affecting millions of people.

Notebooks have the same danger: the document you share may not reflect the actual execution that produced your results.

[2] Herndon, T., Ash, M., & Pollin, R. (2014). "Does high public debt consistently stifle economic growth? A critique of Reinhart and Rogoff." *Cambridge Journal of Economics*, 38(2), 257-279.
```

### The Notebook-to-Script Transition

After Project 1, we'll abandon notebooks for scripts. Here's why scripts are superior for real scientific computing:

| Aspect | Notebooks | Scripts |
|--------|-----------|---------|
| **Execution Order** | Ambiguous, user-determined | Top-to-bottom, always |
| **Hidden State** | Accumulates invisibly | Fresh start each run |
| **Version Control** | JSON mess with outputs | Clean text diffs |
| **Testing** | Nearly impossible | Straightforward |
| **Debugging** | Cell-by-cell only | Professional tools |
| **Collaboration** | Merge conflicts | Standard git workflow |
| **Performance** | Overhead and lag | Direct execution |
| **Reproducibility** | Often impossible | Guaranteed |

```{admonition} 💡 Computational Thinking: Reproducible by Design
:class: important

Reproducibility isn't just about sharing code—it's about ensuring that code produces identical results regardless of who runs it or when. Scripts enforce this by eliminating hidden state and ambiguous execution order. This principle extends beyond Python: declarative configurations, containerization, and infrastructure-as-code all follow the same philosophy of explicit, reproducible computation.

The mantra: "It should work the same way every time, for everyone."
```

## 1.4 Scripts: Write Once, Run Anywhere (Correctly)

Python scripts are simple text files containing Python code, executed from top to bottom, the same way every time. No hidden state, no ambiguity, just predictable execution. Let's build your first robust script.

### From IPython to Script

Start by experimenting in IPython:

```{code-cell} ipython3
# Quick calculation in IPython
earth_mass = 5.97e24  # kg
moon_mass = 7.35e22   # kg
ratio = earth_mass / moon_mass
print(f"Earth is {ratio:.1f}× more massive than the Moon")
```

Now let's create a proper script. Save this as `mass_ratio.py`:

```{code-cell} ipython3
# This shows what would be in mass_ratio.py
script_content = '''#!/usr/bin/env python
"""Calculate mass ratios between celestial bodies."""

# Constants (kg) - uppercase names indicate constants
EARTH_MASS = 5.97e24
MOON_MASS = 7.35e22
SUN_MASS = 1.99e30

def calculate_ratio(mass1, mass2):
    """Calculate mass ratio between two bodies.
    
    Args:
        mass1: Mass of first body (kg)
        mass2: Mass of second body (kg)
        
    Returns:
        float: Ratio of mass1 to mass2
        
    Raises:
        ValueError: If mass2 is zero
    """
    # Defensive programming: check for division by zero
    if mass2 == 0:
        raise ValueError("Cannot divide by zero mass")
    return mass1 / mass2

def main():
    """Main execution function."""
    # Earth to Moon ratio
    earth_moon = calculate_ratio(EARTH_MASS, MOON_MASS)
    print(f"Earth is {earth_moon:.1f}× more massive than the Moon")
    
    # Sun to Earth ratio
    sun_earth = calculate_ratio(SUN_MASS, EARTH_MASS)
    print(f"Sun is {sun_earth:.0f}× more massive than Earth")

# This pattern makes the script both runnable and importable
if __name__ == "__main__":
    main()
'''

print("Script content (save as mass_ratio.py):")
print(script_content)

# Execute just the functions to show results
exec(script_content.split('if __name__')[0] + "main()")
```

Run it from the terminal with: `python mass_ratio.py`

### The `if __name__ == "__main__"` Pattern

```{margin}
**`__name__`**: Python sets this to `"__main__"` when running a file directly, but to the module name when importing.
```

This crucial pattern makes your code both runnable and importable:

```{code-cell} ipython3
# Demonstrating the __name__ pattern
test_code = '''
def useful_function(x):
    """A function others might want to use."""
    return x ** 2

# This print shows what __name__ contains
print(f"Module's __name__ is: {__name__}")

if __name__ == "__main__":
    # This only runs when executed directly
    print("Running as a script!")
    result = useful_function(5)
    print(f"5 squared is {result}")
'''

# When run directly
print("When run as a script:")
exec(test_code)

print("\n" + "="*50)
print("When imported, the test code doesn't run")
print("But the function is still available for use")
```

```{hint} 🤔 Check Your Understanding
Why would you want code that behaves differently when imported versus run directly?

```{tip} Click for Answer
:class: dropdown

This pattern serves multiple purposes:

1. **Testing**: Include test code that runs when developing but not when others use your functions
2. **Reusability**: Others can import your functions without triggering test/demo code
3. **Library Design**: Create modules that work both as tools and standalone programs
4. **Development**: Test functions immediately while writing them

Example: A module calculating orbital periods could be imported by other code OR run directly to calculate specific examples.

```python
# orbital_mechanics.py
def orbital_period(a, M):
    # ... calculation ...
    return period

if __name__ == "__main__":
    # Test with Earth's orbit
    period = orbital_period(1.496e11, 1.989e30)
    print(f"Earth's period: {period/86400:.1f} days")
```
```
```

## 1.5 Creating Reproducible Environments

Your code's behavior depends on its environment—Python version, installed packages, even operating system. Creating reproducible environments ensures your code works identically everywhere.

### The Conda Solution

```{margin}
**Conda**: A package and environment manager that creates isolated Python installations with specific package versions.
```

Conda creates isolated environments—separate Python installations with their own packages:

```{code-cell} ipython3
# Commands you would run in terminal (not Python)
print("""Essential conda commands:

# Create new environment with specific Python version
conda create -n myproject python=3.11

# Activate environment (ALWAYS do this first!)
conda activate myproject

# Install packages
conda install numpy scipy matplotlib

# List installed packages
conda list

# Deactivate when done
conda deactivate

# Remove environment completely
conda env remove -n myproject
""")
```

### Environment Files: Reproducibility in Practice

Create an `environment.yml` file that others can use to recreate your exact setup:

```{code-cell} ipython3
environment_yml = """name: astr596_project
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=1.24
  - scipy=1.11
  - matplotlib=3.7
  - ipython
  - jupyter
  - pip
  - pip:
    - astroquery==0.4.6
"""

print("environment.yml content:")
print(environment_yml)

print("\nOthers recreate your environment with:")
print("conda env create -f environment.yml")
print("conda activate astr596_project")
```

```{admonition} 🚨 Common Bug Alert: Channel Confusion
:class: warning

**Problem**: Package not found or wrong version installed

**Cause**: Different conda channels have different package versions

**Solution**: Always specify channels in environment.yml

**Best Practice**: Use `conda-forge` channel for scientific packages—it's community-maintained and has the most up-to-date scientific software

```yaml
channels:
  - conda-forge  # Always list this first
  - defaults     # Fallback to defaults if needed
```
```

### Proper Path Management

Stop hardcoding paths that break on other systems:

```{code-cell} ipython3
from pathlib import Path
import os

# BAD: Only works on your machine
bad_path = '/Users/yourname/research/data.txt'
print(f"BAD: Hardcoded path: {bad_path}")

# GOOD: Works everywhere (relative to script)
# Note: __file__ would be the script's path in a real script
script_dir = Path.cwd()  # Using current dir for demo
data_file = script_dir / 'data' / 'observations.txt'
print(f"GOOD: Relative path: {data_file}")

# BETTER: Handle missing files gracefully
if data_file.exists():
    print(f"  ✓ Found data at: {data_file}")
else:
    print(f"  ✗ Data not found at: {data_file}")
    print(f"    (Expected - this is just a demo)")

# BEST: Use configuration with environment variables
data_dir = Path(os.getenv('DATA_DIR', './data'))
print(f"BEST: Configurable path: {data_dir}")
```

### Random Seed Control

Make "random" results reproducible:

```{code-cell} ipython3
import random

def reproducible_random(seed=42):
    """Generate reproducible random numbers.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        list: Five random numbers (always same for same seed)
    """
    # Set the seed for reproducibility
    random.seed(seed)
    
    # These will be the same every time with same seed
    values = [random.random() for _ in range(5)]
    return values

# Run multiple times - same results with same seed
print("First run: ", [f"{x:.3f}" for x in reproducible_random(42)])
print("Second run:", [f"{x:.3f}" for x in reproducible_random(42)])

# Different seed = different results
print("New seed:  ", [f"{x:.3f}" for x in reproducible_random(137)])

print("\nAlways document your random seeds in papers!")
```

```{admonition} 🔬 Why This Matters: The LIGO Discovery
:class: tip

When LIGO detected gravitational waves in 2015, skeptics worldwide wanted to verify the analysis. The LIGO Scientific Collaboration provided their exact environment specifications, analysis scripts, and used seeded random number generation for their noise analysis [3]. Scientists globally could reproduce the Nobel Prize-winning analysis exactly. Without reproducible environments, this historic discovery might have been dismissed as a computational artifact.

[3] Abbott, B. P., et al. (2016). "Observation of gravitational waves from a binary black hole merger." *Physical Review Letters*, 116(6), 061102.
```

## 1.6 Essential Debugging Strategies

When code fails (and it will), systematic debugging saves hours of frustration. Here are strategies that work every time.

### The Universal First Check

Before anything else, verify your environment:

```{code-cell} ipython3
import sys
import os
from pathlib import Path

def environment_check():
    """Universal debugging first check.
    
    Returns:
        bool: True if in correct environment, False otherwise
    """
    print("=== Environment Debug Check ===")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Current dir: {os.getcwd()}")
    
    # Check for conda environment
    # Look for 'conda' or 'miniconda' in the Python path
    if 'conda' in sys.executable or 'miniconda' in sys.executable:
        # Extract environment name from path
        path_parts = sys.executable.split(os.sep)
        if 'envs' in path_parts:
            env_idx = path_parts.index('envs')
            env_name = path_parts[env_idx + 1] if env_idx + 1 < len(path_parts) else "base"
            
            # Check if it's the correct environment
            if 'astr596' in env_name:
                print(f"✓ Correct environment: {env_name}")
                return True
            else:
                print(f"✗ Wrong environment: {env_name}")
                print("  Fix: conda activate astr596")
                return False
    else:
        print("✗ Not in a conda environment")
        print("  Fix: conda activate astr596")
        return False

# Run the check
environment_check()
```

### Systematic Import Debugging

When imports fail, use this systematic approach:

```{code-cell} ipython3
import subprocess
import sys

def debug_import_error(module_name):
    """Systematically debug why an import fails.
    
    Args:
        module_name: Name of module to check
        
    Returns:
        bool: True if import succeeds, False otherwise
    """
    print(f"Debugging import: {module_name}")
    
    # Step 1: Show Python location
    print(f"\n1. Python: {sys.executable}")
    
    # Step 2: Try importing the module
    try:
        module = __import__(module_name)
        print(f"\n2. ✓ Import successful!")
        # Show where module is located if it has a file
        if hasattr(module, '__file__'):
            print(f"   Located at: {module.__file__}")
        return True
    except ImportError as e:
        print(f"\n2. ✗ Import failed: {e}")
    
    # Step 3: Check if package is installed using pip
    try:
        # Use pip to check if package is installed
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', module_name],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            print(f"\n3. Package is installed but can't import")
            print("   Likely wrong environment or corrupted install")
        else:
            print(f"\n3. Package not installed")
            print(f"   Fix: conda install {module_name}")
    except:
        print(f"\n3. Could not check installation status")
    
    return False

# Test with common modules
debug_import_error('math')  # Should always work
print("\n" + "="*50 + "\n")
debug_import_error('astropy')  # Might not be installed
```

### Using IPython's Debugger

When code crashes, IPython's `%debug` magic enters the debugger at the crash point:

```{code-cell} ipython3
# Example of code that would crash
def divide_list(numbers, divisor):
    """Divide all numbers in a list.
    
    Args:
        numbers: List of numbers to divide
        divisor: Number to divide by
        
    Returns:
        list: Results of division
    """
    # This will crash if divisor is zero
    return [n / divisor for n in numbers]

# This would crash with ZeroDivisionError:
# result = divide_list([1, 2, 3], 0)

print("""In IPython, after an error occurs, type: %debug

You'll enter the debugger with these commands:
  p variable  - print variable value
  l          - list code around error
  u/d        - go up/down the call stack  
  c          - continue execution
  n          - next line
  s          - step into function
  q          - quit debugger
  
Example debugger session:
  ipdb> p divisor
  0
  ipdb> p numbers
  [1, 2, 3]
  ipdb> l
  (shows code around the error)
  ipdb> q
""")
```

```{admonition} 💡 Computational Thinking: Defensive Programming
:class: important

Defensive programming means assuming things will go wrong and coding accordingly. Instead of hoping files exist, check first. Instead of assuming imports work, verify them. This mindset—expect failure, handle it gracefully—separates robust scientific code from scripts that work "sometimes."

This pattern appears everywhere: network requests that might timeout, sensors that might malfunction, or data that might be corrupted. Always ask: "What could go wrong here?"

Example:
```python
# Fragile code
data = open('file.txt').read()

# Defensive code
if Path('file.txt').exists():
    with open('file.txt') as f:
        data = f.read()
else:
    print("Warning: file.txt not found, using defaults")
    data = default_data
```
```

## Practice Exercises

### Exercise 1.1: IPython Mastery

Complete these IPython tasks to build proficiency:

```{code-cell} ipython3
# Part A: Timing Comparison
print("Part A: In IPython, compare these approaches:")
print("  %timeit sum([i**2 for i in range(1000)])")
print("  %timeit sum(map(lambda x: x**2, range(1000)))")
print("Which is faster? By how much?")

# Part B: Exploration Challenge  
print("\nPart B: Use tab completion to find:")
print("  All functions in math module containing 'log'")
print("  Hint: math.*log*?<TAB>")

# Part C: Documentation Discovery
print("\nPart C: Use ? and ?? to explore:")
print("  math.floor() vs math.trunc()")
print("  What's the difference for negative numbers?")
print("  Example: floor(-2.3) vs trunc(-2.3)")
```

### Exercise 1.2: Notebook State Detective

Given this notebook execution history, predict the final output:

```{code-cell} ipython3
print("""Execution order: Cell 1, Cell 3, Cell 2, Cell 4, Cell 2, Cell 4

Cell 1: counter = 0
        data = []

Cell 2: counter += 1
        data.append(counter)

Cell 3: counter = 10

Cell 4: print(f"Counter: {counter}, Data: {data}")

Work through the execution step by step.
What makes this confusing?
Why would this be hard to debug?
""")

# Solution trace (work it out first!)
solution = """
Execution trace:
1. Cell 1: counter=0, data=[]
2. Cell 3: counter=10, data=[]
3. Cell 2: counter=11, data=[11]
4. Cell 4: prints "Counter: 11, Data: [11]"
5. Cell 2: counter=12, data=[11, 12]
6. Cell 4: prints "Counter: 12, Data: [11, 12]"
"""
```

### Exercise 1.3: Environment Diagnostic Tool

Write a comprehensive diagnostic script:

```{code-cell} ipython3
def full_diagnostic():
    """Complete environment diagnostic.
    
    Generates a comprehensive report about the Python environment,
    installed packages, and system configuration.
    """
    import sys
    import subprocess
    from pathlib import Path
    import os
    
    print("=== Full Environment Diagnostic ===\n")
    
    # 1. Python version and location
    print(f"1. Python Version: {sys.version.split()[0]}")
    print(f"   Location: {sys.executable}")
    
    # 2. Active conda environment (if any)
    conda_env = "Not in conda"
    if 'conda' in sys.executable:
        # Parse environment name from path
        parts = sys.executable.split(os.sep)
        if 'envs' in parts:
            idx = parts.index('envs')
            conda_env = parts[idx + 1] if idx + 1 < len(parts) else "base"
    print(f"\n2. Conda Environment: {conda_env}")
    
    # 3. Check for key scientific packages
    print("\n3. Scientific Packages:")
    packages = ['numpy', 'scipy', 'matplotlib', 'pandas', 'astropy']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except:
            print(f"   ✗ {pkg}")
    
    # 4. Current working directory
    print(f"\n4. Working Directory: {os.getcwd()}")
    
    # 5. Python module search paths (first 5)
    print("\n5. Python Search Paths (first 5):")
    for i, path in enumerate(sys.path[:5], 1):
        # Replace home directory with ~ for readability
        display = str(path).replace(str(Path.home()), "~")
        print(f"   {i}. {display}")
    
    # 6. Total package count (bonus feature)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True, text=True, timeout=5
        )
        # Count lines minus header
        count = len(result.stdout.strip().split('\n')) - 2
        print(f"\n6. Total Installed Packages: {count}")
    except:
        print("\n6. Could not count packages")
    
    # Save report to file
    output_file = Path("diagnostic_report.txt")
    print(f"\n7. Report saved to: {output_file}")
    
    return True

# Run the diagnostic
full_diagnostic()
```

### Exercise 1.4: Script Conversion Challenge

Convert this problematic notebook pattern into a robust script:

```{code-cell} ipython3
# Here's the robust script solution
script_solution = '''
#!/usr/bin/env python
"""Robust data processing script replacing problematic notebook."""

from pathlib import Path
import sys

def read_data(filepath):
    """Read data from file with error handling.
    
    Args:
        filepath: Path to data file
        
    Returns:
        list: Lines from file, or None if error
    """
    filepath = Path(filepath)
    
    # Check if file exists before trying to read
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return None
    
    # Read with proper file handling
    with open(filepath) as f:
        data = f.read().splitlines()
    return data

def process_data(data):
    """Process data with validation.
    
    Args:
        data: List of data to process
        
    Returns:
        list: Processed data
    """
    # Validate input
    if not data:
        print("Warning: No data to process")
        return []
    
    # Your processing logic here
    processed = [line.upper() for line in data]
    return processed

def save_results(results, output_path):
    """Save results with overwrite protection.
    
    Args:
        results: Processed results to save
        output_path: Where to save results
        
    Returns:
        bool: True if saved successfully
    """
    output_path = Path(output_path)
    
    # Check for existing file
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Save cancelled")
            return False
    
    # Save with proper file handling
    with open(output_path, 'w') as f:
        f.write('\\n'.join(results))
    print(f"Results saved to {output_path}")
    return True

def main():
    """Main execution with proper flow."""
    # Always runs in correct order
    data = read_data('input.txt')
    if data:
        results = process_data(data)
        if results:
            save_results(results, 'output.txt')

# Standard pattern for scripts
if __name__ == "__main__":
    main()
'''

print("Robust script solution:")
print(script_solution)

print("\nKey improvements over notebooks:")
print("1. Linear execution - no ambiguity")
print("2. Error handling at each step")
print("3. No hidden state accumulation")
print("4. Can be imported or run standalone")
```

## Main Takeaways

This chapter has revealed the hidden complexity underlying every Python program you'll write. You've learned that when code fails to run or produces different results on different machines, it's rarely due to the code itself—it's the environment surrounding that code. Understanding this distinction transforms you from someone who gets frustrated by errors to someone who systematically diagnoses and fixes them.

IPython is more than just an enhanced Python prompt—it's a scientific laboratory where ideas become code. The ability to quickly test hypotheses, examine results, and iterate on solutions is fundamental to computational science. The magic commands like `%timeit` and `%debug` aren't just conveniences; they're essential tools that separate casual coding from professional scientific computing. Master IPython now, and you'll use it daily throughout your research career.

The Jupyter notebook trap is real and dangerous. While notebooks seem perfect for scientific work—mixing code, results, and narrative—their hidden state and execution ambiguity make them unsuitable for serious scientific computing. The out-of-order execution problem isn't a minor inconvenience; it's a fundamental flaw that can corrupt your scientific results. After Project 1, you'll leave notebooks behind for the reliability of scripts, but understanding their dangers now will help you recognize similar issues in other tools.

Scripts enforce reproducibility through simplicity. By executing top-to-bottom every time, they eliminate the ambiguity that plagues notebooks. The `if __name__ == "__main__"` pattern might seem like unnecessary boilerplate now, but it's the key to writing code that's both immediately useful and reusable by others. This pattern embodies a core principle: good scientific code serves multiple purposes without compromising any of them.

Creating reproducible environments isn't just about making your code run on other machines—it's about scientific integrity. When you can't reproduce your own results from six months ago, you've lost the thread of your research. The tools you've learned—conda environments, environment files, proper path handling, and seed control—aren't optional extras. They're the foundation of trustworthy computational science. Every major discovery in computational science, from gravitational waves to exoplanet detection, has depended on reproducible environments.

The debugging strategies you've learned will save you countless hours. The universal first check—verifying your environment—solves most "mysterious" errors. Systematic import debugging reveals exactly why modules can't be found. IPython's debugger lets you examine failures at the moment they occur. These aren't just troubleshooting techniques; they're the difference between guessing and knowing.

Remember: computational science isn't just about writing code that works once. It's about creating reliable, reproducible tools that advance human knowledge. The practices you've learned in this chapter—from IPython exploration to environment management—are the foundation of that reliability.

## Definitions

**Conda**: A package and environment management system that creates isolated Python installations with specific package versions, ensuring reproducibility across different machines.

**Defensive programming**: Writing code that anticipates and handles potential failures gracefully, rather than assuming everything will work correctly.

**Environment**: An isolated Python installation with its own interpreter, packages, and settings, preventing conflicts between projects with different requirements.

**Import**: The process of loading Python code from external files (modules) into your current program, making their functions and variables available for use.

**Input/Output history**: IPython's system of storing all commands (`In`) and their results (`Out`) in numbered variables for later reference.

**IPython**: Interactive Python—an enhanced version of the basic Python interpreter designed specifically for scientific computing, offering features like magic commands, tab completion, and integrated help.

**Jupyter**: A web-based interactive computing platform that allows you to create notebooks combining code, results, and text in a single document.

**Magic command**: Special IPython commands prefixed with `%` (line magics) or `%%` (cell magics) that provide functionality beyond standard Python, such as timing code or debugging.

**Module**: A Python file containing code (functions, classes, variables) that can be imported and used in other Python programs.

**Notebook**: A Jupyter document containing cells of code and text that can be executed individually, maintaining state between executions.

**Path**: The location of a file or directory in your filesystem, either absolute (full path from root) or relative (path from current location).

**Reproducibility**: The ability to obtain consistent results using the same data and code, regardless of who runs it or when.

**Script**: A plain text file containing Python code that executes from top to bottom when run, providing consistent and reproducible execution.

**sys.path**: Python's list of directories to search when importing modules, checked in order until a matching module is found.

**`__name__`**: A special Python variable that equals `"__main__"` when a script is run directly, or the module name when imported.

## Key Takeaways

✓ **IPython is your primary tool**: Use it for exploration, testing, and quick calculations—not the basic Python interpreter

✓ **Environment problems cause most "broken code"**: When code fails, check your environment first with `sys.executable`

✓ **Notebooks corrupt scientific computing**: Hidden state and out-of-order execution make results irreproducible

✓ **Scripts enforce reproducibility**: Top-to-bottom execution eliminates ambiguity and hidden state

✓ **The `__name__` pattern enables reusability**: Code can be both runnable and importable without modification

✓ **Conda environments isolate projects**: Each project gets its own Python and packages, preventing conflicts

✓ **Always specify package versions**: Use environment.yml files to ensure others can recreate your exact setup

✓ **Paths should be relative, not absolute**: Use `pathlib.Path` for cross-platform compatibility

✓ **Control randomness with seeds**: Set random seeds for reproducible "random" results

✓ **Systematic debugging saves time**: Check environment → verify imports → test incrementally

✓ **Defensive programming prevents disasters**: Assume things will fail and handle errors gracefully

## Quick Reference Tables

### Essential IPython Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `%timeit` | Time code execution | `%timeit sum(range(1000))` |
| `%run` | Run script keeping variables | `%run analysis.py` |
| `%debug` | Enter debugger after error | `%debug` |
| `%who` | List all variables | `%who` |
| `%whos` | Detailed variable information | `%whos` |
| `%reset` | Clear all variables | `%reset -f` |
| `%history` | Show command history | `%history -n 10` |
| `%save` | Save code to file | `%save script.py 1-10` |
| `%load` | Load code from file | `%load script.py` |
| `%magic` | List all magic commands | `%magic` |
| `?` | Quick help | `len?` |
| `??` | Show source code | `len??` |

### Environment Debugging Checklist

| Check | Command | What to Look For |
|-------|---------|------------------|
| Python location | `which python` | Should show conda environment path |
| Python version | `python --version` | Should match project requirements |
| Active environment | `conda info --envs` | Asterisk marks active environment |
| Installed packages | `conda list` | Verify required packages present |
| Import paths | `python -c "import sys; print(sys.path)"` | Should include project directories |
| Package location | `python -c "import pkg; print(pkg.__file__)"` | Should be in conda environment |
| Environment details | `conda info` | Shows channels, package cache, envs |

### Script vs Notebook Comparison

| Feature | Script | Notebook |
|---------|--------|----------|
| **Execution order** | Always top-to-bottom | User-determined, ambiguous |
| **State management** | Fresh on each run | Accumulates between cells |
| **Version control** | Clean text diffs | JSON mess with outputs |
| **Debugging** | Professional tools | Cell-by-cell only |
| **Testing** | Straightforward | Nearly impossible |
| **Performance** | Direct execution | Overhead and lag |
| **Collaboration** | Standard git workflow | Merge conflicts common |
| **Reproducibility** | Guaranteed | Often impossible |
| **Memory usage** | Predictable | Can accumulate invisibly |
| **Production ready** | Yes | Never |

## Next Chapter Preview

Now that you understand your computational environment and can work effectively in IPython, {doc}`Chapter 2 <../02-python-calculator/index>` will transform Python into a powerful scientific calculator. You'll discover why `0.1 + 0.2 ≠ 0.3` in Python (and every programming language), learn to handle the numerical precision issues that plague computational physics, and understand how computers actually represent numbers. These fundamentals might seem basic, but small numerical errors compound exponentially—a tiny rounding error in an orbital calculation can send your simulated spacecraft to the wrong planet. Get ready to master the subtle art of computational arithmetic where every digit matters and where understanding floating-point representation can mean the difference between a successful mission and a spectacular failure.