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

- Configure and navigate **IPython** as your primary interactive computing environment
- Diagnose why identical code produces different results on different machines
- Explain how Python locates and loads code when you type `import`
- Identify the hidden dangers of **Jupyter notebooks** that corrupt scientific results
- Create reproducible computational environments using **conda**
- Debug environment problems systematically using diagnostic tools
- Transform notebook explorations into reproducible Python **scripts**
- Execute Python code effectively from the terminal and IPython environment

## Prerequisites Check

:::{admonition} ✅ Before Starting This Chapter
:class: note

- [ ] You have completed the Getting Started module (basic setup, CLI, git)
- [ ] You can navigate directories using `cd`, `ls`, and `pwd`
- [ ] You have Miniforge installed with the `astr596` environment created
- [ ] You can activate your conda environment: `conda activate astr596`
- [ ] You understand file paths (absolute vs. relative)

If any boxes are unchecked, review the Getting Started module first.
:::

## Chapter Overview

Picture this: You download code from a groundbreaking astronomy paper, eager to reproduce their results. You run it exactly as instructed. Instead of the published results, you get error messages, or worse — completely different numbers with no indication why. This frustrating scenario happens to nearly every computational scientist, from undergraduates to professors. The problem isn't bad code or user error; it's that scientific computing happens in complex **environments** where tiny differences can cascade into complete failures.

:::{margin}
**environment**
An isolated Python installation with its own packages and settings
:::

This chapter reveals the hidden machinery that makes Python work (or not work) on your computer. You'll discover why the same code produces different results on different machines, master **IPython** as your computational laboratory, understand the dangers of **Jupyter notebooks**, and learn to create truly reproducible computational environments. These aren't just technical skills — they're the foundation of trustworthy computational science.

By chapter's end, you'll transform from someone who hopes code works to someone who knows exactly why it works (or doesn't). You'll diagnose "module not found" errors in seconds, create environments that work identically on any machine, and understand the critical difference between exploration and reproducible science. Let's begin by exploring the tool that will become your new best friend: IPython.

## 1.1 IPython: Your Computational Laboratory

:::{margin}
**IPython**
Interactive Python - an enhanced interpreter for scientific computing
:::

While you could use the basic Python interpreter by typing `python`, **IPython** (just type `ipython` instead) transforms your terminal into a powerful environment for scientific exploration. Think of it as the difference between a basic calculator and a graphing calculator — both do math, but one is designed for serious work. Let's see why every professional computational scientist prefers IPython over the basic Python **REPL**.

:::{margin}
**REPL**
Read-Eval-Print Loop - an interactive programming environment
:::

### Launching Your Laboratory

First, ensure you're in the right environment, then launch IPython:

```{code-cell} ipython3
# IMPORTANT: These are terminal commands, not Python code!
# Type these in your terminal/command prompt, not in Python

# Step 1: Open your terminal (Terminal on Mac/Linux, Anaconda Prompt on Windows)
# Step 2: Activate your conda environment
print("In terminal: conda activate astr596")

# Step 3: Launch IPython
print("In terminal: ipython")

# You'll see something like this appear:
print("""
Python 3.11.5 | packaged by conda-forge
IPython 8.14.0 -- An enhanced Interactive Python
In [1]: 
""")

# The 'In [1]:' prompt means you're now in IPython, ready to type Python code
print("Now you can type Python commands at the In [1]: prompt")
```

Notice the prompt says `In [1]:` instead of `>>>` (which is what basic Python shows). This numbering system is your first hint that IPython is different — it remembers everything. Each command you type gets a number, making it easy to reference previous work.

### The Power of Memory

IPython maintains a complete history of your session, accessible through special variables:

```{code-cell} ipython3
# Type these commands one at a time in IPython
import math

# Earth's radius in km
radius = 6371  

# Calculate Earth's volume
volume = (4/3) * math.pi * radius**3
print(f"Earth's volume: {volume:.2e} km³")

# In IPython, you can reference previous work:
print("In IPython, Out[n] stores outputs, In[n] stores inputs")
print("The underscore _ references the last output")
```

:::{admonition} 🤔 Check Your Understanding
:class: hint

What's the difference between `In[5]` and `Out[5]` in IPython?
:::

:::{admonition} Solution
:class: tip, dropdown

- `In[5]` contains the actual text/code you typed in cell 5 (as a string)
- `Out[5]` contains the result/value that cell 5 produced (if any)

For example:
- `In[5]` might be `"2 + 2"`
- `Out[5]` would be `4`

This history system lets you reference and reuse previous computations without retyping.
:::

### Tab Completion: Your Exploration Tool

Tab completion helps you discover what's available without memorizing everything:

```{code-cell} ipython3
import math

# In real IPython, type: math.<TAB>
# Shows all available functions
available_functions = [item for item in dir(math) if not item.startswith('_')]
print("math module contains:", available_functions[:10], "...")

# To see functions containing 'sin' (math.*sin*?<TAB> in IPython):
sin_functions = [item for item in dir(math) if 'sin' in item]
print("\nFunctions with 'sin':", sin_functions)
```

### Magic Commands: IPython's Superpowers

:::{margin}
**magic command**
IPython commands prefixed with % providing special functionality
:::

IPython's **magic commands** give you capabilities far beyond standard Python:

```{code-cell} ipython3
# In IPython, you would use: %timeit sum(range(1000))
# Here we simulate timing comparisons
import timeit

# Time list comprehension
time1 = timeit.timeit('[i**2 for i in range(100)]', number=10000)
print(f"List comprehension: {time1*100:.4f} µs per loop")

# Time map/lambda approach
time2 = timeit.timeit('list(map(lambda x: x**2, range(100)))', number=10000)
print(f"Map with lambda: {time2*100:.4f} µs per loop")

# Show which is faster
print(f"\nList comprehension is {time2/time1:.1f}x faster")
```

:::{admonition} 🚨 Common Bug Alert: Platform-Specific Timing
:class: warning

Timing results vary significantly between machines due to:
- CPU speed and architecture
- System load and background processes
- Python version and compilation options

Never assume timing results from one machine apply to another. Always benchmark on your target system.
:::

:::{admonition} 🌟 The More You Know: When 0.1 Seconds = 28 Dead
:class: tip, dropdown

**TODO:** should be be in Chapter 2 instead in the floating-point/precision section? Most likely yes

On February 25, 1991, during the Gulf War, a Patriot missile defense system failed to intercept an incoming Iraqi Scud missile, resulting in 28 American soldiers killed and 98 wounded. The cause? A timing error that accumulated due to floating-point precision issues.

The system's internal clock counted time in tenths of seconds, but 0.1 cannot be exactly represented in binary floating-point arithmetic—it's actually 0.099999999... in binary. After running for 100 hours, this tiny error had accumulated to 0.34 seconds. In that time, a Scud missile travels over 600 meters, causing the Patriot system to look in the wrong part of the sky ([GAO Report, 1992](https://www.gao.gov/products/imtec-92-26)).

The truly tragic part? The software fix had already been written and was in transit to the base when the attack occurred. As noted in the official report: "The Patriot system was designed to operate for only 14 hours at a time... No one expected the system to run continuously for 100 hours."

This disaster teaches us two critical lessons: First, tiny numerical errors compound over time—what seems like a trivial rounding error can have deadly consequences. Second, environment assumptions matter—software designed for one use case (14-hour operation) failed catastrophically when used differently (100-hour operation). When you learn about floating-point precision in Chapter 2, remember: those "boring" numerical details can literally be matters of life and death.
:::

### Getting Help Instantly

IPython makes documentation accessible without leaving your workflow:

```{code-cell} ipython3
import math

# In real IPython, use: math.sqrt?
print("In IPython, use ? for quick help:")
print("  math.sqrt?  - shows documentation")
print("  math.sqrt?? - shows source code (if available)")
print("\nExample documentation for math.sqrt:")
print("  Return the square root of x.")
print("  Domain: x ≥ 0, Range: result ≥ 0")
```

:::{admonition} 💡 Computational Thinking: Interactive Exploration
:class: important

The ability to quickly test ideas and explore APIs interactively is fundamental to computational science. IPython's environment encourages experimentation:

**Explore** → **Test** → **Refine**

This rapid iteration cycle is how algorithms are born and bugs are discovered. This pattern appears everywhere: interactive debuggers, REPLs in other languages, and computational notebooks all follow this explore-test-refine cycle.
:::

### Managing Your Workspace

```{code-cell} ipython3
import sys

# Create some variables for demonstration
data = [1, 2, 3, 4, 5]
result = sum(data)
name = "Earth"

# Show variables (simulating %who in IPython)
current_vars = [var for var in dir() 
                if not var.startswith('_') and var not in ['sys', 'timeit', 'math']]
print("Variables in workspace (%who in IPython):", current_vars[:3])

# Detailed info (simulating %whos in IPython)
print("\nDetailed variable info (%whos in IPython):")
for var in current_vars[:3]:
    obj = eval(var)
    print(f"  {var:10} {type(obj).__name__:10} {str(obj)[:30]}")
```

:::{admonition} 🌟 The More You Know: The Reproducibility Crisis
:class: tip, dropdown

In 2016, a survey in *Nature* found that more than 70% of researchers failed to reproduce another scientist's experiments, and more than 50% failed to reproduce their own experiments ([Baker, 2016](https://doi.org/10.1038/533452a)). While this survey focused on experimental sciences, computational reproducibility faces unique challenges.

A study by Stodden et al. (2018) attempted to reproduce 204 computational results from *Science* magazine and found that only 26% could be reproduced without author assistance ([Stodden et al., 2018](https://doi.org/10.1073/pnas.1708290115)). The main barriers? Missing dependencies, absent random seeds, and undocumented computational environments—exactly what you're learning to manage with IPython's session history and environment controls.

Tools like IPython's `%history` and `%save` commands create an audit trail that helps ensure your future self (and others) can reproduce your work. It's not just good practice—it's becoming a requirement at many journals!
:::

## 1.2 Understanding Python's Hidden Machinery

:::{margin}
**module**
A Python file containing code that can be imported and used in other programs
:::

When you type a simple line like `import math`, a complex process unfolds behind the scenes. Understanding this machinery is the difference between guessing why code fails and knowing exactly how to fix it.

### The Import System Exposed

:::{margin}
**import system**
Python's mechanism for loading code from external files
:::

Let's peek behind the curtain to understand Python's **import system**:

```{code-cell} ipython3
import sys
from pathlib import Path

# Where is Python running from?
print(f"Python executable: {sys.executable}")

# What version are we using?
print(f"Python version: {sys.version.split()[0]}")

# Where will Python look for modules?
print("\nPython searches these locations (in order):")
for i, path in enumerate(sys.path[:5], 1):
    # Shorten paths for readability
    display_path = str(path).replace(str(Path.home()), "~")
    print(f"  {i}. {display_path}")
print("  ... and more")
```

:::{margin}
**sys.path**
Python's list of directories to search when importing modules
:::

This search path (**sys.path**) determines everything. When you `import something`, Python checks each directory in order and uses the first match it finds.

### Debugging Import Problems

Here's a diagnostic function you'll use throughout your career, broken into manageable stages:

**Stage 1: Check Environment**

```{code-cell} ipython3
def check_python_environment():
    """Verify we're in the correct Python environment."""
    import sys
    
    env_path = sys.executable
    if 'astr596' in env_path:
        return True, f"✓ Correct environment: {env_path}"
    return False, f"✗ Wrong environment: {env_path}"

status, message = check_python_environment()
print(message)
```

**Stage 2: Try Import**

```{code-cell} ipython3
def try_import(module_name):
    """Attempt to import a module with informative error."""
    try:
        module = __import__(module_name)
        if hasattr(module, '__file__'):
            location = module.__file__
        else:
            location = "built-in module"
        return True, f"✓ Found: {location}"
    except ImportError as e:
        return False, f"✗ Failed: {str(e)}"

# Test with standard library
success, msg = try_import('math')
print(msg)
```

**Stage 3: Suggest Fix**

```{code-cell} ipython3
def suggest_import_fix(module_name, error):
    """Suggest solutions for import failures."""
    if "No module named" in str(error):
        return f"Install with: conda install {module_name}"
    elif "cannot import name" in str(error):
        return "Check version compatibility"
    return "Verify environment activation"

# Example usage
fix = suggest_import_fix('numpy', "No module named 'numpy'")
print(fix)
```

:::{admonition} 🤔 Check Your Understanding
:class: hint

You get `ModuleNotFoundError: No module named 'astropy'`. What are three possible causes?
:::

:::{admonition} Solution
:class: tip, dropdown

1. **Wrong environment**: You're not in the conda environment where astropy is installed
2. **Not installed**: Astropy isn't installed in the current environment
3. **Path issues**: Python's sys.path doesn't include the directory containing astropy

To diagnose:
```bash
which python          # Are you using the right Python?
conda list astropy    # Is it installed?
python -c "import sys; print(sys.path)"  # Where is Python looking?
```

The most common cause is forgetting to activate your conda environment!
:::

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
]

print("Potential Python locations on Unix-like systems:")
for path in possible_pythons:
    expanded_path = Path(path).expanduser()
    exists = "✓" if expanded_path.exists() else "✗"
    print(f"  {exists} {path}")

print("\nThis is why 'conda activate' is crucial!")
```

:::{admonition} 🚨 Common Bug Alert: The Wrong Python
:class: warning

**Symptom**: Code works in terminal but fails in IDE, or vice versa

**Cause**: Different tools using different Python installations

**Fix**: Always verify with:
```bash
which python       # Unix/Mac
where python       # Windows
```

**Prevention**: Always activate your conda environment first!
:::

## 1.3 Jupyter Notebooks: Beautiful Disasters Waiting to Happen

:::{margin}
**Jupyter notebook**
Web-based platform that executes code in cells while maintaining state
:::

**Jupyter notebooks** seem perfect for scientific computing - you can mix code, results, and explanations in one document. However, they harbor dangerous flaws that can corrupt your scientific results. You may use them for Short Project 1 to understand their appeal, but then you must abandon them for more robust approaches.

### The Seductive Power of Notebooks

To start Jupyter (after activating your environment):

```{code-cell} ipython3
# In terminal:
# conda activate astr596
# jupyter lab

print("Jupyter Lab would open at: http://localhost:8888")
print("You can create notebooks, write code in cells, and see results inline")
```

### The Hidden State Monster

The most insidious problem: notebooks maintain the hidden state between cell executions:

```{code-cell} ipython3
# Simulating notebook cells with execution order problems

# Cell 1 (first execution)
gravity = 980  # cm/s², Earth's gravity
print(f"Cell 1: Set gravity = {gravity}")

# Cell 2 (depends on gravity)
import math
def calculate_fall_time(height):
    """Calculate time using gravity from when function was created."""
    return math.sqrt(2 * height / gravity)

print(f"Cell 2: Defined function with gravity = {gravity}")

# Cell 3 (changes gravity)
gravity = 371  # cm/s², Mars gravity
print(f"Cell 3: Changed gravity = {gravity}")

# Cell 4 (which gravity does this use?)
time = calculate_fall_time(100)
print(f"Cell 4: Fall time = {time:.2f} seconds")
print(f"  But function still uses gravity = 980!")
print(f"  This hidden state causes wrong results!")
```

:::{admonition} 🔧 Debug This!
:class: challenge

A student's notebook has these cells:

```python
Cell 1: data = [1, 2, 3]
Cell 2: result = sum(data) / len(data)
Cell 3: data.append(4)
Cell 4: print(f"Average: {result}")
```

They run cells in order: 1, 2, 3, 4, 2, 4. What prints the second time?
:::

:::{admonition} Solution
:class: solution, dropdown

The second execution of Cell 4 prints: `Average: 2.5`

Execution trace:

1. Cell 1: `data = [1, 2, 3]`
2. Cell 2: `result = 2.0` (sum=6, len=3)
3. Cell 3: `data = [1, 2, 3, 4]`
4. Cell 4: Prints `"Average: 2.0"`
5. Cell 2 again: `result = 2.5` (sum=10, len=4)
6. Cell 4 again: Prints `"Average: 2.5"`

This demonstrates how re-running cells creates different states than sequential execution!
:::

### Memory Accumulation Disasters

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

print("\nEach run ADDS to memory - notebooks don't reset!")
```

:::{admonition} 🌟 The More You Know: The $6 Trillion Excel Error
:class: tip, dropdown

In 2013, graduate student Thomas Herndon couldn't reproduce the results from a highly influential economics paper by Carmen Reinhart and Kenneth Rogoff. This paper, ["Growth in a Time of Debt,"](https://www.aeaweb.org/articles?id=10.1257/aer.100.2.573) had been cited by politicians worldwide to justify austerity policies affecting millions of people.

When Herndon finally obtained the original Excel spreadsheet, he discovered a coding error: five countries were accidentally excluded from a calculation due to an Excel formula that didn't include all rows. This simple mistake skewed the results, showing that high debt caused negative growth when the corrected analysis showed much weaker effects ([Herndon, Ash, and Pollin, 2014](https://doi.org/10.1093/cje/bet075)). The implications were staggering — this spreadsheet error influenced global economic policy.

Just like hidden state in Jupyter notebooks, the error was invisible in the final spreadsheet. The lesson? Computational transparency and reproducibility aren't just academic exercises — they have real-world consequences. Always make your computational process visible and reproducible!
:::

### The Notebook-to-Script Transition

:::{margin}
**script**
Plain text file (`.py` extension) with Python code that executes top-to-bottom
:::

After Project 1, we'll abandon notebooks for **scripts**. Here's why scripts are superior:

:::{list-table} Script vs Notebook Comparison
:header-rows: 1
:widths: 30 35 35

* - Aspect
  - Notebooks
  - Scripts
* - Execution Order
  - Ambiguous, user-determined
  - Top-to-bottom, always
* - Hidden State
  - Accumulates invisibly
  - Fresh start each run
* - Version Control
  - JSON mess with outputs
  - Clean text diffs
* - Testing
  - Nearly impossible
  - Straightforward
* - Debugging
  - Cell-by-cell only
  - Professional tools
* - Reproducibility
  - Often impossible
  - Guaranteed
:::

:::{admonition} 💡 Computational Thinking: Reproducible by Design
:class: important

**Reproducibility** isn't just about sharing code — it's about ensuring that code produces identical results regardless of who runs it or when. Scripts enforce this by eliminating hidden state and ambiguous execution order.

This principle extends beyond Python: declarative configurations, containerization, and infrastructure-as-code all follow the same philosophy of explicit, reproducible computation.

Remember the mantra: "It should work the same way every time, for everyone."
:::

## 1.4 Scripts: Write Once, Run Anywhere (Correctly)

Python scripts are simple text files containing Python code, executed from top to bottom, the same way every time. No hidden state, no ambiguity, just predictable execution.

### From IPython to Script

Start by experimenting in IPython:

```{code-cell} ipython3
# Quick calculation in IPython
earth_mass = 5.97e27  # g
moon_mass = 7.35e25   # g
ratio = earth_mass / moon_mass
print(f"Earth is {ratio:.1f}× more massive than the Moon")
```

Now create a proper script. Save this as `mass_ratio.py`:

```{code-cell} ipython3
#!/usr/bin/env python
"""Calculate mass ratios between celestial bodies."""

# Constants in CGS units [g]
EARTH_MASS = 5.97e27
MOON_MASS = 7.35e25
SUN_MASS = 1.99e33

def calculate_ratio(mass1, mass2):
    """Calculate mass ratio between two bodies.
    
    Args:
        mass1: Mass of first body [g]
        mass2: Mass of second body [g]
        
    Returns:
        float: Ratio of mass1 to mass2
    """
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
```

### The `if __name__ == "__main__"` Pattern

:::{margin}
**__name__**
Python variable that equals "__main__" when run directly, or the module name when imported
:::

This crucial pattern makes your code both runnable and importable. Don't worry if this seems confusing at first — it's a pattern you'll use in every script you write, and it will become second nature:

```{code-cell} ipython3
# Understanding the __name__ variable
def useful_function(x):
    """A function others might want to use."""
    return x ** 2

# Python automatically sets __name__ based on how the file is used
print(f"Module's __name__ is: {__name__}")

# This is the magic pattern - memorize it even if you don't fully understand it yet
if __name__ == "__main__":
    # This code ONLY runs when you execute the script directly
    # It does NOT run when someone imports your script
    print("Running as a script!")
    result = useful_function(5)
    print(f"5 squared is {result}")
    
# Think of it like this:
# - Running directly: Python sets __name__ to "__main__", so the code runs
# - Importing: Python sets __name__ to the filename, so the code doesn't run
```

:::{admonition} 🤔 Check Your Understanding
:class: hint

Why would you want code that behaves differently when imported versus run directly?
:::

:::{admonition} Solution
:class: tip, dropdown

This pattern serves multiple purposes:

1. **Testing**: Include test code that runs when developing but not when others use your functions
2. **Reusability**: Others can import your functions without triggering test/demo code
3. **Library Design**: Create modules that work both as tools and standalone programs
4. **Development**: Test functions immediately while writing them

Example:
```python
# orbital_mechanics.py
def orbital_period(a, M):
    # ... calculation ...
    return period

if __name__ == "__main__":
    # Test with Earth's orbit
    period = orbital_period(1.496e13, 1.989e33)
    print(f"Earth's period: {period/86400:.1f} days")
```
:::

## 1.5 Creating Reproducible Environments

:::{margin}
**conda**
Package and environment manager for isolated Python installations
:::

:::{margin}
**virtual environment**
An isolated Python installation with its own packages, preventing conflicts between projects
:::

Your code's behavior depends on its **environment** — Python version, installed packages, even operating system. Creating reproducible environments ensures your code works identically everywhere.

### The Conda Solution

**Conda** creates isolated environments — separate Python installations with their own packages:

```{code-cell} ipython3
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

:::{admonition} 🚨 Common Bug Alert: Channel Confusion
:class: warning

**Problem**: Package not found or wrong version installed

**Cause**: Different conda channels have different package versions

**Solution**: Always specify channels in environment.yml

**Best Practice**: Use `conda-forge` channel for scientific packages

```yaml
channels:
  - conda-forge  # Always list this first
  - defaults     # Fallback if needed
```
:::

### Proper Path Management

Stop hardcoding paths that break on other systems. If you've only used Jupyter notebooks, you might not realize that file paths like `C:\Users\YourName\data.txt` only work on YOUR computer:

```{code-cell} ipython3
from pathlib import Path
import os

# BAD: Only works on one specific computer
bad_path = '/Users/yourname/research/data.txt'
print(f"BAD (hardcoded): {bad_path}")
print("  Problem: This exact folder structure doesn't exist on other computers!")

# GOOD: Works everywhere (relative to where your script is)
script_dir = Path.cwd()  # cwd = "current working directory" (where you are now)
data_file = script_dir / 'data' / 'observations.txt'  # The / operator joins paths!
print(f"\nGOOD (relative): {data_file}")
print("  This looks for 'data' folder relative to where your script runs")

# BETTER: Handle missing files gracefully (defensive programming!)
if data_file.exists():
    print(f"\n✓ Found data at: {data_file}")
else:
    print(f"\n✗ Data not found at: {data_file}")
    print(f"    (Expected - this is just a demo)")

# BEST: Use configuration with environment variables
# Environment variables are settings your computer stores
# You can set DATA_DIR to different paths on different computers
data_dir = Path(os.getenv('DATA_DIR', './data'))  # Use DATA_DIR or default to ./data
print(f"\nBEST (configurable): {data_dir}")
print("  Falls back to './data' if DATA_DIR environment variable not set")
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

:::{admonition} 🌟 The More You Know: Reproducing a Nobel Prize Discovery
:class: tip, dropdown

When the LIGO Scientific Collaboration announced the first detection of gravitational waves on February 11, 2016, the scientific community demanded proof. This wasn't just any claim - it confirmed a 100-year-old prediction by Einstein and would earn the 2017 Nobel Prize in Physics.

The LIGO team didn't just publish their results ([Abbott et al., 2016](https://doi.org/10.1103/PhysRevLett.116.061102)) - they provided complete reproducibility. They released:

1. **Raw data**: 32 seconds of strain data around the event (GW150914) from both detectors
2. **Analysis code**: Complete Python scripts and Jupyter notebooks on [GitHub](https://github.com/ligo-scientific-collaboration/GW150914)
3. **Environment specifications**: Exact versions of all software dependencies
4. **Random seeds**: For all Monte Carlo analyses in their statistical validation

The result? Within days, independent scientists worldwide confirmed the detection. As the team noted in their software paper, "The analyses used to support the first direct detection... have been reproduced by independent scientists using the publicly available data and software" ([Abbott et al., 2021](https://doi.org/10.1088/1361-6382/abfd85)).

The LIGO team even created "blind injections" — fake signals secretly added to the data stream to test their analysis pipeline. Only after the analysis was complete would they reveal whether a signal was real or injected. This prevented confirmation bias and ensured their computational methods were robust.

Your `random.seed()` calls and environment files might seem trivial now, but they're the same practices that enabled one of the most important physics discoveries of the 21st century to be independently verified. When you set a random seed, you're following in the footsteps of Nobel laureates!
:::

## 1.6 Essential Debugging Strategies

:::{margin}
**defensive programming**
Writing code that anticipates and handles failures gracefully
:::

When code fails (and it will), systematic debugging saves hours of frustration. Here are strategies that work every time, demonstrating **defensive programming**.

### The Universal First Check

Before anything else, verify your environment:

```{code-cell} ipython3
import sys
import os
from pathlib import Path

def environment_check():
    """Universal debugging first check.
    
    Returns:
        bool: True if in correct environment
    """
    print("=== Environment Debug Check ===")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Current dir: {os.getcwd()}")
    
    # Check for conda environment
    if 'conda' in sys.executable or 'miniconda' in sys.executable:
        # Extract environment name
        path_parts = sys.executable.split(os.sep)
        if 'envs' in path_parts:
            env_idx = path_parts.index('envs')
            env_name = path_parts[env_idx + 1] if env_idx + 1 < len(path_parts) else "base"
            
            if 'astr596' in env_name:
                print(f"✓ Correct environment: {env_name}")
                return True
            else:
                print(f"✗ Wrong environment: {env_name}")
                print("  Fix: conda activate astr596")
                return False
    
    print("✗ Not in a conda environment")
    print("  Fix: conda activate astr596")
    return False

# Run the check
environment_check()
```

### Using IPython's Debugger

When code crashes, IPython's `%debug` magic enters the debugger at the crash point:

```{code-cell} ipython3
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

print("""In IPython, after an error occurs, type: %debug

Debugger commands:
  p variable  - print variable value
  l          - list code around error
  u/d        - go up/down the call stack
  c          - continue execution
  n          - next line
  s          - step into function
  q          - quit debugger
  
Example session:
  ipdb> p divisor
  0
  ipdb> p numbers
  [1, 2, 3]
  ipdb> q
""")
```

:::{admonition} 💡 Computational Thinking: Defensive Programming
:class: important

Defensive programming means assuming things will go wrong and coding accordingly. Instead of hoping files exist, check first. Instead of assuming imports work, verify them.

This mindset—expect failure, handle it gracefully — separates robust scientific code from scripts that work "sometimes."

Pattern:
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

This pattern appears everywhere: network requests that might timeout, sensors that might malfunction, or data that might be corrupted.
:::

## Practice Exercises

### Exercise 1.1: IPython Mastery

:::{admonition} Part A: Follow These Steps (5 min)
:class: exercise, dropdown

Execute these commands exactly in IPython:

```{code-cell} ipython3
# Step 1: Import and explore
import math

# Step 2: Check available functions
print("Functions:", [x for x in dir(math) if not x.startswith('_')][:5])

# Step 3: Time a simple calculation
import timeit
time = timeit.timeit('sum(range(100))', number=10000)
print(f"Time: {time*100:.4f} µs per loop")
```
:::

:::{admonition} Part B: Modify the Approach (10 min)
:class: exercise, dropdown

Compare two methods for squaring numbers:

```{code-cell} ipython3
import timeit

# Method 1: List comprehension
time1 = timeit.timeit('[i**2 for i in range(100)]', number=1000)

# Method 2: Map with lambda
time2 = timeit.timeit('list(map(lambda x: x**2, range(100)))', number=1000)

print(f"List comp: {time1*1000:.2f} ms")
print(f"Map/lambda: {time2*1000:.2f} ms")
print(f"Ratio: {time2/time1:.2f}x")

# Which is faster? Why?
```
:::

:::{admonition} Part C: Apply Your Knowledge (15 min)
:class: exercise, dropdown

Design your own timing experiment for calculating factorials:

```{code-cell} ipython3
# Create three different ways to calculate factorials for 1-20
# Hint: Consider recursion, iteration, and math.factorial

import math
import timeit

# Method 1: Using math.factorial
def method1(n):
    return [math.factorial(i) for i in range(1, n+1)]

# Method 2: Iterative approach
def method2(n):
    results = []
    for i in range(1, n+1):
        fact = 1
        for j in range(1, i+1):
            fact *= j
        results.append(fact)
    return results

# Method 3: Recursive (be careful with large n!)
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n-1)

def method3(n):
    return [factorial_recursive(i) for i in range(1, n+1)]

# Time each approach
t1 = timeit.timeit('method1(20)', globals=globals(), number=1000)
t2 = timeit.timeit('method2(20)', globals=globals(), number=1000)
t3 = timeit.timeit('method3(20)', globals=globals(), number=1000)

print(f"math.factorial: {t1*1000:.2f} ms")
print(f"Iterative:      {t2*1000:.2f} ms")
print(f"Recursive:      {t3*1000:.2f} ms")
print(f"\nFastest: math.factorial (built-in C implementation)")
```

:::

### Exercise 1.2: Notebook State Detective

:::{admonition} Part A: Trace Execution (5 min)
:class: exercise, dropdown

Given this execution order, trace the state:

```{code-cell} ipython3
print("""Execution order: Cell 1, Cell 3, Cell 2, Cell 4, Cell 2, Cell 4

Cell 1: counter = 0
        data = []

Cell 2: counter += 1
        data.append(counter)

Cell 3: counter = 10

Cell 4: print(f"Counter: {counter}, Data: {data}")

Work through each step on paper first.""")
```
:::

:::{admonition} Part B: Find the Final State (10 min)
:class: exercise, dropdown

```{code-cell} ipython3
# Simulate the execution
counter = 0      # Cell 1
data = []        # Cell 1
counter = 10     # Cell 3
counter += 1     # Cell 2
data.append(counter)  # Cell 2
print(f"After first Cell 4: Counter: {counter}, Data: {data}")

counter += 1     # Cell 2 again
data.append(counter)  # Cell 2 again
print(f"After second Cell 4: Counter: {counter}, Data: {data}")
```
:::

:::{admonition} Part C: Explain the Danger (15 min)
:class: exercise, dropdown

Write a paragraph explaining why this execution pattern would be:
1. Hard to debug
2. Impossible to reproduce
3. Dangerous for scientific computing

Consider: What if this was calculating orbital trajectories?
:::

### Exercise 1.3: Environment Diagnostic Tool

:::{admonition} Part A: Basic Diagnostic (5 min)
:class: exercise, dropdown

```{code-cell} ipython3
import sys
import os

# Basic environment info
print(f"Python: {sys.version.split()[0]}")
print(f"Location: {sys.executable}")
print(f"Working dir: {os.getcwd()}")
```
:::

:::{admonition} Part B: Check Key Packages (10 min)
:class: exercise, dropdown

```{code-cell} ipython3
# Check for scientific packages
packages = ['numpy', 'scipy', 'matplotlib', 'pandas']

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")
```
:::

:::{admonition} Part C: Complete Diagnostic (15 min)
:class: exercise, dropdown

```{code-cell} ipython3
import sys
import subprocess
from pathlib import Path
import os

def full_diagnostic():
    """Complete environment diagnostic."""
    print("=== Full Environment Diagnostic ===\n")
    
    # 1. Python info
    print(f"1. Python: {sys.version.split()[0]}")
    print(f"   Location: {sys.executable}")
    
    # 2. Environment name
    if 'conda' in sys.executable:
        parts = sys.executable.split(os.sep)
        if 'envs' in parts:
            idx = parts.index('envs')
            env = parts[idx + 1] if idx + 1 < len(parts) else "base"
            print(f"\n2. Environment: {env}")
    
    # 3. Search paths (first 3)
    print("\n3. Python paths:")
    for i, path in enumerate(sys.path[:3], 1):
        display = str(path).replace(str(Path.home()), "~")
        print(f"   {i}. {display}")
    
    # 4. Package count
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True, text=True, timeout=5
        )
        count = len(result.stdout.strip().split('\n')) - 2
        print(f"\n4. Total packages: {count}")
    except:
        print("\n4. Could not count packages")
    
    return True

full_diagnostic()
```
:::

### Exercise 1.4: Variable Star Exercise Thread

```{code-cell} ipython3
# Chapter 1: Variable Star Basics - Setting up our environment
# This simple start will grow into a full VariableStar class by Chapter 6

# Define basic properties of a Cepheid variable star
star_name = "Delta Cephei"
period = 5.366319  # days
magnitude_mean = 3.95
magnitude_amplitude = 0.88

# Store in a simple dictionary for now
variable_star = {
    'name': star_name,
    'period': period,
    'mag_mean': magnitude_mean,
    'mag_amp': magnitude_amplitude
}

print(f"Variable star {variable_star['name']}:")
print(f"  Period: {variable_star['period']:.3f} days")
print(f"  Magnitude: {variable_star['mag_mean']:.2f} ± {variable_star['mag_amp']:.2f}")

# Save to file for next chapter with error handling (defensive programming!)
import json
try:
    with open('variable_star_ch1.json', 'w') as f:
        json.dump(variable_star, f, indent=2)
    print("\n✓ Data saved successfully for Chapter 2!")
    
    # Verify the save worked by reading it back
    with open('variable_star_ch1.json', 'r') as f:
        verification = json.load(f)
    print(f"  Verified: {verification['name']} data intact")
    
except IOError as e:
    print(f"\n✗ Warning: Could not save data: {e}")
    print("  You'll need to recreate this in Chapter 2")
except json.JSONDecodeError as e:
    print(f"\n✗ Warning: Data corruption detected: {e}")
    print("  Try running this cell again")
```

## Main Takeaways

This chapter has revealed the hidden complexity underlying every Python program you'll write. You've learned that when code fails to run or produces different results on different machines, it's rarely due to the code itself — it's the **environment** surrounding that code. Understanding this distinction transforms you from someone who gets frustrated by errors to someone who systematically diagnoses and fixes them.

**IPython** is more than just an enhanced Python prompt - it's a scientific laboratory where ideas become code. The ability to quickly test hypotheses, examine results, and iterate on solutions is fundamental to computational science. The **magic commands** like `%timeit` and `%debug` aren't just conveniences; they're essential tools that separate casual coding from professional scientific computing. Master IPython now, and you'll use it daily throughout your research career.

The **Jupyter notebook** trap is real and dangerous. While notebooks seem perfect for scientific work — mixing code, results, and narrative — their hidden state and execution ambiguity make them unsuitable for serious scientific computing. The out-of-order execution problem isn't a minor inconvenience; it's a fundamental flaw that can corrupt your scientific results. After Project 1, you'll leave notebooks behind for the reliability of **scripts**, but understanding their dangers now will help you recognize similar issues in other tools.

Scripts enforce **reproducibility** through simplicity. By executing top-to-bottom every time, they eliminate the ambiguity that plagues notebooks. The `if __name__ == "__main__"` pattern might seem like unnecessary boilerplate now, but it's the key to writing code that's both immediately useful and reusable by others. This pattern embodies a core principle: good scientific code serves multiple purposes without compromising any of them.

Creating reproducible environments isn't just about making your code run on other machines — it's about scientific integrity. When you can't reproduce your own results from six months ago, you've lost the thread of your research. The tools you've learned — **conda** environments, environment files, proper path handling, and seed control — aren't optional extras. They're the foundation of trustworthy computational science. Every major discovery in computational science, from gravitational waves to exoplanet detection, has depended on reproducible environments.

The debugging strategies you've learned will save you countless hours. The universal first check — verifying your environment — solves most "mysterious" errors. Systematic import debugging reveals exactly why **modules** can't be found. IPython's debugger lets you examine failures at the moment they occur. These aren't just troubleshooting techniques; they're the difference between guessing and knowing.

Remember: computational science isn't just about writing code that works once. It's about creating reliable, reproducible tools that advance human knowledge. The practices you've learned in this chapter — from IPython exploration to environment management — are the foundation of that reliability. **Defensive programming** isn't paranoia; it's professionalism.

## Definitions

**conda**: A package and environment management system that creates isolated Python installations with specific package versions, ensuring reproducibility across different machines.

**defensive programming**: Writing code that anticipates and handles potential failures gracefully, checking inputs, validating assumptions, and providing informative error messages rather than assuming everything will work correctly.

**environment**: An isolated Python installation with its own interpreter, packages, and settings, preventing conflicts between projects with different requirements.

**import system**: Python's mechanism for loading code from external files (modules) into your current program, searching through directories listed in sys.path in order.

**IPython**: Interactive Python — an enhanced version of the basic Python interpreter designed specifically for scientific computing, offering features like magic commands, tab completion, and integrated help.

**Jupyter notebook**: A web-based interactive computing platform that allows you to create notebooks combining code, results, and text in cells that can be executed individually while maintaining state between executions.

**magic command**: Special IPython commands prefixed with `%` (line magics) or `%%` (cell magics) that provide functionality beyond standard Python, such as timing code execution or entering the debugger.

**module**: A Python file containing code (functions, classes, variables) that can be imported and used in other Python programs, enabling code reuse and organization.

**REPL**: Read-Eval-Print Loop — an interactive programming environment that takes single expressions, evaluates them, and returns the result immediately, facilitating exploration and testing.

**reproducibility**: The ability to obtain consistent results using the same data and code, regardless of who runs it, when, or on what machine — fundamental to scientific integrity.

**script**: A plain text file containing Python code that executes from top to bottom when run, providing consistent and reproducible execution without hidden state.

**sys.path**: Python's list of directories to search when importing modules, checked in order until a matching module is found, determining what code is available to import.

**__name__**: A special Python variable that equals `"__main__"` when a script is run directly, or the module name when imported, enabling code to behave differently in each context.

## Key Takeaways

✓ **IPython is your primary tool**: Use it for exploration, testing, and quick calculations — not the basic Python interpreter

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

```{list-table} Essential IPython Commands
:header-rows: 1
:widths: 20 30 50

* - Command
  - Purpose
  - Example
* - `%timeit`
  - Time code execution
  - `%timeit sum(range(1000))`
* - `%run`
  - Run script keeping variables
  - `%run analysis.py`
* - `%debug`
  - Enter debugger after error
  - `%debug`
* - `%who`
  - List all variables
  - `%who`
* - `%whos`
  - Detailed variable information
  - `%whos`
* - `%reset`
  - Clear all variables
  - `%reset -f`
* - `%history`
  - Show command history
  - `%history -n 10`
* - `%save`
  - Save code to file
  - `%save script.py 1-10`
* - `?`
  - Quick help
  - `len?`
* - `??`
  - Show source code
  - `len??`
```

```{list-table} Environment Debugging Checklist
:header-rows: 1
:widths: 25 35 40

* - Check
  - Command
  - What to Look For
* - Python location
  - `which python`
  - Should show conda environment path
* - Python version
  - `python --version`
  - Should match project requirements
* - Active environment
  - `conda info --envs`
  - Asterisk marks active environment
* - Installed packages
  - `conda list`
  - Verify required packages present
* - Import paths
  - `python -c "import sys; print(sys.path)"`
  - Should include project directories
* - Package location
  - `python -c "import pkg; print(pkg.__file__)"`
  - Should be in conda environment
```

## Python Module & Method Reference

:::{note}
**Your Growing Python Toolkit**

This reference section starts with Chapter 1's tools and will expand with each chapter. By course end, you'll have built a comprehensive personal Python reference covering everything from basic file operations to advanced machine learning libraries. Think of this as your Python cookbook that you're writing as you learn — bookmark it, you'll use it constantly!
:::

This reference section catalogs all Python modules, functions, and methods introduced in this chapter. It will grow throughout the course as you learn new tools. Think of this as your personal Python toolkit that you're building piece by piece.

### Standard Library Modules

**`sys` module** - System-specific parameters and functions
```python
import sys
```
- `sys.executable` - Path to the Python interpreter currently running
- `sys.version` - Python version information as a string
- `sys.path` - List of directories Python searches for modules
- `sys.getsizeof(object)` - Returns size of object in bytes

**`os` module** - Operating system interface
```python
import os
```
- `os.getcwd()` - Returns current working directory as string
- `os.getenv('VAR_NAME', default)` - Gets environment variable value
- `os.sep` - Path separator for your OS ('/' on Unix, '\\' on Windows)

**`math` module** - Mathematical functions
```python
import math
```
- `math.pi` - Mathematical constant π (3.14159...)
- `math.sqrt(x)` - Square root of x
- `math.factorial(n)` - Returns n! (n factorial)

**`pathlib` module** - Object-oriented filesystem paths
```python
from pathlib import Path
```
- `Path()` - Creates a path object
- `Path.cwd()` - Current working directory as Path object
- `Path.home()` - User's home directory
- `path.exists()` - Returns True if path exists
- `path.expanduser()` - Expands ~ to home directory
- `path / 'subdir'` - Joins paths using `/` operator

**`random` module** - Generate random numbers
```python
import random
```
- `random.seed(n)` - Initialize random number generator for reproducibility
- `random.random()` - Random float between 0.0 and 1.0

**`timeit` module** - Time small code snippets
```python
import timeit
```
- `timeit.timeit(code, number=N)` - Times code execution N times, returns total seconds

**`subprocess` module** - Run external commands
```python
import subprocess
```
- `subprocess.run(cmd, capture_output=True, text=True)` - Executes command and captures output

**`json` module** - JSON encoder and decoder
```python
import json
```
- `json.dump(obj, file)` - Write Python object as JSON to file
- `json.load(file)` - Read JSON from file into Python object

### Built-in Functions (No Import Needed)

**Core Functions**
- `print(*args)` - Display output to console
- `len(obj)` - Returns length/size of object
- `sum(iterable)` - Sum all elements in an iterable
- `range(start, stop, step)` - Generate sequence of numbers
- `enumerate(iterable, start=0)` - Returns index-value pairs
- `dir(object)` - List all attributes of an object
- `type(object)` - Returns the type of an object
- `eval(string)` - Evaluates a Python expression string (use carefully!)

**Type Conversions**
- `int(x)` - Convert to integer
- `float(x)` - Convert to floating-point number
- `str(x)` - Convert to string
- `list(x)` - Convert to list

**Object Inspection**
- `hasattr(obj, 'attribute')` - Check if object has an attribute
- `getattr(obj, 'attribute', default)` - Get attribute value or default
- `__import__(name)` - Import module by name (rarely used directly)

**File Operations**
```python
with open('file.txt', 'r') as f:
    content = f.read()
```
- `open(filename, mode)` - Open file ('r' read, 'w' write, 'a' append)
- `file.read()` - Read entire file as string
- `file.write(string)` - Write string to file
- `with` statement - Ensures file is properly closed after use

### IPython Magic Commands

These special commands only work in IPython, not regular Python scripts. They're prefixed with `%` for line magics or `%%` for cell magics.

**Timing and Profiling**
- `%timeit code` - Time execution with statistical analysis
- `%time code` - Time single execution

**Workspace Management**
- `%who` - List all variables
- `%whos` - Detailed variable information with types and values
- `%reset` - Clear all variables from memory
- `%reset -f` - Force reset without confirmation

**Code Execution**
- `%run script.py` - Execute Python script in IPython
- `%save filename.py n-m` - Save lines n through m to file
- `%load filename.py` - Load code from file into cell

**Debugging**
- `%debug` - Enter debugger after an exception
- `%pdb` - Automatic debugger on exceptions

**Help and History**
- `?` - Quick help (e.g., `len?`)
- `??` - Show source code (e.g., `len??`)
- `%history` - Show command history
- `%history -n 10` - Show last 10 commands

**IPython Special Variables**
- `In[n]` - Input history (code you typed in cell n)
- `Out[n]` - Output history (result from cell n)
- `_` - Previous output
- `__` - Second-to-last output
- `___` - Third-to-last output

### Data Structures and Patterns

**List Comprehensions**
```python
# Pattern: [expression for item in iterable if condition]
squares = [x**2 for x in range(10)]
filtered = [x for x in data if x > 0]
```

**Dictionary Creation**
```python
# Creating dictionaries for data organization
star_data = {
    'name': 'Delta Cephei',
    'period': 5.366,
    'magnitude': 3.95
}
```

**The `if __name__ == "__main__"` Pattern**
```python
# Makes code both importable and runnable
if __name__ == "__main__":
    # This code only runs when script is executed directly
    main()
```

### Quick Usage Examples

```{code-cell} ipython3
# Finding what's in a module
import math
math_functions = [x for x in dir(math) if not x.startswith('_')]
print(f"math has {len(math_functions)} functions")

# Checking your environment
import sys
print(f"Python lives at: {sys.executable}")

# Safe file reading
from pathlib import Path
file = Path("data.txt")
if file.exists():
    content = file.read_text()
else:
    print("File not found - using defaults")

# Reproducible randomness
import random
random.seed(42)  # Always produces same "random" sequence
```

### When to Use What?

**For file paths**: Always use `pathlib.Path`, never hardcode strings
**For timing code**: Use `%timeit` in IPython, `timeit.timeit()` in scripts  
**For checking Python**: Use `sys.executable` and `sys.version`
**For randomness**: Always set `random.seed()` for reproducibility
**For debugging**: Use `%debug` in IPython after errors occur
**For exploration**: Use `dir()` and `?` to discover functionality

This reference will expand as you learn new modules. Keep it handy, you'll refer to it often as you build your computational toolkit!

## Next Chapter Preview

Now that you understand your computational environment and can work effectively in IPython, Chapter 2 will transform Python into a powerful scientific calculator. You'll discover why `0.1 + 0.2 ≠ 0.3` in Python (and every programming language), learn to handle the numerical precision issues that plague computational physics, and understand how computers actually represent numbers. These fundamentals might seem basic, but small numerical errors compound exponentially - a tiny rounding error in an orbital calculation can send your simulated spacecraft to the wrong planet. Get ready to master the subtle art of computational arithmetic where every digit matters and where understanding floating-point representation can mean the difference between a successful mission and a spectacular failure.