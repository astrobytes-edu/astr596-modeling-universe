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

1. **Configure and navigate IPython** as your primary interactive computing environment for scientific projects and astronomical data analysis.
2. **Diagnose and fix three common causes** of environment-dependent behavior in scientific code.
3. **Explain step-by-step how Python's import system** locates and loads modules when you type `import module`.
4. **Identify five specific dangers** of Jupyter notebooks that can corrupt scientific results.
5. **Create fully reproducible computational environments** using `conda` with proper version pinning.
6. **Debug environment problems systematically** using a four-stage diagnostic protocol.
7. **Transform notebook explorations into reproducible Python scripts** following best practices.
8. **Execute Python code effectively** from both terminal and IPython environments.

## Prerequisites Check

:::{important} ✅ Prerequisites Self-Assessment
:class: dropdown

Before starting this chapter, verify you have completed these items:

- [ ] You can open a terminal/command prompt and are able to navigate directories (via `cd`, `ls`, `lsdir`, `pwd`).
- [ ] You have Miniforge installed and created a `astr596` python conda environment for this course.
- [ ] You have a text editor (e.g., Vim)or IDE (e.g., VS Code) installed.
- [ ] You have `git` installed and can you use basic `git` commands (e.g.,`clone`, `add`, `commit`, `pull`).

⚠️ If you checked 'no' to any item, see the [Getting Started](../../02-getting-started/index) module . 
:::

## Chapter Overview

Picture this: You download code from a groundbreaking exoplanet detection paper, eager to reproduce their radial velocity analysis. You run it exactly as instructed. Instead of the published planetary parameters, you get error messages, or worse — completely different orbital periods with no indication why. This frustrating scenario happens to nearly every astrophysicist, from graduate students to professors. The problem isn't bad code or user error; it's that scientific computing happens in complex **environments** where tiny differences can cascade into complete failures.

:::{margin}
**environment**
An isolated Python installation with its own packages and settings
:::

This chapter reveals the hidden machinery that makes Python work (or not work) on your computer. You'll discover why the same spectral analysis code produces different results on different machines, master **IPython** as your computational laboratory for rapid prototyping, understand the dangers of **Jupyter notebooks** in scientific computing, and learn to create truly reproducible computational environments for your research. These aren't just technical skills — they're the foundation of trustworthy astrophysics research.

By chapter's end, you'll transform from someone who hopes code works to someone who knows exactly why it works (or doesn't). You'll diagnose "No module named 'astropy'" errors in seconds, create environments that work identically on any supercomputer cluster, and understand the critical difference between exploration and reproducible science. Let's begin by exploring the tool that will become your new best friend: **IPython**.

:::{margin}
**IPython**
Interactive Python - an enhanced interpreter designed for scientific computing
:::

## 1.1 IPython: Your Computational Laboratory

While you could use the basic Python interpreter by typing `python`, **IPython** (type `ipython` at the terminal instead) transforms your terminal into a powerful environment for scientific computing and **exploratory data analysis** (EDA). Think of it as the difference between an basic amateur telescope and one with adaptive optics — both observe, but one is designed for serious scientific work. Let's see why every professional computational astrophysicist prefers IPython over the basic Python **REPL**.

:::{margin}
**exploratory data analysis** (EDA)
A systematic approach to investigating datasets through visualization and summary statistics to uncover patterns, detect anomalies, and identify relationships between variables before formal modeling or hypothesis testing.
:::

### Launching Your Laboratory

First, ensure you're in the right environment, then launch IPython. Here's how to do it properly:

:::{margin}
**REPL**
Read-Eval-Print Loop - an interactive programming environment
:::

:::{note}
**Terminal Commands vs Python Code**

Throughout this book:
- Lines starting with `$` are terminal/shell commands
- Lines starting with `In []:` are IPython commands
- Regular code blocks are Python scripts
:::

```bash
# Terminal commands (type these in your terminal, not in Python!)
$ conda activate astr596
$ ipython

# You'll see something like this appear:
Python 3.11.5 | packaged by conda-forge
IPython 8.14.0 -- An enhanced Interactive Python
In [1]: 
```

Notice the prompt says `In [1]:` instead of `>>>` (which is what basic Python shows). This numbering system is your first hint that IPython is different — it remembers everything. Each command you type gets a number, making it easy to reference previous work.

### The Power of Memory

IPython maintains a complete history of your session, accessible through special variables:

```{code-cell} ipython3
# Type these commands one at a time in IPython
import numpy as np

# Hubble constant in km/s/Mpc
H0 = 70.0  

# Calculate Hubble time (age of universe for flat, matter-only cosmology)
t_hubble = 1.0 / H0  # in units where c = 1
t_hubble_gyr = t_hubble * 978  # Convert to Gyr (approximation)

print(f"Hubble time: {t_hubble_gyr:.1f} Gyr")

# In IPython, you can reference previous work:
print("In IPython, Out[n] stores outputs, In[n] stores inputs")
print("The underscore _ references the last output")
```

```{code-cell} ipython3
# Type these commands one at a time in IPython
# Kepler's Third Law: Orbital Mechanics
# Another fundamental calculation - orbital periods
a_earth = 1.496e13  # cm (1 AU)
M_sun = 1.989e33    # g (solar mass)
G = 6.674e-8        # cm^3 g^-1 s^-2

# Calculate Earth's orbital period using Kepler's third law
# P^2 = 4π^2 a^3 / (GM)

import math

P_squared = (4 * math.pi**2 * a_earth**3) / (G * M_sun)

P_seconds = math.sqrt(P_squared)
P_days = P_seconds / (24 * 3600)

print(f"\nEarth's orbital period: {P_days:.1f} days")

print(f"(Actual: 365.25 days - pretty close!)")
```

::::{hint} 🤔 Check Your Understanding
:class: dropdown

What's the difference between `In[5]` and `Out[5]` in IPython?

:::{admonition} Solution
:class: tip, dropdown

- `In[5]` contains the actual text/code you typed in cell 5 (as a string)
- `Out[5]` contains the result/value that cell 5 produced (if any)

For example:

- `In[5]` might be `"np.sqrt(2)"`
- `Out[5]` would be `1.4142135623730951`

This history system lets you reference and reuse previous computations without retyping — crucial when analyzing large datasets or iterating on algorithms.
:::
::::

### Tab Completion: Your Exploration Tool

Tab completion helps you discover astronomical libraries without memorizing everything:

```{code-cell} ipython3
import numpy as np

# In real IPython, type: np.<TAB>
# Shows all available functions

available_functions = [item for item in dir(np) if not item.startswith('_')]
print("NumPy contains:", len(available_functions), "functions/attributes")
print("Sample:", available_functions[:10])

# To see functions for FFT analysis (np.*fft*?<TAB> in IPython):
fft_functions = [item for item in dir(np) if 'fft' in item.lower()]
print("\nFFT-related functions:", fft_functions)
```

### Magic Commands: IPython's Superpowers

:::{margin}
**magic command**
IPython commands prefixed with % providing special functionality
:::

IPython's **magic commands** give you capabilities far beyond standard Python. Here's a practical example comparing different methods to calculate stellar magnitudes:

```{code-cell} python
# In IPython, you would use: %timeit
# Here we simulate timing different approaches to photometry calculations

import timeit
import numpy as np

# Method 1: List comprehension for magnitude calculation
def mag_list_comp(fluxes):
    return [-2.5 * np.log10(f/3631) for f in fluxes if f > 0]

# Method 2: NumPy vectorized
def mag_numpy(fluxes):
    return -2.5 * np.log10(fluxes[fluxes > 0] / 3631)

# Generate test data: stellar fluxes in Jansky
np.random.seed(42)
test_fluxes = np.random.lognormal(7, 1, 1000)

# Time both methods
time1 = timeit.timeit(lambda: mag_list_comp(test_fluxes), number=100)
time2 = timeit.timeit(lambda: mag_numpy(test_fluxes), number=100)

print(f"List comprehension: {time1*10:.4f} ms per call")
print(f"NumPy vectorized:   {time2*10:.4f} ms per call")
print(f"\nNumPy is {time1/time2:.1f}x faster for photometry!")
```

:::{warning} 🚨 Common Bug Alert: Platform-Specific Timing
:class: dropdown

Timing results vary significantly between machines due to:

- CPU speed and architecture (Intel vs AMD vs ARM)
- NumPy compilation (MKL vs OpenBLAS vs BLIS)
- System load and background processes
- Python version and compilation options

**Best Practice**: Always benchmark on your target system (laptop vs cluster)
:::

### Getting Help Instantly

IPython makes documentation accessible without leaving your workflow:

```{code-cell} python
# Demonstrating IPython's help system
print("In IPython, use ? for quick help:")
print("  np.fft.fft?  - shows documentation")
print("  np.fft.fft?? - shows source code (if available)")
print("\nExample documentation for np.fft.fft:")
print("  Compute the one-dimensional discrete Fourier Transform.")
print("  Used for: spectral analysis, period finding, filtering")
print("  Returns: complex array of Fourier coefficients")
```

:::{important} 💡 Computational Thinking: Interactive Exploration
:class: dropdown

The ability to quickly test ideas and explore APIs interactively is fundamental to computational astrophysics. IPython's environment encourages experimentation:

**Explore** → **Test** → **Refine** → **Validate**

This rapid iteration cycle is how algorithms are born and bugs are discovered. You might:
1. Explore a new spectral library's API
2. Test different periodogram algorithms
3. Refine parameters for optimal performance
4. Validate against known variable stars

This pattern appears everywhere: from testing cosmological simulations to debugging telescope control software.
:::

### Managing Your Workspace

```{code-cell} python
import sys
import numpy as np

# Create some astronomical variables for demonstration
redshift = 0.5
luminosity_distance = 2590.3  # Mpc for z=0.5, standard cosmology
apparent_mag = 18.5
filters = ['u', 'g', 'r', 'i', 'z']

# Show variables (simulating %who in IPython)
current_vars = [var for var in dir() 
                if not var.startswith('_') and not var.startswith('__')]
print("Variables in workspace (%who in IPython):")
print(", ".join(current_vars[:8]))

# Detailed info (simulating %whos in IPython)
print("\nDetailed variable info (%whos in IPython):")
print(f"{'Variable':<20} {'Type':<15} {'Value/Info'}")
print("-"*55)
print(f"{'redshift':<20} {'float':<15} {redshift}")
print(f"{'luminosity_distance':<20} {'float':<15} {luminosity_distance:.1f} Mpc")
print(f"{'filters':<20} {'list':<15} {len(filters)} SDSS filters")
```

:::{important} 🎯 Why This Matters: Your First Day of Research
:class: dropdown

Next week when your advisor hands you a FITS file and says "can you check if this galaxy is interesting?", you'll open IPython and in 5 minutes:
- Load the data with `astropy.io.fits`
- Check the header with tab completion
- Plot a quick spectrum with `%matplotlib`
- Test if it's a quasar with `z = wavelength_obs/wavelength_rest - 1`

Without IPython, this becomes a 30-minute script-writing exercise. With IPython, you'll have an answer before your advisor finishes their coffee.
:::

:::{tip} 🌟 The More You Know: The Reproducibility Crisis in Science
:class: dropdown

A 2018 study attempted to obtain data and code from 204 computational papers in *Science* magazine, finding that materials were available for only 26% of articles, with many of those still having reproducibility issues ([Stodden et al., 2018](https://doi.org/10.1073/pnas.1708290115)). Common reproducibility barriers included:

- Missing dependencies and software versions
- Hardcoded file paths to data
- Undocumented parameter choices
- Missing random seeds for simulations

The Sloan Digital Sky Survey (SDSS) has addressed reproducibility through versioned data releases (DR1 through DR18), standardized pipelines, and detailed documentation of software versions. Their approach ensures that a spectrum analyzed with specific data release tools will give consistent results years later.

Tools like IPython's `%history` and `%save` commands create an audit trail that helps ensure your future self (and others) can reproduce your analysis.

**Additional Resources:**

- [Nature's reproducibility survey (2016)](https://doi.org/10.1038/533452a) - More than 70% of researchers have failed to reproduce another scientist's experiments
- [SDSS Data Release Documentation](https://www.sdss.org/dr18/)
:::

---

## Section 1.2: Understanding Python's Hidden Machinery

When you type `import astropy`, a complex process unfolds behind the scenes. Understanding this machinery is the difference between guessing why `ImportError: No module named 'astropy.cosmology'` fails and knowing exactly how to fix it.

**Why Import Systems Matter in Astronomy:** Modern astronomical analysis relies on dozens of specialized packages. A typical spectroscopic pipeline might import `astropy` for FITS handling, `specutils` for spectral analysis, `astroquery` for archive access, and `matplotlib` for visualization. When you're reducing data at 3 AM at the telescope, understanding how Python finds and loads these packages can save your observing run.

### The Import System Exposed

Python's **import system** is like a librarian searching through a card catalog. When you request a book (module), the librarian (Python) has a specific search order (sys.path) and won't randomly guess where to look. This systematic approach ensures consistency but can cause confusion when multiple versions exist.

:::{margin}
**import system**
Python's mechanism for loading code from external files
:::

Let's peek behind the curtain to understand this process, particularly for astronomical libraries. Every time Python starts, it builds a search path based on your environment, installation method, and current directory. This path determines everything about which code gets loaded:

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

# Check for astronomy packages
print("\nChecking for key astronomy packages:")
for pkg in ['numpy', 'astropy', 'matplotlib']:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg:12} version {version}")
    except ImportError:
        print(f"  ✗ {pkg:12} NOT INSTALLED")

# Virial Theorem Application: Galaxy Cluster Temperature
print("\n--- Computational Example: Virial Theorem ---")
# For a self-gravitating system in equilibrium: 2K + U = 0
# This gives us the virial temperature of a galaxy cluster

# Galaxy cluster parameters (CGS units)
M_cluster = 1e15 * 1.989e33  # 10^15 solar masses in grams
R_cluster = 1 * 3.086e24     # 1 Mpc in cm (1 Mpc = 10^6 pc = 3.086e24 cm)
m_proton = 1.673e-24         # Proton mass in g
k_B = 1.381e-16              # Boltzmann constant in erg/K
G = 6.674e-8                 # Gravitational constant

# Virial temperature: kT ~ GMm_p/(3R)
T_virial = (G * M_cluster * m_proton) / (3 * k_B * R_cluster)
T_virial_keV = k_B * T_virial / 1.602e-9  # Convert to keV

print(f"Galaxy cluster mass: {M_cluster/1.989e33:.1e} M☉")
print(f"Cluster radius: {R_cluster/3.086e24:.1f} Mpc")
print(f"Virial temperature: {T_virial:.2e} K")
print(f"Temperature in keV: {T_virial_keV:.1f} keV")
print("(Typical observed: 2-10 keV - we're in the right ballpark!)")
```

The search path (**sys.path**) acts as Python's roadmap for finding modules. Think of it like the light path through a telescope: light follows a specific route through primary mirror, secondary mirror, and eyepiece. Similarly, Python follows sys.path in order, using the first matching module it finds. This is why having multiple versions of the same package can cause confusion—Python doesn't look for the "best" version, just the first one.

:::{margin}
**sys.path**
Python's list of directories to search when importing modules
:::

This ordered search has important implications for astronomical software development. If you have a file named `astropy.py` in your current directory, Python will import that instead of the real astropy package. This is a common source of mysterious errors when students name their test scripts after the packages they're learning.

**cache**
A temporary storage area that keeps frequently accessed data for quick retrieval, avoiding repeated expensive operations
:::

:::{important} 💡 Computational Thinking: The Import Resolution Algorithm
:class: dropdown

When Python executes `import astropy.cosmology`, it follows this algorithm:

1. **Check cache**: Is `astropy` already in `sys.modules`?
2. **Search paths**: For each directory in `sys.path`:
   - Look for `astropy/` directory with `__init__.py`
   - Look for `astropy.py` file
   - Look for compiled extension `astropy.so/.pyd`
3. **Load module**: Execute the module code once
4. **Cache result**: Store in `sys.modules` to avoid reloading
5. **Access submodule**: Repeat for `cosmology` within `astropy`

Understanding this algorithm helps you debug why `import astropy` works but `from astropy import cosmology` might fail (missing subpackage installation).
:::

```{mermaid}
flowchart TD
    A[import astropy.cosmology] --> B{Is 'astropy' in<br/>sys.modules cache?}
    B -->|Yes| H[Use cached module]
    B -->|No| C[Search sys.path directories]
    C --> D{Found<br/>astropy/?}
    D -->|No| E{Found<br/>astropy.py?}
    E -->|No| F{Found<br/>astropy.so?}
    F -->|No| G[ImportError!]
    D -->|Yes| I[Load & execute module]
    E -->|Yes| I
    F -->|Yes| I
    I --> J[Store in sys.modules]
    J --> K[Look for 'cosmology' submodule]
    H --> K
    
    style G fill:#ff6b6b
    style I fill:#51cf66
    style J fill:#339af0
```

### Debugging Import Problems

Here's a diagnostic function specifically for astronomical libraries:

:::{tip} **Stage 1: Check Environment**
:class: dropdown

```{code-cell} ipython3
def check_astro_environment():
    """Verify we're in the correct astronomy Python environment."""
    import sys
    
    env_path = sys.executable
    env_name = "Unknown"
    
    # Extract environment name from path
    if 'envs' in env_path:
        parts = env_path.split('/')
        if 'envs' in parts:
            idx = parts.index('envs')
            env_name = parts[idx + 1] if idx + 1 < len(parts) else "base"
    
    # Check for astronomy-specific environment
    if 'astr' in env_name.lower() or 'astro' in env_name.lower():
        return True, f"✓ Astronomy environment: {env_name}"
    elif 'base' in env_name:
        return False, f"⚠️ In base environment (not recommended)"
    else:
        return None, f"🔍 Current environment: {env_name}"

status, message = check_astro_environment()
print(message)
```

:::

:::{tip} **Stage 2: Test Astronomy Imports**
:class: dropdown

```{code-cell} ipython3
def test_astro_imports():
    """Test critical astronomy package imports with versions."""
    critical_packages = {
        'numpy': 'Numerical computing',
        'astropy': 'Core astronomy library',
        'scipy': 'Scientific algorithms',
        'matplotlib': 'Plotting',
        'pandas': 'Data manipulation'
    }
    
    print("Astronomy Package Status:")
    print("-" * 50)
    
    for pkg, description in critical_packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', '???')
            location = getattr(mod, '__file__', 'built-in')
            print(f"✓ {pkg:12} v{version:8} - {description}")
        except ImportError as e:
            print(f"✗ {pkg:12} MISSING    - {description}")
            print(f"  Fix: conda install -c conda-forge {pkg}")
    
    return True

test_astro_imports()
```

:::

:::{tip} **Stage 3: Verify Subpackages**
:class: dropdown

```{code-cell} python
def check_astropy_subpackages():
    """Check if key astropy subpackages are accessible."""
    subpackages = [
        ('astropy.io.fits', 'FITS file I/O'),
        ('astropy.coordinates', 'Coordinate transformations'),
        ('astropy.cosmology', 'Cosmological calculations'),
        ('astropy.units', 'Unit conversions'),
        ('astropy.time', 'Time systems')
    ]
    
    print("\nAstropy Subpackage Check:")
    print("-" * 50)
    
    for subpkg, description in subpackages:
        try:
            __import__(subpkg)
            print(f"✓ {subpkg:25} - {description}")
        except ImportError:
            print(f"✗ {subpkg:25} - {description}")
    
    return True

# Only run if astropy is available
try:
    import astropy
    check_astropy_subpackages()
except ImportError:
    print("Astropy not installed - skipping subpackage check")
```

:::

::::{hint} 🤔 Check Your Understanding
:class: dropdown

You get `ModuleNotFoundError: No module named 'astropy.io.fits'`. What are three possible causes and their solutions?

:::{admonition} Solution
:class: tip, dropdown

Three possible causes and solutions:

1. **Incomplete astropy installation**: Some distributions split astropy
   - Solution: `conda install -c conda-forge astropy` (get complete package)

2. **Old astropy version**: Pre-1.0 versions had different structure
   - Check: `python -c "import astropy; print(astropy.__version__)"`
   - Solution: `conda update astropy`

3. **Mixed pip/conda installation**: Conflicting installations
   - Check: `conda list astropy` vs `pip list | grep astropy`
   - Solution: Stick to conda for scientific packages

The most robust fix:
```bash
conda create -n astro_clean python=3.11
conda activate astro_clean
conda install -c conda-forge astropy numpy scipy matplotlib
```

:::
::::

### Multiple Pythons: A Common Disaster

Most systems have multiple Python installations, especially on shared computing clusters:

```{code-cell} ipython3
from pathlib import Path
import sys

# Common Python locations on astronomy systems
astronomy_pythons = {
    'System Python': '/usr/bin/python3',
    'Conda (base)': '~/miniforge3/bin/python',
    'Conda (astro env)': '~/miniforge3/envs/astro/bin/python',
    'AstroConda': '~/astroconda3/bin/python',
    'Homebrew (Mac)': '/usr/local/bin/python3',
    'Module system': '/software/astro/python/bin/python'
}

print("Common Python locations on astronomy systems:")
print("-" * 50)
for name, path in astronomy_pythons.items():
    expanded_path = Path(path).expanduser()
    exists = "✓" if expanded_path.exists() else "✗"
    current = "← CURRENT" if str(expanded_path) in sys.executable else ""
    print(f"  {exists} {name:20} {path:30} {current}")

print("\n⚠️  This is why 'conda activate' is crucial!")
print("📚 Each Python has its own packages - they don't share!")
```

:::{warning} 🚨 Common Bug Alert: The Cluster Python Confusion
:class: dropdown

**Symptom**: Code works on laptop but fails on computing cluster

**Cause**: Different Python modules loaded by default

**Example**: On many clusters:
```bash
$ module load python  # Loads system Python
$ python script.py    # Uses wrong Python!
```

**Fix**: Always use explicit paths or conda:
```bash
$ conda activate astr596
$ which python  # Verify it's YOUR Python
$ python script.py
```

**Best Practice**: Add to your job scripts:
```bash
#!/bin/bash
#SBATCH --job-name=astro_analysis

# Always activate your environment first!
source ~/miniforge3/etc/profile.d/conda.sh
conda activate astr596
python your_script.py
```

:::

## 1.3 Jupyter Notebooks: Beautiful Disasters Waiting to Happen

:::{margin}
**Jupyter notebook**
Web-based platform that executes code in cells while maintaining state
:::

**Jupyter notebooks** seem perfect for scientific computing and data analysis - you can mix code, plots, and explanations in one document. You'll see them in tutorials and even published papers. However, they harbor dangerous flaws that can corrupt your scientific results. You are allowed to use them for `Short Project 1` to understand their appeal - and also because its likely the only way you've been taught Python, but then you must abandon them for more robust approaches.

### The Seductive Power of Notebooks

To start Jupyter (after activating your environment):

```bash
# Terminal commands:
$ conda activate astr596
$ jupyter lab

# Opens browser at http://localhost:8888
# You can create notebooks, write code, see plots inline
```

### The Hidden State Monster

The most insidious problem: notebooks maintain hidden state between cell executions. Here's an astronomical example:

```{code-cell} ipython3
# Simulating notebook cells with execution order problems

# Cell 1: Set cosmological parameters
H0 = 70.0  # Hubble constant [km/s/Mpc]
omega_m = 0.3  # Matter density
print(f"Cell 1: Set H0 = {H0}, Ωm = {omega_m}")

# Cell 2: Define distance calculation
def luminosity_distance(z):
    """Calculate luminosity distance (simplified flat universe)."""
    # This captures H0 and omega_m from when function was defined!
    c = 3e5  # km/s
    return (c * z / H0) * (1 + z/2 * (1 - omega_m))  # Approximation

print(f"Cell 2: Defined function with H0 = {H0}")

# Cell 3: User updates cosmology for Planck results
H0 = 67.4  # Updated Hubble constant
omega_m = 0.315  # Updated matter density
print(f"Cell 3: Updated to Planck cosmology H0 = {H0}")

# Cell 4: Calculate distance - which H0 does this use?
z_galaxy = 1.0
d_L = luminosity_distance(z_galaxy)
print(f"Cell 4: Distance to z=1 galaxy = {d_L:.0f} Mpc")
print(f"  But function still uses OLD H0 = 70!")
print(f"  This gives WRONG distance by {(70/67.4-1)*100:.1f}%!")
```

::::{important} 🔧 Debug This!
:class: dropdown

An astronomy student's notebook analyzes variable star data:

```ipython3
Cell 1: periods = [0.5, 1.2, 2.3]  # days
        magnitudes = [12.5, 13.1, 11.8]

Cell 2: mean_period = np.mean(periods)
        mean_mag = np.mean(magnitudes)

Cell 3: periods.append(5.4)  # Add Cepheid
        magnitudes.append(10.2)

Cell 4: print(f"Average period: {mean_period:.2f} days")
        print(f"Average magnitude: {mean_mag:.2f}")

Cell 5: # Classify based on period
        if mean_period < 1:
            print("RR Lyrae stars")
        elif mean_period < 10:
            print("Cepheids")
```

They run cells: 1, 2, 3, 4, 2, 4, 5. What's the classification? Is it correct?

:::{admonition} Solution
:class: solution, dropdown

**Execution trace:**

1. Cell 1: `periods = [0.5, 1.2, 2.3]`, `magnitudes = [12.5, 13.1, 11.8]`
2. Cell 2: `mean_period = 1.33`, `mean_mag = 12.47`
3. Cell 3: Lists become `[0.5, 1.2, 2.3, 5.4]` and `[12.5, 13.1, 11.8, 10.2]`
4. Cell 4: Prints "Average period: 1.33 days" (OLD value!)
5. Cell 2 again: `mean_period = 2.35`, `mean_mag = 11.90` (NEW values)
6. Cell 4 again: Prints "Average period: 2.35 days"
7. Cell 5: Classifies as "Cepheids" (1 < 2.35 < 10)

**Problems:**

- Classification uses updated mean (2.35) but that includes the Cepheid itself!
- This is circular reasoning - using a Cepheid to classify as Cepheids
- The correct mean without the Cepheid is 1.33 days (RR Lyrae range)

**This demonstrates how notebook state corruption leads to incorrect scientific conclusions!**
:::
::::

```{mermaid}
graph LR
    subgraph "What You Think Happens"
        A1[Cell 1: Set variables] --> A2[Cell 2: Calculate mean]
        A2 --> A3[Cell 3: Add data]
        A3 --> A4[Cell 4: Print results]
    end
    
    subgraph "What Actually Happened"
        B1[Cell 1: periods=[0.5,1.2,2.3]] --> B3[Cell 3: periods=[0.5,1.2,2.3,5.4]]
        B3 --> B2a[Cell 2: mean=1.33]
        B2a --> B4a[Cell 4: Prints 1.33]
        B4a --> B2b[Cell 2 again: mean=2.35]
        B2b --> B4b[Cell 4: Prints 2.35]
    end
    
    subgraph "Hidden State"
        HS[Variables persist and<br/>accumulate between runs!]
    end
    
    style HS fill:#ff6b6b
    style B2b fill:#ffd43b
```

### Memory Accumulation in Data Analysis

```{code-cell} ipython3
import sys
import numpy as np

# Simulating repeated cell execution with telescope data
spectra_list = []

print("Initial memory state")

# First run: Load night 1 data
for i in range(100):
    # Simulate 100 spectra, 4000 wavelength bins each
    spectrum = np.random.randn(4000)
    spectra_list.append(spectrum)
    
# Calculate memory usage
n_spectra = len(spectra_list)
n_pixels = n_spectra * 4000
memory_mb = (n_pixels * 8) / (1024 * 1024)  # 8 bytes per float64
print(f"After night 1: {n_spectra} spectra, ~{memory_mb:.1f} MB")

# Second run (cell executed again) - adds MORE data!
for i in range(100):
    spectrum = np.random.randn(4000)
    spectra_list.append(spectrum)
    
n_spectra = len(spectra_list)
n_pixels = n_spectra * 4000
memory_mb = (n_pixels * 8) / (1024 * 1024)
print(f"After re-run: {n_spectra} spectra, ~{memory_mb:.1f} MB")

print("\n⚠️  Each run ADDS data - notebook doesn't reset!")
print("📈 With real spectroscopic surveys (millions of spectra),")
print("   this crashes your kernel and loses all work!")
```

:::{tip} 🌟 The More You Know: The $125 Billion Excel Error
:class: dropdown

In 2013, graduate student Thomas Herndon couldn't reproduce the results from a highly influential economics paper by Carmen Reinhart and Kenneth Rogoff. This paper, ["Growth in a Time of Debt,"](https://www.aeaweb.org/articles?id=10.1257/aer.100.2.573) had been cited by politicians worldwide to justify austerity policies affecting millions of people.

When Herndon finally obtained the original Excel spreadsheet, he discovered a coding error: five countries were accidentally excluded from a calculation due to an Excel formula that didn't include all rows. This simple mistake skewed the results, showing that high debt caused negative growth when the corrected analysis showed much weaker effects ([Herndon, Ash, and Pollin, 2014](https://doi.org/10.1093/cje/bet075)). The implications were staggering — this spreadsheet error influenced global economic policy.

Just like hidden state in Jupyter notebooks, the error was invisible in the final spreadsheet. The lesson? Computational transparency and reproducibility aren't just academic exercises — they have real-world consequences. Always make your computational process visible and reproducible!
:::

:::{tip} 🌟 The More You Know: The BICEP2 "Discovery" Retraction
:class: tip, dropdown

In March 2014, the BICEP2 collaboration announced the detection of primordial gravitational waves via B-mode polarization in the cosmic microwave background - evidence for cosmic inflation and a likely Nobel Prize discovery.

By September 2014, they retracted the claim. The issue? Dust contamination in their analysis pipeline. While not directly a notebook problem, it illustrates how hidden state and assumptions in analysis workflows can corrupt results. The team had underestimated galactic dust contribution, and their analysis pipeline didn't properly propagate this uncertainty through all steps.

The lesson extends to notebooks: when your analysis state isn't clear and reproducible, you can mistake foreground contamination for cosmological signals. The BICEP2 team now publishes their entire analysis pipeline as reproducible scripts, not notebooks.

This $10+ million experiment's false discovery shows why computational reproducibility matters. Always use scripts for final analysis!
:::

### The Notebook-to-Script Transition

:::{margin}
**script**
Plain text file with Python code that executes top-to-bottom, the same way every time
:::

After Project 1, we'll abandon notebooks for **scripts**. Here's why scripts are superior for astronomical research:

:::{list-table} Script vs Notebook: Astronomical Data Analysis
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
* - Large Data Processing
  - Memory leaks common
  - Controlled memory usage
* - Cluster Jobs
  - Can't run with SLURM
  - Easy batch submission
* - Version Control
  - JSON mess with outputs
  - Clean text diffs
* - Pipeline Integration
  - Nearly impossible
  - Straightforward
* - Reproducible Results
  - Often impossible
  - Guaranteed
:::

:::{important} 🎯 Why This Matters: Your Paper's Data Analysis Must Be Bulletproof
:class: dropdown

When you submit your first paper to ApJ or MNRAS. The referee may ask: "Can you verify that your period-finding algorithm consistently identifies the 0.5673-day period in your RR Lyrae sample?"

With a **notebook**, you'll panic:

- Which cells did you run to get that result?
- Did you update the detrending before or after finding that period?
- Your memory says one thing, but re-running gives different periods

With a **script**, you'll confidently respond:

- "Run `python find_periods.py --input data/rr_lyrae.csv --method lomb-scargle`"
- "Results are identical: P = 0.5673 ± 0.0002 days"
- "See our GitHub repository for version-controlled analysis code"

Real example: The TESS mission requires all planet discoveries to be verified with independent analysis pipelines. These are **always scripts**, never notebooks. Why? Because when claiming you've found an Earth-like exoplanet, there's no room for hidden state corruption. Your career depends on reproducible results.

Remember: **Notebooks are for exploration. Scripts are for science.**
:::

```{mermaid}
graph TD
    subgraph "Notebook Execution"
        N1[Any cell] --> N2[Any cell]
        N2 --> N3[Any cell]
        N3 --> N1
        N1 -.->|Hidden State| NS[(Persistent Memory)]
        N2 -.->|Hidden State| NS
        N3 -.->|Hidden State| NS
    end
    
    subgraph "Script Execution"
        S1[Line 1] --> S2[Line 2]
        S2 --> S3[Line 3]
        S3 --> S4[Line 4]
        S4 --> S5[Fresh start each run]
    end
    
    style NS fill:#ff6b6b
    style S5 fill:#51cf66
```
:::{important} 💡 Computational Thinking: Reproducible Analysis Pipelines
:class: dropdown

Modern astronomical surveys process terabytes of data through complex pipelines. Consider the Vera Rubin Observatory (LSST):

**Data Flow**: Raw images → Calibration → Source detection → Photometry → Catalogs

Each step must be:

- **Deterministic**: Same input = same output
- **Versioned**: Track software versions
- **Logged**: Record all parameters
- **Testable**: Unit tests for each component

Notebooks fail at every requirement. Scripts excel at all of them. This is why major surveys use workflow managers (Snakemake, Pegasus) orchestrating Python scripts, never notebooks.

Remember: *"If it's not reproducible, it's not science."*
:::

---

## 1.4 Scripts: Write Once, Run Anywhere (Correctly)

Python scripts are simple text files containing Python code, executed from top to bottom, the same way every time. No hidden state, no ambiguity, just predictable execution - essential for scientific computing.

### From IPython to Scripting

Start by experimenting in IPython with a real astronomical calculation:

```{code-cell} ipython3
# Quick calculation in IPython: Schwarzschild radius
import numpy as np

# Constants (CGS units)
G = 6.67430e-8   # cm^3 g^-1 s^-2
c = 2.99792458e10  # cm/s
M_sun = 1.98847e33  # g

# Calculate for stellar-mass black hole
M_bh = 10 * M_sun  # 10 solar mass black hole
r_s = 2 * G * M_bh / c**2

print(f"Schwarzschild radius for 10 M☉ black hole: {r_s/1e5:.1f} km")
```

Now create a proper script. Save this as `schwarzschild.py`:

:::{admonition} Complete schwarzschild.py Script
:class: dropdown

```{code-cell} python
#!/usr/bin/env python
"""
Calculate Schwarzschild radii for various astrophysical objects.

This module provides functions to calculate the Schwarzschild radius
(event horizon) for black holes of different masses.
"""

import numpy as np
import argparse

# Physical constants (CGS units)
G = 6.67430e-8      # Gravitational constant [cm^3 g^-1 s^-2]
c = 2.99792458e10   # Speed of light [cm/s]
M_sun = 1.98847e33  # Solar mass [g]

def schwarzschild_radius(mass_g):
    """
    Calculate Schwarzschild radius for a given mass.
    
    The Schwarzschild radius is the radius of the event horizon
    for a non-rotating black hole.
    
    Parameters
    ----------
    mass_g : float
        Mass in grams
        
    Returns
    -------
    float
        Schwarzschild radius in centimeters
        
    Examples
    --------
    >>> r_s = schwarzschild_radius(10 * M_sun)
    >>> print(f"{r_s/1e5:.1f} km")
    29.5 km
    """
    if mass_g <= 0:
        raise ValueError("Mass must be positive")
    return 2 * G * mass_g / c**2

def classify_black_hole(mass_solar):
    """
    Classify black hole by mass.
    
    Parameters
    ----------
    mass_solar : float
        Mass in solar masses
        
    Returns
    -------
    str
        Classification (stellar, intermediate, supermassive)
    """
    if mass_solar < 100:
        return "Stellar-mass black hole"
    elif mass_solar < 1e5:
        return "Intermediate-mass black hole"
    else:
        return "Supermassive black hole"

def main():
    """Main execution function with example calculations."""
    
    # Example objects
    objects = {
        "Cygnus X-1": 21.2,           # Solar masses
        "GW150914 remnant": 62,       # First LIGO detection
        "Sagittarius A*": 4.154e6,    # Milky Way center
        "M87*": 6.5e9,                # First black hole image
    }
    
    print("Schwarzschild Radii of Famous Black Holes")
    print("=" * 50)
    
    for name, mass_solar in objects.items():
        mass_g = mass_solar * M_sun
        r_s = schwarzschild_radius(mass_g)
        classification = classify_black_hole(mass_solar)
        
        # Convert to appropriate units
        if r_s < 1e5:
            r_s_display = f"{r_s:.1f} cm"
        elif r_s < 1e8:
            r_s_display = f"{r_s/1e5:.1f} km"
        elif r_s < 1.5e13:  # 1 AU in cm
            r_s_display = f"{r_s/1e11:.1f} million km"
        else:
            r_s_display = f"{r_s/1.496e13:.2f} AU"
            
        print(f"\n{name}:")
        print(f"  Mass: {mass_solar:.2e} M☉")
        print(f"  Type: {classification}")
        print(f"  Event horizon: {r_s_display}")

# This pattern makes the script both runnable and importable
if __name__ == "__main__":
    main()
```

:::

### The `if __name__ == "__main__"` Pattern for Python Scripts

:::{margin}
**__name__**
Python variable that equals "__main__" when run directly, or the module name when imported
:::

This crucial pattern makes your astronomy code both runnable and importable:

```{code-cell} ipython3
# Understanding __name__ in scientific context
def planck_function(wavelength_nm, temperature_K):
    """
    Calculate Planck function for blackbody radiation.
    
    Used for stellar spectra modeling.
    """
    import numpy as np
    
    # Constants (CGS units)
    h = 6.62607015e-27  # Planck constant [erg⋅s]
    c = 2.99792458e10   # Speed of light [cm/s]
    k_B = 1.380649e-16  # Boltzmann constant [erg/K]
    
    # Convert wavelength to centimeters
    wavelength = wavelength_nm * 1e-7
    
    # Planck function
    exp_term = np.exp(h * c / (wavelength * k_B * temperature_K))
    B = (2 * h * c**2 / wavelength**5) / (exp_term - 1)
    
    return B

# Python sets __name__ based on how the file is used
print(f"Current __name__ is: {__name__}")

if __name__ == "__main__":
    # This runs ONLY when script is executed directly
    # Perfect for testing your functions
    print("\nTesting Planck function for the Sun:")
    
    # Sun's peak wavelength should be ~500 nm (green)
    wavelengths = [400, 500, 600, 700]  # nm (violet to red)
    T_sun = 5778  # K
    
    for wl in wavelengths:
        B = planck_function(wl, T_sun)
        print(f"  λ={wl}nm: B={B:.2e} erg/cm²/s/sr/cm")
        
    # When others import this file, this test code won't run
    # They can just use: from your_module import planck_function
```

::::{hint} 🤔 Check Your Understanding
:class: dropdown

Why is the `if __name__ == "__main__"` pattern crucial for scientific instrument control software?

:::{tip} Solution
:class: dropdown

In scientific instrument control and data acquisition:

1. **Safety**: Test functions without activating equipment
   ```python
   def move_to_position(x, y, z):
       # Moves expensive/dangerous equipment!
       pass
   
   if __name__ == "__main__":
       # Safe testing with simulated coordinates
       print("Testing movement (not really moving):")
       # move_to_position(10.0, 20.0, 5.0)  # Commented for safety
   ```

2. **Module Testing**: Test detector readout without taking real data
3. **Pipeline Components**: Each script works standalone or in pipeline
4. **Calibration Scripts**: Can process test data or real observations

Real example from laboratory operations:
```python
# focus_control.py
def optimize_focus(detector, n_steps=10):
    """Find optimal focus position."""
    # ... complex focusing routine ...
    return best_focus

if __name__ == "__main__":
    # Test with simulated detector, not real hardware
    from simulator import FakeDetector
    test_detector = FakeDetector()
    focus = optimize_focus(test_detector, n_steps=5)
    print(f"Test focus position: {focus}")
```

:::
::::

---

## 1.5 Creating Reproducible Environments

:::{margin}
**conda**
Package and environment manager for isolated Python installations
:::

Your astronomical analysis depends on its **environment** — Python version, astropy version, even NumPy's linear algebra backend. Creating reproducible environments ensures your code produces identical results on any system, from your laptop to a supercomputer.

### The Conda Solution

**Conda** creates isolated environments — separate Python installations with their own packages. This is essential for research where different projects need different package versions:

```bash
# Essential conda commands for astronomy

# Create environment for photometry project
$ conda create -n photometry python=3.11
$ conda activate photometry
$ conda install -c conda-forge numpy scipy astropy photutils

# Create separate environment for spectroscopy
$ conda create -n spectroscopy python=3.10
$ conda activate spectroscopy  
$ conda install -c conda-forge numpy scipy astropy specutils

# List all your environments
$ conda env list

# Switch between projects
$ conda deactivate
$ conda activate photometry
```

:::{attention} 🎯 Why This Matters: Cluster Time = Money
:class: dropdown

Your university pays ~$0.10 per CPU-hour on the cluster. A typical research project uses 10,000+ hours. If your code crashes after 8 hours because of environment issues, you've wasted:

- $80 in compute time
- 8 hours of waiting
- Your queue priority (back to the end of the line!)

Get your environment right ONCE, and every subsequent run just works. This chapter will literally save you hundreds of dollars and weeks of time.
:::

### Environment Files: Share Your Exact Setup

Create an `environment.yml` file for your research project:

:::{tip} Example environment.yml for exoplanet research
:class: dropdown

```{code-cell} python
environment_yml = """name: exoplanet_analysis
channels:
  - conda-forge
  - astropy
dependencies:
  # Core scientific stack
  - python=3.11
  - numpy=1.24.*
  - scipy=1.11.*
  - matplotlib=3.7.*
  - pandas=2.0.*
  
  # Astronomy specific
  - astropy=5.3.*
  - astroquery=0.4.*
  - photutils=1.9.*
  - astroplan=0.9
  - ccdproc=2.4.*
  
  # Exoplanet packages
  - batman-package=2.4.*
  - emcee=3.1.*
  - corner=2.2.*
  
  # Development tools
  - ipython
  - jupyter
  - pytest
  
  # Additional packages via pip
  - pip
  - pip:
    - exoplanet==0.5.3
    - lightkurve==2.4.0
"""

print("environment.yml for exoplanet research:")
print(environment_yml)

print("\nCollaborators recreate your exact environment:")
print("$ conda env create -f environment.yml")
print("$ conda activate exoplanet_analysis")
```

:::

:::{warning} 🚨 Common Bug Alert: The AstroConda Deprecation
:class: dropdown

**Old tutorials may reference AstroConda** - it's deprecated!

**Don't use:**
```bash
$ conda config --add channels http://ssb.stsci.edu/astroconda  # DEPRECATED
```

**Instead use conda-forge and astropy channels:**
```bash
$ conda config --add channels conda-forge
$ conda config --add channels astropy
$ conda config --set channel_priority strict
```

**Why this matters:**

- AstroConda stopped updating in 2023
- conda-forge has more recent versions
- Better compatibility with modern Python

**For STScI tools specifically:**

```bash
$ conda install -c conda-forge stsynphot
$ pip install webbpsf  # Some tools installed only via pip
```

:::

### Proper Path Management

Stop hardcoding paths that break when moving between laptop and cluster:

```{code-cell} python
from pathlib import Path
import os

# BAD: Hardcoded path to telescope data
bad_path = '/Users/jane/Desktop/observations/2024-03-15/raw/science_001.fits'
print(f"BAD (hardcoded): {bad_path}")
print("  Problem: Doesn't exist on cluster or collaborator's machine!")

# GOOD: Relative to data directory
data_root = Path.cwd() / 'data'
night = '2024-03-15'
fits_file = data_root / night / 'raw' / 'science_001.fits'
print(f"\nGOOD (relative): {fits_file}")

# BETTER: Configuration-based approach
# In your script or config file:
DATA_DIR = Path(os.getenv('ASTRO_DATA', './data'))  
PROCESSED_DIR = Path(os.getenv('ASTRO_PROCESSED', './processed'))

def get_observation_path(date_obs, frame_num, data_type='raw'):
    """
    Construct path to observation file.
    
    Parameters
    ----------
    date_obs : str
        Observation date (YYYY-MM-DD)
    frame_num : int
        Frame number
    data_type : str
        'raw', 'reduced', or 'calibrated'
    """
    filename = f"science_{frame_num:03d}.fits"
    return DATA_DIR / date_obs / data_type / filename

# Usage
obs_path = get_observation_path('2024-03-15', 1)
print(f"\nBEST (configurable): {obs_path}")

# Check if file exists before processing
if obs_path.exists():
    print(f"  ✓ Ready to process: {obs_path.name}")
else:
    print(f"  ✗ File not found - check DATA_DIR environment variable")
    print(f"    Expected location: {obs_path}")
```

### Random Seed Control for Monte Carlo

Make Monte Carlo simulations reproducible by using a default random number generator seed value:

```{code-cell} ipython3
import numpy as np

def simulate_photometric_errors(n_stars=100, seed=42):
    """
    Simulate photometric measurements with realistic errors.
    
    Parameters
    ----------
    n_stars : int
        Number of stars to simulate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        True magnitudes, observed magnitudes, errors
    """
    # CRITICAL: Set seed for reproducibility
    np.random.seed(seed)
    
    # True stellar magnitudes (roughly following IMF)
    true_mags = np.random.uniform(12, 18, n_stars)
    
    # Photometric errors increase with magnitude
    # Typical for ground-based observations
    base_error = 0.01  # Bright star error
    errors = base_error * np.exp((true_mags - 12) / 4)
    
    # Add Gaussian noise
    observed_mags = true_mags + np.random.normal(0, errors)
    
    return true_mags, observed_mags, errors

# Run simulation multiple times - same results!
for run in range(3):
    true, obs, err = simulate_photometric_errors(n_stars=5, seed=42)
    print(f"Run {run+1}: First obs mag = {obs[0]:.3f} ± {err[0]:.3f}")

print("\n⚠️  Different seed = different results:")
true2, obs2, err2 = simulate_photometric_errors(n_stars=5, seed=137)
print(f"Seed 137: First obs mag = {obs2[0]:.3f} ± {err2[0]:.3f}")

print("\n🔍 Always document seeds in papers for reproducibility!")
```

:::{tip} 🌟 The More You Know: When Code Sharing Isn't Enough
:class: dropdown

In 2022, researchers attempted to run over 9,000 R scripts from 2,000+ publicly shared research datasets in the Harvard Dataverse repository ([Trisovic et al., 2022](https://doi.org/10.1038/s41597-022-01143-6)). The results were sobering: **74% of the R files failed to run** on the first attempt. Even after automated code cleaning to fix common issues, **56% still failed**.

The most common errors weren't complex algorithmic problems but basic issues:

- Missing package imports (`library()` statements)
- Hardcoded file paths that don't exist on other systems
- Dependencies on variables defined in other scripts
- Assuming specific working directories

What makes this particularly striking is that these researchers had already taken the crucial step of sharing their code—they were trying to do the right thing! But code availability alone doesn't guarantee reproducibility.

The study found that simple practices could have prevented most failures:

- Using relative paths instead of absolute paths
- Explicitly loading all required libraries at the script beginning
- Setting random seeds for any stochastic processes
- Including session information (R version, package versions)

This echoes our discussion about environments: sharing code without documenting its environment is like sharing a recipe without mentioning it's for a high-altitude kitchen. The code might be perfect, but it still won't work!
:::

---

## Section 1.6: Essential Debugging Strategies

:::{margin}
**defensive programming**
Writing code that anticipates and handles failures gracefully
:::

When your star cluster simulation produces unphysical orbits after running for 12 hours, your N-body dynamics code likely violates energy conservation, or your spectral reduction pipeline crashes during an observing run, systematic debugging saves the day. Debugging isn't just about fixing errors—**it's about understanding why they occurred and preventing them in the future**. Here are battle-tested strategies from both computational laboratories and observatories that will serve you throughout your research career, whether you're modeling galaxy formation, solving the equations of stellar structure, or processing telescope data.

**The Psychology of Debugging:** When code fails, especially if you're new to Python or transitioning from Jupyter notebooks, the problem is often a bug in your code—a typo, incorrect indentation, wrong variable name, or logical error. These are normal and expected! However, before diving into line-by-line debugging, a quick environment check can save you hours if the problem is actually a missing package or wrong Python version. Think of it as triage: the environment check takes 5 seconds and catches ~30% of problems immediately. The other 70%? Those are real bugs that require careful debugging.

### The Universal First Check

Before examining your algorithm for why virial equilibrium isn't converging, before questioning whether your Runge-Kutta integrator is correctly implemented, before doubting your understanding of the Saha equation—always, always verify your environment first. This simple discipline will save you hours of frustration:

**Why Environment Checks Matter:** Your code doesn't exist in isolation. It runs within a complex ecosystem of Python interpreters, installed packages, system libraries, and configuration files. A mismatch in any of these layers can cause mysterious failures. This is especially critical when moving code between laptops, workstations, and high-performance computing clusters where you run large simulations.

:::{tip} Full Environment Diagnostic Function Example
:class: dropdown

```{code-cell} ipython3
import sys
import os
from pathlib import Path

def astronomy_environment_check():
    """
    Universal debugging first check for astronomical Python.
    
    Returns
    -------
    dict
        Environment status and diagnostics
    """
    print("=" * 60)
    print("ASTRONOMICAL PYTHON ENVIRONMENT CHECK")
    print("=" * 60)
    
    # 1. Python location and version
    print("\n1. PYTHON INTERPRETER:")
    print(f"   Path: {sys.executable}")
    print(f"   Version: {sys.version.split()[0]}")
    
    # 2. Environment detection
    print("\n2. CONDA ENVIRONMENT:")
    env_path = sys.executable
    if 'conda' in env_path or 'miniforge' in env_path:
        if 'envs' in env_path:
            parts = env_path.split(os.sep)
            idx = parts.index('envs')
            env_name = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
            print(f"   ✓ Active environment: {env_name}")
        else:
            print(f"   ⚠ Base environment (not recommended)")
    else:
        print(f"   ✗ Not in conda environment")
    
    # 3. Critical astronomy packages
    print("\n3. ASTRONOMY PACKAGES:")
    critical = ['numpy', 'astropy', 'matplotlib']
    for pkg in critical:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', '?')
            print(f"   ✓ {pkg:12} {ver}")
        except ImportError:
            print(f"   ✗ {pkg:12} NOT FOUND")
    
    # 4. Data directories
    print("\n4. DATA PATHS:")
    data_dir = Path(os.getenv('ASTRO_DATA', './data'))
    print(f"   ASTRO_DATA: {data_dir.absolute()}")
    print(f"   Exists: {'✓' if data_dir.exists() else '✗'}")
    
    print("\n" + "=" * 60)
    return True

# Run the diagnostic
astronomy_environment_check()
```

:::

This diagnostic function should be your first line of defense. Copy it, modify it for your specific needs, and run it whenever something seems wrong. The few seconds it takes to run can save hours of misguided debugging.

### Using IPython's Debugger

When your code does crash — *and it will* — IPython's `%debug` magic command lets you perform a post-mortem examination. Think of it as having a time machine that takes you back to the moment of failure, letting you inspect all variables and understand exactly what went wrong:

**The Power of Post-Mortem Debugging:** Unlike adding print statements everywhere (which changes your code's behavior and timing), the debugger lets you explore the crash site without modifications. You can examine variables, test hypotheses, and even run new code in the context of the failure. This is invaluable when debugging complex algorithms where the error might be subtle—a sign error in your gravitational potential, an incorrect boundary condition in your PDE solver, or bad pixel values in your CCD reduction.

```{code-cell} ipython3
def process_photometry(fluxes, zero_point=25.0):
    """
    Convert instrumental fluxes to magnitudes.
    
    Parameters
    ----------
    fluxes : array-like
        Instrumental fluxes (ADU)
    zero_point : float
        Photometric zero point
        
    Returns
    -------
    array
        Calibrated magnitudes
    """
    import numpy as np
    
    # This will crash if any flux is negative or zero!
    magnitudes = zero_point - 2.5 * np.log10(fluxes)
    return magnitudes

# Example of debugging workflow
print("""When this crashes in IPython:

>>> fluxes = [1000, 500, -10, 2000]  # Bad data!
>>> mags = process_photometry(fluxes)
ValueError: math domain error

>>> %debug  # Enter debugger

ipdb> p fluxes
[1000, 500, -10, 2000]

ipdb> p fluxes[2]
-10  # Found the problem!

ipdb> import numpy as np
ipdb> np.where(np.array(fluxes) <= 0)
(array([2]),)  # Index of bad value

ipdb> q  # Quit debugger

# Fix: Filter bad data
>>> good_fluxes = [f for f in fluxes if f > 0]
>>> mags = process_photometry(good_fluxes)
""")
```

**Debugging Strategies for Theoretical/Computational Astrophysics:**

1. **Unit Conversion Errors:** The #1 killer of calculations
   - Always write units in comments: `R_cluster = 1 * 3.086e24  # 1 Mpc in cm`
   - Test with known values: Solar radius = 6.96e10 cm, Earth orbit = 1.496e13 cm
   - Use **dimensional analysis**: Check that [Energy] = [Mass][Length]²[Time]⁻²

2. **Math Transcription Errors:** From equations to code
   - Compare implementation character-by-character with paper equations
   - Common mistakes: Missing parentheses, wrong exponents, forgotten 2π factors
   - Split up complex equations into multiple calculations (e.g., `planck_fcn = num/denom`)
   - Test limiting cases: Newtonian limit of GR, non-relativistic limit of SR

3. **Numerical Method Failures:**
   - **Integration instabilities:** Symplectic integrators for long-term stability
   - **Stiff ODEs:** Use implicit methods (`scipy.integrate.solve_ivp` with 'Radau')
   - **Boundary conditions:** Ghost zones for hydro codes
   - **Convergence:** Richardson extrapolation to test resolution dependence

4. **Physical Validity Checks:**
   - **Conservation laws**: Energy, momentum, angular momentum, mass
   - **Causality**: Nothing faster than c
   - **Thermodynamics**: Entropy shouldn't decrease (except with cooling)
   - **Stability**: Jeans mass, Chandrasekhar limit, Eddington luminosity

5. **Scale and Precision Issues:**
   - **Log-space for extreme ratios, order of magnitude ranges**
   - **Catastrophic cancellation**: Rewrite `1 - cos(x)` as `2*sin(x/2)**2` for small `x` (i.e., Taylor series expansion)
   - **Double precision limits**: ~15 decimal digits (problems for age of universe calculations)

**Always start simple**: test your algorithms on simplified/idealized test problems, where the answer is known - e.g., testing your N-body ODE integrator on the $N=2$ Earth-Sun system before jumping to simulating a $N \gg 2$ star cluster.

**Debugging Strategies for Observational Astronomy:**

1. **Instrument-Specific Issues:**
   - Bad pixels, hot pixels, cosmic rays: Use median filtering
   - Flat fielding errors: Check twilight flats vs dome flats
   - Dark current: Temperature-dependent, check CCD temperature logs

2. **Coordinate and Time Systems:**
   - Frame confusion: ICRS vs Galactic vs ecliptic
   - Epoch differences: J2000 vs current epoch for proper motion
   - Time scales: UTC vs TAI vs TDB (critical for pulsar timing)

3. **Data Pipeline Problems:**
   - NaN propagation: Use np.nanmean(), check after each step
   - Memory overflow: Process in chunks for large surveys
   - File I/O: FITS header corruption, endianness issues

Example debugging session:

:::{important} 💡 Computational Thinking: Defensive Astronomy Programming
:class: dropdown

Astronomical data is messy. Defensive programming anticipates common failures:

**Common Issues & Defensive Solutions:**

Example defensive patterns:
```python
def safe_magnitude(flux, zero_point=25.0):
    """Calculate magnitude with error handling."""
    if flux <= 0:
        return np.nan  # Don't crash, return NaN
    
    try:
        mag = zero_point - 2.5 * np.log10(flux)
        if not np.isfinite(mag):
            return np.nan
        return mag
    except (ValueError, TypeError):
        return np.nan
```

This robustness is essential when processing thousands of images automatically!

```python
# THEORETICAL: Sign error in gravitational potential
# WRONG: Missing negative sign
phi = G * M / r  # Should be negative!

# CORRECT: Gravitational potential is negative
phi = -G * M / r # this happens why more than you think!
```

:::

## ⚠️ Practice Exercises

### ⚠️ Exercise 1.1: IPython Mastery

:::{admonition} Part A: Explore Scientific Libraries (5 min)
:class: exercise, dropdown

Execute these commands in IPython to explore `astropy`:

```python
# In IPython:
import astropy
import astropy.units as u
import astropy.constants as const

# Explore available constants
print("Astronomical constants:", 
      [x for x in dir(const) if not x.startswith('_')][:10])

# Quick calculation: Jeans mass
T = 10 * u.K  # Molecular cloud temperature
n = 1e4 * u.cm**-3  # Number density

# Calculate Jeans mass (simplified)
M_J = 2.0 * (const.k_B * T / (const.G * const.m_p))**(3/2) * n**(-1/2)
print(f"Jeans mass: {M_J.to(u.M_sun):.1f}")
```

:::

:::{admonition} ⚠️ Part B: Time Array Operations (10 min)
:class: exercise, dropdown

Compare different methods for calculating stellar distances:

```python
import numpy as np
import timeit

# Method 1: List comprehension
def distance_list(parallaxes_mas):
    """Calculate distances using list comprehension."""
    return [1000.0/p if p > 0 else np.nan 
            for p in parallaxes_mas]

# Method 2: NumPy vectorized
def distance_numpy(parallaxes_mas):
    """Calculate distances using NumPy."""
    par = np.array(parallaxes_mas)
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = 1000.0 / par
        distances[par <= 0] = np.nan
    return distances

# Test data: parallaxes in milliarcseconds
# Include some bad data (negative, zero)
np.random.seed(42)
test_parallaxes = np.random.exponential(2, 1000)
test_parallaxes[::50] = -1  # Some bad measurements

# Time both methods
t1 = timeit.timeit(
    lambda: distance_list(test_parallaxes), 
    number=100
)
t2 = timeit.timeit(
    lambda: distance_numpy(test_parallaxes), 
    number=100
)

print(f"List comprehension: {t1*10:.3f} ms")
print(f"NumPy vectorized:   {t2*10:.3f} ms")
print(f"NumPy is {t1/t2:.1f}x faster!")
```

:::

:::{admonition} ⚠️ Part C: Create Your Own Analysis (15 min)
:class: exercise, dropdown

Design a timing experiment for period-finding algorithms:

```python
import numpy as np
import timeit

# Generate synthetic light curve
np.random.seed(42)
n_points = 1000
times = np.sort(np.random.uniform(0, 100, n_points))
true_period = 2.35  # days
true_amplitude = 0.5  # magnitudes

# Create variable star signal
signal = true_amplitude * np.sin(2 * np.pi * times / true_period)
noise = np.random.normal(0, 0.05, n_points)
magnitudes = 15.0 + signal + noise

# Method 1: Lomb-Scargle periodogram (simplified)
def lomb_scargle_simple(t, y, periods):
    """Simplified Lomb-Scargle (for demonstration)."""
    powers = []
    for period in periods:
        omega = 2 * np.pi / period
        cos_wt = np.cos(omega * t)
        sin_wt = np.sin(omega * t)
        
        # Simplified power calculation
        c = np.sum(y * cos_wt)
        s = np.sum(y * sin_wt)
        power = c**2 + s**2
        powers.append(power)
    return np.array(powers)

# Method 2: String-length method
def string_length(t, y, periods):
    """String-length period finding."""
    lengths = []
    for period in periods:
        # Fold light curve
        phases = (t % period) / period
        # Sort by phase
        idx = np.argsort(phases)
        y_sorted = y[idx]
        
        # Calculate string length
        length = np.sum(np.abs(np.diff(y_sorted)))
        lengths.append(length)
    return np.array(lengths)

# Test periods
test_periods = np.linspace(0.5, 5.0, 100)

# Time both methods
t1 = timeit.timeit(
    lambda: lomb_scargle_simple(times, magnitudes, test_periods),
    number=10
)
t2 = timeit.timeit(
    lambda: string_length(times, magnitudes, test_periods),
    number=10
)

print(f"Lomb-Scargle: {t1*100:.1f} ms")
print(f"String-length: {t2*100:.1f} ms")
print(f"Ratio: {t1/t2:.2f}x")

# Find the period
powers = lomb_scargle_simple(times, magnitudes, test_periods)
best_period = test_periods[np.argmax(powers)]
print(f"\nTrue period: {true_period:.3f} days")
print(f"Found period: {best_period:.3f} days")
```
:::

### ⚠️ Exercise 1.2: Notebook State Detective - Cosmology Edition

:::{admonition} Part A: Trace the Cosmological Calculation (5 min)
:class: exercise, dropdown

Given this notebook execution order, trace the state:

```python
# Execution order: Cell 1, Cell 3, Cell 2, Cell 4, Cell 2, Cell 4

Cell 1: H0 = 70.0  # km/s/Mpc
        omega_m = 0.3
        omega_lambda = 0.7

Cell 2: def age_of_universe():
            # Simplified calculation
            from math import sqrt
            H0_SI = H0 * 1000 / 3.086e22  # Convert to 1/s
            age = (2/3) / H0_SI / sqrt(omega_lambda)
            return age / (365.25 * 24 * 3600 * 1e9)  # Gyr

Cell 3: H0 = 67.4  # Planck value
        omega_lambda = 0.685

Cell 4: print(f"Age: {age_of_universe():.2f} Gyr")

```

**What age gets printed each time? Which cosmology is used?**

:::

:::{admonition} ⚠️ Part B: Find the Bug (10 min)
:class: exercise, dropdown

```python
# Simulate the execution to find the bug
H0 = 70.0  # Cell 1
omega_m = 0.3
omega_lambda = 0.7

H0 = 67.4  # Cell 3
omega_lambda = 0.685

# Cell 2 - function captures H0 and omega_lambda NOW
def age_of_universe():
    from math import sqrt
    H0_SI = H0 * 1000 / 3.086e22
    age = (2/3) / H0_SI / sqrt(omega_lambda)
    return age / (365.25 * 24 * 3600 * 1e9)

print(f"First Cell 4: Age = {age_of_universe():.2f} Gyr")

# Cell 2 again - NOW it uses updated values!
def age_of_universe():
    from math import sqrt
    H0_SI = H0 * 1000 / 3.086e22
    age = (2/3) / H0_SI / sqrt(omega_lambda)
    return age / (365.25 * 24 * 3600 * 1e9)

print(f"Second Cell 4: Age = {age_of_universe():.2f} Gyr")
print("\nThe function definition captures values when defined!")
```
:::

:::{admonition} ⚠️ Part C: Explain the Scientific Impact (15 min)
:class: exercise, dropdown

Write a paragraph explaining how this notebook behavior could affect:

1. **Cosmological parameter estimation** - Wrong H₀ leads to wrong distances
2. **Reproducibility** - Collaborators get different results
3. **Scientific conclusions** - Age estimates could be off by gigayears

**Consider:** What if this was analyzing Planck CMB data or Type Ia supernovae for dark energy constraints?
:::

### ⚠️ Exercise 1.3: Environment Diagnostic

:::{admonition} Part A: Check Your Setup (5 min)
:class: exercise, dropdown

```python
import sys
import importlib

# Check Python and key packages
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("\nAstronomy packages:")

packages = {
    'numpy': 'Numerical computing',
    'scipy': 'Scientific algorithms', 
    'matplotlib': 'Plotting',
    'astropy': 'Core astronomy',
    'astroquery': 'Archive queries',
    'photutils': 'Photometry',
    'specutils': 'Spectroscopy'
}

for pkg, description in packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg:12} {version:10} - {description}")
    except ImportError:
        print(f"  ✗ {pkg:12} MISSING     - {description}")
```
:::

:::{admonition} ⚠️ Part B: Test Data Access (10 min)
:class: exercise, dropdown

```python
from pathlib import Path
import os

# Check standard astronomy data locations
def check_astronomy_data():
    """Check for standard astronomical data directories."""
    
    # Common environment variables
    env_vars = {
        'ASTRO_DATA': 'Local observation data',
        'CALDB': 'Calibration database',
        'PYSYN_CDBS': 'Synphot reference data',
        'WEBBPSF_PATH': 'Webb PSF data',
        'CRDS_PATH': 'Calibration references'
    }
    
    print("Astronomy Data Paths:")
    print("-" * 50)
    
    for var, description in env_vars.items():
        path = os.getenv(var)
        if path:
            path_obj = Path(path)
            exists = "✓" if path_obj.exists() else "✗"
            print(f"{exists} {var:15} = {path}")
            print(f"  └─ {description}")
        else:
            print(f"- {var:15} not set")
            print(f"  └─ {description}")
    
    # Check for common data directories
    print("\nLocal data directories:")
    for dirname in ['data', 'raw', 'reduced', 'catalogs']:
        path = Path(dirname)
        if path.exists():
            n_files = len(list(path.glob('*')))
            print(f"  ✓ ./{dirname}/ ({n_files} items)")
        else:
            print(f"  - ./{dirname}/ not found")

check_astronomy_data()
```
:::

:::{admonition} ⚠️ Part C: Complete Observatory Diagnostic (15 min)
:class: exercise, dropdown

```python
import sys
import subprocess
from pathlib import Path
import platform

def observatory_diagnostic():
    """
    Complete diagnostic for astronomical computing environment.
    
    Checks everything needed for telescope data analysis.
    """
    print("=" * 60)
    print("OBSERVATORY COMPUTING ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    # 1. System info
    print("\n1. SYSTEM INFORMATION:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # 2. Memory check (important for large images)
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n2. MEMORY:")
        print(f"   Total: {mem.total / 1e9:.1f} GB")
        print(f"   Available: {mem.available / 1e9:.1f} GB")
    except ImportError:
        print("\n2. MEMORY: psutil not installed")
    
    # 3. Check for astronomy tools
    print("\n3. ASTRONOMY TOOLS:")
    tools = {
        'ds9': 'SAOImage DS9 viewer',
        'fv': 'FITS viewer',
        'topcat': 'Table analysis',
        'aladin': 'Sky atlas'
    }
    
    for tool, description in tools.items():
        try:
            result = subprocess.run(
                ['which', tool], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"   ✓ {tool:8} - {description}")
            else:
                print(f"   - {tool:8} - {description}")
        except:
            print(f"   ? {tool:8} - {description}")
    
    # 4. Python packages for specific telescopes
    print("\n4. TELESCOPE-SPECIFIC PACKAGES:")
    telescope_packages = {
        'drizzlepac': 'HST data',
        'jwst': 'JWST pipeline',
        'ccdproc': 'CCD reduction',
        'pyraf': 'IRAF tasks'
    }
    
    for pkg, telescope in telescope_packages.items():
        try:
            __import__(pkg)
            print(f"   ✓ {pkg:12} - {telescope}")
        except ImportError:
            print(f"   - {pkg:12} - {telescope}")
    
    print("\n" + "=" * 60)
    return True

# Run the diagnostic
observatory_diagnostic()
```
:::

### ⚠️ Exercise 1.4: Variable Star Exercise Thread

**TODO:** Fix to a simpler problem, this or similar exercise should be in Chapter 6 (i.e., provide partial class implementation for them to complete.)

:::{admonition} Chapter 1: Variable Star Analysis Foundation
:class: exercise, dropdown

```python
# Chapter 1: Variable Star Analysis - Professional Foundation
# This will grow into a complete variable star analysis pipeline

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

class VariableStarObservation:
    """
    Container for variable star observations.
    
    This class will be expanded in each chapter to include:
    - Chapter 2: Numerical precision for period analysis
    - Chapter 3: Time series arrays and phase folding
    - Chapter 4: Period finding algorithms
    - Chapter 5: Statistical analysis and error propagation
    - Chapter 6: Full object-oriented analysis pipeline
    """
    
    def __init__(self, star_name, star_type, period, 
                 mag_mean, mag_amplitude, epoch=None):
        """
        Initialize variable star observation.
        
        Parameters
        ----------
        star_name : str
            Star designation (e.g., 'Delta Cephei')
        star_type : str
            Variable type ('Cepheid', 'RR Lyrae', etc.)
        period : float
            Period in days
        mag_mean : float
            Mean magnitude
        mag_amplitude : float
            Peak-to-peak amplitude
        epoch : float, optional
            Reference epoch (JD)
        """
        self.star_name = star_name
        self.star_type = star_type
        self.period = period
        self.mag_mean = mag_mean
        self.mag_amplitude = mag_amplitude
        self.epoch = epoch or 2451545.0  # J2000.0 default
        
        # Metadata for reproducibility
        self.metadata = {
            'created': datetime.now().isoformat(),
            'python_version': sys.version.split()[0],
            'environment': sys.executable,
            'numpy_version': np.__version__
        }
    
    def __str__(self):
        """String representation."""
        return (f"{self.star_name} ({self.star_type}): "
                f"P={self.period:.4f}d, "
                f"<m>={self.mag_mean:.2f}, "
                f"Δm={self.mag_amplitude:.2f}")
    
    def phase_fold(self, times, magnitudes):
        """
        Phase fold observations (preview of Chapter 3).
        
        Parameters
        ----------
        times : array-like
            Observation times (JD)
        magnitudes : array-like
            Observed magnitudes
            
        Returns
        -------
        phases : array
            Phases (0-1)
        folded_mags : array
            Phase-folded magnitudes
        """
        # Calculate phases
        phases = ((times - self.epoch) % self.period) / self.period
        
        # Sort by phase
        sort_idx = np.argsort(phases)
        return phases[sort_idx], np.array(magnitudes)[sort_idx]
    
    def save(self, filename=None):
        """
        Save observation with full metadata.
        
        Parameters
        ----------
        filename : str, optional
            Output filename (default: star_name_ch1.json)
        """
        if filename is None:
            # Safe filename from star name
            safe_name = self.star_name.replace(' ', '_').replace('*', 'star')
            filename = f"{safe_name}_ch1.json"
        
        data = {
            'star': {
                'name': self.star_name,
                'type': self.star_type,
                'period': self.period,
                'mag_mean': self.mag_mean,
                'mag_amplitude': self.mag_amplitude,
                'epoch': self.epoch
            },
            'metadata': self.metadata
        }
        
        path = Path(filename)
        try:
            path.write_text(json.dumps(data, indent=2))
            print(f"✓ Saved to {path.absolute()}")
            
            # Verify save
            verify = json.loads(path.read_text())
            assert verify['star']['name'] == self.star_name
            print(f"✓ Verified: {self.star_name} data intact")
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"✗ Error saving: {e}")
            return None
        
        return path
    
    @classmethod
    def load(cls, filename):
        """
        Load observation from file.
        
        Parameters
        ----------
        filename : str
            Input filename
            
        Returns
        -------
        VariableStarObservation
            Loaded observation object
        """
        path = Path(filename)
        data = json.loads(path.read_text())
        
        star_data = data['star']
        obs = cls(
            star_name=star_data['name'],
            star_type=star_data['type'],
            period=star_data['period'],
            mag_mean=star_data['mag_mean'],
            mag_amplitude=star_data['mag_amplitude'],
            epoch=star_data.get('epoch', 2451545.0)
        )
        
        # Preserve original metadata
        obs.metadata.update(data.get('metadata', {}))
        return obs


# Create example variable stars
print("Creating variable star catalog...")
print("=" * 50)

# Classical Cepheid
delta_cep = VariableStarObservation(
    star_name="Delta Cephei",
    star_type="Classical Cepheid",
    period=5.366319,
    mag_mean=3.95,
    mag_amplitude=0.88,
    epoch=2451545.0
)
print(delta_cep)
delta_cep.save()

# RR Lyrae star
rr_lyr = VariableStarObservation(
    star_name="RR Lyrae",
    star_type="RR Lyrae",
    period=0.56686776,
    mag_mean=7.92,
    mag_amplitude=1.04,
    epoch=2451545.0
)
print(rr_lyr)
rr_lyr.save()

# Eclipsing binary
algol = VariableStarObservation(
    star_name="Algol",
    star_type="Eclipsing Binary",
    period=2.8673043,
    mag_mean=2.12,
    mag_amplitude=1.27,
    epoch=2451545.0
)
print(algol)
algol.save()

print("\n✓ Chapter 1 complete: Environment configured")
print("✓ Variable star foundation established")
print("→ Next: Chapter 2 will add numerical precision analysis")
```
:::

## Main Takeaways

This chapter has revealed the hidden complexity underlying every astronomical Python analysis you'll perform. You've learned that when your spectral fitting code fails or produces different radial velocities on different systems, it's rarely the algorithm — it's the **environment** surrounding that code. Understanding this distinction transforms you from someone frustrated by `ImportError: No module named 'astropy.modeling'` to someone who systematically diagnoses and fixes environment issues in seconds.

**IPython** is more than an enhanced prompt - it's your astronomical data exploration laboratory. The ability to quickly test period-finding algorithms, explore new spectroscopy libraries, and time different approaches to photometry is fundamental to computational astrophysics. The **magic commands** like `%timeit` for benchmarking and `%debug` for post-mortem analysis aren't conveniences; they're essential tools for developing robust data reduction pipelines. Master IPython now, because you'll use it every day at the telescope and in your office.

The **Jupyter notebook** trap is particularly dangerous in astronomy where we often explore large datasets interactively. While notebooks seem perfect for examining spectra or plotting light curves, their hidden state makes them unsuitable for serious analysis. That beautiful notebook showing exoplanet transit fits might give different planet radii each time it's run due to out-of-order execution. After Project 1, you'll transition to **scripts** that guarantee reproducibility — essential when your results might influence million-dollar telescope time allocations.

Scripts enforce **reproducibility** through predictable execution. The `if __name__ == "__main__"` pattern enables you to build modular analysis tools that work both standalone and as part of larger pipelines — crucial for survey astronomy where individual components must integrate into massive data processing systems. This pattern is why you can `import photometry` from a colleague's module while they can still run it directly to process their data.

Creating reproducible environments is about scientific integrity, not just convenience. When you can't reproduce your own gravitational lens modeling from six months ago because NumPy updated and changed its random number generator, you've lost crucial research continuity. The tools you've learned — **conda** environments with version pinning, `environment.yml` files for exact reproduction, proper path handling for cluster compatibility — are the foundation of trustworthy computational astrophysics. Every major discovery, from exoplanets to gravitational waves, depends on reproducible computational environments.

The debugging strategies you've learned will save you countless hours at the telescope. The universal environment check solves most "mysterious" failures before they waste observing time. Systematic import debugging reveals why `astropy.io.fits` can't be found (usually forgetting to activate your environment). IPython's debugger lets you examine why your centroiding algorithm failed without rerunning the entire night's reduction.

**Remember:** Computational astrophysics isn't just about implementing algorithms from papers. It's about creating reliable, reproducible tools that can process terabytes of telescope data and produce trustworthy scientific results. The practices you've learned — from IPython exploration to environment management — are the foundation that enables discoveries. **Defensive programming** isn't paranoia; it's what keeps pipelines running when processing millions of galaxy spectra.

## Definitions

**conda**: Package and environment management system that creates isolated Python installations with specific package versions, essential for maintaining different analysis environments for different telescopes or surveys.

**defensive programming**: Writing code that anticipates failures (bad pixels, missing FITS headers, cosmic rays) and handles them gracefully rather than crashing, crucial for automated telescope pipelines.

**environment**: An isolated Python installation with its own interpreter, packages, and settings, preventing conflicts between different projects or different versions of astronomy software.

**import system**: Python's mechanism for loading code from external modules, searching through directories listed in sys.path in order until finding the requested package.

**IPython**: Interactive Python — an enhanced interpreter designed for scientific computing, offering features like magic commands, tab completion, and post-mortem debugging essential for astronomical data analysis.

**Jupyter notebook**: Web-based interactive computing platform combining code, results, and text in cells that maintain state between executions, useful for exploration but dangerous for reproducible science.

**magic command**: Special IPython commands prefixed with `%` or `%%` providing functionality beyond standard Python, such as timing code (`%timeit`), debugging (`%debug`), or profiling memory usage.

**module**: A Python file containing functions, classes, and variables that can be imported and used in other programs, enabling code reuse across different analysis scripts.

**REPL**: Read-Eval-Print Loop — an interactive programming environment that immediately evaluates expressions, essential for testing algorithms and exploring telescope data.

**reproducibility**: The ability to obtain identical scientific results using the same data and code, regardless of when or where it's run — fundamental to validating astronomical discoveries.

**script**: Plain text file containing Python code that executes from top to bottom predictably, providing reproducible execution essential for telescope data pipelines.

**sys.path**: Python's list of directories to search when importing modules, determining which version of astropy or other packages gets loaded.

**__name__**: Special Python variable that equals `"__main__"` when a script runs directly or the module name when imported, enabling code to serve both as a standalone tool and importable library.

## Key Takeaways

✓ **IPython is your primary astronomical tool**: Use it for testing algorithms, exploring data, and rapid prototyping — not the basic Python REPL

✓ **Environment problems cause most "broken" analysis code**: When imports fail, check your environment first with `sys.executable` and `conda list`

✓ **Notebooks corrupt scientific analysis**: Hidden state and execution ambiguity make results irreproducible — use them only for initial exploration

✓ **Scripts enforce reproducibility**: Top-to-bottom execution eliminates ambiguity essential for publishable results

✓ **The `__name__` pattern enables pipeline integration**: Code can be both a standalone tool and an importable module

✓ **Conda environments isolate telescope projects**: Each survey or instrument can have its own package versions without conflicts

✓ **Always version-pin packages**: Use `environment.yml` files to ensure collaborators can reproduce your exact analysis

✓ **Paths must be configurable**: Use environment variables and Path objects for code that works on both laptops and clusters

✓ **Control randomness with seeds**: Always set and document random seeds for Monte Carlo simulations

✓ **Systematic debugging saves telescope time**: Environment check → verify imports → test with known data

✓ **Defensive programming handles messy astronomical data**: Assume bad pixels, cosmic rays, and missing headers

## Quick Reference Tables

:::{list-table} Essential IPython Commands for Astronomy
:header-rows: 1
:widths: 20 30 50

* - Command
  - Purpose
  - Astronomy Example
* - `%timeit`
  - Time code execution
  - `%timeit photometry(image)`
* - `%run`
  - Run script keeping variables
  - `%run reduce_spectra.py`
* - `%debug`
  - Debug after error
  - Debug failed source extraction
* - `%who`
  - List variables
  - Check loaded catalogs
* - `%whos`
  - Detailed variable info
  - Inspect array dimensions
* - `%matplotlib`
  - Configure plotting
  - `%matplotlib inline` for notebooks
* - `%load`
  - Load code file
  - `%load photometry_utils.py`
* - `%save`
  - Save session code
  - `%save reduction.py 1-50`
* - `?`
  - Quick help
  - `astropy.io.fits.open?`
* - `??`
  - Show source
  - `photutils.aperture??`
:::

:::{list-table} Astronomy Environment Debugging
:header-rows: 1
:widths: 25 35 40

* - Check
  - Command
  - What to Look For
* - Python location
  - `which python`
  - Should show conda environment
* - Astropy version
  - `python -c "import astropy; print(astropy.__version__)"`
  - Version 5.0+ recommended
* - Environment name
  - `conda info --envs`
  - Asterisk marks active
* - Astronomy packages
  - `conda list | grep astro`
  - astropy, astroquery, etc.
* - FITS support
  - `python -c "from astropy.io import fits"`
  - Should import without error
* - Data paths
  - `echo $ASTRO_DATA`
  - Your data directory
:::

## Python Module & Method Reference

:::{note}
**Building Your Astronomy Python Toolkit**

This reference focuses on modules and methods essential for astronomical data analysis. Each chapter adds new tools to your arsenal. By course end, you'll have a comprehensive reference covering everything from FITS I/O to cosmological calculations.
:::

### Astronomy-Specific Modules

**`astropy` core modules**
```python
import astropy
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coord
import astropy.time as time
import astropy.io.fits as fits
```

Key components:
- `u.Quantity(value, unit)` - Numbers with units
- `const.c`, `const.G`, `const.M_sun` - Physical constants
- `coord.SkyCoord(ra, dec)` - Celestial coordinates
- `time.Time(value, format='jd')` - Astronomical time
- `fits.open(filename)` - FITS file I/O

**`numpy` for astronomical arrays**
```python
import numpy as np
```
Essential for astronomy:
- `np.median()` - Robust against cosmic rays
- `np.nanmean()` - Handle masked pixels
- `np.fft.fft()` - Fourier analysis for periods
- `np.random.seed()` - Reproducible Monte Carlo
- `np.memmap()` - Large FITS files

**`matplotlib` for astronomical plots**
```python
import matplotlib.pyplot as plt
```
Astronomy-specific:
- `plt.imshow(image, origin='lower')` - Display FITS images
- `plt.scatter(phase, mag)` - Phase-folded light curves
- `plt.errorbar(x, y, yerr)` - Measurements with errors
- `plt.subplot(projection=wcs)` - WCS projections

### Standard Library for Astronomy

**`pathlib` for data management**
```python
from pathlib import Path

data_dir = Path('/data/observations')
night = '2024-03-15'
fits_path = data_dir / night / 'raw' / 'science_001.fits'
```

**`json` for observation logs**
```python
import json

obs_log = {
    'target': 'M31',
    'exposure': 300,
    'filter': 'r',
    'airmass': 1.2
}
json.dump(obs_log, open('obs_log.json', 'w'))
```

### IPython Magic for Astronomy

**Astronomy-specific magics**
- `%timeit photometry(image)` - Time reduction steps
- `%debug` - Debug failed extractions
- `%run reduce_night.py` - Run reduction scripts
- `%load_ext autoreload` - Reload modified modules
- `%matplotlib widget` - Interactive plots

### Environment Management

**Conda for astronomy**
```bash
# Create astronomy environment
conda create -n astro python=3.11
conda activate astro

# Essential astronomy stack
conda install -c conda-forge \
    numpy scipy matplotlib \
    astropy astroquery photutils \
    specutils ccdproc

# Telescope-specific
conda install -c astropy ginga  # FITS viewer
pip install webbpsf  # JWST PSFs
```

### Quick Astronomy Patterns

**Safe FITS file reading**
```python
from astropy.io import fits
from pathlib import Path

def read_fits_safely(filename):
    """Read FITS with error handling."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {filename}")
    
    try:
        with fits.open(path) as hdul:
            data = hdul[0].data.copy()
            header = hdul[0].header.copy()
        return data, header
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None
```

**Reproducible Monte Carlo**
```python
def monte_carlo_errors(data, n_iterations=1000, seed=42):
    """Monte Carlo error propagation."""
    np.random.seed(seed)  # Always set seed!
    
    results = []
    for i in range(n_iterations):
        simulated = data + np.random.normal(0, data_err)
        results.append(analyze(simulated))
    
    return np.mean(results), np.std(results)
```

### When to Use What?

- **For coordinates**: Always use `astropy.coordinates`, never roll your own
- **For time**: Use `astropy.time.Time` for precision (handles leap seconds)
- **For units**: Use `astropy.units` to prevent Mars Climate Orbiter disasters
- **For FITS**: Use `astropy.io.fits`, not older pyfits
- **For plotting**: Start with matplotlib, consider plotly for interactive
- **For large arrays**: Use numpy with memmap for files > RAM
- **For Monte Carlo**: Always set random seed for reproducibility

## Next Chapter Preview

Now that you've mastered your computational environment, Chapter 2 will transform Python into a powerful astronomical calculator. You'll discover why `0.1 + 0.2 ≠ 0.3` matters when calculating planetary orbits, learn how floating-point errors compound during numerical integration of stellar evolution, and understand why spacecraft trajectories require quadruple precision arithmetic. You'll implement algorithms for coordinate transformations, time system conversions, and cosmological calculations — all while managing the numerical precision that separates a successful Mars landing from a crater. Get ready to understand why the Patriot missile disaster happened and how to prevent similar catastrophes in your astronomical computations!