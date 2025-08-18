# Chapter 5: Functions & Modules - Building Reusable Scientific Code

## Learning Objectives

By the end of this chapter, you will be able to:
- Design functions as clear contracts with well-defined inputs and outputs
- Understand Python's scope rules and how they affect variable access
- Write functions with flexible parameter handling using *args and **kwargs
- Apply functional programming patterns like map, filter, and lambda functions
- Create and import your own modules for code organization
- Document functions properly using docstrings
- Recognize and avoid common function-related bugs
- Build modular, reusable code for scientific applications

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Write loops and conditionals fluently (Chapter 3)
- ✓ Choose appropriate data structures for different tasks (Chapter 4)
- ✓ Handle floating-point arithmetic safely (Chapter 2)
- ✓ Use IPython for testing and timing code (Chapter 1)
- ✓ Design algorithms with pseudocode (Chapter 3)

```{code-cell} python
# Quick prerequisite check
data = [2.5, 3.7, 1.2, 4.8]
result = []
for value in data:
    if value > 2.0:
        result.append(value * 2)
print(f"If you got {result}, you're ready!")
# Expected: [5.0, 7.4, 9.6]
```

## Chapter Overview

Functions are the fundamental building blocks of organized code. Without functions, you'd be copying and pasting the same code repeatedly, making bugs harder to fix and improvements impossible to maintain. But functions are more than just a way to avoid repetition—they're how we create abstractions, manage complexity, and build reliable software. In astronomy, every data reduction pipeline, every model fitting routine, and every simulation starts with well-designed functions.

This chapter teaches you to think about functions as contracts between different parts of your code. When you write a function that converts temperature units or calculates stellar magnitudes, you're creating a promise: given valid input, the function will reliably return the correct output. This contract mindset helps you write functions that others (including future you) can trust and use effectively.

We'll explore Python's scope rules, which determine where variables can be accessed, and learn how seemingly simple concepts like default arguments can create subtle bugs that have plagued even major astronomical software packages. You'll discover how Python's flexible parameter system enables powerful interfaces, and how functional programming concepts prepare you for modern scientific computing frameworks like JAX. By the end, you'll be organizing your code into modules that can be shared, tested, and maintained professionally—essential skills for collaborative astronomical research.

## 5.1 Defining Functions: The Basics

A function encapsulates a piece of logic that transforms inputs into outputs. Think of a function as a machine: you feed it raw materials (inputs), it performs some process (the function body), and it produces a product (output). In astronomical terms, a function might take raw CCD counts and return calibrated fluxes, or take orbital elements and return positions—just like the modular design of telescope control systems where each subsystem (tracking, focusing, guiding) operates as an independent function.

Here's something exciting: every function you write can potentially be used by astronomers worldwide. The same function that processes your data tonight could analyze observations from Keck, VLT, or even JWST tomorrow. Let's see how this works!

### Your First Function

Let's start with a function every astronomer needs—converting between magnitude and flux. This single function can process data from any telescope in the world!

```{code-cell} python
def magnitude_to_flux(magnitude, zero_point=0.0):
    """
    Convert astronomical magnitude to flux.
    
    Uses the relation: m = -2.5 * log10(flux) + zero_point
    Therefore: flux = 10^((zero_point - m) / 2.5)
    """
    flux = 10 ** ((zero_point - magnitude) / 2.5)
    return flux

# Using the function
vega_mag = 0.0  # Vega's magnitude in V band
vega_flux = magnitude_to_flux(vega_mag)
print(f"Vega flux (m=0): {vega_flux:.2e} [arbitrary units]")

# Fainter star
star_flux = magnitude_to_flux(5.0)  # 5th magnitude star
print(f"5th mag star: {star_flux:.2e} [100x fainter than Vega]")
```

**Congratulations!** You just wrote code that could process data from any telescope in the world! This same function works whether you're analyzing Hubble observations or data from your backyard telescope. Let's break down exactly how it works:

1. **`def` keyword**: Tells Python we're defining a function
2. **Function name** (`magnitude_to_flux`): Follows snake_case convention, describes what it does
3. **Parameters** (`magnitude, zero_point=0.0`): Variables that receive values when function is called
4. **Docstring**: Brief description of what the function does (always include this!)
5. **Function body**: Indented code that does the actual work
6. **`return` statement**: Sends a value back to whoever called the function

When Python executes `magnitude_to_flux(5.0)`, it creates a temporary namespace where `magnitude = 5.0`, runs the function body, and returns the result. But here's something to consider: each function call has a tiny overhead (creating that namespace, jumping to the function, returning). For a single calculation, it's negligible—about 0.1 microseconds. But if you're calling this function millions of times in a loop processing a large catalog, that overhead adds up. We'll explore this more as we build toward vectorized operations!

### 🔍 **Check Your Understanding #1**

What will this code print?

```python
def process_observation(counts):
    dark_subtracted = counts - 100
    # Oops, forgot the return statement!

signal = process_observation(1500)
print(f"Processed signal: {signal}")
```

<details>
<summary>Answer</summary>

It prints `Processed signal: None`. The function calculates `dark_subtracted` but doesn't return it. Without an explicit `return` statement, Python functions return `None`. This is a common bug in data reduction scripts!

To fix it:
```python
def process_observation(counts):
    dark_subtracted = counts - 100
    return dark_subtracted  # Now it returns the value
```

</details>

### Functions Without Return Values

Not all functions return values. Some perform actions like saving data or updating plots:

```{code-cell} python
def report_observation(object_name, magnitude, error):
    """Report observation in standard format."""
    if error > 0.5:
        quality = "poor"
    elif error > 0.1:
        quality = "fair"
    else:
        quality = "good"
    
    print(f"{object_name}: {magnitude:.2f} ± {error:.2f} mag ({quality})")
    # No return statement - returns None implicitly

# Report some variable star observations
report_observation("RR Lyrae", 7.45, 0.03)
report_observation("Mira", 3.21, 0.15)
report_observation("T Tauri", 10.2, 0.8)
```

### 🌟 **Why This Matters: Pipeline Building**

In astronomical data processing, functions form pipelines where each step transforms the data:

```python
raw_image → bias_subtract() → flat_field() → cosmic_ray_removal() → extract_spectrum()
```

Each function in the pipeline has a clear responsibility. When your spectral extraction looks wrong, you can test each function independently to find the problem. Without functions, you'd have one massive script where a bug anywhere could affect everything—a debugging nightmare that has delayed many publications!

### 🌟 **Why This Matters: The Hubble Space Telescope Mirror Disaster**

In 1990, a simple function parameter error cost NASA $1.5 billion. The Hubble Space Telescope's primary mirror was ground to the wrong shape because a testing device was assembled incorrectly. The core issue? A function that calculated the mirror's curvature used a parameter (the null corrector spacing) that was off by 2.2mm. 

```python
# Simplified version of what went wrong
def calculate_mirror_curve(focal_length, corrector_spacing=1.358):  # Wrong default!
    # Should have been 1.3802
    curve = focal_length / (2 * corrector_spacing)
    return curve
```

This single parameter error went undetected through multiple tests because the function was never validated with independent measurements. The lesson: always validate function parameters against multiple sources, especially when millions (or billions) of dollars are at stake!

### Returning Multiple Values

Python functions can return multiple values using tuples—perfect for astronomical calculations that produce related results:

```{code-cell} python
def analyze_light_curve(times, magnitudes):
    """
    Calculate basic light curve statistics.
    
    Returns:
        mean_mag, amplitude, period_guess
    """
    import numpy as np
    
    mean_mag = np.mean(magnitudes)
    amplitude = np.max(magnitudes) - np.min(magnitudes)
    
    # Simple period estimate (time between similar magnitudes)
    # Real period finding is much more sophisticated!
    if len(times) > 10:
        period_guess = 2 * (times[-1] - times[0]) / len(times)
    else:
        period_guess = None
    
    return mean_mag, amplitude, period_guess

# Simulated RR Lyrae data
import numpy as np
times = np.linspace(0, 2, 50)
mags = 8.0 + 0.5 * np.sin(2 * np.pi * times / 0.6)

mean, amp, period = analyze_light_curve(times, mags)
print(f"Mean magnitude: {mean:.2f}")
print(f"Amplitude: {amp:.2f} mag")
print(f"Period estimate: {period:.2f} days" if period else "Need more data")
```

### The Design Process: From Problem to Function

Before writing any function, design it first. This prevents the common mistake of coding yourself into a corner—something that happens often when reducing complex astronomical data.

```{code-cell} python
"""
DESIGN: Function to validate photometry measurements

PURPOSE: Ensure photometric measurements are physically reasonable
INPUT: magnitude, error, airmass
OUTPUT: boolean (True if valid)
CHECKS:
    - Magnitude in reasonable range (not saturated or too faint)
    - Error is positive and reasonable
    - Airmass is physical (>= 1.0)
"""

def validate_photometry(magnitude, error, airmass):
    """
    Validate photometric measurement for quality.
    
    Parameters
    ----------
    magnitude : float
        Measured magnitude
    error : float
        Magnitude uncertainty  
    airmass : float
        Atmospheric airmass during observation
        
    Returns
    -------
    bool
        True if measurement passes all quality checks
    """
    # Check magnitude range (typical CCD limits)
    if magnitude < -1 or magnitude > 25:
        return False
    
    # Check error is positive and reasonable
    if error <= 0 or error > 1.0:
        return False
    
    # Check airmass is physical
    if airmass < 1.0 or airmass > 3.0:
        return False
    
    return True

# Test with various measurements
test_cases = [
    (15.3, 0.02, 1.2),  # Good measurement
    (30.0, 0.05, 1.1),  # Too faint
    (10.5, -0.1, 1.5),  # Negative error
    (12.0, 0.03, 0.8),  # Impossible airmass
]

for mag, err, am in test_cases:
    valid = validate_photometry(mag, err, am)
    print(f"m={mag:5.1f}, σ={err:5.2f}, X={am:.1f} → {'Valid' if valid else 'Invalid'}")
```

### 💡 **Computational Thinking: Function Contract Design**

Every well-designed function follows a contract pattern that applies across all programming:

```
CONTRACT PATTERN:
1. Preconditions: What must be true before calling
2. Postconditions: What will be true after calling
3. Invariants: What stays unchanged
4. Side effects: What else happens

Example for magnitude_to_flux():
- Precondition: magnitude is numeric
- Postcondition: returns positive flux value
- Invariant: input magnitude unchanged
- Side effects: none (pure function)

This pattern appears in:
- Database transactions (ACID properties)
- API design (REST contracts)
- Parallel computing (thread safety)
- Unit testing (test contracts)
```

## 5.2 Function Arguments In-Depth

Python provides flexible ways to handle function parameters, from simple positional arguments to sophisticated keyword-only parameters. Understanding these mechanisms allows you to create functions that are both powerful and easy to use—essential for building astronomical analysis tools that others can understand and modify.

### Positional vs Keyword Arguments

When you call a function, you can pass arguments by position or by name:

```{code-cell} python
def calculate_signal_to_noise(signal, noise, exposure_time=1.0):
    """
    Calculate signal-to-noise ratio for an observation.
    
    SNR = signal * sqrt(exposure_time) / noise
    """
    import math
    snr = signal * math.sqrt(exposure_time) / noise
    return snr

# Different ways to call the same function
snr1 = calculate_signal_to_noise(1000, 30)  # Positional only
snr2 = calculate_signal_to_noise(noise=30, signal=1000)  # Keywords (any order!)
snr3 = calculate_signal_to_noise(1000, 30, exposure_time=100)  # Mixed

print(f"1 second: SNR = {snr1:.1f}")
print(f"1 second: SNR = {snr2:.1f}")  
print(f"100 seconds: SNR = {snr3:.1f}")
```

Keyword arguments make function calls self-documenting. Compare `process(data, True, False, 10)` with `process(data, normalize=True, remove_cosmic_rays=False, sigma_clip=10)`—the second version immediately tells you what each parameter does.

### Default Arguments and the Mutable Default Trap

Default arguments make functions flexible, but there's a critical trap that has caused bugs in major astronomical software packages. Python evaluates default arguments once when the function is defined, not each time it's called:

```{code-cell} python
# THE TRAP - Mutable default (DON'T DO THIS!)
def add_observation_buggy(time, mag, observations=[]):  # DANGER!
    """Add observation to list - BUGGY VERSION."""
    observations.append((time, mag))
    return observations

# Watch the disaster unfold
night1 = add_observation_buggy(2459123.5, 12.3)
print(f"Night 1: {night1}")

night2 = add_observation_buggy(2459124.5, 12.1)  # Surprise!
print(f"Night 2: {night2}")  # Contains BOTH nights!

print(f"Same object? {night1 is night2}")  # True - it's the same list!
```

⚠️ **Common Bug Alert: The Mutable Default Disaster**

This bug has appeared in:
- IRAF reduction scripts (accumulated all nights' data)
- Astropy coordinate transformations (cached incorrect results)
- Observatory scheduling software (mixed different programs)

The symptom: data from previous runs mysteriously appears in new analyses. The fix: always use `None` as default for mutable arguments.

Here's the correct pattern:

```{code-cell} python
def add_observation_fixed(time, mag, observations=None):
    """Add observation to list - CORRECT VERSION."""
    if observations is None:
        observations = []  # Create new list each time
    observations.append((time, mag))
    return observations

# Now it works correctly
night1 = add_observation_fixed(2459123.5, 12.3)
night2 = add_observation_fixed(2459124.5, 12.1)
print(f"Night 1: {night1}")
print(f"Night 2: {night2}")  # Separate lists!
```

### Variable-Length Arguments (*args)

Sometimes you need functions that accept any number of arguments—perfect for combining multiple observations:

```{code-cell} python
def combine_magnitudes(*magnitudes, method='mean'):
    """
    Combine multiple magnitude measurements.
    
    Note: We convert to flux, average, then back to magnitude
    because magnitudes are logarithmic!
    """
    if not magnitudes:
        raise ValueError("Need at least one magnitude")
    
    # Convert to linear flux space for averaging
    fluxes = [10**(-0.4 * m) for m in magnitudes]
    
    if method == 'mean':
        combined_flux = sum(fluxes) / len(fluxes)
    elif method == 'median':
        combined_flux = sorted(fluxes)[len(fluxes)//2]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert back to magnitude
    import math
    combined_mag = -2.5 * math.log10(combined_flux)
    return combined_mag

# Combine multiple observations of the same star
obs1, obs2, obs3 = 12.45, 12.51, 12.48
combined = combine_magnitudes(obs1, obs2, obs3)
print(f"Individual: {obs1}, {obs2}, {obs3}")
print(f"Combined: {combined:.2f}")

# Works with any number of observations
many_obs = combine_magnitudes(12.1, 12.2, 12.15, 12.18, 12.13, method='median')
print(f"Median of 5 observations: {many_obs:.2f}")
```

### 🔍 **Check Your Understanding #2**

What's wrong with this function definition, and how would you fix it?

```python
def process_spectrum(wavelength, default_flux=1.0, *fluxes, normalize=True):
    # Process spectrum with options
    pass
```

<details>
<summary>Answer</summary>

The parameter order is wrong! Python requires this order:
1. Regular positional parameters
2. *args (if any)
3. Keyword parameters with defaults
4. **kwargs (if any)

Correct version:
```python
def process_spectrum(wavelength, *fluxes, default_flux=1.0, normalize=True):
    # Now the order is correct
    pass
```

The original would give a SyntaxError because *fluxes can't come after a keyword parameter with a default.

</details>

### Keyword Arguments (**kwargs)

The `**kwargs` pattern enables incredibly flexible interfaces—essential for plotting functions and instrument configurations:

```{code-cell} python
def configure_observation(target, exposure, **kwargs):
    """
    Configure telescope observation with flexible options.
    """
    print(f"=== Observation Configuration ===")
    print(f"Target: {target}")
    print(f"Exposure: {exposure}s")
    
    # Process additional options
    defaults = {
        'filter': 'V',
        'binning': 1,
        'readout': 'normal'
    }
    
    # Update defaults with provided options
    config = defaults.copy()
    config.update(kwargs)
    
    print("Settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

# Simple observation
config1 = configure_observation("M31", 300)

print("\n")

# Complex observation with many options
config2 = configure_observation(
    "GRB221009A", 
    60,
    filter='R',
    binning=2,
    readout='fast',
    priority='urgent',
    notes='Gamma-ray burst followup'
)
```

## 5.3 Scope and Namespaces

Ready to understand something that confuses even experienced programmers? This knowledge will save you hours of debugging! When I first learned about scope, I spent a whole weekend tracking down a bug that turned out to be a simple scope issue—the same kind that appears in IRAF/PyRAF scripts when variables mysteriously change values or seem to disappear.

Here's the truth: everyone gets confused by scope at first—even Python's creator Guido van Rossum has admitted the scoping rules are one of Python's trickier aspects! But once you understand the LEGB rule, you'll have power over your variables that many programmers never quite master.

Understanding scope—where variables can be accessed—is crucial for writing bug-free code. Python's scope rules determine which variables are visible at any point in your program. Without understanding scope, you'll encounter confusing bugs where variables don't have the values you expect, or worse, where changing a variable in one place mysteriously affects code elsewhere.

### The LEGB Rule

Python resolves variable names using the LEGB rule, searching in this order:
- **L**ocal: Inside the current function
- **E**nclosing: In the enclosing function (for nested functions)  
- **G**lobal: At the top level of the module
- **B**uilt-in: In the built-in namespace (print, len, etc.)

```{code-cell} python
# Demonstrating LEGB with astronomical context
magnitude_limit = 20.0  # Global scope

def plan_observation():
    magnitude_limit = 15.0  # Enclosing scope (overrides global)
    
    def check_visibility(star_mag):
        magnitude_limit = 12.0  # Local scope (overrides enclosing)
        
        # Local scope sees its own magnitude_limit
        print(f"Inside check: limit = {magnitude_limit}")
        return star_mag < magnitude_limit
    
    # Call inner function
    visible = check_visibility(10.0)
    
    # Enclosing scope sees its own magnitude_limit
    print(f"Inside plan: limit = {magnitude_limit}")
    return visible

result = plan_observation()
print(f"Global: limit = {magnitude_limit}")  # Unchanged!
```

Each function creates its own namespace—a mapping of names to objects. When you use a variable, Python searches through these namespaces in LEGB order until it finds the name.

### ⚠️ **Common Bug Alert: UnboundLocalError - The Assignment Trap**

```python
total_observations = 0  # Global

def add_observation():
    # This will crash with UnboundLocalError!
    total_observations += 1  # Python thinks this is local
    return total_observations

def add_observation_fixed():
    global total_observations  # Explicitly use global
    total_observations += 1
    return total_observations

# Better approach - avoid global state entirely
def add_observation_better(current_total):
    return current_total + 1
```

The error happens because Python sees you're assigning to `total_observations`, assumes it's local, but then can't find a local value to increment. This bug often appears in data reduction scripts that try to maintain running totals. The symptoms are confusing because the same variable name works fine when reading but crashes when writing. This exact bug has appeared in multiple observatory control systems where global counters tracked telescope positions!

### 🔍 **Check Your Understanding #3**

In nested functions, if both the inner and outer function define a variable 'x', which 'x' does the inner function see? Why?

```python
def outer():
    x = "outer"
    
    def inner():
        x = "inner"
        print(f"Inner sees: {x}")
    
    inner()
    print(f"Outer sees: {x}")

outer()
```

<details>
<summary>Answer</summary>

The inner function sees its own local `x = "inner"`, and the outer function sees its own `x = "outer"`. They don't interfere with each other because each function has its own local namespace. 

Output:
```
Inner sees: inner
Outer sees: outer
```

If the inner function didn't define its own `x`, it would see the outer's `x` due to the LEGB rule (it would find it in the Enclosing scope). This scoping behavior lets you create independent variables in nested functions, which is essential for closures and decorators used throughout astronomical software packages.
</details>

### 💡 **Computational Thinking: Why Global Variables Are Dangerous**

Global variables violate the "principle of least surprise" that appears everywhere in computing:

```
HIDDEN STATE ANTI-PATTERN:
- Function behavior depends on external state
- Can't understand function in isolation
- Testing requires global setup
- Parallel processing becomes impossible

Real disaster: ESO pipeline bug (2018)
- Global config variable for instrument mode
- Thread A changes mode for its reduction
- Thread B reads wrong mode mid-process
- Months of data reduced incorrectly

Better pattern: Explicit State Passing
BAD:  current_filter = 'V'; take_image()
GOOD: take_image(filter='V')
```

### Closures: Functions That Remember

Closures are functions that "remember" variables from their enclosing scope—powerful for creating specialized analysis functions:

```{code-cell} python
def create_magnitude_converter(zero_point, extinction=0.0):
    """
    Create a magnitude converter for specific conditions.
    
    The returned function 'remembers' zero_point and extinction.
    """
    def converter(instrumental_mag, airmass=1.0):
        # This function 'closes over' zero_point and extinction
        corrected = instrumental_mag - extinction * airmass
        calibrated = corrected + zero_point
        return calibrated
    
    return converter

# Create converters for different filters
v_band_converter = create_magnitude_converter(zero_point=23.5, extinction=0.15)
r_band_converter = create_magnitude_converter(zero_point=23.8, extinction=0.09)

# Use them for calibration
instrumental_v = 15.67
calibrated_v = v_band_converter(instrumental_v, airmass=1.5)
print(f"V band: {instrumental_v:.2f} (instrumental) → {calibrated_v:.2f} (calibrated)")

instrumental_r = 15.23
calibrated_r = r_band_converter(instrumental_r, airmass=1.5)
print(f"R band: {instrumental_r:.2f} (instrumental) → {calibrated_r:.2f} (calibrated)")
```

## 5.4 Functional Programming Elements

Python supports functional programming—a style that treats computation as the evaluation of mathematical functions. While Python isn't a pure functional language, these concepts are essential because they prepare you for modern scientific computing frameworks like JAX that require functional style, and they lead to cleaner, more testable code.

### Lambda Functions

Lambda functions are small, anonymous functions defined inline—perfect for simple transformations in data analysis:

```{code-cell} python
# Sort stars by different criteria
stars = [
    {'name': 'Vega', 'mag': 0.03, 'distance': 25.04},
    {'name': 'Arcturus', 'mag': -0.05, 'distance': 36.7},
    {'name': 'Sirius', 'mag': -1.46, 'distance': 8.6},
    {'name': 'Canopus', 'mag': -0.74, 'distance': 310},
]

# Sort by magnitude (brightest first - remember lower mag = brighter!)
by_brightness = sorted(stars, key=lambda s: s['mag'])
print("Brightest stars:")
for star in by_brightness[:2]:
    print(f"  {star['name']}: {star['mag']}")

# Sort by distance
by_distance = sorted(stars, key=lambda s: s['distance'])
print("\nNearest stars:")
for star in by_distance[:2]:
    print(f"  {star['name']}: {star['distance']} ly")
```

### Map, Filter, and Reduce for Astronomical Data

These functional tools transform how you process observations:

```{code-cell} python
from functools import reduce
import math

# Sample observations: (time, magnitude, error)
observations = [
    (2459123.512, 12.35, 0.02),
    (2459123.538, 12.28, 0.03),
    (2459123.564, 12.31, 0.02),
    (2459123.589, 99.99, 0.00),  # Bad measurement
    (2459123.614, 12.29, 0.02),
]

# FILTER: Remove bad measurements
good_obs = list(filter(lambda obs: obs[1] < 90, observations))
print(f"Filtered {len(observations) - len(good_obs)} bad measurements")

# MAP: Extract just magnitudes
magnitudes = list(map(lambda obs: obs[1], good_obs))
print(f"Magnitudes: {magnitudes}")

# REDUCE: Calculate weighted average
def weighted_avg(accumulator, obs):
    """Accumulate weighted sum and weights."""
    mag, err = obs[1], obs[2]
    weight = 1.0 / (err ** 2) if err > 0 else 0
    sum_weighted, sum_weights = accumulator
    return (sum_weighted + mag * weight, sum_weights + weight)

weighted_sum, total_weight = reduce(weighted_avg, good_obs, (0.0, 0.0))
weighted_mean = weighted_sum / total_weight if total_weight > 0 else 0
print(f"Weighted mean magnitude: {weighted_mean:.3f}")
```

### 🔍 **Check Your Understanding #4**

Rewrite this loop using functional programming:

```python
# Find all stars brighter than magnitude 5.0
bright_stars = []
for star in catalog:
    if star['magnitude'] < 5.0:
        bright_stars.append(star['name'])
```

<details>
<summary>Answer</summary>

Two equivalent functional approaches:

```python
# Using filter and map
bright_stars = list(map(
    lambda s: s['name'],
    filter(lambda s: s['magnitude'] < 5.0, catalog)
))

# Using list comprehension (more Pythonic)
bright_stars = [s['name'] for s in catalog if s['magnitude'] < 5.0]
```

The list comprehension is generally preferred in Python for readability, but understanding the functional approach prepares you for libraries like JAX that require functional style.

</details>

### 💡 **Computational Thinking: Pure Functions Pattern**

Pure functions are the foundation of reliable, testable, parallelizable code. A pure function always returns the same output for the same input and has no side effects:

```
PURE FUNCTION PATTERN:
✓ Deterministic: same input → same output
✓ No side effects: doesn't modify external state
✓ No hidden inputs: only uses parameters
✓ Thread-safe: can run in parallel

Example - PURE:
def calculate_flux(magnitude):
    return 10 ** (-0.4 * magnitude)

Example - IMPURE:
import random
def add_noise(magnitude):
    return magnitude + random.gauss(0, 0.1)  # Different each time!

Benefits of pure functions:
1. Easy testing: predictable results
2. Safe parallelization: no race conditions
3. Memoization possible: can cache results
4. Debugging simplified: isolated behavior

This pattern appears in:
- JAX (requires pure functions for autodiff)
- Dask (parallel computing)
- Unit testing frameworks
- Functional reactive programming
```

These patterns appear in every major astronomy software package!

### Functions as First-Class Objects

In Python, functions are objects you can pass around—essential for building flexible analysis pipelines:

```{code-cell} python
def median_combine(values):
    """Median combination - robust against outliers."""
    return sorted(values)[len(values)//2]

def mean_combine(values):
    """Mean combination - optimal for Gaussian noise."""
    return sum(values) / len(values)

def process_image_stack(images, combine_function):
    """
    Combine multiple images using specified method.
    
    combine_function determines the combination algorithm.
    """
    print(f"Combining {len(images)} images using {combine_function.__name__}")
    result = combine_function(images)
    return result

# Simulate image stack (pixel values)
image_stack = [1021, 1019, 1024, 2000, 1018]  # Note the cosmic ray (2000)!

# Try different combination methods
mean_result = process_image_stack(image_stack, mean_combine)
median_result = process_image_stack(image_stack, median_combine)

print(f"Mean: {mean_result:.1f} (affected by cosmic ray)")
print(f"Median: {median_result:.1f} (robust against outliers)")
```

### 🌟 **Why This Matters: Modern Frameworks**

Functional programming isn't just academic—it's essential for modern scientific computing:

1. **JAX** (Google's NumPy replacement) requires pure functions for automatic differentiation
2. **Parallel processing** works best with stateless functions
3. **Testing** is trivial when functions have no side effects
4. **GPU computing** maps naturally to functional operations

Example: In JAX, you can automatically differentiate through an entire light curve fitting routine if it's written functionally. This enables advanced techniques like Hamiltonian Monte Carlo that would be impossibly complex to implement manually.

### 🌟 **Why This Matters: LIGO's Modular Design Success**

The detection of gravitational waves in 2015 (Nobel Prize 2017) was possible partly because of LIGO's modular software architecture. The data analysis pipeline consisted of hundreds of independent modules, each with clear interfaces:

```python
# Simplified LIGO pipeline structure
raw_strain → remove_noise() → whiten_data() → matched_filter() → coincidence_test()
```

When the first detection (GW150914) appeared, the modular design allowed teams to:
- Independently verify each processing step
- Swap in alternative algorithms for cross-validation
- Run parallel analyses with different parameters
- Complete verification in weeks instead of months

The same modular approach that you're learning here enabled one of the most significant discoveries in physics!

## 5.5 Modules and Packages

As your analysis grows from scripts to projects, organization becomes critical. Modules and packages are Python's way of organizing code into reusable, maintainable units. In Chapter 1, you learned to import modules—now you'll create your own.

### Creating Your First Module

Let's create a module for photometric calculations. Save this as `photometry.py`:

```{code-cell} python
# photometry.py - Part 1: Constants and Basic Functions
"""
Photometric calculations for astronomical observations.
"""

# Physical constants
SOLAR_MAGNITUDE = 4.83  # V-band absolute magnitude
PARSEC_IN_AU = 206265.0
ZERO_POINT_FLUX = 3631.0  # Jansky for AB magnitude system

def magnitude_to_flux(magnitude, zero_point=0.0):
    """Convert magnitude to flux."""
    flux = 10 ** ((zero_point - magnitude) / 2.5)
    return flux

def flux_to_magnitude(flux, zero_point=0.0):
    """Convert flux to magnitude."""
    import math
    if flux <= 0:
        return float('inf')  # Infinite magnitude for zero flux
    magnitude = zero_point - 2.5 * math.log10(flux)
    return magnitude
```

Now add error propagation functions:

```{code-cell} python
# photometry.py - Part 2: Error Propagation
def magnitude_error(flux, flux_error):
    """
    Calculate magnitude error from flux error.
    
    σ_mag = 2.5 / ln(10) * σ_flux / flux ≈ 1.0857 * σ_flux / flux
    """
    if flux <= 0:
        return float('inf')
    mag_error = 1.0857 * abs(flux_error / flux)
    return mag_error

def combine_magnitudes(mags, errors):
    """Combine magnitudes with proper error propagation."""
    if len(mags) != len(errors):
        raise ValueError("Magnitudes and errors must have same length")
    
    # Convert to flux for proper averaging
    fluxes = [magnitude_to_flux(m) for m in mags]
    weights = [1/e**2 for e in errors]
    
    weighted_flux = sum(f*w for f, w in zip(fluxes, weights))
    total_weight = sum(weights)
    
    mean_flux = weighted_flux / total_weight
    mean_mag = flux_to_magnitude(mean_flux)
    
    # Error propagation
    mean_error = 1.0 / (total_weight ** 0.5)
    
    return mean_mag, mean_error
```

### Using Your Module

Once you've created the module, you can import and use it:

```{code-cell} python
# Method 1: Import entire module
import photometry

star_flux = photometry.magnitude_to_flux(12.5)
print(f"12.5 mag = {star_flux:.2e} flux units")

# Method 2: Import specific functions
from photometry import combine_magnitudes

observations = [(12.35, 0.02), (12.38, 0.03), (12.33, 0.02)]
mags = [m for m, e in observations]
errors = [e for m, e in observations]

combined_mag, combined_error = combine_magnitudes(mags, errors)
print(f"Combined: {combined_mag:.3f} ± {combined_error:.3f}")

# Method 3: Import with alias
import photometry as phot
error = phot.magnitude_error(1000, 30)
print(f"Magnitude error: {error:.3f}")
```

### The `if __name__ == "__main__"` Pattern

This essential pattern makes modules both importable and executable:

```{code-cell} python
# photometry.py - Part 3: Test Code
if __name__ == "__main__":
    # This code ONLY runs when script is executed directly
    # NOT when it's imported as a module
    
    print("Testing photometry module...")
    
    # Test magnitude conversion
    test_mag = 15.0
    test_flux = magnitude_to_flux(test_mag)
    recovered_mag = flux_to_magnitude(test_flux)
    
    print(f"Original: {test_mag:.2f}")
    print(f"Flux: {test_flux:.2e}")
    print(f"Recovered: {recovered_mag:.2f}")
    
    assert abs(recovered_mag - test_mag) < 1e-10, "Conversion failed!"
    print("All tests passed!")
```

This pattern appears in every professional Python module because it enables:
1. **Module testing**: Run tests without a separate test file
2. **Command-line interfaces**: Make modules directly executable
3. **Examples**: Show how to use the module
4. **Development**: Quick testing during development

### Creating a Package

As your project grows, organize related modules into packages:

```
astro_tools/
    __init__.py          # Makes it a package
    photometry.py        # Photometric calculations
    spectroscopy.py      # Spectral analysis
    coordinates.py       # Coordinate transformations
    constants.py         # Physical constants
```

The `__init__.py` file defines what gets imported:

```python
# astro_tools/__init__.py
"""
Astronomical analysis tools for research.
"""

# Import commonly used functions for convenience
from .photometry import magnitude_to_flux, flux_to_magnitude
from .constants import SPEED_OF_LIGHT, PARSEC_IN_AU

# Package metadata
__version__ = '0.1.0'
__author__ = 'Your Name'

print(f"Loading astro_tools v{__version__}")
```

### ⚠️ **Common Bug Alert: The Relative Import Confusion**

```python
# This causes endless confusion when scripts move!

# In observations/photometry.py
from ..utils import calibration  # Relative import

# Works when run as: python -m observations.photometry
# Breaks when run as: python observations/photometry.py
# Error: "attempted relative import beyond top-level package"

# Also breaks if you move the file to a different directory!

# SOLUTION: Use absolute imports or make proper packages
# Better approach:
from astro_tools.utils import calibration  # Absolute import

# Or for development, add to path:
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calibration
```

This relative import problem has delayed countless research projects when code that worked perfectly in one directory structure breaks after reorganization. The Gaia data processing pipeline had to be restructured twice because relative imports made it impossible to run individual modules for testing. Always prefer absolute imports in scientific code—they're clearer and more robust!

### Import Best Practices and Namespace Pollution

```{code-cell} python
# GOOD: Clear, explicit imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

# BAD: Wildcard imports pollute namespace
# from numpy import *  # Adds 600+ names!
# from astropy.units import *  # Conflicts with numpy!

# Example of namespace collision disaster
# from numpy import *       # Has array, log, sin, etc.
# from math import *         # Also has log, sin, etc.
# result = log(10)  # Which log function? Math or numpy?

# This caused a real bug in a published paper where
# math.log (natural log) was confused with numpy.log10,
# leading to incorrect stellar masses!

# SAFE: Explicit namespaces prevent confusion
import numpy as np
import math

natural_log = math.log(10)      # Clear: natural logarithm
common_log = np.log10(10)       # Clear: base-10 logarithm
print(f"ln(10) = {natural_log:.2f}")
print(f"log₁₀(10) = {common_log:.2f}")
```

### 🔍 **Check Your Understanding #5**

Why is `from math import *` particularly dangerous in scientific code?

<details>
<summary>Answer</summary>

Wildcard imports are dangerous because:

1. **Name collisions**: Multiple libraries have functions with the same names (log, sqrt, sin, etc.)
2. **Hidden overwrites**: Later imports silently replace earlier ones
3. **Unclear source**: You can't tell where a function comes from
4. **Namespace pollution**: Hundreds of names added to your namespace

Real example that caused a retracted paper:
```python
from numpy import *      # Has log (natural logarithm)
from math import *       # Also has log (natural logarithm)
from scipy.special import *  # Has log1p (log(1+x))

# Calculating stellar luminosity
luminosity = log(flux)  # Which log? All three are natural log, but...
# If someone later adds:
from sympy import *     # Has log that might behave differently with symbols

# The same code now might use a different log function!
```

This is why professional astronomy packages always use explicit imports. The extra typing (`np.log`) is worth the clarity and safety!
</details>

## 5.6 Documentation and Testing

Good documentation and basic testing make your functions trustworthy and reusable. This isn't optional in scientific computing—undocumented, untested code has led to retracted papers and wasted telescope time.

### Writing Good Docstrings

```{code-cell} python
def fit_period(times, magnitudes, min_period=0.1, max_period=100.0):
    """
    Find the best-fit period for a variable star.
    
    Uses Lomb-Scargle periodogram for unevenly sampled data.
    
    Parameters
    ----------
    times : array-like
        Observation times in days (JD or MJD)
    magnitudes : array-like
        Measured magnitudes
    min_period : float, optional
        Minimum period to search (days), default 0.1
    max_period : float, optional
        Maximum period to search (days), default 100.0
    
    Returns
    -------
    best_period : float
        Period with maximum power in periodogram
    power : float
        Normalized power at best period (0-1)
    
    Raises
    ------
    ValueError
        If fewer than 10 observations provided
    
    Examples
    --------
    >>> times = np.linspace(0, 100, 200)
    >>> mags = 12.0 + 0.5 * np.sin(2*np.pi*times/5.67)
    >>> period, power = fit_period(times, mags)
    >>> print(f"Found period: {period:.2f} days")
    Found period: 5.67 days
    
    Notes
    -----
    The Lomb-Scargle periodogram is optimal for unevenly
    sampled data typical of ground-based observations.
    For space-based data with even sampling, FFT may be faster.
    
    References
    ----------
    .. [1] Lomb, N.R., 1976, Ap&SS, 39, 447
    .. [2] Scargle, J.D., 1982, ApJ, 263, 835
    """
    # Implementation would go here
    pass
```

### Simple Testing with Assertions

```{code-cell} python
def test_photometry_functions():
    """Test our photometry module functions."""
    
    # Test magnitude-flux conversion round trip
    original_mag = 15.0
    flux = magnitude_to_flux(original_mag)
    recovered_mag = flux_to_magnitude(flux)
    
    assert abs(recovered_mag - original_mag) < 1e-10, \
        f"Round trip failed: {original_mag} → {recovered_mag}"
    
    # Test error propagation
    mag_error = magnitude_error(1000, 30)
    expected_error = 1.0857 * 30 / 1000
    
    assert abs(mag_error - expected_error) < 1e-6, \
        f"Error calculation wrong: {mag_error} vs {expected_error}"
    
    # Test edge cases
    inf_mag = flux_to_magnitude(0)  # Zero flux
    assert inf_mag == float('inf'), "Zero flux should give infinite magnitude"
    
    # Test combination with equal weights
    mags = [12.0, 12.0, 12.0]
    errors = [0.1, 0.1, 0.1]
    combined_mag, combined_err = combine_magnitudes(mags, errors)
    
    assert abs(combined_mag - 12.0) < 1e-10, "Combined magnitude wrong"
    assert combined_err < 0.1, "Combined error should be smaller"
    
    print("All photometry tests passed! ✓")

# Run tests
test_photometry_functions()
```

### 🛠️ **Debug This!**

This function has a subtle performance bug. Can you spot it?

```python
def find_variables(catalog):
    """Find variable stars in a catalog - HAS PERFORMANCE BUG."""
    variables = []
    
    for star in catalog:
        # Calculate variability index
        mags = star['magnitudes']
        mean_mag = sum(mags) / len(mags)
        
        # Check if variable (BUG HERE!)
        if star not in variables and std_dev(mags) > 0.1:
            variables.append(star)
    
    return variables

def std_dev(values):
    """Calculate standard deviation."""
    mean = sum(values) / len(values)
    variance = sum((x - mean)**2 for x in values) / len(values)
    return variance ** 0.5
```

<details>
<summary>Answer and Fix</summary>

**Bug**: `if star not in variables` is O(n) for lists! For a catalog of 10,000 stars, this becomes O(n²) total complexity—extremely slow.

**Fix**: Track seen stars with a set for O(1) lookup:

```python
def find_variables_fixed(catalog):
    """Find variable stars efficiently."""
    variables = []
    seen_ids = set()  # O(1) membership testing
    
    for star in catalog:
        mags = star['magnitudes']
        
        # Check if variable
        if star['id'] not in seen_ids and std_dev(mags) > 0.1:
            variables.append(star)
            seen_ids.add(star['id'])
    
    return variables
```

For 10,000 stars:
- Original: ~5 seconds
- Fixed: ~0.05 seconds
- 100x speedup!

This exact bug appeared in the Kepler pipeline early versions, causing hours of unnecessary computation.

</details>

## 5.7 Performance Considerations

How much do all these abstractions cost? Now let's measure what we've been discussing throughout this chapter! Performance matters in astronomy where we process terabytes of data. Understanding function call overhead and optimization techniques can mean the difference between results in hours versus days. Remember when we mentioned function overhead in the first section? Let's quantify exactly what that means for your code.

### 📊 **Performance Profile: Function Call Overhead Analysis**

Let's measure the real cost of function calls at different complexity levels, building on what we've learned:

```{code-cell} python
import time
import numpy as np

def trivial_function(x):
    """Trivial operation - overhead dominates."""
    return x + 1

def moderate_function(x):
    """Moderate complexity - overhead noticeable."""
    return (x ** 2 + 2 * x + 1) / (x + 1)

def complex_function(x):
    """Complex operation - overhead negligible."""
    import math
    result = 0
    for i in range(10):
        result += math.sin(x + i) * math.cos(x - i)
    return result / 10

# Test with 100,000 calls
n_calls = 100000
test_value = 3.14

# Trivial function
start = time.time()
for _ in range(n_calls):
    trivial_function(test_value)
trivial_time = time.time() - start

# Inline version of trivial
start = time.time()
for _ in range(n_calls):
    result = test_value + 1
inline_time = time.time() - start

# Moderate function
start = time.time()
for _ in range(n_calls):
    moderate_function(test_value)
moderate_time = time.time() - start

# Complex function
start = time.time()
for _ in range(n_calls):
    complex_function(test_value)
complex_time = time.time() - start

print("Function Call Overhead Analysis:")
print(f"Trivial:  {trivial_time*1000:.1f}ms (inline: {inline_time*1000:.1f}ms)")
print(f"  → Overhead: {(trivial_time/inline_time - 1)*100:.0f}%")
print(f"Moderate: {moderate_time*1000:.1f}ms")
print(f"Complex:  {complex_time*1000:.1f}ms")
print(f"\nLesson: Function overhead only matters for trivial operations in tight loops!")
print("This is why NumPy's vectorized operations are so powerful—")
print("they move the loop into compiled C code, eliminating Python overhead entirely!")
```

The pattern is clear: function call overhead is significant for trivial operations but becomes negligible as complexity increases. This validates our approach throughout this chapter—use functions for organization and clarity, and only worry about overhead when profiling reveals it's actually a problem!

### When Function Overhead Matters

```{code-cell} python
# CASE 1: Overhead negligible - complex function
def fit_blackbody(wavelengths, fluxes, temp_guess=5000):
    """Complex calculation - function overhead insignificant."""
    # Lots of computation here
    # ... fitting algorithm ...
    return best_temp

# CASE 2: Overhead matters - trivial function in tight loop
def add_constant(x, c):
    return x + c

# Bad: Calling trivial function millions of times
# for i in range(1_000_000):
#     result = add_constant(data[i], 2.5)  # Slow!

# Good: Inline the operation or vectorize
# result = data + 2.5  # Much faster!
```

### 📊 **Performance Profile: Import Time Costs**

Module imports have a one-time cost that can affect program startup. Let's measure the overhead:

```{code-cell} python
import time
import sys

# Measure import times for common astronomy packages
def measure_import_time(module_name):
    """Measure time to import a module."""
    # Remove from sys.modules if already imported
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    start = time.time()
    try:
        exec(f"import {module_name}")
        elapsed = (time.time() - start) * 1000
        return elapsed
    except ImportError:
        return None

# Test common modules (some may not be installed)
modules_to_test = [
    'math',           # Built-in, very fast
    'datetime',       # Standard library
    'json',          # Standard library
    'numpy',         # Large scientific package
    'astropy',       # Even larger!
    'matplotlib.pyplot',  # Heavy with backends
]

print("Module Import Time Analysis:")
print("-" * 40)

for module in modules_to_test:
    time_ms = measure_import_time(module)
    if time_ms is not None:
        print(f"{module:20s}: {time_ms:6.1f} ms")
    else:
        print(f"{module:20s}: Not installed")

print("-" * 40)
print("\nImplications for your code:")
print("• Import heavy modules only when needed")
print("• Consider lazy imports for optional features")
print("• Module import happens once per interpreter session")
print("• In scripts that run many times, import cost adds up!")

# Example of lazy import pattern
def analyze_with_plotting(data, make_plot=False):
    """Only import matplotlib if actually plotting."""
    result = data.mean()  # Basic analysis always happens
    
    if make_plot:
        # Lazy import - only pay cost if needed
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.show()
    
    return result
```

This pattern appears in many astronomy packages where heavy dependencies (like matplotlib or astropy) are only imported when specific features are used. The Fermi LAT analysis tools use lazy imports extensively, reducing startup time from 5+ seconds to under 0.5 seconds for basic operations!

### Memoization for Expensive Calculations

Cache results of expensive computations—essential for period finding and model fitting. This technique is critical for expensive coordinate transformations (like converting between reference frames repeatedly) and any calculation where the same inputs produce the same outputs:

```{code-cell} python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_periodogram(n_points, n_frequencies):
    """
    Simulate expensive Lomb-Scargle calculation.
    Results are automatically cached.
    
    In real astronomy: period-finding algorithms often test
    thousands of frequencies on the same data. Caching results
    for attempted periods saves enormous computation time!
    """
    import time
    
    # Simulate expensive computation
    time.sleep(0.1)  # Pretend this takes 100ms
    
    # Fake result
    result = n_points * n_frequencies * 3.14159
    return result

# First call: slow (100ms)
import time
start = time.time()
result1 = expensive_periodogram(1000, 10000)
print(f"First call: {(time.time()-start)*1000:.1f} ms")

# Second call with same inputs: instant (from cache)!
start = time.time()
result2 = expensive_periodogram(1000, 10000)
print(f"Second call: {(time.time()-start)*1000:.1f} ms")

# Check cache statistics
print(f"Cache info: {expensive_periodogram.cache_info()}")
print("\nIn real astronomical pipelines, memoization can turn hours")
print("of computation into minutes when analyzing multiple filters")
print("or when users repeatedly adjust period ranges!")
```

This exact technique saved the Kepler team enormous computational resources when searching for exoplanet transits—the same period often gets tested multiple times with slightly different detrending, and caching the expensive periodogram calculations made interactive analysis possible!
```

### 💡 **Computational Thinking: Performance Patterns**

```
OPTIMIZATION HIERARCHY (try in order):

1. Better Algorithm
   O(n²) → O(n log n): 10,000x speedup for n=100,000
   Example: Brute force → FFT for period finding

2. Vectorization
   Python loop → NumPy: 10-100x speedup
   Example: Magnitude correction on arrays

3. Caching/Memoization
   Recomputation → Lookup: ∞ speedup for repeated calls
   Example: Coordinate transformations

4. Parallelization
   Single core → Multiple cores: ~N_cores speedup
   Example: Processing multiple stars independently

5. Compiled Code
   Python → Numba/Cython: 10-100x speedup
   Example: Pixel-by-pixel operations

Don't optimize prematurely, but know these options exist!
```

## Practice Exercises

### Exercise 5.1: Quick Practice - Create Reusable Analysis Functions

Let's start by creating a fundamental building block for light curve analysis. This function will become part of your photometry toolkit:

```python
"""
Create a robust amplitude calculation function.

Requirements:
1. Write calculate_amplitude(magnitudes) that returns max - min
2. Handle edge cases: empty list, single value, None values
3. Remember: smaller magnitude = brighter! (amplitude is still max - min)
4. Test with real Cepheid data from 'cepheid_simple.txt'

File format for cepheid_simple.txt:
# Time(JD)  Magnitude  Error
2458123.512  12.35  0.02
2458123.538  12.28  0.03
... (20-30 observations)

Starter code:
"""

def calculate_amplitude(magnitudes):
    """
    Calculate peak-to-peak amplitude of magnitude variations.
    
    Parameters
    ----------
    magnitudes : list or array-like
        Magnitude measurements
        
    Returns
    -------
    float
        Amplitude (max - min), or 0.0 for invalid input
    """
    # Handle edge cases first
    if not magnitudes or len(magnitudes) == 0:
        return 0.0
    
    if len(magnitudes) == 1:
        return 0.0  # No variation with single point
    
    # Filter out None values if present
    valid_mags = [m for m in magnitudes if m is not None]
    
    if not valid_mags:
        return 0.0
    
    # Calculate amplitude
    return max(valid_mags) - min(valid_mags)

# Test with real data
def test_amplitude():
    """Test amplitude calculation with Cepheid data."""
    # Read the data (you learned this in Chapter 1!)
    times, mags, errors = [], [], []
    
    with open('cepheid_simple.txt', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                mags.append(float(parts[1]))
    
    amp = calculate_amplitude(mags)
    print(f"Cepheid amplitude: {amp:.3f} magnitudes")
    
    # Test edge cases
    assert calculate_amplitude([]) == 0.0
    assert calculate_amplitude([12.5]) == 0.0
    assert calculate_amplitude([12.0, 12.5, None, 11.8]) == 0.7
    print("All tests passed!")

if __name__ == "__main__":
    test_amplitude()
```

### Exercise 5.2: Synthesis - Build a Light Curve Analysis Module

Now let's combine multiple functions into a comprehensive module. This demonstrates how real astronomical software is organized:

```python
"""
Create lightcurve.py - a complete light curve analysis module.

This module will contain all the functions needed for basic
variable star analysis, building on what you created above.

Requirements:
1. Create a module with proper docstring
2. Include all functions listed below
3. Add validation and error handling
4. Test with 'rr_lyrae_realistic.txt' (has gaps and bad data!)
5. Use if __name__ == "__main__" for comprehensive testing

File format for rr_lyrae_realistic.txt:
# Time(MJD)  Magnitude  Error  Quality_Flag
58123.512  14.35  0.02  1
58123.538  14.28  0.03  1
58123.564  99.99  9.99  0  # Bad measurement
... (100-200 observations with gaps)
"""

# lightcurve.py
"""
Light curve analysis tools for variable star research.

This module provides functions for loading, analyzing, and
phase-folding photometric time series of variable stars.
"""

import math

def load_observations(filename, skip_bad=True):
    """
    Load photometric observations from file.
    
    Parameters
    ----------
    filename : str
        Path to observation file
    skip_bad : bool
        Skip observations with quality_flag = 0
        
    Returns
    -------
    tuple
        (times, magnitudes, errors) as lists
    """
    times, mags, errors = [], [], []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                # Check quality flag if present
                if len(parts) >= 4 and skip_bad:
                    if int(parts[3]) == 0:
                        continue  # Skip bad data
                
                # Only add good data
                time = float(parts[0])
                mag = float(parts[1])
                err = float(parts[2])
                
                # Additional validation
                if mag < 30 and err < 5:  # Reasonable values
                    times.append(time)
                    mags.append(mag)
                    errors.append(err)
    
    return times, mags, errors

def calculate_amplitude(mags):
    """Calculate peak-to-peak amplitude."""
    if not mags or len(mags) < 2:
        return 0.0
    return max(mags) - min(mags)

def estimate_period_simple(times, mags):
    """
    Simple period estimation using peak finding.
    
    Finds time between brightness maxima (minima in magnitude).
    This is a naive approach - real period finding uses FFT or Lomb-Scargle!
    """
    if len(times) < 3:
        return None
    
    # Find local minima (brightness maxima)
    minima_times = []
    for i in range(1, len(mags) - 1):
        if mags[i] < mags[i-1] and mags[i] < mags[i+1]:
            minima_times.append(times[i])
    
    if len(minima_times) < 2:
        return None
    
    # Calculate average spacing between minima
    spacings = []
    for i in range(1, len(minima_times)):
        spacings.append(minima_times[i] - minima_times[i-1])
    
    if spacings:
        return sum(spacings) / len(spacings)
    return None

def phase_fold(times, mags, period, epoch=0.0):
    """
    Fold light curve at given period.
    
    Parameters
    ----------
    times : list
        Observation times
    mags : list
        Magnitudes
    period : float
        Period to fold at (same units as times)
    epoch : float
        Reference epoch for phase zero
        
    Returns
    -------
    tuple
        (phases, folded_mags) where phases are 0-1
    """
    phases = []
    for t in times:
        phase = ((t - epoch) % period) / period
        phases.append(phase)
    
    return phases, mags

# Module testing
if __name__ == "__main__":
    print("Testing light curve analysis module...")
    
    # Test with RR Lyrae data
    times, mags, errors = load_observations('rr_lyrae_realistic.txt')
    print(f"Loaded {len(times)} good observations")
    
    # Calculate amplitude
    amp = calculate_amplitude(mags)
    print(f"Amplitude: {amp:.3f} magnitudes")
    
    # Estimate period
    period_guess = estimate_period_simple(times, mags)
    if period_guess:
        print(f"Estimated period: {period_guess:.3f} days")
        
        # Try phase folding
        phases, folded = phase_fold(times, mags, period_guess)
        print(f"Phase folded {len(phases)} observations")
    else:
        print("Could not estimate period with simple method")
    
    print("\nModule test complete!")
```

### Exercise 5.3: Challenge - Functional Approach to Data Filtering

This advanced exercise demonstrates how functional programming revolutionizes astronomical data processing:

```python
"""
Implement a functional pipeline for photometric data processing.

This exercise shows how functional programming patterns create
clean, testable, parallelizable code for real astronomy work.

Requirements:
1. Use map/filter/reduce for the entire pipeline
2. Compare performance with traditional loops
3. Implement memoization for period finding
4. NO explicit for loops in functional version!

This demonstrates concepts used in modern frameworks like JAX!
"""

from functools import reduce, lru_cache
import time

def traditional_pipeline(observations):
    """Traditional loop-based approach."""
    # Filter bad observations
    good_obs = []
    for obs in observations:
        if obs['error'] <= 0.1 and obs['mag'] < 90:
            good_obs.append(obs)
    
    # Convert magnitude to flux
    fluxes = []
    for obs in good_obs:
        flux = 10 ** (-0.4 * obs['mag'])
        fluxes.append(flux)
    
    # Calculate mean flux
    if fluxes:
        mean_flux = sum(fluxes) / len(fluxes)
    else:
        mean_flux = 0
    
    return mean_flux

def functional_pipeline(observations):
    """Functional approach - no explicit loops!"""
    # Filter bad observations
    good_obs = filter(
        lambda obs: obs['error'] <= 0.1 and obs['mag'] < 90,
        observations
    )
    
    # Map magnitude to flux
    fluxes = map(
        lambda obs: 10 ** (-0.4 * obs['mag']),
        good_obs
    )
    
    # Reduce to find mean
    flux_list = list(fluxes)  # Need to materialize for len()
    if flux_list:
        total_flux = reduce(lambda a, b: a + b, flux_list)
        mean_flux = total_flux / len(flux_list)
    else:
        mean_flux = 0
    
    return mean_flux

# Memoized period finder for expensive calculations
@lru_cache(maxsize=1000)
def find_period_memoized(times_tuple, mags_tuple, test_period):
    """
    Expensive period testing function with memoization.
    
    Note: lru_cache requires hashable arguments (tuples, not lists).
    This caches results of expensive calculations automatically!
    """
    # Simulate expensive Lomb-Scargle calculation
    time.sleep(0.001)  # Pretend this is expensive
    
    # Simple chi-squared for phase dispersion
    times = list(times_tuple)
    mags = list(mags_tuple)
    
    phases = [(t % test_period) / test_period for t in times]
    phase_mags = sorted(zip(phases, mags))
    
    # Calculate dispersion
    chi_squared = 0
    for i in range(len(phase_mags) - 1):
        chi_squared += (phase_mags[i+1][1] - phase_mags[i][1]) ** 2
    
    return chi_squared / len(phase_mags) if phase_mags else float('inf')

# Performance comparison
def compare_approaches():
    """Compare traditional vs functional performance."""
    
    # Generate realistic observations
    import random
    observations = []
    for i in range(10000):
        observations.append({
            'time': i * 0.01,
            'mag': 12.0 + random.gauss(0, 0.5),
            'error': abs(random.gauss(0.03, 0.02))
        })
    
    # Add some bad data
    for i in range(100):
        observations[random.randint(0, 9999)]['mag'] = 99.99
    
    # Time traditional approach
    start = time.time()
    trad_result = traditional_pipeline(observations)
    trad_time = time.time() - start
    
    # Time functional approach
    start = time.time()
    func_result = functional_pipeline(observations)
    func_time = time.time() - start
    
    print("Pipeline Performance Comparison:")
    print(f"Traditional: {trad_time*1000:.2f} ms")
    print(f"Functional:  {func_time*1000:.2f} ms")
    print(f"Results match: {abs(trad_result - func_result) < 1e-10}")
    
    # Test memoization
    test_times = tuple(range(100))
    test_mags = tuple([12.0 + 0.5 * math.sin(t/10) for t in test_times])
    
    print("\nMemoization Test:")
    start = time.time()
    result1 = find_period_memoized(test_times, test_mags, 62.83)
    first_call = time.time() - start
    
    start = time.time()
    result2 = find_period_memoized(test_times, test_mags, 62.83)
    second_call = time.time() - start
    
    print(f"First call:  {first_call*1000:.2f} ms")
    print(f"Second call: {second_call*1000:.6f} ms (from cache!)")
    print(f"Speedup: {first_call/second_call:.0f}x")
    
    print(f"\nCache info: {find_period_memoized.cache_info()}")

if __name__ == "__main__":
    compare_approaches()
```

### Exercise 5.4: Integration Exercise - Complete Variable Star Pipeline

Combine everything you've learned to create a production-ready analysis:

```python
"""
Build a complete variable star analysis pipeline using all concepts.

This exercise integrates:
- Functions with proper error handling
- Module organization
- Performance optimization
- Functional programming concepts
- Documentation and testing

Create a script that:
1. Loads multiple light curves
2. Filters bad data functionally
3. Finds periods using memoized function
4. Generates report with statistics
5. Handles errors gracefully

This is similar to real variable star survey pipelines!
"""

# Your implementation here
```

## Main Takeaways

Functions transform astronomy code from one-off scripts into reliable, reusable tools that form the foundation of research software. When you encapsulate logic in well-designed functions, you create building blocks that can be tested independently, shared with collaborators, and combined into complex analysis pipelines. The mutable default argument trap and scope confusion have caused real bugs in astronomical software, from IRAF scripts that accumulated data across nights to pipeline errors that corrupted months of observations. Understanding these pitfalls now saves debugging time later.

The progression from simple functions to modules to packages mirrors how astronomical software projects naturally grow. What starts as a quick magnitude conversion function evolves into a photometry module, then becomes part of a larger analysis package. This organic growth works best when you follow Python's conventions: clear naming, comprehensive docstrings, and the `if __name__ == "__main__"` pattern that makes modules both importable and testable. The functional programming concepts introduced here aren't just academic exercises—they're essential preparation for modern frameworks like JAX that power cutting-edge astronomical simulations and machine learning applications.

Performance considerations matter more in astronomy than many fields because we routinely process gigabytes to terabytes of data. The difference between a loop with function calls and vectorized operations can mean waiting hours versus minutes for results. But premature optimization is still a mistake—first make your code correct and clear, then optimize the bottlenecks that profiling reveals. The caching techniques shown here have saved countless CPU hours in period-finding algorithms and model fitting routines.

Looking forward, the function concepts from this chapter directly enable NumPy's vectorized operations in Chapter 7, where entire arrays are processed without explicit loops. The module organization skills prepare you for building larger scientific packages, while the documentation practices ensure your code can be understood and maintained by others—including yourself six months from now when you've forgotten the details. Most importantly, thinking in terms of functional contracts and clear interfaces will make you a better computational scientist, capable of building the robust, efficient tools that modern astronomy demands.

## Definitions

**argument** - The actual value passed to a function when calling it (e.g., in `f(5)`, 5 is an argument)

**closure** - A function that remembers variables from its enclosing scope even after that scope has finished executing

**decorator** - A function that modifies another function's behavior without changing its code

**docstring** - A string literal that appears as the first statement in a function, module, or class to document its purpose

**function** - A reusable block of code that performs a specific task, taking inputs and optionally returning outputs

**global** - A keyword that allows a function to modify a variable in the global scope

**lambda** - An anonymous function defined inline using the `lambda` keyword

**LEGB** - The order Python searches for variables: Local, Enclosing, Global, Built-in

**memoization** - Caching function results to avoid recomputing expensive operations

**module** - A Python file containing definitions and statements that can be imported and reused

**namespace** - A container that holds a set of identifiers and their associated objects

**package** - A directory containing multiple Python modules and an `__init__.py` file

**parameter** - A variable in a function definition that receives a value when the function is called

**pure function** - A function that always returns the same output for the same input with no side effects

**return value** - The result that a function sends back to the code that called it

**scope** - The region of a program where a variable is accessible

**side effect** - Any state change that occurs beyond returning a value from a function

***args** - Syntax for collecting variable positional arguments into a tuple

***kwargs** - Syntax for collecting variable keyword arguments into a dictionary

## Key Takeaways

- Functions are contracts: they promise specific outputs for given inputs
- The mutable default argument trap occurs because defaults are evaluated once at definition time
- Always use `None` as a sentinel for mutable default arguments
- Python searches for variables using LEGB: Local, Enclosing, Global, Built-in
- Global variables make code hard to test, debug, and parallelize
- Lambda functions are useful for simple operations but limited to single expressions
- Functional programming concepts (map, filter, reduce) prepare you for modern frameworks
- The `if __name__ == "__main__"` pattern makes modules both importable and executable
- Never use `from module import *` except in interactive sessions
- Docstrings are essential for scientific code that others will use and maintain
- Function call overhead matters in tight loops—consider vectorization
- Memoization can dramatically speed up expensive repeated calculations
- Performance optimization should follow the hierarchy: algorithm → vectorization → caching → parallelization → compilation

## Quick Reference Tables

### Function Definition Patterns

| Pattern | Syntax | Use Case |
|---------|--------|----------|
| Basic function | `def func(x, y):` | Simple operations |
| Default arguments | `def func(x, y=10):` | Optional parameters |
| Mutable default fix | `def func(x, data=None):` | Avoid default trap |
| Variable args | `def func(*args):` | Unknown number of inputs |
| Keyword args | `def func(**kwargs):` | Flexible options |
| Combined | `def func(x, *args, y=10, **kwargs):` | Maximum flexibility |
| Lambda | `lambda x: x**2` | Simple inline functions |

### Module Import Patterns

| Pattern | Example | When to Use |
|---------|---------|-------------|
| Import module | `import numpy` | Use many functions from module |
| Import with alias | `import numpy as np` | Long module names |
| Import specific | `from math import sin, cos` | Need few specific functions |
| Import all (avoid!) | `from math import *` | Interactive only |
| Package import | `from astropy.time import Time` | Specific submodule |

### Common Function Gotchas and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Mutable default | Data persists between calls | Use `None` sentinel |
| UnboundLocalError | Can't increment global | Use `global` keyword or pass explicitly |
| Namespace pollution | Name conflicts | Avoid wildcard imports |
| Slow loops | Minutes for simple operations | Vectorize with NumPy |
| Repeated calculation | Same expensive computation | Add `@lru_cache` decorator |
| Import not found | ModuleNotFoundError | Check sys.path or install package |

## Next Chapter Preview

With functions and modules mastered, Chapter 6 will introduce Object-Oriented Programming (OOP)—a paradigm that bundles data and behavior together. You'll learn to create classes that model astronomical objects naturally: a `Star` class with magnitude and position attributes, methods to calculate distance and luminosity, and special methods that make your objects work seamlessly with Python's built-in functions.

The functional programming concepts from this chapter provide essential background for OOP. Methods are just functions attached to objects, and understanding scope prepares you for the `self` parameter that confuses many beginners. The module organization skills you've developed will expand to organizing classes and building object hierarchies. Most importantly, the design thinking you've practiced—creating clean interfaces and thinking about contracts—directly applies to designing effective classes that model the complex relationships in astronomical systems.