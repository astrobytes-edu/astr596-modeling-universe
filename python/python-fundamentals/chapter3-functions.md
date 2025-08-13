# Chapter 3: Functions - The Art of Abstraction

*"The function of good software is to make the complex appear to be simple." - Grady Booch*

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [Why Functions Matter](#why-functions-matter)
   - [The Philosophy of Abstraction](#the-philosophy-of-abstraction)
   - [DRY: Don't Repeat Yourself](#dry-dont-repeat-yourself)
3. [Function Fundamentals](#function-fundamentals)
   - [Basic Syntax and Return](#basic-syntax-and-return)
   - [Parameters vs Arguments](#parameters-vs-arguments)
   - [The None Return](#the-none-return)
4. [Function Arguments: The Full Story](#function-arguments-the-full-story)
   - [Positional Arguments](#positional-arguments)
   - [Default Arguments and Their Dangers](#default-arguments-and-their-dangers)
   - [Keyword Arguments](#keyword-arguments)
   - [*args: Variable Positional Arguments](#args-variable-positional-arguments)
   - [**kwargs: Variable Keyword Arguments](#kwargs-variable-keyword-arguments)
   - [The Complete Order](#the-complete-order)
5. [Scope and Namespaces](#scope-and-namespaces)
   - [The LEGB Rule](#the-legb-rule)
   - [Global Variables: Handle with Care](#global-variables-handle-with-care)
   - [Nonlocal: Nested Function Scopes](#nonlocal-nested-function-scopes)
6. [Functions as First-Class Objects](#functions-as-first-class-objects)
   - [Functions as Arguments](#functions-as-arguments)
   - [Functions Returning Functions](#functions-returning-functions)
   - [Lambda Functions](#lambda-functions)
   - [Decorators: Function Transformers](#decorators-function-transformers)
7. [Documentation and Type Hints](#documentation-and-type-hints)
   - [Docstrings That Matter](#docstrings-that-matter)
   - [Type Hints for Clarity](#type-hints-for-clarity)
8. [Advanced Patterns](#advanced-patterns)
   - [Recursion](#recursion)
   - [Closures](#closures)
   - [Partial Functions](#partial-functions)
9. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)
10. [Exercises](#exercises)
11. [Key Takeaways](#key-takeaways)

---

## Learning Objectives

```{admonition} By the end of this chapter you will:
:class: info
- Write functions that are modular, reusable, and testable
- Master all argument types: positional, keyword, *args, **kwargs
- Understand scope rules and avoid common namespace pitfalls
- Use functions as first-class objects (pass, return, transform)
- Write comprehensive docstrings and use type hints
- Apply functional programming concepts to scientific code
- Recognize when and how to use advanced patterns like decorators
```

## Why Functions Matter

### The Philosophy of Abstraction

Functions are humanity's tool for managing complexity. They let us think at different levels:

```{code-cell} ipython3
# Level 1: The messy details
def calculate_schwarzschild_radius_verbose():
    G = 6.67430e-11  # gravitational constant in m³/kg/s²
    c = 299792458    # speed of light in m/s
    M = 1.98892e30   # solar mass in kg
    
    # Schwarzschild radius formula
    numerator = 2 * G * M
    denominator = c * c
    r_s = numerator / denominator
    
    print(f"For calculation:")
    print(f"  G = {G:.5e} m³/kg/s²")
    print(f"  c = {c:.5e} m/s")
    print(f"  M = {M:.5e} kg")
    print(f"  r_s = 2GM/c² = {r_s:.3f} m")
    
    return r_s

# Level 2: The abstraction
def schwarzschild_radius(mass_kg):
    """Calculate Schwarzschild radius for given mass."""
    G = 6.67430e-11
    c = 299792458
    return 2 * G * mass_kg / c**2

# Level 3: The application
def is_black_hole(mass_kg, radius_m):
    """Check if object is within its Schwarzschild radius."""
    return radius_m < schwarzschild_radius(mass_kg)

# Now we can think at the problem level, not the math level
solar_mass = 1.98892e30
print(f"Sun compressed to 1km: Black hole? {is_black_hole(solar_mass, 1000)}")
print(f"Sun compressed to 10km: Black hole? {is_black_hole(solar_mass, 10000)}")
```

### DRY: Don't Repeat Yourself

Every duplicated piece of code is a bug waiting to happen:

```{code-cell} ipython3
# ❌ BAD: Repeated code
mag1 = 5.2
flux1 = 100 * 10**(-0.4 * mag1)  # Pogson's formula
print(f"Magnitude {mag1} → Flux {flux1:.2f}")

mag2 = 3.7
flux2 = 100 * 10**(-0.4 * mag2)  # Same formula, repeated
print(f"Magnitude {mag2} → Flux {flux2:.2f}")

mag3 = 6.1
flux3 = 100 * 10**(-0.4 * mag3)  # And again...
print(f"Magnitude {mag3} → Flux {flux3:.2f}")

print("\n" + "="*40 + "\n")

# ✅ GOOD: Function eliminates repetition
def magnitude_to_flux(magnitude, zero_point_flux=100):
    """Convert astronomical magnitude to flux using Pogson's formula."""
    return zero_point_flux * 10**(-0.4 * magnitude)

# Now if we need to change the formula, we change it in ONE place
for mag in [5.2, 3.7, 6.1]:
    flux = magnitude_to_flux(mag)
    print(f"Magnitude {mag} → Flux {flux:.2f}")
```

---

## Function Fundamentals

### Basic Syntax and Return

```{code-cell} ipython3
def kinetic_energy(mass, velocity):
    """
    Calculate kinetic energy.
    
    Parameters
    ----------
    mass : float
        Mass in kg
    velocity : float
        Velocity in m/s
        
    Returns
    -------
    float
        Kinetic energy in Joules
    """
    return 0.5 * mass * velocity**2

# Multiple return values
def orbital_parameters(semi_major_axis_au, eccentricity):
    """Calculate perihelion and aphelion distances."""
    perihelion = semi_major_axis_au * (1 - eccentricity)
    aphelion = semi_major_axis_au * (1 + eccentricity)
    return perihelion, aphelion  # Returns a tuple

# Using the functions
energy = kinetic_energy(1000, 7900)  # 1000kg at orbital velocity
print(f"Kinetic energy: {energy:.2e} J")

peri, aph = orbital_parameters(1.0, 0.0167)  # Earth's orbit
print(f"Earth: Perihelion = {peri:.3f} AU, Aphelion = {aph:.3f} AU")
```

### Parameters vs Arguments

```{code-cell} ipython3
# Parameters are the variables in the function definition
def greet(name, greeting="Hello"):  # 'name' and 'greeting' are parameters
    return f"{greeting}, {name}!"

# Arguments are the actual values passed when calling
result = greet("Andromeda", "Greetings")  # "Andromeda" and "Greetings" are arguments
print(result)

# This distinction matters when discussing function behavior!
```

### The None Return

Functions always return something, even if you don't specify:

```{code-cell} ipython3
def print_only(message):
    """This function doesn't explicitly return anything."""
    print(f"Message: {message}")
    # Implicit: return None

result = print_only("Testing")
print(f"Return value: {result}")
print(f"Type: {type(result)}")

# Explicit None return for early exit
def safe_divide(a, b):
    """Divide with safety check."""
    if b == 0:
        return None  # Explicit None for error case
    return a / b

print(f"10/2 = {safe_divide(10, 2)}")
print(f"10/0 = {safe_divide(10, 0)}")
```

---

## Function Arguments: The Full Story

### Positional Arguments

Order matters for positional arguments:

```{code-cell} ipython3
def calculate_redshift(observed_wavelength, rest_wavelength):
    """Calculate redshift z from wavelengths."""
    return (observed_wavelength - rest_wavelength) / rest_wavelength

# Order matters!
z1 = calculate_redshift(656.3, 486.1)  # Wrong order
z2 = calculate_redshift(486.1, 656.3)  # Also wrong
z_correct = calculate_redshift(750.0, 656.3)  # Correct: observed, then rest

print(f"Wrong: z = {z1:.3f}")
print(f"Also wrong: z = {z2:.3f}")
print(f"Correct (H-alpha redshifted): z = {z_correct:.3f}")
```

### Default Arguments and Their Dangers

Default arguments are evaluated ONCE when the function is defined:

```{code-cell} ipython3
# ⚠️ DANGER: Mutable default argument
def add_observation_bad(obs, obs_list=[]):  # DON'T DO THIS!
    obs_list.append(obs)
    return obs_list

# Watch what happens:
list1 = add_observation_bad("Galaxy A")
list2 = add_observation_bad("Galaxy B")  # Where did Galaxy A come from?!
print(f"list1: {list1}")
print(f"list2: {list2}")
print(f"Same object? {list1 is list2}")  # They're the same list!

print("\n" + "="*40 + "\n")

# ✅ CORRECT: Use None as default for mutable objects
def add_observation_good(obs, obs_list=None):
    if obs_list is None:
        obs_list = []  # Create new list each time
    obs_list.append(obs)
    return obs_list

# Now it works correctly:
list3 = add_observation_good("Galaxy C")
list4 = add_observation_good("Galaxy D")
print(f"list3: {list3}")
print(f"list4: {list4}")
print(f"Same object? {list3 is list4}")  # Different lists!
```

```{warning} The Mutable Default Trap
This is one of Python's most common gotchas! 
- Default values are evaluated ONCE when the function is defined
- Lists, dicts, and sets are mutable and will be shared across calls
- Always use `None` as default for mutable objects
```

### Keyword Arguments

Use names for clarity and flexibility:

```{code-cell} ipython3
def simulate_orbit(
    mass1, mass2,  # Positional arguments
    eccentricity=0,  # Keyword arguments with defaults
    inclination=0,
    time_steps=1000,
    integrator="verlet"
):
    """Simulate a two-body orbit."""
    print(f"Simulating: M1={mass1}, M2={mass2}")
    print(f"  e={eccentricity}, i={inclination}°")
    print(f"  {time_steps} steps using {integrator}")
    return f"Orbit data for {time_steps} steps"

# Can use positional and keyword arguments
result1 = simulate_orbit(1.0, 0.5)  # Just positional
result2 = simulate_orbit(1.0, 0.5, eccentricity=0.3)  # Mix
result3 = simulate_orbit(1.0, 0.5, time_steps=5000, eccentricity=0.1)  # Any order for keywords!
```

### *args: Variable Positional Arguments

Accept any number of positional arguments:

```{code-cell} ipython3
def total_luminosity(*star_luminosities):
    """
    Calculate total luminosity of multiple stars.
    
    Parameters
    ----------
    *star_luminosities : float
        Variable number of luminosity values (solar units)
    """
    print(f"Received {len(star_luminosities)} stars")
    print(f"Type of args: {type(star_luminosities)}")  # It's a tuple!
    
    total = sum(star_luminosities)
    return total

# Can call with any number of arguments
print(f"Single star: {total_luminosity(1.0)} L☉")
print(f"Binary: {total_luminosity(1.0, 0.5)} L☉")
print(f"Cluster: {total_luminosity(1.0, 0.5, 2.3, 0.1, 3.5)} L☉")

# Can also unpack a list with *
cluster = [1.0, 0.5, 2.3, 0.1, 3.5]
print(f"From list: {total_luminosity(*cluster)} L☉")  # Note the *
```

### **kwargs: Variable Keyword Arguments

Accept any number of keyword arguments:

```{code-cell} ipython3
def create_star_catalog(**star_properties):
    """
    Create a star catalog with arbitrary properties.
    
    Parameters
    ----------
    **star_properties : various
        Arbitrary keyword arguments for star properties
    """
    print(f"Type of kwargs: {type(star_properties)}")  # It's a dict!
    
    catalog = "Star Catalog:\n"
    for property_name, value in star_properties.items():
        catalog += f"  {property_name}: {value}\n"
    
    return catalog

# Can pass any keyword arguments
print(create_star_catalog(
    name="Sirius A",
    spectral_type="A1V",
    magnitude=-1.46,
    distance_pc=2.64
))

# Can also unpack a dictionary with **
vega_data = {
    'name': 'Vega',
    'spectral_type': 'A0V',
    'magnitude': 0.03,
    'distance_pc': 7.68,
    'rotation_km_s': 236
}
print(create_star_catalog(**vega_data))  # Note the **
```

### The Complete Order

When combining all argument types, they must appear in this order:

```{code-cell} ipython3
def ultimate_function(
    pos1, pos2,           # Regular positional arguments
    *args,                # Variable positional arguments
    kwonly1, kwonly2=10,  # Keyword-only arguments (after *args)
    **kwargs              # Variable keyword arguments
):
    """Demonstrates the complete argument order."""
    print(f"Positional: {pos1}, {pos2}")
    print(f"*args: {args}")
    print(f"Keyword-only: {kwonly1}, {kwonly2}")
    print(f"**kwargs: {kwargs}")

# Must provide keyword-only arguments by name
ultimate_function(
    1, 2,                    # Positional
    3, 4, 5,                 # Extra positional (*args)
    kwonly1="required",      # Keyword-only (required)
    kwonly2="optional",      # Keyword-only (has default)
    extra1="bonus",          # Extra keywords (**kwargs)
    extra2="more"
)

# Force keyword-only without *args using bare *
def keyword_only_example(*, name, value=0):
    """After *, all arguments must be passed by name."""
    return f"{name} = {value}"

# keyword_only_example("test", 5)  # ERROR: won't work
print(keyword_only_example(name="test", value=5))  # Must use names
```

---

## Scope and Namespaces

### The LEGB Rule

Python searches for variables in this order: Local → Enclosing → Global → Built-in

```{code-cell} ipython3
# Global scope
galaxy_name = "Milky Way"  # Global variable

def outer_function():
    # Enclosing scope
    star_count = 400_000_000_000  # Enclosing for inner_function
    
    def inner_function():
        # Local scope
        planet_count = 8  # Local variable
        
        # LEGB in action
        print(f"Local: {planet_count} planets")
        print(f"Enclosing: {star_count} stars")
        print(f"Global: In the {galaxy_name}")
        print(f"Built-in: sum function is {sum}")
        
    inner_function()

outer_function()

# Shadowing: Local variables can hide outer ones
def shadow_example():
    galaxy_name = "Andromeda"  # Shadows global galaxy_name
    print(f"Inside function: {galaxy_name}")

shadow_example()
print(f"Outside function: {galaxy_name}")  # Global unchanged
```

### Global Variables: Handle with Care

```{code-cell} ipython3
# Global variable (generally avoid these!)
observation_count = 0

def add_observation_global():
    global observation_count  # Declare we're modifying global
    observation_count += 1
    return observation_count

# Modifying global state (usually bad practice)
print(f"Initial count: {observation_count}")
add_observation_global()
add_observation_global()
print(f"After two calls: {observation_count}")

# Better approach: Pass and return state
def add_observation_pure(count):
    """Pure function - no side effects."""
    return count + 1

# Much cleaner and testable
count = 0
count = add_observation_pure(count)
count = add_observation_pure(count)
print(f"Pure function result: {count}")
```

```{admonition} Why Avoid Global Variables?
:class: warning
- Makes code hard to test (tests affect each other)
- Creates hidden dependencies
- Concurrent code becomes dangerous
- Debugging becomes difficult

**Better alternatives:**
- Pass parameters explicitly
- Use classes to encapsulate state
- Return updated values
```

### Nonlocal: Nested Function Scopes

```{code-cell} ipython3
def make_counter():
    """Create a closure that counts calls."""
    count = 0
    
    def increment():
        nonlocal count  # Modify enclosing scope variable
        count += 1
        return count
    
    return increment

# Create independent counters
counter1 = make_counter()
counter2 = make_counter()

print(f"Counter1: {counter1()}, {counter1()}, {counter1()}")
print(f"Counter2: {counter2()}, {counter2()}")  # Independent!
```

---

## Functions as First-Class Objects

In Python, functions are objects like any other - you can pass them, return them, and store them:

### Functions as Arguments

```{code-cell} ipython3
import math

def apply_to_list(data, function):
    """Apply a function to each element in a list."""
    return [function(x) for x in data]

# Different functions to apply
magnitudes = [1.5, 2.3, 0.8, 3.1]

# Pass different functions
fluxes = apply_to_list(magnitudes, lambda m: 10**(-0.4 * m))
logs = apply_to_list(magnitudes, math.log10)
squares = apply_to_list(magnitudes, lambda x: x**2)

print(f"Magnitudes: {magnitudes}")
print(f"To fluxes: {[f'{f:.3f}' for f in fluxes]}")
print(f"Logarithms: {[f'{l:.3f}' for l in logs]}")
print(f"Squares: {[f'{s:.3f}' for s in squares]}")

# Real example: Numerical integration with different functions
def integrate(func, a, b, n=1000):
    """Simple numerical integration using rectangles."""
    dx = (b - a) / n
    total = 0
    for i in range(n):
        x = a + i * dx
        total += func(x) * dx
    return total

# Integrate different functions
result1 = integrate(math.sin, 0, math.pi)  # ∫sin(x) from 0 to π
result2 = integrate(lambda x: x**2, 0, 1)  # ∫x² from 0 to 1

print(f"\n∫sin(x) from 0 to π = {result1:.4f} (expected: 2)")
print(f"∫x² from 0 to 1 = {result2:.4f} (expected: 0.333...)")
```

### Functions Returning Functions

```{code-cell} ipython3
def make_magnitude_converter(zero_point_flux):
    """
    Create a magnitude-to-flux converter for a specific zero point.
    
    This is a 'function factory' - it returns customized functions.
    """
    def converter(magnitude):
        return zero_point_flux * 10**(-0.4 * magnitude)
    
    # Return the inner function
    return converter

# Create converters for different photometric systems
vega_converter = make_magnitude_converter(3.631e-20)  # Vega system
ab_converter = make_magnitude_converter(3.631e-23)    # AB system

mag = 20.0
print(f"Magnitude {mag}:")
print(f"  Vega system: {vega_converter(mag):.3e} W/m²/Hz")
print(f"  AB system: {ab_converter(mag):.3e} W/m²/Hz")

# Another example: Creating custom filters
def make_filter(lower, upper):
    """Create a filter function for a wavelength range."""
    def filter_func(wavelength):
        return lower <= wavelength <= upper
    
    filter_func.__name__ = f"filter_{lower}_{upper}"
    return filter_func

# Create filters for different bands
u_band = make_filter(300, 400)  # U band in nm
g_band = make_filter(400, 550)  # G band in nm

wavelength = 450
print(f"\nWavelength {wavelength}nm:")
print(f"  In U band? {u_band(wavelength)}")
print(f"  In G band? {g_band(wavelength)}")
```

### Lambda Functions

Anonymous functions for simple operations:

```{code-cell} ipython3
# Lambda syntax: lambda arguments: expression

# Regular function
def square(x):
    return x**2

# Equivalent lambda
square_lambda = lambda x: x**2

print(f"Regular: {square(5)}")
print(f"Lambda: {square_lambda(5)}")

# Lambdas shine in functional programming
data = [
    {'name': 'Sirius', 'mag': -1.46},
    {'name': 'Canopus', 'mag': -0.74},
    {'name': 'Arcturus', 'mag': -0.05},
    {'name': 'Vega', 'mag': 0.03}
]

# Sort by magnitude
data_sorted = sorted(data, key=lambda star: star['mag'])
print("\nStars by brightness:")
for star in data_sorted:
    print(f"  {star['name']}: {star['mag']}")

# Filter bright stars
bright = filter(lambda s: s['mag'] < 0, data)
print("\nBright stars (mag < 0):")
for star in bright:
    print(f"  {star['name']}")
```

### Decorators: Function Transformers

Decorators modify or enhance functions:

```{code-cell} ipython3
import time
import functools

# Simple decorator to time function execution
def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.6f} seconds")
        return result
    return wrapper

# Apply decorator with @
@timer
def slow_calculation(n):
    """Simulate a slow calculation."""
    total = 0
    for i in range(n):
        total += i**2
    return total

result = slow_calculation(1000000)
print(f"Result: {result}")

# Decorator with arguments
def validate_range(min_val, max_val):
    """Decorator factory that validates input range."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(value):
            if not min_val <= value <= max_val:
                raise ValueError(f"Value {value} outside range [{min_val}, {max_val}]")
            return func(value)
        return wrapper
    return decorator

@validate_range(0, 1)
def calculate_probability(p):
    """Calculate something with probability."""
    return p * (1 - p)

print(f"\nProbability(0.3) = {calculate_probability(0.3):.3f}")
# calculate_probability(1.5)  # Would raise ValueError
```

---

## Documentation and Type Hints

### Docstrings That Matter

```{code-cell} ipython3
def calculate_orbital_period(semi_major_axis: float, 
                           total_mass: float) -> float:
    """
    Calculate orbital period using Kepler's third law.
    
    For a two-body system, calculates the orbital period given
    the semi-major axis and total system mass.
    
    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis in AU
    total_mass : float
        Total mass of system in solar masses
    
    Returns
    -------
    float
        Orbital period in years
        
    Notes
    -----
    Uses the simplified form of Kepler's third law:
    P² = a³/M where P is in years, a in AU, M in solar masses
    
    Examples
    --------
    >>> calculate_orbital_period(1.0, 1.0)  # Earth around Sun
    1.0
    >>> calculate_orbital_period(5.2, 1.0)  # Jupiter around Sun
    11.86
    
    References
    ----------
    .. [1] Carroll & Ostlie, "An Introduction to Modern Astrophysics"
    """
    return (semi_major_axis**3 / total_mass)**0.5

# Access docstring
print(calculate_orbital_period.__doc__)
```

### Type Hints for Clarity

```{code-cell} ipython3
from typing import List, Tuple, Optional, Union, Callable, Dict

def process_spectrum(
    wavelengths: List[float],
    fluxes: List[float],
    normalize: bool = True,
    smooth_window: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Process a spectrum with optional normalization and smoothing.
    
    Type hints make the expected inputs and outputs clear!
    """
    # Processing would happen here
    return wavelengths, fluxes

def find_spectral_lines(
    spectrum: Dict[str, List[float]],
    threshold: float = 3.0,
    method: Callable[[List[float]], float] = max
) -> Union[List[float], None]:
    """
    Find spectral lines in a spectrum.
    
    Shows complex type hints including Callable and Union.
    """
    if not spectrum:
        return None
    
    # Line finding logic here
    return [656.3, 486.1]  # H-alpha, H-beta

# Type hints help IDEs provide better autocomplete and catch errors!
```

---

## Advanced Patterns

### Recursion

Functions calling themselves - elegant but use with care:

```{code-cell} ipython3
def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(f"5! = {factorial(5)}")

# More complex: Binary tree traversal for hierarchical structures
def calculate_cluster_mass(cluster):
    """
    Recursively calculate mass of hierarchical structure.
    
    Each cluster can contain stars or sub-clusters.
    """
    if isinstance(cluster, (int, float)):  # Base case: single star
        return cluster
    
    # Recursive case: sum all components
    total_mass = 0
    for component in cluster:
        total_mass += calculate_cluster_mass(component)
    return total_mass

# Hierarchical cluster structure
globular_cluster = [
    1.5,  # Single star
    [0.8, 1.2],  # Binary system
    [0.5, [0.3, 0.4]],  # Triple system
    [[0.9, 1.1], [1.3, 0.7]]  # Two binaries
]

total = calculate_cluster_mass(globular_cluster)
print(f"Total cluster mass: {total:.1f} M☉")
```

```{warning} Recursion Limits
Python has a recursion limit (usually 1000) to prevent stack overflow:
```python
import sys
print(sys.getrecursionlimit())  # Usually 1000
```
For deep recursion, use iteration or increase the limit carefully.
```

### Closures

Functions that "remember" their environment:

```{code-cell} ipython3
def make_doppler_calculator(rest_wavelength):
    """
    Create a Doppler shift calculator for a specific spectral line.
    
    This is a closure - the inner function 'remembers' rest_wavelength.
    """
    def calculate_velocity(observed_wavelength):
        # This function has access to rest_wavelength from outer scope
        z = (observed_wavelength - rest_wavelength) / rest_wavelength
        c = 299792.458  # km/s
        return z * c
    
    # Add some metadata
    calculate_velocity.rest_wavelength = rest_wavelength
    calculate_velocity.__name__ = f"doppler_{rest_wavelength}"
    
    return calculate_velocity

# Create specialized calculators
h_alpha_doppler = make_doppler_calculator(656.28)  # H-alpha line
h_beta_doppler = make_doppler_calculator(486.13)   # H-beta line

# Use them
observed = 658.0
print(f"Observed wavelength: {observed} nm")
print(f"H-alpha velocity: {h_alpha_doppler(observed):.1f} km/s")
print(f"H-beta velocity: {h_beta_doppler(observed):.1f} km/s")
print(f"H-alpha rest λ: {h_alpha_doppler.rest_wavelength} nm")
```

### Partial Functions

Pre-fill some arguments of a function:

```{code-cell} ipython3
from functools import partial

def planck_law(wavelength, temperature, scale=1e-9):
    """
    Planck's law for blackbody radiation.
    
    Parameters
    ----------
    wavelength : float
        Wavelength in meters
    temperature : float
        Temperature in Kelvin
    scale : float
        Scale factor for units
    """
    import math
    h = 6.626e-34  # Planck constant
    c = 2.998e8    # Speed of light
    k = 1.381e-23  # Boltzmann constant
    
    numerator = 2 * h * c**2 / wavelength**5
    denominator = math.exp(h * c / (wavelength * k * temperature)) - 1
    return scale * numerator / denominator

# Create specialized functions for specific temperatures
sun_spectrum = partial(planck_law, temperature=5778)  # Sun
sirius_spectrum = partial(planck_law, temperature=9940)  # Sirius

# Now we can use them easily
wavelength = 500e-9  # 500 nm (green light)
print(f"At {wavelength*1e9:.0f} nm:")
print(f"  Sun: {sun_spectrum(wavelength):.2e} W/m²/m")
print(f"  Sirius: {sirius_spectrum(wavelength):.2e} W/m²/m")
```

---

## Common Pitfalls and Best Practices

```{code-cell} ipython3
# Pitfall 1: Modifying arguments
def bad_append(item, lst=[]):  # DON'T: mutable default
    lst.append(item)
    return lst

def good_append(item, lst=None):  # DO: None default
    if lst is None:
        lst = []
    lst.append(item)
    return lst

# Pitfall 2: Too many parameters
def bad_function(a, b, c, d, e, f, g, h):  # Too many!
    pass

def good_function(config_dict):  # Group related parameters
    pass

# Pitfall 3: Side effects in unexpected places
total = 0
def bad_accumulator(value):
    global total  # Hidden side effect!
    total += value
    return total

def good_accumulator(value, running_total):  # Explicit
    return running_total + value

# Pitfall 4: Functions doing too much
def bad_do_everything(data):
    # Load data
    # Process data
    # Analyze data
    # Plot results
    # Save output
    pass  # Too many responsibilities!

# Better: Single responsibility
def load_data(filename): pass
def process_data(data): pass
def analyze_data(processed): pass
def plot_results(analysis): pass
def save_output(results, filename): pass
```

---

## Exercises

### Exercise 1: Advanced Argument Handling

```{exercise} Flexible Data Processor
Create a function that:
1. Takes required positional arguments for data and method
2. Accepts any number of filters as *args
3. Takes optional keyword arguments for configuration
4. Accepts any additional metadata as **kwargs

The function should:
- Apply all filters to the data
- Process using the specified method
- Return results with metadata attached

Test with astronomical data filtering scenarios.
```

### Exercise 2: Function Factory

```{exercise} Custom Integrator Generator
Write a function that returns customized numerical integrators:
1. Takes integration method ('euler', 'rk4', 'verlet') as input
2. Returns a function configured for that method
3. The returned function should integrate any differential equation

Example usage:
```python
euler_integrator = make_integrator('euler')
result = euler_integrator(dydt, y0, t_span)
```

Test with orbital dynamics equations.
```

### Exercise 3: Decorator Challenge

```{exercise} Performance Monitor Decorator
Create a decorator that:
1. Times function execution
2. Logs input arguments
3. Catches and logs exceptions
4. Can be configured with verbosity level

Apply to various astronomical calculations and analyze performance.
```

### Exercise 4: Recursive Tree Search

```{exercise} Galaxy Cluster Finder
Galaxies form hierarchical structures. Write a recursive function that:
1. Takes a tree structure of galaxy positions
2. Finds all groups within a given distance threshold
3. Returns nested structure of identified clusters

Handle edge cases like empty regions and single galaxies.
```

---

## Key Takeaways

```{admonition} Chapter 3 Summary
:class: success

✅ **Functions are abstractions**: Hide complexity, expose simplicity

✅ **DRY Principle**: Don't Repeat Yourself - factor out common code

✅ **Argument mastery**: Positional, keyword, *args, **kwargs - know when to use each

✅ **Beware mutable defaults**: Use None and create inside function

✅ **LEGB scope rule**: Local → Enclosing → Global → Built-in

✅ **Functions are objects**: Pass them, return them, transform them

✅ **Decorators enhance functions**: Add functionality without modifying code

✅ **Type hints clarify intent**: Make your code self-documenting

✅ **Single responsibility**: Each function should do one thing well
```

```{admonition} Next Chapter Preview
:class: info
Chapter 4: Data Structures & Algorithms - Choosing the right tool for astronomical data
```

## Quick Reference Card

```python
# Function definition
def func(pos, default=None, *args, **kwargs):
    """Docstring here."""
    return result

# Argument unpacking
func(*list_args, **dict_kwargs)

# Lambda functions
lambda x: x**2

# Decorators
@decorator
def func():
    pass

# Type hints
def func(x: int, y: float = 0.0) -> str:
    pass

# Scope modifiers
global var_name
nonlocal var_name

# Function as argument
map(func, iterable)
filter(func, iterable)
sorted(items, key=func)

# Partial functions
from functools import partial
new_func = partial(old_func, arg1=value)

# Common patterns
if param is None:
    param = []  # Mutable default pattern
```