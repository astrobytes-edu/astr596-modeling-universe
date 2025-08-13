# Chapter 1: Python as a Scientific Calculator

*"Before you run, you must walk. Before you compute galaxies, you must compute numbers."*

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [Why Start Here?](#why-start-here)
3. [Basic Operations & Data Types](#basic-operations-data-types)
   - [Numbers in Python](#numbers-in-python)
   - [Basic Math Operations](#basic-math-operations)
   - [Scientific Notation](#scientific-notation)
4. [The Reality of Computer Math](#the-reality-of-computer-math)
   - [Machine Precision](#machine-precision)
   - [Why 0.1 + 0.2 ≠ 0.3](#why-01-02-03)
   - [Implications for Astronomy](#implications-for-astronomy)
5. [Variables & Memory](#variables-memory)
   - [Naming Your Data](#naming-your-data)
   - [Multiple Assignment](#multiple-assignment)
   - [Constants in Science](#constants-in-science)
6. [Strings & Scientific Formatting](#strings-scientific-formatting)
   - [String Basics](#string-basics)
   - [F-strings for Science](#f-strings-for-science)
   - [Formatting Numbers](#formatting-numbers)
7. [Type Conversion & Checking](#type-conversion-checking)
8. [Exercises](#exercises)
9. [Key Takeaways](#key-takeaways)

---

## Learning Objectives

```{admonition} By the end of this chapter you will:
:class: info
- Use Python as a powerful scientific calculator
- Understand fundamental data types (int, float, complex, string)
- Recognize and handle floating-point precision limitations
- Create meaningful variable names following scientific conventions
- Format numerical output for scientific communication
- Convert between data types safely
```

## Why Start Here?

Every simulation, every data analysis, every model starts with basic calculations. Python can replace your scientific calculator, but it's far more powerful—and has some quirks you need to understand.

```{code-cell} ipython3
# Python as calculator - try these!
print(2**10)  # Powers
print(355/113)  # Better approximation of pi than 22/7
print(1.23e-7)  # Scientific notation
```

---

## Basic Operations & Data Types

### Numbers in Python

Python has three numeric types, each with a specific purpose:

```{code-cell} ipython3
# Integers - exact whole numbers
photon_count = 1000000
print(f"Type: {type(photon_count)}, Value: {photon_count}")

# Floats - decimal numbers (approximate!)
redshift = 2.345
print(f"Type: {type(redshift)}, Value: {redshift}")

# Complex - for wave functions, Fourier transforms
wave = 3 + 4j
print(f"Type: {type(wave)}, Magnitude: {abs(wave)}")
```

```{admonition} Why Different Types?
:class: note
- **Integers**: Exact counts, array indices, loop counters
- **Floats**: Measurements, calculations, anything with decimals
- **Complex**: Quantum mechanics, signal processing, Fourier analysis
```

### Basic Math Operations

```{code-cell} ipython3
# Standard operations
a, b = 10, 3

print(f"Addition:       {a} + {b} = {a + b}")
print(f"Subtraction:    {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division:       {a} / {b} = {a / b:.4f}")  # Always returns float
print(f"Integer div:    {a} // {b} = {a // b}")    # Floors the result
print(f"Remainder:      {a} % {b} = {a % b}")      # Modulo
print(f"Power:          {a} ** {b} = {a ** b}")
```

```{exercise} Order of Operations
What does Python calculate for: `2 + 3 * 4 ** 2 / 8 - 1`

Think through it step by step, then verify.
```

```{solution}
:class: dropdown

Python follows PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction):
- First: `4 ** 2 = 16`
- Then: `3 * 16 = 48`
- Then: `48 / 8 = 6.0`
- Then: `2 + 6.0 = 8.0`
- Finally: `8.0 - 1 = 7.0`

```python
result = 2 + 3 * 4 ** 2 / 8 - 1
print(result)  # 7.0
```
```

### Scientific Notation

In astronomy, we deal with enormous and tiny numbers:

```{code-cell} ipython3
# Scientific notation using 'e'
light_speed = 3e8  # m/s
planck_constant = 6.626e-34  # J⋅s
solar_mass = 1.989e30  # kg

# Python preserves precision
distance_to_andromeda = 2.537e6  # light years
print(f"Distance to Andromeda: {distance_to_andromeda:.3e} ly")
print(f"In meters: {distance_to_andromeda * 9.461e15:.3e} m")
```

---

## The Reality of Computer Math

### Machine Precision

```{warning} Critical Concept
Computers cannot store infinite precision. Every calculation is approximate!
```

```{code-cell} ipython3
import sys

# What's the smallest number we can add to 1.0 and see a difference?
epsilon = sys.float_info.epsilon
print(f"Machine epsilon: {epsilon}")
print(f"1.0 + epsilon/2 == 1.0? {1.0 + epsilon/2 == 1.0}")
print(f"1.0 + epsilon == 1.0? {1.0 + epsilon == 1.0}")

# Floats have limits
print(f"\nLargest float: {sys.float_info.max:.3e}")
print(f"Smallest positive float: {sys.float_info.min:.3e}")
```

### Why 0.1 + 0.2 ≠ 0.3

The most famous example of floating-point weirdness:

```{code-cell} ipython3
# The shocking truth
a = 0.1
b = 0.2
c = a + b

print(f"0.1 + 0.2 = {c}")
print(f"0.1 + 0.2 == 0.3? {c == 0.3}")
print(f"Actual value: {c:.20f}")

# Why? Binary representation can't exactly store decimal 0.1
print(f"\nBinary approximation of 0.1: {0.1:.55f}")
```

```{admonition} Why This Happens
:class: important

Computers store numbers in binary (base 2). Just like 1/3 = 0.333... repeats forever in decimal, 
0.1 in binary is 0.0001100110011... repeating forever. The computer truncates this, causing tiny errors.

**Rule**: Never use `==` to compare floats! Use "close enough":
```

```{code-cell} ipython3
# The right way to compare floats
import math

def floats_are_equal(a, b, tolerance=1e-9):
    """Check if two floats are 'close enough'."""
    return abs(a - b) < tolerance

# Or use Python's built-in
print(f"Using custom function: {floats_are_equal(0.1 + 0.2, 0.3)}")
print(f"Using math.isclose: {math.isclose(0.1 + 0.2, 0.3)}")
```

### Implications for Astronomy

These tiny errors matter in astronomical computations:

```{code-cell} ipython3
# Simulating orbital mechanics for 1 year
days_per_year = 365.25
seconds_per_day = 86400
time_step = 0.1  # seconds

# Naive approach - accumulating error
time_naive = 0.0
steps = int(days_per_year * seconds_per_day / time_step)
for _ in range(steps):
    time_naive += time_step

# Better approach - minimize accumulation
time_better = steps * time_step

print(f"Expected time: {days_per_year * seconds_per_day} seconds")
print(f"Naive approach: {time_naive} seconds")
print(f"Better approach: {time_better} seconds")
print(f"Error in naive: {time_naive - days_per_year * seconds_per_day} seconds")
print(f"Error in better: {time_better - days_per_year * seconds_per_day} seconds")
```

```{exercise} Catastrophic Cancellation
When you subtract two nearly equal numbers, you lose precision. Try this:

```python
a = 1.0000001
b = 1.0000000
print(f"Difference: {a - b}")
print(f"Relative error affects {___}% of the result")  # Fill in
```

Why might this be a problem when calculating small changes in stellar positions?
```

```{solution}
:class: dropdown

The difference is 1e-7, but we started with 7 significant figures and ended with 1. 
That's a loss of 6 significant figures - 86% precision loss!

This is critical in astronomy when calculating:
- Proper motions (tiny changes in position)
- Radial velocities (small Doppler shifts)
- Parallax measurements (minuscule angular changes)

Better approach: Reformulate to avoid subtraction of similar numbers when possible.
```

---

## Variables & Memory

### Naming Your Data

Variable names are documentation. Choose wisely:

```{code-cell} ipython3
# Bad - what does this mean?
x = 5.972e24
v = 7.9e3
t = 86400

# Good - self-documenting code
earth_mass_kg = 5.972e24
orbital_velocity_ms = 7.9e3  # m/s
seconds_per_day = 86400

# Python naming conventions (PEP 8)
stellar_temperature = 5778  # lowercase_with_underscores
MAX_ITERATIONS = 1000000   # CONSTANTS_IN_CAPS
StarClass = "G2V"          # CapitalizedWords for classes (later)
```

```{warning} Python Keywords
Never use these as variable names: 
`and, as, assert, break, class, continue, def, del, elif, else, except, False, finally, for, from, global, if, import, in, is, lambda, None, nonlocal, not, or, pass, raise, return, True, try, while, with, yield`
```

### Multiple Assignment

Python allows elegant multiple assignment:

```{code-cell} ipython3
# Simultaneous assignment
ra, dec = 266.405, -28.936  # Galactic center coordinates
print(f"Sgr A* position: RA={ra}°, Dec={dec}°")

# Swapping without temporary variable
a, b = 10, 20
print(f"Before swap: a={a}, b={b}")
a, b = b, a
print(f"After swap: a={a}, b={b}")

# Unpacking from calculations
quotient, remainder = divmod(100, 7)
print(f"100 = 7 * {quotient} + {remainder}")
```

### Constants in Science

Python doesn't have true constants, but we use CAPS by convention:

```{code-cell} ipython3
# Physical constants (SI units)
SPEED_OF_LIGHT = 299792458  # m/s (exact)
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/kg/s²
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s (exact as of 2019)

# Astronomical constants
SOLAR_MASS = 1.98892e30  # kg
PARSEC = 3.0857e16  # m
ASTRONOMICAL_UNIT = 1.495978707e11  # m

# Using constants in calculations
schwarzschild_radius_sun = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
print(f"Schwarzschild radius of Sun: {schwarzschild_radius_sun:.1f} m")
print(f"That's about {schwarzschild_radius_sun/1000:.1f} km")
```

---

## Strings & Scientific Formatting

### String Basics

Strings store text data - essential for labels, filenames, and output:

```{code-cell} ipython3
# Creating strings
object_name = "PSR J0348+0432"  # Single or double quotes
observation_note = 'High-mass pulsar in binary system'

# String concatenation
full_description = object_name + ": " + observation_note
print(full_description)

# String methods
print(f"Uppercase: {object_name.upper()}")
print(f"Is it PSR? {object_name.startswith('PSR')}")
print(f"Replace: {object_name.replace('PSR', 'Pulsar')}")
```

### F-strings for Science

F-strings (formatted string literals) are Python's best formatting tool:

```{code-cell} ipython3
# Basic f-string formatting
star_name = "Betelgeuse"
temperature = 3500  # Kelvin
radius = 887  # Solar radii

print(f"{star_name}: T = {temperature} K, R = {radius} R☉")

# Controlling decimal places
pi = 3.14159265359
print(f"π to 2 decimals: {pi:.2f}")
print(f"π to 5 decimals: {pi:.5f}")

# Scientific notation
avogadro = 6.02214076e23
print(f"Avogadro's number: {avogadro:.3e}")
print(f"Also written as: {avogadro:.3E}")
```

### Formatting Numbers

Advanced formatting for scientific output:

```{code-cell} ipython3
# Alignment and padding
data = [
    ("Sirius A", 1.711, 9940),
    ("Vega", 2.135, 9602),
    ("Arcturus", 1.08, 4286)
]

print("Star Name    Mass (M☉)  Temp (K)")
print("-" * 35)
for name, mass, temp in data:
    print(f"{name:<12} {mass:>9.3f}  {temp:>8d}")

# Percentage formatting
detection_efficiency = 0.8765
print(f"Detection efficiency: {detection_efficiency:.1%}")

# Adding thousand separators
galaxy_count = 2000000000000
print(f"Observable galaxies: {galaxy_count:,}")
print(f"In scientific notation: {galaxy_count:.2e}")
```

```{exercise} Format the Output
Create a nicely formatted table of planetary data:
- Planet name (left-aligned, 10 characters)
- Mass (in Earth masses, 2 decimals, right-aligned)
- Orbital period (in days, 1 decimal)

Data: [("Mercury", 0.055, 87.969), ("Venus", 0.815, 224.701), ("Earth", 1.0, 365.256)]
```

```{solution}
:class: dropdown

```python
planets = [
    ("Mercury", 0.055, 87.969),
    ("Venus", 0.815, 224.701),
    ("Earth", 1.0, 365.256)
]

print(f"{'Planet':<10} {'Mass (M⊕)':>10} {'Period (days)':>13}")
print("-" * 35)
for name, mass, period in planets:
    print(f"{name:<10} {mass:>10.2f} {period:>13.1f}")
```

Output:
```
Planet      Mass (M⊕)  Period (days)
-----------------------------------
Mercury          0.06          88.0
Venus            0.82         224.7
Earth            1.00         365.3
```
```

---

## Type Conversion & Checking

Converting between types is common when processing data:

```{code-cell} ipython3
# Type checking
value = 42
print(f"Is {value} an integer? {isinstance(value, int)}")
print(f"Is {value} a float? {isinstance(value, float)}")
print(f"Is {value} a number? {isinstance(value, (int, float))}")

# Explicit conversion
user_input = "123.45"  # Simulating input
number = float(user_input)
print(f"String '{user_input}' converted to float: {number}")

# Integer conversion truncates!
pi = 3.14159
print(f"int({pi}) = {int(pi)}")  # Doesn't round!

# Rounding properly
print(f"round({pi}) = {round(pi)}")
print(f"round({pi}, 2) = {round(pi, 2)}")
```

```{warning} Common Conversion Pitfalls
- `int()` truncates, doesn't round - use `round()` first if needed
- `float('inf')` and `float('nan')` are valid!
- Converting to int can overflow in other languages, but Python handles arbitrary precision
```

```{code-cell} ipython3
# Special float values
import math

infinity = float('inf')
not_a_number = float('nan')

print(f"Infinity > 1e308? {infinity > 1e308}")
print(f"NaN == NaN? {not_a_number == not_a_number}")  # Always False!
print(f"Check for NaN: {math.isnan(not_a_number)}")
print(f"Check for infinity: {math.isinf(infinity)}")
```

---

## Exercises

### Exercise 1: Stellar Magnitude Calculator

```{exercise} Magnitude and Flux
The magnitude system in astronomy uses a logarithmic scale. The relationship between two stars' magnitudes and fluxes is:

$$m_1 - m_2 = -2.5 \log_{10}(F_1/F_2)$$

1. Calculate the flux ratio between a magnitude 1 star and a magnitude 6 star
2. If Star A is 2.5 magnitudes brighter than Star B, what's their flux ratio?
3. Why do we use this "backwards" system where brighter objects have smaller numbers?
```

### Exercise 2: Floating Point Detective

```{exercise} Finding the Limits
Write code to determine:
1. The smallest positive float Python can represent
2. The largest integer that can be stored exactly in a float
3. What happens when you exceed these limits?

Hint: Try powers of 2, and remember that floats use 53 bits for the mantissa.
```

### Exercise 3: Scientific Constants Library

```{exercise} Build Your Constants Module
Create variables for these astronomical constants with appropriate names:
- Speed of light (c)
- Gravitational constant (G)  
- Solar luminosity (L☉)
- Earth mass (M⊕)
- Hubble constant (H₀)

Then calculate:
1. The Schwarzschild radius of Earth
2. The critical density of the universe
3. How many Earth masses equal one Solar mass
```

---

## Key Takeaways

```{admonition} Chapter 1 Summary
:class: success

✅ **Python as Calculator**: Python handles basic math, scientific notation, and complex numbers naturally

✅ **Floating-Point Reality**: All calculations have limited precision (~15-17 decimal digits)

✅ **Never Compare Floats with ==**: Use `math.isclose()` or check if difference < tolerance

✅ **Variable Names Matter**: `stellar_temperature_kelvin` beats `temp` every time

✅ **F-strings for Formatting**: `f"{value:.3e}"` gives you control over scientific output

✅ **Type Conversion**: Be explicit and careful, especially with user input

✅ **Constants Convention**: USE_CAPS for values that shouldn't change
```

```{admonition} Next Chapter Preview
:class: info
Chapter 2: Control Flow & Logic - Teaching your computer to make decisions based on data
```

## Quick Reference Card

```python
# Numbers
integer = 42
floating = 3.14159
scientific = 6.626e-34
complex_num = 3 + 4j

# Operations
x ** y  # Power
x // y  # Integer division
x % y   # Remainder/modulo

# Formatting
f"{value:.3f}"   # 3 decimal places
f"{value:.2e}"   # Scientific notation
f"{value:>10}"   # Right-align, width 10
f"{value:,}"     # Thousands separator

# Type Conversion
int(3.14)        # → 3 (truncates!)
float("3.14")    # → 3.14
round(3.14159, 2)  # → 3.14

# Floating-point comparison
import math
math.isclose(a, b, rel_tol=1e-9)
```