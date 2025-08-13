# Chapter 2: Control Flow & Logic - Teaching Computers to Think

*"Logic is the beginning of wisdom, not the end." - Spock*

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [Why Logic Matters](#why-logic-matters)
   - [From Philosophy to Circuits](#from-philosophy-to-circuits)
   - [Boolean Algebra: The Mathematics of Decision](#boolean-algebra-the-mathematics-of-decision)
3. [Truth and Conditions](#truth-and-conditions)
   - [Comparison Operators](#comparison-operators)
   - [Boolean Values and Truthiness](#boolean-values-and-truthiness)
   - [Logical Operators](#logical-operators)
4. [If-Then Logic: Making Decisions](#if-then-logic-making-decisions)
   - [Basic If Statements](#basic-if-statements)
   - [If-Elif-Else Chains](#if-elif-else-chains)
   - [Guard Clauses vs Nested Ifs](#guard-clauses-vs-nested-ifs)
5. [Loops: Repetition with Purpose](#loops-repetition-with-purpose)
   - [For Loops - When You Know How Many](#for-loops-when-you-know-how-many)
   - [While Loops - Until a Condition](#while-loops-until-a-condition)
   - [Loop Control: Break, Continue, Else](#loop-control-break-continue-else)
6. [Comprehensions: Elegant Iteration](#comprehensions-elegant-iteration)
   - [List Comprehensions](#list-comprehensions)
   - [When to Use Comprehensions](#when-to-use-comprehensions)
7. [Common Patterns in Scientific Computing](#common-patterns-in-scientific-computing)
8. [Exercises](#exercises)
9. [Key Takeaways](#key-takeaways)

---

## Learning Objectives

```{admonition} By the end of this chapter you will:
:class: info
- Understand Boolean logic and its connection to mathematics and philosophy
- Write conditional statements that make intelligent decisions
- Use loops to automate repetitive calculations
- Recognize when to use different loop patterns
- Apply logical thinking to solve astronomical problems
- Debug logical errors in program flow
```

## Why Logic Matters

### From Philosophy to Circuits

Logic isn't just programming—it's the foundation of rational thought, mathematics, and computation itself.

```{code-cell} ipython3
# Logic has a rich history
print("Aristotle (384 BCE): Syllogistic logic - All stars are suns, Proxima is a star, therefore...")
print("Boole (1854): Boolean algebra - True/False as 1/0")
print("Shannon (1937): Logic gates in circuits")
print("Today: Every if-statement in your code")

# It all reduces to True and False
print(f"\nIn Python: True = {int(True)}, False = {int(False)}")
print(f"This is why: True + True + False = {True + True + False}")
```

```{admonition} The Philosophical Connection
:class: note
**Logical reasoning forms the basis of the scientific method:**
- **Deduction**: If all stars fuse hydrogen (premise) and the Sun is a star (premise), then the Sun fuses hydrogen (conclusion)
- **Induction**: We've observed 1000 pulsars rotating rapidly, therefore all pulsars probably rotate rapidly
- **Abduction**: The CMB has these properties, the best explanation is the Big Bang

Your code embodies logical reasoning!
```

### Boolean Algebra: The Mathematics of Decision

Before computers, George Boole showed that logic follows mathematical rules:

```{code-cell} ipython3
# Boolean algebra in action
A = True
B = False

# Basic operations (same as logic gates in circuits!)
print(f"NOT A = {not A}")
print(f"A AND B = {A and B}")
print(f"A OR B = {A or B}")
print(f"A XOR B = {A != B}")  # Exclusive OR

# De Morgan's Laws - fundamental to logic
print("\nDe Morgan's Laws:")
print(f"not (A and B) = (not A) or (not B): {not (A and B) == (not A) or (not B)}")
print(f"not (A or B) = (not A) and (not B): {not (A or B) == (not A) and (not B)}")
```

---

## Truth and Conditions

### Comparison Operators

Every decision starts with a comparison:

```{code-cell} ipython3
# Astronomical example: Is this a habitable zone planet?
star_luminosity = 0.5  # Solar units
planet_distance = 0.7  # AU
inner_habitable = 0.95 * (star_luminosity ** 0.5)
outer_habitable = 1.37 * (star_luminosity ** 0.5)

print(f"Star luminosity: {star_luminosity} L☉")
print(f"Planet distance: {planet_distance} AU")
print(f"Habitable zone: {inner_habitable:.2f} - {outer_habitable:.2f} AU")
print()

# All comparison operators
print(f"Distance > inner edge? {planet_distance > inner_habitable}")
print(f"Distance < outer edge? {planet_distance < outer_habitable}")
print(f"Distance >= inner? {planet_distance >= inner_habitable}")
print(f"Distance <= outer? {planet_distance <= outer_habitable}")
print(f"Exactly at inner edge? {planet_distance == inner_habitable}")
print(f"Not at inner edge? {planet_distance != inner_habitable}")
```

```{warning} Floating Point Comparisons (Again!)
Remember from Chapter 1: Never use `==` with floats!
```python
# WRONG
if orbit_period == 365.25:
    
# RIGHT
if abs(orbit_period - 365.25) < 0.01:
```
```

### Boolean Values and Truthiness

Python has a broader concept of "truth" than just True/False:

```{code-cell} ipython3
# What's considered True or False?
values_to_test = [
    True, False,           # Booleans
    1, 0, -1,             # Numbers
    "", "hello",          # Strings
    [], [1, 2, 3],        # Lists
    None,                 # Special null value
    0.0, 0.000001,        # Floats
]

print("Value".ljust(15), "Bool".ljust(8), "Type")
print("-" * 35)
for value in values_to_test:
    print(f"{str(value):15} {bool(value)!s:8} {type(value).__name__}")
```

```{admonition} The Truthiness Rule
:class: important
**Falsy values**: `False`, `0`, `0.0`, `""`, `[]`, `{}`, `None`
**Everything else is Truthy!**

This enables elegant code:
```python
# Instead of: if len(observations) > 0:
if observations:  # Empty list is False!
    process(observations)
```
```

### Logical Operators

Combine conditions to express complex logic:

```{code-cell} ipython3
# Stellar classification logic
temperature = 5800  # Kelvin
luminosity = 1.0    # Solar units
mass = 1.0          # Solar masses

# Complex conditions
is_main_sequence = 0.08 < mass < 150  # Stars have mass limits
is_sun_like = 5300 < temperature < 6000 and 0.8 < luminosity < 1.2
is_giant = luminosity > 10 and temperature < 5000
is_white_dwarf = luminosity < 0.01 and temperature > 8000

print(f"Temperature: {temperature}K, Luminosity: {luminosity}L☉, Mass: {mass}M☉")
print(f"Main sequence? {is_main_sequence}")
print(f"Sun-like? {is_sun_like}")
print(f"Giant? {is_giant}")
print(f"White dwarf? {is_white_dwarf}")

# Operator precedence (like math!)
# not > and > or
result = True or False and False  # What's this?
print(f"\nTrue or False and False = {result}")
print("Because 'and' binds tighter than 'or': True or (False and False)")
```

```{exercise} Short-Circuit Evaluation
Python stops evaluating as soon as it knows the answer. Why do these matter?

```python
# This is safe even if divisor is 0
if divisor != 0 and value/divisor > 10:
    print("Large ratio")

# This would crash!
if value/divisor > 10 and divisor != 0:
    print("Large ratio")
```

When might this be useful in astronomy code?
```

```{solution}
:class: dropdown

Short-circuit evaluation is crucial for:

1. **Avoiding division by zero**:
```python
if parallax != 0 and 1/parallax < 100:  # Safe!
    print("Nearby star")
```

2. **Checking existence before access**:
```python
if spectrum is not None and spectrum.max() > threshold:
    print("Bright source")
```

3. **Performance** - expensive operations last:
```python
if quick_check() and expensive_calculation():
    process()
```
```

---

## If-Then Logic: Making Decisions

### Basic If Statements

The fundamental decision structure:

```{code-cell} ipython3
def classify_star(temperature):
    """Classify star by temperature using Harvard spectral classification."""
    
    spectral_class = "Unknown"
    
    if temperature > 30000:
        spectral_class = "O"  # Blue
    elif temperature > 10000:
        spectral_class = "B"  # Blue-white
    elif temperature > 7500:
        spectral_class = "A"  # White
    elif temperature > 6000:
        spectral_class = "F"  # Yellow-white
    elif temperature > 5200:
        spectral_class = "G"  # Yellow (Sun)
    elif temperature > 3700:
        spectral_class = "K"  # Orange
    elif temperature > 2400:
        spectral_class = "M"  # Red
    else:
        spectral_class = "L/T/Y"  # Brown dwarfs
    
    return spectral_class

# Test the classifier
test_temps = [40000, 9700, 5778, 3500, 1000]
for temp in test_temps:
    print(f"{temp:5}K -> Class {classify_star(temp)}")
```

### If-Elif-Else Chains

Order matters in elif chains!

```{code-cell} ipython3
def determine_evolution_stage(mass, luminosity, temperature):
    """Determine stellar evolution stage from observable parameters."""
    
    # Check in order of likelihood/importance
    if luminosity < 0.01 and temperature > 10000:
        return "White Dwarf"
    elif luminosity > 1000:
        return "Supergiant"
    elif luminosity > 100 and temperature < 4500:
        return "Red Giant"
    elif luminosity > 100:
        return "Blue Giant"
    elif 0.08 < mass < 0.5 and luminosity < 0.08:
        return "Red Dwarf"
    elif abs(luminosity - mass**3.5) < 0.5 * mass**3.5:  # Within 50% of main sequence
        return "Main Sequence"
    else:
        return "Peculiar/Variable"

# Test cases
stars = [
    (1.0, 1.0, 5778),      # Sun
    (0.1, 0.001, 3000),    # Red dwarf
    (10, 10000, 20000),    # Supergiant
    (0.6, 0.0001, 15000),  # White dwarf
]

for mass, lum, temp in stars:
    stage = determine_evolution_stage(mass, lum, temp)
    print(f"M={mass:4.1f}M☉, L={lum:7.4f}L☉, T={temp:5}K -> {stage}")
```

### Guard Clauses vs Nested Ifs

Write cleaner code with guard clauses:

```{code-cell} ipython3
# ❌ Nested approach - hard to follow
def process_observation_nested(data):
    if data is not None:
        if len(data) > 0:
            if data.max() > 0:
                if data.min() < 1000:
                    # Finally do the work!
                    return data.mean()
                else:
                    return "Signal too strong"
            else:
                return "No positive values"
        else:
            return "Empty dataset"
    else:
        return "No data"

# ✅ Guard clause approach - much cleaner!
def process_observation_clean(data):
    # Handle error cases first and exit early
    if data is None:
        return "No data"
    if len(data) == 0:
        return "Empty dataset"
    if data.max() <= 0:
        return "No positive values"
    if data.min() >= 1000:
        return "Signal too strong"
    
    # Main logic is unindented and clear
    return data.mean()

# Both give same results, but one is much more readable!
```

```{admonition} Code Philosophy: Fail Fast
:class: tip
**Guard clauses** embody the "fail fast" principle:
1. Check for error conditions first
2. Return/exit immediately if something's wrong
3. Keep the "happy path" unindented and clear

This matches how we think: "If this is wrong, stop. If that's wrong, stop. Otherwise, proceed."
```

---

## Loops: Repetition with Purpose

### For Loops - When You Know How Many

For loops are perfect when you know the iterations in advance:

```{code-cell} ipython3
# Classic for loop with range
print("Counting photons in bins:")
for bin_number in range(5):
    photon_count = 100 + bin_number * 50  # Simulated data
    print(f"Bin {bin_number}: {photon_count} photons")

print("\n" + "="*40 + "\n")

# Iterating over data directly
wavelengths = [656.3, 486.1, 434.0, 410.2]  # Hydrogen Balmer series
print("Balmer series wavelengths (nm):")
for wavelength in wavelengths:
    print(f"  λ = {wavelength} nm")

print("\n" + "="*40 + "\n")

# Enumerate when you need index AND value
elements = ["Hydrogen", "Helium", "Carbon", "Oxygen"]
print("Cosmic abundances (by number):")
abundances = [0.92, 0.078, 0.0003, 0.0005]
for i, element in enumerate(elements):
    print(f"  {i+1}. {element}: {abundances[i]:.4%}")
```

```{code-cell} ipython3
# Range variations
print("range(5):", list(range(5)))            # 0 to 4
print("range(2, 8):", list(range(2, 8)))      # 2 to 7
print("range(0, 10, 2):", list(range(0, 10, 2)))  # Even numbers
print("range(10, 0, -1):", list(range(10, 0, -1)))  # Countdown

# Practical example: Observing schedule
print("\nObservation schedule (hours from midnight):")
for hour in range(20, 28, 2):  # 8pm to 4am, every 2 hours
    actual_hour = hour if hour < 24 else hour - 24
    am_pm = "PM" if 12 <= hour < 24 else "AM"
    display_hour = actual_hour if actual_hour <= 12 else actual_hour - 12
    if display_hour == 0:
        display_hour = 12
    print(f"  Observation at {display_hour:2d}:00 {am_pm}")
```

### While Loops - Until a Condition

While loops continue until a condition becomes false:

```{code-cell} ipython3
# Newton's method for finding square roots
def sqrt_newton(n, tolerance=1e-10):
    """Calculate square root using Newton's method."""
    if n < 0:
        return None
    
    guess = n / 2  # Initial guess
    iterations = 0
    
    while abs(guess * guess - n) > tolerance:
        guess = (guess + n/guess) / 2  # Newton's formula
        iterations += 1
        
        # Safety check
        if iterations > 100:
            print("Warning: Max iterations reached")
            break
    
    return guess, iterations

# Test it
import math
number = 2.0
my_sqrt, iters = sqrt_newton(number)
print(f"Newton's sqrt({number}): {my_sqrt} after {iters} iterations")
print(f"Python's sqrt({number}): {math.sqrt(number)}")
print(f"Difference: {abs(my_sqrt - math.sqrt(number)):.2e}")
```

```{warning} Infinite Loop Danger!
Always ensure your while loop condition will eventually become False!

```python
# DANGER: This runs forever!
while True:
    print("Help, I'm stuck!")
    
# SAFE: Always have an exit strategy
max_iterations = 1000
count = 0
while condition and count < max_iterations:
    # do work
    count += 1
```
```

### Loop Control: Break, Continue, Else

Fine-tune loop behavior:

```{code-cell} ipython3
# Break: Exit loop early
def find_first_giant_planet(planets):
    """Find first planet with mass > 50 Earth masses."""
    for i, planet in enumerate(planets):
        if planet['mass'] > 50:
            print(f"Found giant planet at index {i}: {planet['name']}")
            break
    else:  # This runs if loop completes without break!
        print("No giant planets found")

planets = [
    {'name': 'Kepler-452b', 'mass': 5},
    {'name': 'HD 209458 b', 'mass': 220},
    {'name': 'Proxima b', 'mass': 1.3}
]

find_first_giant_planet(planets)

print("\n" + "="*40 + "\n")

# Continue: Skip to next iteration
print("Processing observations (skipping bad data):")
observations = [100, -5, 200, 0, 150, -999, 300]

for obs in observations:
    if obs <= 0:  # Bad data
        print(f"  Skipping invalid value: {obs}")
        continue
    
    # Process valid data
    magnitude = -2.5 * math.log10(obs/100)
    print(f"  Flux: {obs:3d} -> Magnitude: {magnitude:+5.2f}")
```

```{admonition} The Mysterious Loop-Else
:class: note
Python's `else` clause on loops is unique:
- Executes if loop completes normally (no `break`)
- Useful for search patterns: "Find X, else report not found"
- Works with both `for` and `while` loops
```

---

## Comprehensions: Elegant Iteration

### List Comprehensions

Transform loops into concise expressions:

```{code-cell} ipython3
# Traditional loop approach
magnitudes = [5.2, 3.1, 6.8, 4.5, 2.3]
fluxes_loop = []
for mag in magnitudes:
    flux = 10**(-0.4 * mag)
    fluxes_loop.append(flux)

# List comprehension - same result, one line!
fluxes_comp = [10**(-0.4 * mag) for mag in magnitudes]

print("Traditional loop result:", len(fluxes_loop), "values")
print("Comprehension result:", len(fluxes_comp), "values")
print(f"Results identical? {fluxes_loop == fluxes_comp}")

print("\n" + "="*40 + "\n")

# Comprehensions with conditions
all_stars = [
    {'name': 'Sirius', 'mag': -1.46, 'type': 'A'},
    {'name': 'Betelgeuse', 'mag': 0.42, 'type': 'M'},
    {'name': 'Rigel', 'mag': 0.13, 'type': 'B'},
    {'name': 'Aldebaran', 'mag': 0.85, 'type': 'K'},
    {'name': 'Vega', 'mag': 0.03, 'type': 'A'},
]

# Get bright stars (mag < 0.5)
bright_stars = [star['name'] for star in all_stars if star['mag'] < 0.5]
print(f"Bright stars: {bright_stars}")

# Get blue stars with their magnitudes
blue_stars = [(s['name'], s['mag']) for s in all_stars if s['type'] in ['O', 'B', 'A']]
print(f"Blue stars: {blue_stars}")
```

### When to Use Comprehensions

```{code-cell} ipython3
# ✅ GOOD: Simple transformation
# Clear and concise
squares = [x**2 for x in range(10)]

# ✅ GOOD: Filtering with simple condition
positives = [x for x in data if x > 0]

# ❌ BAD: Too complex - use regular loop
# Hard to read and debug
# result = [process(x) if complex_condition(x) and other_check(x) 
#          else alternate_process(x) for x in data if pre_filter(x)]

# ❌ BAD: Side effects - comprehensions shouldn't print or modify external state
# Don't do this:
# [print(x) for x in range(10)]  # Use regular loop instead

# ✅ GOOD: Nested comprehensions for matrices
# Create a 3x3 identity matrix
identity = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
print("\nIdentity matrix:")
for row in identity:
    print(row)
```

```{admonition} Comprehension Philosophy
:class: tip
**Use comprehensions when:**
- The operation is simple and clear
- You're building a new list from an existing iterable
- The logic fits comfortably on 1-2 lines

**Use regular loops when:**
- Logic is complex or multi-step
- You need to break or handle errors
- You're not building a list (just processing)
```

---

## Common Patterns in Scientific Computing

### Pattern 1: Accumulator

```{code-cell} ipython3
# Sum pattern
def calculate_total_luminosity(star_luminosities):
    """Sum luminosities of star cluster."""
    total = 0  # Initialize accumulator
    for luminosity in star_luminosities:
        total += luminosity  # Accumulate
    return total

# Product pattern  
def calculate_probability_all_detect(individual_probs):
    """Probability that ALL telescopes detect the source."""
    combined = 1  # Initialize for product
    for prob in individual_probs:
        combined *= prob  # Accumulate via multiplication
    return combined

cluster = [1.0, 0.5, 0.3, 2.1, 0.8]  # Solar luminosities
print(f"Total cluster luminosity: {calculate_total_luminosity(cluster):.1f} L☉")

detection_probs = [0.9, 0.85, 0.95]
print(f"Combined detection probability: {calculate_probability_all_detect(detection_probs):.3f}")
```

### Pattern 2: Search/Filter

```{code-cell} ipython3
def find_habitable_planets(planets):
    """Find all planets in the habitable zone."""
    habitable = []
    
    for planet in planets:
        # Calculate habitable zone for planet's star
        inner = 0.95 * math.sqrt(planet['star_luminosity'])
        outer = 1.37 * math.sqrt(planet['star_luminosity'])
        
        # Check if planet is in zone
        if inner <= planet['distance'] <= outer:
            habitable.append(planet)
    
    return habitable

# Example exoplanet data
exoplanets = [
    {'name': 'Proxima b', 'distance': 0.05, 'star_luminosity': 0.0017},
    {'name': 'Kepler-452b', 'distance': 1.05, 'star_luminosity': 1.2},
    {'name': 'TRAPPIST-1e', 'distance': 0.029, 'star_luminosity': 0.000524},
]

habitable = find_habitable_planets(exoplanets)
print(f"Potentially habitable: {[p['name'] for p in habitable]}")
```

### Pattern 3: Convergence

```{code-cell} ipython3
def calculate_pi_leibniz(tolerance=1e-6):
    """Calculate π using Leibniz formula until convergence."""
    # π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
    
    pi_estimate = 0
    term_number = 0
    
    while True:
        term = (-1)**term_number / (2*term_number + 1)
        pi_estimate += term
        
        # Check convergence
        if abs(term) < tolerance:
            break
            
        term_number += 1
    
    return 4 * pi_estimate, term_number

pi_approx, iterations = calculate_pi_leibniz()
print(f"π ≈ {pi_approx:.8f} after {iterations} iterations")
print(f"Error: {abs(pi_approx - math.pi):.2e}")
```

---

## Exercises

### Exercise 1: Logical Thinking

```{exercise} Stellar Classification Logic
Write a function that takes temperature, luminosity, and mass as inputs and returns:
- "Impossible" if the star violates basic physics (e.g., luminosity > 10^6 * mass^3.5)
- "White Dwarf" if high temp, low luminosity
- "Main Sequence" if it follows L ∝ M^3.5 approximately
- "Giant" if luminosity is much higher than expected for its mass
- "Not classified" otherwise

Test with:
- Sun: T=5778K, L=1.0, M=1.0
- Sirius B: T=25000K, L=0.026, M=1.0
- Betelgeuse: T=3500K, L=100000, M=20
```

### Exercise 2: Prime Number Sieve

```{exercise} Sieve of Eratosthenes
Implement the ancient algorithm for finding prime numbers:
1. Create a list of numbers from 2 to n
2. Start with the first number (2)
3. Mark all its multiples as composite
4. Move to the next unmarked number
5. Repeat until you've processed all numbers

Find all primes less than 100. How many are there?

Bonus: Why might astronomers care about prime numbers? (Hint: think about periodic signals and aliases)
```

### Exercise 3: Monte Carlo Integration

```{exercise} Escape Velocity Distribution
A globular cluster has stars with random velocities. Use a loop to:
1. Generate 1000 random velocities (Gaussian distribution, mean=10 km/s, std=3 km/s)
2. Count how many exceed the cluster's escape velocity (15 km/s)
3. Calculate what fraction will escape
4. Use a comprehension to get the list of escaping velocities

How does this relate to cluster evaporation over time?
```

### Exercise 4: Convergence Testing

```{exercise} Iterative Orbit Calculation
When calculating orbits, we often need to solve Kepler's equation iteratively:
E - e*sin(E) = M

Where E is eccentric anomaly, e is eccentricity, M is mean anomaly.

Write a function that:
1. Uses a while loop to solve for E given M and e
2. Stops when the change in E is less than 1e-10
3. Limits iterations to prevent infinite loops
4. Returns both E and the number of iterations

Test with e=0.5, M=π/4. How many iterations does it take?
```

---

## Key Takeaways

```{admonition} Chapter 2 Summary
:class: success

✅ **Logic is Universal**: From Aristotle to CPUs, logic underlies all reasoning

✅ **Boolean Algebra**: True/False operations follow mathematical rules (De Morgan's Laws)

✅ **Truthiness**: Empty containers and zeros are False; most everything else is True

✅ **Guard Clauses**: Check error conditions first, keep happy path unindented

✅ **For vs While**: Use `for` when iterations are known, `while` for conditions

✅ **Break/Continue/Else**: Control loop flow precisely

✅ **Comprehensions**: Elegant one-liners for simple transformations

✅ **Patterns**: Accumulator, Search/Filter, and Convergence appear everywhere
```

```{admonition} Next Chapter Preview
:class: info
Chapter 3: Functions - Your First Abstraction. Learn to write reusable, testable code that does one thing well.
```

## Quick Reference Card

```python
# Comparisons
<, >, <=, >=, ==, !=
math.isclose(a, b)  # For floats!

# Logical operators (in order of precedence)
not x
x and y  # Short-circuits
x or y   # Short-circuits

# If statements
if condition:
    pass
elif other_condition:
    pass
else:
    pass

# Guard clause pattern
if error_condition:
    return early
# Happy path here

# For loops
for i in range(n):  # 0 to n-1
for i, val in enumerate(list):  # Index and value
for item in collection:  # Direct iteration

# While loops
while condition:
    # work
    if done:
        break
    if skip:
        continue
else:
    # Runs if no break

# Comprehensions
[expr for item in iterable if condition]

# Common patterns
total = 0
for x in data:
    total += x  # Accumulator
```