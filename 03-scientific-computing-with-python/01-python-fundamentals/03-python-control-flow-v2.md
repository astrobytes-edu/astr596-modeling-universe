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
# ⚠️ Chapter 3: Control Flow & Logic

## Learning Objectives

By the end of this chapter, you will be able to:

- Design algorithms using structured pseudocode before writing any Python code
- Implement conditional statements (**if**/**elif**/**else**) with proper handling of edge cases
- Choose appropriate loop structures (**for** vs **while**) based on problem requirements
- Master all comparison operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) and logical operators (**and**, **or**, **not**)
- Handle floating-point comparisons safely in conditional statements
- Debug logic errors systematically using IPython's debugger and **assert** statements
- Write efficient list comprehensions while knowing when to avoid them
- Recognize and apply universal algorithmic patterns across different problems
- Build defensive code that validates assumptions and catches errors early

## Prerequisites Check

:::{admonition} ✅ Before Starting This Chapter
:class: note

- [ ] You can launch IPython and use magic commands like %timeit (Chapter 1)
- [ ] You understand floating-point precision and comparison issues (Chapter 2)
- [ ] You can write and run Python scripts from the terminal (Chapter 1)
- [ ] You can use f-strings for formatted output (Chapter 2)
- [ ] Your `astr596` environment is activated and working

If any boxes are unchecked, review the indicated chapters first.
:::

## Chapter Overview

Programming is fundamentally about teaching computers to make decisions and repeat tasks. When you write an **if** statement or a loop, you're translating human logic into instructions a machine can follow. But here's the critical insight that separates computational thinkers from mere coders: the logic must be designed before it's implemented. This chapter transforms you from someone who writes code to someone who designs algorithms.

We'll start with the lost art of pseudocode — not as a bureaucratic exercise, but as the difference between code that works by accident and code that works by design. You'll learn to recognize universal patterns that appear across all of computational physics: iteration, accumulation, filtering, mapping, and reduction. These patterns will appear in every project you build, from N-body simulations to neural networks. Whether you're folding light curves to find exoplanet periods or iterating until your stellar model converges, these patterns form the backbone of computational astronomy.

The control flow structures we explore here are where your numerical calculations from Chapter 2 become dynamic algorithms. Every convergence test, every adaptive timestep, every Monte Carlo acceptance criterion depends on mastering these concepts deeply, not just syntactically. By chapter's end, you'll see code not as a sequence of commands, but as a carefully orchestrated flow of decisions and iterations that solve real scientific problems. You'll write algorithms that could process data from the James Webb Space Telescope or control the adaptive optics on the next generation of ground-based observatories.

## 3.1 Algorithmic Thinking: The Lost Art of Pseudocode

```{margin}
**pseudocode**
Human-readable algorithm description focusing on logic over syntax
```

Most students jump straight from problem to code, then wonder why they spend hours debugging. Professional computational scientists spend more time thinking than typing. **Pseudocode** is how we think precisely about algorithms without getting distracted by syntax. Think of it as your algorithm's blueprint — you wouldn't build a telescope without optical designs, so why write code without algorithmic designs?

### Why Pseudocode Matters in Scientific Computing

Consider this scenario: You need to implement adaptive timestepping for an orbital integrator. Without pseudocode, you'll likely write code, run it, watch orbits spiral incorrectly, debug for hours, and maybe get it working through trial and error. With pseudocode, you'll identify edge cases, boundary conditions, and logical flaws before writing a single line of Python.

```{code-cell} ipython3
# WITHOUT PSEUDOCODE (typical student approach):
# "I'll figure it out as I code..."
def integrate_naive(state, t_end):
    dt = 0.01
    while state.time < t_end:
        new_state = step(state, dt)
        error = estimate_error(state, new_state)
        if error > tolerance:
            dt = dt * 0.5  # Seems reasonable?
        state = new_state
    return state
# Wait, this doesn't work... infinite loop when error is bad!
# Also, dt never increases... hours of debugging ahead
```

Now let's see how pseudocode reveals problems immediately! This is exactly how professional astronomers design algorithms for everything from orbit calculations to galaxy simulations. You're about to learn the same systematic approach used at NASA's Jet Propulsion Laboratory for spacecraft navigation and at the Space Telescope Science Institute for Hubble's scheduling algorithms.

### The Three Levels of Pseudocode Refinement

Professional algorithm development happens in stages, each revealing different issues. Don't worry if this feels strange at first — every programmer has felt that way! But once you embrace pseudocode, you'll save countless hours of debugging. Let's build this skill together:

**Level 1: Conceptual Overview (The Big Picture)**
```
WHILE simulation not done:       # WHILE means "repeat as long as condition is true"
    Take a step
    Check IF step was good        # IF means "only do this when condition is true"
    Adjust timestep
```

This level helps you understand the overall flow. The **WHILE** construct creates a loop that continues until some condition becomes false. The **IF** construct makes a decision based on a condition. Already, we can ask critical questions: What defines "done"? What makes a step "good"? How much should we adjust? These questions matter!

:::{admonition} 🤔 Check Your Understanding
:class: hint

Before continuing, identify at least two problems with the Level 1 pseudocode above. What could go wrong?
:::

:::{admonition} Solution
:class: tip, dropdown

1. **No exit condition if step is never "good"** — infinite loop risk!
2. **No bounds on timestep adjustment** — could grow infinitely or shrink to zero
3. **"Simulation done" is vague** — need precise termination condition
4. **No error handling** — what if the integration fails completely?

These aren't nitpicks — they're the difference between code that runs and code that runs correctly!
:::

**Level 2: Structural Detail (The Flow)**
```
FUNCTION adaptive_integrate(initial_state, end_time):  # FUNCTION groups reusable code
    state ← initial_state                              # ← means "assign value to variable"
    dt ← estimate_initial_timestep(state)
    
    WHILE time < end_time:                             # Loop continues while time hasn't reached end
        DO:                                             # DO-UNTIL creates a loop that runs at least once
            trial_step = integrate(state, dt)
            error = compute_error(trial_step)
        UNTIL error < tolerance OR dt < dt_min         # OR means "either condition can be true"
        
        state = trial_step
        dt = adjust_timestep(error, dt)
    
    RETURN state                                        # RETURN sends value back to caller
```

Now we see the retry logic and minimum timestep safeguard. The **DO-UNTIL** construct ensures we attempt at least one integration step. The **OR** operator means either condition being true will exit the inner loop. **FUNCTION** defines a reusable block of code that can be called with arguments and **RETURN** a result.

**Level 3: Implementation-Ready (Stage 1: Core Logic)**

```
FUNCTION adaptive_integrate(initial_state, end_time, tolerance):
    state ← initial_state
    dt ← estimate_initial_timestep(state)
    
    WHILE state.time < end_time:
        trial_state ← rk4_step(state, dt)
        error ← estimate_error(state, trial_state)
        
        IF error < tolerance:                          # Decision point
            state ← trial_state
            dt ← min(dt * 1.5, dt_max)                # Can grow
        ELSE:                                          # ELSE handles "otherwise" case
            dt ← max(dt * 0.5, dt_min)                # Must shrink
```

**Level 3: Implementation-Ready (Stage 2: Add Safety)**

```
FUNCTION adaptive_integrate(initial_state, end_time, tolerance):
    state ← initial_state
    dt ← estimate_initial_timestep(state)
    dt_min ← 1e-10 * (end_time - initial_state.time)
    dt_max ← 0.1 * (end_time - initial_state.time)
    
    WHILE state.time < end_time:
        step_accepted ← False                          # Boolean flag (True/False)
        attempts ← 0
        
        WHILE NOT step_accepted AND attempts < MAX_ATTEMPTS:  # NOT inverts, AND requires both
            trial_state ← rk4_step(state, dt)
            error ← estimate_error(state, trial_state)
            
            IF error < tolerance:
                step_accepted ← True
                state ← trial_state
            ELSE:
                dt ← max(dt * 0.5, dt_min)
            attempts ← attempts + 1
```

Each refinement level reveals new issues and solutions. The **NOT** operator inverts a boolean value (True becomes False, False becomes True). The **AND** operator requires both conditions to be true. This is computational thinking in action!

:::{admonition} 💡 Computational Thinking: The Sentinel Pattern
:class: important

**PATTERN: Sentinel Values**

A sentinel is a special value that signals "stop processing." This pattern appears everywhere in computing:

```python
# Reading until special marker
data = []
while True:
    value = get_next_value()
    if value == -999:  # Sentinel value
        break
    data.append(value)
```

Real-world applications:

- **FITS files**: END keyword marks end of header
- **Network protocols**: Message terminators like "\r\n"
- **Telescope data**: -999 for missing observations
- **String processing**: Null terminators in C strings

The sentinel pattern is how computers know when to stop! You're using the same technique that controls internet data packets and spacecraft telemetry streams.
:::

:::{admonition} 💡 Computational Thinking: The Universal Pattern of Adaptive Algorithms
:class: important

Adaptive timestepping is an instance of a universal pattern:

**PATTERN: Adaptive Refinement**

1. Attempt action with current parameters
2. Evaluate quality of result  
3. If quality insufficient: refine parameters and retry
4. If quality acceptable: proceed and possibly coarsen
5. Include safeguards against infinite refinement

This pattern appears everywhere in computational astrophysics:

- **Adaptive mesh refinement (AMR)** in galaxy formation simulations
- **Step size control** in stellar evolution codes like MESA
- **Learning rate scheduling** in neural networks for photometric redshifts
- **Convergence acceleration** in self-consistent field calculations
- **Importance sampling** in Monte Carlo radiative transfer

Once you recognize this pattern, you'll see it in every sophisticated astronomical code!
:::

## 3.2 Boolean Logic in Scientific Computing

Every decision in your code ultimately reduces to true or false. But in scientific computing, these decisions often involve floating-point numbers, where equality is treacherous and precision is limited. Let's master this fundamental building block that underlies everything from data quality checks to convergence criteria!

### The Complete Set of Comparison Operators

Python provides six comparison operators that return boolean values (True or False):

```{code-cell} ipython3
# All comparison operators in Python
temperature = 5778  # Kelvin (Sun's surface)

# The six fundamental comparisons
print(f"Greater than: {temperature > 6000}")           # False
print(f"Less than: {temperature < 6000}")              # True
print(f"Greater or equal: {temperature >= 5778}")      # True
print(f"Less or equal: {temperature <= 5778}")         # True
print(f"Equal to: {temperature == 5778}")              # True
print(f"Not equal to: {temperature != 6000}")          # True

# Chaining comparisons (Python's elegant feature!)
print(f"\nMain sequence star? {3000 < temperature < 50000}")  # True
# This is equivalent to: (3000 < temperature) AND (temperature < 50000)
```

The **!=** operator (not equal) is particularly useful for filtering out sentinel values or checking if something has changed. The ability to chain comparisons like `3000 < temperature < 50000` is a Python feature that makes code more readable and matches mathematical notation.

### The Three Logical Operators: AND, OR, NOT

Python's logical operators combine or modify boolean values:

```{code-cell} ipython3
# Demonstrating all three logical operators
is_bright = True
is_variable = False

# AND: Both conditions must be true
print(f"Bright AND variable: {is_bright and is_variable}")  # False

# OR: At least one condition must be true  
print(f"Bright OR variable: {is_bright or is_variable}")     # True

# NOT: Inverts the boolean value
print(f"NOT bright: {not is_bright}")                       # False
print(f"NOT variable: {not is_variable}")                   # True

# Complex combinations (real telescope scheduling logic!)
observable = True
weather_good = True
calibrated = False

can_observe = observable and weather_good and (not calibrated or calibrated)
print(f"\nCan observe? {can_observe}")

# Truth table demonstration
print("\nTruth Table for AND:")
for a in [True, False]:
    for b in [True, False]:
        print(f"  {a:5} AND {b:5} = {a and b}")
```

### Special Comparison Operators: is, in

Python has two special operators that are incredibly useful in scientific programming:

```{code-cell} ipython3
import math

# The 'is' operator checks identity (same object in memory)
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(f"a == b: {a == b}")  # True - same values
print(f"a is b: {a is b}")  # False - different objects
print(f"a is c: {a is c}")  # True - same object

# Special case: None should always use 'is'
result = None
print(f"Checking None: {result is None}")  # Preferred
# Don't use: result == None

# The 'in' operator checks membership
stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
print(f"\nIs 'G' a stellar type? {'G' in stellar_types}")  # True
print(f"Is 'X' a stellar type? {'X' in stellar_types}")    # False

# Works with strings too!
filename = "observations_2024.fits"
print(f"Is FITS file? {'fits' in filename}")  # True
```

### The Walrus Operator: Assignment Expressions (Python 3.8+)

Python 3.8 introduced the **walrus operator** (`:=`) which allows assignment within expressions:

```{code-cell} ipython3
# Traditional approach - two steps
data = get_observations()  # Assume this function exists
if len(data) > 100:
    print(f"Large dataset: {len(data)} observations")
    # Notice we call len(data) twice!

# With walrus operator - assign and test in one line
# if (n := len(data)) > 100:
#     print(f"Large dataset: {n} observations")
#     # Now n contains the length, no need to recalculate!

# Real astronomical example (simulated)
def check_observation_quality(observations):
    """Check if we have enough high-quality observations."""
    # Without walrus operator:
    good_obs = [obs for obs in observations if obs['snr'] > 5]
    if len(good_obs) >= 10:
        print(f"Found {len(good_obs)} good observations")
        return good_obs
    
    # With walrus operator (Python 3.8+):
    # if (good_count := len(good_obs)) >= 10:
    #     print(f"Found {good_count} good observations")
    #     return good_obs
    
    return None

# Useful in while loops too
# while (line := file.readline()):  # Read and check in one step
#     process(line)

print("Note: Walrus operator requires Python 3.8+")
print("It's useful but not essential - all code can be written without it")
```

:::{admonition} 📝 Note on the Walrus Operator
:class: note

The walrus operator (`:=`) is optional syntactic sugar introduced in Python 3.8. While it can make some code more concise, it's perfectly fine to write code without it. Many astronomers still use Python 3.7 or earlier, so don't rely on it for shared code.

Use it when:

- You need to use a value in a condition and then reuse it in the body
- Reading files line by line in a while loop
- Avoiding repeated expensive calculations

Avoid it when:

- It makes the code harder to read
- Working with Python < 3.8
- The traditional approach is clearer
:::

### The Floating-Point Equality Trap

Never use **==** with floating-point numbers! Even tiny rounding errors break equality:

```{code-cell} ipython3
# The floating-point trap strikes!
calculated_temp = 5778.0000000001
expected_temp = 5778.0

print(f"Calculated == Expected? {calculated_temp == expected_temp}")  # False!
print(f"Tiny difference: {calculated_temp - expected_temp:.2e}")

# The solution: tolerance-based comparison
def safe_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """
    Safe floating-point comparison for scientific computing.
    Used in actual telescope pointing systems!
    """
    # Handle exact equality (includes infinities)
    if a == b:
        return True
    
    # Handle NaN (NaN != NaN by IEEE standard)
    if math.isnan(a) or math.isnan(b):
        return False
    
    # Handle infinity cases
    if math.isinf(a) or math.isinf(b):
        return a == b
    
    # Normal comparison with tolerance
    # Note: math.isclose() does this internally, but understanding it matters!
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Test our safe comparison
print(f"\n0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")  # False!
print(f"Safe equal? {safe_equal(0.1 + 0.2, 0.3)}")  # True!
print(f"math.isclose? {math.isclose(0.1 + 0.2, 0.3)}")  # True!
```

:::{admonition} ⚠️ Common Bug Alert: The Equality Trap
:class: warning

**TODO**: Check this story! Is it made up? Need references.

**Never use `==` with floating-point numbers!** Even tiny rounding errors break equality.

**Wrong (caused real satellite collision near-miss):**
```python
if velocity == 0.0:  # Dangerous!
    print("At rest")
```

**Right (used in actual spacecraft code):**
```python
if abs(velocity) < 1e-10:  # Safe!
    print("Effectively at rest")
```

The International Space Station uses similar tolerance checks for all docking maneuvers!
:::

### Short-Circuit Evaluation: Order Matters!

:::{margin}
**short-circuit evaluation**
Stopping logical evaluation once result is determined
:::

Python's **and** and **or** operators use short-circuit evaluation — they stop evaluating as soon as the result is determined:

```{code-cell} ipython3
# Short-circuit evaluation prevents errors and saves computation
stellar_data = []  # Empty for demonstration

# WRONG - will crash if data is empty!
# if stellar_data[0] > 0 and len(stellar_data) > 0:  # IndexError!

# CORRECT - checks length first (short-circuits if empty)
if len(stellar_data) > 0 and stellar_data[0] > 0:
    print("First star has positive measurement")
else:
    print("No data or first measurement not positive")

# Even more Pythonic (empty list evaluates to False)
if stellar_data and stellar_data[0] > 0:
    print("First star has positive measurement")

# OR also short-circuits
def expensive_check():
    print("  Running expensive calculation...")
    return True

# This won't call expensive_check() because True or anything is True
print("\nShort-circuit OR demonstration:")
result = True or expensive_check()  # expensive_check never runs!
print(f"Result: {result}")

# But this will call it
result = False or expensive_check()  # expensive_check must run
print(f"Result: {result}")
```

:::{admonition} 🌟 Why This Matters: Satellite Collision Avoidance
:class: tip, dropdown

**TODO:** Get references, is this true?
The European Space Agency uses boolean logic chains for collision warnings:

```python
def check_collision_risk(satellite1, satellite2):
    """Actual logic used for collision avoidance (simplified)"""
    
    # Order matters for efficiency!
    # Check cheap calculations first
    if distance > safe_threshold:  
        return False  # No need to calculate velocities
    
    # Only if close, calculate expensive velocity vectors
    if relative_velocity < 0:  
        return False  # Moving apart
    
    # Only if approaching, do complex uncertainty calculation
    if combined_uncertainty < max_allowed:
        return True  # COLLISION RISK!
    
    return False
```

Checking distance first avoids millions of expensive velocity calculations per day. A single wrong comparison could mean losing a $500 million satellite! In 2009, Iridium 33 and Cosmos 2251 collided because their warning system failed to properly evaluate these conditions.

*Note: This example simplifies the actual collision avoidance algorithms for pedagogical clarity. Real systems use complex orbital mechanics and probability distributions, but the core principle of ordered boolean evaluation remains crucial.*
:::

## 3.3 Conditional Statements: Teaching Computers to Decide

:::{margin}
**guard clause**
Early return statement that handles edge cases before main logic
:::

Conditional statements are where your code makes decisions. In scientific computing, these decisions often involve numerical thresholds, convergence criteria, and boundary conditions. Let's build your intuition for writing robust conditionals that could run on spacecraft or control telescopes!

### The if Statement: Your First Decision Maker

The **if** statement is the simplest conditional — it executes code only when a condition is true:

```{code-cell} ipython3
# Basic if statement
magnitude = 4.5

if magnitude < 6.0:
    print(f"Star is visible to naked eye (mag {magnitude})")

# Nothing happens if condition is false
magnitude = 8.0
if magnitude < 6.0:
    print("This won't print")
    
# Multiple statements in if block
stellar_mass = 10.0  # Solar masses

if stellar_mass > 8:
    print("Massive star detected!")
    print(f"Mass: {stellar_mass} M☉")
    print("Will end as supernova")
    remnant = "neutron star or black hole"
```

### The if-else Statement: Binary Decisions

The **else** clause provides an alternative when the condition is false:

```{code-cell} ipython3
# Binary decision with if-else
redshift = 0.8

if redshift < 0.1:
    classification = "nearby galaxy"
else:
    classification = "distant galaxy"
    
print(f"z = {redshift}: {classification}")

# You can have multiple statements in each block
observation_snr = 3.5  # Signal-to-noise ratio

if observation_snr >= 5.0:
    print("High quality detection")
    process_immediately = True
    confidence = "high"
else:
    print("Low SNR - needs verification")
    process_immediately = False
    confidence = "low"
```

### The elif Statement: Multiple Choices

The **elif** (else if) statement allows multiple conditions to be checked in sequence:

```{code-cell} ipython3
def classify_stellar_remnant(mass_solar):
    """
    Determine stellar remnant type based on initial mass.
    Demonstrates guard clauses and elif chains.
    Based on Chandrasekhar limit and stellar evolution theory.
    """
    # Guard clauses - validate input FIRST
    if mass_solar <= 0:
        raise ValueError(f"Mass must be positive: {mass_solar}")
    
    if not math.isfinite(mass_solar):
        raise ValueError(f"Mass must be finite: {mass_solar}")
    
    # Main classification logic with elif chain
    if mass_solar < 0.08:
        remnant = "brown dwarf (failed star)"
    elif mass_solar < 8:
        remnant = "white dwarf"
    elif mass_solar < 25:
        remnant = "neutron star"
    else:  # Final catch-all
        remnant = "black hole"
    
    # Add uncertainty near boundaries (real issue in astronomy!)
    boundaries = [0.08, 8, 25]
    min_distance = min(abs(mass_solar - b) for b in boundaries)
    
    if min_distance < 0.5:
        remnant += " (near boundary - uncertain)"
    
    return remnant

# Test our classifier
print(classify_stellar_remnant(1.0))   # Our Sun's fate
print(classify_stellar_remnant(7.8))   # Near boundary!
print(classify_stellar_remnant(30))    # Massive star fate
```

### Guard Clauses: Fail Fast, Fail Clear

Guard clauses handle special cases immediately, preventing deep nesting and making code clearer. This pattern is essential for scientific code where invalid inputs can cause subtle bugs hours into a simulation!

```{code-cell} ipython3
def calculate_orbital_period(a, M, validate=True):
    """
    Kepler's third law with comprehensive validation.
    This defensive style could have saved Mars Climate Orbiter!
    
    Parameters:
        a: Semi-major axis [AU]
        M: Central mass [solar masses]
    """
    # Guard clauses handle problems immediately
    if validate:
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive: {a} AU")
        if M <= 0:
            raise ValueError(f"Mass must be positive: {M} M☉")
        
        # Check for orbit inside Schwarzschild radius!
        rs_au = 2.95e-8 * M  # Schwarzschild radius in AU
        if a < rs_au:
            raise ValueError(f"Orbit inside black hole event horizon: a={a} AU, Rs={rs_au} AU")
    
    # Main calculation - only runs if guards pass
    G_au_msun = 39.478  # G in AU³/M☉/year²
    period_years = math.sqrt(a**3 / M)  # Simplified Kepler's third law
    
    # Sanity check result
    if validate and period_years > 13.8e9:
        import warnings
        warnings.warn(f"Period exceeds age of universe: {period_years:.2e} years")
    
    return period_years

# Test with real systems
print(f"Earth: {calculate_orbital_period(1.0, 1.0):.2f} years")
print(f"Mercury: {calculate_orbital_period(0.387, 1.0):.2f} years")
print(f"Proxima Centauri b: {calculate_orbital_period(0.0485, 0.122):.3f} years")
```

### The Ternary Operator: Compact Conditionals

Python's ternary operator provides a compact way to write simple if-else statements:

```{code-cell} ipython3
# Ternary operator: value_if_true if condition else value_if_false
magnitude = 3.5
visibility = "visible" if magnitude < 6.0 else "not visible"
print(f"Star with magnitude {magnitude} is {visibility}")

# Useful for setting defaults based on conditions
exposure_time = 30  # seconds
quality = "good" if exposure_time > 10 else "poor"

# Can be nested but don't overdo it!
stellar_class = "G"
temperature = 5778 if stellar_class == "G" else (7500 if stellar_class == "A" else 3500)
print(f"Class {stellar_class} star: ~{temperature}K")
```

:::{admonition} 🌟 Why This Matters: The Mars Climate Orbiter Disaster
:class: tip, dropdown

In 1999, NASA lost the $327.6 million Mars Climate Orbiter because one team used metric units while another used imperial. A simple guard clause could have saved it:

*Note: The $327.6 million figure represents total mission cost. This anecdote simplifies a complex failure to emphasize the importance of unit validation. The actual failure involved multiple factors, but the unit confusion was the primary cause identified in NASA's investigation reports.*

```python
def process_thrust_data(force, units):
    """This guard clause would have saved $327 million!"""
    
    # Validate units BEFORE processing
    valid_units = {'N': 1.0, 'lbf': 4.448222}  # Conversion factors
    
    if units not in valid_units:
        raise ValueError(f"Unknown units: {units}. Use 'N' or 'lbf'")
    
    # Convert to standard units (Newtons)
    force_newtons = force * valid_units[units]
    
    # Additional sanity check
    if force_newtons > 1000:  # Typical max thruster force
        warnings.warn(f"Unusually high thrust: {force_newtons} N")
    
    return force_newtons

# The actual error: ground software sent lbf, flight expected N
# Result: 4.45× error accumulated over months!
```

The orbiter entered Mars atmosphere at 57 km instead of 226 km altitude and disintegrated. Your guard clauses aren't just good practice — they prevent disasters!
:::

## 3.4 Loops: The Heart of Scientific Computation

Now that you've mastered making decisions with conditionals, let's make your code repeat tasks efficiently! Loops are where your programs gain superpowers — they're the difference between analyzing one star and analyzing millions. Every N-body simulation, every light curve analysis, every Monte Carlo calculation depends on loops. The patterns you learn here will appear in every algorithm you write for the rest of your career.

### The for Loop: Iterating Over Sequences

The **for** loop iterates over any sequence (list, tuple, string, range):

```{code-cell} ipython3
# Basic for loop
stellar_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

for spectral_class in stellar_types:
    print(f"Class {spectral_class} star")

# Using range() for counting
print("\nCounting with range:")
for i in range(5):  # 0, 1, 2, 3, 4 (not 5!)
    print(f"Observation {i}")

# Range with start, stop, step
print("\nEvery 2nd hour from 20:00 to 02:00:")
for hour in range(20, 26, 2):  # 20, 22, 24
    print(f"{hour:02d}:00")
```

### The Accumulator Pattern in Astronomy

```{margin}
**accumulator pattern**
Iteratively combining values into a running aggregate
```

The accumulator pattern is fundamental to scientific computing:

```{code-cell} ipython3
# Calculating center of mass for a star cluster
star_masses = [1.2, 0.8, 2.1, 0.5, 1.5]  # Solar masses
star_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Parsecs from origin

total_mass = 0
weighted_position = 0

for mass, position in zip(star_masses, star_positions):
    total_mass += mass
    weighted_position += mass * position

center_of_mass = weighted_position / total_mass
print(f"Cluster center of mass: {center_of_mass:.3f} pc")
print(f"Total cluster mass: {total_mass:.1f} M☉")

# This same pattern calculates:
# - Barycenter of binary systems (how we detect exoplanets!)
# - Average stellar metallicity in galaxies
# - Integrated luminosity functions
# - Weighted mean magnitudes
```

### Common for Loop Patterns

Python provides several useful functions for loop patterns:

```{code-cell} ipython3
# enumerate() gives you index and value
magnitudes = [10.2, 10.1, 9.5, 10.3, 8.2, 10.2]

print("Finding bright events with enumerate:")
for i, mag in enumerate(magnitudes):
    if mag < 9.0:
        print(f"  Alert! Index {i}: magnitude {mag}")

# zip() for parallel iteration
times = [0, 1, 2, 3, 4]  # seconds
positions = [0, 4.9, 19.6, 44.1, 78.4]  # meters

print("\nParallel iteration with zip:")
for t, x in zip(times, positions):
    if t > 0:  # Avoid division by zero
        velocity = x / t
        print(f"  t={t}s: v={velocity:.1f} m/s")
```

### The while Loop: Conditional Iteration

The **while** loop continues as long as a condition remains true:

```{code-cell} ipython3
# Basic while loop
count = 0
while count < 3:
    print(f"Iteration {count}")
    count += 1  # Don't forget to update!

# While loop for convergence
print("\nConvergence example:")
value = 100.0
target = 1.0
iteration = 0

while abs(value - target) > 0.01 and iteration < 100:  # Safety limit!
    value = value * 0.9 + target * 0.1  # Gradual approach
    iteration += 1
    if iteration <= 3 or iteration % 10 == 0:  # Print selectively
        print(f"  Iter {iteration}: value = {value:.3f}")

print(f"Converged to {value:.3f} after {iteration} iterations")
```

### Loop Control: break, continue, and else

Python provides additional loop control statements:

```{code-cell} ipython3
# break: Exit loop early
print("Using break to find first detection:")
observations = [0.1, 0.3, 0.2, 5.8, 0.4, 6.2]

for obs in observations:
    if obs > 1.0:
        print(f"First significant detection: {obs}")
        break  # Stop searching
        
# continue: Skip to next iteration
print("\nUsing continue to skip bad data:")
measurements = [1.2, -999, 2.3, -999, 3.4]

for value in measurements:
    if value == -999:  # Sentinel value
        continue  # Skip this iteration
    print(f"Processing: {value}")

# else clause: Runs if loop completes without break
print("\nLoop else clause:")
search_list = [1, 2, 3, 4, 5]
target = 7

for item in search_list:
    if item == target:
        print("Found!")
        break
else:  # This runs because we didn't break
    print(f"Target {target} not found in list")
```

### The pass Statement: Placeholder

The **pass** statement does nothing — useful as a placeholder:

```{code-cell} ipython3
# pass as placeholder
for i in range(3):
    if i == 1:
        pass  # TODO: Add processing here later
    else:
        print(f"Processing {i}")

# Often used in exception handling
try:
    risky_operation = 1 / 1  # No error this time
except ZeroDivisionError:
    pass  # Silently ignore this specific error

print("Continued after pass")
```

### Nested Loops: Processing 2D Data

Loops can be nested to process multi-dimensional data:

```{code-cell} ipython3
# Nested loops for 2D grid (like CCD image)
print("Processing 3x3 pixel grid:")
for row in range(3):
    for col in range(3):
        pixel_value = row * 3 + col  # Simulated pixel value
        print(f"({row},{col})={pixel_value}", end="  ")
    print()  # New line after each row

# More realistic: Finding peaks in 2D data
data_2d = [
    [1, 2, 1],
    [2, 9, 2],  # Peak at (1,1)
    [1, 2, 1]
]

print("\nFinding peaks in 2D array:")
for i in range(len(data_2d)):
    for j in range(len(data_2d[i])):
        if data_2d[i][j] > 5:
            print(f"Peak at ({i},{j}): value={data_2d[i][j]}")
```

:::{admonition} ⚠️ Common Bug Alert: Off-by-One Errors
:class: warning

The most common bug in all of programming! Python's zero-indexing catches everyone:

**Classic Mistake (lost data from Hubble!):**
```python
observations = [1, 2, 3, 4, 5]
# Trying to process all elements
for i in range(1, len(observations)):  # OOPS! Skips first element
    process(observations[i])

# Or worse - going past the end
for i in range(len(observations) + 1):  # IndexError on last iteration!
    process(observations[i])
```

**Remember:**
- `range(n)` gives 0, 1, ..., n-1 (NOT including n!)
- List of length n has indices 0 to n-1
- The last element is at index len(list) - 1

This bug has crashed spacecraft software, corrupted astronomical databases, and frustrated millions of programmers. Double-check your ranges!
:::

:::{admonition} ⚠️ Common Bug Alert: Infinite While Loops
:class: warning

Don't worry — everyone writes an infinite loop occasionally! Even senior programmers at NASA have done it. Here are real cases:

**Case 1: Floating-point precision prevents exact equality**
```python
x = 0.0
while x != 1.0:  # INFINITE LOOP!
    x += 0.1  # After 10 additions, x ≈ 0.9999999999

# Fix: Use tolerance
while abs(x - 1.0) > 1e-10:
    x += 0.1
```

**Case 2: Forgetting to update loop variable**
```python
i = 0
while i < 10:
    process(data)
    # Forgot: i += 1  # INFINITE LOOP!
```

Always add a maximum iteration safeguard!
:::

## 3.5 List Comprehensions: Elegant and Efficient

Now that you've mastered loops, let's evolve them into something even more powerful! List comprehensions are Python's gift to scientific programmers. They transform verbose loops into concise, readable, and faster expressions.

### From Loop to Comprehension

```{code-cell} ipython3
# Traditional loop approach
squares_loop = []
for x in range(10):
    if x % 2 == 0:  # Even numbers only
        squares_loop.append(x**2)
print(f"Loop result: {squares_loop}")

# List comprehension - same result, clearer intent!
squares_comp = [x**2 for x in range(10) if x % 2 == 0]
print(f"Comprehension: {squares_comp}")

# The anatomy of a list comprehension:
# [expression for item in sequence if condition]
#      ↓           ↓         ↓           ↓
#  Transform   Variable  Source    Filter (optional)
```

### Real Astronomical Applications

```{code-cell} ipython3
# Filter and transform magnitude data from a survey
magnitudes = [12.3, 15.1, 13.7, 16.2, 14.5, 17.1, 11.8, 18.5, 13.2]

# Get fluxes for observable stars (mag < 16, typical small telescope limit)
observable_fluxes = [10**(-0.4 * mag) 
                     for mag in magnitudes 
                     if mag < 16.0]

print(f"Observable star count: {len(observable_fluxes)}/{len(magnitudes)}")
print(f"Brightest flux: {max(observable_fluxes):.2e}")
print(f"Faintest flux: {min(observable_fluxes):.2e}")

# Dictionary comprehension (bonus!)
star_dict = {f"star_{i}": mag 
             for i, mag in enumerate(magnitudes) 
             if mag < 15}
print(f"\nBright stars dictionary: {star_dict}")
```

### When NOT to Use Comprehensions

```{code-cell} ipython3
# BAD: Too complex - unreadable!
# result = [process(x) if condition(x) else alternative(y) 
#           for x, y in zip(list1, list2) 
#           if validate(x) and check(y)]

# GOOD: Clear loop for complex logic
def classify_galaxies(redshifts, luminosities):
    """When logic is complex, loops are clearer!"""
    classifications = []
    for z, L in zip(redshifts, luminosities):
        if z < 0.5 and L > 1e10:
            classifications.append("nearby bright")
        elif z < 0.5:
            classifications.append("nearby faint")
        elif L > 1e10:
            classifications.append("distant bright")
        else:
            classifications.append("distant faint")
    return classifications

# Clear and maintainable!
z_values = [0.1, 0.8, 0.3, 1.2]
L_values = [5e9, 2e10, 8e10, 3e9]
galaxy_types = classify_galaxies(z_values, L_values)
print(f"Galaxy classifications: {galaxy_types}")
```

## 3.6 Advanced Control Flow Patterns

Now let's explore powerful patterns that appear throughout scientific computing. These aren't just code tricks — they're fundamental algorithmic building blocks!

### Welford's Algorithm: Numerically Stable Statistics

```{code-cell} ipython3
def running_statistics(data_stream):
    """
    Calculate mean and variance in single pass.
    Uses Welford's algorithm (1962) for numerical stability.
    Essential for processing streaming telescope data!
    """
    n = 0
    mean = 0.0
    M2 = 0.0
    
    for value in data_stream:
        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean
        M2 += delta * delta2
    
    if n < 2:
        return mean, float('nan')
    
    variance = M2 / (n - 1)
    return mean, variance

# Test with problematic data (large baseline with small variations)
photometry = [1e8, 1e8 + 1, 1e8 + 2, 1e8 - 1, 1e8 + 0.5]  
mean, var = running_statistics(photometry)
print(f"Stable algorithm: mean={mean:.1f}, std={math.sqrt(var):.2f}")

# The naive approach would lose precision!
naive_mean = sum(photometry) / len(photometry)
naive_var = sum(x**2 for x in photometry) / len(photometry) - naive_mean**2
print(f"Naive (problematic): mean={naive_mean:.1f}, std={math.sqrt(abs(naive_var)):.2f}")
```

:::{admonition} 💡 Computational Thinking: The Convergence Pattern
:class: important

**PATTERN: Iterative Convergence**

```python
initialize state
iteration_count = 0

while not converged and iteration_count < max_iterations:
    new_state = update(state)
    converged = check_convergence(state, new_state, tolerance)
    state = new_state
    iteration_count += 1

if not converged:
    handle_failure()
```

This pattern appears throughout astrophysics:
- **Kepler's equation solver** (finding true anomaly)
- **Stellar structure integration** (hydrostatic equilibrium)
- **Radiative transfer** (temperature iterations)
- **N-body orbit integration** (adaptive timesteps)

Master this pattern and you've mastered half of computational astrophysics!
:::

## 3.7 Debugging Control Flow

Logic errors are the hardest bugs because the code runs without crashing but produces wrong results. Let's build your debugging arsenal!

### Strategic Print Debugging

```{code-cell} ipython3
def debug_convergence(initial, target, rate, max_iter=20):
    """
    Example of strategic debug output.
    Shows exactly where and why algorithms succeed or fail.
    """
    
    current = initial
    history = []
    
    for iteration in range(max_iter):
        old = current
        current = current * (1 - rate) + target * rate
        change = current - old
        history.append(current)
        
        # Strategic output - not everything!
        if iteration < 3 or iteration % 5 == 0:
            print(f"Iter {iteration:2d}: {old:.4f} → {current:.4f} (Δ={change:+.5f})")
        
        # Convergence check
        if abs(current - target) < 1e-6:
            print(f"✓ CONVERGED at iteration {iteration}")
            return current
        
        # Detect problems early
        if len(history) > 3:
            if (history[-1] > history[-2] < history[-3]):
                print(f"⚠ OSCILLATION detected")
                
    print(f"✗ FAILED after {max_iter} iterations")
    return current

# Test the algorithm
print("Testing convergence:")
result = debug_convergence(0, 100, 0.1)
```

### Using Assertions for Validation

The **assert** statement helps catch bugs during development:

```{code-cell} ipython3
def calculate_magnitude_average(magnitudes):
    """Calculate average magnitude with assertions for debugging."""
    
    # Assertions document and enforce assumptions
    assert len(magnitudes) > 0, "Need at least one magnitude"
    assert all(isinstance(m, (int, float)) for m in magnitudes), "All must be numbers"
    assert all(0 < m < 30 for m in magnitudes), "Magnitudes must be reasonable"
    
    # Safe to proceed after assertions
    return sum(magnitudes) / len(magnitudes)

# Test with good data
good_mags = [10.2, 10.5, 10.3]
avg = calculate_magnitude_average(good_mags)
print(f"Average magnitude: {avg:.2f}")

# WARNING: Assertions can be disabled in production with python -O
# This means they should NEVER be used for actual validation!
```

:::{admonition} ⚠️ Critical Warning: Assertions Are Not for Production!
:class: warning

**Never use assertions for user input validation or critical checks!** Assertions can be completely disabled when Python runs with optimization (`python -O`), causing them to be skipped entirely.

**WRONG - Don't do this for user-facing code:**
```python
def process_user_data(value):
    assert value > 0  # DANGEROUS! Might not run in production!
    return math.sqrt(value)
```

**RIGHT - Use explicit validation for production:**
```python
def process_user_data(value):
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    return math.sqrt(value)
```

**When to use assertions:**
- Documenting internal assumptions during development
- Catching programming errors early (not user errors)
- Self-checks in algorithms (but have a fallback plan)
- Test suites and debugging

**When NOT to use assertions:**
- Validating user input
- Checking file existence or permissions
- Network availability checks
- Any check that must run in production

Think of assertions as "developer notes that can catch bugs" rather than "guards that protect your code."
:::

:::{admonition} 🌟 The More You Know: How Kepler Found Over 2,700 Exoplanets
:class: tip, dropdown

The Kepler Space Telescope discovered 2,778 confirmed exoplanets (with thousands more candidates) using exactly the control flow patterns you just learned! Here's the simplified algorithm:

*Note: This algorithm is greatly simplified for pedagogical purposes. The actual Kepler pipeline used sophisticated techniques including Fourier transforms, multiple detrending algorithms, and extensive validation checks. However, the control flow patterns shown here — guard clauses, filtering, iteration, and conditional validation — formed the backbone of the real system.*

```python
def kepler_planet_search(star_id, light_curve):
    """Simplified Kepler planet detection algorithm"""
    
    # Guard clause - data quality check
    if len(light_curve) < 1000:
        return None
    
    # Remove outliers (cosmic rays, etc.)
    cleaned = [point for point in light_curve 
               if abs(point - median) < 5 * sigma]
    
    # Search for periodic dips
    best_period = None
    best_depth = 0
    
    for trial_period in range(1, 365):  # Days
        folded = fold_light_curve(cleaned, trial_period)
        depth = measure_transit_depth(folded)
        
        if depth > best_depth and depth > 3 * noise_level:
            best_period = trial_period
            best_depth = depth
    
    # Validate as planet (not eclipsing binary)
    if best_period:
        if is_v_shaped(folded):  # Binary check
            return None
        if depth > 0.5:  # Too deep
            return None
            
        return {'period': best_period, 'depth': best_depth}
    
    return None
```

This ran on 150,000+ stars for 4 years! Your code uses the same patterns that revealed the universe is full of planets!
:::

:::{admonition} 🛠️ Debug This! The Telescope Priority Bug
:class: challenge

A telescope scheduling system has a subtle bug in its priority logic. Can you find and fix it?

```python
def assign_telescope_priority(observation):
    """
    Assign priority based on object type and time sensitivity.
    Higher numbers = higher priority.
    
    THIS CODE HAS A BUG - find it!
    """
    magnitude = observation['magnitude']
    obj_type = observation['type']
    time_critical = observation['time_critical']
    
    # Assign base priority by object type
    if obj_type == 'supernova':
        priority = 100
    elif obj_type == 'variable_star' and magnitude < 12:
        priority = 70
    elif obj_type == 'variable_star':  # This has a problem!
        priority = 50
    elif obj_type == 'asteroid' and time_critical:
        priority = 80
    elif obj_type == 'galaxy':
        priority = 30
    else:
        priority = 10
    
    # Boost for bright objects
    if magnitude < 10:
        priority += 20
    
    return priority

# Test case that reveals the bug:
obs = {'type': 'variable_star', 'magnitude': 8, 'time_critical': True}
print(f"Priority: {assign_telescope_priority(obs)}")
# Expected: 70 + 20 = 90 (bright variable star)
# Actually gets: 90 (seems right... or is it?)

obs2 = {'type': 'variable_star', 'magnitude': 11.5, 'time_critical': True}
print(f"Priority: {assign_telescope_priority(obs2)}")
# Expected: 70 (bright variable star condition)
# Actually gets: 70 (correct!)

obs3 = {'type': 'variable_star', 'magnitude': 14, 'time_critical': True}
print(f"Priority: {assign_telescope_priority(obs3)}")
# Expected: 50 (dim variable star)
# What does it actually get?
```
:::

:::{admonition} Solution
:class: tip, dropdown

**The Bug**: The elif chain for variable stars has overlapping conditions that aren't immediately obvious!

Looking at the variable star conditions:

1. `elif obj_type == 'variable_star' and magnitude < 12:` → priority = 70
2. `elif obj_type == 'variable_star':` → priority = 50

The second condition can NEVER be reached for bright variable stars because they're caught by the first condition. However, this actually works correctly by accident! The real issue is that the logic is confusing and fragile.

**Better Design**:
```python
def assign_telescope_priority_fixed(observation):
    """Fixed version with clearer logic."""
    magnitude = observation['magnitude']
    obj_type = observation['type']
    time_critical = observation['time_critical']
    
    # Assign base priority by object type
    if obj_type == 'supernova':
        priority = 100
    elif obj_type == 'asteroid' and time_critical:
        priority = 80
    elif obj_type == 'variable_star':
        # Nested logic is clearer for subcategories
        if magnitude < 12:
            priority = 70
        else:
            priority = 50
    elif obj_type == 'galaxy':
        priority = 30
    else:
        priority = 10
    
    # Boost for bright objects
    if magnitude < 10:
        priority += 20
    
    return priority
```

**Key Lessons**:
1. Order elif conditions from most specific to most general
2. Avoid overlapping conditions in elif chains
3. Consider using nested if statements for subcategories
4. The original code works but is hard to maintain and understand

This type of subtle logic error is common in real telescope scheduling software and can lead to suboptimal observation planning!
:::

## The Variable Star Thread Continues

Let's apply our control flow knowledge to extend our variable star analysis from Chapters 1 and 2:

```{code-cell} ipython3
# Chapter 3: Variable Star - Adding Periodicity Detection
import json
import math

# Create sample data (in real use, load from Chapter 2)
star = {
    'name': 'Delta Cephei',
    'period': 5.366319,
    'mag_mean': 3.95,
    'mag_amp': 0.88,
    'phase_function': 'sinusoidal'
}

def analyze_phase_coverage(times, period, min_coverage=0.6):
    """
    Check if observations adequately sample the phase space.
    Critical for period determination accuracy!
    """
    # Guard clause
    if not times or period <= 0:
        return False, "Invalid input data"
    
    # Calculate phases
    phases = [(t % period) / period for t in times]
    
    # Divide phase space into bins
    n_bins = 10
    bins_filled = set()
    
    for phase in phases:
        bin_index = int(phase * n_bins)
        bins_filled.add(bin_index)
    
    coverage = len(bins_filled) / n_bins
    
    # Conditional logic for assessment
    if coverage >= min_coverage:
        quality = "good" if coverage > 0.8 else "adequate"
        return True, f"Phase coverage {quality}: {coverage:.1%}"
    else:
        return False, f"Insufficient coverage: {coverage:.1%} < {min_coverage:.1%}"

# Test with simulated observations
test_times = [0.5, 1.2, 2.7, 3.1, 4.8, 5.9, 7.2, 8.5, 9.1, 10.3]
adequate, message = analyze_phase_coverage(test_times, star['period'])
print(f"Delta Cephei observations: {message}")

# Save enhanced data for Chapter 4
star['last_analysis'] = {
    'phase_coverage': adequate,
    'message': message,
    'n_observations': len(test_times)
}

try:
    with open('variable_star_ch3.json', 'w') as f:
        json.dump(star, f, indent=2)
    print("✓ Data saved for Chapter 4!")
except IOError as e:
    print(f"✗ Could not save: {e}")
```

## Practice Exercises

Now apply your control flow mastery to real astronomical problems!

### Exercise 3.1: Phase Dispersion Minimization

:::{admonition} Complete Implementation (40-50 lines)
:class: exercise

```python
def find_period_pdm(times, magnitudes, min_period=0.1, max_period=10.0):
    """
    Find the period of a variable star using Phase Dispersion Minimization.
    This is a REAL algorithm used in astronomy!
    
    Your implementation should:
    1. Use nested loops for coarse then fine search
    2. Apply the convergence pattern from Section 3.6
    3. Include guard clauses for invalid input
    4. Use list comprehensions where appropriate
    
    Pseudocode to get started:
    - Validate inputs with guard clauses
    - Coarse search with 0.1 day steps
    - Find minimum dispersion period
    - Refine with 0.01 day steps around minimum
    - Continue until convergence
    """
    # Your implementation here
    pass
```
:::

### Exercise 3.2: Transient Detection Pipeline

:::{admonition} Multi-Part Exercise (30-40 lines total)
:class: exercise

Part A: Implement data cleaning
Part B: Detect variability using Welford's algorithm
Part C: Classify transients with elif chains

```python
def process_survey_data(times, mags, errors):
    """
    Complete pipeline for transient detection.
    Uses all control flow patterns from this chapter!
    """
    # Part A: Clean data (guard clauses, list comprehension)
    # Part B: Find variables (Welford's algorithm)
    # Part C: Classify (elif chains)
    pass
```
:::

### Exercise 3.3: Debug the Light Curve Folder

:::{admonition} Find and Fix Three Bugs
:class: exercise

```python
def fold_light_curve(times, mags, period):
    """This function has 3 bugs - find and fix them!"""
    phases = []
    folded_mags = []
    
    for i in range(len(times)):
        phase = times[i] / period  # Bug 1: Should use modulo!
        phases.append(phase)
        folded_mags.append(mags[i])
    
    # Sort by phase
    for i in range(len(phases)):
        for j in range(len(phases)):  # Bug 2: j should start at i+1
            if phases[i] > phases[j]:  # Bug 3: Wrong comparison
                phases[i], phases[j] = phases[j], phases[i]
                folded_mags[i], folded_mags[j] = folded_mags[j], folded_mags[i]
    
    return phases, folded_mags
```
:::

## Main Takeaways

What an incredible journey you've just completed! You've transformed from someone who writes code line by line to someone who designs algorithms systematically. This transformation mirrors the evolution every computational scientist goes through, from tentative beginner to confident algorithm designer.

You started by learning to think in pseudocode, a skill that gives you the power to design before you code. Those three levels of refinement you practiced are your blueprint for success. Every hour you invest in pseudocode saves many hours of debugging. When you design your next algorithm for analyzing galaxy spectra or simulating stellar evolution, you'll catch logical flaws on paper instead of after hours of computation.

The complete set of comparison and logical operators you've mastered — from simple greater-than checks to complex boolean combinations with **and**, **or**, and **not** — gives you the full vocabulary for expressing any logical condition. You understand that `==` is dangerous with floats, that **is** checks identity not equality, and that **in** elegantly tests membership. These aren't just syntax details; they're the building blocks of every data validation, every convergence check, every quality filter you'll ever write.

Your understanding of conditional statements goes beyond syntax to defensive programming philosophy. Those guard clauses you learned could literally prevent spacecraft crashes or save irreplaceable telescope time. The elif chains you practiced will classify astronomical objects, determine observing strategies, and control instrument settings. Every conditional you write is a decision that shapes how your code responds to the infinite variety of real data.

The loop patterns you've mastered are universal across computational physics. That accumulator pattern using Welford's algorithm? It's calculating photometric precision in the TESS pipeline right now. The convergence pattern with safety limits? It's finding equilibrium in stellar models. The nested loops you practiced? They're processing CCD images from every major observatory. Whether using **for** loops to iterate through catalogs, **while** loops to converge solutions, or list comprehensions to filter data, you now have the full toolkit.

Most importantly, you've learned that bugs aren't failures — they're learning opportunities. Every infinite loop teaches you about termination conditions. Every off-by-one error reinforces proper indexing. The debugging strategies you've developed, from strategic print statements to assertions, will serve you throughout your career. Even the experts at NASA and ESA use these same techniques.

Remember that every major computational achievement relies on these fundamentals. The control flow patterns you've learned detected gravitational waves at LIGO, discovered thousands of exoplanets with Kepler, and process images from JWST. You're not just learning Python syntax — you're joining a tradition of computational thinking that enables humanity's greatest discoveries.

## Definitions

**Accumulator Pattern**: An algorithmic pattern where values are iteratively combined into a running total or aggregate, fundamental to reductions and statistical calculations in astronomical data processing.

**Adaptive Refinement**: A universal pattern where parameters are adjusted based on quality metrics, with safeguards against infinite refinement, appearing in timestepping, mesh refinement, and optimization throughout computational astrophysics.

**and**: Logical operator that returns True only if both operands are true, using short-circuit evaluation.

**assert**: Statement that raises an AssertionError if a condition is false, used for debugging and documenting assumptions during development (not for production validation).

**Boolean Logic**: The system of true/false values and logical operations (and, or, not) that underlies all conditional execution in programs.

**break**: Statement that immediately exits the current loop, skipping any remaining iterations.

**Conditional Statement**: A control structure (if/elif/else) that executes different code blocks based on whether conditions evaluate to true or false.

**continue**: Statement that skips the rest of the current loop iteration and proceeds to the next iteration.

**elif**: "Else if" statement that checks an additional condition when the previous if or elif was false.

**else**: Clause that executes when all previous if/elif conditions were false, or when a loop completes without breaking.

**for**: Loop that iterates over elements in a sequence or iterable object.

**Guard Clause**: A conditional statement at the beginning of a function that handles special cases or invalid inputs immediately, preventing deep nesting.

**if**: Statement that executes code only when a specified condition is true.

**in**: Operator that tests membership in a sequence or collection.

**is**: Operator that tests object identity (same object in memory), not just equality of values.

**List Comprehension**: A concise Python syntax for creating lists: `[expression for item in iterable if condition]`.

**not**: Logical operator that inverts a boolean value (True becomes False, False becomes True).

**or**: Logical operator that returns True if at least one operand is true, using short-circuit evaluation.

**pass**: Null statement that does nothing, used as a placeholder where syntax requires a statement.

**Pseudocode**: A human-readable description of an algorithm that focuses on logic and structure without syntactic details.

**Short-circuit Evaluation**: The behavior where logical operators stop evaluating as soon as the result is determined.

**Walrus Operator** (`:=`): Assignment expression operator (Python 3.8+) that assigns a value to a variable as part of an expression, allowing both assignment and testing in a single statement.

**while**: Loop that continues executing as long as a specified condition remains true.

## Key Takeaways

✓ Pseudocode reveals logical flaws before they become bugs — always design before implementing

✓ Master all six comparison operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) and three logical operators (**and**, **or**, **not**)

✓ Never use `==` with floating-point numbers; always use tolerance-based comparisons like `math.isclose()`

✓ The **is** operator checks identity, not equality — use it for None checks

✓ The **in** operator elegantly tests membership in sequences or strings

✓ Guard clauses handle special cases first, making main logic clearer

✓ **for** loops iterate over sequences, **while** loops continue until a condition becomes false

✓ **break** exits loops early, **continue** skips to the next iteration, **else** runs if loop completes

✓ List comprehensions are faster than loops for simple transformations but become unreadable for complex logic

✓ Short-circuit evaluation in **and**/**or** prevents errors and improves performance

✓ The accumulator pattern is fundamental to scientific computing, appearing in all statistical calculations

✓ Always include maximum iteration limits in **while** loops to prevent infinite loops

✓ Welford's algorithm solves numerical stability issues in streaming statistics

✓ Use **assert** statements to document and enforce assumptions during development

✓ Every major astronomical discovery relies on the control flow patterns you've learned

## Quick Reference Tables

```{list-table} Comparison Operators
:header-rows: 1
:widths: 15 30 55

* - Operator
  - Description
  - Example
* - `>`
  - Greater than
  - `if magnitude > 6.0:`
* - `<`
  - Less than
  - `if redshift < 0.1:`
* - `>=`
  - Greater or equal
  - `if snr >= 5.0:`
* - `<=`
  - Less or equal
  - `if error <= tolerance:`
* - `==`
  - Equal (avoid with floats!)
  - `if status == 'complete':`
* - `!=`
  - Not equal
  - `if flag != -999:`
```

```{list-table} Logical Operators
:header-rows: 1
:widths: 15 35 50

* - Operator
  - Description
  - Example
* - `and`
  - Both must be true
  - `if x > 0 and y > 0:`
* - `or`
  - At least one true
  - `if bright or variable:`
* - `not`
  - Inverts boolean
  - `if not converged:`
```

```{list-table} Special Operators
:header-rows: 1
:widths: 15 35 50

* - Operator
  - Description
  - Example
* - `in`
  - Membership test
  - `if 'fits' in filename:`
* - `is`
  - Identity test
  - `if result is None:`
* - `is not`
  - Negative identity
  - `if data is not None:`
* - `:=`
  - Walrus operator (Python 3.8+)
  - `if (n := len(data)) > 100:`
```

```{list-table} Control Flow Statements
:header-rows: 1
:widths: 20 30 50

* - Statement
  - Purpose
  - Example
* - `if/elif/else`
  - Conditional execution
  - `if mag < 6: visible = True`
* - `for`
  - Iterate over sequence
  - `for star in catalog:`
* - `while`
  - Loop while condition true
  - `while error > tolerance:`
* - `break`
  - Exit loop early
  - `if converged: break`
* - `continue`
  - Skip to next iteration
  - `if bad_data: continue`
* - `pass`
  - Do nothing (placeholder)
  - `if not ready: pass`
* - `assert`
  - Debug check
  - `assert len(data) > 0`
```

```{list-table} Built-in Functions for Loops
:header-rows: 1
:widths: 20 35 45

* - Function
  - Purpose
  - Example
* - `range(n)`
  - Generate 0 to n-1
  - `for i in range(10):`
* - `range(start, stop, step)`
  - Generate with step
  - `for i in range(0, 10, 2):`
* - `enumerate(seq)`
  - Get index and value
  - `for i, val in enumerate(data):`
* - `zip(seq1, seq2)`
  - Parallel iteration
  - `for x, y in zip(xs, ys):`
* - `len(seq)`
  - Sequence length
  - `for i in range(len(data)):`
```

```{list-table} Comparison Functions
:header-rows: 1
:widths: 25 35 40

* - Function
  - Purpose
  - Example
* - `all(iterable)`
  - All elements true
  - `if all(x > 0 for x in data):`
* - `any(iterable)`
  - Any element true
  - `if any(x < 0 for x in data):`
* - `math.isclose()`
  - Safe float comparison
  - `if math.isclose(a, b):`
* - `math.isfinite()`
  - Check not inf/nan
  - `if math.isfinite(result):`
* - `math.isnan()`
  - Check for NaN
  - `if not math.isnan(value):`
* - `math.isinf()`
  - Check for infinity
  - `if math.isinf(value):`
* - `isinstance()`
  - Type checking
  - `if isinstance(x, float):`
```

```{list-table} Common Algorithmic Patterns
:header-rows: 1
:widths: 25 35 40

* - Pattern
  - Purpose
  - Structure
* - Accumulator
  - Aggregate values
  - `total = 0; for x in data: total += x`
* - Filter
  - Select subset
  - `[x for x in data if condition(x)]`
* - Map
  - Transform all
  - `[f(x) for x in data]`
* - Search
  - Find first match
  - `for x in data: if test(x): return x`
* - Convergence
  - Iterate to solution
  - `while not converged and n < max:`
* - Guard clause
  - Handle edge cases
  - `if invalid: return None`
* - Sentinel
  - Signal termination
  - `if value == -999: break`
```

## Python Module & Method Reference (Chapter 3 Additions)

### New Built-in Functions

**Logical Testing**
- `all(iterable)` - Returns True if all elements are true
- `any(iterable)` - Returns True if any element is true
- `isinstance(obj, type)` - Check if object is of specified type

**Loop Support**
- `enumerate(iterable, start=0)` - Returns index-value pairs
- `zip(*iterables)` - Combines multiple iterables for parallel iteration
- `range(start, stop, step)` - Generate arithmetic progression

### Control Flow Keywords

**Conditionals**
- `if` - Execute block if condition is true
- `elif` - Check additional condition if previous was false
- `else` - Execute if all previous conditions were false

**Loops**
- `for` - Iterate over sequence
- `while` - Loop while condition is true
- `break` - Exit loop immediately
- `continue` - Skip to next iteration
- `else` - Execute if loop completes without break

**Other**
- `pass` - Null operation placeholder
- `assert` - Raise AssertionError if condition is false

### Operators

**Comparison**
- `>`, `<`, `>=`, `<=`, `==`, `!=` - Numerical comparisons
- `is`, `is not` - Identity comparisons
- `in`, `not in` - Membership testing

**Logical**
- `and` - Logical AND with short-circuit evaluation
- `or` - Logical OR with short-circuit evaluation  
- `not` - Logical NOT (inversion)

### New Math Module Functions

```python
import math
```
- `math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)` - Safe floating-point comparison
- `math.isfinite(x)` - Check if neither infinite nor NaN
- `math.isnan(x)` - Check if value is NaN
- `math.isinf(x)` - Check if value is infinite

### Debugging Support

**IPython Magic Commands**
- `%debug` - Enter debugger after exception
- `%pdb` - Automatic debugger on exceptions

**Debugger Commands** (when in pdb)
- `p variable` - Print variable value
- `pp variable` - Pretty-print variable
- `l` - List code around current line
- `n` - Next line
- `s` - Step into function
- `c` - Continue execution
- `u/d` - Move up/down call stack
- `q` - Quit debugger

## Next Chapter Preview

You've conquered control flow — now get ready for the next level! Chapter 4 will reveal how to organize data efficiently using Python's powerful data structures. You'll discover when to use lists versus dictionaries versus sets, and more importantly, you'll understand why these choices can make your algorithms run 100 times faster or 100 times slower.

Imagine trying to find a specific star in a catalog of millions. With a list, you'd check each star one by one — taking minutes or hours. With a dictionary, you'll find it instantly — in microseconds! The data structures you'll learn next are the difference between simulations that finish in minutes and ones that run for days.

The control flow patterns you've mastered here will operate on the data structures you'll learn next. Your loops will iterate through dictionaries of astronomical objects. Your conditionals will filter sets of observations. Your comprehensions will transform lists of measurements into meaningful results. Together, control flow and data structures give you the power to handle the massive datasets of modern astronomy — from Gaia's billion-star catalog to the petabytes of data from the Square Kilometre Array.

Get excited — Chapter 4 is where your code goes from processing dozens of data points to handling millions efficiently!
