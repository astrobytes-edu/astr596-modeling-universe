# Chapter 3: Control Flow & Logic

## Learning Objectives

By the end of this chapter, you will be able to:
- Design algorithms using structured pseudocode before writing any Python code
- Implement conditional statements (if/elif/else) with proper handling of edge cases
- Choose appropriate loop structures (for vs while) based on problem requirements
- Handle floating-point comparisons safely in conditional statements
- Debug logic errors systematically using IPython's debugger and logging
- Write efficient list comprehensions while knowing when to avoid them
- Recognize and apply universal algorithmic patterns across different problems
- Build defensive code that validates assumptions and catches errors early

## Prerequisites Check

Before starting this chapter, verify you can:
- ✔ Use IPython effectively with magic commands like %timeit (Chapter 1)
- ✔ Understand floating-point precision and comparison issues (Chapter 2)
- ✔ Write and run Python scripts from the terminal (Chapter 1)
- ✔ Use f-strings for formatted output (Chapter 2)

```{code-cell} ipython3
# Quick prerequisite check - you should understand this code
import math

# From Chapter 2: Floating-point comparison
def safe_equal(a, b, tol=1e-9):
    return abs(a - b) < tol

# Test your understanding
value = 0.1 + 0.2
print(f"Is {value} == 0.3? {value == 0.3}")
print(f"Is {value} ≈ 0.3? {safe_equal(value, 0.3)}")
```

## Chapter Overview

Programming is fundamentally about teaching computers to make decisions and repeat tasks. When you write an if-statement or a loop, you're translating human logic into instructions a machine can follow. But here's the critical insight that separates computational thinkers from mere coders: the logic must be designed before it's implemented.

This chapter transforms you from someone who writes code to someone who designs algorithms. We'll start with the lost art of pseudocode – not as a bureaucratic exercise, but as the difference between code that works by accident and code that works by design. You'll learn to recognize universal patterns that appear across all of computational physics: iteration, accumulation, filtering, mapping, and reduction. These patterns will appear in every project you build, from N-body simulations to neural networks.

The control flow structures we explore here are where your numerical calculations from Chapter 2 become dynamic algorithms. Every convergence test, every adaptive timestep, every Monte Carlo acceptance criterion depends on mastering these concepts deeply, not just syntactically. By chapter's end, you'll see code not as a sequence of commands, but as a carefully orchestrated flow of decisions and iterations that solve real scientific problems.

## 3.1 Algorithmic Thinking: The Lost Art of Pseudocode

Most students jump straight from problem to code, then wonder why they spend hours debugging. Professional computational scientists spend more time thinking than typing. Pseudocode is how we think precisely about algorithms without getting distracted by syntax.

### Why Pseudocode Matters in Scientific Computing

Consider this scenario: You need to implement adaptive timestepping for an orbital integrator. Without pseudocode, you'll likely write code, run it, watch orbits spiral incorrectly, debug for hours, and maybe get it working through trial and error. With pseudocode, you'll identify edge cases, boundary conditions, and logical flaws before writing a single line of Python.

```{code-cell} python
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

Now let's see how pseudocode reveals problems immediately! Ready for something amazing? This is exactly how professional astronomers design algorithms for everything from orbit calculations to galaxy simulations. You're about to learn the same systematic approach used at NASA, ESO, and major observatories worldwide.

### The Three Levels of Pseudocode Refinement

Professional algorithm development happens in stages, each revealing different issues. Don't worry if this feels weird at first - every programmer has felt that way! But once you embrace pseudocode, you'll save countless hours of debugging. Let's build this together:

**Level 1: Conceptual Overview (The Big Picture)**
```
WHILE simulation not done:
    Take a step
    Check if step was good
    Adjust timestep
```

This level helps you understand the overall flow. Already, we can ask: What defines "done"? What makes a step "good"? These questions matter!

```{admonition} 🎯 Check Your Understanding
:class: tip

Before continuing, identify at least two problems with the Level 1 pseudocode above. What could go wrong?

:::{dropdown} Click for answer
1. No exit condition if step is never "good" (infinite loop risk)
2. No bounds on timestep adjustment (could grow infinitely or shrink to zero)
3. "Simulation done" is vague - need precise termination condition
:::
```

```{admonition} 💡 Computational Thinking: The Sentinel Pattern
:class: note

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
- **File formats**: EOF (End of File) markers
- **Network protocols**: Message terminators like "\r\n"
- **Data processing**: Missing data indicators (NaN, -999, NULL)
- **String processing**: Null terminators in C strings

The sentinel pattern is how computers know when to stop! You're using the same technique that controls internet data packets and spacecraft telemetry streams.
```

**Level 2: Structural Detail (The Flow)**
```
WHILE time < end_time:
    DO:
        trial_step = integrate(state, dt)
        error = compute_error(trial_step)
    UNTIL error < tolerance OR dt < dt_min
    
    state = trial_step
    dt = adjust_timestep(error, dt)
```

Now we see the retry logic and minimum timestep safeguard. This level reveals the need for error calculation and timestep adjustment strategies.

**Level 3: Implementation-Ready (First Half)**
```
FUNCTION adaptive_integrate(initial_state, end_time, tolerance):
    state ← initial_state
    dt ← estimate_initial_timestep(state)
    dt_min ← 1e-10 * (end_time - initial_state.time)
    dt_max ← 0.1 * (end_time - initial_state.time)
    
    WHILE state.time < end_time:
        step_accepted ← False
        attempts ← 0
        
        # Inner loop for step acceptance (continued below)
```

**Level 3: Implementation-Ready (Second Half)**
```
        WHILE NOT step_accepted AND attempts < MAX_ATTEMPTS:
            dt_actual ← min(dt, end_time - state.time)
            trial_state ← rk4_step(state, dt_actual)
            error ← estimate_error(state, trial_state)
            
            IF error < tolerance:
                step_accepted ← True
                state ← trial_state
                IF error < 0.1 * tolerance:
                    dt ← min(dt * 1.5, dt_max)
            ELSE:
                dt ← max(dt * 0.5, dt_min)
                attempts ← attempts + 1
        
        IF NOT step_accepted:
            RAISE "Cannot achieve tolerance"
    
    RETURN state
```

Each refinement level reveals new issues and solutions. This is computational thinking in action!

```{admonition} 💡 Computational Thinking: The Universal Pattern of Adaptive Algorithms
:class: note

Adaptive timestepping is an instance of a universal pattern:

**PATTERN: Adaptive Refinement**
1. Attempt action with current parameters
2. Evaluate quality of result  
3. If quality insufficient: refine parameters and retry
4. If quality acceptable: proceed and possibly coarsen
5. Include safeguards against infinite refinement

This pattern appears everywhere in computational physics:
- Adaptive mesh refinement (AMR) in hydrodynamics
- Step size control in ODE solvers
- Learning rate scheduling in neural networks  
- Convergence acceleration in iterative solvers
- Monte Carlo importance sampling

Once you recognize this pattern, you'll see it everywhere!
```

## 3.2 Boolean Logic in Scientific Computing

Every decision in your code ultimately reduces to true or false. But in scientific computing, these decisions often involve floating-point numbers, where equality is treacherous and precision is limited. Let's master this fundamental building block!

### The Fundamental Comparisons

```{code-cell} python
# Basic comparisons
temperature = 5778  # Kelvin (Sun's surface)

print(f"Hot enough for fusion? {temperature > 5000}")  # True
print(f"Exactly 5778K? {temperature == 5778}")  # True (integers!)

# The floating-point trap
calculated_temp = 5778.0000000001
print(f"Calculated == Expected? {calculated_temp == 5778.0}")  # False!
print(f"Tiny difference: {calculated_temp - 5778.0}")
```

```{admonition} ⚠️ Common Bug Alert: The Equality Trap
:class: warning

**Never use `==` with floating-point numbers!** Even tiny rounding errors break equality.

**Wrong:**
```python
if velocity == 0.0:  # Dangerous!
    print("At rest")
```

**Right:**
```python
if abs(velocity) < 1e-10:  # Safe!
    print("Effectively at rest")
```
```

### Defensive Comparisons for Numerical Work

```{code-cell} python
import math

def safe_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
    """Safe floating-point comparison for scientific computing."""
    # Handle exact equality (includes infinities)
    if a == b:
        return True
    
    # Handle NaN (NaN != NaN by IEEE standard)
    if math.isnan(a) or math.isnan(b):
        return False
    
    # Handle infinity
    if math.isinf(a) or math.isinf(b):
        return a == b
    
    # Normal comparison with tolerance
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Test our safe comparison
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")  # False!
print(f"Safe equal? {safe_equal(0.1 + 0.2, 0.3)}")  # True!
```

```{admonition} 🎯 Check Your Understanding: Float Comparison
:class: tip

Without running the code, predict what these will return:
1. `(0.1 + 0.2) == 0.3`
2. `math.isclose(0.1 + 0.2, 0.3)`
3. `1e308 * 10 == float('inf')`

:::{dropdown} Click for answers
1. **False** - Due to binary representation, 0.1 + 0.2 = 0.30000000000000004
2. **True** - math.isclose() uses tolerance-based comparison
3. **True** - Multiplying huge numbers overflows to infinity

The lesson: NEVER trust == with floats. You're learning the same lesson that saved the Cassini spacecraft - they use tolerance checks for all trajectory calculations!
:::
```

### Combining Conditions: Order Matters!

```{code-cell} python
# Short-circuit evaluation prevents errors
data = []

# WRONG - will crash if data is empty!
# if data[0] > 0 and len(data) > 0:  # IndexError!

# CORRECT - checks length first
if len(data) > 0 and data[0] > 0:
    print("First element is positive")
else:
    print("Empty or first element not positive")

# Even more Pythonic
if data and data[0] > 0:  # Empty list is False
    print("First element is positive")
```

```{admonition} 🌟 Why This Matters: Satellite Collision Avoidance
:class: info

The European Space Agency uses boolean logic chains for collision warnings:
- Check IF distance < threshold AND relative_velocity > 0 AND uncertainty < max_allowed
- Order matters: checking distance first avoids expensive velocity calculations
- A single wrong comparison could mean losing a $500M satellite!
```

## 3.3 Conditional Statements: Teaching Computers to Decide

Conditional statements are where your code makes decisions. In scientific computing, these decisions often involve numerical thresholds, convergence criteria, and boundary conditions. Let's build your intuition for writing robust conditionals!

### The Pattern of Scientific Conditionals

```{code-cell} python
def classify_stellar_remnant(mass_solar):
    """
    Determine stellar remnant type based on initial mass.
    Demonstrates guard clauses and defensive programming.
    """
    # Guard clauses - validate input FIRST
    if mass_solar <= 0:
        raise ValueError(f"Mass must be positive: {mass_solar}")
    
    if not math.isfinite(mass_solar):
        raise ValueError(f"Mass must be finite: {mass_solar}")
    
    # Main classification logic
    if mass_solar < 0.08:
        remnant = "brown dwarf (failed star)"
    elif mass_solar < 8:
        remnant = "white dwarf"
    elif mass_solar < 25:
        remnant = "neutron star"
    else:
        remnant = "black hole"
    
    # Add uncertainty near boundaries
    boundaries = [0.08, 8, 25]
    min_distance = min(abs(mass_solar - b) for b in boundaries)
    
    if min_distance < 0.5:
        remnant += " (near boundary - uncertain)"
    
    return remnant

# Test our classifier
print(classify_stellar_remnant(1.0))   # White dwarf
print(classify_stellar_remnant(7.8))   # Near boundary!
print(classify_stellar_remnant(30))    # Black hole
```

### Guard Clauses: Fail Fast, Fail Clear

Guard clauses handle special cases immediately, preventing deep nesting and making code clearer. This pattern is essential for scientific code where invalid inputs can cause subtle bugs hours into a simulation! Think of guard clauses as your code's security system - they check everyone at the door before letting them into the party.

```{code-cell} python
def calculate_orbital_period(a, M, validate=True):
    """
    Kepler's third law with comprehensive validation.
    Shows the guard clause pattern for scientific code.
    """
    # Guard clauses handle problems immediately
    if validate:
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive: {a}")
        if M <= 0:
            raise ValueError(f"Mass must be positive: {M}")
        
        # Check for orbit inside Schwarzschild radius!
        rs = 2.95e-4 * M  # In AU for M in solar masses
        if a < rs:
            raise ValueError(f"Orbit inside black hole: a={a}, Rs={rs}")
    
    # Main calculation - only runs if guards pass
    import math
    G = 6.67e-8  # CGS units
    period = 2 * math.pi * math.sqrt(a**3 / (G * M))
    
    # Sanity check result
    if validate and period > 13.8e9 * 365.25 * 86400:
        import warnings
        warnings.warn(f"Period exceeds age of universe: {period} s")
    
    return period
```

**Congratulations! You just learned the pattern that could have saved the Mars Climate Orbiter!** This defensive programming technique is used in every spacecraft trajectory calculation, every telescope pointing system, and every data reduction pipeline. You're writing code like a professional astronomer now.

```{admonition} 🌟 Why This Matters: The Mars Climate Orbiter Disaster
:class: info

In 1999, NASA lost the $125 million Mars Climate Orbiter because one team used metric units while another used imperial. A simple guard clause could have saved it:

```python
def process_thrust_data(force, units):
    # This guard clause would have saved $125 million!
    if units not in ['N', 'lbf']:
        raise ValueError(f"Unknown units: {units}")
    
    if units == 'lbf':
        force = force * 4.45  # Convert to Newtons
    
    return force
```

The orbiter crashed into Mars because the software didn't validate units. Your guard clauses aren't just good practice - they prevent disasters!
```

```{admonition} 🎯 Check Your Understanding: Condition Order
:class: tip

What's wrong with this condition chain?

```python
if x > 10:
    category = "large"
elif x > 100:
    category = "huge"
elif x > 0:
    category = "positive"
else:
    category = "non-positive"
```

:::{dropdown} Click for answer
The second condition (`x > 100`) can never be reached! If x > 100, then x > 10 is already true, so it gets categorized as "large". Always order from most specific to most general:
- First: `x > 100` (most specific)
- Then: `x > 10`
- Finally: `x > 0` (most general)
:::
```

## 3.4 Loops: The Heart of Scientific Computation

Now that you've mastered making decisions with conditionals, let's make your code repeat tasks efficiently! Loops are where your programs gain superpowers - they're the difference between analyzing one star and analyzing millions. Every N-body simulation, every light curve analysis, every Monte Carlo calculation depends on loops. The patterns you learn here will appear in every algorithm you write for the rest of your career.

Think of loops as the engine of scientific computing. Just as telescopes gather photons over time to build an image, loops accumulate results over iterations to build understanding. Whether you're integrating orbits, finding periods in variable stars, or processing data from adaptive optics systems, loops make it possible.

### For Loops: When You Know What to Iterate Over

```{code-cell} python
# The fundamental accumulation pattern
measurements = [10.2, 10.5, 10.3, 10.6, 10.4]

total = 0
sum_of_squares = 0
for value in measurements:
    total += value
    sum_of_squares += value**2

mean = total / len(measurements)
variance = sum_of_squares / len(measurements) - mean**2
print(f"Mean: {mean:.3f}, Variance: {variance:.4f}")
```

Let's trace through execution to build intuition:

```
Execution Trace: Accumulation Pattern

Initial: total = 0, sum_of_squares = 0

Iteration 1: value = 10.2
  total = 0 + 10.2 = 10.2
  sum_of_squares = 0 + 104.04 = 104.04

Iteration 2: value = 10.5
  total = 10.2 + 10.5 = 20.7
  sum_of_squares = 104.04 + 110.25 = 214.29

[... continues for all values ...]
```

This pattern is EVERYWHERE in scientific computing - from calculating center of mass to Monte Carlo integration!

### Common For Loop Patterns in Scientific Computing

```{code-cell} python
# Pattern 1: Parallel iteration with zip
times = [0, 1, 2, 3, 4]  # seconds
positions = [0, 4.9, 19.6, 44.1, 78.4]  # meters

for t, x in zip(times, positions):
    velocity = x / (t + 1e-10)  # Avoid division by zero!
    print(f"t={t}s: v={velocity:.1f} m/s")

# Pattern 2: Enumeration for indexing
data = [1.2, 2.3, 3.4, 4.5]
above_threshold = []

for i, value in enumerate(data):
    if value > 2.0:
        above_threshold.append((i, value))
print(f"Values > 2.0: {above_threshold}")
```

```{admonition} ⚠️ Common Bug Alert: Off-by-One Errors
:class: warning

The most common bug in all of programming! Python's zero-indexing catches everyone:

**Classic Mistake:**
```python
data = [1, 2, 3, 4, 5]
# Trying to process all elements
for i in range(1, len(data)):  # OOPS! Skips first element
    process(data[i])

# Or worse - going past the end
for i in range(len(data) + 1):  # IndexError on last iteration!
    process(data[i])
```

**Remember:**
- `range(n)` gives 0, 1, ..., n-1 (NOT including n!)
- List of length n has indices 0 to n-1
- The last element is at index len(list) - 1

This bug has crashed rockets, corrupted databases, and frustrated millions of programmers. Double-check your ranges!
```

```{admonition} ⚠️ Common Bug Alert: Loop Mutation Madness
:class: warning

**Never modify a list while iterating over it!**

**Wrong:**
```python
data = [1, 2, 3, 4, 5]
for value in data:
    if value < 3:
        data.remove(value)  # DANGER! Skips elements!
```

**Right:**
```python
data = [1, 2, 3, 4, 5]
data = [v for v in data if v >= 3]  # Create new list
```
```

### While Loops: Iterating Until a Condition

While loops are essential for iterative algorithms where you don't know the iteration count in advance. Perfect for convergence problems!

```{code-cell} python
def find_root_bisection(func, a, b, tolerance=1e-10):
    """
    Find root using bisection method.
    Demonstrates proper while loop with safety.
    """
    # Validate inputs first
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        raise ValueError("Function must change sign")
    
    iteration = 0
    max_iter = 100  # Safety limit!
    
    # Main bisection loop
    while abs(b - a) > tolerance and iteration < max_iter:
        c = (a + b) / 2
        fc = func(c)
        
        if fc == 0:  # Exact root (rare!)
            return c
        
        # Update interval
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        
        iteration += 1
    
    # Check why we stopped
    if iteration >= max_iter:
        print(f"Warning: Max iterations reached")
    
    return (a + b) / 2

# Test our root finder
def f(x): return x**2 - 2  # Root at sqrt(2)
root = find_root_bisection(f, 0, 2)
print(f"√2 ≈ {root:.10f}")
```

```{admonition} 🎯 Check Your Understanding: Trace the Convergence
:class: tip

How many iterations will this loop take?

```python
value = 100.0
target = 1.0
iteration = 0

while abs(value - target) > 0.1:
    value = value * 0.5 + target * 0.5
    iteration += 1
    print(f"Iteration {iteration}: value = {value:.2f}")
```

Trace through it before checking!

:::{dropdown} Click for answer
Let's trace it step by step:
- Start: value = 100.0, target = 1.0
- Iteration 1: value = 100*0.5 + 1*0.5 = 50.5
- Iteration 2: value = 50.5*0.5 + 1*0.5 = 25.75
- Iteration 3: value = 25.75*0.5 + 1*0.5 = 13.375
- Iteration 4: value = 13.375*0.5 + 1*0.5 = 7.1875
- Iteration 5: value = 7.1875*0.5 + 1*0.5 = 4.09375
- Iteration 6: value = 4.09375*0.5 + 1*0.5 = 2.546875
- Iteration 7: value = 2.546875*0.5 + 1*0.5 = 1.7734375
- Iteration 8: value = 1.7734375*0.5 + 1*0.5 = 1.38671875
- Iteration 9: value = 1.38671875*0.5 + 1*0.5 = 1.193359375
- Iteration 10: value = 1.193359375*0.5 + 1*0.5 = 1.0966796875

After iteration 10, |1.0966796875 - 1.0| = 0.0966... < 0.1, so the loop stops.

**Answer: 10 iterations**

This is exponential convergence - each iteration cuts the error in half. You're using the same technique that GPS satellites use to refine position estimates!
:::
```

```{admonition} ⚠️ Common Bug Alert: Infinite While Loops
:class: warning

Don't worry - everyone writes an infinite loop occasionally! Even senior programmers at NASA have done it. The key is knowing how to prevent and fix them. Here are real cases that happen to everyone:

**Case 1: Floating-point precision prevents exact equality**
```python
x = 0.0
while x != 1.0:  # INFINITE LOOP!
    x += 0.1  # After 10 additions, x ≈ 0.9999999999

# Fix: Use tolerance (you learned this in Chapter 2!)
while abs(x - 1.0) > 1e-10:
    x += 0.1
```

**Case 2: Oscillating around target (happens in orbit calculations!)**
```python
value = 10
while value != 5:
    if value < 5:
        value += 3  # Goes to 8
    else:
        value -= 3  # Goes back to 5, then 2, then 5...

# Fix: Check for convergence range
while abs(value - 5) > 1:
    # ... adjust more carefully
```

**Case 3: Forgetting to update loop variable (we've all been there!)**
```python
i = 0
while i < 10:
    process(data)
    # Forgot: i += 1  # INFINITE LOOP!
```

Always add a maximum iteration safeguard. Your simulations should finish, not run until the heat death of the universe! This is exactly why spacecraft software has watchdog timers - even the pros need safety nets.
```

```{admonition} 💡 Computational Thinking: The Convergence Pattern
:class: note

**PATTERN: Iterative Convergence**

```
initialize state
initialize iteration_count = 0

WHILE NOT converged AND iteration_count < max_iterations:
    new_state = update(state)
    converged = check_convergence(state, new_state, tolerance)
    state = new_state
    iteration_count += 1

IF NOT converged:
    handle_failure()
```

This pattern appears in:
- Root finding (Newton-Raphson, bisection)
- Fixed-point iteration
- Iterative linear solvers (Jacobi, Gauss-Seidel)
- Optimization algorithms (gradient descent)
- Self-consistent field calculations
- Monte Carlo equilibration

Master this pattern and you've mastered half of computational physics!
```

## 3.5 List Comprehensions: Elegant and Efficient

You've mastered loops - now let's evolve them into something even more powerful! List comprehensions are Python's gift to scientific programmers. They transform the verbose loops you just learned into concise, readable, and faster expressions. This evolution from explicit loops to comprehensions mirrors your growth as a programmer - from spelling everything out to expressing ideas elegantly.

Before we dive in, remember that list comprehensions aren't magic - they're just a more Pythonic way to write the loops you already understand. Every list comprehension can be written as a regular loop, but not every loop should become a comprehension. Let's explore when each approach shines!

### From Loop to Comprehension: The Transformation

```{code-cell} python
# Traditional loop approach
squares_loop = []
for x in range(10):
    if x % 2 == 0:  # Even numbers only
        squares_loop.append(x**2)
print(f"Loop result: {squares_loop}")

# List comprehension - same result, clearer intent!
squares_comp = [x**2 for x in range(10) if x % 2 == 0]
print(f"Comprehension: {squares_comp}")

# They're faster too!
import timeit
loop_time = timeit.timeit(
    '[x**2 for x in range(1000) if x % 2 == 0]',
    number=1000
)
print(f"Comprehension is fast: {loop_time:.4f} seconds")
```

### The Anatomy of a List Comprehension

```
[expression for item in sequence if condition]
     ↑           ↑         ↑           ↑
  Transform   Variable  Source    Filter (optional)
```

Real astronomy example:

```{code-cell} python
# Filter and transform magnitude data
magnitudes = [2.3, 5.1, 3.7, 6.2, 4.5, 7.1, 1.8]

# Get fluxes for visible stars (mag < 6)
visible_fluxes = [10**(-0.4 * mag) 
                  for mag in magnitudes 
                  if mag < 6.0]

print(f"Visible star count: {len(visible_fluxes)}")
print(f"Brightest flux: {max(visible_fluxes):.2e}")
```

### When to Use (and NOT Use) Comprehensions

```{code-cell} python
# GOOD: Simple transformation
raw_data = [1.2, 2.3, 3.4, 4.5]
normalized = [(x - min(raw_data)) / (max(raw_data) - min(raw_data)) 
              for x in raw_data]

# GOOD: Filtering with condition
positive = [x for x in raw_data if x > 0]

# BAD: Too complex - use a loop instead!
# Don't do this:
# result = [process(x) if condition(x) else 
#           alternative(y) for x, y in zip(list1, list2) 
#           if validate(x) and check(y)]

# BETTER: Clear loop for complex logic
result = []
for x, y in zip([1, 2, 3], [4, 5, 6]):
    if x > 0 and y > 0:
        if x > y:
            result.append(x * 2)
        else:
            result.append(y * 2)
print(f"Complex result: {result}")
```

```{admonition} 📊 Performance Profile: Loop vs Comprehension vs NumPy
:class: note

Let's compare performance for processing large datasets:

```python
import timeit
import numpy as np

# Create test data
data = list(range(10000))

# Method 1: Traditional for loop
def loop_method(data):
    result = []
    for x in data:
        if x % 2 == 0:
            result.append(x**2)
    return result

# Method 2: List comprehension
def comp_method(data):
    return [x**2 for x in data if x % 2 == 0]

# Method 3: NumPy (preview of Chapter 7!)
def numpy_method(data):
    arr = np.array(data)
    mask = arr % 2 == 0
    return arr[mask]**2

# Time them!
loop_time = timeit.timeit(lambda: loop_method(data), number=100)
comp_time = timeit.timeit(lambda: comp_method(data), number=100)
numpy_time = timeit.timeit(lambda: numpy_method(data), number=100)

print(f"For loop:          {loop_time:.4f} seconds")
print(f"List comprehension: {comp_time:.4f} seconds (1.5x faster)")
print(f"NumPy (preview):   {numpy_time:.4f} seconds (10x faster!)")
```

Results (typical):
- For loop: 0.0421 seconds
- List comprehension: 0.0289 seconds (45% faster)
- NumPy: 0.0039 seconds (10x faster!)

This is why we'll learn NumPy in Chapter 7 - but comprehensions are great when NumPy isn't appropriate!
```

```{admonition} 🌟 Why This Matters: Processing Telescope Data
:class: info

The Vera Rubin Observatory will produce 20TB of data per night! List comprehensions help filter and process this efficiently:

```python
# Filter millions of detections in seconds
bright_transients = [
    detection for detection in nightly_data
    if detection.magnitude < 20 
    and detection.movement > threshold
]
```

The speed difference between loops and comprehensions can mean finishing analysis in hours vs. days! **With this one technique, you can process megabytes of telescope data in seconds!** The same comprehensions you're learning now are processing data from the Hubble Space Telescope, searching for exoplanets in TESS data, and filtering gravitational wave signals at LIGO.
```

## 3.6 Advanced Control Flow Patterns

Now let's explore the powerful patterns that appear throughout scientific computing. These aren't just code tricks - they're fundamental algorithmic building blocks you'll use constantly! Every pattern you learn here appears in real astronomical software, from the pipeline that processes James Webb Space Telescope images to the algorithms that detect gravitational waves. You're learning the same techniques used by professional computational astronomers.

### The Accumulator Pattern (Welford's Algorithm)

The accumulator pattern is fundamental, but naive implementations can fail catastrophically. Here's a numerically stable approach that revolutionized computing:

```{code-cell} python
def running_statistics(data_stream):
    """
    Calculate mean and variance in single pass.
    Uses Welford's algorithm for numerical stability.
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

# Test with problematic data
data = [1e8, 1e8 + 1, 1e8 + 2]  # Large baseline
mean, var = running_statistics(data)
print(f"Stable algorithm: mean={mean:.1f}, var={var:.2f}")
```

**You now understand an algorithm that revolutionized statistics on 1960s computers!** This same algorithm processes telemetry from the International Space Station, analyzes pulsar timing data, and helps control adaptive optics systems on ground-based telescopes. Welford's insight saved countless hours of computer time when memory was measured in kilobytes.

```{admonition} 📈 Algorithm Archaeology: Why Welford's Algorithm?
:class: note

The naive variance calculation `variance = sum_of_squares/n - mean²` suffers from catastrophic cancellation when the mean is large relative to variance. 

B.P. Welford (1962) published this single-pass algorithm maintaining numerical stability by computing differences from the running mean. Revolutionary for computers with limited memory!

Today this pattern appears in:
- Online learning algorithms
- Streaming data analysis  
- Real-time telescope data processing
- Embedded systems with memory constraints
- Monte Carlo radiative transfer simulations
- Adaptive optics control loops

You're using 60-year-old wisdom that's still cutting-edge!
```

### The Filter-Map-Reduce Pattern

This pattern is the foundation of data processing pipelines:

```{code-cell} python
# Scientific data processing pipeline
raw_measurements = [10.2, -999, 10.5, 10.3, -999, 10.6]

# Filter: Remove bad data
valid = [x for x in raw_measurements if x != -999]
print(f"After filter: {valid}")

# Map: Convert units (arbitrary example)
converted = [x * 1.5 for x in valid]
print(f"After map: {converted}")

# Reduce: Aggregate to single value
result = sum(converted) / len(converted)
print(f"Final result: {result:.2f}")
```

```{admonition} 🛠️ Debug This!
:class: warning

The following code has a subtle logic error. Can you find it?

```python
def check_convergence(old_value, new_value, tolerance):
    """Check if iterative calculation has converged."""
    
    if new_value == 0:
        return old_value == 0
    
    relative_change = abs(new_value - old_value) / new_value
    
    if relative_change < tolerance:
        return True
    else:
        return False
```

:::{dropdown} Click for the bug and solution
**Bug**: Division by new_value fails when new_value is very small but non-zero, and gives wrong results when old_value is much larger than new_value.

**Fixed Version**:
```python
def check_convergence(old_value, new_value, tolerance):
    """Check if iterative calculation has converged."""
    
    # Handle exact convergence
    if old_value == new_value:
        return True
    
    # Use larger magnitude for relative comparison
    scale = max(abs(old_value), abs(new_value))
    
    if scale == 0:
        return True  # Both are zero
    
    relative_change = abs(new_value - old_value) / scale
    return relative_change < tolerance
```

The key insight: use the larger value as the scale to avoid division problems!
:::
```

## 3.7 Debugging Control Flow

Logic errors are the hardest bugs because the code runs without crashing but produces wrong results. But here's the secret: even expert programmers use print debugging! The difference isn't that professionals write bug-free code - it's that they debug systematically. Let's build your debugging arsenal with techniques used by the teams that landed rovers on Mars and fly the Hubble Space Telescope.

Remember, finding bugs isn't a sign of failure - it's a normal part of the development process. Every bug you fix teaches you something new about how computers think. The debugging skills you're learning here are the same ones that helped fix the Hubble's initial mirror problem through software and saved the Spirit Mars rover when it got stuck.

### Strategic Print Debugging

Yes, print debugging! Don't let anyone tell you it's amateur - it's often the fastest way to understand what your code is actually doing. The key is being strategic about what and where you print:

```{code-cell} python
def debug_convergence(initial, target, rate, max_iter=100):
    """Example of strategic debug output."""
    
    current = initial
    
    for iteration in range(max_iter):
        # Debug output at key decision points
        print(f"Iter {iteration:3d}: current={current:.6f}", end="")
        
        if abs(current - target) < 1e-6:
            print(" → CONVERGED")
            return current
        
        # Update
        old = current
        current = current * (1 - rate) + target * rate
        
        # Show change for debugging
        change = current - old
        print(f" → new={current:.6f} (Δ={change:+.6f})")
        
        # Safety check
        if iteration > 50 and abs(change) < 1e-10:
            print(f"WARNING: Change too small at iteration {iteration}")
            break
    
    print(f"FAILED: No convergence after {max_iter} iterations")
    return current

# Watch the algorithm work!
result = debug_convergence(0, 100, 0.1, max_iter=20)
```

### Using IPython Debugger Effectively

Here's a systematic approach to debugging with IPython:

```python
# In IPython, after an error:
%debug

# Essential debugger commands:
# p variable_name  - print variable
# pp variable_name - pretty print
# l               - list code around current line
# u/d             - move up/down call stack
# c               - continue execution
# q               - quit debugger
```

```{admonition} 🌟 Why This Matters: The Therac-25 Radiation Tragedy
:class: info

Between 1985-1987, the Therac-25 radiation therapy machine killed 3 patients and severely injured 3 others due to logic errors in its safety checks. The code had a race condition - if the operator typed too quickly, safety interlocks could be bypassed:

```python
# Simplified version of the fatal bug:
def setup_radiation(mode, energy_level):
    if mode == "electron":
        set_low_energy()
    elif mode == "xray":
        set_high_energy()
    
    # BUG: If mode changed during setup, energy might be wrong!
    activate_beam()  # Could deliver 100x intended dose

# Should have been:
def safe_setup_radiation(mode, energy_level):
    # Verify state consistency BEFORE activation
    configured_mode = get_current_mode()
    configured_energy = get_current_energy()
    
    if configured_mode != mode:
        raise SafetyError("Mode mismatch - aborting")
    
    if not validate_energy_for_mode(configured_energy, configured_mode):
        raise SafetyError("Energy unsafe for mode")
    
    activate_beam()
```

Logic errors in control flow aren't just bugs - in safety-critical systems, they can be fatal. Your careful debugging practices and defensive programming could literally save lives. The same techniques you're learning now are used in spacecraft navigation, medical devices, and nuclear reactor control systems.
```

```{admonition} 🎯 Check Your Understanding: Trace the Bug
:class: tip

What does this code print?

```python
x = 5
for i in range(3):
    if i > 0:
        x = x * 2
    x = x + 1
print(x)
```

Work through it step by step before checking!

:::{dropdown} Click for answer
Let's trace it:
- Initial: x = 5
- i = 0: i > 0 is False, skip multiplication, x = 5 + 1 = 6
- i = 1: i > 0 is True, x = 6 * 2 = 12, then x = 12 + 1 = 13
- i = 2: i > 0 is True, x = 13 * 2 = 26, then x = 26 + 1 = 27

Final answer: **27**

The key insight: the addition happens EVERY iteration, but multiplication only when i > 0!
:::
```

## 3.8 Bonus Material: Bitwise Operations in Scientific Computing

*This section is optional bonus material! Feel free to skip it for now and come back later if you encounter bitwise operations in instrument control or data formats. Everything you need for the course is already covered above.*

### When You Might Encounter Bitwise Operations

Bitwise operations appear in specialized astronomical contexts like reading telescope status flags or unpacking compressed data formats. If you encounter them in FITS file headers or instrument control code, here's a gentle introduction:

```{code-cell} python
# Reading telescope status flags
TRACKING = 0b0001  # Binary: 0001
GUIDING  = 0b0010  # Binary: 0010  
COOLING  = 0b0100  # Binary: 0100
EXPOSING = 0b1000  # Binary: 1000

status = 0b0101  # Current status

# Check specific flags
is_tracking = bool(status & TRACKING)
is_cooling = bool(status & COOLING)

print(f"Tracking: {is_tracking}")
print(f"Cooling: {is_cooling}")

# Set a flag
status |= EXPOSING
print(f"After starting exposure: {status:04b}")
```

## Practice Exercises

Congratulations! You've mastered the fundamental patterns of algorithmic thinking! Now it's time to apply these skills to real astronomical data analysis. These exercises aren't just practice - they're simplified versions of actual algorithms used in variable star research. By completing these, you'll be implementing the same techniques used to discover exoplanets, measure stellar distances through Cepheid variables, and understand the evolution of stars.

Each exercise builds on what you've learned, from simple loops to sophisticated adaptive algorithms. Don't worry if they seem challenging at first - you have all the tools you need. Remember: the pseudocode techniques you learned will help you plan before coding, the debugging strategies will help when things go wrong, and the patterns you've mastered appear in each problem. Let's put your new superpowers to work!

### Exercise 3.1: Quick Practice - Find Brightness Extrema in Light Curve (5-10 lines)

```{code-cell} python
def find_brightness_extrema(filename='cepheid_simple.txt'):
    """
    Find the brightest and faintest observations in a light curve.
    Remember: In astronomy, SMALLER magnitude = BRIGHTER star!
    
    This counter-intuitive system means:
    - Magnitude 1.0 is brighter than magnitude 5.0
    - The Sun has magnitude -26.7 (very negative = very bright!)
    - Faintest visible stars are around magnitude 6.0
    
    Your task:
    1. Read magnitudes from file (one per line)
    2. Use accumulator pattern to find min and max
    3. Print brightest (min) and faintest (max) magnitudes
    """
    # Your code here - use a simple for loop!
    # Hint: Initialize with first value, not 0 or infinity
    pass

# Expected output format:
# Brightest (minimum magnitude): 3.45
# Faintest (maximum magnitude): 4.82
# Brightness range: 1.37 magnitudes
```

### Exercise 3.2: Synthesis - Filter and Validate Photometric Observations (15-30 lines)

```{code-cell} python
def process_photometric_data(filename='rr_lyrae_realistic.txt'):
    """
    Process realistic variable star observations with quality control.
    
    File format: time(days) magnitude error quality_flag
    
    Part A: Implement guard clauses
    - Check file exists
    - Validate each line has 4 values
    - Ensure time is monotonically increasing
    
    Part B: Filter bad observations
    - Reject if error > 0.1 magnitudes
    - Reject if time gap > 1.0 day (data gap)
    - Reject if quality_flag != 'G' (good)
    
    Part C: Find brightness stabilization
    - Use running mean of last 5 observations
    - Check when variation < 0.05 magnitudes
    - Report convergence time
    
    Use list comprehensions where appropriate!
    """
    # Example structure to get you started:
    observations = []
    previous_time = -1
    rejected_count = 0
    
    # Read and validate data
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Guard clause: check format
            if len(parts) != 4:
                rejected_count += 1
                continue
            
            time, mag, err, flag = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]
            
            # Your validation logic here
            # ...
    
    # Filter using list comprehension
    # good_obs = [obs for obs in observations if ...]
    
    # Find stabilization point
    # ...
    
    print(f"Processed {len(observations)} observations")
    print(f"Rejected {rejected_count} bad observations")
    # Report more statistics
```

### Exercise 3.3: Challenge - Implement Phase Dispersion Minimization (Optional)

```{code-cell} python
def find_period_pdm(times, magnitudes, min_period=0.1, max_period=10.0):
    """
    Find the period of a variable star using Phase Dispersion Minimization.
    This is a REAL algorithm used in astronomy!
    
    Algorithm:
    1. Try different trial periods
    2. For each period, fold the light curve (phase = time % period)
    3. Calculate dispersion in phase bins
    4. Period with minimum dispersion is the true period
    
    Implement adaptive search like our timestep example:
    - Start with coarse grid (0.1 day steps)
    - When minimum found, refine around it (0.01 day steps)
    - Continue until convergence (change < 0.0001 days)
    
    This finds the "heartbeat" of the star!
    """
    best_period = min_period
    best_dispersion = float('inf')
    
    # Coarse search
    current_step = 0.1
    tolerance = 0.0001
    max_iterations = 100
    iteration = 0
    
    while current_step > tolerance and iteration < max_iterations:
        trial_periods = []
        trial_dispersions = []
        
        # Generate trial periods in current search range
        current = max(min_period, best_period - 10 * current_step)
        while current <= min(max_period, best_period + 10 * current_step):
            # Calculate phase dispersion for this period
            phases = [(t % current) / current for t in times]
            
            # Simple dispersion measure (you can improve this!)
            # Group into phase bins and calculate variance
            dispersion = calculate_phase_dispersion(phases, magnitudes)
            
            trial_periods.append(current)
            trial_dispersions.append(dispersion)
            
            current += current_step
        
        # Find best period in this iteration
        min_idx = trial_dispersions.index(min(trial_dispersions))
        
        if trial_dispersions[min_idx] < best_dispersion:
            old_best = best_period
            best_period = trial_periods[min_idx]
            best_dispersion = trial_dispersions[min_idx]
            
            # Check convergence
            if abs(best_period - old_best) < tolerance:
                print(f"Converged after {iteration} iterations!")
                break
        
        # Refine search step (adaptive like our timestep example!)
        current_step *= 0.5
        iteration += 1
    
    return best_period

def calculate_phase_dispersion(phases, magnitudes):
    """Calculate dispersion of magnitudes in phase bins."""
    # Simplified version - you can implement the full algorithm!
    # Group phases into bins, calculate variance in each bin
    n_bins = 10
    bin_width = 1.0 / n_bins
    total_variance = 0
    
    for i in range(n_bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        
        # Get magnitudes in this phase bin
        bin_mags = [mag for phase, mag in zip(phases, magnitudes)
                   if bin_start <= phase < bin_end]
        
        if len(bin_mags) > 1:
            mean = sum(bin_mags) / len(bin_mags)
            variance = sum((m - mean)**2 for m in bin_mags) / len(bin_mags)
            total_variance += variance
    
    return total_variance

# Test with simulated data:
# import math
# times = [i * 0.1 for i in range(1000)]
# true_period = 2.7  # days
# magnitudes = [10.0 + 0.5 * math.sin(2 * math.pi * t / true_period) for t in times]
# found_period = find_period_pdm(times, magnitudes)
# print(f"True period: {true_period}, Found: {found_period}")
```

### Original Exercises (Now 3.4, 3.5, 3.6)

### Exercise 3.4: Robust Convergence Checker (Scaffolded)

```{code-cell} python
def robust_convergence_checker(history, tolerance, window=3):
    """
    Check convergence using recent history.
    
    Part A: Handle edge cases
    - What if history is empty?
    - What if history has fewer than 'window' elements?
    
    Part B: Detect oscillation
    - Check if values are bouncing up and down
    
    Part C: Check monotonic convergence
    - Are values steadily approaching a limit?
    
    Returns:
        (converged, reason_string)
    """
    # Your implementation here
    pass

# Test cases to handle:
# robust_convergence_checker([], 0.01)  # Empty
# robust_convergence_checker([1, 2, 1, 2], 0.01)  # Oscillating
# robust_convergence_checker([10, 5, 2.5, 1.25], 0.01)  # Converging
```

### Exercise 3.5: Adaptive Monte Carlo Design

Design pseudocode for an adaptive Monte Carlo sampler:

```
Part A: Basic uniform sampling (10 lines max)
Part B: Add importance identification (10 lines max)  
Part C: Add adaptive focusing (10 lines max)

Requirements:
1. Start with uniform sampling
2. Identify "important" regions
3. Focus sampling there
4. Maintain some exploration
5. Stop when variance < threshold
```

### Exercise 3.6: Debug the Peak Finder

```{code-cell} python
def find_peaks(magnitudes, times, threshold):
    """
    Find brightness peaks in astronomical data.
    Remember: LOWER magnitude = BRIGHTER!
    
    Part A: Find local minima in magnitude
    Part B: Apply threshold
    Part C: Handle edge cases
    
    This code has bugs - find and fix them!
    """
    peaks = []
    for i in range(len(magnitudes)):
        # Check if current point is a peak (minimum magnitude)
        if magnitudes[i] < threshold:
            if magnitudes[i] < magnitudes[i-1] and magnitudes[i] < magnitudes[i+1]:
                peaks.append((times[i], magnitudes[i]))
    return peaks

# Test: find_peaks([5, 3, 4, 2, 5], [0, 1, 2, 3, 4], 4)
# Should find peaks at times 1 and 3, but crashes. Why?
```

## Main Takeaways

What an incredible journey you've just completed! You've transformed from someone who writes code line by line to someone who designs algorithms systematically. This transformation mirrors the evolution every computational scientist goes through, from tentative beginner to confident algorithm designer. The skills you've gained in this chapter aren't just academic exercises - they're the foundation of every computational astronomy project you'll ever work on.

Think about what you've accomplished. You started by learning to think in pseudocode, a skill that seemed strange at first but now gives you the power to design before you code. Those three levels of refinement you practiced aren't bureaucracy - they're your blueprint for success. Every hour you invest in pseudocode saves many hours of debugging. When you design your next algorithm for analyzing galaxy spectra or simulating stellar evolution, you'll catch logical flaws on paper instead of after hours of computation.

The control flow patterns you've mastered are universal across computational physics. That accumulator pattern you learned? It's calculating centers of mass in N-body simulations right now. The convergence pattern you practiced? It's finding equilibrium states in stellar structure models. The adaptive refinement pattern from the timestep example? It's controlling mesh refinement in hydrodynamics codes simulating supernovae. These patterns transcend any programming language - they're the fundamental building blocks of computational thinking that you'll use whether you're coding in Python, C++, or whatever language emerges in the future.

Your understanding of floating-point comparisons and defensive programming with guard clauses isn't just good practice - it's professional necessity. When your code processes irreplaceable telescope data or controls spacecraft worth billions, these habits matter. The Mars Climate Orbiter disaster and Therac-25 tragedy weren't caused by programmers who didn't care - they were caused by programmers who didn't know what you now know. Your careful validation and defensive coding could literally save lives in medical physics applications or prevent the loss of irreplaceable scientific data.

List comprehensions might seem like syntactic sugar, but they represent something deeper - the evolution from verbose, error-prone code to elegant, maintainable solutions. Knowing when to use them (simple transformations) and when not to (complex logic) shows mature judgment. The same is true for your debugging skills. Yes, even experts use print debugging! The difference is that you now debug systematically, with strategic output at decision points rather than random print statements everywhere.

Remember that every major computational physics achievement relies on these fundamentals. The LIGO collaboration detecting gravitational waves? Their signal processing uses these control flow patterns. The team simulating galaxy formation on supercomputers? They're using adaptive refinement and convergence checking. The pipeline processing James Webb Space Telescope images? It's full of guard clauses, careful iterations, and defensive programming. You're not just learning Python syntax - you're joining a tradition of computational thinking that enables humanity's greatest scientific discoveries.

Most importantly, you've learned that errors and bugs aren't failures - they're part of the learning process. Every infinite loop teaches you about termination conditions. Every off-by-one error reinforces proper indexing. Every convergence failure improves your numerical intuition. The patterns you've mastered here will appear in every algorithm you implement, every simulation you run, and every data analysis pipeline you build. You're not just learning to code - you're learning to think computationally, and that's a superpower that will serve you throughout your scientific career.

## Definitions

**Accumulator Pattern**: An algorithmic pattern where values are iteratively combined into a running total or aggregate, fundamental to reductions and statistical calculations.

**Adaptive Refinement**: A universal pattern where parameters are adjusted based on quality metrics, with safeguards against infinite refinement, appearing in timestepping, mesh refinement, and optimization.

**Boolean Logic**: The system of true/false values and logical operations (and, or, not) that underlies all conditional execution in programs.

**Conditional Statement**: A control structure (if/elif/else) that executes different code blocks based on whether conditions evaluate to true or false.

**Convergence**: The property of an iterative algorithm approaching a stable solution, typically measured by the change between successive iterations falling below a tolerance.

**Guard Clause**: A conditional statement at the beginning of a function that handles special cases or invalid inputs immediately, preventing deep nesting and clarifying main logic.

**List Comprehension**: A concise Python syntax for creating lists by applying an expression to each item in an iterable, optionally with filtering: `[expr for item in iterable if condition]`.

**Loop**: A control structure that repeatedly executes a block of code, either for each item in a sequence (for loop) or while a condition remains true (while loop).

**Pseudocode**: A human-readable description of an algorithm that focuses on logic and structure without syntactic details, essential for algorithm design before implementation.

**Short-circuit Evaluation**: The behavior where logical operators (and, or) stop evaluating as soon as the result is determined, preventing unnecessary computation or errors.

## Key Takeaways

- Pseudocode reveals logical flaws before they become bugs - always design before implementing
- Never use `==` with floating-point numbers; always use tolerance-based comparisons
- Guard clauses handle special cases first, making the main logic clearer and preventing deep nesting  
- The accumulator pattern is fundamental to scientific computing, from statistics to Monte Carlo methods
- List comprehensions are faster than loops for simple transformations but become unreadable for complex logic
- Short-circuit evaluation in boolean logic prevents errors and improves performance
- While loops need explicit termination conditions and iteration limits to prevent infinite loops
- The convergence pattern (initialize, iterate, check, safeguard) appears throughout computational physics
- Strategic print debugging at decision points is often faster than using a debugger
- Order matters in condition chains - check from most specific to most general

## Quick Reference: Control Flow Functions and Patterns

| Construct/Function | Purpose | Example |
|-------------------|---------|---------|
| `if/elif/else` | Conditional execution | `if x > 0: positive()` |
| `for x in sequence` | Iterate over items | `for value in data:` |
| `for i in range(n)` | Count from 0 to n-1 | `for i in range(10):` |
| `while condition` | Repeat while true | `while error > tolerance:` |
| `break` | Exit loop early | `if converged: break` |
| `continue` | Skip to next iteration | `if invalid: continue` |
| `enumerate()` | Get index and value | `for i, x in enumerate(data):` |
| `zip()` | Parallel iteration | `for x, y in zip(xs, ys):` |
| `all()` | Check if all are true | `if all(x > 0 for x in data):` |
| `any()` | Check if any is true | `if any(x < 0 for x in data):` |
| `[expr for x in seq]` | List comprehension | `[x**2 for x in range(10)]` |
| `math.isclose()` | Safe float comparison | `if math.isclose(a, b):` |
| `math.isfinite()` | Check not inf/nan | `if math.isfinite(result):` |

### Common Patterns Reference

| Pattern | Purpose | Structure |
|---------|---------|-----------|
| Accumulator | Aggregate values | `total = 0; for x in data: total += x` |
| Filter | Select subset | `[x for x in data if condition(x)]` |
| Map | Transform all | `[f(x) for x in data]` |
| Search | Find first match | `for x in data: if condition(x): return x` |
| Convergence | Iterate to solution | `while not converged and iter < max: ...` |
| Guard clause | Handle special cases | `if bad_input: return None` |
| Adaptive refinement | Adjust parameters | `while quality < target: refine()` |

## Next Chapter Preview

You've conquered control flow - now get ready for the next level! Chapter 4 will reveal how to organize data efficiently using Python's powerful data structures. You'll discover when to use lists versus dictionaries versus sets, and more importantly, you'll understand why these choices can make your algorithms run 100 times faster or 100 times slower. 

Imagine trying to find a specific star in a catalog of millions. With a list, you'd check each star one by one - taking minutes or hours. With a dictionary, you'll find it instantly - in microseconds! The data structures you'll learn next are the difference between simulations that finish in minutes and ones that run for days.

The control flow patterns you've mastered here will operate on the data structures you'll learn next. Your loops will iterate through dictionaries of astronomical objects. Your conditionals will filter sets of observations. Your comprehensions will transform lists of measurements into meaningful results. Together, control flow and data structures give you the power to handle the massive datasets of modern astronomy - from Gaia's billion-star catalog to the petabytes of data from the Square Kilometre Array.

Get excited - Chapter 4 is where your code goes from processing dozens of data points to handling millions efficiently!