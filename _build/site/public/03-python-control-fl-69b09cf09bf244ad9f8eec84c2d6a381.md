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
- ✓ Use IPython effectively with magic commands like %timeit (Chapter 1)
- ✓ Understand floating-point precision and comparison issues (Chapter 2)
- ✓ Write and run Python scripts from the terminal (Chapter 1)
- ✓ Use f-strings for formatted output (Chapter 2)

## Chapter Overview

Programming is fundamentally about teaching computers to make decisions and repeat tasks. When you write an if-statement or a loop, you're translating human logic into instructions a machine can follow. But here's the critical insight that separates computational thinkers from mere coders: the logic must be designed before it's implemented.

This chapter transforms you from someone who writes code to someone who designs algorithms. We'll start with the lost art of pseudocode — not as a bureaucratic exercise, but as the difference between code that works by accident and code that works by design. You'll learn to recognize universal patterns that appear across all of computational physics: iteration, accumulation, filtering, mapping, and reduction. These patterns will appear in every project you build, from N-body simulations to neural networks.

The control flow structures we explore here are where your numerical calculations from Chapter 2 become dynamic algorithms. Every convergence test, every adaptive timestep, every Monte Carlo acceptance criterion depends on mastering these concepts deeply, not just syntactically.

## 3.1 Algorithmic Thinking: The Lost Art of Pseudocode

Most students jump straight from problem to code, then wonder why they spend hours debugging. Professional computational scientists spend more time thinking than typing. Pseudocode is how we think precisely about algorithms without getting distracted by syntax.

### Why Pseudocode Matters in Scientific Computing

Consider this scenario: You need to implement adaptive timestepping for an orbital integrator. Without pseudocode, you'll likely write code, run it, watch orbits spiral incorrectly, debug for hours, and maybe get it working through trial and error. With pseudocode, you'll identify edge cases, boundary conditions, and logical flaws before writing a single line of Python.

Let's see the difference:

```python
# WITHOUT PSEUDOCODE (typical student approach):
# "I'll figure it out as I code..."
def integrate(state, t_end):
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

Now with proper pseudocode design:

```
ALGORITHM: Adaptive Timestep Integration
INPUT: initial_state, t_end, tolerance
OUTPUT: final_state

INITIALIZE:
    current_state ← initial_state
    dt ← initial_guess_timestep
    min_dt ← machine_epsilon * timescale
    max_dt ← 0.1 * total_time

WHILE current_state.time < t_end:
    attempted_step ← False
    
    WHILE NOT attempted_step:
        trial_state ← integrate_step(current_state, dt)
        error ← estimate_error(current_state, trial_state)
        
        IF error > tolerance:
            dt ← max(dt * 0.5, min_dt)  # Prevent infinite shrinking
            IF dt == min_dt:
                WARN "Minimum timestep reached"
                attempted_step ← True  # Accept with warning
        ELSE:
            attempted_step ← True
            
    current_state ← trial_state
    
    # Adjust dt for next step
    IF error < 0.1 * tolerance:
        dt ← min(dt * 2, max_dt)  # Grow if very accurate

RETURN current_state
```

The pseudocode reveals issues immediately: What if error never gets small enough? What if dt grows too large? How do we handle the final step that might overshoot t_end? These questions are easier to answer in pseudocode than in Python.

### The Three Levels of Pseudocode Refinement

Professional algorithm development happens in stages, each revealing different issues:

**Level 1: Conceptual Overview**
```
WHILE simulation not done:
    Take a step
    Check if step was good
    Adjust timestep
```

**Level 2: Structural Detail**
```
WHILE time < end_time:
    DO:
        trial_step = integrate(state, dt)
        error = compute_error(trial_step)
    UNTIL error < tolerance OR dt < dt_min
    
    state = trial_step
    dt = adjust_timestep(error, dt)
```

**Level 3: Implementation-Ready**
```
FUNCTION adaptive_integrate(initial_state, end_time, tolerance):
    state ← initial_state
    dt ← estimate_initial_timestep(state)
    dt_min ← 1e-10 * (end_time - initial_state.time)
    dt_max ← 0.1 * (end_time - initial_state.time)
    
    WHILE state.time < end_time:
        step_accepted ← False
        attempts ← 0
        
        WHILE NOT step_accepted AND attempts < MAX_ATTEMPTS:
            dt_actual ← min(dt, end_time - state.time)  # Don't overshoot
            
            trial_state ← rk4_step(state, dt_actual)
            error_estimate ← || trial_state - rk2_step(state, dt_actual) ||
            
            IF error_estimate < tolerance:
                step_accepted ← True
                state ← trial_state
                
                # Adjust dt for next step
                IF error_estimate < 0.1 * tolerance:
                    dt ← min(dt * 1.5, dt_max)
            ELSE:
                dt ← max(dt * 0.5, dt_min)
                attempts ← attempts + 1
        
        IF NOT step_accepted:
            RAISE "Cannot achieve tolerance at minimum timestep"
    
    RETURN state
```

Each level of refinement reveals new issues and solutions. This is computational thinking in action.

### 📦 **Computational Thinking Box: The Universal Pattern of Adaptive Algorithms**

Adaptive timestepping is an instance of a universal pattern that appears throughout computational physics:

```
PATTERN: Adaptive Refinement
1. Attempt action with current parameters
2. Evaluate quality of result  
3. If quality insufficient: refine parameters and retry
4. If quality acceptable: proceed and possibly coarsen parameters
5. Include safeguards against infinite refinement

This pattern appears in:
- Adaptive mesh refinement (AMR) in hydrodynamics
- Step size control in ODE solvers
- Learning rate scheduling in neural networks  
- Convergence acceleration in iterative solvers
- Monte Carlo importance sampling
```

### 🔍 **Check Your Understanding**

Before continuing, write pseudocode for this problem: Given a list of stellar magnitudes, find all stars visible to the naked eye (magnitude ≤ 6), but exclude any that are within 0.1 magnitudes of the threshold (to account for measurement uncertainty).

<details>
<summary>Sample Solution</summary>

```
ALGORITHM: Find Reliably Visible Stars
INPUT: magnitude_list, visibility_threshold = 6.0, uncertainty = 0.1
OUTPUT: visible_stars

visible_stars ← empty list

FOR EACH magnitude IN magnitude_list:
    IF magnitude < (visibility_threshold - uncertainty):
        ADD magnitude TO visible_stars
    # Note: We exclude stars in the uncertainty zone
    # magnitude ∈ [5.9, 6.0] are excluded for safety

RETURN visible_stars
```

</details>

## 3.2 Boolean Logic in Scientific Computing

Every decision in your code ultimately reduces to true or false. But in scientific computing, these decisions often involve floating-point numbers, where equality is treacherous and precision is limited.

### The Fundamental Comparisons

```python
In [1]: # Basic comparisons
In [2]: temperature = 5778  # Kelvin (Sun's surface)

In [3]: temperature > 5000   # Hot enough for certain reactions
Out[3]: True

In [4]: temperature == 5778  # Exact equality (dangerous with floats!)
Out[4]: True  # Only because we used integers

In [5]: # The floating-point trap
In [6]: calculated_temp = 5778.0000000001
In [7]: calculated_temp == 5778.0
Out[7]: False  # Tiny difference breaks equality!
```

### Defensive Comparisons for Numerical Work

```python
In [8]: import math

In [9]: def safe_equal(a, b, rel_tol=1e-9, abs_tol=1e-12):
   ...:     """Safe floating-point comparison."""
   ...:     # Check for exact equality first (handles infinities)
   ...:     if a == b:
   ...:         return True
   ...:     
   ...:     # Check for NaN (NaN != NaN by definition)
   ...:     if math.isnan(a) or math.isnan(b):
   ...:         return False
   ...:     
   ...:     # Check for infinity
   ...:     if math.isinf(a) or math.isinf(b):
   ...:         return a == b
   ...:     
   ...:     # Normal comparison with tolerance
   ...:     return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
```

### ⚠️ **Common Bug Alert: Floating-Point Equality in Loops**

```python
# DANGEROUS - might never terminate!
step = 0.1
position = 0.0
target = 1.0

while position != target:  # BAD!
    position += step
    print(f"Position: {position}")

# After 10 steps, position = 0.9999999999999999, not 1.0!
# This loop runs forever!

# SAFE VERSION
while position < target - 1e-10:  # Good!
    position += step
```

### Combining Conditions: Order Matters

```python
In [10]: # Short-circuit evaluation can prevent errors
In [11]: data = []

In [12]: # WRONG - might crash
In [13]: if data[0] > 0 and len(data) > 0:  # IndexError if empty!
   ...:     print("First element is positive")

In [14]: # CORRECT - safe order
In [15]: if len(data) > 0 and data[0] > 0:  # Checks length first
   ...:     print("First element is positive")

In [16]: # Or more Pythonic
In [17]: if data and data[0] > 0:  # Empty list is False
   ...:     print("First element is positive")
```

### 📊 **Performance Profile: Short-Circuit Evaluation**

```python
In [18]: def expensive_check():
   ...:     """Simulates costly validation."""
   ...:     import time
   ...:     time.sleep(0.1)
   ...:     return True

In [19]: # Short-circuit saves time
In [20]: %timeit False and expensive_check()
154 ns ± 2.3 ns per loop  # Doesn't call expensive_check!

In [21]: %timeit True and expensive_check()  
100 ms ± 523 µs per loop  # Calls expensive_check

# Use this pattern in convergence checks:
if iteration > max_iterations or has_converged(state):
    break  # Checks iteration count FIRST
```

## 3.3 Conditional Statements: Teaching Computers to Decide

Conditional statements are where your code makes decisions. In scientific computing, these decisions often involve numerical thresholds, convergence criteria, and boundary conditions.

### The Pattern of Scientific Conditionals

```python
def classify_stellar_remnant(mass_solar):
    """
    Determine stellar remnant type based on initial mass.
    Demonstrates guard clauses and defensive programming.
    """
    # Guard clause - validate input first
    if mass_solar <= 0:
        raise ValueError(f"Stellar mass must be positive: {mass_solar}")
    
    if not math.isfinite(mass_solar):
        raise ValueError(f"Stellar mass must be finite: {mass_solar}")
    
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
    boundary_distances = [
        abs(mass_solar - 0.08),
        abs(mass_solar - 8),
        abs(mass_solar - 25)
    ]
    min_distance = min(boundary_distances)
    
    if min_distance < 0.5:
        remnant += " (near classification boundary)"
    
    return remnant
```

### Guard Clauses: Fail Fast, Fail Clear

Guard clauses handle special cases immediately, preventing deep nesting and making code clearer:

```python
def calculate_orbital_period(a, M, validate=True):
    """
    Kepler's third law with comprehensive validation.
    
    Shows the pattern of guard clauses for scientific code.
    """
    # Guard clauses handle problems immediately
    if validate:
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive: {a}")
        if M <= 0:
            raise ValueError(f"Mass must be positive: {M}")
        if a < 2.95e-4 * M:  # Inside Schwarzschild radius!
            raise ValueError(f"Orbit inside black hole: a={a}, Rs={2.95e-4*M}")
    
    # Main calculation - only runs if all guards pass
    import math
    G = 6.67e-8  # CGS
    period = 2 * math.pi * math.sqrt(a**3 / (G * M))
    
    # Sanity check on result
    if validate and period > 13.8e9 * 365.25 * 86400:  # Age of universe
        import warnings
        warnings.warn(f"Orbital period exceeds age of universe: {period} s")
    
    return period
```

### 🐛 **Debug This!**

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

<details>
<summary>Bug and Solution</summary>

**Bug**: Division by new_value fails when new_value is very small but non-zero, and gives wrong results when old_value is much larger than new_value.

**Fixed Version**:
```python
def check_convergence(old_value, new_value, tolerance):
    """Check if iterative calculation has converged."""
    
    # Handle exact convergence
    if old_value == new_value:
        return True
    
    # Use the larger magnitude for relative comparison
    scale = max(abs(old_value), abs(new_value))
    
    if scale == 0:
        return True  # Both are zero
    
    relative_change = abs(new_value - old_value) / scale
    return relative_change < tolerance
```

</details>

## 3.4 Loops: The Heart of Scientific Computation

Loops are how we process data, iterate until convergence, and simulate time evolution. Choosing the right loop structure and implementing it correctly determines whether your simulation finishes in minutes or days.

### For Loops: When You Know What to Iterate Over

```python
# Basic iteration pattern
measurements = [10.2, 10.5, 10.3, 10.6, 10.4]

# Accumulation pattern - fundamental to all reductions
total = 0
sum_of_squares = 0
for value in measurements:
    total += value
    sum_of_squares += value**2

mean = total / len(measurements)
variance = sum_of_squares / len(measurements) - mean**2
```

Let's trace through the execution to build intuition:

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

Final: total = 51.5, sum_of_squares = 530.39
mean = 51.5 / 5 = 10.3
variance = 530.39 / 5 - 10.3² = 0.0178
```

### Common For Loop Patterns in Scientific Computing

```python
# Pattern 1: Parallel iteration with zip
times = [0, 1, 2, 3, 4]  # seconds
positions = [0, 4.9, 19.6, 44.1, 78.4]  # meters

for t, x in zip(times, positions):
    velocity = x / (t + 1e-10)  # Avoid division by zero
    print(f"t={t}s: x={x}m, v={velocity:.1f}m/s")

# Pattern 2: Enumeration for indexing
data = [1.2, 2.3, 3.4, 4.5]
filtered = []
indices = []

for i, value in enumerate(data):
    if value > 2.0:
        filtered.append(value)
        indices.append(i)
print(f"Values > 2.0 at indices: {indices}")

# Pattern 3: Sliding window (for smoothing, derivatives)
window_size = 3
smoothed = []

for i in range(len(data) - window_size + 1):
    window = data[i:i + window_size]
    smoothed.append(sum(window) / window_size)
```

### While Loops: Iterating Until a Condition

While loops are essential for iterative algorithms where the number of iterations isn't known in advance:

```python
def find_root_bisection(func, a, b, tolerance=1e-10, max_iter=100):
    """
    Find root using bisection method.
    Demonstrates proper while loop with safety checks.
    """
    # Validate inputs
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    iteration = 0
    
    # Main bisection loop
    while abs(b - a) > tolerance and iteration < max_iter:
        c = (a + b) / 2
        fc = func(c)
        
        # Check if we found exact root
        if fc == 0:
            return c
        
        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
        iteration += 1
        
        # Optional: track convergence
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: interval=[{a}, {b}], width={b-a}")
    
    # Check why we stopped
    if iteration >= max_iter:
        import warnings
        warnings.warn(f"Maximum iterations reached. Precision: {b-a}")
    
    return (a + b) / 2
```

### 📦 **Computational Thinking Box: The Convergence Pattern**

```
PATTERN: Iterative Convergence

initialize state
initialize iteration_count = 0

WHILE NOT converged AND iteration_count < max_iterations:
    new_state = update(state)
    converged = check_convergence(state, new_state, tolerance)
    state = new_state
    iteration_count += 1

IF NOT converged:
    handle_failure()

This pattern appears in:
- Root finding (Newton-Raphson, bisection)
- Fixed-point iteration
- Iterative linear solvers (Jacobi, Gauss-Seidel)
- Optimization algorithms (gradient descent)
- Self-consistent field calculations
- Monte Carlo equilibration
```

### ⏸️ **Pause and Predict**

What will this code print?

```python
x = 0.0
while x != 1.0:
    x += 0.1
    print(f"{x:.17f}")
    if x > 2:  # Safety check
        break
```

<details>
<summary>Answer</summary>

It prints 20+ lines and hits the safety check! After 10 additions:
```
0.10000000000000001
0.20000000000000001
0.30000000000000004
...
0.99999999999999989  # Not 1.0!
1.09999999999999987
...
2.09999999999999964
```

The accumulated rounding errors prevent x from ever exactly equaling 1.0.

</details>

## 3.5 List Comprehensions: Elegant and Efficient

List comprehensions provide a concise way to create lists, but they're more than syntactic sugar — they can be significantly faster than equivalent loops.

### From Loop to Comprehension

```python
# Traditional loop approach
squares = []
for x in range(10):
    if x % 2 == 0:  # Even numbers only
        squares.append(x**2)

# List comprehension - same result, clearer intent
squares = [x**2 for x in range(10) if x % 2 == 0]

# Performance comparison
In [50]: %timeit [x**2 for x in range(1000) if x % 2 == 0]
47.3 µs ± 312 ns per loop

In [51]: %%timeit
   ...: squares = []
   ...: for x in range(1000):
   ...:     if x % 2 == 0:
   ...:         squares.append(x**2)
73.2 µs ± 1.02 µs per loop

# Comprehension is ~35% faster!
```

### When to Use (and Not Use) Comprehensions

```python
# GOOD: Simple transformation
magnitudes = [2.3, 5.1, 3.7, 6.2, 4.5]
fluxes = [10**(-0.4 * mag) for mag in magnitudes]

# GOOD: Filtering with condition
visible = [mag for mag in magnitudes if mag < 6.0]

# BAD: Too complex, hard to read
# result = [process(x) if condition(x) else 
#           alternative(y) for x, y in zip(list1, list2) 
#           if validate(x) and check(y)]

# BETTER: Use a loop for complex logic
result = []
for x, y in zip(list1, list2):
    if validate(x) and check(y):
        if condition(x):
            result.append(process(x))
        else:
            result.append(alternative(y))
```

### Nested Comprehensions: Handle with Care

```python
# Creating a distance matrix
positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

# Readable nested comprehension
distances = [[math.sqrt((x1-x2)**2 + (y1-y2)**2) 
              for x2, y2 in positions]
             for x1, y1 in positions]

# When nesting gets deep, use loops for clarity
```

## 3.6 Advanced Control Flow Patterns

### The Accumulator Pattern

The accumulator pattern is fundamental to scientific computing:

```python
def running_statistics(data_stream):
    """
    Calculate mean and variance in a single pass.
    Demonstrates Welford's algorithm for numerical stability.
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

# This is numerically stable even for large datasets!
```

### The Filter-Map-Reduce Pattern

```python
# Scientific data processing pipeline
raw_measurements = [10.2, -999, 10.5, 10.3, -999, 10.6]  # -999 = bad data

# Filter: Remove bad data
valid_data = [x for x in raw_measurements if x != -999]

# Map: Convert to different unit
converted = [x * 1.5 for x in valid_data]  # Some conversion

# Reduce: Aggregate to single value
result = sum(converted) / len(converted)

# Or as a single expression (less readable):
result = sum(x * 1.5 for x in raw_measurements if x != -999) / \
         sum(1 for x in raw_measurements if x != -999)
```

### 📈 **Algorithm Archaeology: Why Welford's Algorithm?**

The naive variance calculation `variance = sum_of_squares/n - mean²` suffers from catastrophic cancellation when the mean is large relative to the variance. 

In 1962, B.P. Welford published a single-pass algorithm that maintains numerical stability by computing differences from the running mean. This was revolutionary for computers with limited memory that couldn't store all data for a second pass.

Today, this pattern appears in:
- Online learning algorithms
- Streaming data analysis
- Embedded systems with memory constraints
- Real-time telescope data processing

## 3.7 Debugging Control Flow

Logic errors are the hardest bugs because the code runs without errors but produces wrong results.

### Strategic Print Debugging

```python
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
        
        # Debug: show change
        change = current - old
        print(f" → new={current:.6f} (Δ={change:+.6f})")
        
        # Safety check with informative message
        if iteration > 50 and abs(change) < 1e-10:
            print(f"WARNING: Change too small at iteration {iteration}")
            break
    
    print(f"FAILED: No convergence after {max_iter} iterations")
    return current
```

### Using IPython Debugger

```python
In [60]: def buggy_function(data):
   ...:     result = []
   ...:     for i in range(len(data)):
   ...:         if data[i] > data[i+1]:  # Bug: goes past end!
   ...:             result.append(data[i])
   ...:     return result

In [61]: buggy_function([3, 1, 4, 1, 5])
# IndexError!

In [62]: %debug
> buggy_function()
      3     for i in range(len(data)):
----> 4         if data[i] > data[i+1]:
      5             result.append(data[i])

ipdb> i
4
ipdb> len(data)
5
ipdb> # Aha! When i=4, i+1=5 is out of bounds!
```

## 3.8 Optional: Bitwise Operations in Scientific Computing

*This section is optional but included for completeness, as bitwise operations appear in instrument control, data compression, and FITS file handling.*

### When You Encounter Bitwise Operations

```python
# Reading telescope status flags
TRACKING = 0b0001  # Binary: 0001
GUIDING  = 0b0010  # Binary: 0010  
COOLING  = 0b0100  # Binary: 0100
EXPOSING = 0b1000  # Binary: 1000

status = 0b0101  # Binary representation of status

# Check specific flags
is_tracking = bool(status & TRACKING)  # AND operation
is_cooling = bool(status & COOLING)

# Set a flag
status |= EXPOSING  # OR operation to set bit

# Clear a flag  
status &= ~COOLING  # AND with NOT to clear bit

print(f"Status: {status:04b}")  # Binary formatting
```

### Practical Example: Packed Data

```python
def unpack_compressed_coords(packed):
    """
    Some instruments pack x,y coordinates into single 32-bit integer.
    Upper 16 bits = x, Lower 16 bits = y
    """
    x = (packed >> 16) & 0xFFFF  # Shift right and mask
    y = packed & 0xFFFF          # Mask lower bits
    return x, y

packed = 0x00640032  # x=100, y=50
x, y = unpack_compressed_coords(packed)
print(f"Unpacked: x={x}, y={y}")
```

## Practice Exercises

### Exercise 3.1: Robust Convergence Checker

```python
def robust_convergence_checker(history, tolerance, window=3):
    """
    Check convergence using recent history, not just last two values.
    
    Args:
        history: List of values from iterations
        tolerance: Convergence threshold
        window: Number of recent values to check
        
    Returns:
        (converged, reason)
    
    Your implementation should:
    1. Handle empty or short histories
    2. Check if values are oscillating
    3. Check if values are converging monotonically
    4. Return informative reason string
    """
    # Your code here
    pass
```

### Exercise 3.2: Adaptive Algorithm Design

```python
"""
Design pseudocode for an adaptive Monte Carlo sampler that:
1. Starts with uniform sampling
2. Identifies regions of high "importance" 
3. Focuses sampling in important regions
4. Maintains some exploration of full space
5. Stops when variance is below threshold

Write three levels of pseudocode refinement.
Then implement in Python.
"""
```

### Exercise 3.3: Debug the Logic

```python
def find_peak(data, threshold):
    """
    This function should find peaks above threshold,
    but it has multiple logic errors. Find and fix them.
    """
    peaks = []
    for i in range(len(data)):
        # Check if current point is a peak
        if data[i] > threshold:
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
    return peaks

# Test: find_peak([1, 3, 2, 5, 1], 2)
# Should return [1, 3] but crashes. Why?
```

## Key Takeaways

Pseudocode is not optional bureaucracy but essential algorithm design. Every hour spent on pseudocode saves many hours of debugging. The three-level refinement process reveals issues before they become bugs.

Floating-point comparisons require defensive programming. Never use `==` with floats. Always include tolerances and handle special values (inf, nan) explicitly.

Universal patterns appear throughout computational physics. The accumulator pattern, convergence pattern, and adaptive refinement pattern you learned here will appear in every project.

Guard clauses and early returns prevent deep nesting and make code clearer. Handle special cases first, then focus on the main algorithm.

List comprehensions are powerful but not always appropriate. Use them for simple transformations and filtering. Use explicit loops when logic is complex.

Debugging logic errors requires systematic approaches. Strategic printing at decision points, using the debugger effectively, and testing edge cases are essential skills.

## Quick Reference: Control Flow Functions and Patterns

| Construct/Function | Purpose | Example |
|-------------------|---------|---------|
| `if/elif/else` | Conditional execution | `if x > 0: ...` |
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
| `[... if condition]` | Filtered comprehension | `[x for x in data if x > 0]` |
| `range(start, stop, step)` | Generate sequence | `range(0, 10, 2)` → 0,2,4,6,8 |
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
| Sliding window | Local operations | `for i in range(len(data)-window+1): ...` |
| Guard clause | Handle special cases | `if bad_input: return None` |

## Next Chapter Preview

Now that you can control program flow and implement algorithms systematically, Chapter 4 will explore how to organize data efficiently. You'll learn when to use lists versus dictionaries versus sets, understand the performance implications of each choice, and see how data structure selection can make the difference between algorithms that finish in seconds versus hours. The control flow patterns you've mastered here will operate on the data structures you'll learn next, combining to create efficient scientific algorithms.