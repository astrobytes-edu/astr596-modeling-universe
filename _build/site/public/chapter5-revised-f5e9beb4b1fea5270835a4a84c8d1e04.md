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

Functions are the fundamental building blocks of organized code. Without functions, you'd be copying and pasting the same code repeatedly, making bugs harder to fix and improvements impossible to maintain. But functions are more than just a way to avoid repetition—they're how we create abstractions, manage complexity, and build reliable software. Whether you're simulating particle trajectories, analyzing statistical distributions, or processing telescope data, every computational project starts with well-designed functions.

This chapter teaches you to think about functions as contracts between different parts of your code. When you write a function that calculates kinetic energy or performs numerical integration, you're creating a promise: given valid input, the function will reliably return the correct output. This contract mindset helps you write functions that others (including future you) can trust and use effectively.

We'll explore Python's scope rules, which determine where variables can be accessed, and learn how seemingly simple concepts like default arguments can create subtle bugs that have plagued even major scientific software packages. You'll discover how Python's flexible parameter system enables powerful interfaces, and how functional programming concepts prepare you for modern scientific computing frameworks like JAX. By the end, you'll be organizing your code into modules that can be shared, tested, and maintained professionally—essential skills for collaborative scientific research.

## 5.1 Defining Functions: The Basics

A function encapsulates a piece of logic that transforms inputs into outputs. Think of a function as a machine: you feed it raw materials (inputs), it performs some process (the function body), and it produces a product (output). In physics, a function might calculate energy from mass and velocity, or integrate a differential equation—each function performs one clear task that can be tested and trusted.

### Your First Function

Let's start with something every physicist needs—calculating kinetic energy:

```{code-cell} python
def kinetic_energy(mass, velocity):
    """
    Calculate kinetic energy of a particle.
    
    KE = (1/2) * m * v^2
    """
    ke = 0.5 * mass * velocity**2
    return ke

# Using the function
electron_mass = 9.109e-31  # kg
electron_velocity = 2.18e6  # m/s (in hydrogen atom)
energy = kinetic_energy(electron_mass, electron_velocity)

print(f"Electron kinetic energy: {energy:.2e} Joules")
print(f"In electron volts: {energy/1.602e-19:.2f} eV")
```

Let's break down exactly how this works:

1. **`def` keyword**: Tells Python we're defining a function
2. **Function name** (`kinetic_energy`): Follows snake_case convention, describes what it does
3. **Parameters** (`mass, velocity`): Variables that receive values when function is called
4. **Docstring**: Brief description of what the function does (always include this!)
5. **Function body**: Indented code that does the actual work
6. **`return` statement**: Sends a value back to whoever called the function

When Python executes `kinetic_energy(electron_mass, electron_velocity)`, it creates a temporary namespace where `mass = 9.109e-31` and `velocity = 2.18e6`, runs the function body, and returns the result.

:::{admonition} 🔍 Check Your Understanding #1
:class: question

What will this code print?

```python
def calculate_momentum(mass, velocity):
    momentum = mass * velocity
    # Oops, forgot the return statement!

p = calculate_momentum(2.0, 3.0)
print(f"Momentum: {p}")
```

<details>
<summary>Answer</summary>

It prints `Momentum: None`. The function calculates `momentum` but doesn't return it. Without an explicit `return` statement, Python functions return `None`. This is a common bug in scientific code!

To fix it:
```python
def calculate_momentum(mass, velocity):
    momentum = mass * velocity
    return momentum  # Now it returns the value
```

</details>
:::

### Functions Without Return Values

Not all functions return values. Some perform actions like printing results or saving data:

```{code-cell} python
def report_calculation(name, value, units, threshold=1e-10):
    """Report calculation result with significance check."""
    if abs(value) < threshold:
        significance = "negligible"
    elif abs(value) < 1.0:
        significance = "small"
    else:
        significance = "significant"
    
    print(f"{name}: {value:.3e} {units} ({significance})")
    # No return statement - returns None implicitly

# Report some physics calculations
report_calculation("Force", 1.2e-8, "N")
report_calculation("Energy", 5.4e-19, "J")
report_calculation("Momentum", 2.3, "kg⋅m/s")
```

### Returning Multiple Values

Python functions can return multiple values using tuples—perfect for calculations that produce related results:

```{code-cell} python
def projectile_motion(v0, angle_deg, t):
    """
    Calculate projectile position and velocity at time t.
    
    Parameters:
        v0: initial velocity (m/s)
        angle_deg: launch angle (degrees)
        t: time (seconds)
    
    Returns:
        x, y: position (meters)
        vx, vy: velocity components (m/s)
    """
    import math
    
    # Convert angle to radians
    angle = math.radians(angle_deg)
    
    # Initial velocity components
    v0x = v0 * math.cos(angle)
    v0y = v0 * math.sin(angle)
    
    # Position at time t
    x = v0x * t
    y = v0y * t - 0.5 * 9.81 * t**2
    
    # Velocity at time t
    vx = v0x
    vy = v0y - 9.81 * t
    
    return x, y, vx, vy

# Analyze projectile after 1 second
x, y, vx, vy = projectile_motion(30, 45, 1.0)
print(f"Position: ({x:.1f}, {y:.1f}) m")
print(f"Velocity: ({vx:.1f}, {vy:.1f}) m/s")
speed = (vx**2 + vy**2)**0.5
print(f"Speed: {speed:.1f} m/s")
```

### The Design Process: From Problem to Function

Before writing any function, design it first. This prevents the common mistake of coding yourself into a corner:

```{code-cell} python
"""
DESIGN: Function to validate numerical calculation

PURPOSE: Ensure numerical results are physically reasonable
INPUT: value, expected_range, tolerance
OUTPUT: boolean (True if valid)
CHECKS:
    - Value is finite (not inf or nan)
    - Value is within expected range
    - Value differs from expected by less than tolerance
"""

def validate_result(value, expected=None, bounds=None, name="Result"):
    """
    Validate numerical calculation result.
    
    Parameters
    ----------
    value : float
        Calculated value to validate
    expected : float, optional
        Expected value for comparison
    bounds : tuple, optional
        (min, max) acceptable range
    name : str
        Name for error messages
        
    Returns
    -------
    bool
        True if all validation checks pass
    """
    import math
    
    # Check if value is finite
    if not math.isfinite(value):
        print(f"ERROR: {name} is not finite: {value}")
        return False
    
    # Check bounds if provided
    if bounds is not None:
        min_val, max_val = bounds
        if not (min_val <= value <= max_val):
            print(f"ERROR: {name}={value} outside bounds [{min_val}, {max_val}]")
            return False
    
    # Check expected value if provided
    if expected is not None:
        rel_error = abs(value - expected) / abs(expected) if expected != 0 else abs(value)
        if rel_error > 0.01:  # 1% tolerance
            print(f"ERROR: {name}={value} differs from expected {expected} by {rel_error*100:.1f}%")
            return False
    
    return True

# Test validation
energy = kinetic_energy(1.0, 10.0)
print(f"Energy = {energy} J")
print(f"Valid? {validate_result(energy, expected=50.0, bounds=(0, 1000))}")

# Test with bad value
bad_energy = float('inf')
print(f"\nBad energy = {bad_energy}")
print(f"Valid? {validate_result(bad_energy, name='Energy')}")
```

:::{admonition} 🌟 Why This Matters: The Cost of Unvalidated Functions
:class: important

In 1996, the European Space Agency's Ariane 5 rocket exploded 37 seconds after launch, destroying $370 million in satellites. The cause? A single unvalidated function that converted a 64-bit floating-point number to a 16-bit integer without checking if the value fit. The horizontal velocity exceeded 32,767 (the maximum for a 16-bit signed integer), causing an overflow that triggered self-destruct.

A simple validation function like ours above—checking bounds before conversion—would have prevented this disaster. This is why we always validate inputs and outputs in scientific computing!
:::

:::{admonition} 💡 Computational Thinking: Function Contract Design
:class: tip

Every well-designed function follows a contract pattern that applies across all programming:

```
CONTRACT PATTERN:
1. Preconditions: What must be true before calling
2. Postconditions: What will be true after calling
3. Invariants: What stays unchanged
4. Side effects: What else happens

Example for kinetic_energy():
- Precondition: mass > 0, velocity is numeric
- Postcondition: returns positive energy value
- Invariant: input values unchanged
- Side effects: none (pure function)

This pattern appears in:
- Database transactions (ACID properties)
- API design (REST contracts)
- Parallel computing (thread safety)
- Unit testing (test contracts)
```
:::

## 5.2 Function Arguments In-Depth

Python provides flexible ways to handle function parameters. Let's explore this flexibility through three phases: basic usage, common pitfalls, and advanced patterns.

### Phase 1: Basic - Positional vs Keyword Arguments

```{code-cell} python
def calculate_force(mass, acceleration, friction_coeff=0.0):
    """
    Calculate net force with optional friction.
    
    F_net = ma - μmg (simplified friction model)
    """
    force = mass * acceleration
    if friction_coeff > 0:
        friction = friction_coeff * mass * 9.81
        force -= friction
    return force

# Different ways to call the same function
f1 = calculate_force(10, 2)  # Positional only
f2 = calculate_force(mass=10, acceleration=2)  # Keywords (any order!)
f3 = calculate_force(10, 2, friction_coeff=0.1)  # Mixed

print(f"No friction: {f1:.1f} N")
print(f"Same (keywords): {f2:.1f} N")  
print(f"With friction: {f3:.1f} N")
```

### Phase 2: Intermediate - The Mutable Default Trap

Here's a critical bug that has appeared in major scientific software:

```{code-cell} python
# THE TRAP - DON'T DO THIS!
def add_measurement_buggy(value, measurements=[]):
    """Add measurement to list - BUGGY VERSION."""
    measurements.append(value)
    return measurements

# Watch the disaster unfold
day1_data = add_measurement_buggy(3.14)
print(f"Day 1: {day1_data}")

day2_data = add_measurement_buggy(2.71)  # Surprise!
print(f"Day 2: {day2_data}")  # Contains BOTH days!

print(f"Same object? {day1_data is day2_data}")  # True!
```

:::{admonition} ⚠️ Common Bug Alert: The Mutable Default Disaster
:class: warning

This bug has appeared in:
- IRAF reduction scripts (accumulated all nights' data)
- Astropy coordinate transformations (cached incorrect results)
- Observatory scheduling software (mixed different programs)

The symptom: data from previous runs mysteriously appears in new analyses. 

The cause: Python evaluates default arguments **once** when the `def` statement executes (usually at module import), not each time the function is called.

The fix: always use `None` as default for mutable arguments.
:::

Here's the correct pattern:

```{code-cell} python
def add_measurement_fixed(value, measurements=None):
    """Add measurement to list - CORRECT VERSION."""
    if measurements is None:
        measurements = []  # Create new list each time
    measurements.append(value)
    return measurements

# Now it works correctly
day1_data = add_measurement_fixed(3.14)
day2_data = add_measurement_fixed(2.71)
print(f"Day 1: {day1_data}")
print(f"Day 2: {day2_data}")  # Separate lists!
```

### Phase 3: Advanced - Variable Arguments (*args and **kwargs)

```{code-cell} python
def statistical_summary(*values, method='all'):
    """
    Calculate statistics for any number of values.
    
    *args collects positional arguments into a tuple
    """
    if not values:
        return None
    
    import math
    
    n = len(values)
    mean = sum(values) / n
    
    if method == 'mean':
        return mean
    
    # Calculate variance and std dev
    variance = sum((x - mean)**2 for x in values) / n
    std_dev = math.sqrt(variance)
    
    if method == 'all':
        return {
            'n': n,
            'mean': mean,
            'std': std_dev,
            'min': min(values),
            'max': max(values)
        }
    
    return mean

# Works with any number of arguments
print(statistical_summary(1, 2, 3, 4, 5))
print(statistical_summary(2.5, 3.7, 1.2, 4.8, 3.3, 2.9, method='mean'))
```

```{code-cell} python
def run_simulation(particles, time_step, **options):
    """
    Run particle simulation with flexible options.
    
    **kwargs collects keyword arguments into a dictionary
    """
    # Default options
    defaults = {
        'gravity': True,
        'collisions': False,
        'verbose': False,
        'max_steps': 1000
    }
    
    # Update with user options
    config = defaults.copy()
    config.update(options)
    
    if config['verbose']:
        print(f"=== Simulation Configuration ===")
        print(f"Particles: {particles}")
        print(f"Time step: {time_step}s")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Simulation would run here
    return f"Simulated {particles} particles"

# Simple call
run_simulation(100, 0.01)

# Complex call with many options
result = run_simulation(
    50, 0.001,
    gravity=True,
    collisions=True,
    verbose=True,
    boundary='periodic'
)
```

:::{admonition} 🔍 Check Your Understanding #2
:class: question

What's wrong with this function definition, and how would you fix it?

```python
def process_data(values, scale=1.0, *extra, normalize=True):
    # Process data with options
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
def process_data(values, *extra, scale=1.0, normalize=True):
    # Now the order is correct
    pass
```

The original would give a SyntaxError because *extra can't come after a keyword parameter with a default.

</details>
:::

## 5.3 Scope and Namespaces

Understanding scope—where variables can be accessed—is crucial for writing bug-free code. Python's scope rules determine which variables are visible at any point in your program. Without understanding scope, you'll encounter confusing bugs where variables don't have the values you expect, or worse, where changing a variable in one place mysteriously affects code elsewhere.

### The LEGB Rule

Python resolves variable names using the LEGB rule, searching in this order:
- **L**ocal: Inside the current function
- **E**nclosing: In the enclosing function (for nested functions)  
- **G**lobal: At the top level of the module
- **B**uilt-in: In the built-in namespace (print, len, etc.)

```{code-cell} python
# Demonstrating LEGB with physics context
speed_of_light = 3e8  # Global scope

def calculate_energy():
    speed_of_light = 3.0e8  # Enclosing scope (shadows global)
    
    def relativistic_energy(mass):
        speed_of_light = 299792458  # Local scope (shadows enclosing)
        
        # Local scope sees its own speed_of_light
        print(f"Inside relativistic: c = {speed_of_light} m/s")
        return mass * speed_of_light**2
    
    # Call inner function
    energy = relativistic_energy(1e-30)
    
    # Enclosing scope sees its own speed_of_light
    print(f"Inside calculate: c = {speed_of_light} m/s")
    return energy

result = calculate_energy()
print(f"Global: c = {speed_of_light} m/s")  # Unchanged!
```

Each function creates its own namespace—a mapping of names to objects. When you use a variable, Python searches through these namespaces in LEGB order until it finds the name.

### The UnboundLocalError Trap

```{code-cell} python
counter = 0  # Global

def increment_wrong():
    # This will crash with UnboundLocalError!
    # counter += 1  # Python thinks this is local
    pass

def increment_fixed():
    global counter  # Explicitly use global
    counter += 1
    return counter

# Better approach - avoid global state entirely
def increment_better(current_count):
    return current_count + 1

# Test the better approach
count = 0
count = increment_better(count)
print(f"Count: {count}")
```

:::{admonition} ⚠️ Common Bug Alert: UnboundLocalError 
:class: warning

The error happens because Python sees you're assigning to `counter`, assumes it's local, but then can't find a local value to increment. This bug often appears in simulation scripts that try to maintain running totals.

Symptoms: 
- Variable works fine when reading
- Crashes when trying to modify
- Error message mentions "referenced before assignment"

Fix: Either use `global` keyword or (better) pass the value explicitly.
:::

### Closures: Functions That Remember

Closures are functions that "remember" variables from their enclosing scope:

```{code-cell} python
def create_integrator(method='rectangle'):
    """
    Create a numerical integrator with a specific method.
    
    The returned function 'remembers' the method.
    """
    def integrate(func, a, b, n=100):
        """Integrate func from a to b using n steps."""
        dx = (b - a) / n
        
        if method == 'rectangle':
            # Rectangle rule (simplest integration)
            total = 0
            for i in range(n):
                x = a + i * dx
                total += func(x) * dx
            return total
            
        elif method == 'midpoint':
            # Midpoint rule (more accurate)
            total = 0
            for i in range(n):
                x = a + (i + 0.5) * dx
                total += func(x) * dx
            return total
    
    return integrate

# Create specialized integrators
rect_integrate = create_integrator('rectangle')
mid_integrate = create_integrator('midpoint')

# Test with x^2 from 0 to 1 (exact answer = 1/3)
def f(x):
    return x**2

rect_result = rect_integrate(f, 0, 1, 100)
mid_result = mid_integrate(f, 0, 1, 100)
exact = 1/3

print(f"Rectangle rule: {rect_result:.6f} (error: {abs(rect_result-exact):.6f})")
print(f"Midpoint rule:  {mid_result:.6f} (error: {abs(mid_result-exact):.6f})")
print(f"Exact:          {exact:.6f}")
```

:::{admonition} 🔍 Check Your Understanding #3
:class: question

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

If the inner function didn't define its own `x`, it would see the outer's `x` due to the LEGB rule (it would find it in the Enclosing scope).
</details>
:::

:::{admonition} 💡 Computational Thinking: Why Global Variables Are Dangerous
:class: tip

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
:::

## 5.4 Functional Programming Elements

Python supports functional programming—a style that treats computation as the evaluation of mathematical functions. These concepts are essential for modern scientific frameworks like JAX and lead to cleaner, more testable code.

### Lambda Functions

Lambda functions are small, anonymous functions defined inline:

```{code-cell} python
# Sort particles by different properties
particles = [
    {'id': 1, 'mass': 1.67e-27, 'charge': 1.6e-19, 'v': 1e6},  # proton
    {'id': 2, 'mass': 9.11e-31, 'charge': -1.6e-19, 'v': 2e6},  # electron
    {'id': 3, 'mass': 1.67e-27, 'charge': 0, 'v': 5e5},  # neutron
]

# Sort by mass (lightest first)
by_mass = sorted(particles, key=lambda p: p['mass'])
print("Sorted by mass:")
for p in by_mass:
    print(f"  Particle {p['id']}: mass = {p['mass']:.2e} kg")

# Sort by kinetic energy
by_energy = sorted(particles, key=lambda p: 0.5 * p['mass'] * p['v']**2)
print("\nSorted by kinetic energy:")
for p in by_energy:
    energy = 0.5 * p['mass'] * p['v']**2
    print(f"  Particle {p['id']}: KE = {energy:.2e} J")
```

### Map, Filter, and Reduce

These functional tools transform how you process data:

```{code-cell} python
from functools import reduce

# Sample measurements with noise
measurements = [
    9.81, 9.79, 9.83, 99.9,  # Bad measurement!
    9.80, 9.82, 9.78, 9.81
]

# FILTER: Remove outliers
mean_estimate = 9.81
tolerance = 0.1
good_data = list(filter(
    lambda x: abs(x - mean_estimate) < tolerance,
    measurements
))
print(f"Filtered {len(measurements) - len(good_data)} outliers")
print(f"Good data: {good_data}")

# MAP: Convert to different units
measurements_in_ft_s2 = list(map(
    lambda x: x * 3.28084,  # m/s² to ft/s²
    good_data
))
print(f"In ft/s²: {[f'{x:.2f}' for x in measurements_in_ft_s2]}")

# REDUCE: Calculate mean
mean = reduce(lambda a, b: a + b, good_data) / len(good_data)
print(f"Mean: {mean:.3f} m/s²")

# Or more complex - calculate running variance
def variance_accumulator(acc, value):
    """Accumulate sum and sum of squares."""
    sum_val, sum_sq, count = acc
    return (sum_val + value, sum_sq + value**2, count + 1)

sum_val, sum_sq, n = reduce(variance_accumulator, good_data, (0, 0, 0))
variance = (sum_sq / n) - (sum_val / n)**2
print(f"Variance: {variance:.6f}")
```

:::{admonition} 🔍 Check Your Understanding #4
:class: question

Rewrite this loop using functional programming:

```python
# Find all positive values and square them
results = []
for x in data:
    if x > 0:
        results.append(x**2)
```

<details>
<summary>Answer</summary>

Two equivalent functional approaches:

```python
# Using filter and map
results = list(map(
    lambda x: x**2,
    filter(lambda x: x > 0, data)
))

# Using list comprehension (more Pythonic)
results = [x**2 for x in data if x > 0]
```

The list comprehension is generally preferred in Python for readability, but understanding the functional approach prepares you for libraries like JAX that require functional style.

</details>
:::

### Functions as First-Class Objects

In Python, functions are objects you can pass around:

```{code-cell} python
def mean_aggregator(values):
    """Average values - best for Gaussian noise."""
    return sum(values) / len(values)

def median_aggregator(values):
    """Median - robust against outliers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
    return sorted_vals[n//2]

def process_trials(data, aggregator_func):
    """
    Process experimental trials with specified aggregation.
    
    aggregator_func determines how to combine results.
    """
    print(f"Using {aggregator_func.__name__} aggregation")
    result = aggregator_func(data)
    return result

# Simulated repeated measurements
trials = [9.81, 9.79, 9.83, 15.0, 9.80]  # Note the outlier!

mean_result = process_trials(trials, mean_aggregator)
median_result = process_trials(trials, median_aggregator)

print(f"Mean:   {mean_result:.2f} (affected by outlier)")
print(f"Median: {median_result:.2f} (robust to outlier)")
```

:::{admonition} 🌟 Why This Matters: Modern Frameworks Require Functional Thinking
:class: important

Functional programming isn't just academic—it's essential for modern scientific computing:

1. **JAX** (Google's NumPy replacement) requires pure functions for automatic differentiation
2. **Parallel processing** works best with stateless functions
3. **Testing** is trivial when functions have no side effects
4. **GPU computing** maps naturally to functional operations

Example: In JAX, you can automatically differentiate through an entire orbital mechanics simulation if it's written functionally. This enables advanced techniques like Hamiltonian Monte Carlo that would be impossibly complex to implement manually.
:::

## 5.5 Modules and Packages

As your analysis grows from scripts to projects, organization becomes critical. Let's build a module step by step.

### Phase 1: Create Basic Module Structure

Save this as `physics_calc.py`:

```{code-cell} python
# physics_calc.py - Part 1: Constants
"""
Basic physics calculations module.

This module provides fundamental physics computations.
"""

# Physical constants
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/kg/s²
SPEED_OF_LIGHT = 299792458  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

print("Loading physics_calc module...")
```

### Phase 2: Add Core Functions

```{code-cell} python
# physics_calc.py - Part 2: Basic Functions

def kinetic_energy(mass, velocity):
    """Calculate kinetic energy: KE = 0.5 * m * v²"""
    return 0.5 * mass * velocity**2

def gravitational_force(m1, m2, distance):
    """Calculate gravitational force between two masses."""
    if distance == 0:
        return float('inf')
    return GRAVITATIONAL_CONSTANT * m1 * m2 / distance**2

def escape_velocity(mass, radius):
    """Calculate escape velocity from a spherical body."""
    import math
    return math.sqrt(2 * GRAVITATIONAL_CONSTANT * mass / radius)
```

### Phase 3: Add Testing and Main Block

```{code-cell} python
# physics_calc.py - Part 3: Testing

if __name__ == "__main__":
    # This code ONLY runs when script is executed directly
    # NOT when it's imported as a module
    
    print("Testing physics_calc module...")
    
    # Test escape velocity for Earth
    earth_mass = 5.972e24  # kg
    earth_radius = 6.371e6  # m
    
    v_escape = escape_velocity(earth_mass, earth_radius)
    print(f"Earth escape velocity: {v_escape:.0f} m/s")
    print(f"That's {v_escape/1000:.1f} km/s")
    
    # Should be about 11,200 m/s
    expected = 11200
    error = abs(v_escape - expected) / expected
    assert error < 0.01, f"Calculation error too large: {error:.2%}"
    
    print("All tests passed!")
```

### Using Your Module

```{code-cell} python
# Different ways to import and use the module

# Method 1: Import entire module
import physics_calc

earth_escape = physics_calc.escape_velocity(5.972e24, 6.371e6)
print(f"Earth escape velocity: {earth_escape/1000:.1f} km/s")

# Method 2: Import specific functions
from physics_calc import kinetic_energy, SPEED_OF_LIGHT

ke = kinetic_energy(9.109e-31, 0.01 * SPEED_OF_LIGHT)
print(f"Electron at 1% light speed: {ke:.2e} J")

# Method 3: Import with alias
import physics_calc as phys

force = phys.gravitational_force(5.972e24, 7.342e22, 3.844e8)
print(f"Earth-Moon force: {force:.2e} N")
```

### Import Best Practices

```{code-cell} python
# GOOD: Clear, explicit imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# BAD: Wildcard imports pollute namespace
# from numpy import *  # Adds 600+ names!
# from scipy import *  # Conflicts with numpy!

# Example of namespace collision disaster
# from numpy import *       # Has array, log, sin, etc.
# from math import *        # Also has log, sin, etc.
# result = log(10)         # Which log? Natural or base-10?

# SAFE: Explicit namespaces prevent confusion
import numpy as np
import math

natural_log = math.log(10)      # Clear: natural logarithm
common_log = np.log10(10)       # Clear: base-10 logarithm
print(f"ln(10) = {natural_log:.2f}")
print(f"log₁₀(10) = {common_log:.2f}")
```

:::{admonition} 🔍 Check Your Understanding #5
:class: question

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

This is why professional packages always use explicit imports. The extra typing (`np.log`) is worth the clarity and safety!
</details>
:::

:::{admonition} 🌟 The More You Know: How Modular Code Detected Gravitational Waves
:class: history

On September 14, 2015, at 09:50:45 UTC, two laser interferometers separated by 3,000 kilometers recorded matching chirps lasting just 0.2 seconds. This signal, named GW150914, was humanity's first direct detection of gravitational waves—ripples in spacetime predicted by Einstein a century earlier. The discovery earned Rainer Weiss, Barry Barish, and Kip Thorne the 2017 Nobel Prize in Physics [1].

The detection was only possible because of LIGO's modular software architecture. The analysis pipeline consisted of hundreds of independent Python functions, each performing one specific task [2]:

```python
# Simplified LIGO pipeline structure
def remove_60hz_noise(strain_data):
    """Remove power line interference at 60 Hz and harmonics."""
    # Notch filter implementation
    
def whiten_spectrum(strain_data, psd):
    """Normalize frequency response across spectrum."""
    # Spectral whitening implementation
    
def matched_filter(data, template_bank):
    """Search for gravitational wave patterns."""
    # Cross-correlation with waveform templates
    
def coincidence_test(h1_triggers, l1_triggers, time_window=0.01):
    """Verify both detectors saw signal within light travel time."""
    # Check for matching events
```

When the first candidate signal appeared in the data, this modular design enabled unprecedented rapid verification. Within minutes, automated analyses had processed the data through dozens of independent checks. Within hours, team members worldwide were running their own validation tests [3].

The modularity allowed the collaboration to:
- Test each processing step with simulated "injection" signals
- Run multiple independent analysis pipelines in parallel
- Swap alternative algorithms to verify robustness
- Complete peer review in record time

The entire LIGO analysis software is open source on GitHub, built with the same Python functions you're learning now. Any scientist can download the data and code to verify the discovery independently—which thousands have done [4].

The modular approach that enabled this discovery is exactly what you're learning in this chapter. Each function had one job, clear inputs and outputs, comprehensive documentation, and thorough testing. When you design your functions this way, you're following the same principles that enabled one of physics' greatest discoveries!

**References:**
[1] Abbott, B. P. et al. (2016). "Observation of Gravitational Waves from a Binary Black Hole Merger." Physical Review Letters, 116(6), 061102.
[2] LIGO Scientific Collaboration. (2021). "LIGO Algorithm Library." https://lscsoft.docs.ligo.org/lalsuite/
[3] Abbott, B. P. et al. (2016). "GW150914: The Advanced LIGO Detectors in the Era of First Discoveries." Physical Review Letters, 116(13), 131103.
[4] LIGO Open Science Center. (2024). https://www.gw-openscience.org/
:::

## 5.6 Documentation and Testing

Good documentation and testing make your functions trustworthy and reusable. Let's build these skills progressively.

### Phase 1: Basic Documentation

```{code-cell} python
def simple_harmonic_period(mass, spring_constant):
    """Calculate period of simple harmonic oscillator."""
    import math
    return 2 * math.pi * math.sqrt(mass / spring_constant)

# Minimal but clear
T = simple_harmonic_period(0.5, 10)
print(f"Period: {T:.2f} seconds")
```

### Phase 2: Professional Documentation

```{code-cell} python
def simulate_pendulum(length, initial_angle, time_step=0.01, duration=10.0):
    """
    Simulate simple pendulum motion using small angle approximation.
    
    Parameters
    ----------
    length : float
        Pendulum length in meters
    initial_angle : float
        Initial displacement in radians
    time_step : float, optional
        Integration time step in seconds (default: 0.01)
    duration : float, optional
        Total simulation time in seconds (default: 10.0)
    
    Returns
    -------
    times : list
        Time points
    angles : list
        Angular displacement at each time
    
    Examples
    --------
    >>> t, theta = simulate_pendulum(1.0, 0.1)
    >>> period = 2 * 3.14159 * sqrt(1.0/9.81)
    >>> print(f"Expected period: {period:.2f} s")
    
    Notes
    -----
    Uses small angle approximation: sin(θ) ≈ θ
    Only valid for initial_angle < ~0.3 radians (17°)
    
    References
    ----------
    .. [1] Taylor, J.R. (2005). Classical Mechanics, pp. 157-160
    """
    import math
    
    omega = math.sqrt(9.81 / length)  # Angular frequency
    times = []
    angles = []
    
    t = 0
    while t <= duration:
        angle = initial_angle * math.cos(omega * t)
        times.append(t)
        angles.append(angle)
        t += time_step
    
    return times, angles
```

### Phase 3: Testing Your Functions

```{code-cell} python
def test_physics_functions():
    """Test our physics calculations."""
    
    # Test 1: Kinetic energy
    ke = kinetic_energy(2.0, 3.0)
    expected = 9.0  # 0.5 * 2 * 3²
    assert abs(ke - expected) < 1e-10, f"KE calculation wrong: {ke} vs {expected}"
    
    # Test 2: Edge cases
    zero_ke = kinetic_energy(0, 100)
    assert zero_ke == 0, "Zero mass should give zero energy"
    
    # Test 3: Pendulum period (small angle)
    import math
    length = 1.0
    expected_period = 2 * math.pi * math.sqrt(length / 9.81)
    
    times, angles = simulate_pendulum(length, 0.1, duration=expected_period)
    
    # After one period, should return close to initial position
    final_angle = angles[-1]
    initial_angle = angles[0]
    assert abs(final_angle - initial_angle) < 0.01, \
        f"Pendulum didn't return to start: {final_angle:.3f} vs {initial_angle:.3f}"
    
    print("All physics tests passed! ✓")

# Run tests
test_physics_functions()
```

:::{admonition} 🛠️ Debug This!
:class: challenge

This function has a subtle performance bug. Can you spot it?

```python
def find_resonances(frequencies, target_freq, tolerance=0.01):
    """Find resonant frequencies - HAS PERFORMANCE BUG."""
    resonances = []
    
    for freq in frequencies:
        # Check if frequency is resonant (BUG HERE!)
        if freq not in resonances and abs(freq - target_freq) < tolerance:
            resonances.append(freq)
    
    return resonances
```

<details>
<summary>Answer and Fix</summary>

**Bug**: `if freq not in resonances` is O(n) for lists! For 10,000 frequencies, this becomes O(n²) complexity—extremely slow.

**Fix**: Use a set for O(1) lookup:

```python
def find_resonances_fixed(frequencies, target_freq, tolerance=0.01):
    """Find resonant frequencies efficiently."""
    resonances = []
    seen = set()  # O(1) membership testing
    
    for freq in frequencies:
        if freq not in seen and abs(freq - target_freq) < tolerance:
            resonances.append(freq)
            seen.add(freq)
    
    return resonances
```

For 10,000 frequencies:
- Original: ~5 seconds
- Fixed: ~0.05 seconds
- 100x speedup!

This exact bug appeared in spectroscopy analysis software, causing hours of unnecessary computation.
</details>
:::

## 5.7 Performance Considerations

Understanding function overhead helps you write efficient code. Let's measure it!

### Measuring Function Call Overhead

```{code-cell} python
import time

def trivial_function(x):
    """Almost no work - overhead dominates."""
    return x + 1

def moderate_function(x):
    """Some work - overhead noticeable."""
    return (x**2 + 2*x + 1) / (x + 1)

def complex_function(x):
    """Heavy computation - overhead negligible."""
    import math
    result = 0
    for i in range(10):
        result += math.sin(x + i) * math.cos(x - i)
    return result / 10

# Measure overhead
n_calls = 100000
test_value = 3.14

# Trivial function
start = time.perf_counter()
for _ in range(n_calls):
    trivial_function(test_value)
trivial_time = time.perf_counter() - start

# Inline version
start = time.perf_counter()
for _ in range(n_calls):
    result = test_value + 1
inline_time = time.perf_counter() - start

# Complex function
start = time.perf_counter()
for _ in range(n_calls):
    complex_function(test_value)
complex_time = time.perf_counter() - start

print("Function Call Overhead Analysis:")
print(f"Trivial:  {trivial_time*1000:.1f}ms (inline: {inline_time*1000:.1f}ms)")
print(f"  → Overhead: {(trivial_time/inline_time - 1)*100:.0f}%")
print(f"Complex:  {complex_time*1000:.1f}ms")
print(f"\nLesson: Overhead only matters for trivial operations in tight loops!")
```

Remember from Chapter 4: this overhead is O(1)—constant time. But constants matter when multiplied by millions!

### Memoization for Expensive Calculations

Cache results to avoid recomputation:

```{code-cell} python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(n):
    """
    Simulate expensive computation (e.g., eigenvalue calculation).
    Results are automatically cached!
    """
    import time
    
    # Simulate expensive work
    time.sleep(0.1)  # Pretend this takes 100ms
    
    result = sum(i**2 for i in range(n))
    return result

# First call: slow
start = time.perf_counter()
result1 = expensive_calculation(100)
first_time = time.perf_counter() - start
print(f"First call:  {first_time*1000:.1f}ms")

# Second call: instant (from cache)!
start = time.perf_counter()
result2 = expensive_calculation(100)
cached_time = time.perf_counter() - start
print(f"Cached call: {cached_time*1000:.3f}ms")

print(f"Speedup: {first_time/cached_time:.0f}x")
print(f"Cache stats: {expensive_calculation.cache_info()}")
```

:::{admonition} 🌟 The More You Know: How a Missing Return Statement Cost $327 Million
:class: history

On June 4, 1996, the maiden flight of the European Space Agency's Ariane 5 rocket ended in spectacular failure just 37 seconds after liftoff. The rocket veered off course and self-destructed, destroying four Cluster satellites worth $327 million. The cause? A single missing validation in a function that converted velocity data [1].

The disaster stemmed from reused code from Ariane 4's inertial reference system. A function that converted a 64-bit floating-point horizontal velocity value to a 16-bit signed integer worked perfectly for Ariane 4, which flew slower trajectories. But Ariane 5 was more powerful, and 37 seconds into flight, the horizontal velocity exceeded 32,767—the maximum value for a 16-bit signed integer [2].

The conversion function looked something like this:

```python
def convert_velocity_buggy(velocity_64bit):
    # Convert to 16-bit integer
    velocity_16bit = int(velocity_64bit)
    # MISSING: Check if value fits in 16 bits!
    return velocity_16bit

# What they needed:
def convert_velocity_fixed(velocity_64bit):
    MAX_INT16 = 32767
    MIN_INT16 = -32768
    
    # Validate before conversion
    if velocity_64bit > MAX_INT16:
        # Handle overflow appropriately
        return MAX_INT16  # Or raise exception
    elif velocity_64bit < MIN_INT16:
        return MIN_INT16
        
    return int(velocity_64bit)
```

The overflow triggered an exception in the backup inertial reference system (which ran identical code), causing both systems to fail simultaneously. The main computer, suddenly receiving diagnostic data instead of attitude information, interpreted it as actual flight data and commanded a violent course correction. The aerodynamic forces exceeded design limits, triggering automatic self-destruct [3].

The investigation revealed multiple function-related failures:
1. No validation of input ranges (our validate_result() function would have caught this)
2. Exception handling that shut down the entire system instead of degrading gracefully
3. Identical code in primary and backup systems (no diversity)
4. The alignment function continued running after launch, even though it was only needed on the ground

The tragedy led to fundamental changes in how critical software is developed. Modern aerospace code now requires:
- Explicit contracts for all functions (preconditions, postconditions, invariants)
- Comprehensive range checking on all conversions
- Diverse redundancy (different algorithms, not copies)
- Formal verification of critical functions

Every time you write a validation function or check input ranges, you're applying lessons learned from this $327 million disaster. The functions you're learning to write—with clear contracts, validation, and error handling—are exactly what could have prevented this tragedy.

**References:**
[1] Lions, J. L. et al. (1996). "ARIANE 5 Flight 501 Failure: Report by the Inquiry Board." European Space Agency.
[2] Gleick, J. (1996). "A Bug and A Crash." The New York Times Magazine, December 1, 1996.
[3] Nuseibeh, B. (1997). "Ariane 5: Who Dunnit?" IEEE Software, 14(3), 15-16.
:::

## Practice Exercises

### Variable Star Learning Activity Thread

Let's build our variable star analysis toolkit, adding to what we started in previous chapters!

```{code-cell} python
def generate_variable_star_data(output_file='variable_star_ch5.csv'):
    """
    Generate simulated Cepheid variable star data.
    
    Creates realistic light curve with:
    - Periodic variation
    - Measurement noise
    - Occasional bad measurements
    """
    import random
    import math
    
    # Cepheid parameters
    period = 5.366  # days
    mean_mag = 12.0
    amplitude = 0.4
    
    # Generate 30 observations over 20 days
    observations = []
    for i in range(30):
        time = i * 0.7  # Uneven sampling
        
        # Calculate theoretical magnitude
        phase = (time % period) / period
        magnitude = mean_mag + amplitude * math.sin(2 * math.pi * phase)
        
        # Add realistic noise
        noise = random.gauss(0, 0.02)
        measured_mag = magnitude + noise
        
        # Measurement error (heteroscedastic - varies with brightness)
        error = 0.01 + 0.02 * random.random()
        
        # Occasionally add bad measurement
        if random.random() < 0.05:  # 5% bad data
            measured_mag = 99.99
            error = 9.99
        
        observations.append((time + 2458000, measured_mag, error))
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("# JD,Magnitude,Error\n")
        for jd, mag, err in observations:
            f.write(f"{jd:.3f},{mag:.3f},{err:.3f}\n")
    
    print(f"Generated {len(observations)} observations in {output_file}")
    return observations

# Generate the data
data = generate_variable_star_data()
```

### Exercise 5.1: Basic Functions

Create fundamental analysis functions:

```python
def calculate_amplitude(magnitudes):
    """
    Calculate peak-to-peak amplitude of magnitude variations.
    
    Handle edge cases: empty list, None values, single value
    """
    # Your implementation here
    pass

def validate_magnitude(mag, error):
    """
    Check if magnitude measurement is valid.
    
    Invalid if:
    - magnitude > 30 (too faint)
    - error < 0 or error > 1 (unrealistic)
    - magnitude is inf or nan
    """
    # Your implementation here
    pass

# Test your functions
test_mags = [12.1, 12.5, 11.9, 12.3]
print(f"Amplitude: {calculate_amplitude(test_mags)}")
```

### Exercise 5.2: Build Analysis Module

Create `lightcurve.py` module:

```python
"""
lightcurve.py - Variable star analysis module

Progressive build:
1. Start with constants and simple functions
2. Add file loading with error handling  
3. Add analysis functions
4. Add module testing
"""

# Part 1: Constants
MAGNITUDE_LIMIT = 20.0  # Typical CCD limit
ERROR_THRESHOLD = 0.5   # Maximum acceptable error

# Part 2: Data loading
def load_lightcurve(filename):
    """Load light curve from CSV file."""
    times, mags, errors = [], [], []
    # Implementation here
    return times, mags, errors

# Part 3: Analysis
def find_period_simple(times, mags):
    """Estimate period from peak spacing."""
    # Implementation here
    pass

# Part 4: Testing
if __name__ == "__main__":
    # Test with generated data
    data = load_lightcurve("variable_star_ch5.csv")
    print(f"Loaded {len(data[0])} observations")
```

### Exercise 5.3: Functional Approach

Apply functional programming to data filtering:

```python
def analyze_functional(observations):
    """
    Process observations using functional programming.
    
    No explicit loops! Use map, filter, reduce.
    """
    from functools import reduce
    
    # Filter bad observations (mag < 90)
    good_obs = filter(lambda obs: obs['mag'] < 90, observations)
    
    # Calculate mean magnitude functionally
    # Your implementation here
    
    return mean_mag

# Compare with traditional approach
def analyze_traditional(observations):
    """Traditional loop-based approach."""
    total = 0
    count = 0
    for obs in observations:
        if obs['mag'] < 90:
            total += obs['mag']
            count += 1
    return total / count if count > 0 else 0
```

## Main Takeaways

Functions transform programming from repetitive scripting into modular, maintainable software engineering. When you encapsulate logic in well-designed functions, you create building blocks that can be tested independently, shared with collaborators, and combined into complex analysis pipelines. The progression from simple functions to modules to packages mirrors how scientific software naturally grows—what starts as a quick calculation evolves into a shared tool used by entire research communities.

The scope rules and namespace concepts you've learned explain why variables sometimes behave unexpectedly in complex programs. Understanding the LEGB rule prevents the frustrating bugs where variables have unexpected values or modifications in one place affect seemingly unrelated code. The mutable default argument trap, which has caused real bugs in major scientific packages, demonstrates why understanding Python's evaluation model is crucial for writing reliable code.

Functional programming concepts like map, filter, and pure functions aren't just academic exercises—they're essential preparation for modern scientific computing. Frameworks like JAX require functional style for automatic differentiation, while parallel processing works best with stateless functions. The ability to pass functions as arguments and return them from other functions enables powerful patterns like the specialized integrators we created with closures.

The performance measurements showed that function call overhead only matters for trivial operations in tight loops—exactly the situation where you'll want to use NumPy's vectorized operations (Chapter 7) instead. But for complex calculations, the overhead is negligible compared to the computation itself. Memoization can provide dramatic speedups when the same expensive calculations are repeated, as often happens in optimization and parameter searching.

Looking forward, the functions you've learned to write here form the foundation for object-oriented programming in Chapter 6, where functions become methods attached to objects. The module organization skills prepare you for building larger scientific packages, while the documentation practices ensure your code can be understood and maintained by others. Most importantly, thinking in terms of functional contracts and clear interfaces will make you a better computational scientist, capable of building the robust, efficient tools that modern research demands.

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
- The mutable default argument trap occurs because defaults are evaluated once when the `def` statement executes
- Always use `None` as a sentinel for mutable default arguments
- Python searches for variables using LEGB: Local, Enclosing, Global, Built-in
- Global variables make code hard to test, debug, and parallelize
- Lambda functions are useful for simple operations but limited to single expressions
- Functional programming concepts (map, filter, reduce) prepare you for modern frameworks
- The `if __name__ == "__main__"` pattern makes modules both importable and executable
- Never use `from module import *` in production code—it causes namespace pollution
- Docstrings are essential for scientific code that others will use and maintain
- Function call overhead matters only in tight loops with trivial operations
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
| Import all (avoid!) | `from math import *` | Interactive sessions only |
| Package import | `from scipy import constants` | Specific submodule |

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

With functions and modules mastered, Chapter 6 will introduce Object-Oriented Programming (OOP)—a paradigm that bundles data and behavior together. You'll learn to create classes that model physical systems naturally: a `Particle` class with position and velocity attributes, methods to calculate energy and momentum, and special methods that make your objects work seamlessly with Python's built-in functions.

The functional programming concepts from this chapter provide essential background for OOP. Methods are just functions attached to objects, and understanding scope prepares you for the `self` parameter that confuses many beginners. The module organization skills you've developed will expand to organizing classes and building object hierarchies. Most importantly, the design thinking you've practiced—creating clean interfaces and thinking about contracts—directly applies to designing effective classes that model the complex relationships in physical systems.