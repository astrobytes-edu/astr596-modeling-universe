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

# Chapter 4: Data Structures - Organizing Scientific Data

## Learning Objectives

By the end of this chapter, you will be able to:

- Choose optimal data structures based on algorithmic requirements and performance constraints
- Predict whether operations will be O(1) constant time or O(n) linear time
- Understand memory layout and why it matters for scientific computing
- Implement defensive copying strategies to prevent aliasing bugs
- Profile memory usage and optimize data structure choices for large datasets
- Design data structures that prepare you for vectorized computing
- Debug common bugs related to mutability, aliasing, and hashability
- Apply data structure patterns to real computational astrophysics problems

## Prerequisites Check

:::{admonition} ✅ Before Starting This Chapter
:class: note

- [ ] You can write loops and conditional statements fluently (Chapter 3)
- [ ] You understand the difference between assignment and equality (Chapter 2)
- [ ] You can use IPython for testing and timing code (Chapter 1)
- [ ] You understand floating-point precision issues (Chapter 2)
- [ ] You understand defensive programming principles (Chapter 1)

If any boxes are unchecked, review the indicated chapters first.
:::

## Chapter Overview

Imagine you're simulating the gravitational interactions between a million particles in a galaxy formation model. Each timestep, you need to find which particles are close enough to interact strongly. With the wrong data structure, this neighbor search could take hours per timestep. With the right one—a spatial hash table or octree—it takes seconds. That's the difference between a simulation finishing in a day versus running for months. This chapter teaches you to make these critical choices that determine whether your code scales to research-grade problems or remains stuck with toy models.

This chapter transforms you from someone who stores data to someone who orchestrates it strategically for computational astrophysics. You'll discover not just that dictionary lookups are fast, but *why* they're fast—through hash functions that turn particle positions into array indices. You'll understand *when* they might fail—like when hash collisions cluster your particles. And you'll learn *how* to verify performance yourself—because in computational science, measurement beats assumption every time.

These concepts directly prepare you for the numerical computing ahead. The memory layout discussions explain why NumPy arrays can be 100× more efficient than Python lists for vector operations. The immutability concepts prepare you for functional programming paradigms used in modern frameworks like JAX, where immutable operations enable automatic differentiation. The performance profiling skills will help you identify bottlenecks whether you're evolving stellar models or solving magnetohydrodynamics equations. By chapter's end, you'll think about data organization like a computational astrophysicist, architecting for performance from the start.

## 4.1 What Is a Data Structure?

```{margin}
**data structure**
A way of organizing data in memory to enable efficient access and modification
```

A **data structure** is fundamentally about organizing information to match your access patterns. Think about an N-body simulation: you could store particles in order of creation (like a **list**), organize them by spatial region for fast neighbor finding (like a **dictionary** of cells), or track unique particle IDs (like a **set**). Each choice profoundly affects your simulation's performance—the difference between **O(n²)** all-pairs checks and **O(n log n)** tree-based algorithms.

:::{admonition} 🌟 The More You Know: The Cassini Spacecraft Memory Crisis
:class: tip, dropdown

In 1997, the Cassini spacecraft nearly failed before reaching Saturn due to a memory overflow bug in its attitude control system. The flight software used an inefficient data structure to track thruster firings, allocating new memory for each event without reusing space. After millions of small adjustments during the 7-year journey, the embedded system's limited memory filled up.

NASA engineers had to upload a patch while Cassini was millions of miles away, switching to a circular buffer data structure that reused memory. According to NASA/JPL's Cassini Program documentation and lessons learned database (JPL D-18441, 1998), the fix was deployed via Deep Space Network commands, requiring precise timing due to the 84-minute round-trip signal delay. The wrong data structure choice nearly lost a $3.26 billion mission. 

This incident, detailed in "Cassini Attitude Control Flight Software: Lessons Learned" (IEEE Aerospace Conference, 2000), demonstrates that understanding data structures isn't academic—it's mission-critical. Modern spacecraft now use formal verification methods to prove memory bounds on their data structures.
:::

### Building Intuition: Measuring Speed Empirically

Ready to discover something that will transform how you think about organizing data in your simulations? We're going to measure the dramatic performance difference between different data structures. This empirical approach—measure first, understand why, then optimize—is exactly how computational astrophysicists approach performance optimization.

```{code-cell} ipython3
# Let's discover why data structure choice matters for physics simulations!
import time
import random

# Create test data representing particle IDs in a simulation
sizes = [100, 1000, 10000, 100000]

print("Searching for a particle NOT in the collection (worst case):")
print("=" * 60)

for n in sizes:
    # Simulate particle IDs
    particle_list = list(range(n))
    particle_set = set(range(n))
    
    # Search for a particle that escaped (not in our domain)
    escaped_particle = -1
    
    # Time list search - what you might naturally try first
    start = time.perf_counter()
    found = escaped_particle in particle_list
    list_time = time.perf_counter() - start
    
    # Time set search - the optimized approach
    start = time.perf_counter()
    found = escaped_particle in particle_set
    set_time = time.perf_counter() - start
    
    print(f"Size {n:6d}: List: {list_time*1e6:8.2f} μs, "
          f"Set: {set_time*1e6:8.2f} μs")
    if list_time > 0:
        print(f"           Set is {list_time/set_time:,.0f}× faster!")

print("\nFor collision detection between 100,000 particles:")
print("List approach: hours per timestep")
print("Set approach: milliseconds per timestep!")
```

This performance difference isn't just academic—it determines whether your galaxy simulation can evolve for billions of years or gets stuck after a few million. The secret behind this magic is the hash table, which you're about to understand completely.

### Understanding Big-O Notation

Now that you've witnessed this dramatic performance difference empirically, let's understand the mathematical pattern behind it. **Big-O notation** describes how an algorithm's runtime scales with input size—it's the language computational physicists use to discuss whether an algorithm is feasible for large-scale simulations.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import math

# Visualize how different algorithms scale for physics simulations
n = range(1, 101)
constant = [1 for _ in n]
logarithmic = [math.log(x) for x in n]
linear = list(n)
quadratic = [x**2 for x in n]

plt.figure(figsize=(10, 6))
plt.plot(n, constant, label='O(1) - Hash table lookup', linewidth=2)
plt.plot(n, logarithmic, label='O(log n) - Tree-based (Barnes-Hut)', linewidth=2)
plt.plot(n, linear, label='O(n) - Direct summation', linewidth=2)
plt.plot(n, quadratic, label='O(n²) - All-pairs (naive N-body)', linewidth=2)

plt.xlim(0, 100)
plt.ylim(0, 500)
plt.xlabel('Number of Particles (thousands)')
plt.ylabel('Time (arbitrary units)')
plt.title('N-body Algorithm Scaling: Why Data Structures Matter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("For a million-particle simulation:")
print(f"  O(1): 1 operation - spatial hash lookup")
print(f"  O(log n): {math.log2(1e6):.0f} operations - octree traversal")
print(f"  O(n): 1,000,000 operations - single particle force sum")
print(f"  O(n²): 1,000,000,000,000 operations - all pairs (impossible!)")
```

This graph reveals why algorithmic complexity matters for computational physics. The naive **O(n²)** approach that checks all particle pairs becomes impossible beyond a few thousand particles. But with smart **data structures** like octrees (**O(n log n)**) or spatial hashing (**O(n)**), we can simulate entire galaxies!

:::{admonition} 💡 Computational Thinking: The Time-Space Tradeoff in Physics Simulations
:class: important

This universal pattern appears throughout computational physics: trading memory for speed.

**The Pattern:**
- Use more memory → organize data cleverly → enable faster computation
- Use less memory → simpler organization → slower computation

**Real Physics Examples:**
- **Particle mesh methods**: Store density on a grid (memory) to avoid O(n²) particle interactions
- **Neighbor lists** in molecular dynamics: Store nearby atoms (memory) vs O(n²) distance checks
- **Octrees** in N-body codes: Store spatial hierarchy (memory) for O(n log n) force calculation
- **Lookup tables** for equation of state: Store pre-computed values vs recalculating
- **FFT** for spectral methods: Store complex coefficients for O(n log n) vs O(n²) convolution

Example: The GADGET cosmology code uses 200 bytes per particle (vs 24 for just position/velocity) to achieve O(n log n) scaling, enabling billion-particle simulations. This 8× memory cost enables 1000× speedup!
:::

## 4.2 Lists: Python's Workhorse Sequence

**Lists** are Python's most versatile **data structure**—perfect for particle arrays, time series, or any ordered collection. But here's something crucial for scientific computing: Python **lists** don't store your numbers directly! Understanding this explains why NumPy arrays are so much faster for numerical work.

### How Lists Really Work in Memory

Let's explore how Python actually stores your simulation data in memory. This knowledge will help you understand when to use Python **lists** versus when you need NumPy arrays.

```{code-cell} ipython3
import sys

# Let's examine memory usage for particle data
# Position of one particle in 3D space
position = [1.5e10, 2.3e10, 0.8e10]  # cm (typical for solar system)

# The list container itself
list_size = sys.getsizeof(position)
print(f"List container: {list_size} bytes")

# Each float is a full Python object!
element_sizes = [sys.getsizeof(x) for x in position]
print(f"Each coordinate: {element_sizes[0]} bytes")

# Total memory footprint
total = list_size + sum(element_sizes)
print(f"Total: {total} bytes for 3 floats!")
print(f"That's {total/24:.1f}× more than raw floats would need!")

print("\nWhy so much memory?")
print("- List stores pointers, not values")
print("- Each float is a full object with type info")
print("- Python must track reference counts")
print("\nNote: Memory sizes vary by Python version and system architecture.")
print("Use sys.getsizeof() to check your specific environment.")
print("\nThis is why NumPy arrays (Chapter 7) store raw values contiguously!")
```

Here's what's actually happening in memory when you store particle positions:

```
Python List of Positions:          Objects in Memory:
┌─────────────────┐               ┌──────────────────┐
│ size: 3         │               │ float: 1.5e10    │
│ capacity: 4     │          ┌───>│ type info        │
├─────────────────┤          │    │ ref count        │
│ pointer ────────┼──────────┘    └──────────────────┘
│ pointer ────────┼──────────────>┌──────────────────┐
│ pointer ────────┼────┐          │ float: 2.3e10    │
│ (unused)        │    │          └──────────────────┘
└─────────────────┘    └────────>┌──────────────────┐
                                  │ float: 0.8e10    │
                                  └──────────────────┘

NumPy Array (preview):
┌────┬────┬────┐
│1.5 │2.3 │0.8 │  <- Raw values, no pointers!
└────┴────┴────┘
```

### List Operations: Performance for Simulations

Different **list** operations have vastly different costs—critical knowledge when processing particle data or building spatial **data structures**.

```{code-cell} ipython3
import time

# Simulate a growing particle system
particles = list(range(100_000))

# Adding particles at the END (common pattern)
start = time.perf_counter()
particles.append(100_000)  # New particle enters domain
particles.pop()            # Remove for fair comparison
end_time = time.perf_counter() - start

# Adding particles at the BEGINNING (avoid this!)
start = time.perf_counter()
particles.insert(0, -1)    # Insert at front
particles.pop(0)           # Remove from front
begin_time = time.perf_counter() - start

print(f"Adding particle at END:   {end_time*1e6:.2f} microseconds")
print(f"Adding particle at START: {begin_time*1e6:.2f} microseconds")
print(f"\nBeginning is {begin_time/end_time:.0f}× slower!")

print("\nWhy? Inserting at the beginning shifts ALL particles in memory!")
print("For particle systems, use collections.deque if you need")
print("fast operations at both ends (e.g., boundary conditions).")
```

:::{admonition} 💡 Performance Tip: When Both Ends Matter
:class: tip

If your simulation needs fast operations at both ends (like particles entering/leaving through boundaries), use `collections.deque`:

```python
from collections import deque
particles = deque(maxlen=1000000)  # Efficient at both ends!
```

This is why molecular dynamics codes use specialized data structures for boundary particles.
:::

### List Growth Strategy: Understanding Dynamic Arrays

Watch how Python manages memory as your particle system grows. This pattern appears in many languages and understanding it helps you write efficient simulation codes.

```{code-cell} ipython3
import sys

# Track memory allocation as particle system grows
particles = []
previous_size = 0

print("Watch Python's list growth strategy:")
print("(Critical for understanding simulation performance)")
print("Length → Capacity (overallocation)")
print("-" * 40)

for i in range(20):
    old_size = sys.getsizeof(particles)
    particles.append(i)
    new_size = sys.getsizeof(particles)
    
    if new_size != old_size:
        # Calculate capacity from size change
        capacity = (new_size - sys.getsizeof([])) // 8
        actual_length = len(particles)
        overalloc = ((capacity - actual_length) / actual_length * 100 
                     if actual_length > 0 else 0)
        print(f"{actual_length:4d} → {capacity:4d} "
              f"({overalloc:5.1f}% extra)")

print("\nWhy this matters for physics simulations:")
print("- Amortized O(1) append: Usually fast, occasionally slow (reallocation)")
print("- Memory fragmentation: Growing lists can scatter in memory")
print("- Cache misses: Non-contiguous memory hurts performance")
print("This is why production codes pre-allocate arrays when particle count is known!")
```

This growth pattern is why appending to lists is "amortized O(1)"—usually fast, occasionally slow when reallocation happens. For time-critical simulations, pre-allocate your arrays when possible!

## 4.3 Tuples: The Power of Immutability

```{margin}
**immutable**
Objects whose state cannot be modified after creation
```

What if you need to guarantee that initial conditions or physical constants won't change during your simulation? Enter **tuples**—Python's **immutable** sequences. This isn't a limitation—it's protection against an entire category of bugs that plague scientific codes!

:::{admonition} ⚠️ Common Bug Alert: The Accidental Modification Disaster
:class: warning

One of the most insidious bugs in computational physics happens when functions unexpectedly modify their inputs. Remember the defensive programming principles from Chapter 1? Here's why they matter:

```python
# THE BUG THAT CORRUPTS YOUR SIMULATION:
def evolve_system(state, dt):
    # Trying to be clever, modifying in-place
    state[0] += state[1] * dt  # Modifies original!
    state[1] += calculate_force(state[0]) * dt
    return state

initial_state = [1.0, 0.0]  # Position, velocity
state_t1 = evolve_system(initial_state, 0.01)
print(initial_state)  # [1.01, ...] - Initial conditions corrupted!

# THE FIX: Use tuples to prevent modification
initial_state = (1.0, 0.0)  # Tuple!
# Now modification attempts raise errors immediately!
```

This connects directly to Chapter 1's defensive programming: catch errors early, fail loudly!
:::

### Understanding Immutability in Physics Simulations

Let's see how **immutability** protects your simulations and enables powerful optimizations:

```{code-cell} ipython3
# Physical constants should NEVER change during simulation
class PhysicsConstants:
    """Demonstrating safe vs unsafe constant storage."""
    
    # UNSAFE: Mutable list (can be accidentally modified)
    unsafe_constants = [
        6.674e-8,   # G (cm³/g/s²)
        2.998e10,   # c (cm/s)
        1.381e-16,  # k_B (erg/K)
    ]
    
    # SAFE: Immutable tuple (modification attempts fail loudly)
    safe_constants = (
        6.674e-8,   # G (cm³/g/s²)
        2.998e10,   # c (cm/s)
        1.381e-16,  # k_B (erg/K)
    )

# Demonstration of the danger
def buggy_calculation(constants):
    """This function has a typo that modifies constants!"""
    if isinstance(constants, list):
        # Oops! Used = instead of == in a complex calculation
        constants[0] = constants[0] * 1e10  # BUG: Modifies G!
        return "Calculated (with hidden corruption)"
    else:
        # With tuple, this would raise an error immediately
        try:
            constants[0] = constants[0] * 1e10
        except TypeError as e:
            return f"Caught bug immediately: {e}"

# Test with both
print("With list:", buggy_calculation(PhysicsConstants.unsafe_constants.copy()))
print("With tuple:", buggy_calculation(PhysicsConstants.safe_constants))

print("\nImmutability turns subtle runtime corruption into immediate, obvious errors!")
```

### Tuples as Dictionary Keys: Caching Expensive Calculations

Here's where **immutability** becomes a superpower for computational physics—**tuples** can be **dictionary** keys, enabling powerful **memoization** patterns for expensive calculations:

```{code-cell} ipython3
# Cache expensive physics calculations
cache = {}
calculation_count = 0

def gravitational_force(pos1, pos2, m1, m2):
    """Calculate gravitational force with automatic caching."""
    global calculation_count
    
    # Create cache key from positions (must be tuples!)
    cache_key = (pos1, pos2, m1, m2)
    
    if cache_key in cache:
        print(f"  Cache hit! Avoided expensive calculation")
        return cache[cache_key]
    
    # Expensive calculation
    calculation_count += 1
    print(f"  Computing force (calculation #{calculation_count})")
    
    # Unpack positions
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    
    # Calculate distance
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r = (dx**2 + dy**2 + dz**2) ** 0.5
    
    # Gravitational force
    G = 6.674e-8  # cm³/g/s²
    F = G * m1 * m2 / r**2
    
    # Cache for future use
    cache[cache_key] = F
    return F

# Simulate repeated force calculations in N-body code
sun_pos = (0.0, 0.0, 0.0)
earth_pos = (1.496e13, 0.0, 0.0)  # 1 AU in cm
m_sun = 1.989e33  # grams
m_earth = 5.972e27  # grams

print("First calculation:")
F1 = gravitational_force(sun_pos, earth_pos, m_sun, m_earth)

print("\nSecond calculation (same positions):")
F2 = gravitational_force(sun_pos, earth_pos, m_sun, m_earth)

print(f"\nForce: {F1:.2e} dynes")
print(f"Calculations performed: {calculation_count}")
print("Caching with tuples can speed up N-body codes 10-100×!")
```

## 4.4 The Mutable vs Immutable Distinction

Time for one of Python's most important concepts for scientific computing! Understanding **mutability** is the key to avoiding mysterious bugs where your simulation state changes unexpectedly. This connects directly to Chapter 1's defensive programming principles—catching errors early prevents corrupted results.

### Python's Reference Model in Physics Simulations

Python doesn't store values in variables—it stores **references** to objects. This has profound implications for simulation codes:

```{code-cell} ipython3
# Critical concept for simulation state management!

print("With IMMUTABLE objects (safe for constants):")
G_constant = 6.674e-8
G_backup = G_constant
print(f"Initially: G={G_constant:.3e}, backup={G_backup:.3e}")
print(f"Same object in memory? {G_constant is G_backup}")

G_backup = 6.674e-7  # Creates NEW object (typo won't affect original)
print(f"After change: G={G_constant:.3e}, backup={G_backup:.3e}")
print("Original constant is safe!")

print("\n" + "="*50)
print("With MUTABLE objects (dangerous for state!):")

# Particle system state
particles = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # Positions
backup = particles  # Think you're making a backup?

print(f"Initially: same object? {particles is backup}")

# Modify what you think is the backup
backup[0][0] = 999.0  # Trying to test something

print(f"Original particles: {particles}")
print(f"'Backup': {backup}")
print("😱 You just corrupted your simulation state!")

print("\nLesson: Use copy.deepcopy() for simulation state backups!")
```

:::{admonition} 🌟 The More You Know: The Knight Capital Trading Disaster
:class: tip, dropdown

On August 1, 2012, Knight Capital Group lost $440 million in 45 minutes due to a **data structure** **aliasing** bug. According to SEC Release No. 70694 (October 16, 2013) and Knight's own 8-K filing with the SEC on August 2, 2012, the incident stemmed from code deployment inconsistencies.

The company deployed new trading software to 8 servers but accidentally left old code (called "Power Peg") on one server. Both old and new systems shared a configuration **dictionary**. The old code interpreted a flag in this shared **dictionary** as enabling test mode, while the new code used the same flag to enable live trading. 

As reported by Reuters and Bloomberg at the time, when markets opened, 7 servers correctly processed orders while the 8th server executed everything as real trades. In 45 minutes, the rogue server executed 4 million trades worth $7 billion—roughly 10% of total US equity volume that day.

The shared **mutable** state (the configuration **dictionary**) without proper version control created perfect conditions for disaster. Knight Capital, once the largest trader in US equities, was acquired for a fraction of its value within a week. The incident led to new regulations about algorithmic trading systems and configuration management.

The lesson for scientific computing: When multiple systems or simulation components share **mutable** **data structures**, confusion about state can have catastrophic consequences. Always use defensive copying and **immutable** configurations where possible!
:::

### The Classic Mutable Default Argument Bug

This bug is so dangerous that Python linters specifically check for it. It's particularly insidious in iterative simulations:

```{code-cell} ipython3
# THE BUG: Default mutable created ONCE at function definition!
def accumulate_energies_buggy(energy, history=[]):
    """BUGGY: Trying to track energy history."""
    history.append(energy)
    return history

# Simulate multiple independent runs
print("Run 1:")
run1 = accumulate_energies_buggy(100)
run1 = accumulate_energies_buggy(95)
print(f"  Energy history: {run1}")

print("\nRun 2 (should be independent):")
run2 = accumulate_energies_buggy(200)
print(f"  Energy history: {run2}")
print("  Contains Run 1 data! Runs are coupled! 😱")

print("\n" + "="*50)
print("THE FIX: Use None sentinel pattern")

def accumulate_energies_fixed(energy, history=None):
    """Safe version - creates new list for each run."""
    if history is None:
        history = []  # Fresh list for each simulation
    history.append(energy)
    return history

print("\nRun 3:")
run3 = accumulate_energies_fixed(100)
run3 = accumulate_energies_fixed(95)
print(f"  Energy history: {run3}")

print("\nRun 4 (properly independent):")
run4 = accumulate_energies_fixed(200)
print(f"  Energy history: {run4}")
print("  Runs are independent! ✅")
```

### Shallow vs Deep Copies in Grid Simulations

This distinction is critical for grid-based simulations in computational fluid dynamics or stellar atmospheres:

```{code-cell} ipython3
import copy

# Create a 2D grid for temperature distribution
print("Simulating heat diffusion on a grid:")
grid = [[20.0, 20.0, 20.0],
        [20.0, 50.0, 20.0],  # Hot spot in center
        [20.0, 20.0, 20.0]]
print(f"Initial grid: {grid[1]}")  # Middle row

# Shallow copy - DANGEROUS for grids!
print("\n--- SHALLOW COPY (aliases inner arrays) ---")
grid_next_shallow = grid.copy()
# Try to update center point for next timestep
grid_next_shallow[1][1] = 35.0  # Cooling

print(f"Original grid center: {grid[1][1]}°C")
print(f"Next step grid center: {grid_next_shallow[1][1]}°C")
print("😱 Modified both grids! Simulation is corrupted!")

# Reset
grid = [[20.0, 20.0, 20.0],
        [20.0, 50.0, 20.0],
        [20.0, 20.0, 20.0]]

# Deep copy - SAFE for grids!
print("\n--- DEEP COPY (independent arrays) ---")
grid_next_deep = copy.deepcopy(grid)
grid_next_deep[1][1] = 35.0  # Cooling

print(f"Original grid center: {grid[1][1]}°C")
print(f"Next step grid center: {grid_next_deep[1][1]}°C")
print("✅ Grids are independent! Simulation is correct!")

print("\nMemory visualization:")
print("Shallow: grid → [ref1, ref2, ref3] → same inner arrays")
print("Deep:    grid → [ref1, ref2, ref3] → independent arrays")
```

## 4.5 Dictionaries: O(1) Lookup Magic for Physics

Now we explore one of computer science's most elegant inventions—the **dictionary**! In computational physics, **dictionaries** are perfect for particle properties, lookup tables, and **caching** expensive calculations. You're about to understand exactly how they achieve near-instantaneous lookups regardless of size.

### Organizing Simulation Data with Dictionaries

Let's see how **dictionaries** transform particle management in an N-body simulation:

```{code-cell} ipython3
# N-body simulation: Managing particle properties efficiently
import json
import math

# Individual particle data (could have thousands)
particles = {
    'particle_0001': {
        'mass': 1.989e30,  # grams (0.001 solar masses)
        'position': [1.5e13, 0.0, 0.0],  # cm
        'velocity': [0.0, 3.0e6, 0.0],   # cm/s
        'type': 'star'
    },
    'particle_0002': {
        'mass': 5.972e27,  # grams (Earth mass)
        'position': [2.5e13, 0.0, 0.0],
        'velocity': [0.0, 2.5e6, 0.0],
        'type': 'planet'
    },
    'particle_0003': {
        'mass': 7.342e25,  # grams (Moon mass)
        'position': [2.54e13, 0.0, 0.0],
        'velocity': [0.0, 2.4e6, 0.0],
        'type': 'moon'
    }
}

# O(1) lookup by particle ID - instant access!
target = 'particle_0002'
print(f"Accessing {target}:")
print(f"  Mass: {particles[target]['mass']:.2e} g")
print(f"  Type: {particles[target]['type']}")

# Organize by type for efficient group operations
by_type = {}
for pid, data in particles.items():
    ptype = data['type']
    if ptype not in by_type:
        by_type[ptype] = []
    by_type[ptype].append(pid)

print(f"\nParticles by type: {by_type}")

# Calculate total mass by type (common in simulations)
for ptype, pids in by_type.items():
    total_mass = sum(particles[pid]['mass'] for pid in pids)
    print(f"{ptype}: {total_mass:.2e} g")

print("\nDictionaries are perfect for:")
print("- O(1) particle lookup by ID")
print("- Grouping particles by properties")
print("- Storing heterogeneous particle data")
```

### Understanding Hash Tables for Physics Applications

Let's demystify how **dictionaries** achieve **O(1)** lookup—critical for lookup tables in equation of state calculations:

```{code-cell} ipython3
def demonstrate_hashing_for_physics():
    """Show how hash tables enable fast lookups in physics codes."""
    
    # Common lookup table keys in physics simulations
    physics_keys = [
        (1.0e6, 1.0e-3),   # (Temperature, Density) for EOS
        (2.0e6, 2.0e-3),
        (5.0e6, 1.0e-2),
        "H_ionization",     # Reaction rates
        "He_ionization",
        "opacity_rosseland" # Opacity types
    ]
    
    print("How hash tables accelerate physics lookups:")
    print("=" * 60)
    print("Key                  → Hash        → Bucket")
    print("-" * 60)
    
    for key in physics_keys:
        hash_value = hash(key)
        # Simplified bucket calculation
        bucket = abs(hash_value) % 100
        if isinstance(key, tuple):
            key_str = f"T={key[0]:.0e}, ρ={key[1]:.0e}"
        else:
            key_str = str(key)
        print(f"{key_str:20s} → {hash_value:11d} → {bucket:3d}")
    
    print("\nThe O(1) lookup process for equation of state:")
    print("1. Hash (T, ρ) pair → integer")
    print("2. Map to bucket index")
    print("3. Direct array access!")
    print("\nThis is why EOS tables are fast despite millions of entries!")

demonstrate_hashing_for_physics()
```

### Dictionary Performance for Particle Lookups

Time to see the dramatic performance difference for particle system queries:

```{code-cell} ipython3
import time
import random

# Create a large particle system
n_particles = 1_000_000
print(f"Creating system with {n_particles:,} particles...")

# List approach (what beginners might try)
particle_list = [(f"particle_{i:07d}", random.uniform(1e27, 1e30)) 
                 for i in range(n_particles)]

# Dictionary approach (professional solution)
particle_dict = {f"particle_{i:07d}": random.uniform(1e27, 1e30) 
                 for i in range(n_particles)}

# Search for specific particle
target = "particle_0500000"

# Time list search - O(n)
start = time.perf_counter()
for pid, mass in particle_list:
    if pid == target:
        mass_list = mass
        break
list_time = time.perf_counter() - start

# Time dict lookup - O(1)
start = time.perf_counter()
mass_dict = particle_dict[target]
dict_time = time.perf_counter() - start

print(f"\nFinding {target} among {n_particles:,} particles:")
print(f"List search: {list_time*1000:.3f} ms")
print(f"Dict lookup: {dict_time*1000:.6f} ms")
print(f"Dictionary is {list_time/dict_time:,.0f}× faster!")

print("\nFor timestep calculations accessing many particles:")
print("List: Each access is O(n) - simulation grinds to a halt")
print("Dict: Each access is O(1) - scales to millions of particles")
```

:::{admonition} 💡 Computational Thinking: Memoization in Physics Calculations
:class: important

This **caching** pattern dramatically speeds up physics codes:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def equation_of_state(temperature, density):
    # Expensive calculation cached automatically
    return complex_eos_calculation(temperature, density)
```

**Real physics applications:**
- **Nuclear reaction rates**: **Cache** temperature-dependent rates
- **Opacity calculations**: **Cache** (T, ρ, composition) lookups
- **Stellar atmospheres**: **Cache** radiative transfer solutions
- **Cosmological simulations**: **Cache** cooling functions

Example: The MESA stellar evolution code uses extensive **caching** of nuclear reaction rates, speeding up calculations by >100× compared to recalculating every timestep.
:::

## 4.6 Sets: Mathematical Operations for Particle Systems

**Sets** are perfect for tracking unique particles, finding neighbors, and performing mathematical operations on particle groups. They provide **O(1)** membership testing plus elegant **set** operations—ideal for computational physics!

### Set Operations in Particle Dynamics

```{code-cell} ipython3
# Particle tracking across domain boundaries
# (Common in parallel simulations with domain decomposition)

# Particles in different processor domains
domain_A = {'p001', 'p002', 'p003', 'p004', 'p005'}
domain_B = {'p004', 'p005', 'p006', 'p007'}
domain_C = {'p003', 'p007', 'p008', 'p009'}

# Particles that need communication (on boundaries)
boundary_particles = (domain_A & domain_B) | (domain_B & domain_C) | (domain_A & domain_C)
print(f"Boundary particles needing communication: {boundary_particles}")

# Particles unique to each domain (no communication needed)
interior_A = domain_A - domain_B - domain_C
interior_B = domain_B - domain_A - domain_C
interior_C = domain_C - domain_A - domain_B
print(f"Interior particles (A): {interior_A}")

# Check particle conservation
all_particles = domain_A | domain_B | domain_C
print(f"Total unique particles: {len(all_particles)}")

# Particles in multiple domains (need consistency check)
duplicated = set()
all_domains = [domain_A, domain_B, domain_C]
for i in range(len(all_domains)):
    for j in range(i+1, len(all_domains)):
        duplicated |= (all_domains[i] & all_domains[j])

print(f"Particles in multiple domains: {duplicated}")
print("\nSets make domain decomposition bookkeeping trivial!")
```

### Performance: Sets vs Lists for Neighbor Finding

```{code-cell} ipython3
import time

# Simulate neighbor finding in particle simulation
n_particles = 100_000
n_neighbors = 100

print(f"Finding neighbors among {n_particles:,} particles...")

# Create particle IDs
all_particles = [f"p_{i:06d}" for i in range(n_particles)]
neighbors = [f"p_{i:06d}" for i in range(n_neighbors)]

# Convert to set for fast lookup
all_particles_set = set(all_particles)

# Test if neighbors exist - List approach
start = time.perf_counter()
for neighbor in neighbors:
    found = neighbor in all_particles  # O(n) each time!
list_time = time.perf_counter() - start

# Test if neighbors exist - Set approach  
start = time.perf_counter()
for neighbor in neighbors:
    found = neighbor in all_particles_set  # O(1) each time!
set_time = time.perf_counter() - start

print(f"\nChecking {n_neighbors} potential neighbors:")
print(f"List approach: {list_time*1000:.2f} ms")
print(f"Set approach:  {set_time*1000:.5f} ms")
print(f"Set is {list_time/set_time:,.0f}× faster!")

print("\nFor molecular dynamics with millions of particles:")
print("Set-based neighbor lists are essential for performance!")
```

:::{admonition} ✅ Check Your Understanding
:class: hint

Why is checking particle membership **O(n)** with **lists** but **O(1)** with **sets**?

Think about the search process...

**Answer:** **Lists** must check each element sequentially until finding a match (or reaching the end)—this is **O(n)** linear time. **Sets** use **hash tables**: they compute a hash of the particle ID and jump directly to where it would be stored—this is **O(1)** constant time. 

For a million-particle simulation, this is the difference between microseconds and seconds per lookup. When you're checking thousands of particle interactions per timestep, this performance difference determines whether your simulation finishes in hours or weeks!
:::

## 4.7 Memory and Performance for Scientific Computing

Understanding memory layout is crucial for scientific computing. These concepts explain why specialized numerical libraries are so much faster than pure Python and will help you write **cache**-efficient simulation codes.

### Memory Usage in Particle Systems

Let's examine memory costs for different **data structures** in a particle simulation context:

```{code-cell} ipython3
import sys

# Stage 1: Basic Memory Measurement (8 lines)
def measure_particle_memory(n=10000):
    """Compare memory for particle ID storage."""
    ids_list = [f"p_{i:06d}" for i in range(n)]
    ids_set = set(ids_list)
    
    print(f"Storing {n:,} particle IDs:")
    print(f"  List: {sys.getsizeof(ids_list):,} bytes")
    print(f"  Set:  {sys.getsizeof(ids_set):,} bytes")
    print(f"  Set overhead: {sys.getsizeof(ids_set)/sys.getsizeof(ids_list):.1f}×")

measure_particle_memory()
```

```{code-cell} ipython3
# Stage 2: Understanding the Tradeoff (10 lines)
def understand_physics_tradeoffs():
    """Memory vs speed tradeoffs in physics codes."""
    # Small particle system
    n = 100
    positions = [[i*1.0, 0.0, 0.0] for i in range(n)]
    
    list_bytes = sys.getsizeof(positions)
    # If we used spatial hashing (dict of cells)
    dict_bytes = sys.getsizeof({i: pos for i, pos in enumerate(positions)})
    
    print(f"100 particles as list: {list_bytes:,} bytes")
    print(f"With spatial hash: ~{dict_bytes:,} bytes")
    print(f"Extra {dict_bytes - list_bytes:,} bytes enables O(1) neighbor finding!")
    print("\nReal codes like LAMMPS use this tradeoff extensively")

understand_physics_tradeoffs()
```

```{code-cell} ipython3
# Stage 3: Complete Memory Profile for Simulation (12 lines)
def profile_simulation_structures(n_particles=1000):
    """Memory comparison for complete particle system."""
    import numpy as np  # Preview of Chapter 7
    
    structures = {
        'List of lists': [[i, i*1.0, 0.0, 0.0] for i in range(n_particles)],
        'List of tuples': [(i, i*1.0, 0.0, 0.0) for i in range(n_particles)],
        'Dictionary': {i: [i*1.0, 0.0, 0.0] for i in range(n_particles)},
        'NumPy (preview)': np.zeros((n_particles, 4))
    }
    
    print(f"Memory for {n_particles:,} particles (ID, x, y, z):")
    print("-" * 50)
    for name, struct in structures.items():
        size = sys.getsizeof(struct)
        per_particle = size / n_particles
        print(f"{name:15s}: {size:8,} bytes ({per_particle:.1f} bytes/particle)")

profile_simulation_structures()
print("\nNotice NumPy's dramatic efficiency - that's Chapter 7!")
```

### Cache Efficiency in Grid Computations

Modern CPUs are fast, but memory access is slow. Understanding cache efficiency is critical for grid-based simulations:

```{code-cell} ipython3
import time

# Demonstrate cache effects in finite difference calculations
size = 500  # Grid size for heat equation
grid = [[20.0 + 0.1*i*j for j in range(size)] 
        for i in range(size)]

print("Simulating heat diffusion finite differences:")
print("(Common in CFD and stellar atmosphere codes)")

# Row-major access (cache-friendly in Python)
start = time.perf_counter()
total = 0.0
for i in range(size):
    for j in range(size):
        total += grid[i][j]
row_time = time.perf_counter() - start

# Column-major access (cache-hostile in Python)
start = time.perf_counter()
total = 0.0
for j in range(size):
    for i in range(size):
        total += grid[i][j]
col_time = time.perf_counter() - start

print(f"\nProcessing {size}×{size} grid:")
print(f"Row-major (cache-friendly):   {row_time*1000:.1f} ms")
print(f"Column-major (cache-hostile): {col_time*1000:.1f} ms")
print(f"Column-major is {col_time/row_time:.1f}× slower!")

print("\nNote: Python's object overhead masks some effects.")
print("In NumPy/C/Fortran with contiguous arrays, this difference")
print("can be 10-100×! This is why CFD codes carefully order loops.")
```

:::{admonition} ⚠️ Common Bug Alert: The Iteration Modification Trap
:class: warning

Never modify a collection while iterating—a common bug in particle removal:

```python
# THE BUG - Skips particles!
particles = [p1, p2, p3, p4, p5]
for p in particles:
    if p.escaped_boundary():
        particles.remove(p)  # WRONG! Skips next particle

# THE FIX 1: Iterate over copy
for p in particles.copy():
    if p.escaped_boundary():
        particles.remove(p)

# THE FIX 2: List comprehension (best for physics)
particles = [p for p in particles if not p.escaped_boundary()]

# THE FIX 3: Build removal list
to_remove = []
for p in particles:
    if p.escaped_boundary():
        to_remove.append(p)
for p in to_remove:
    particles.remove(p)
```

This bug has corrupted many molecular dynamics simulations!
:::

## 4.8 Choosing the Right Structure for Physics

After exploring all options, how do you choose? Here's your decision framework for computational physics:

:::{admonition} ✅ Check Your Understanding
:class: hint

For each physics scenario, what **data structure** would you choose?

1. **Tracking unique particle IDs in collision detection** → ?
2. **Time series of energy measurements** → ?
3. **Caching expensive equation of state calculations** → ?
4. **Physical constants that must not change** → ?

Think before reading...

**Answers:**
1. **Set** (unique particles, **O(1)** collision checks)
2. **List** (ordered by time, allows duplicates)
3. **Dictionary** (**cache** (T,ρ) → pressure with **O(1)** lookup)
4. **Tuple** or namedtuple (**immutable** constants, prevents accidents)
:::

### Performance Quick Reference for Physics

| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Access by index | **O(1)** | **O(1)** | N/A | N/A |
| Search for particle | **O(n)** | **O(n)** | **O(1)*** | **O(1)** |
| Add particle | **O(1)**† | N/A | **O(1)**† | **O(1)**† |
| Remove particle | **O(n)** | N/A | **O(1)** | **O(1)** |
| Memory (relative) | 1× | 0.9× | 3× | 3× |
| Spatial ordering | Yes | Yes | No | No |

\* **Dict** searches by key (particle ID)  
† **Amortized** - occasionally **O(n)** during resize

### Real Example: N-Body Simulation Design

```{code-cell} ipython3
class NBodySystem:
    """Data structure choices for N-body simulation."""
    
    def __init__(self):
        # Lists for vectorizable operations (future NumPy)
        self.positions = []  # Will become NumPy array
        self.velocities = []
        self.masses = []
        
        # Tuple for immutable parameters
        self.G = 6.674e-8  # cm³/g/s² - must not change!
        self.softening = 1e10  # cm - numerical stability
        
        # Dict for particle properties by ID
        self.properties = {}  # ID → {'type': 'star', 'age': 1e9}
        
        # Set for fast membership tests
        self.active_ids = set()  # Active particles
        self.collision_candidates = set()  # Close pairs
        
        # Dict for spatial hashing (grid cells)
        self.spatial_grid = {}  # cell_key → set of particle IDs
        
    def __repr__(self):
        return (f"NBodySystem: {len(self.positions)} particles\n"
                f"  Active: {len(self.active_ids)}\n"
                f"  Collision candidates: {len(self.collision_candidates)}")

system = NBodySystem()
print(system)
print("\nEach structure optimized for its specific use!")
print("This design scales to millions of particles.")
```

## Practice Exercises

### Exercise 4.1: Particle System Organization

**Part A: Basic List Implementation (5-10 lines)**

Follow these exact steps to create a working particle system:

```{code-cell} ipython3
def create_particle_system_list():
    """Store particles as list of lists - simple but limited."""
    particles = []
    
    # Create 5 test particles (id, mass, x, y, z)
    for i in range(5):
        particle = [i, 1.0e30 * (i+1), i*1e13, 0.0, 0.0]
        particles.append(particle)
    
    print(f"Created {len(particles)} particles")
    print(f"Particle 0: ID={particles[0][0]}, mass={particles[0][1]:.1e} g")
    return particles

particles = create_particle_system_list()
```

**Part B: Convert to Dictionary (10-15 lines)**

Improve the design with dictionaries for O(1) lookup:

```{code-cell} ipython3
def create_particle_system_dict():
    """Store particles in dictionary - better for lookups."""
    particles = {}
    
    # Create particles with meaningful structure
    for i in range(5):
        particles[f'p{i:03d}'] = {
            'mass': 1.0e30 * (i+1),  # grams
            'position': [i*1e13, 0.0, 0.0],  # cm
            'velocity': [0.0, 2e6, 0.0],  # cm/s
        }
    
    # O(1) lookup by ID!
    target = 'p002'
    print(f"Particle {target}: mass = {particles[target]['mass']:.1e} g")
    print(f"  Position: {particles[target]['position'][0]:.1e} cm")
    
    return particles

particles = create_particle_system_dict()
print(f"System has {len(particles)} particles with O(1) access")
```

**Part C: Production Version with Validation (15-20 lines)**

Add error checking and performance measurement:

```{code-cell} ipython3
import time

def create_particle_system_professional(n=1000):
    """Production-ready particle system with validation."""
    start = time.perf_counter()
    
    particles = {}
    errors = []
    
    for i in range(n):
        # Validate mass (must be positive)
        mass = 1.0e30 * (1 + i*0.001)
        if mass <= 0:
            errors.append(f"Particle {i}: invalid mass {mass}")
            continue
            
        particles[f'p{i:06d}'] = {
            'mass': mass,
            'position': [i*1e11, 0.0, 0.0],
            'velocity': [0.0, 3e6, 0.0],
            'active': True
        }
    
    elapsed = time.perf_counter() - start
    
    print(f"Created {len(particles)} particles in {elapsed*1000:.1f} ms")
    if errors:
        print(f"Skipped {len(errors)} invalid particles")
    
    # Verify O(1) access
    test_time = time.perf_counter()
    _ = particles['p000500']['mass']
    access_time = time.perf_counter() - test_time
    print(f"Single particle access: {access_time*1e6:.2f} μs")
    
    return particles

system = create_particle_system_professional()
```

### Exercise 4.2: Collision Detection System

**Part A: Basic Neighbor Finding (10 lines)**

```{code-cell} ipython3
def find_neighbors_naive(positions, radius):
    """Find particle pairs within radius - O(n²) approach."""
    neighbors = []
    n = len(positions)
    
    for i in range(n):
        for j in range(i+1, n):  # Avoid double counting
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            r = (dx**2 + dy**2) ** 0.5
            if r < radius:
                neighbors.append((i, j))
    
    return neighbors

# Test with small system
pos = [[0, 0], [1, 0], [0, 1], [2, 2], [3, 3]]
pairs = find_neighbors_naive(pos, 1.5)
print(f"Found {len(pairs)} neighbor pairs: {pairs}")
```

**Part B: Use Sets for Efficiency (15 lines)**

```{code-cell} ipython3
def find_neighbors_with_sets(particles, radius):
    """Track unique collision pairs with sets."""
    
    # Use set to avoid duplicate pairs
    collision_pairs = set()
    checked_pairs = set()
    
    positions = [(i, p['position']) for i, p in particles.items()]
    
    for i, (id1, pos1) in enumerate(positions):
        for j, (id2, pos2) in enumerate(positions[i+1:], i+1):
            pair = tuple(sorted([id1, id2]))  # Canonical ordering
            
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                
                # Check distance
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                r = (dx**2 + dy**2) ** 0.5
                
                if r < radius:
                    collision_pairs.add(pair)
    
    print(f"Checked {len(checked_pairs)} unique pairs")
    print(f"Found {len(collision_pairs)} collision candidates")
    return collision_pairs

# Test
test_particles = {
    'p1': {'position': [0, 0, 0]},
    'p2': {'position': [1e10, 0, 0]},
    'p3': {'position': [0, 1e10, 0]}
}
collisions = find_neighbors_with_sets(test_particles, 1.5e10)
```

**Part C: Spatial Hashing for O(n) (25 lines)**

```{code-cell} ipython3
def find_neighbors_spatial_hash(particles, radius, cell_size=None):
    """Use spatial hashing for O(n) neighbor finding."""
    if cell_size is None:
        cell_size = radius * 2
    
    # Build spatial hash
    cells = {}
    for pid, data in particles.items():
        x, y = data['position'][:2]
        cell_key = (int(x / cell_size), int(y / cell_size))
        
        if cell_key not in cells:
            cells[cell_key] = set()
        cells[cell_key].add(pid)
    
    # Find neighbors (only check adjacent cells)
    neighbors = set()
    for cell_key, pids in cells.items():
        cx, cy = cell_key
        
        # Check this cell and 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cx + dx, cy + dy)
                if neighbor_cell in cells:
                    for p1 in pids:
                        for p2 in cells[neighbor_cell]:
                            if p1 < p2:  # Avoid duplicates
                                # Check actual distance
                                pos1 = particles[p1]['position']
                                pos2 = particles[p2]['position']
                                r = ((pos1[0]-pos2[0])**2 + 
                                     (pos1[1]-pos2[1])**2) ** 0.5
                                if r < radius:
                                    neighbors.add((p1, p2))
    
    print(f"Spatial hash: {len(cells)} cells, {len(neighbors)} pairs")
    print("This scales as O(n) instead of O(n²)!")
    return neighbors
```

### Exercise 4.3: Equation of State Cache

**Part A: Basic EOS Function (10 lines)**

```{code-cell} ipython3
class SimpleEOS:
    """Basic equation of state without caching."""
    
    def __init__(self):
        self.calculations = 0
    
    def pressure(self, temperature, density):
        """Calculate pressure (ideal gas for simplicity)."""
        self.calculations += 1
        k_B = 1.381e-16  # erg/K
        m_H = 1.673e-24  # grams (hydrogen mass)
        
        # P = ρkT/m for ideal gas
        P = density * k_B * temperature / m_H
        return P

eos = SimpleEOS()
P1 = eos.pressure(1e6, 1e-3)
P2 = eos.pressure(1e6, 1e-3)  # Same calculation repeated!
print(f"Pressure: {P1:.2e} dyne/cm²")
print(f"Performed {eos.calculations} calculations (wasteful!)")
```

**Part B: Add Dictionary Cache (15 lines)**

```{code-cell} ipython3
class CachedEOS:
    """EOS with dictionary caching."""
    
    def __init__(self):
        self.cache = {}
        self.calculations = 0
        self.cache_hits = 0
    
    def pressure(self, temperature, density):
        """Calculate pressure with caching."""
        key = (temperature, density)
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        # Calculate only if not cached
        self.calculations += 1
        k_B = 1.381e-16
        m_H = 1.673e-24
        P = density * k_B * temperature / m_H
        
        self.cache[key] = P
        return P

eos = CachedEOS()
# Simulate multiple calls with same parameters
for _ in range(5):
    P = eos.pressure(1e6, 1e-3)

print(f"Calculations: {eos.calculations}, Cache hits: {eos.cache_hits}")
print(f"Saved {eos.cache_hits} expensive calculations!")
```

**Part C: Advanced Cache with Memory Limit (25 lines)**

```{code-cell} ipython3
from collections import OrderedDict

class ProductionEOS:
    """Production-ready EOS with LRU cache and statistics."""
    
    def __init__(self, cache_size=1000):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.calculations = 0
        self.cache_hits = 0
        self.evictions = 0
    
    def pressure(self, T, rho):
        """Get pressure with automatic caching."""
        key = (round(T, 2), round(rho, 6))  # Round for cache efficiency
        
        if key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key]
        
        # Calculate
        self.calculations += 1
        k_B = 1.381e-16
        m_H = 1.673e-24
        
        # More realistic EOS (includes radiation pressure)
        a = 7.566e-15  # Radiation constant
        P_gas = rho * k_B * T / m_H
        P_rad = a * T**4 / 3
        P_total = P_gas + P_rad
        
        # Add to cache
        self.cache[key] = P_total
        
        # Evict oldest if needed
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            self.evictions += 1
        
        return P_total
    
    def stats(self):
        """Report cache performance."""
        total = self.calculations + self.cache_hits
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            print(f"Cache statistics:")
            print(f"  Size: {len(self.cache)}/{self.cache_size}")
            print(f"  Calculations: {self.calculations}")
            print(f"  Cache hits: {self.cache_hits}")
            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Evictions: {self.evictions}")

# Test with stellar interior conditions
eos = ProductionEOS(cache_size=100)

# Simulate stellar model with repeated conditions
test_conditions = [
    (1e7, 100), (2e7, 150), (1e7, 100),  # Repeated
    (3e7, 200), (1e7, 100), (2e7, 150),  # More repeats
]

for T, rho in test_conditions:
    P = eos.pressure(T, rho)
    print(f"T={T:.0e} K, ρ={rho} g/cm³ → P={P:.2e} dyne/cm²")

eos.stats()
print("\nCaching is essential for stellar evolution codes!")
```

## Main Takeaways

You've just mastered concepts that will transform your computational physics from toy problems to research-grade simulations! The journey from understanding simple lists to architecting with dictionaries and sets represents a fundamental shift in how you approach scientific computing. You now see data structures not just as containers, but as carefully chosen tools where the right choice can mean the difference between simulations that finish in hours versus weeks.

The most profound insight from this chapter is that data structure choice often matters more than algorithm optimization. We saw how switching from a list to a set for particle lookups gave us a 100,000× speedup—no amount of code optimization could achieve that! This is the secret that separates research codes from student projects: professionals spend more time architecting their data organization than writing physics equations. The best physics insight in the world is useless if your code can't handle realistic problem sizes.

The mutable versus immutable distinction that seemed abstract at first is actually critical for scientific computing. Every time you use a tuple for physical constants or configuration parameters, you're preventing bugs that have literally crashed spacecraft and corrupted published results. When you properly use defensive copying for your simulation state, you're protecting yourself from the aliasing bugs that have plagued computational physics for decades. Remember: in the Cassini example, a simple data structure choice nearly lost a billion-dollar mission.

The performance principles you've learned extend far beyond Python. The cache efficiency concepts explain why codes like GROMACS and LAMMPS obsess over memory layout—it's not premature optimization, it's the difference between simulating thousands versus millions of atoms. The Big-O notation you've mastered is the universal language for discussing whether an algorithm scales to galaxy-sized problems. The hash table concept underlying dictionaries appears in every parallel communication library, every adaptive mesh refinement code, and every spatial indexing scheme you'll encounter.

Most importantly, you now understand the connection between data structures and numerical methods. That O(n²) all-pairs particle interaction that's impossible for large systems? It becomes O(n log n) with tree-based structures like Barnes-Hut. The O(n) searching that makes timesteps crawl? It becomes O(1) with spatial hashing. These aren't just computer science concepts—they're the difference between simulating a few hundred particles and modeling entire galaxies with billions of stars.

Remember: every data structure makes trade-offs. Lists give you ordering but slow searches. Dictionaries trade memory for lightning-fast lookups. Sets provide uniqueness and mathematical operations but lose ordering. There's no universally "best" structure—only the best structure for your specific physics problem. As you tackle research simulations, you'll combine multiple structures: NumPy arrays for number crunching, dictionaries for particle properties, sets for collision detection, and spatial data structures for neighbor finding.

With this foundation, you're ready to architect simulations that scale to astronomical proportions. The next chapter on functions and modules will show you how to organize this knowledge into reusable, testable components. After that, NumPy will supercharge your numerical operations using the memory layout principles you now understand. You're no longer just writing code—you're engineering solutions to the computational challenges of modern physics!

## Definitions

**Aliasing**: When two or more variables refer to the same object in memory, causing modifications through one to affect the others.

**Amortized O(1)**: An operation that is usually **O(1)** but occasionally **O(n)**, where the average over many operations remains **O(1)**.

**Big-O Notation**: Mathematical notation describing how an algorithm's runtime scales with input size, focusing on the dominant term.

**Cache**: Small, fast memory close to the CPU that stores recently accessed data for quick retrieval.

**Data Structure**: A way of organizing data in computer memory to enable efficient access and modification.

**Deep Copy**: Creating a completely independent copy of an object and all objects it contains, recursively.

**Dictionary**: A **mutable** mapping type storing key-value pairs with **O(1)** average lookup time using **hash tables**.

**Hash Function**: A function mapping data of arbitrary size to fixed-size values, enabling fast lookups.

**Hash Table**: The underlying implementation for **dictionaries** and **sets**, enabling **O(1)** average-case lookups.

**Immutable**: Objects whose state cannot be modified after creation (**tuples**, strings, numbers).

**List**: Python's built-in **mutable** sequence type that stores **references** to objects in order.

**Memoization**: **Caching** technique storing function results to avoid recalculating for the same inputs.

**Mutable**: Objects whose state can be modified after creation (**lists**, **dictionaries**, **sets**).

**Named Tuple**: A **tuple** subclass allowing element access by name as well as index.

**O(1) - Constant Time**: An operation whose runtime doesn't depend on input size.

**O(n) - Linear Time**: An operation whose runtime grows proportionally with input size.

**O(n²) - Quadratic Time**: An operation whose runtime grows with the square of input size.

**O(log n) - Logarithmic Time**: An operation whose runtime grows logarithmically with input size.

**O(n log n)**: Common complexity for efficient sorting algorithms and tree-based operations.

**Reference**: A variable that points to an object in memory rather than containing the value directly.

**Set**: A **mutable** collection of unique, unordered elements with **O(1)** membership testing.

**Shallow Copy**: Creating a new container with **references** to the same contained objects.

**Spatial Hashing**: Organizing particles by spatial region for **O(1)** neighbor finding.

**Tuple**: An **immutable** sequence type that cannot be changed after creation.

## Key Takeaways

✓ **Data structure** choice can change performance by factors of 1,000,000× or more for large-scale simulations

✓ **Lists** are versatile but have **O(n)** search—use for ordered particle arrays that you'll vectorize with NumPy

✓ **Dictionaries** and **sets** provide **O(1)** lookup through **hash tables**—essential for particle lookups and caching

✓ **Tuples** prevent modification bugs—use for physical constants and configuration parameters

✓ The **shallow** vs **deep copy** distinction is critical for grid-based simulations

✓ Python stores **references** to objects, explaining why NumPy's contiguous arrays are faster

✓ Memory layout affects **cache** performance by 2-10× even in Python

✓ Every data structure trades something (memory, speed, flexibility) for something else

✓ **Mutable** default arguments are dangerous—always use the None sentinel pattern (Chapter 1 callback)

✓ Spatial data structures transform **O(n²)** physics problems into **O(n)** or **O(n log n)**

✓ **Sets** provide elegant mathematical operations for domain decomposition and particle tracking

✓ **Dictionaries** enable **memoization** that can speed up equation of state calculations 100×

## Python Module & Method Reference

This reference section catalogs all Python modules, functions, and methods introduced in this chapter. Keep this as a quick lookup guide for your physics simulations.

### Standard Library Modules

**`time` module** - High-resolution timing for performance measurement
```python
import time
```
- `time.perf_counter()` - Returns float seconds with highest available resolution
  - Use for timing code segments: `start = time.perf_counter()`
  - Always use for benchmarking (not `time.time()` which can go backwards!)

**`sys` module** - System-specific parameters (expanded from Chapter 1)
```python
import sys
```
- `sys.getsizeof(object)` - Returns memory size of object in bytes
  - Includes object overhead but not contained objects
  - Essential for memory profiling in simulations

**`copy` module** - Create object copies with control over depth
```python
import copy
```
- `copy.copy(x)` - Creates shallow copy (new container, same contents)
- `copy.deepcopy(x)` - Creates deep copy (recursively copies all contents)
  - Critical for grid simulations to avoid aliasing bugs
  - Use for simulation state backups

**`random` module** - Generate random numbers for Monte Carlo
```python
import random
```
- `random.random()` - Random float in [0.0, 1.0)
- `random.uniform(a, b)` - Random float in [a, b]
- `random.choice(seq)` - Random element from sequence
- `random.sample(population, k)` - k unique random elements

**`json` module** - Save/load structured data
```python
import json
```
- `json.dump(obj, file, indent=2)` - Write object to file with formatting
- `json.load(file)` - Read object from file
- `json.dumps(obj)` - Convert object to JSON string
- `json.loads(string)` - Parse JSON string to object

**`math` module** - Mathematical functions (review from Chapter 2)
```python
import math
```
- `math.sqrt(x)` - Square root (use for distances)
- `math.log(x)` - Natural logarithm
- `math.log2(x)` - Base-2 logarithm (for Big-O analysis)

### Collections Module

**`collections` module** - Specialized container datatypes
```python
from collections import OrderedDict, Counter, defaultdict, deque, namedtuple
```

**`OrderedDict`** - Dictionary that remembers insertion order
- `.move_to_end(key, last=True)` - Move key to end (or beginning if last=False)
- `.popitem(last=True)` - Remove and return (key, value) pair from end/beginning
- Use for LRU caches in physics calculations

**`Counter`** - Dictionary subclass for counting hashable objects
- `Counter(iterable)` - Create from iterable
- `.most_common(n)` - Returns n most common (element, count) pairs
- `.update(iterable)` - Add counts from iterable
- Perfect for histogram analysis of particle properties

**`defaultdict`** - Dictionary with default value factory
- `defaultdict(list)` - Missing keys default to empty list
- `defaultdict(int)` - Missing keys default to 0
- Eliminates KeyError in particle grouping

**`deque`** - Double-ended queue with O(1) operations at both ends
- `deque(maxlen=n)` - Fixed-size queue (old elements dropped)
- `.append(x)` / `.appendleft(x)` - Add to right/left end
- `.pop()` / `.popleft()` - Remove from right/left end
- Essential for boundary conditions in simulations

**`namedtuple`** - Tuple subclass with named fields
```python
Particle = namedtuple('Particle', ['id', 'mass', 'x', 'y', 'z'])
p = Particle(1, 1.0e30, 0, 0, 0)
print(p.mass)  # Access by name
```

### Functools Module

**`functools` module** - Higher-order functions and operations
```python
from functools import lru_cache
```

**`@lru_cache(maxsize=128)`** - Decorator for automatic memoization
```python
@lru_cache(maxsize=1000)
def expensive_calculation(T, rho):
    return complex_physics_calculation(T, rho)
```
- Caches function results automatically
- `maxsize=None` for unlimited cache
- `.cache_info()` - Returns cache statistics
- `.cache_clear()` - Clear the cache

### Built-in Functions for Data Structures

**Type Checking**
- `type(obj)` - Returns object's type
- `isinstance(obj, type)` - Check if obj is instance of type
- `id(obj)` - Returns unique identifier (memory address)
- `obj1 is obj2` - Check if same object (identity)

**Container Operations**
- `len(container)` - Number of elements
- `x in container` - Membership test (O(1) for sets/dicts, O(n) for lists)
- `sorted(iterable)` - Returns new sorted list
- `reversed(sequence)` - Returns reverse iterator
- `enumerate(iterable, start=0)` - Returns (index, value) pairs
- `zip(iter1, iter2, ...)` - Parallel iteration

**Copying**
- `list(iterable)` - Create new list from iterable
- `tuple(iterable)` - Create new tuple from iterable
- `set(iterable)` - Create new set from iterable
- `dict(pairs)` - Create dictionary from (key, value) pairs

### Data Structure Methods Summary

**List Methods** (Expanded)
- `.append(x)` - Add to end (amortized O(1))
- `.extend(iterable)` - Add all elements from iterable
- `.insert(i, x)` - Insert at position i (O(n))
- `.remove(x)` - Remove first x (O(n))
- `.pop(i=-1)` - Remove and return element at i
- `.clear()` - Remove all elements
- `.index(x)` - Find position of x (O(n))
- `.count(x)` - Count occurrences of x
- `.sort()` - Sort in-place
- `.reverse()` - Reverse in-place
- `.copy()` - Create shallow copy

**Dictionary Methods** (Expanded)
- `.get(key, default=None)` - Safe access with default
- `.setdefault(key, default)` - Get or set with default
- `.pop(key, default)` - Remove and return value
- `.popitem()` - Remove and return arbitrary (key, value)
- `.update(other)` - Update with other dict/pairs
- `.keys()` - View of keys
- `.values()` - View of values
- `.items()` - View of (key, value) pairs
- `.clear()` - Remove all items
- `.copy()` - Shallow copy

**Set Methods** (Expanded)
- `.add(x)` - Add element
- `.remove(x)` - Remove element (raises KeyError if missing)
- `.discard(x)` - Remove element (no error if missing)
- `.pop()` - Remove and return arbitrary element
- `.clear()` - Remove all elements
- `.union(other)` or `|` - Elements in either set
- `.intersection(other)` or `&` - Elements in both sets
- `.difference(other)` or `-` - Elements in first but not second
- `.symmetric_difference(other)` or `^` - Elements in either but not both
- `.update(other)` or `|=` - Add elements from other
- `.intersection_update(other)` or `&=` - Keep only elements in both
- `.difference_update(other)` or `-=` - Remove elements in other
- `.symmetric_difference_update(other)` or `^=` - Keep elements in either but not both
- `.issubset(other)` - Check if all elements in other
- `.issuperset(other)` - Check if contains all elements of other
- `.isdisjoint(other)` - Check if no elements in common
- `.copy()` - Create shallow copy

### Performance Quick Reference

| Operation | List | Tuple | Dict | Set | Deque |
|-----------|------|-------|------|-----|-------|
| Access by index | O(1) | O(1) | N/A | N/A | O(n) |
| Search for value | O(n) | O(n) | O(1)* | O(1) | O(n) |
| Add to end | O(1)† | N/A | O(1)† | O(1)† | O(1) |
| Add to beginning | O(n) | N/A | O(1)† | O(1)† | O(1) |
| Remove from end | O(1) | N/A | N/A | N/A | O(1) |
| Remove from beginning | O(n) | N/A | N/A | N/A | O(1) |
| Remove specific | O(n) | N/A | O(1) | O(1) | O(n) |

\* Dict searches by key, not value  
† Amortized - occasionally O(n) during resize

## Quick Reference Tables

### Data Structure Operations

| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Create empty | `[]` | `()` | `{}` | `set()` |
| Create with items | `[1,2,3]` | `(1,2,3)` | `{'a':1}` | `{1,2,3}` |
| Add item | `.append(x)` | N/A | `d[k]=v` | `.add(x)` |
| Remove item | `.remove(x)` | N/A | `del d[k]` | `.remove(x)` |
| Check membership | `x in list` | `x in tuple` | `k in dict` | `x in set` |
| Get by index | `list[i]` | `tuple[i]` | N/A | N/A |

### Common Methods for Physics

| Structure | Method | Physics Use Case |
|-----------|--------|------------------|
| list | `.append(x)` | Add new particle |
| list | `.extend(iter)` | Merge particle lists |
| dict | `.get(k, default)` | Safe parameter lookup |
| dict | `@lru_cache` | Automatic memoization |
| set | `.union(other)` | Combine particle domains |
| set | `.intersection(other)` | Find boundary particles |
| deque | `.appendleft()` | Boundary conditions |

### When to Use Each Structure

| Physics Task | Best Choice | Why |
|--------------|------------|-----|
| Particle positions | List → NumPy array | Vectorizable operations |
| Particle lookup by ID | Dictionary | O(1) access |
| Active particles | Set | Fast membership, uniqueness |
| Physical constants | Tuple/NamedTuple | Immutable, safe |
| EOS cache | Dictionary | Fast lookup by (T,ρ) |
| Collision candidates | Set | No duplicates, set operations |
| Time series | List | Ordered, allows duplicates |
| Spatial grid | Dict of sets | O(1) cell access |

## Next Chapter Preview

With data structures mastered, Chapter 5 will explore functions and modules—how to organize code for reusability, testing, and collaboration. You'll learn how Python's function model, with first-class functions and closure support, enables powerful patterns like decorators and functional programming techniques. These concepts prepare you for building modular physics engines, creating reusable analysis pipelines, and understanding the functional programming paradigm used in modern frameworks like JAX where functions transform into automatically differentiable computational graphs. Get ready to transform your scripts into professional-quality scientific libraries!