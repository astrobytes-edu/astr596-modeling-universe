---
title: "Part 5: Performance Optimization"
subtitle: "Module 3: ODE Methods & Conservation | ASTR 596"
---

**Navigation:**
[← Part 4: Stability Analysis](./04-module3-part4-stability.md) | [Synthesis & Summary →](./06-module3-synthesis.md)

## Learning Outcomes

By the end of this section, you will be able to:

- [ ] **Transform** scalar loops to vectorized operations
- [ ] **Achieve** 10-100× performance improvements
- [ ] **Understand** memory layout and cache optimization
- [ ] **Profile** code to identify bottlenecks
- [ ] **Apply** vectorization to N-body and other problems

---

## The Performance Revolution

Modern processors achieve peak performance through parallelism. A single CPU core can process 4-8 floating-point numbers simultaneously using SIMD (Single Instruction, Multiple Data) instructions. GPUs can process thousands. But accessing this power requires structuring our code correctly.

The difference between naive and optimized code can be dramatic:
- **Naive loops**: Process one value at a time
- **Vectorized code**: Process entire arrays simultaneously
- **Performance gain**: Often 10-100× faster

## Memory Layout: The Hidden Performance Factor

:::{margin}
**Cache**: Fast memory close to the CPU. Modern processors have L1 (32KB), L2 (256KB), and L3 (8MB) caches. Accessing main memory is ~100× slower than L1 cache.
:::

How we organize data in memory dramatically affects performance. CPUs load data in **cache lines** (typically 64 bytes = 8 doubles). Accessing memory sequentially is orders of magnitude faster than random access.

### Array of Structures (AoS) vs Structure of Arrays (SoA)

:::{margin}
**Structure of Arrays (SoA)**: Organizing data so all x-coordinates are contiguous, all y-coordinates are contiguous, etc. Optimizes cache usage and vectorization.
:::

Consider storing particle data:

**Bad: Array of Structures (AoS)**

```python
# Each particle is a structure, array of particles
particles = [
    {'x': x1, 'y': y1, 'z': z1, 'vx': vx1, 'vy': vy1, 'vz': vz1},
    {'x': x2, 'y': y2, 'z': z2, 'vx': vx2, 'vy': vy2, 'vz': vz2},
    # ... accessing x-coordinates requires jumping through memory
]
```

**Good: Structure of Arrays (SoA)**

```python
# All x-coordinates contiguous, then all y, etc.
positions_x = np.array([x1, x2, x3, ..., xn])
positions_y = np.array([y1, y2, y3, ..., yn])
positions_z = np.array([z1, z2, z3, ..., zn])
velocities_x = np.array([vx1, vx2, vx3, ..., vxn])
# ... sequential memory access!
```

This enables:

- **SIMD instructions**: Process 4-8 values per CPU cycle
- **Cache efficiency**: Sequential access maximizes cache hits
- **GPU compatibility**: Coalesced memory access

<!-- Figure: Memory Layout Performance Impact
Create a figure with two panels:
Top: Visualize AoS vs SoA memory layout with cache lines highlighted
Bottom: Performance graph showing computation time vs number of particles for both layouts
Include cache miss indicators for AoS
Why: Memory layout can cause 10x performance differences but is invisible in the code
Caption: "Memory layout dramatically affects performance. (Top) AoS scatters related data across memory, causing cache misses. SoA keeps related data contiguous, maximizing cache efficiency. (Bottom) Performance comparison shows SoA can be 10× faster for large particle counts due to better cache utilization and vectorization."
-->

## Vectorization in NumPy

NumPy operations are implemented in optimized C and automatically use SIMD instructions. The key is expressing operations on entire arrays rather than elements.

### Example: Computing All Pairwise Distances

**Scalar approach (slow):**

```python
def distances_scalar(positions):
    n = len(positions)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            distances[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return distances
```

**Vectorized approach (fast):**

```python
def distances_vectorized(positions):
    # Broadcasting creates all pairwise differences
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    # Compute all distances at once
    distances = np.linalg.norm(diff, axis=2)
    return distances
```

The vectorized version is typically 50-100× faster!

## Broadcasting: NumPy's Secret Weapon

:::{margin}
**Broadcasting**: NumPy's ability to perform operations on arrays of different shapes by automatically expanding dimensions.
:::

**Broadcasting** allows operations between arrays of different shapes without explicit loops:

```python
# positions shape: (n, 3)
# Create all pairwise differences using broadcasting
# positions[np.newaxis, :, :] has shape (1, n, 3)
# positions[:, np.newaxis, :] has shape (n, 1, 3)
# Broadcasting expands to (n, n, 3) for subtraction
diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
```

## Practical Example: Gravitational Forces

Let's vectorize force calculation for your N-body project:

**Inefficient scalar version:**

```python
def forces_scalar(positions, masses, G=1.0):
    n = len(masses)
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dr = positions[j] - positions[i]
                r = np.linalg.norm(dr)
                F_mag = G * masses[i] * masses[j] / r**3
                forces[i] += F_mag * dr
    
    return forces
```

**Efficient vectorized version:**
```python
def forces_vectorized(positions, masses, G=1.0):
    # All pairwise displacements: shape (n, n, 3)
    dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    # All pairwise distances: shape (n, n)
    r = np.linalg.norm(dr, axis=2)
    
    # Avoid self-interaction
    np.fill_diagonal(r, np.inf)
    
    # Force magnitudes: shape (n, n)
    F_mag = G * masses[:, np.newaxis] * masses[np.newaxis, :] / r**3
    
    # Force vectors: shape (n, n, 3)
    F_vec = F_mag[:, :, np.newaxis] * dr
    
    # Total force on each particle
    return F_vec.sum(axis=1)
```

### Performance Comparison

| N particles | Scalar Time | Vectorized Time | Speedup |
|------------|-------------|-----------------|---------|
| 100 | 0.1s | 0.002s | 50× |
| 1000 | 10s | 0.02s | 500× |
| 10000 | 1000s | 2s | 500× |

The speedup increases with problem size!

## Profiling: Finding Bottlenecks

Before optimizing, profile to find where time is actually spent:

```python
import cProfile
import pstats

def profile_code():
    # Your simulation code here
    pass

# Profile the code
cProfile.run('profile_code()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time consumers
```

Typical bottlenecks in N-body codes:
1. Force calculation (90% of time)
2. Neighbor finding (for short-range forces)
3. Memory allocation in loops
4. Unnecessary array copies

## Advanced Vectorization Techniques

### Avoiding Temporary Arrays

```python
# Bad: Creates temporary arrays
result = np.sqrt(x**2 + y**2 + z**2)

# Good: In-place operations
result = x**2
result += y**2
result += z**2
np.sqrt(result, out=result)
```

### Using Numba for Custom Operations

When NumPy isn't enough, Numba compiles Python to machine code:

```python
from numba import jit

@jit(nopython=True)
def custom_force(positions, masses):
    n = len(masses)
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i+1, n):  # Exploit symmetry
            dr = positions[j] - positions[i]
            r2 = dr[0]**2 + dr[1]**2 + dr[2]**2
            F = masses[i] * masses[j] / (r2 * np.sqrt(r2)) * dr
            forces[i] += F
            forces[j] -= F  # Newton's third law
    
    return forces
```

## Rules for Efficient Vectorization

1. **Eliminate loops**: Replace with array operations
2. **Use broadcasting**: Let NumPy handle dimension expansion
3. **Preallocate arrays**: Never grow arrays in loops
4. **Access memory sequentially**: Use SoA layout
5. **Minimize temporaries**: Use in-place operations
6. **Profile before optimizing**: Measure, don't guess

<!-- Figure: Vectorization Performance Scaling
Create a log-log plot showing:
- X-axis: Number of particles (100 to 100,000)
- Y-axis: Computation time
- Three lines: Pure Python loops, NumPy vectorized, Numba compiled
- Include theoretical O(N²) scaling line
Why: Shows dramatic performance differences and scaling behavior
Caption: "Performance scaling for N-body force calculation. Pure Python loops become impractical beyond 1000 particles. NumPy vectorization provides 50-500× speedup. Numba can provide additional 2-5× improvement for complex custom operations. All methods show O(N²) scaling as expected."
-->

## Tips for Your N-body Project

:::{admonition} N-body Performance Tips
:class: tip
1. **Start with correctness**: Get a working scalar version first
2. **Vectorize force calculation**: This is 90% of runtime
3. **Use SoA layout**: Separate x, y, z arrays
4. **Avoid repeated calculations**: Compute r³ once, not r then r³
5. **Exploit Newton's third law**: F_ij = -F_ji saves half the work
6. **Consider cutoffs**: For large N, use tree codes or cutoff radius
7. **Monitor memory usage**: Large arrays can exceed cache/RAM
:::

## Common Performance Mistakes

:::{admonition} Common Mistakes in Vectorization
:class: warning
1. **Growing arrays in loops** - Causes repeated memory reallocation
2. **Wrong memory layout** - Row-major vs column-major matters
3. **Unnecessary copies** - Use views when possible
4. **Over-vectorizing** - Sometimes a simple loop is clearer and fast enough
5. **Ignoring cache effects** - Random access patterns kill performance
6. **Not profiling** - Optimizing the wrong part of code
:::

## When NOT to Vectorize

Vectorization isn't always the answer:

1. **Small arrays** (N < 100): Overhead exceeds benefit
2. **Complex logic**: If-statements break vectorization
3. **Sequential dependencies**: Can't parallelize recursive operations
4. **Memory limited**: Vectorization uses more memory
5. **Clarity matters more**: Readable code beats fast code for non-bottlenecks

## Choosing Integration Methods - Complete Decision Tree

Combining everything we've learned:

```
Start: ODE System
    │
    ├─ Stiff? ──Yes──→ Implicit Methods (Backward Euler, etc.)
    │    │
    │    No
    │    ↓
    ├─ Long Integration? ──Yes──→ Conservative? ──Yes──→ Symplectic
    │    │                              │                    │
    │    No                             No                   ├─ O(h²): Leapfrog
    │    ↓                              ↓                    └─ O(h⁴): Yoshida
    │                                  RK45
    └─ Many Particles? ──Yes──→ Vectorize Everything!
         │
         No → Simple RK4
```

:::{admonition} Check Your Understanding
:class: question
1. Why does memory layout affect performance so dramatically?
2. What is broadcasting and how does it eliminate loops?
3. When would scalar code outperform vectorized code?
4. Why is profiling essential before optimization?
5. How does cache size limit the benefits of vectorization?
6. What's the trade-off between code clarity and performance?
:::

---

## Bridge to Synthesis: Bringing It All Together

You now understand the complete landscape of ODE integration:
- How methods fail (Part 1: Euler's catastrophe)
- How to achieve accuracy (Part 2: Runge-Kutta)
- How to preserve physics (Part 3: Symplectic)
- When methods explode (Part 4: Stability)
- How to make them fast (Part 5: Performance)

In the synthesis, we'll connect these ideas to show how they form a unified framework for computational dynamics. The principles you've learned here—balancing accuracy, stability, conservation, and performance—apply throughout computational physics.

*Next: Synthesis & Summary*