---
title: "Part 4: Stability and Performance"
subtitle: "Module 3: ODE Methods & Conservation | ASTR 596"
---

**Navigation:**
[← Part 3: Symplectic Integration](./03-symplectic.md) | [Synthesis & Summary →](./05-synthesis-summary.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Analyze** stability regions for different methods
- **Diagnose** and handle stiff equations
- **Transform** scalar loops to vectorized operations
- **Achieve** 10-100× performance improvements
- **Choose** appropriate methods based on stability requirements

---

## Linear Stability Theory

To understand when methods fail, we analyze their behavior on the test equation:

$$\frac{dy}{dt} = \lambda y$$

where $\lambda \in \mathbb{C}$. The true solution is $y(t) = y_0 e^{\lambda t}$.

:::{margin}
**Stability function**: The amplification factor $R(z)$ relating consecutive numerical solution values: $y_{n+1} = R(h\lambda)y_n$.
:::

When we apply a numerical method, we get:

$$y_{n+1} = R(h\lambda) y_n$$

where $R(z)$ is the **stability function**.

:::{margin}
**Stability region**: The set of complex values $z = h\lambda$ for which $|R(z)| \leq 1$, ensuring bounded solutions.
:::

For bounded solutions, we need $|R(h\lambda)| \leq 1$. The **stability region** is where this holds.

## Stability Functions for Common Methods

**Euler:** $R(z) = 1 + z$ (stable in circle of radius 1 centered at -1)

**RK2:** $R(z) = 1 + z + \frac{z^2}{2}$

**RK4:** $R(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}$ (Taylor series of $e^z$!)

### Visualizing Stability Regions

```
Complex z-plane (z = hλ):

        Im(z)
          ↑
          |     RK4 stability region
    ······|······
   ·      |      ·   Euler
  ·   ----+----   ·  stability
   ·      |      ·   region
    ······|······
          |
    ------+-----→ Re(z)
         -2

Euler: Small circle
RK2: Larger region
RK4: Even larger
Leapfrog: Imaginary axis only
```

## Physical Interpretation

For a circular orbit with frequency $\omega$:
- Eigenvalues: $\lambda = \pm i\omega$
- Euler: $h < 2/\omega$ (barely stable, severe dissipation)
- RK4: $h < 2.8/\omega$ (larger timesteps allowed)
- Leapfrog: $h < 2/\omega$ (marginally stable, no dissipation)

## Stiff Equations

:::{margin}
**Stiff equation**: An ODE where stability requirements force much smaller timesteps than accuracy requirements.
:::

A **stiff equation** contains widely separated timescales. Consider:

$$\frac{dy}{dt} = -1000(y - \cos(t)) - \sin(t)$$

The solution has:
- Fast transient: $e^{-1000t}$ (decays in ~0.001 time units)
- Slow oscillation: $\cos(t)$ (period ~6.28 time units)

After the transient dies, we're just tracking $\cos(t)$. But explicit methods still need $h < 0.002$ for stability!

### Example: Chemical Reaction Networks

```python
def stiff_example():
    """
    Robertson's chemical kinetics - classic stiff system
    A → B (slow)
    B + B → C + B (fast)
    B + C → A + C (fast)
    """
    def robertson(y, t):
        A, B, C = y
        dA = -0.04*A + 1e4*B*C
        dB = 0.04*A - 1e4*B*C - 3e7*B**2
        dC = 3e7*B**2
        return np.array([dA, dB, dC])
    
    # Timescales differ by 10^9!
    # Explicit methods need h ~ 10^-9
    # But solution changes on timescale ~ 1
```

## Implicit Methods for Stiff Problems

:::{margin}
**Implicit method**: A numerical scheme where the unknown appears on both sides of the equation, requiring solution of an algebraic system.
:::

**Implicit methods** evaluate the derivative at the new point:

### Backward Euler

$$y_{n+1} = y_n + h f(y_{n+1}, t_{n+1})$$

Stability function: $R(z) = \frac{1}{1-z}$

This is stable for the entire left half-plane! The price: we must solve a (potentially nonlinear) equation at each step.

### Implementation

```python
def backward_euler(y, t, h, f, tol=1e-10):
    """
    Backward Euler for stiff equations
    Requires solving implicit equation
    """
    def residual(y_new):
        return y_new - y - h*f(y_new, t + h)
    
    # Newton iteration to solve implicit equation
    y_new = y  # Initial guess
    for _ in range(10):
        F = residual(y_new)
        if np.linalg.norm(F) < tol:
            return y_new
        
        # Compute Jacobian (finite difference)
        J = np.eye(len(y))
        eps = 1e-8
        for i in range(len(y)):
            y_pert = y_new.copy()
            y_pert[i] += eps
            J[:, i] = (residual(y_pert) - F) / eps
        
        # Newton update
        y_new = y_new - np.linalg.solve(J, F)
    
    return y_new
```

---

## Vectorization - From Loops to Arrays

### The Performance Revolution

Modern processors achieve peak performance through parallelism. Vectorization transforms our algorithms to exploit this hardware capability, routinely achieving 10-100× speedups.

### Memory Layout: The Hidden Performance Factor

:::{margin}
**Structure of Arrays (SoA)**: Organizing data so all x-coordinates are contiguous, all y-coordinates are contiguous, etc. Optimizes cache usage and vectorization.
:::

How we organize data dramatically affects performance. **Structure of Arrays (SoA)** stores each component contiguously:

```python
# Bad: Array of Structures (AoS)
particles = [
    {'x': x1, 'y': y1, 'z': z1, 'vx': vx1, ...},
    {'x': x2, 'y': y2, 'z': z2, 'vx': vx2, ...},
]

# Good: Structure of Arrays (SoA)
positions_x = np.array([x1, x2, x3, ..., xn])
positions_y = np.array([y1, y2, y3, ..., yn])
positions_z = np.array([z1, z2, z3, ..., zn])
```

This enables:
- **SIMD instructions**: Process 4-8 values per cycle
- **Cache efficiency**: Sequential access patterns
- **GPU compatibility**: Coalesced memory access

### N-body Forces: The Transformation

**Scalar approach (slow):**
```python
def compute_forces_scalar(positions, masses, G):
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

**Vectorized approach (fast):**
```python
def compute_forces_vectorized(positions, masses, G):
    """
    Fully vectorized N-body force calculation
    10-100x faster than scalar version
    """
    # All pairwise displacements
    dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    # All pairwise distances
    r = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(r, 1)  # Avoid self-interaction
    
    # Force magnitudes
    F_mag = G * masses[:, np.newaxis] * masses[np.newaxis, :] / r**3
    np.fill_diagonal(F_mag, 0)
    
    # Force vectors
    F_vectors = F_mag[:, :, np.newaxis] * dr
    
    # Total forces
    return F_vectors.sum(axis=1)
```

### Performance Comparison

| N particles | Scalar Time | Vectorized Time | Speedup |
|------------|-------------|-----------------|---------|
| 100 | 0.1s | 0.002s | 50× |
| 1000 | 10s | 0.02s | 500× |
| 10000 | 1000s | 2s | 500× |

### Rules for Efficient Vectorization

1. **Eliminate loops** - Replace with array operations
2. **Use broadcasting** - Let NumPy handle dimension expansion
3. **Preallocate arrays** - Never grow arrays in loops
4. **Access memory sequentially** - Use SoA layout
5. **Profile before optimizing** - Measure, don't guess

---

## Choosing Integration Methods - A Decision Tree

```
ODE System
    ├── Stiff?
    │   ├── Yes → Implicit Methods
    │   └── No → Continue
    │
    └── Long-time Integration?
        ├── Yes → Need Energy Conservation?
        │   ├── Yes → Symplectic Methods
        │   │   ├── O(h²) accuracy? → Leapfrog
        │   │   └── O(h⁴) accuracy? → Yoshida4
        │   └── No → RK4 or RK45
        │
        └── No → Accuracy Critical?
            ├── High → RK45 Adaptive
            └── Moderate → RK4 Fixed Step
```

## Common Integration Pitfalls and Solutions

| Problem | Symptom | Root Cause | Solution |
|---------|---------|------------|----------|
| Energy drift | Orbits slowly spiral | Non-symplectic method | Use leapfrog/Verlet |
| Sudden explosion | NaN/Inf after stable evolution | Timestep exceeds stability limit | Reduce h or use implicit |
| Phase error | Period wrong but amplitude OK | Accumulation of O(h²) error | Higher-order method |
| Performance | Simulation takes weeks | Scalar loops | Vectorize everything |

## Debugging Integration Problems

When your integration fails, systematic debugging saves weeks:

```python
def diagnose_integration(integrator, system, h):
    """
    Comprehensive integration diagnostics
    """
    # Test 1: Energy conservation
    E_initial = compute_energy(system)
    system_evolved = integrate(integrator, system, h, 1000)
    E_final = compute_energy(system_evolved)
    energy_drift = (E_final - E_initial) / E_initial
    
    # Test 2: Time reversibility
    forward = integrate(integrator, system, h, 100)
    backward = integrate(integrator, forward, -h, 100)
    reversibility_error = np.linalg.norm(backward - system) / np.linalg.norm(system)
    
    # Test 3: Convergence order
    errors = []
    for h_test in [h, h/2, h/4, h/8]:
        result = integrate_fixed_time(integrator, system, h_test)
        reference = integrate_fixed_time(integrator, system, h_test/100)
        errors.append(np.linalg.norm(result - reference))
    
    measured_order = np.log2(errors[0]/errors[1])
    
    return {
        'energy_drift': energy_drift,
        'reversibility': reversibility_error,
        'measured_order': measured_order
    }
```

## Typical Timesteps for Astrophysical Systems

| System | Characteristic Time | Method | Typical h |
|--------|-------------------|---------|-----------|
| Earth orbit | 365.25 days | Leapfrog | 0.5-1 day |
| Binary pulsar | 7.75 hours | Symplectic | 1-10 seconds |
| Star cluster | 10 Myr crossing | Leapfrog | 1000 years |
| Galaxy merger | 100 Myr | Leapfrog | 0.1-1 Myr |

:::{admonition} Check Your Understanding
:class: question
1. Why is the imaginary axis important for orbital problems?
2. What makes an equation "stiff" from a stability perspective?
3. Why don't we always use implicit methods if they're so stable?
4. How does vectorization achieve 100× speedups?
:::

---

## Bridge to Synthesis: Bringing It All Together

You now understand the complete landscape of ODE integration: from catastrophic failure to geometric preservation, from stability boundaries to performance optimization. Each method makes specific trade-offs, and choosing wisely determines whether your simulation reveals nature's secrets or numerical artifacts.

In the synthesis, we'll connect these methods to the broader framework of computational physics, showing how the principles you've learned here appear throughout scientific computing.

*Next: Synthesis & Summary*