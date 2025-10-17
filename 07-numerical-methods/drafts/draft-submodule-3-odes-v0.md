# SUBMODULE 3: ODE METHODS & CONSERVATION
*Week 4 - "Making Time Flow While Preserving Physics"*

## Learning Objectives
By the end of this submodule, you will:
- Implement and compare Euler, RK2, RK4, and symplectic integrators
- Understand why energy conservation requires special methods
- Analyze stability regions and choose appropriate timesteps
- Transform sequential thinking to vectorized array operations

---

## Part 5: ODE Integration - Building Up from Euler to RK4

### The Fundamental Problem

> 🎯 **What We Want vs What We Get**
> 
> **Physics says**: $\frac{dx}{dt} = f(x,t)$ with solution $x(t) = x_0 + \int_0^t f(x,s) ds$
> 
> **Computer says**: Can't integrate continuously! Must approximate with finite steps.
> 
> **Every ODE method is just a different way to approximate this integral!**

Think of it like this:
- **Rectangle rule** → Euler method
- **Midpoint rule** → RK2
- **Simpson's rule** → RK4

### The Failure Cascade - Learning Through Disaster

#### Stage 1: Euler Method (1st Order) - The Naive Approach
```python
def euler_step(x, v, dt, accel_func):
    """
    Simplest method: freeze physics during timestep
    Local error: O(dt²), Global error: O(dt)
    """
    a = accel_func(x)
    x_new = x + v * dt      # Constant velocity
    v_new = v + a * dt      # Constant acceleration
    return x_new, v_new

# Test on harmonic oscillator (should be periodic!)
def harmonic_accel(x):
    return -x  # F = -kx with k=m=1

# Watch energy EXPLODE
x, v = 1.0, 0.0  # Initial conditions
for _ in range(1000):
    x, v = euler_step(x, v, 0.01, harmonic_accel)
    energy = 0.5*(x**2 + v**2)
    # Energy grows linearly with time!
```

**Why Euler Fails**: Updates position and velocity independently, breaking the coupling between them.

#### Stage 2: Midpoint Method (2nd Order) - Better But Not Good Enough
```python
def rk2_midpoint(x, v, dt, accel_func):
    """
    Sample acceleration at midpoint for better estimate
    Local error: O(dt³), Global error: O(dt²)
    """
    a = accel_func(x)
    
    # Estimate midpoint
    x_mid = x + 0.5*dt*v
    v_mid = v + 0.5*dt*a
    
    # Use midpoint derivatives for full step
    a_mid = accel_func(x_mid)
    x_new = x + dt*v_mid
    v_new = v + dt*a_mid
    
    return x_new, v_new
```

**Improvement**: Samples physics at intermediate point - captures some curvature.

#### Stage 3: RK4 - The Workhorse (4th Order)
```python
def rk4_step(x, v, dt, accel_func):
    """
    Simpson's rule for ODEs - weighted average of slopes
    Local error: O(dt⁵), Global error: O(dt⁴)
    """
    # Sample derivative at 4 points
    k1v = accel_func(x)
    k1x = v
    
    k2v = accel_func(x + 0.5*dt*k1x)
    k2x = v + 0.5*dt*k1v
    
    k3v = accel_func(x + 0.5*dt*k2x)
    k3x = v + 0.5*dt*k2v
    
    k4v = accel_func(x + dt*k3x)
    k4x = v + dt*k3v
    
    # Simpson's weights: 1/6, 1/3, 1/3, 1/6
    x_new = x + dt*(k1x + 2*k2x + 2*k3x + k4x)/6
    v_new = v + dt*(k1v + 2*k2v + 2*k3v + k4v)/6
    
    return x_new, v_new
```

**The Magic**: Samples derivative at start, middle (twice), and end - optimal weights cancel errors through O(dt⁴).

### Empirical Order Verification
```python
def verify_convergence_order():
    """Measure how error scales with timestep"""
    timesteps = [0.1, 0.05, 0.025, 0.0125]
    methods = {
        'Euler': euler_step,
        'RK2': rk2_midpoint,
        'RK4': rk4_step
    }
    
    for name, method in methods.items():
        errors = []
        for dt in timesteps:
            # Integrate one orbit
            x, v = 1.0, 0.0
            for _ in range(int(2*np.pi/dt)):
                x, v = method(x, v, dt, harmonic_accel)
            
            # Error from starting point
            error = np.sqrt((x-1)**2 + v**2)
            errors.append(error)
        
        # Calculate order from error ratios
        order = np.log2(errors[0]/errors[1])
        print(f"{name}: Order = {order:.1f}")
```

**Expected Output**:
- Euler: Order ≈ 1.0
- RK2: Order ≈ 2.0
- RK4: Order ≈ 4.0

### 📊 Conceptual Checkpoint
Before moving on, verify you understand:
- [ ] Why higher order methods sample at multiple points
- [ ] How local error accumulates to global error
- [ ] Why RK4 uses those specific weights (Simpson's rule!)
- [ ] That ALL these methods have energy drift

### The Critical Insight

> ⚠️ **Higher Order ≠ Better for Everything**
> 
> RK4 has tiny errors per step, but for million-year simulations:
> - Energy still drifts (just slower)
> - Phase errors accumulate
> - Eventually, planets spiral into the sun!
> 
> **This motivates symplectic methods...**

---

## Part 6: Symplectic Methods - When Conservation Laws Drive Algorithms

### The Revelation

Standard methods integrate position and velocity:
$$\begin{cases}
\dot{x} = v \\
\dot{v} = a(x)
\end{cases}$$

Symplectic methods integrate position and momentum:
$$\begin{cases}
\dot{x} = \frac{\partial H}{\partial p} \\
\dot{p} = -\frac{\partial H}{\partial x}
\end{cases}$$

This "trivial" change preserves phase space volume!

### Leapfrog/Verlet - The Elegant Solution
```python
def leapfrog_step(x, v, dt, accel_func):
    """
    The staggered update that preserves energy
    Still 2nd order, but symplectic!
    """
    # Half-step velocity (leap)
    a = accel_func(x)
    v_half = v + 0.5*dt*a
    
    # Full-step position (frog)
    x_new = x + dt*v_half
    
    # Half-step velocity (leap)
    a_new = accel_func(x_new)
    v_new = v_half + 0.5*dt*a_new
    
    return x_new, v_new
```

**Why It Works**: 
- Time-reversible (can integrate backward!)
- Preserves phase space volume (Liouville's theorem)
- Energy oscillates but stays bounded

### Visual: Phase Space Preservation
```python
def visualize_phase_space():
    """Compare how methods deform phase space"""
    # Initial circle of test particles
    theta = np.linspace(0, 2*np.pi, 100)
    x0 = 0.1 * np.cos(theta)
    v0 = 0.1 * np.sin(theta)
    
    dt = 0.1
    steps = 100
    
    # Evolve with each method
    for method_name, method in [
        ('Euler', euler_step),
        ('RK4', rk4_step),
        ('Leapfrog', leapfrog_step)
    ]:
        x, v = x0.copy(), v0.copy()
        
        for _ in range(steps):
            for i in range(len(x)):
                x[i], v[i] = method(x[i], v[i], dt, harmonic_accel)
        
        # Calculate area (should be preserved!)
        area = np.pi * np.std(x) * np.std(v) * 4
        print(f"{method_name}: Final area = {area:.3f}")
        # Only Leapfrog preserves area!
```

### The Modified Hamiltonian - Why Symplectic Works

Symplectic methods don't solve the exact problem perfectly. They solve a slightly different problem exactly!

```python
# Original Hamiltonian: H = p²/2m + V(q)
# Leapfrog actually conserves: H_modified = H + O(dt²)

def analyze_modified_hamiltonian():
    """Show that Leapfrog conserves *something*"""
    x, v = 1.0, 0.0
    energies = []
    
    for _ in range(10000):
        x, v = leapfrog_step(x, v, 0.01, harmonic_accel)
        E = 0.5*(x**2 + v**2)
        energies.append(E)
    
    print(f"Energy oscillation amplitude: {np.std(energies):.6f}")
    print(f"Energy drift: {energies[-1] - energies[0]:.6f}")
    # Oscillates but no secular drift!
```

### 🤝 Peer Instruction Question
*Think, then discuss:*

"Your 1000-year solar system simulation shows planets slowly spiraling outward. Most likely cause?"

A) Timestep too large  
B) Using RK4 instead of symplectic method  
C) Round-off error accumulation  
D) Wrong gravitational constant  

*Answer: B - Even tiny energy errors in RK4 accumulate over millions of orbits*

### When to Use Symplectic vs High-Order

| Scenario | Best Choice | Reason |
|----------|------------|--------|
| Long-term orbits | Symplectic | Energy conservation |
| Chaotic systems | Symplectic | Preserves structure |
| Short smooth trajectory | RK4 | High accuracy |
| Dissipative systems | RK4 | No energy to conserve |
| Molecular dynamics | Symplectic | Many particles, long times |

---

## Part 7: Stability Analysis - When Methods Explode

### Linear Stability Theory

For the test equation $\dot{y} = \lambda y$:
- True solution: $y(t) = y_0 e^{\lambda t}$
- Euler approximation: $y_{n+1} = (1 + \lambda \Delta t) y_n$

**Stable when**: $|1 + \lambda \Delta t| < 1$

### Stability Regions Visualization
```python
def plot_stability_regions():
    """Where in complex plane is each method stable?"""
    # Create grid of λΔt values
    x = np.linspace(-3, 1, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    z = X + 1j*Y
    
    # Amplification factors
    R_euler = np.abs(1 + z)
    R_rk2 = np.abs(1 + z + 0.5*z**2)
    R_rk4 = np.abs(1 + z + z**2/2 + z**3/6 + z**4/24)
    
    # Plot (stable where |R| < 1)
    for R, name in [(R_euler, 'Euler'), 
                     (R_rk2, 'RK2'), 
                     (R_rk4, 'RK4')]:
        stable = R < 1
        print(f"{name} stable region area: {np.sum(stable)}")
    
    # RK4 has largest stable region!
```

### Physical Timestep Constraints

```python
def compute_timestep_limits(system):
    """Dimensional analysis for safe timesteps"""
    
    if system == "earth_orbit":
        # Orbital frequency
        omega = np.sqrt(G*M_sun/AU**3)
        dt_max = 0.01 * (2*np.pi/omega)  # 1% of period
        print(f"Max dt: {dt_max/86400:.1f} days")
        
    elif system == "star_cluster":
        # Multiple timescales!
        t_cross = pc / (10*km/s)  # Crossing time
        t_relax = N * t_cross / np.log(N)  # Relaxation
        t_binary = day  # Close binary period
        
        dt_max = min(0.01*t_binary, 0.001*t_cross)
        print(f"Competing scales: {t_cross/t_binary:.1e}")
        
    return dt_max
```

### ⚠️ Common Misconception Alert
> **"Smaller timestep always better"**
> 
> **FALSE! Consider:**
> - Below ~10⁻⁸ seconds, round-off dominates
> - More steps = more accumulated error
> - Optimal timestep balances accuracy and stability

---

## Part 13: Vectorization & Modern Computing

### The Paradigm Shift: Loops → Arrays

Traditional (slow) approach:
```python
# DON'T DO THIS - Sequential nightmare
forces = []
for i in range(N):
    f_i = np.zeros(3)
    for j in range(N):
        if i != j:
            dr = pos[j] - pos[i]
            r = np.linalg.norm(dr)
            f_i += G*mass[j]*dr/r**3
    forces.append(f_i)
```

Modern (fast) vectorized approach:
```python
# DO THIS - Parallel beauty
def compute_forces_vectorized(pos, mass):
    """All N² forces at once!"""
    # Pairwise vectors: shape (N, N, 3)
    dr = pos[:, None, :] - pos[None, :, :]
    
    # Distances: shape (N, N)
    r = np.linalg.norm(dr, axis=2)
    r[r==0] = 1  # Avoid self-force
    
    # Forces: F_ij = G*m_i*m_j*dr_ij/r³
    F = G * mass[:, None, None] * mass[None, :, None] * dr / r[:, :, None]**3
    
    return np.sum(F, axis=1)  # Sum over j
```

**Performance difference**: 10-100× faster!

### Preview: JAX for Your Final Project
```python
# Coming in your neural network project:
import jax.numpy as jnp
from jax import jit, vmap, grad

@jit  # Compile to machine code!
def leapfrog_jax(state, dt):
    """Same algorithm, but JAX-powered"""
    x, v = state
    
    # Automatic differentiation for forces!
    a = -grad(potential)(x)
    
    v_half = v + 0.5*dt*a
    x_new = x + dt*v_half
    a_new = -grad(potential)(x_new)
    v_new = v_half + 0.5*dt*a_new
    
    return (x_new, v_new)

# Process 1000 orbits in parallel!
parallel_integrate = vmap(leapfrog_jax)
```

### Why Vectorization Matters

| Operation | Loop Time | Vectorized | Hardware |
|-----------|-----------|------------|----------|
| 1000-body forces | 1 s | 0.01 s | CPU SIMD |
| Matrix multiply | O(n³) | O(n³) but parallel | GPU |
| Monte Carlo batch | Sequential | Parallel | Multi-core |
| Neural network | Layer by layer | Matrix ops | TPU |

### 🤔 Metacognitive Reflection
*How does array thinking change your approach?*

1. **Problem formulation**: Think "all particles at once"
2. **Memory patterns**: Coalesced access patterns
3. **Debugging**: Harder but worth it
4. **Scaling**: Same code works for 10 or 10,000 particles

:::{admonition} 🚀 Computational Thinking: From Serial to Parallel Universes
:class: info, dropdown

**The Future is Parallel - Here's Why**

Modern astrophysics simulations don't run on single CPUs - they run on supercomputers with millions of cores. Understanding parallel thinking is essential.

**The Hierarchy of Parallelism:**

```python
# Level 1: Vectorization (this module)
# Single CPU, multiple data (SIMD)
forces = G * masses[:, None] * masses[None, :] / r**2

# Level 2: Multi-threading (shared memory)
# Multiple CPU cores, same memory
from multiprocessing import Pool
with Pool(8) as p:
    forces = p.map(compute_force, particles)

# Level 3: MPI (distributed memory)
# Multiple computers, message passing
from mpi4py import MPI
comm = MPI.COMM_WORLD
local_forces = compute_local_forces(local_particles)
total_forces = comm.allreduce(local_forces)

# Level 4: GPU (massive parallelism)
# Thousands of simple cores
@cuda.jit
def force_kernel(positions, forces):
    i = cuda.grid(1)
    if i < positions.size:
        forces[i] = compute_force_gpu(positions[i])
```

**Amdahl's Law - The Harsh Reality:**
```python
def parallel_speedup(parallel_fraction, n_cores):
    """Maximum possible speedup"""
    serial_fraction = 1 - parallel_fraction
    speedup = 1 / (serial_fraction + parallel_fraction/n_cores)
    
    # Example: 95% parallel code
    print(f"95% parallel on 100 cores: {speedup:.1f}× speedup")
    # Result: Only 17× faster, not 100×!
    
    # The serial 5% becomes the bottleneck
    return speedup
```

**Real-World Scales:**
- Millennium Simulation: 10¹⁰ particles, 512 CPUs, 28 days
- Illustris: 10¹⁰ particles, 8,192 cores, 19 million CPU-hours
- Your Project 2: 10³ particles, 1 CPU, minutes

**Parallel Thinking Principles:**

1. **Minimize Communication**
```python
# BAD: Communicate every timestep
for step in range(1000):
    forces = gather_all_forces()  # Expensive!
    
# GOOD: Communicate occasionally
for step in range(1000):
    if step % 100 == 0:
        forces = gather_all_forces()
```

2. **Load Balance**
```python
# BAD: Some processors idle
if rank == 0:
    do_lots_of_work()
else:
    do_tiny_task()
    
# GOOD: Everyone busy
work = distribute_evenly(total_work, n_processors)
```

3. **Data Locality**
```python
# BAD: Random memory access
for i in random_order:
    process(data[i])
    
# GOOD: Sequential access
for i in range(n):
    process(data[i])  # Cache-friendly
```

**Why This Matters for Your Future:**
- Every laptop has 4-8 cores (use them!)
- GPUs have thousands of cores (JAX leverages these)
- Cloud computing makes supercomputers accessible
- Big discoveries need big simulations

**The Mental Shift:**
- Serial thinking: "Do this, then that, then the other"
- Parallel thinking: "Do everything that can be done simultaneously"
- It's not about coding - it's about problem decomposition

**Your N-body is Already Parallel-Ready:**
```python
# Force calculation is "embarrassingly parallel"
# Each particle pair is independent!
# 1000 particles = 500,000 independent calculations
# Perfect for GPUs!
```

**Bottom Line**: Vectorization (this module) is your gateway drug to parallel computing. Master array thinking now, scale to supercomputers later.
:::

---

## Integration Challenge: Bringing It All Together

```python
def compare_all_methods():
    """
    Complete comparison for binary star orbit
    Track: accuracy, energy conservation, speed
    """
    # Binary star parameters
    m1, m2 = 1.0, 0.5  # Solar masses
    a = 1.0  # AU separation
    
    # Initial circular orbit
    x1 = np.array([a*m2/(m1+m2), 0, 0])
    x2 = np.array([-a*m1/(m1+m2), 0, 0])
    v_circ = np.sqrt(G*(m1+m2)/a)
    v1 = np.array([0, v_circ*m2/(m1+m2), 0])
    v2 = np.array([0, -v_circ*m1/(m1+m2), 0])
    
    methods = {
        'Euler': euler_step,
        'RK2': rk2_midpoint,
        'RK4': rk4_step,
        'Leapfrog': leapfrog_step
    }
    
    # Integrate for 10 orbits
    # Compare: energy drift, phase error, CPU time
    # Which method would you choose for:
    # a) 1-day prediction?
    # b) 1000-year evolution?
    # c) Million-particle cluster?
```

---

## Connections to Course Projects

### Project 2 (N-body) - Immediate Application
- Implement Leapfrog for stable orbits
- Compare energy conservation across methods
- Vectorize force calculations

### Future Projects
- **Project 4**: Leapfrog becomes HMC sampler
- **Project 5**: Vectorization essential for GP operations
- **Final**: JAX autodiff for neural network training

---

## Practice Problems

1. **Order Verification**: Empirically measure order of each method
2. **Energy Monitor**: Track energy for 1000 orbits, plot drift
3. **Stability Boundary**: Find critical timestep for each method
4. **Vectorization Challenge**: Convert 3-body problem to arrays

---

## Summary

You've discovered the fundamental tension in numerical integration:
- **Accuracy** (RK4) vs **Structure Preservation** (Symplectic)
- **Stability regions** determine maximum timesteps
- **Vectorization** enables modern-scale simulations
- **Method choice** depends on problem requirements

Key insight: Sometimes preserving physics (energy, momentum) matters more than minimizing local error!

Next submodule: Advanced techniques for when these methods aren't enough...