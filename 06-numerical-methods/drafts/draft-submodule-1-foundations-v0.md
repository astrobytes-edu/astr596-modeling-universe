# SUBMODULE 1: FOUNDATIONS OF DISCRETE COMPUTING
*Week 2 - "From Continuous Physics to Discrete Reality"*

## Learning Objectives
By the end of this submodule, you will:
- Understand why computers fundamentally cannot perform calculus
- Identify and predict different types of numerical errors
- Apply Taylor series to construct numerical approximations
- Determine optimal step sizes balancing truncation and round-off errors

---

## Part 0: The Fundamental Paradox - Calculus on Computers

### The Core Problem

> 🎯 **The Limit That Can Never Be**
> 
> **Mathematics says**: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$
> 
> **Computer says**: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ where $h$ can NEVER be zero!
> 
> **The paradox**: 
> - Make $h$ too big → large truncation error (poor approximation)
> - Make $h$ too small → catastrophic cancellation (floating-point disaster)
> - Sweet spot: $h \approx \sqrt{\epsilon_{machine}} \approx 10^{-8}$ for double precision

### The Parallel Universes: Continuous vs Discrete

| Continuous Calculus | Discrete Computer | Physical Example | What Goes Wrong |
|---------------------|-------------------|------------------|-----------------|
| $\lim_{h→0}$ | $h = \Delta t$ (fixed) | Timestep in orbit integration | Can't make arbitrarily small |
| $\int f(x)dx$ | $\sum f(x_i)\Delta x$ | Total luminosity from spectrum | Finite sampling misses features |
| $\frac{df}{dx}$ | $\frac{f(x+h)-f(x)}{h}$ | Force from potential gradient | Round-off error at small h |
| Smooth curves | Polygonal paths | Spacecraft trajectory | Corners where there shouldn't be |
| Exact conservation | Approximate conservation | Energy in N-body simulation | Systematic drift |

### Worked Example: Finding the Optimal h
```python
# Finding optimal h for derivative of sin(x) at x=1
import numpy as np
x = 1.0
true_deriv = np.cos(1.0)  # Analytical: 0.5403...

h_values = np.logspace(-16, 0, 50)  # From 10^-16 to 1
errors = []

for h in h_values:
    numerical_deriv = (np.sin(x + h) - np.sin(x)) / h
    error = abs(numerical_deriv - true_deriv)
    errors.append(error)
    
# Minimum error occurs at h ≈ 10^-8 
# This is sqrt(machine_epsilon)!
```

### 🤝 Peer Instruction Question
*Think for 30 seconds, then discuss with your neighbor:*

"You're computing a spacecraft's acceleration numerically. The derivative calculation shows huge errors. Which is the most likely cause?"
- A) Wrong physics equation (F = ma)
- B) h too small (round-off error dominates)
- C) h too large (truncation error dominates)
- D) Need to know the value of h to answer

*After discussion: Most spacecraft dynamics vary on minute/hour timescales. If using microsecond timesteps, answer is B. If using day timesteps, answer is C.*

---

## Part 1: Numbers Aren't Real - Computer Arithmetic & Cosmic Consequences

### Why This Matters for Astronomy

Astronomy spans the largest range of scales in science:
- Planck length: $10^{-35}$ m
- Atomic scale: $10^{-10}$ m  
- Planetary: $10^{7}$ m
- Stellar: $10^{9}$ m
- Galactic: $10^{21}$ m
- Observable universe: $10^{26}$ m

**Total range**: 61 orders of magnitude! But double precision only gives us ~16 decimal digits.

### Finding Machine Epsilon
```python
# Machine epsilon: smallest distinguishable difference
def find_machine_epsilon():
    eps = 1.0
    while 1.0 + eps != 1.0:
        eps_last = eps
        eps /= 2.0
    return eps_last  # Returns ~2.2e-16 for float64

# Below this scale, computer arithmetic breaks down
# 1 + epsilon = 1 in the computer's universe!
```

### The Three Types of Error

#### 1. **Round-off Error** (from finite precision)
```python
# Catastrophic cancellation example
def parallax_bad(d1, d2):
    """Parallax angle from two distances"""
    return 1.0/d1 - 1.0/d2  # Disaster when d1 ≈ d2!

def parallax_good(d1, d2):
    """Reformulated to preserve precision"""
    return (d2 - d1)/(d1 * d2)  # Much better!

# Example: Earth's orbit parallax
d1, d2 = 1.495978e11, 1.495979e11  # 1km difference in 1 AU
print(f"Bad: {parallax_bad(d1, d2):.3e}")  # Large error
print(f"Good: {parallax_good(d1, d2):.3e}")  # Accurate
```

#### 2. **Truncation Error** (from approximations)
- Comes from cutting off Taylor series
- Example: Using Euler method instead of RK4
- Controllable by algorithm choice and step size

#### 3. **Propagation Error** (accumulation over iterations)
- Errors compound through calculations
- Critical for long-term simulations
- Why we need symplectic methods for million-year orbits

### ⚠️ Common Misconception Alert
> **"Using quadruple precision (128-bit) solves all problems"**
> 
> **FALSE!** 
> - If your algorithm is unstable, errors grow exponentially regardless
> - Example: Chaotic systems where errors double every timestep
> - Better algorithm > more precision

### Real Astrophysics Disaster: Ariane 5
On June 4, 1996, Ariane 5 rocket exploded 39 seconds after launch.
- **Cause**: 64-bit float converted to 16-bit integer
- **Value**: Horizontal velocity > 32,767 (integer overflow)
- **Result**: $500 million loss
- **Lesson**: Understanding numerical limits prevents disasters

### 📊 Conceptual Checkpoint
Before proceeding, ensure you can:
- [ ] Explain why 0.1 + 0.2 ≠ 0.3 in binary computers
- [ ] Identify when catastrophic cancellation will occur
- [ ] Predict which operations lose the most precision
- [ ] Calculate machine epsilon for your system

---

## Part 2: Taylor Series - The Universe's Local Approximation

### From Calculus to Code

The Taylor series is the bridge between continuous physics and discrete computation:

$$f(x + h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + ... + \frac{h^{n+1}}{(n+1)!}f^{(n+1)}(\xi)$$

where $\xi \in [x, x+h]$ (the remainder/truncation error term).

### Building Numerical Methods from Taylor Series
```python
def position_update(x, v, a, jerk, dt, order):
    """Update position using Taylor series to given order"""
    x_new = x + v*dt                      # 1st order (Euler)
    if order >= 2: 
        x_new += 0.5*a*dt**2               # 2nd order
    if order >= 3: 
        x_new += (1.0/6.0)*jerk*dt**3      # 3rd order
    # Each term is smaller by factor of dt
    return x_new
```

### Understanding Error Types

#### Visual Representation
```
True value:        ●═══════════════════════════════
                   ↑
Computed value:    ○═══════════════════════════════
                   ↑
                   └─ Absolute Error = |● - ○|
                   
Relative Error = |● - ○| / |●|  (dimensionless, often %)

Taylor series truncated at order n:
f(x+h) ≈ f(x) + hf'(x) + ... + h^n f^(n)(x)/n!
         └────────────────────────┘
              We compute this
                                    └─ Truncation error (what we threw away)
```

### Order of a Method - Physical Meaning

> 📊 **Method Order = Leading Power of Truncation Error**

A method has **order p** if local truncation error ~ $O(h^{p+1})$:

| Order | Local Error | Halve h → Error ÷ | Example | Orbits to 1% Error |
|-------|-------------|-------------------|---------|-------------------|
| 1 | $O(h^2)$ | 4 | Euler | ~10 |
| 2 | $O(h^3)$ | 8 | Midpoint | ~100 |
| 4 | $O(h^5)$ | 32 | RK4 | ~10,000 |

:::{admonition} 🔬 Predict-Observe-Explain: Error Scaling
:class: tip, dropdown

**Activity: Empirically Discover Error Scaling**

Before implementing any method, predict its behavior:

**1. PREDICT** (2 minutes)
```python
# If Euler has error=0.01 at dt=0.1
# What error at dt=0.05? dt=0.025?
# Write your predictions:
prediction_dt_0_05 = ___  # Your guess
prediction_dt_0_025 = ___  # Your guess
```

**2. OBSERVE** (5 minutes)
```python
def test_euler_scaling():
    """Actually measure the error scaling"""
    true_solution = 1.0  # cos(2π) after one period
    
    for dt in [0.1, 0.05, 0.025, 0.0125]:
        x, v = 1.0, 0.0
        steps = int(2*np.pi/dt)
        for _ in range(steps):
            a = -x  # Harmonic oscillator
            x, v = x + v*dt, v + a*dt
        
        error = abs(x - true_solution)
        print(f"dt={dt}: error={error:.4f}")
```

**3. EXPLAIN** (3 minutes)
- Did errors scale as predicted?
- If 1st order: error should halve when dt halves
- If 2nd order: error should quarter when dt halves
- What does this mean for computational cost?

**Why This Matters**: In real research, you must predict method behavior BEFORE expensive simulations. This builds intuition for computational cost vs accuracy trade-offs.
:::

### Guided Practice: Measuring Order Empirically
```python
# Template: Determine method order experimentally
def measure_order(method, exact_solution, h_values):
    """Empirically determine method order"""
    errors = []
    for h in h_values:
        approx = method(h)
        errors.append(abs(exact_solution - approx))
    
    # On log-log plot, slope = -(order + 1)
    for i in range(len(errors)-1):
        ratio = errors[i] / errors[i+1]
        h_ratio = h_values[i] / h_values[i+1]
        order = np.log(ratio) / np.log(h_ratio) - 1
        print(f"Estimated order: {order:.1f}")
```

### Independent Practice: Planet Position Predictor
```python
# TODO: Predict Jupiter's position after 1 day
# Given: current position, velocity, and force
# Use Taylor series to different orders
# Compare accuracy vs computational cost
```

### The Convergence Radius Problem

Not all Taylor series converge everywhere! Example from general relativity:

```python
# Gravitational potential Taylor series
# Converges only outside Schwarzschild radius!
def potential_series(r, r_s, terms=5):
    """r_s = 2GM/c² (Schwarzschild radius)"""
    if r <= r_s:
        return float('inf')  # Series diverges!
    
    phi = 0
    for n in range(terms):
        phi += (-1)**n * (r_s/r)**(n+1) / (n+1)
    return phi * c**2
```

**Physical meaning**: Can't Taylor expand through an event horizon - the physics becomes fundamentally different!

### 🐛 Debugging Challenge
```python
# BUG HUNT: This Taylor series has 2 bugs
def taylor_sin(x, terms=5):
    result = 0
    for n in range(terms):
        result += (-1)**n * x**(2*n) / factorial(2*n+1)  # Bug 1
    return result  # Bug 2: factorial not defined

# Fix 1: x**(2*n+1) not x**(2*n)
# Fix 2: Import factorial or compute it
```

### 🤔 Metacognitive Reflection
*Take 2 minutes to reflect on these questions:*

1. **Why does higher order not always mean better?**
   - Higher order = more function evaluations = more round-off error
   - For rough functions, high derivatives might not exist

2. **When would you choose 2nd order over 4th order?**
   - When function evaluations are expensive
   - When you need many quick approximate solutions
   - When the function isn't smooth enough for 4th order

3. **What happens at machine precision limits?**
   - Taylor series terms become smaller than machine epsilon
   - Adding more terms adds only round-off error
   - Optimal truncation exists!

---

## Synthesis: Optimal Step Size Selection

Bringing together all three parts, we can now understand the fundamental trade-off:

```python
def optimal_h_demo():
    """Find optimal h balancing all error sources"""
    # Total error = Truncation + Round-off
    # E(h) ≈ Ch^p + ε/h  (p = method order)
    # Minimize: dE/dh = 0
    # Result: h_opt = (ε/C)^(1/(p+1))
    
    machine_eps = 2.2e-16
    
    # For different method orders:
    h_opt_euler = machine_eps**(1/2)    # ~10^-8
    h_opt_rk4 = machine_eps**(1/5)      # ~10^-3
    
    print(f"Optimal h for Euler: {h_opt_euler:.2e}")
    print(f"Optimal h for RK4: {h_opt_rk4:.2e}")
    # RK4 can use larger steps!
```

---

## Connections to Course Projects

### Project 2 (N-body Dynamics)
- You'll experience round-off accumulation over millions of timesteps
- Taylor series leads directly to integration methods
- Optimal timestep selection crucial for stable orbits

### Project 3 (Monte Carlo Radiative Transfer)
- Random sampling reduces round-off correlations
- Statistical error decreases as 1/√N (different from deterministic!)

### Project 4 (MCMC)
- Proposal distributions need careful numerical design
- Acceptance ratios can suffer from catastrophic cancellation

### Project 5 (Gaussian Processes)
- Covariance matrices near-singular (numerical challenges)
- Cholesky decomposition stability issues

### Final Project (Neural Networks)
- Gradient vanishing/explosion from repeated multiplication
- Careful initialization prevents numerical instability

---

## Quick Reference Card

### Error Type Hierarchy
1. **Round-off**: ~$10^{-16}$ (unavoidable)
2. **Truncation**: ~$h^{p+1}$ (controllable by method)
3. **Propagation**: Accumulation (controllable by algorithm)

### When Choosing Step Size h
- Too large: Truncation error dominates (inaccurate)
- Too small: Round-off error dominates (cancellation)
- Just right: $h \approx \epsilon^{1/(p+1)}$ for order p method

### Red Flags in Your Code
- Subtracting nearly equal numbers
- Dividing by small differences
- Adding tiny increments to large numbers
- Using == for floating-point comparison

---

## Practice Problems

1. **Parallax Calculation**: Implement both versions and measure precision loss
2. **Machine Epsilon**: Find it for different data types (float32, float64)
3. **Taylor Series**: Approximate cos(0.1) to different orders, plot convergence
4. **Optimal h**: Find empirically for d/dx[exp(x)] at x=1

---

## Summary

You've learned that numerical computing is fundamentally about managing approximations:
- Computers can't do calculus - only finite differences
- Every number has limited precision - plan for it
- Taylor series connects continuous to discrete - but must truncate
- Optimal choices balance competing error sources

Next submodule: We'll use these foundations to solve real astrophysical problems - finding equilibria and measuring the universe through numerical integration.