---
title: "Part 2: Building Better Methods - Runge-Kutta"
subtitle: "Module 3: ODE Methods & Conservation | ASTR 596"
---

**Navigation:**
[← Part 1: Failure of Naive Integration](./01-failure-of-naive.md) | [Part 3: Symplectic Integration →](./03-symplectic.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Derive** RK2 and RK4 from multivariate Taylor expansions
- **Connect** RK weights to quadrature rules from Module 2
- **Implement** adaptive timestep control
- **Analyze** when higher-order methods help vs harm
- **Choose** appropriate RK methods based on problem requirements

---

## The Key Insight: Sample Multiple Points

Euler's fatal flaw is assuming the derivative stays constant over the entire timestep. Real functions curve. Runge-Kutta methods evaluate the derivative at carefully chosen intermediate points, then combine these samples with specific weights to achieve higher-order accuracy.

## RK2 - The Midpoint Method

### Geometric Intuition

Instead of blindly following the tangent from the starting point, we:
1. Take a half-step using the initial derivative
2. Evaluate the derivative at this midpoint
3. Use this midpoint derivative for the full step

### Complete Mathematical Formulation

:::{margin}
**Predictor-corrector method**: A two-stage process where an initial estimate (predictor) is refined using better information (corrector).
:::

The RK2 midpoint method is a **predictor-corrector method**:

**Stage 1 - Predictor:**
$$k_1 = f(x_n, t_n)$$
$$x_{mid} = x_n + \frac{h}{2}k_1$$

**Stage 2 - Corrector:**
$$k_2 = f(x_{mid}, t_n + \frac{h}{2})$$
$$x_{n+1} = x_n + h k_2$$

### Error Analysis via Taylor Series

Using multivariate Taylor expansion, the midpoint derivative becomes:
$$f(x_{mid}, t_{mid}) = f + \frac{h}{2}\left(\frac{\partial f}{\partial t} + f\frac{\partial f}{\partial x}\right) + O(h^2)$$

The RK2 update matches the true solution through $O(h^2)$:
- Local truncation error: $O(h^3)$
- Global error: $O(h^2)$

RK2 is second-order accurate - halving the timestep reduces error by a factor of 4.

### Implementation

```python
def rk2_step(x, t, h, f):
    """
    Single RK2 (midpoint method) step
    Second-order accurate
    """
    # Stage 1: Evaluate at starting point
    k1 = f(x, t)
    
    # Predictor: Half step to midpoint
    x_mid = x + 0.5 * h * k1
    t_mid = t + 0.5 * h
    
    # Stage 2: Evaluate at midpoint
    k2 = f(x_mid, t_mid)
    
    # Corrector: Full step using midpoint derivative
    return x + h * k2
```

## RK4 - The Classical Workhorse

:::{margin}
**Runge-Kutta 4 (RK4)**: The most popular ODE method, achieving 4th-order accuracy with 4 function evaluations.
:::

**RK4** samples the derivative at four carefully chosen points:

**Stage 1:** $k_1 = f(x_n, t_n)$

**Stage 2:** $k_2 = f(x_n + \frac{h}{2}k_1, t_n + \frac{h}{2})$

**Stage 3:** $k_3 = f(x_n + \frac{h}{2}k_2, t_n + \frac{h}{2})$

**Stage 4:** $k_4 = f(x_n + hk_3, t_n + h)$

**Final update:** $x_{n+1} = x_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

The weights $\frac{1}{6}, \frac{2}{6}, \frac{2}{6}, \frac{1}{6}$ are precisely Simpson's rule weights! RK4 is essentially applying Simpson's quadrature to the integral.

### The Connection to Quadrature

Remember from Module 2:
- **Euler = Rectangle rule**: Sample at left endpoint
- **RK2 = Midpoint rule**: Sample at center
- **RK4 = Simpson's rule**: Weighted average with parabolic fit

This connection reveals why RK4 is so effective - it inherits the fourth-order accuracy of Simpson's rule.

### Implementation

```python
def rk4_step(x, t, h, f):
    """
    Single RK4 step
    Fourth-order accurate
    """
    k1 = f(x, t)
    k2 = f(x + 0.5*h*k1, t + 0.5*h)
    k3 = f(x + 0.5*h*k2, t + 0.5*h)
    k4 = f(x + h*k3, t + h)
    
    return x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
```

### Error Analysis

Through Taylor series expansion:
- Local truncation error: $O(h^5)$
- Global error: $O(h^4)$

This means halving the timestep reduces error by a factor of 16!

## The General Runge-Kutta Framework

All explicit RK methods follow this pattern:

$$k_i = f\left(x_n + h\sum_{j=1}^{i-1} a_{ij}k_j, t_n + c_ih\right)$$

$$x_{n+1} = x_n + h\sum_{i=1}^s b_i k_i$$

The coefficients $(a_{ij}, b_i, c_i)$ define the method and are typically displayed in a Butcher tableau:

```
c₁ |
c₂ | a₂₁
c₃ | a₃₁ a₃₂
⋮  | ⋮
cₛ | aₛ₁ aₛ₂ ... aₛ,ₛ₋₁
---|--------------------
   | b₁  b₂  ...  bₛ
```

For RK4:
```
0   |
1/2 | 1/2
1/2 | 0   1/2
1   | 0   0   1
----|----------------
    | 1/6 2/6 2/6 1/6
```

## Adaptive Timestep Control

Real problems have varying timescales. Adaptive methods adjust $h$ dynamically by comparing two estimates of different order and using their difference to estimate error.

### The Runge-Kutta-Fehlberg Method (RK45)

RK45 computes both 4th and 5th order solutions using the same function evaluations:

```python
def rk45_adaptive(x, t, h, f, tolerance):
    """
    Adaptive RK45 with error control
    Automatically adjusts timestep
    """
    # Compute RK4 and RK5 estimates (shared k values)
    k1 = f(x, t)
    k2 = f(x + h/4*k1, t + h/4)
    k3 = f(x + 3*h/32*k1 + 9*h/32*k2, t + 3*h/8)
    k4 = f(x + 1932*h/2197*k1 - 7200*h/2197*k2 + 7296*h/2197*k3, t + 12*h/13)
    k5 = f(x + 439*h/216*k1 - 8*h*k2 + 3680*h/513*k3 - 845*h/4104*k4, t + h)
    k6 = f(x - 8*h/27*k1 + 2*h*k2 - 3544*h/2565*k3 + 1859*h/4104*k4 - 11*h/40*k5, t + h/2)
    
    # Fourth-order estimate
    x_4th = x + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
    
    # Fifth-order estimate
    x_5th = x + h*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
    
    # Error estimate
    error = np.linalg.norm(x_5th - x_4th)
    
    # Optimal timestep from error scaling
    h_opt = h * 0.9 * (tolerance/error)**(1/5)
    
    if error < tolerance:
        return x_5th, h_opt, True  # Accept step
    else:
        return None, h_opt, False  # Reject, retry with smaller h
```

### When to Use Adaptive Methods

**Good for:**
- Problems with varying timescales
- Unknown dynamics
- Achieving specified accuracy
- Efficiency (larger steps when possible)

**Bad for:**
- Long-term Hamiltonian systems (destroys symplecticity)
- Problems requiring uniform time grid
- Parallel simulations (variable timesteps complicate synchronization)

## Performance Comparison

Let's compare methods on the Kepler problem (elliptical orbit):

| Method | Order | Steps/Orbit | Energy Error After 100 Orbits | CPU Time |
|--------|-------|-------------|-------------------------------|----------|
| Euler | 1 | 1000 | +10% | 1.0× |
| RK2 | 2 | 100 | +0.1% | 0.2× |
| RK4 | 4 | 30 | +0.0001% | 0.12× |
| RK45 | 4/5 | 15-50 | +0.00001% | 0.15× |

RK4 achieves 100,000× better accuracy than Euler with 8× less computation!

## The Dark Side of High-Order Methods

Despite their accuracy, RK methods have a fatal flaw for long-term integration:

### Energy Drift Comparison

After 10,000 orbits with timestep $h = 0.01$:
- **Euler**: +100% energy drift (orbit doubled in size!)
- **RK2**: +10% energy drift (significant growth)
- **RK4**: +0.1% energy drift (small but systematic)

Even RK4, accurate to O(h⁴) per step, exhibits systematic energy drift! After millions of orbits (typical for solar system simulations), even this tiny drift becomes catastrophic.

### Why RK Methods Drift

The fundamental issue: RK methods don't preserve the geometric structure of phase space. They slightly expand or contract volumes, violating Liouville's theorem. Each step compounds this violation until energy drifts unboundedly.

Higher order reduces the drift rate but doesn't eliminate it. For billion-year simulations, we need qualitatively different methods.

:::{admonition} Check Your Understanding
:class: question
1. Why does RK2 sample at the midpoint rather than the endpoint?
2. What's the connection between RK4 weights and Simpson's rule?
3. When would adaptive timesteps be essential vs harmful?
4. Why doesn't higher order eliminate energy drift?
:::

---

## Bridge to Part 3: The Need for Geometric Integration

The Runge-Kutta family achieves impressive local accuracy through careful sampling of derivatives. RK4's fourth-order accuracy seems like it should be sufficient for any problem. Yet for long-term integration of Hamiltonian systems, even this isn't enough.

The issue isn't accuracy—it's structure preservation. In Part 3, we'll explore symplectic integrators that sacrifice local accuracy for global stability. These methods preserve the fundamental geometry of phase space, keeping energy bounded even over cosmic timescales.

The transition from RK to symplectic methods represents a philosophical shift: from minimizing error to preserving physics. As we'll see, a second-order symplectic method that keeps your solar system stable for a billion years beats a tenth-order method that spirals planets into the sun.

*Next: Part 3 - Symplectic Integration*