---
title: "Part 1: The Failure of Naive Integration"
subtitle: "Module 3: ODE Methods & Conservation | ASTR 596"
---

**Navigation:**
[← Module Overview](./00-overview.md) | [Part 2: Runge-Kutta Methods →](./02-runge-kutta.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Implement** Euler's method and **witness** its catastrophic energy drift
- **Analyze** local vs global error accumulation through Taylor series
- **Understand** why higher accuracy doesn't guarantee better long-term behavior
- **Recognize** the geometric failure modes in phase space
- **Predict** when and how integration methods will fail

---

## From Continuous to Discrete

The fundamental ODE initial value problem asks us to find a function $x(t)$ given its rate of change:

$$\frac{dx}{dt} = f(x, t), \quad x(t_0) = x_0$$

This has the formal integral solution:

$$x(t) = x_0 + \int_{t_0}^t f(x(s), s) ds$$

:::{margin}
**Initial Value Problem (IVP)**: Finding a function given its derivative and starting value - the fundamental problem of dynamics.
:::

But this **Initial Value Problem** contains a circular dependency: to find $x(t)$, we need to integrate $f(x(s), s)$, but $f$ depends on the unknown solution $x(s)$ itself!

Numerical methods break this circular dependency by making assumptions about how $f$ behaves over small time intervals:

1. **Constant derivative assumption** (Euler): Assume $f$ stays constant over $[t_n, t_{n+1}]$
2. **Linear variation assumption** (Trapezoidal): Assume $f$ varies linearly between endpoints
3. **Polynomial approximation** (Runge-Kutta): Assume $f$ follows a polynomial of degree $p$

Each assumption leads to a different family of methods with different error behaviors, stability properties, and conservation characteristics.

## The Dimensional Analysis of Timesteps

Before diving into specific methods, let's understand what constrains our choice of timestep $h$ from pure dimensional analysis.

For any oscillatory system with characteristic frequency $\omega$:
```
Nyquist requirement: h < π/ω (must sample twice per oscillation)
Accuracy requirement: h < 0.1 × 2π/ω (need ~60 points per period)
Typical choice: h ~ 0.01 × 2π/ω (600 points per period)
```

For gravitational N-body systems, multiple timescales compete:
```
Orbital period: T_orb = 2π√(a³/GM)
Close encounter: T_coll = R/(v_rel)
Shortest scale: T_min = min(all pairwise encounter times)
Required timestep: h < 0.01 × T_min

Example - Earth-Sun system:
Period: T = 365.25 days = 3.156 × 10⁷ seconds
Typical timestep: h ~ 1 day = 8.64 × 10⁴ seconds (T/365)
High accuracy: h ~ 0.1 day = 8.64 × 10³ seconds (T/3650)
```

---

## Euler's Method - The Simplest Approach

### Mathematical Formulation

:::{margin}
**Euler's method**: The simplest ODE integration scheme, using a constant derivative over each timestep.
:::

**Euler's method** is the most straightforward discretization possible. Given the ODE $\frac{dx}{dt} = f(x,t)$, we approximate:

$$x_{n+1} = x_n + h f(x_n, t_n)$$

This assumes the derivative $f$ remains constant over the entire interval $[t_n, t_{n+1}]$ - essentially extending the tangent line at $t_n$ forward by distance $h$.

### Taylor Series Analysis of Error

To understand Euler's error precisely, we need the Taylor series. The true solution at $t_{n+1}$ is:

$$x(t_n + h) = x(t_n) + h x'(t_n) + \frac{h^2}{2}x''(t_n) + \frac{h^3}{6}x'''(t_n) + O(h^4)$$

Since $x'(t_n) = f(x(t_n), t_n)$ by definition of our ODE, Euler's method gives:

$$x_{n+1} = x_n + h f(x_n, t_n)$$

:::{margin}
**Local truncation error**: Error introduced in a single step, assuming perfect starting values.
:::

The **local truncation error** - the error in one step - is:

$$\tau_n = x(t_{n+1}) - x_{n+1} = \frac{h^2}{2}x''(t_n) + O(h^3)$$

The leading error term is $O(h^2)$, making Euler locally second-order accurate.

:::{margin}
**Global error**: Total accumulated error over the entire integration interval.
:::

But errors accumulate! Over $N = T/h$ steps to reach final time $T$, the **global error** becomes:

$$E_{global} = \sum_{i=0}^{N-1} \tau_i \approx N \cdot \frac{h^2}{2}x''(\xi) = \frac{T}{h} \cdot \frac{h^2}{2}x''(\xi) = \frac{Th}{2}x''(\xi) = O(h)$$

Euler is globally first-order - halving the timestep only halves the total error.

## The Energy Drift Catastrophe

Let's see Euler fail catastrophically on the harmonic oscillator:

$$\ddot{x} = -\omega^2 x$$

Converting to first-order system:
$$\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix} = \begin{pmatrix} v \\ -\omega^2 x \end{pmatrix}$$

The true solution has constant energy $E = \frac{1}{2}(v^2 + \omega^2 x^2)$.

### Implementation and Analysis

```python
def euler_harmonic(x0, v0, omega, h, n_steps):
    """
    Integrate harmonic oscillator with Euler's method
    Demonstrates catastrophic energy drift
    """
    x, v = x0, v0
    E0 = 0.5 * (v0**2 + omega**2 * x0**2)
    
    energies = [E0]
    positions = [x0]
    
    for i in range(n_steps):
        # Euler update
        a = -omega**2 * x
        x_new = x + h * v
        v_new = v + h * a
        
        # Energy should be constant, but...
        E = 0.5 * (v_new**2 + omega**2 * x_new**2)
        energies.append(E)
        
        x, v = x_new, v_new
        positions.append(x)
    
    # Energy error grows linearly!
    return positions, energies
```

The shocking result: energy grows approximately linearly with time! After 1000 orbital periods, energy has typically increased by 10%. The orbit spirals outward, violating conservation of energy.

## Why Euler Fails: The Geometric View

In phase space (position-velocity space), the harmonic oscillator traces a circle. Each Euler step moves along the tangent to the circle, placing the new point slightly outside. The phase space area increases, violating Liouville's theorem that phase space volume must be preserved in Hamiltonian systems.

### Visualizing the Failure

```
Phase Space View of Euler's Method:

True orbit (circle):          After 10 steps:
      v                             v
      ↑                             ↑
   ●──●──●                       ●    ●
  ●       ●                    ●        ●
 ●         ●                  ●          ●
●     ○     ● → x            ●            ● → x
 ●         ●                  ●          ●
  ●       ●                    ●        ●
   ●──●──●                       ●    ●

Starting point: ○             Spiral outward!
```

## Mathematical Analysis of Energy Growth

For the harmonic oscillator with Euler's method, we can derive the exact energy growth rate. After one timestep:

$$E_{n+1} = E_n(1 + h^2\omega^2) + O(h^3)$$

The amplification factor $(1 + h^2\omega^2) > 1$ means energy grows exponentially! After $N$ steps:

$$E_N \approx E_0(1 + h^2\omega^2)^N \approx E_0 e^{Nh^2\omega^2} = E_0 e^{Th\omega^2}$$

For Earth's orbit with $h = 1$ day and $T = 1000$ years:
$$\frac{E_{final}}{E_0} \approx e^{0.1} \approx 1.1$$

A 10% energy increase - Earth would drift into a higher orbit!

## The Phase Error Problem

Even if we could tolerate energy drift, Euler has another fatal flaw: phase error. The frequency of oscillation is wrong:

$$\omega_{numerical} = \omega_{true}(1 + O(h^2))$$

After $N = T/h$ oscillations, the phase error is:
$$\Delta\phi = N \cdot O(h^2) = \frac{T}{h} \cdot O(h^2) = O(h)$$

For a binary pulsar observed for 40 years:
- True orbits: ~45,000
- Phase error with Euler: ~100 radians
- The pulsar would be completely out of phase!

## When Does Euler Work?

Despite these failures, Euler has legitimate uses:

1. **Very short integrations** where $Nh^2 \ll 1$
2. **Highly dissipative systems** where energy should decay
3. **Quick explorations** before using better methods
4. **Teaching** why better methods are needed!

## The Fundamental Lesson

Euler's method reveals a profound truth about numerical integration:

> **Local accuracy does not guarantee global stability**

Euler is locally second-order accurate ($O(h^2)$ per step), yet it systematically violates conservation laws. This isn't a bug - it's a fundamental property of the discretization.

The tangent line approximation, while locally accurate, doesn't respect the curved geometry of phase space. Each step compounds this geometric error until the qualitative behavior is wrong.

:::{admonition} Check Your Understanding
:class: question
1. Why does Euler's method have O(h²) local but O(h) global error?
2. If we used Euler backwards in time, would orbits spiral in or out?
3. For a circular orbit, by what factor does the radius grow per orbit?
4. Could we fix Euler by just using a tiny timestep?
:::

---

## Bridge to Part 2: The Quest for Better Methods

Euler's catastrophic failure motivates the search for better integration methods. We need algorithms that:
- Achieve higher-order accuracy
- Maintain stability over long times
- Preserve conservation laws
- Handle multiple timescales

In Part 2, we'll explore the Runge-Kutta family - methods that sample the derivative at multiple points to achieve higher accuracy. But as we'll discover, even fourth-order accuracy isn't enough to preserve energy over cosmic timescales.

The journey from Euler to modern integration methods is a journey from naive approximation to deep understanding of geometric structure. Each method we develop addresses specific failures of its predecessors, leading ultimately to symplectic integrators that preserve the fundamental geometry of physics.

*Next: Part 2 - Building Better Methods*