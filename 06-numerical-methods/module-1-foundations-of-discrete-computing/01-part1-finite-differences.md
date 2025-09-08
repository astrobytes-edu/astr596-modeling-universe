---
title: "Part 1: The Fundamental Paradox - Calculus on Computers"
subtitle: "Foundations of Discrete Computing | Numerical Methods Module 1 | ASTR 596"
---

**Navigation:**
[← Module Overview](./00-overview.md) | [Part 2: Numbers Aren't Real →](./02-numbers-arent-real.md)

## Learning Outcomes

By the end of this section, you will be able to:

- [ ] **Explain** why computers cannot take the limit $h \to 0$ and **analyze** the implications for numerical derivatives
- [ ] **Derive** all finite difference approximations (forward, backward, central, higher-order) from Taylor series
- [ ] **Calculate** the optimal step size $h$ by **analyzing** the trade-off between truncation error ($\propto h^p$) and round-off error ($\propto \epsilon/h$)
- [ ] **Implement** and **evaluate** all finite difference approximations with appropriate error handling
- [ ] **Predict** error scaling behavior before running code and **verify** predictions empirically

---

## Introduction: Why Numerical Derivatives Matter

Before diving into technical details, let's understand why numerical derivatives and finite precision arithmetic are foundational to computational astrophysics. Nearly every calculation you'll perform depends on these concepts.

**Derivatives are everywhere in astrophysics:**

- **Velocities and accelerations**: Every N-body simulation computes
$$\vec{v} = \frac{d\vec{r}}{dt} \quad \text{and} \quad \vec{a} = \frac{d\vec{v}}{dt}$$
where $\vec{r}$ is position, $\vec{v}$ is velocity, and $\vec{a}$ is acceleration.

- **Stellar structure**: The equations of stellar evolution involve gradients like
$$\frac{dP}{dr}, \quad \frac{dT}{dr}, \quad \text{and} \quad \frac{dM_r}{dr}$$
where $P = $ pressure, $T = $ temperature, $M_r = $ is the enclosed mass within radius $r.$

- **Radiative transfer**: The change in intensity $I$ along a ray:
$$\frac{dI}{ds} = -\kappa I + j$$ 
where $s$ is the path length traveled, $\kappa$ is the opacity, and $j$ is the emissivity.

- **Orbital mechanics**: Kepler's equation for planetary motion requires finding $\tfrac{dE}{dt}$ where $E$ is eccentric anomaly.

- **Cosmological evolution**: The Friedmann equations involve $\tfrac{da}{dt}$ where $a$ is the scale factor of the universe.

**Why numerical methods are essential:**
Most astrophysical problems cannot be solved analytically. Consider the three-body problem - there's no closed-form solution for the general case. Instead, we must use numerical approximations, which means:

- Converting continuous differential equations to discrete difference equations
- Managing errors that arise from finite precision arithmetic
- Balancing accuracy against computational cost
- Understanding when our approximations break down

**The computational challenge:**
In practice, astrophysical simulations must handle enormous dynamic ranges that push the limits of finite precision arithmetic:

- **Gravitational dynamics**: N-body simulations track forces varying by factors up to ~$10^{18}$ (from close binaries at ~0.01 AU to globular cluster scales at ~100 pc), exceeding the ~$10^{16}$ dynamic range of double precision and forcing us to use hierarchical methods or separate coordinate systems
- **Star formation**: Simulating giant molecular clouds requires resolving both the ~100 pc cloud scale and ~100 AU protostellar disks - a factor of $10^{6}$ in length scale and $10^{11}$ in density (from $10^{-24}$ g/cm³ in the diffuse ISM to $10^{-13}$ g/cm³ in dense molecular cores)
- **Temporal evolution**: Stellar evolution codes span from dynamical timescales (hours) to main sequence lifetimes (billions of years) - a factor of $10^{13}$ where errors accumulate at every timestep
- **Mixed scales**: Galaxy simulations must resolve both individual star-forming regions (~1 pc) and galactic halos (~100 kpc), while maintaining energy and momentum conservation

Double precision provides only ~16 decimal digits. We cannot simultaneously represent these extreme scales with full precision. We must understand:

- How computers represent numbers and where precision is lost
- How errors propagate through calculations
- How to reformulate problems for numerical stability
- When to trust our results and when to be skeptical

:::{admonition} Course Units Reminder
:class: important
Throughout this course, we use CGS units and astronomical units:

- **Length**: cm, AU ($1.496 \times 10^{13}$ cm), pc ($3.086 \times 10^{18}$ cm)
- **Mass**: g, $M_{\odot}$ ($1.989 \times 10^{33}$ g)
- **Time**: s, day (86400 s), year ($3.156 \times 10^{7}$ s)
- **G**: $6.674 \times 10^{-8}$ cm³ g⁻¹ s⁻²
:::

This section provides the rigorous foundation you need to write reliable code for astrophysical simulations. The principles you learn here will apply to every project in this course and throughout your research career.

---

## The Core Problem

Recall that the mathematical definition of a derivative of a function $f(x)$ at a point $x$ is:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

:::{margin}
**Derivative**: The instantaneous rate of change of a function, fundamental to describing motion, forces, and evolution in physics.

**Floating-point arithmetic**: A system for representing real numbers in computers using a fixed number of bits, leading to finite precision.
:::

where $h$ represents a small displacement from the point $x$. This definition relies on taking the limit as $h$ approaches zero - we evaluate the slope of secant lines through points that get arbitrarily close together until we obtain the slope of the tangent line at $x$. The **derivative** is fundamental to physics, describing velocities, accelerations, and all forms of change.

```{figure} figures/1_1_derivative_paradox.png
---
name: fig-derivative-paradox
width: 100%
---
The derivative paradox: mathematical definition requires $h \to 0$, but computers cannot take true limits. Left: analytical derivative (smooth tangent line). Right: numerical approximation using finite $h$ values showing convergence - smaller $h=0.1$ approaches the true slope better than larger $h=0.3$.
```

On a computer, this mathematical ideal encounters a fundamental obstacle. Computers represent numbers using floating-point arithmetic with finite precision (typically 64 bits for a double-precision number).

This **floating-point arithmetic** means:

1. **We cannot represent $h = 0$ exactly** - Division by zero would cause the program to crash or return undefined/infinite values.

2. **We cannot make $h$ arbitrarily small** - Below a certain threshold (around $10^{-16}$ for double precision), the computer cannot distinguish between $x$ and $x+h$ due to rounding. When we try to compute $f(x+h) - f(x)$, both values round to the same floating-point number, giving us $0/h = 0$, which is completely wrong.

3. **Even moderate values of $h$ cause problems** - When $h$ is very small (say $10^{-10}$), and $f(x+h) \approx f(x)$, we're subtracting two nearly equal numbers. Due to finite precision, we might only retain a few significant digits in the difference, leading to catastrophic cancellation.

:::{margin}
**Catastrophic cancellation**: Loss of precision when subtracting nearly equal floating-point numbers.

**Finite difference**: Approximation of derivatives using function values at discrete points rather than taking a limit.

**Truncation error**: Error from approximating an infinite process with finite terms.

**Round-off error**: Error from finite precision arithmetic.
:::

This **catastrophic cancellation** is one of the most common sources of numerical error in scientific computing.

Therefore, we must approximate the derivative using a finite difference:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h} \quad \text{for some finite } h > 0$$

This **finite difference** approximation creates a fundamental trade-off:

- **If $h$ is too large**: The approximation is poor because we're measuring the slope of a secant line far from the point of interest. This introduces **truncation error** proportional to $h$.
- **If $h$ is too small**: Floating-point **round-off errors** dominate. When $f(x+h)$ and $f(x)$ are nearly equal, their difference loses significant digits, and dividing by a tiny $h$ amplifies this error.

The optimal value of $h$ must balance these competing effects. Through rigorous analysis (which we'll derive shortly), this optimal value turns out to be approximately $h \approx \sqrt{\epsilon_{\text{machine}}} \approx 10^{-8}$ for forward differences, where $\epsilon_{\text{machine}} \approx 2.2 \times 10^{-16}$ is the machine epsilon for double precision.

:::{margin}
**Machine epsilon**: The smallest number $\epsilon$ such that $1 + \epsilon \neq 1$ in floating-point arithmetic.
:::

This **machine epsilon** represents the fundamental limit of floating-point precision. This fundamental limitation means that on a computer, we can typically only compute derivatives to about 8 digits of accuracy using forward differences, even though our numbers have 16 digits of precision. This is why understanding numerical methods is crucial.

:::{admonition} Check Your Understanding
:class: question

1. Why can't we just use $h = 10^{-20}$ for maximum accuracy?
2. What happens if we try to compute $(1 + 10^{-17}) - 1$ in double precision?
3. If machine epsilon is $\approx 10^{-16}$, why is optimal $h \approx 10^{-8}$ for forward difference?
:::

## The Complete Finite Difference Landscape

Now that we understand the fundamental challenge, let's explore all the ways we can approximate derivatives numerically. Each method emerges from different manipulations of Taylor series, and each has distinct advantages and trade-offs.

### Intuition: Why Taylor Series?

:::{margin}
**Taylor series expansion**: Representation of a function as an infinite sum of terms calculated from its derivatives at a single point.

**Truncation error**: Error introduced by approximating an infinite process (like a Taylor series or a limit) with a finite number of terms. For example, approximating $e^x$ with $1 + x + x^2/2$ truncates the infinite series, creating an error of $x^3/6 + x^4/24 + ...$
:::

Taylor series answers the question: "How does a function change near a point?" It builds the function from its derivatives, like constructing a curve from its slope, curvature, and higher-order bending.

The **Taylor series expansion** of a function $f(x)$ around a point $x_0$ is:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \frac{f'''(x_0)}{3!}(x-x_0)^3 + \cdots$$

Or more compactly:

$$\boxed{f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n}$$

Each term adds more detail: the first gives the value, the second adds the slope, the third adds curvature, and so on. For numerical methods, we truncate this infinite series after a few terms, which introduces **truncation error**. By understanding which terms we keep and which we discard, we can predict exactly what error our approximations introduce. Think of it as a mathematical microscope that reveals the fine structure of functions.

### Forward Difference: Looking Ahead

:::{margin}
**Forward difference**: Finite difference approximation using the current point and a point ahead.
:::

The **forward difference** method uses information about the function at the current point $x$ and a point ahead at $x+h$. To understand its accuracy, we start with the Taylor expansion of $f(x+h)$ around the point $x$:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + O(h^4)$$

This expansion tells us exactly how $f(x+h)$ relates to the function value and derivatives at $x$. Now, if we solve for $f'(x)$:

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) + O(h^3)$$

This reveals that the forward difference approximation:

$$\boxed{f'(x) \approx \frac{f(x+h) - f(x)}{h}}$$

:::{margin}
**First-order method**: A numerical method where the error decreases linearly with step size.
:::

has a leading error term of $-\frac{h}{2}f''(x)$. This is a **first-order method** because the error is proportional to $h^1$. The negative sign tells us that for functions with positive second derivatives (convex functions), forward difference underestimates the true derivative.

### Backward Difference: Looking Behind

:::{margin}
**Backward difference**: Finite difference approximation using a point behind and the current point.
:::

The **backward difference** method uses the current point $x$ and a point behind at $x-h$. Following the same Taylor series approach:

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + O(h^4)$$

Rearranging to isolate $f'(x)$:

$$f'(x) = \frac{f(x) - f(x-h)}{h} + \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) + O(h^3)$$

The backward difference approximation:

$$\boxed{f'(x) \approx \frac{f(x) - f(x-h)}{h}}$$

has a leading error term of $+\frac{h}{2}f''(x)$, opposite in sign to forward difference! This means backward difference overestimates the derivative for convex functions.

### Central Difference: The Sweet Spot

:::{margin}
**Central difference**: Finite difference approximation using points symmetrically placed around the evaluation point.

**Second-order accurate**: A numerical method where error decreases quadratically with step size.
:::

The **central difference** method is where mathematical elegance meets practical advantage. By using points symmetrically placed around $x$, something remarkable happens. Let's expand both $f(x+h)$ and $f(x-h)$:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \frac{h^4}{24}f^{(4)}(x) + O(h^5)$$

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + \frac{h^4}{24}f^{(4)}(x) + O(h^5)$$

When we subtract these two expressions:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{2h^3}{6}f'''(x) + O(h^5)$$

Notice that all even-order derivative terms cancel exactly! This is due to the symmetry of the method. Solving for $f'(x)$:

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} - \frac{h^2}{6}f'''(x) + O(h^4)$$

The central difference approximation:

$$\boxed{f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}}$$

is **second-order accurate** with a leading error term of $-\frac{h^2}{6}f'''(x)$. This quadratic dependence on $h$ means that halving the step size reduces the error by a factor of 4, compared to only a factor of 2 for forward or backward differences.

### The Power of Symmetry

Symmetry in numerical methods often leads to cancellation of systematic errors. This principle extends beyond finite differences to integration methods (Simpson's rule), PDE solvers (centered schemes), and even Monte Carlo methods (antithetic variates). When you have a choice, symmetric methods typically provide better accuracy for the same computational cost. The central difference method exploits this symmetry to achieve higher accuracy without additional function evaluations.

### Higher-Order Methods: When You Need More Accuracy

:::{margin}
**Fourth-order accuracy**: A numerical method where error decreases as the fourth power of step size.
:::

For applications requiring higher precision, we can use more points to achieve **fourth-order accuracy**. Through careful manipulation of Taylor series, we can derive:

$$\boxed{f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + O(h^4)}$$

The coefficients $(-1, 8, -8, 1)/12$ are specifically chosen to eliminate the $O(h)$, $O(h^2)$, and $O(h^3)$ error terms.

### Beyond Fourth Order: Diminishing Returns

The pattern continues: sixth-order methods use 7 points, eighth-order use 9 points. However, for noisy data or non-smooth functions, higher-order methods can amplify errors rather than reduce them. Additionally, near boundaries or discontinuities, you may not have enough points for high-order methods. This is why second-order central difference remains the default choice in practice - it provides an excellent balance of accuracy, robustness, and simplicity.

```{figure} figures/1_4_error_scaling.png
---
name: fig-error-scaling
width: 100%
---
Error scaling comparison for finite difference methods applied to $f(x)=\sin(x)$ at $x=1$. Forward (dashed) and backward (solid) differences both show $O(h)$ convergence, central difference (dotted) shows $O(h^2)$, and 4th-order central (solid) shows $O(h^4)$ convergence in the theoretical regime. All methods eventually hit the machine epsilon floor ($\approx 10^{-16}$) where round-off dominates. Gray reference lines show exact theoretical scaling.
```

:::{admonition} Connection to N-body (Project 2)
:class: tip
In your N-body simulation, you'll compute (via Euler's method):

- Velocity: $\vec{v}_{n+1} = \vec{v}_n + \vec{a}_n \Delta t$ (forward difference)
- Position: $\vec{r}_{n+1} = \vec{r}_n + \vec{v}_n \Delta t$ (forward difference)

But wait - why not use central difference for better accuracy? Because we're marching forward in time! We can't use future values we haven't computed yet. This is why we need special integration methods ([Module 3](../03-ODE-methods-and-conservation/00-module3-overview.md)).
:::

### Implementation: Compact Code Examples

Here's a minimal implementation of all finite difference methods:

```python
import numpy as np

def finite_diff(f, x, h, method='central'):
    """All finite difference methods in one place"""
    try:
        if method == 'forward':
            return (f(x + h) - f(x)) / h
        elif method == 'backward':
            return (f(x) - f(x - h)) / h
        elif method == 'central':
            return (f(x + h) - f(x - h)) / (2 * h)
        elif method == 'central4':
            return (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h)) / (12*h)
        else:
            raise ValueError(f"Unknown method: {method}")
    except (ZeroDivisionError, OverflowError) as e:
        raise ValueError(f"Numerical error in {method} difference: {e}")
```

:::{warning} **Common Mistakes:**

1. Using `h = epsilon` directly (too small - roundoff dominates!)
2. Forgetting to scale `h` with `x` for large values
3. Using same `h` for all variables regardless of their scales
4. Not checking if `f(x+h) == f(x)` due to underflow
:::

## The Fundamental Trade-off: Rigorous Derivation of Optimal $h$

We've established that choosing $h$ involves balancing truncation error (from approximating the derivative) and round-off error (from finite precision arithmetic). Let's now rigorously derive the optimal value of $h$ for each method.

### Forward Difference: Optimal Step Size

For the forward difference method, we have two sources of error:

1. **Truncation error**: $E_{\text{trunc}} = \frac{h}{2}|f''(x)|$
2. **Round-off error**: $E_{\text{round}} \approx \frac{\epsilon |f(x)|}{h}$

The total error is:

$$E_{\text{total}}(h) = \frac{h}{2}|f''(x)| + \frac{\epsilon |f(x)|}{h}$$

To find the optimal $h$, we minimize this by taking the derivative with respect to $h$:

$$\frac{dE_{\text{total}}}{dh} = \frac{|f''(x)|}{2} - \frac{\epsilon |f(x)|}{h^2} = 0$$

Solving for $h$:

$$h_{\text{opt}} = \sqrt{\frac{2\epsilon |f(x)|}{|f''(x)|}}$$

For functions where $|f(x)|$ and $|f''(x)|$ are of similar magnitude:

$$h_{\text{opt}} \approx \sqrt{\epsilon} \approx 1.5 \times 10^{-8}$$

This is a remarkable result: even though our numbers have 16 digits of precision, we can only compute derivatives to about 8 digits of accuracy using forward differences!

### Central Difference: Even Better Optimal Step Size

For central difference, the error components are:

1. **Truncation error**: $E_{\text{trunc}} = \frac{h^2}{6}|f'''(x)|$
2. **Round-off error**: $E_{\text{round}} = \frac{\epsilon |f(x)|}{h}$

Following the same minimization procedure:

$$h_{\text{opt}} = \left(\frac{3\epsilon |f(x)|}{|f'''(x)|}\right)^{1/3} \approx \epsilon^{1/3} \approx 6 \times 10^{-6}$$

This is crucial: central difference allows a step size that is about 300 times larger than forward difference while achieving better accuracy!

:::{admonition} Check Your Understanding
:class: question

1. If you halve $h$ for central difference, how much does the error reduce?
2. Why does $f(x+h) - f(x-h)$ cancel even-order terms?
3. For a fourth-order method with error $O(h^4)$, what would be the optimal $h$?
:::

## Visualizing the Error Trade-off

**Why this matters**: The competition between truncation and round-off error creates a fundamental limit on numerical accuracy. This visualization shows why we can't just make $h$ arbitrarily small - there's an optimal value where total error is minimized. Understanding this trade-off is crucial for choosing appropriate step sizes in your simulations.

```{figure} figures/1_2_error_landscape.png
---
name: fig-error-tradeoff
width: 100%
---
The fundamental trade-off in numerical derivatives showing why $h$ cannot be arbitrarily small. Thick lines show component errors: truncation error decreases as $O(h)$ for forward differences and $O(h^2)$ for central differences, while round-off error increases as $O(\epsilon/h)$. Total errors (dotted lines) have minima at optimal $h$ values: $\sqrt{2\epsilon}$ for forward and $(6\epsilon)^{1/3}$ for central differences.
```

For large $h$, truncation error dominates. For small $h$, round-off error dominates. The optimal $h$ sits at the bottom of the U.

<!---
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_error_landscape():
    """Visualize the U-shaped error curve that governs numerical accuracy"""
    h_values = np.logspace(-16, 0, 200)
    epsilon = 2.2e-16
    
    # Forward difference errors (normalized function)
    trunc = 0.5 * h_values  # Truncation: O(h)
    round = epsilon / h_values  # Round-off: O(ε/h)
    total = trunc + round
    
    # Central difference errors
    trunc_c = h_values**2 / 6  # Truncation: O(h²)
    total_c = trunc_c + round
    
    # Find optimal h values
    h_opt_forward = h_values[np.argmin(total)]
    h_opt_central = h_values[np.argmin(total_c)]
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.loglog(h_values, total, 'b-', linewidth=2, label='Total (forward)')
    plt.loglog(h_values, total_c, 'g-', linewidth=2, label='Total (central)')
    plt.axvline(h_opt_forward, color='b', linestyle=':', alpha=0.7)
    plt.axvline(h_opt_central, color='g', linestyle=':', alpha=0.7)
    
    plt.xlabel('Step size h')
    plt.ylabel('Error')
    plt.title('The Fundamental Trade-off: Why h Cannot Be Arbitrarily Small')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return h_opt_forward, h_opt_central
```
--->

## Practical Algorithm for Choosing $h$

:::{admonition} Debugging Numerical Derivatives
:class: tip
When derivatives seem wrong:

1. Check if $h$ is appropriate for your $x$ scale
2. Verify $f(x+h) \neq f(x)$ (not lost to rounding)
3. Try multiple $h$ values - error should follow U-curve
4. Compare forward, central, and analytical (if known)
5. Look for catastrophic cancellation in your function
6. Consider if your function is smooth enough for the method order
:::

**Why this matters**: In real applications, you won't know the derivatives $f''(x)$ or $f'''(x)$ needed for the theoretical optimal $h$. This practical algorithm estimates appropriate step sizes based on the problem scale and machine precision. This is what you'll actually use in your research code.

```python
import numpy as np

def practical_optimal_h(x, method='central'):
    """
    Choose h based on method and problem scale
    
    The key insight: h should scale with |x| to maintain
    relative precision for large values
    """
    epsilon = 2.2e-16
    
    if abs(x) < epsilon:
        raise ValueError(f"x too close to zero for reliable derivatives")
    
    x_scale = max(abs(x), 1.0)  # Avoid division issues near zero
    
    if method in ['forward', 'backward']:
        h_opt = np.sqrt(epsilon) * x_scale
    elif method == 'central':
        h_opt = epsilon**(1/3) * x_scale
    elif method == 'central4':
        h_opt = epsilon**(1/5) * x_scale
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure h is not too small relative to x
    h_final = max(h_opt, epsilon * abs(x))
    return h_final

# Example: Choosing h for different problem scales
x_values = [1.0, 1.496e13, 1.989e33]  # dimensionless, 1 AU, 1 M_sun
for x in x_values:
    h = practical_optimal_h(x, 'central')
    print(f"x = {x:.2e}, optimal h = {h:.2e}, h/x = {h/x:.2e}")
```

---

## Bridge to Part 2: From Derivatives to Machine Arithmetic

You now understand the fundamental paradox: computers cannot take true limits, forcing us to balance truncation and round-off error. But where does this round-off error come from?

In Part 2, we'll dive deep into how computers represent numbers. You'll discover:

- Why 0.1 + 0.2 ≠ 0.3 in binary arithmetic
- How to find and work with machine epsilon
- When catastrophic cancellation destroys your calculations
- How errors propagate through millions of timesteps

The derivative methods you've just learned are built on the foundation of floating-point arithmetic. Understanding this foundation will help you write more robust code and diagnose mysterious numerical failures in your simulations.

*Next: Part 2 - Numbers Aren't Real*