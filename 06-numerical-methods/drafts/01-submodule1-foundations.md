---
title: "Module 1: Foundations of Discrete Computing"
subtitle: "Numerical Methods | ASTR 596"
exports:
  - format: pdf
---

**Navigation:**

[← Course Schedule & Important Dates](../01-course-info/02-astr596-schedule.md) | [Module 1 Home](./00-overview.md) | [Part 1: The Fundamental Paradox →](./01-submodule1-foundations.md)

*"From Continuous Physics to Discrete Reality"*

## Learning Outcomes

By the end of this submodule, you will be able to:

- [ ] **Explain** why computers cannot take the limit $h \to 0$ and **analyze** the implications for numerical derivatives
- [ ] **Calculate** machine epsilon for different data types and **predict** when round-off errors dominate
- [ ] **Identify** and **classify** the three types of numerical error (round-off, truncation, propagation) in code
- [ ] **Apply** Taylor series to **derive** numerical methods of different orders
- [ ] **Derive** the optimal step size $h$ by **analyzing** the trade-off between truncation error ($\propto h^p$) and round-off error ($\propto \epsilon/h$)
- [ ] **Implement** and **evaluate** all finite difference approximations (forward, backward, central, higher-order)
- [ ] **Predict** error scaling behavior before running code and **verify** predictions empirically
- [ ] **Design** numerical derivative calculations that minimize error for specific astrophysical problems
- [ ] **Diagnose** numerical issues in derivative calculations and **reformulate** expressions to avoid catastrophic cancellation

---

## Introduction: Why Numerical Derivatives Matter

Before diving into technical details, let's understand why numerical derivatives and finite precision arithmetic are foundational to computational astrophysics. Nearly every calculation you'll perform depends on these concepts.

**Derivatives are everywhere in astrophysics:**
- **Velocities and accelerations**: Every N-body simulation computes $\vec{v} = d\vec{r}/dt$ and $\vec{a} = d\vec{v}/dt$
- **Stellar structure**: The equations of stellar evolution involve gradients like $dP/dr$, $dT/dr$, $dM/dr$
- **Radiative transfer**: The change in intensity along a ray: $dI/ds = -\kappa I + j$
- **Orbital mechanics**: Kepler's equation requires finding $dE/dt$ where $E$ is eccentric anomaly
- **Cosmological evolution**: The Friedmann equations involve $da/dt$ where $a$ is the scale factor

**Why numerical methods are essential:**
Most astrophysical problems cannot be solved analytically. Consider the three-body problem - there's no closed-form solution for the general case. We must use numerical approximations, which means:
- Converting continuous differential equations to discrete difference equations
- Managing errors that arise from finite precision arithmetic
- Balancing accuracy against computational cost
- Understanding when our approximations break down

**The computational challenge:**
Astrophysics spans the widest range of scales in science - from stellar nuclear reactions ($10^{-13}$ cm) to the observable universe ($10^{28}$ cm). No computer can represent this full range simultaneously with finite precision numbers. We must understand:
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

This submodule provides the rigorous foundation you need to write reliable code for astrophysical simulations. The principles you learn here will apply to every project in this course and throughout your research career.

---

## Part 1: The Fundamental Paradox - Calculus on Computers

### The Core Problem

Recall that the mathematical definition of a derivative of a function $f(x)$ at a point $x$ is:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

:::{margin}
**Derivative**: The instantaneous rate of change of a function, fundamental to describing motion, forces, and evolution in physics.
:::

where $h$ represents a small displacement from the point $x$. This definition relies on taking the limit as $h$ approaches zero - we evaluate the slope of secant lines through points that get arbitrarily close together until we obtain the slope of the tangent line at $x$. The **derivative** is fundamental to physics, describing velocities, accelerations, and all forms of change.

On a computer, this mathematical ideal encounters a fundamental obstacle. Computers represent numbers using floating-point arithmetic with finite precision (typically 64 bits for a double-precision number). 

:::{margin}
**Floating-point arithmetic**: A system for representing real numbers in computers using a fixed number of bits, leading to finite precision.
:::

This **floating-point arithmetic** means:

1. **We cannot represent $h = 0$ exactly** - Division by zero would cause the program to crash or return undefined/infinite values.

2. **We cannot make $h$ arbitrarily small** - Below a certain threshold (around $10^{-16}$ for double precision), the computer cannot distinguish between $x$ and $x+h$ due to rounding. When we try to compute $f(x+h) - f(x)$, both values round to the same floating-point number, giving us $0/h = 0$, which is completely wrong.

3. **Even moderate values of $h$ cause problems** - When $h$ is very small (say $10^{-10}$), and $f(x+h) \approx f(x)$, we're subtracting two nearly equal numbers. Due to finite precision, we might only retain a few significant digits in the difference, leading to catastrophic cancellation.

:::{margin}
**Catastrophic cancellation**: Loss of precision when subtracting nearly equal floating-point numbers.
:::

This **catastrophic cancellation** is one of the most common sources of numerical error in scientific computing.

Therefore, we must approximate the derivative using a finite difference:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h} \quad \text{for some finite } h > 0$$

:::{margin}
**Finite difference**: Approximation of derivatives using function values at discrete points rather than taking a limit.
:::

This **finite difference** approximation creates a fundamental trade-off:

:::{margin}
**Truncation error**: Error from approximating an infinite process with finite terms.

**Round-off error**: Error from finite precision arithmetic.
:::

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

### The Complete Finite Difference Landscape

Now that we understand the fundamental challenge, let's explore all the ways we can approximate derivatives numerically. Each method emerges from different manipulations of Taylor series, and each has distinct advantages and trade-offs.

### Intuition: Why Taylor Series?

:::{margin}
**Taylor series**: Representation of a function as an infinite sum of terms calculated from its derivatives at a single point.
:::

**Taylor series** answers the question: "How does a function change near a point?" It builds the function from its derivatives, like constructing a curve from its slope, curvature, and higher-order bending. For numerical methods, we use Taylor series to understand exactly what error we introduce when we approximate. Think of it as a mathematical microscope that reveals the fine structure of functions.

#### Forward Difference: Looking Ahead

:::{margin}
**Forward difference**: Finite difference approximation using the current point and a point ahead.
:::

The **forward difference** method uses information about the function at the current point $x$ and a point ahead at $x+h$. To understand its accuracy, we start with the Taylor expansion of $f(x+h)$ around the point $x$:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + O(h^4)$$

This expansion tells us exactly how $f(x+h)$ relates to the function value and derivatives at $x$. Now, if we solve for $f'(x)$:

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) + O(h^3)$$

This reveals that the forward difference approximation:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

:::{margin}
**First-order method**: A numerical method where the error decreases linearly with step size.
:::

has a leading error term of $-\frac{h}{2}f''(x)$. This is a **first-order method** because the error is proportional to $h^1$. The negative sign tells us that for functions with positive second derivatives (convex functions), forward difference underestimates the true derivative.

#### Backward Difference: Looking Behind

:::{margin}
**Backward difference**: Finite difference approximation using a point behind and the current point.
:::

The **backward difference** method uses the current point $x$ and a point behind at $x-h$. Following the same Taylor series approach:

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + O(h^4)$$

Rearranging to isolate $f'(x)$:

$$f'(x) = \frac{f(x) - f(x-h)}{h} + \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) + O(h^3)$$

The backward difference approximation:

$$f'(x) \approx \frac{f(x) - f(x-h)}{h}$$

has a leading error term of $+\frac{h}{2}f''(x)$, opposite in sign to forward difference! This means backward difference overestimates the derivative for convex functions.

#### Central Difference: The Sweet Spot

:::{margin}
**Central difference**: Finite difference approximation using points symmetrically placed around the evaluation point.
:::

The **central difference** method is where mathematical elegance meets practical advantage. By using points symmetrically placed around $x$, something remarkable happens. Let's expand both $f(x+h)$ and $f(x-h)$:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \frac{h^4}{24}f^{(4)}(x) + O(h^5)$$

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + \frac{h^4}{24}f^{(4)}(x) + O(h^5)$$

When we subtract these two expressions:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{2h^3}{6}f'''(x) + O(h^5)$$

Notice that all even-order derivative terms cancel exactly! This is due to the symmetry of the method. Solving for $f'(x)$:

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} - \frac{h^2}{6}f'''(x) + O(h^4)$$

The central difference approximation:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

:::{margin}
**Second-order accurate**: A numerical method where error decreases quadratically with step size.
:::

is **second-order accurate** with a leading error term of $-\frac{h^2}{6}f'''(x)$. This quadratic dependence on $h$ means that halving the step size reduces the error by a factor of 4, compared to only a factor of 2 for forward or backward differences.

### The Power of Symmetry

Symmetry in numerical methods often leads to cancellation of systematic errors. This principle extends beyond finite differences to integration methods (Simpson's rule), PDE solvers (centered schemes), and even Monte Carlo methods (antithetic variates). When you have a choice, symmetric methods typically provide better accuracy for the same computational cost. The central difference method exploits this symmetry to achieve higher accuracy without additional function evaluations.

#### Higher-Order Methods: When You Need More Accuracy

:::{margin}
**Fourth-order accuracy**: A numerical method where error decreases as the fourth power of step size.
:::

For applications requiring higher precision, we can use more points to achieve **fourth-order accuracy**. Through careful manipulation of Taylor series, we can derive:

$$f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + O(h^4)$$

The coefficients $(-1, 8, -8, 1)/12$ are specifically chosen to eliminate the $O(h)$, $O(h^2)$, and $O(h^3)$ error terms.

### Beyond Fourth Order: Diminishing Returns

The pattern continues: sixth-order methods use 7 points, eighth-order use 9 points. However, for noisy data or non-smooth functions, higher-order methods can amplify errors rather than reduce them. Additionally, near boundaries or discontinuities, you may not have enough points for high-order methods. This is why second-order central difference remains the default choice in practice - it provides an excellent balance of accuracy, robustness, and simplicity.

:::{admonition} Connection to N-body (Project 2)
:class: tip
In your N-body simulation, you'll compute:
- Velocity: $\vec{v}_{n+1} = \vec{v}_n + \vec{a}_n \Delta t$ (forward difference)
- Position: $\vec{r}_{n+1} = \vec{r}_n + \vec{v}_n \Delta t$ (forward difference)

But wait - why not use central difference for better accuracy? Because we're marching forward in time! We can't use future values we haven't computed yet. This is why we need special integration methods (Submodule 3).
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

:::{warning}
**Common Student Mistakes:**
1. Using `h = epsilon` directly (too small - roundoff dominates!)
2. Forgetting to scale `h` with `x` for large values
3. Using same `h` for all variables regardless of their scales
4. Not checking if `f(x+h) == f(x)` due to underflow
:::

### The Fundamental Trade-off: Rigorous Derivation of Optimal $h$

We've established that choosing $h$ involves balancing truncation error (from approximating the derivative) and round-off error (from finite precision arithmetic). Let's now rigorously derive the optimal value of $h$ for each method.

#### Forward Difference: Optimal Step Size

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

#### Central Difference: Even Better Optimal Step Size

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

### Visualizing the Error Trade-off

**Why this matters**: The competition between truncation and round-off error creates a fundamental limit on numerical accuracy. This visualization shows why we can't just make $h$ arbitrarily small - there's an optimal value where total error is minimized. Understanding this trade-off is crucial for choosing appropriate step sizes in your simulations.

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

For large $h$, truncation error dominates. For small $h$, round-off error dominates. The optimal $h$ sits at the bottom of the U.

### Practical Algorithm for Choosing $h$

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

## Part 2: Numbers Aren't Real - Computer Arithmetic & Cosmic Consequences

### Why This Matters for Astronomy

Astronomy presents unique computational challenges because it spans vast scales:

- Stellar nuclear reactions: $10^{-13}$ cm
- Planetary orbits: 1 AU = $1.496 \times 10^{13}$ cm  
- Stellar systems: 1 pc = $3.086 \times 10^{18}$ cm
- Galactic scales: 10 kpc = $3.086 \times 10^{22}$ cm
- Observable universe: $10^{28}$ cm (approximately 10 Gpc)

Similarly for masses:
- Asteroids: $10^{18}$ g
- Earth: $5.972 \times 10^{27}$ g
- Sun: 1 $M_{\odot}$ = $1.989 \times 10^{33}$ g
- Galaxy: $10^{12}$ $M_{\odot}$ = $2 \times 10^{45}$ g

Double precision floating-point provides only ~16 decimal digits. We cannot simultaneously represent planetary positions to millimeter precision and galactic distances. We must carefully choose units and reference frames.

### Finding Machine Epsilon

Machine epsilon $\epsilon$ is the smallest number such that $1 + \epsilon \neq 1$ in floating-point arithmetic:

```python
import numpy as np

def find_machine_epsilon():
    """
    Find the smallest power of 2 such that 1 + eps != 1
    Note: This finds 2^(-52) for double precision
    """
    eps = 1.0
    while (1.0 + eps/2.0) != 1.0:
        eps = eps / 2.0
    return eps

# Test and compare with NumPy's value
eps_computed = find_machine_epsilon()
eps_numpy = np.finfo(float).eps
print(f"Computed ε: {eps_computed:.3e}")
print(f"NumPy's ε:  {eps_numpy:.3e}")
assert abs(eps_computed - eps_numpy) < 1e-20, "Mismatch in epsilon"
```

This value determines the fundamental limits of numerical accuracy.

### The Three Types of Error

#### 1. Round-off Error (from finite precision)

Round-off error occurs when numbers cannot be represented exactly. The most dangerous situation is catastrophic cancellation when subtracting nearly equal numbers.

**Why this matters**: In astrophysics, we often compute small differences between large quantities - like parallax angles, proper motions, or energy changes in orbits. Understanding how to reformulate these calculations preserves precision in your results.

```python
import numpy as np

def demonstrate_catastrophic_cancellation():
    """
    Show how subtracting nearly equal numbers destroys precision
    Example: Computing parallax from distance measurements
    """
    # Earth's orbit baseline: small change in distance to star
    d1_au = 1.0000000  # Distance in AU (winter)
    d2_au = 1.0000067  # Distance in AU (summer) - 1000 km difference
    
    def parallax_bad(d1, d2):
        """Naive: loses precision when d1 ≈ d2"""
        if d1 == 0 or d2 == 0:
            raise ValueError("Distance cannot be zero")
        return 1.0/d1 - 1.0/d2
    
    def parallax_good(d1, d2):
        """Reformulated: preserves precision"""
        if d1 == 0 or d2 == 0:
            raise ValueError("Distance cannot be zero")
        return (d2 - d1)/(d1 * d2)
    
    bad_result = parallax_bad(d1_au, d2_au)
    good_result = parallax_good(d1_au, d2_au)
    relative_error = abs(bad_result-good_result)/good_result*100
    
    print(f"Bad method:  {bad_result:.10e}")
    print(f"Good method: {good_result:.10e}")
    print(f"Relative difference: {relative_error:.2f}%")
    
    return bad_result, good_result, relative_error
```

#### 2. Truncation Error (from approximations)

Truncation error arises from approximating infinite processes with finite ones. For example, truncating the Taylor series for $e^x$:

$$e^x \approx 1 + x + \frac{x^2}{2!} + ... + \frac{x^n}{n!}$$

The error is all the neglected terms beyond $n$.

#### 3. Propagation Error (accumulation over iterations)

:::{margin}
**Propagation error**: The accumulation and amplification of errors through repeated calculations.
:::

**Propagation error** is how small errors compound through repeated calculations. This is particularly dangerous in astronomy where we often integrate orbits over millions of timesteps:

```python
import numpy as np

def demonstrate_error_propagation():
    """
    Show how tiny errors grow through repeated operations
    This models what happens in long-term orbital integration
    """
    value = 1.0
    tiny_error = 1e-15  # Below machine precision
    
    # Track error growth
    steps = [1, 10, 100, 1000, 10000, 100000, 1000000]
    results = []
    
    print("Error Propagation in Iterative Calculations:")
    print("Steps    | Total Error | Growth Factor")
    print("-" * 40)
    
    for n in steps:
        result = 1.0
        for _ in range(n):
            result *= (1 + tiny_error)
        
        actual_error = result - 1.0
        growth_factor = actual_error / tiny_error
        results.append((n, actual_error, growth_factor))
        
        print(f"{n:7d} | {actual_error:.3e} | {growth_factor:.1e}")
    
    return results
```

This is why long-term orbital integrations require special care.

:::{admonition} Check Your Understanding
:class: question
1. Why does $0.1 + 0.2 \neq 0.3$ in binary floating-point arithmetic?
2. How would you reformulate $(a^2 - b^2)/(a - b)$ when $a \approx b$?
3. If error grows by factor of 2 each step, how many steps until error > 1?
:::

---

## Part 3: Taylor Series - The Bridge from Continuous to Discrete

### The Foundation

The Taylor series connects continuous calculus to discrete numerical methods. For a smooth function $f(x)$:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \frac{f'''(x_0)}{3!}(x-x_0)^3 + ...$$

In numerical methods, we truncate this series, introducing truncation error. The art lies in managing this error within finite precision constraints.

### From Taylor Series to Finite Differences

Let's see how Taylor series creates each finite difference formula.

#### Deriving Forward Difference

Starting with Taylor expansion:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)$$

Rearranging:

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) + O(h^2)$$

The forward difference $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ has error $O(h)$.

#### Deriving Central Difference  

Using both directions:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{2h^3}{6}f'''(x) + O(h^5)$$

Therefore:

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$$

The symmetry cancels all even-order terms, giving us second-order accuracy "for free"!

### Verifying Error Predictions

**Why this matters**: Taylor series isn't just abstract mathematics - it accurately predicts the errors in our numerical methods. This verification shows that our theoretical analysis matches reality, giving us confidence in our error estimates. When you implement numerical methods in your research, you can use similar tests to verify your code is working correctly.

```python
import numpy as np

def verify_taylor_predictions():
    """
    Empirically verify that truncation errors match Taylor predictions
    This builds confidence that our theory accurately describes reality
    """
    # Test function: sin(x) - we know all its derivatives analytically
    f = np.sin
    x, h = 1.0, 0.01
    
    # Forward difference actual error
    fd_approx = (f(x+h) - f(x))/h
    fd_error = fd_approx - np.cos(x)  # cos is the true derivative
    
    # Taylor predicts: error ≈ -(h/2)*f''(x) = (h/2)*sin(x)
    fd_predicted = (h/2) * np.sin(x)
    
    # Central difference actual error  
    cd_approx = (f(x+h) - f(x-h))/(2*h)
    cd_error = cd_approx - np.cos(x)
    
    # Taylor predicts: error ≈ -(h²/6)*f'''(x) = (h²/6)*cos(x)
    cd_predicted = (h**2/6) * np.cos(x)
    
    results = {
        'forward': {'actual': fd_error, 'predicted': fd_predicted},
        'central': {'actual': cd_error, 'predicted': cd_predicted}
    }
    
    print("Taylor Series Predictions vs Reality:")
    for method, errors in results.items():
        ratio = errors['actual']/errors['predicted']
        print(f"{method.capitalize():8} - Actual: {errors['actual']:.6e}, "
              f"Predicted: {errors['predicted']:.6e}, Ratio: {ratio:.3f}")
    
    return results
```

### Why Central Difference Wins

Central difference is superior because:

1. **Higher accuracy**: $O(h^2)$ vs $O(h)$ error
2. **Larger optimal $h$**: $\epsilon^{1/3}$ vs $\epsilon^{1/2}$ 
3. **More robust**: Less susceptible to round-off
4. **Symmetric**: Cancels systematic biases

### When NOT to Use Numerical Derivatives

Before implementing numerical derivatives, consider if they're necessary. **Avoid numerical derivatives when**:

1. **Analytical derivatives are available** - Always use exact derivatives when you can derive them
2. **The function is noisy** - Numerical derivatives amplify noise
3. **You need many derivatives** - Consider automatic differentiation (see below)
4. **The function is expensive** - Each derivative needs multiple evaluations

### Modern Alternative: Automatic Differentiation

For complex functions, especially in machine learning, **automatic differentiation** provides exact derivatives (to machine precision) without the errors of finite differences. Tools like JAX (which you'll use in the final project) compute derivatives by tracking operations, not by approximating with finite differences. This gives the best of both worlds: exact derivatives without manual derivation. However, finite differences remain essential for:
- Verifying automatic differentiation implementations
- Functions only available as black boxes
- Understanding numerical behavior
- Quick derivative estimates

---

## Quick Reference Card

### Choosing Your Method

| Situation | Method | Optimal h | Error Order |
|-----------|--------|-----------|-------------|
| General derivative | Central | $\epsilon^{1/3} \approx 10^{-5}$ | $O(h^2)$ |
| Boundary point | Forward/Backward | $\sqrt{\epsilon} \approx 10^{-8}$ | $O(h)$ |
| Very smooth function | 4th-order central | $\epsilon^{1/5} \approx 10^{-3}$ | $O(h^4)$ |
| Noisy data | Larger h + smoothing | Problem-dependent | - |

### Error Types and Mitigation

| Error Type | Source | Mitigation |
|------------|--------|------------|
| Round-off | Finite precision | Reformulate expressions |
| Truncation | Finite approximation | Use higher-order methods |
| Propagation | Accumulation | Use stable algorithms |

---

## Summary and Key Takeaways

You've mastered the foundations of numerical differentiation and finite precision arithmetic.

### The Fundamental Trade-off
Computers cannot take true limits. We must balance:
- **Truncation error** (wants small $h$)
- **Round-off error** (wants large $h$)

### Optimal Step Sizes
- **Forward/Backward**: $h_{\text{opt}} \sim \sqrt{\epsilon} \approx 10^{-8}$
- **Central**: $h_{\text{opt}} \sim \epsilon^{1/3} \approx 10^{-5}$
- **Fourth-order**: $h_{\text{opt}} \sim \epsilon^{1/5} \approx 10^{-3}$

### Method Selection
- **Default**: Use central difference
- **Boundaries**: Use forward/backward
- **Smooth functions**: Consider fourth-order
- **Always**: Verify by varying $h$

### Connections to Projects
- **Project 2**: Velocity/acceleration updates
- **Project 3**: Round-off in photon tracking
- **Project 4**: Gradients for MCMC
- **Final**: Neural network training

:::{admonition} Final Self-Check
:class: question
Before moving on, ensure you can:
1. Derive the optimal $h$ for any finite difference method
2. Identify when catastrophic cancellation will occur
3. Choose appropriate methods for different scenarios
4. Predict error scaling before running code
:::

Remember: in computational astrophysics, we push numerical limits whether tracking billion-year orbits or modeling vast scale differences. These foundations help you recognize, diagnose, and mitigate numerical errors in your research.

---

## Bridge to Submodule 2: From Derivatives to Root Finding and Integration

You've now mastered how to compute derivatives numerically despite finite precision limitations. But derivatives are just one piece of the computational puzzle. Next, we'll explore two fundamental problems that build on these foundations:

### Root Finding: Where Things Balance
In astrophysics, we often need to find equilibrium points where forces balance or where functions equal zero:
- **Lagrange points**: Where gravitational and centrifugal forces cancel
- **Stellar structure**: Finding the radius where pressure balances gravity
- **Kepler's equation**: Solving $E - e\sin(E) = M$ for orbital position

The derivative knowledge you've gained will be crucial here - Newton's method for root finding uses $x_{n+1} = x_n - f(x_n)/f'(x_n)$, where you'll need to compute $f'(x_n)$ numerically using the methods you just learned.

### Numerical Integration: Measuring the Universe
Integration is the inverse of differentiation, and many astrophysical quantities require integration:
- **Luminosities**: $L = \int F_\lambda d\lambda$ (integrating spectra)
- **Masses**: $M = \int \rho dV$ (integrating density profiles)
- **Orbital periods**: $P = \int dt$ (integrating time along orbits)

Just as we approximated derivatives with finite differences, we'll approximate integrals with finite sums. The same error analysis principles apply - balancing truncation error against round-off error, choosing appropriate step sizes, and understanding when methods fail.

### The Deeper Connection
Both root finding and integration rely on the same fundamental principle you've learned: **we cannot achieve infinite precision on finite machines**. The art lies in managing errors intelligently. The optimal $h$ analysis you mastered for derivatives extends directly to choosing step sizes for integration and convergence criteria for root finding.

### What You'll Learn Next
In Submodule 2, you'll discover:
- How to find roots with guaranteed convergence (bisection) vs fast convergence (Newton-Raphson)
- When quadrature methods (Simpson's rule) beat Monte Carlo integration
- Why some problems are "stiff" and need special methods
- How the condition number determines problem difficulty

The finite difference methods you've learned are the building blocks. Now we'll construct more sophisticated algorithms for solving the static problems that appear throughout computational astrophysics.

*Next: Submodule 2 - Static Problems & Quadrature*