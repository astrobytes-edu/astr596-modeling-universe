---
title: "Part 1: Root Finding - Where Physics Reaches Equilibrium"
subtitle: "Module 2: Static Problems & Quadrature | ASTR 596"
---

**Navigation:**
[← Module Overview](./00-overview.md) | [Part 2: Quadrature →](./02-quadrature.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Implement** bisection, Newton-Raphson, and secant methods from scratch
- **Analyze** convergence rates (linear, superlinear, quadratic) mathematically
- **Diagnose** method failures and **design** robust hybrid approaches
- **Apply** root finding to Kepler's equation and Lagrange points
- **Predict** iteration counts and **verify** convergence behavior

---

## The Fundamental Problem

:::{margin}
**Root/Zero**: A value $x^*$ where $f(x^*) = 0$. The x-intercept of the function. In physics, often represents equilibrium points.
:::

:::{margin}
**Transcendental equation**: An equation involving transcendental functions (sin, cos, exp, log) that cannot be solved algebraically.
:::

In physics, equilibrium occurs where the net force vanishes, where energy is minimized, or where competing effects balance. Mathematically, these all reduce to finding **roots** of equations: solving $f(x) = 0$.

But here's the challenge: most astrophysical equations cannot be solved analytically. Consider finding where a star's pressure balances gravity:

$$\frac{dP}{dr} = -\frac{GM(r)\rho(r)}{r^2}$$

With realistic equations of state and composition gradients, this becomes a transcendental equation with no closed-form solution. We need numerical methods.

## Building Intuition: The Geometry of Root Finding

Before diving into algorithms, let's understand the geometry. Finding roots is about discovering where a curve crosses the x-axis. Different methods use different geometric insights:

:::{margin}
**Bracketing**: Having two points where the function has opposite signs, guaranteeing a root between them by the Intermediate Value Theorem.
:::

:::{margin}
**Interpolation**: Approximating a function between known points using simpler functions (lines, parabolas, etc.).
:::

1. **Bracketing methods** (Bisection): Trap the root between two points and squeeze
2. **Local approximation methods** (Newton): Follow the tangent line to the axis
3. **Interpolation methods** (Secant): Approximate the curve with simpler functions

Each approach has trade-offs between:
- **Reliability**: Will it always find a root?
- **Speed**: How many iterations to converge?
- **Requirements**: Do we need derivatives? Good initial guesses?

---

## Method 1: Bisection - The Reliable Workhorse

### The Mathematical Foundation

:::{margin}
**Intermediate Value Theorem**: If $f$ is continuous on $[a,b]$ and $k$ is any value between $f(a)$ and $f(b)$, then there exists at least one $c \in (a,b)$ where $f(c) = k$.
:::

The bisection method is based on the **Intermediate Value Theorem**: 

> If $f$ is continuous on $[a,b]$ and $f(a) \cdot f(b) < 0$, then there exists at least one root $r \in (a,b)$ where $f(r) = 0$.

The algorithm repeatedly halves the interval, keeping the half that contains the root.

### Visual Intuition

```
Initial bracket [a₀, b₀]:
f(a₀) < 0 ----+                    
              |     ×               
              |   ×   ×             
--------------+×-------×------------ x-axis
            × |         ×           
          ×   |          +---- f(b₀) > 0
              ↑
           root somewhere here

After bisection:
        [a₁, b₁]
f(a₁) < 0 --+          
            | ×        
------------+×--×------ x-axis
            |    +-- f(b₁) > 0
            ↑
         narrowed bracket
```

### Algorithm Analysis

Starting with interval $[a_0, b_0]$ where $f(a_0) \cdot f(b_0) < 0$:

1. Compute midpoint: $c_n = \frac{a_n + b_n}{2}$
2. Evaluate $f(c_n)$
3. Update interval:
   - If $f(a_n) \cdot f(c_n) < 0$: root in $[a_n, c_n]$, so set $b_{n+1} = c_n$, $a_{n+1} = a_n$
   - Otherwise: root in $[c_n, b_n]$, so set $a_{n+1} = c_n$, $b_{n+1} = b_n$

After $n$ iterations, the interval width is:

$$|b_n - a_n| = \frac{|b_0 - a_0|}{2^n}$$

:::{margin}
**Linear convergence**: Error decreases by a constant factor each iteration.
:::

The error in our approximation is bounded by half the interval width:

$$|c_n - r| \leq \frac{|b_n - a_n|}{2} = \frac{|b_0 - a_0|}{2^{n+1}}$$

This is **linear convergence** with rate $\frac{1}{2}$.

### Iterations Required

To achieve error $< \epsilon$, we need:

$$\frac{|b_0 - a_0|}{2^{n+1}} < \epsilon$$

Solving for $n$:

$$n > \log_2\left(\frac{|b_0 - a_0|}{\epsilon}\right) - 1$$

For example, to find a root in $[0, 1]$ to 10 decimal places ($\epsilon = 10^{-10}$):

$$n > \log_2(10^{10}) - 1 \approx 33.2 - 1 = 32.2$$

So we need 33 iterations - slow but guaranteed!

### Pseudocode

```
FUNCTION Bisection(f, a, b, tolerance):
    IF f(a) * f(b) ≥ 0:
        ERROR "Need opposite signs at endpoints"
    
    WHILE (b - a)/2 > tolerance:
        c = (a + b) / 2
        
        IF f(c) = 0:
            RETURN c  // Exact root found
        
        IF f(a) * f(c) < 0:
            b = c  // Root in left half
        ELSE:
            a = c  // Root in right half
    
    RETURN (a + b) / 2
```

:::{admonition} Check Your Understanding
:class: question
1. Why does bisection require $f(a) \cdot f(b) < 0$?
2. What happens if there are multiple roots in $[a,b]$?
3. Can bisection find roots where $f$ touches but doesn't cross zero?
4. How would you modify bisection to find all roots in an interval?
:::

---

## Method 2: Newton-Raphson - The Speed Demon

### The Geometric Insight

:::{margin}
**Tangent line approximation**: Replacing a curve with its tangent line at a point, valid for small deviations.
:::

Newton's method uses calculus to accelerate convergence. The key insight: near any point, a smooth function looks approximately linear. We can follow the **tangent line** to where it crosses the axis, getting much closer to the root in a single step.

### Mathematical Derivation

Starting at point $x_n$, the tangent line to $f(x)$ has equation:

$$y - f(x_n) = f'(x_n)(x - x_n)$$

This line crosses the x-axis (where $y = 0$) at:

$$0 - f(x_n) = f'(x_n)(x_{n+1} - x_n)$$

Solving for $x_{n+1}$:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

This is the Newton-Raphson iteration formula.

### Convergence Analysis

To understand the convergence rate, we analyze the error $e_n = x_n - r$ where $r$ is the true root.

Using Taylor series around $r$:

$$f(x_n) = f(r) + f'(r)(x_n - r) + \frac{f''(r)}{2}(x_n - r)^2 + O((x_n - r)^3)$$

Since $f(r) = 0$:

$$f(x_n) = f'(r)e_n + \frac{f''(r)}{2}e_n^2 + O(e_n^3)$$

Similarly:

$$f'(x_n) = f'(r) + f''(r)e_n + O(e_n^2)$$

Substituting into the Newton iteration:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} = x_n - \frac{f'(r)e_n + \frac{f''(r)}{2}e_n^2}{f'(r) + f''(r)e_n}$$

Factoring out $f'(r)$ from numerator and denominator:

$$x_{n+1} = x_n - e_n \cdot \frac{1 + \frac{f''(r)}{2f'(r)}e_n}{1 + \frac{f''(r)}{f'(r)}e_n}$$

Using the approximation $\frac{1}{1+x} \approx 1 - x$ for small $x$:

$$x_{n+1} = x_n - e_n\left(1 + \frac{f''(r)}{2f'(r)}e_n\right)\left(1 - \frac{f''(r)}{f'(r)}e_n\right)$$

Expanding and keeping terms up to $O(e_n^2)$:

$$x_{n+1} = x_n - e_n + \frac{f''(r)}{2f'(r)}e_n^2$$

Therefore:

$$e_{n+1} = x_{n+1} - r = \frac{f''(r)}{2f'(r)}e_n^2 + O(e_n^3)$$

:::{margin}
**Quadratic convergence**: Error is squared each iteration, doubling the number of correct digits.
:::

This shows **quadratic convergence**! The error is squared each iteration, meaning the number of correct digits approximately doubles.

### When Newton Fails

Despite its speed, Newton's method can fail spectacularly:

1. **Zero derivative**: If $f'(x_n) = 0$, the tangent is horizontal and never crosses the axis
2. **Poor initial guess**: May diverge or converge to wrong root
3. **Cycles**: Can oscillate between points without converging
4. **Non-smooth functions**: Requires continuous derivative

### Visual of Newton's Method

```
Tangent line approximation:
       
f(xₙ) ----×     
         /|     True curve f(x)
        / |   ×
       /  | ×
------/---|-------×------- x-axis
     /    |         ×
    /     |          ×
   /      |           ×
  /       xₙ      xₙ₊₁
 /                 ↑
Tangent line    Next iterate
                (where tangent hits axis)
```

### Example of Newton's Method Cycling

For $f(x) = x^3 - 2x + 2$, starting near $x_0 = 0$:
- $x_1 \approx 1$
- $x_2 \approx 0$ 
- $x_3 \approx 1$ (cycling begins!)

This happens because the function has a special symmetry that causes the tangent lines to create a cycle.

### Pseudocode

```
FUNCTION NewtonRaphson(f, f_prime, x0, tolerance, max_iterations):
    x = x0
    
    FOR iteration = 1 TO max_iterations:
        fx = f(x)
        
        IF |fx| < tolerance:
            RETURN x  // Converged
        
        fpx = f_prime(x)
        
        IF |fpx| < machine_epsilon:
            ERROR "Derivative too small"
        
        x_new = x - fx / fpx
        
        IF |x_new - x| < tolerance:
            RETURN x_new  // Converged
        
        x = x_new
    
    ERROR "Failed to converge"
```

:::{admonition} Physical Example: Finding Lagrange Points
:class: tip
In the circular restricted three-body problem, the effective potential is:
$$\Phi_{eff} = -\frac{GM_1}{r_1} - \frac{GM_2}{r_2} - \frac{1}{2}\omega^2(x^2 + y^2)$$

Lagrange points occur where $\nabla\Phi_{eff} = 0$. Newton's method excels here because we can analytically compute the gradient and Hessian!
:::

---

## Method 3: Secant - The Practical Compromise

### Motivation: No Derivatives Required

:::{margin}
**Finite difference approximation**: Estimating derivatives using function values at nearby points.
:::

Newton's method requires $f'(x)$, but what if:
- The derivative is expensive to compute?
- We only have $f(x)$ as a black box?
- The function comes from experimental data?

The secant method approximates the derivative using a **finite difference**:

$$f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}$$

### The Iteration Formula

Substituting this approximation into Newton's formula:

$$x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

Geometrically, we're replacing the tangent line with a secant line through two points.

### Visual of Secant Method

```
Secant Line Approximation:

f(x₁) ----×     
         /|     
        / |   × True curve
       /  | ×
------/---|-------×------- x-axis
     /    |         ×
    /     x₁    x₂   ×
   /              \   
  /                Next iterate
 /                 (secant crosses axis)
×----f(x₀)
```

### Convergence Analysis

The convergence of the secant method is more complex to analyze. Through detailed analysis (involving difference equations), one can show that the order of convergence is:

$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$$

This is the golden ratio! The error satisfies:

$$|e_{n+1}| \approx C|e_n|^{\phi}$$

:::{margin}
**Superlinear convergence**: Faster than linear but slower than quadratic.
:::

This **superlinear convergence** is slower than Newton's quadratic rate but faster than bisection's linear rate.

### Comparison of Convergence Rates

| Method | Order | Error Reduction | Digits Gained/Iteration |
|--------|-------|----------------|------------------------|
| Bisection | 1.0 | $e_{n+1} = \frac{1}{2}e_n$ | 0.30 (log₁₀ 2) |
| Secant | 1.618 | $e_{n+1} \approx Ce_n^{1.618}$ | ~0.62 (when e < 1) |
| Newton | 2.0 | $e_{n+1} \approx Ce_n^2$ | Doubles each iteration |

### Pseudocode

```
FUNCTION Secant(f, x0, x1, tolerance, max_iterations):
    f0 = f(x0)
    f1 = f(x1)
    
    FOR iteration = 1 TO max_iterations:
        IF |f1| < tolerance:
            RETURN x1  // Converged (function value small)
        
        IF |f1 - f0| < machine_epsilon:
            ERROR "Function values too close"
        
        // Secant line through (x0,f0) and (x1,f1)
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        // Check convergence based on x change
        IF |x_new - x1| < tolerance:
            RETURN x_new  // Converged (x values close)
        
        // Update for next iteration
        x0, f0 = x1, f1
        x1 = x_new
        f1 = f(x1)
    
    ERROR "Failed to converge"
```

:::{admonition} Check Your Understanding
:class: question
1. Why is secant's convergence order the golden ratio?
2. When would you choose secant over Newton?
3. What happens if f(x₀) = f(x₁)?
4. How many function evaluations per iteration for each method?
:::

---

## Hybrid Methods: Best of All Worlds

In practice, we often combine methods to leverage their strengths:

### Brent's Method

Combines bisection's reliability with inverse quadratic interpolation's speed:
- Maintains a bracketing interval like bisection
- Tries fast interpolation when possible
- Falls back to bisection if interpolation fails

### Practical Strategy

```
1. Start with bisection to bracket the root
2. Switch to Newton/Secant once close enough
3. Fall back to bisection if Newton diverges
```

---

## Debugging Root Finding

Common root-finding bugs and their diagnosis:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Bisection won't start | Same sign at endpoints | Scan for sign changes first |
| Newton diverges | Poor initial guess or $f'(x) \approx 0$ | Use bisection first to bracket |
| Secant oscillates | Nearly parallel secant lines | Check $\|f(x_1) - f(x_0)\|$ threshold |
| Wrong root found | Multiple roots in region | Narrow search interval |
| Infinite loop | Tolerance too small | Use $\epsilon_{machine} \times \|x\|$ |
| No convergence | Discontinuous function | Check function continuity |

### Practical Initial Bracketing

To find initial brackets for bisection:

```
FUNCTION FindBrackets(f, x_min, x_max, n_scan):
    dx = (x_max - x_min) / n_scan
    brackets = []
    
    FOR i = 0 TO n_scan-1:
        x1 = x_min + i * dx
        x2 = x_min + (i+1) * dx
        
        IF f(x1) * f(x2) < 0:
            brackets.append([x1, x2])
    
    RETURN brackets
```

## Worked Example: Kepler's Equation

The most famous root-finding problem in astronomy is Kepler's equation, which relates mean anomaly $M$ (proportional to time) to eccentric anomaly $E$ (related to position):

$$E - e\sin(E) = M$$

where $e$ is the orbital eccentricity. Given $M$ and $e$, solve for $E$.

### Why This Is Hard

- Transcendental equation (no algebraic solution)
- Must be solved millions of times in orbit propagation
- High eccentricity makes convergence difficult

### Newton's Method Applied

Define $f(E) = E - e\sin(E) - M$

Then $f'(E) = 1 - e\cos(E)$

Newton iteration:
$$E_{n+1} = E_n - \frac{E_n - e\sin(E_n) - M}{1 - e\cos(E_n)}$$

### Initial Guess Strategy

Good initial guesses are crucial:
- For $e < 0.8$: Use $E_0 = M$
- For $e \geq 0.8$: Use $E_0 = \pi$

### Convergence Analysis

The number of iterations depends strongly on eccentricity:

| Eccentricity | Iterations (typical) | Why |
|--------------|---------------------|-----|
| 0.0 (circle) | 0 | E = M exactly |
| 0.1 | 2-3 | Nearly linear problem |
| 0.5 | 3-4 | Moderate nonlinearity |
| 0.9 | 5-7 | Strong nonlinearity |
| 0.99 | 10-15 | Near-parabolic orbit |

:::{admonition} Connection to Project 2 (N-body)
:class: tip
In your N-body simulation, you might need Kepler's equation to:
- Convert orbital elements to positions
- Set up initial conditions for binary systems
- Verify conservation of orbital elements
:::

---

## Worked Example: Photon Sphere Around a Black Hole

For a Schwarzschild black hole, photons can orbit at radius $r$ where the effective potential has a maximum. This occurs where:

$$\frac{d}{dr}\left[\left(1 - \frac{2GM}{rc^2}\right)\left(1 + \frac{L^2}{r^2}\right)\right] = 0$$

Simplifying for circular orbits gives:
$$r^3 - 3GMr^2/c^2 = 0$$

The non-trivial solution is the photon sphere radius:
$$r_{photon} = \frac{3GM}{c^2} = 1.5 r_s$$

where $r_s = 2GM/c^2$ is the Schwarzschild radius.

This is where light can orbit the black hole in unstable circular paths!

---

## Bridge to Part 2: From Finding Zeros to Measuring Areas

You now command three powerful methods for finding where functions equal zero. Each has its place: bisection for reliability, Newton for speed, secant for practicality.

But finding equilibrium points is only half the story. Next, we tackle integration—measuring quantities across space and time. You'll discover how the same principles of approximation and error analysis apply to computing areas under curves, volumes of regions, and integrals over high-dimensional spaces.

The connection runs deep: integration is the inverse of differentiation, and many integration methods rely on root finding. The error analysis skills you've developed here will guide your understanding of quadrature methods.

*Next: Part 2 - Quadrature*