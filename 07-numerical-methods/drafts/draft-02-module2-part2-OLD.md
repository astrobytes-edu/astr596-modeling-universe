---
title: "Part 2: Quadrature - From Photon Counts to Dark Matter Halos"
subtitle: "Module 2: Static Problems & Quadrature | ASTR 596"
---

**Navigation:**
[← Part 1: Root Finding](./01-root-finding.md) | [Part 3: Synthesis →](./03-synthesis.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Derive** rectangular, trapezoidal, and Simpson's rules from Taylor series
- **Implement** Gaussian quadrature for optimal point placement
- **Apply** Monte Carlo integration for high-dimensional problems
- **Choose** appropriate methods based on smoothness, dimension, and noise
- **Analyze** error scaling and **predict** convergence behavior

---

## The Fundamental Challenge

While differentiation is a local operation (needing only nearby points), integration is global - we must consider the entire domain. **Quadrature** transforms the continuous integral:

$$I = \int_a^b f(x) dx$$

into a discrete sum:

$$I \approx \sum_{i=0}^n w_i f(x_i)$$

The art lies in choosing:
- The points $x_i$ (where to sample)
- The weights $w_i$ (how much each sample contributes)
- The number $n$ (balancing accuracy vs computation)

## Physical Motivation: Why Astronomers Integrate

Integration is everywhere in astronomy because we observe integrated quantities:

1. **Spectroscopy**: We don't see individual photons but integrated flux
   $$F = \int_{\lambda_1}^{\lambda_2} F_\lambda d\lambda$$

2. **Photometry**: CCD pixels integrate photons over exposure time
   $$N_{photons} = \int_0^T \Phi(t) dt$$

3. **Structure**: Mass distributions require volume integrals
   $$M(<r) = \int_0^r 4\pi r'^2 \rho(r') dr'$$

4. **Cosmology**: Distances involve integrating through expanding space
   $$d_L = (1+z) \int_0^z \frac{c\,dz'}{H(z')}$$

Each context demands different accuracy, and the function properties (smooth vs oscillatory, bounded vs singular) determine the best method.

---

## Building Integration Methods: From Rectangles to Optimality

### The Riemann Sum Foundation

The definition of the Riemann integral suggests the simplest approximation:

$$\int_a^b f(x)dx = \lim_{n \to \infty} \sum_{i=0}^{n-1} f(x_i^*) \Delta x_i$$

where $x_i^* \in [x_i, x_{i+1}]$ and $\Delta x_i = x_{i+1} - x_i$.

Different choices of $x_i^*$ give different methods:
- Left endpoint: Rectangle rule (left)
- Right endpoint: Rectangle rule (right)
- Midpoint: Midpoint rule
- Average of endpoints: Trapezoidal rule

---

## Method 1: Rectangle Rule - The Starting Point

### Mathematical Formulation

For uniform spacing $h = (b-a)/n$:

$$I \approx h \sum_{i=0}^{n-1} f(a + ih) \quad \text{(left rectangle)}$$

### Error Analysis via Taylor Series

On each subinterval $[x_i, x_{i+1}]$:

$$\int_{x_i}^{x_{i+1}} f(x)dx = \int_{x_i}^{x_{i+1}} \left[f(x_i) + f'(x_i)(x-x_i) + \frac{f''(\xi)}{2}(x-x_i)^2\right]dx$$

Evaluating:
$$= f(x_i)h + f'(x_i)\frac{h^2}{2} + O(h^3)$$

The rectangle rule uses only $f(x_i)h$, so the local error is:

$$E_{local} = \frac{h^2}{2}f'(\xi)$$

Over $n$ intervals:

$$E_{total} = n \cdot O(h^2) = \frac{b-a}{h} \cdot O(h^2) = O(h)$$

:::{margin}
**First-order method**: Global error proportional to step size.
:::

This is a **first-order method** - halving $h$ halves the error.

### Why Not to Use Rectangle Rule

- Only first-order accurate
- Systematically over/underestimates for monotonic functions
- Poor for oscillatory functions
- Mainly pedagogical value

---

## Method 2: Trapezoidal Rule - The Workhorse

### Geometric Intuition

Instead of rectangles, connect consecutive points with straight lines, creating trapezoids.

### Mathematical Formulation

The area of a trapezoid with parallel sides $f(x_i)$ and $f(x_{i+1})$ and height $h$ is:

$$A_i = \frac{h}{2}[f(x_i) + f(x_{i+1})]$$

Summing over all intervals:

$$I \approx \sum_{i=0}^{n-1} \frac{h}{2}[f(x_i) + f(x_{i+1})] = h\left[\frac{f(a)}{2} + \sum_{i=1}^{n-1} f(x_i) + \frac{f(b)}{2}\right]$$

### Error Analysis

Using Taylor expansion around the midpoint $x_i + h/2$:

$$f(x_i) = f(x_i + h/2) - \frac{h}{2}f'(x_i + h/2) + \frac{h^2}{8}f''(x_i + h/2) + O(h^3)$$

$$f(x_{i+1}) = f(x_i + h/2) + \frac{h}{2}f'(x_i + h/2) + \frac{h^2}{8}f''(x_i + h/2) + O(h^3)$$

The trapezoidal approximation on $[x_i, x_{i+1}]$:

$$\frac{h}{2}[f(x_i) + f(x_{i+1})] = hf(x_i + h/2) + \frac{h^3}{12}f''(x_i + h/2) + O(h^4)$$

The exact integral:

$$\int_{x_i}^{x_{i+1}} f(x)dx = hf(x_i + h/2) + O(h^3)$$

Local error: $E_{local} = -\frac{h^3}{12}f''(\xi)$

:::{margin}
**Second-order method**: Global error proportional to step size squared.
:::

Global error: $E_{total} = -\frac{(b-a)h^2}{12}f''(\xi)$ for some $\xi \in [a,b]$

This is a **second-order method**!

### Pseudocode

```
FUNCTION Trapezoid(f, a, b, n):
    h = (b - a) / n
    sum = (f(a) + f(b)) / 2  // Endpoints weighted by 1/2
    
    FOR i = 1 TO n-1:
        x = a + i * h
        sum = sum + f(x)  // Interior points weighted by 1
    
    RETURN h * sum
```

---

## Method 3: Simpson's Rule - Parabolic Perfection

### The Key Insight

Instead of linear interpolation (trapezoids), use quadratic interpolation (parabolas) through consecutive triplets of points.

### Derivation via Interpolation

For three equally-spaced points $x_0, x_1, x_2$ with $x_1 = x_0 + h$ and $x_2 = x_0 + 2h$, the unique parabola through $(x_i, f(x_i))$ is:

$$p(x) = f(x_0)\frac{(x-x_1)(x-x_2)}{2h^2} - f(x_1)\frac{(x-x_0)(x-x_2)}{h^2} + f(x_2)\frac{(x-x_0)(x-x_1)}{2h^2}$$

Integrating this parabola from $x_0$ to $x_2$:

$$\int_{x_0}^{x_2} p(x)dx = \frac{h}{3}[f(x_0) + 4f(x_1) + f(x_2)]$$

These are the famous Simpson weights: 1, 4, 1.

### Composite Simpson's Rule

For $n$ intervals (must be even):

$$I \approx \frac{h}{3}\left[f(a) + 4\sum_{i=1,3,5...}^{n-1} f(x_i) + 2\sum_{i=2,4,6...}^{n-2} f(x_i) + f(b)\right]$$

### Error Analysis

Through careful Taylor series analysis, the local error for Simpson's rule on interval $[x_i, x_{i+2}]$ is:

$$E_{local} = -\frac{h^5}{90}f^{(4)}(\xi)$$

:::{margin}
**Fourth-order method**: Error decreases as $h^4$ - remarkably accurate for smooth functions!
:::

Global error: $E_{total} = -\frac{(b-a)h^4}{180}f^{(4)}(\xi)$

Simpson's rule is **fourth-order accurate** despite using only parabolas! This "superconvergence" occurs because the method is exact for cubic polynomials. Since the error term involves $f^{(4)}$, and the fourth derivative of any cubic is zero, Simpson's rule integrates cubics exactly.

### When Simpson's Rule Shines

- Smooth functions with continuous fourth derivative
- Periodic functions
- When high accuracy is needed with moderate $n$

### Pseudocode

```
FUNCTION Simpson(f, a, b, n):
    IF n is odd:
        n = n + 1  // Must have even number of intervals
    
    h = (b - a) / n
    sum = f(a) + f(b)  // Endpoints: weight 1
    
    // Clear weight assignment:
    // Odd indices (1,3,5,...): weight 4
    // Even indices (2,4,6,...): weight 2
    FOR i = 1 TO n-1:
        x = a + i * h
        IF i is odd:
            sum = sum + 4 * f(x)  // Odd index: weight 4
        ELSE:
            sum = sum + 2 * f(x)  // Even index: weight 2
    
    RETURN (h / 3) * sum
```

### Visual Comparison: Trapezoids vs Parabolas

```
Trapezoidal Rule:          Simpson's Rule:
                          
    Linear                     Parabolic
   /\    /\                   ╱╲    ╱╲
  /  \  /  \                 ╱  ╲  ╱  ╲
 /    \/    \               ╱    ╲╱    ╲
+------+------+            +------+------+
x₀     x₁     x₂          x₀     x₁     x₂

Connects points with       Fits parabola through
straight lines             three points
```

:::{admonition} Check Your Understanding
:class: question
1. Why must n be even for Simpson's rule?
2. What happens if f(x) is exactly a cubic polynomial?
3. How does the error scale if we double n?
4. When would Simpson's rule perform poorly?
:::

---

## Method 4: Gaussian Quadrature - Optimal Point Placement

### The Revolutionary Idea

All previous methods used equally-spaced points. But what if we could choose both the points AND weights optimally?

**Gauss's insight**: For $n$ points, we can make the method exact for all polynomials up to degree $2n-1$ by choosing the right locations!

### Mathematical Foundation

We want to find points $x_i$ and weights $w_i$ such that:

$$\int_{-1}^{1} p(x)dx = \sum_{i=1}^n w_i p(x_i)$$

is exact for all polynomials $p(x)$ of degree $\leq 2n-1$.

:::{margin}
**Legendre polynomials**: A sequence of orthogonal polynomials on $[-1,1]$ with important properties for numerical integration.
:::

The optimal points are the roots of the $n$-th Legendre polynomial $P_n(x)$. This works because Legendre polynomials are orthogonal, meaning:

$$\int_{-1}^{1} P_n(x) P_m(x) dx = 0 \text{ for } n \neq m$$

This orthogonality property ensures that the Gauss points optimally sample the function.

### Example: 2-Point Gaussian Quadrature

For $n=2$, we need exactness for polynomials up to degree 3.

Setting up equations for monomials $1, x, x^2, x^3$:
- $\int_{-1}^{1} 1\,dx = 2 = w_1 + w_2$
- $\int_{-1}^{1} x\,dx = 0 = w_1x_1 + w_2x_2$
- $\int_{-1}^{1} x^2\,dx = \frac{2}{3} = w_1x_1^2 + w_2x_2^2$
- $\int_{-1}^{1} x^3\,dx = 0 = w_1x_1^3 + w_2x_2^3$

Solution: $x_1 = -\frac{1}{\sqrt{3}}$, $x_2 = \frac{1}{\sqrt{3}}$, $w_1 = w_2 = 1$

### Transformation to General Intervals

For interval $[a,b]$, transform from $[-1,1]$:

$$x = \frac{b-a}{2}t + \frac{a+b}{2}$$

$$\int_a^b f(x)dx = \frac{b-a}{2}\int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right)dt$$

### Common Gaussian Quadrature Points and Weights

| n | Gauss Points | Weights | Exact for degree |
|---|-------------|---------|------------------|
| 2 | $\pm 0.57735$ | 1.00000, 1.00000 | 3 |
| 3 | $0.00000, \pm 0.77460$ | 0.88889, 0.55556, 0.55556 | 5 |
| 4 | $\pm 0.33998, \pm 0.86114$ | 0.65215, 0.65215, 0.34785, 0.34785 | 7 |

### Why Gaussian Quadrature is Magical

- Optimal accuracy for given number of function evaluations
- 3 Gauss points often beats 100 equally-spaced points!
- Exact for polynomials up to degree $2n-1$
- Foundation for spectral methods in advanced simulations

---

## Method 5: Monte Carlo Integration - When Dimensions Explode

### The Curse of Dimensionality

:::{margin}
**Curse of dimensionality**: Exponential growth of computational cost with dimension for grid-based methods.
:::

For a $d$-dimensional integral with $n$ points per dimension, deterministic methods need $n^d$ evaluations. This becomes impossible for $d > 5$.

### The Monte Carlo Solution

Randomly sample $N$ points $\vec{x}_i$ in the domain $\Omega$:

$$I = \int_\Omega f(\vec{x})d\vec{x} \approx \frac{V(\Omega)}{N}\sum_{i=1}^N f(\vec{x}_i)$$

where $V(\Omega)$ is the volume of the domain.

### Error Analysis

:::{margin}
**Central Limit Theorem**: The sum of many independent random variables tends toward a normal distribution, regardless of the original distribution.
:::

By the Central Limit Theorem, the error in Monte Carlo integration is approximately normal with standard deviation:

$$\sigma_I = \frac{V(\Omega)\sigma_f}{\sqrt{N}}$$

where $\sigma_f$ is the standard deviation of $f$ over the domain.

:::{margin}
**Dimension-independent convergence**: Monte Carlo error scales as $N^{-1/2}$ regardless of dimension!
:::

**Key insight**: Error $\propto N^{-1/2}$ independent of dimension $d$!

This assumes:
- Independent random samples
- Finite variance of the integrand
- Sufficient samples for CLT to apply (typically N > 30)

### When Monte Carlo Wins

| Dimension | Grid Points for 1% Error | Monte Carlo Points for 1% Error |
|-----------|-------------------------|--------------------------------|
| 1 | 100 | 10,000 |
| 2 | 10,000 | 10,000 |
| 3 | 1,000,000 | 10,000 |
| 5 | 10^10 | 10,000 |
| 10 | 10^20 | 10,000 |

Above ~4 dimensions, Monte Carlo becomes superior!

### Monte Carlo Visual Representation

```
Monte Carlo Integration in 2D:
┌────────────────────────┐
│ • □ • □ □ • □ • □ □ • │  • = point inside region
│ □ • □ • • □ • □ • □ □ │  □ = point outside region
│ • □ • □ □ • □ • □ • • │
│ □ □ • • • • • □ • □ □ │  Area ≈ (hits/total) × box_area
│ • • □ □ • • • • • □ • │
│ □ • • • □ □ • • • • □ │  Error ∝ 1/√N regardless
│ • □ □ • • • • □ □ • • │  of dimension!
│ □ • • □ □ • □ • • □ □ │
└────────────────────────┘
Random sampling pattern
```

### Pseudocode

```
FUNCTION MonteCarlo(f, bounds, N):
    sum = 0
    sum_sq = 0
    
    FOR i = 1 TO N:
        // Random point in d-dimensional box
        x = RandomPoint(bounds)
        fx = f(x)
        sum = sum + fx
        sum_sq = sum_sq + fx^2
    
    // Volume of integration domain
    volume = Product(upper_i - lower_i for all dimensions)
    
    // Estimate integral
    integral = volume * sum / N
    
    // Estimate error (standard error of the mean)
    mean = sum / N
    mean_sq = sum_sq / N
    variance = mean_sq - mean^2  // Variance of f(x)
    std_error = volume * sqrt(variance / N)  // Standard error of integral
    
    RETURN integral, std_error
```

:::{admonition} Connection to Project 3 (MCRT)
:class: tip
Monte Carlo Radiative Transfer uses this principle:
- Photon paths are high-dimensional integrals
- Each photon is a Monte Carlo sample
- Error decreases as $1/\sqrt{N_{photons}}$
- Handles complex geometries naturally
:::

---

## Adaptive Methods and Advanced Topics

### Adaptive Quadrature

When functions vary rapidly in some regions but are smooth elsewhere, adaptive methods adjust the sampling density:

```
FUNCTION AdaptiveQuadrature(f, a, b, tolerance):
    // Compute integral with n and 2n points
    I1 = Simpson(f, a, b, n)
    I2 = Simpson(f, a, b, 2*n)
    
    IF |I2 - I1| < tolerance:
        RETURN I2  // Accurate enough
    ELSE:
        // Subdivide and recurse
        mid = (a + b) / 2
        left = AdaptiveQuadrature(f, a, mid, tolerance/2)
        right = AdaptiveQuadrature(f, mid, b, tolerance/2)
        RETURN left + right
```

### Richardson Extrapolation and Romberg Integration

:::{margin}
**Richardson Extrapolation**: A technique to improve numerical approximations by combining results from different step sizes to cancel leading error terms.
:::

Improve accuracy by combining results from different step sizes:

If $I(h) = I_{exact} + Ch^p + O(h^{p+1})$, then:
$$I_{better} = \frac{2^p I(h/2) - I(h)}{2^p - 1}$$

This cancels the leading error term!

**Romberg Integration** extends this idea systematically:
- Start with trapezoidal rule at different step sizes
- Apply Richardson extrapolation repeatedly
- Achieves high-order accuracy from simple method

### Quasi-Monte Carlo

For better random sampling than pure Monte Carlo:
- **Sobol sequences**: Low-discrepancy sequences that fill space more uniformly
- **Halton sequences**: Simple quasi-random sequences
- **Latin hypercube sampling**: Ensures all regions are sampled

These methods achieve error $O((\log N)^d/N)$ vs Monte Carlo's $O(1/\sqrt{N})$.

### Handling Singularities

For integrands with singularities:
1. **Removable**: Transform variables to eliminate singularity
2. **Integrable**: Use specialized quadrature (e.g., Gauss-Chebyshev for $1/\sqrt{x}$)
3. **Non-integrable**: Regularize or use principal value

## Real Application: Galaxy Luminosity from Spectrum

Astronomers measure galaxy spectra as discrete samples at wavelengths $\lambda_i$ with fluxes $F_i$. The total luminosity requires integration:

$$L = 4\pi d^2 \int_{\lambda_{min}}^{\lambda_{max}} F_\lambda d\lambda$$

### Challenges

1. **Irregular wavelength spacing** (different instruments, resolutions)
2. **Noise** in flux measurements
3. **Missing data** (atmospheric absorption, detector gaps)
4. **Emission lines** (narrow features needing resolution)

### Solution Strategy

```
IF wavelength spacing regular:
    IF smooth continuum:
        USE Simpson's rule
    ELSE:
        USE Trapezoidal rule (robust to noise)
ELSE (irregular spacing):
    USE Trapezoidal rule with variable h:
    L = Σ (λ[i+1] - λ[i]) * (F[i+1] + F[i]) / 2
```

For emission lines, ensure sufficient sampling:
- Need ~10 points across line width
- Or fit Gaussian profile and integrate analytically

---

## Choosing the Right Integration Method

### Decision Tree

```
Is dimension > 4?
    YES → Monte Carlo
    NO → Continue
    
Is function smooth (C^4 continuous)?
    YES → Simpson's Rule or Gaussian Quadrature
    NO → Continue
    
Do you control sample points?
    YES → Gaussian Quadrature
    NO → Continue
    
Is data noisy or experimental?
    YES → Trapezoidal Rule (robust)
    NO → Simpson's Rule (if smooth enough)
```

### Method Comparison Summary

| Method | Order | Pros | Cons | Best For |
|--------|-------|------|------|----------|
| Rectangle | $O(h)$ | Simple | Inaccurate | Teaching only |
| Trapezoid | $O(h^2)$ | Robust, simple | Moderate accuracy | Experimental data |
| Simpson | $O(h^4)$ | Very accurate | Needs smooth f, even n | Smooth functions |
| Gaussian | $O(h^{2n})$ | Optimal accuracy | Complex setup | When you choose points |
| Monte Carlo | $O(N^{-1/2})$ | Dimension-independent | Slow convergence | High dimensions |

---

## Debugging Integration

Common integration bugs and their diagnosis:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Error doesn't decrease with n | Wrong implementation | Check weights, indices |
| Error plateaus at ~10^-14 | Round-off dominance | Use fewer points or higher precision |
| Oscillating convergence | Under-resolving oscillations | Increase n or use adaptive methods |
| Simpson gives odd results | Odd n | Ensure even intervals |
| Negative result for positive f | Overflow in sum | Use Kahan summation |

---

## Bridge to Part 3: The Unity of Numerical Methods

You've now mastered two fundamental classes of problems: finding where functions equal zero and measuring areas under curves. These aren't isolated techniques—they're connected by deep mathematical principles.

In Part 3, we'll explore these connections. You'll see how root finding and integration are inverse operations, how the same error analysis principles govern both, and how to combine methods for complex problems. Most importantly, you'll develop the intuition to choose the right method for any computational challenge.

The synthesis ahead will prepare you for the dynamic problems of Module 3, where these static methods become building blocks for simulating evolution through time.

*Next: Part 3 - Synthesis*