# SUBMODULE 2: STATIC PROBLEMS & QUADRATURE
*"Finding Equilibria and Measuring the Universe"*

## Learning Outcomes

By the end of this submodule, you will be able to:

- [ ] **Implement** root-finding algorithms for astrophysical equilibria with rigorous understanding of convergence
- [ ] **Select** appropriate methods based on problem characteristics and convergence requirements
- [ ] **Apply** numerical integration to measure cosmic quantities with controlled error
- [ ] **Derive** error bounds for quadrature methods from first principles
- [ ] **Connect** quadrature methods to Monte Carlo techniques for high-dimensional problems
- [ ] **Analyze** convergence rates and computational complexity trade-offs
- [ ] **Diagnose** failures in root-finding and integration algorithms
- [ ] **Design** hybrid approaches for challenging problems

---

## Introduction: The Static Universe - Finding Balance and Measuring Totals

Before diving into technical details, let's understand why root finding and integration are foundational to understanding the cosmos. The universe is filled with equilibrium points where competing forces achieve perfect balance, and measuring cosmic quantities requires integrating over scales from stellar surfaces to galactic halos.

:::{margin}
**Equilibrium**: A state where all forces balance, resulting in no net change. Mathematically, where derivatives equal zero.
:::

:::{margin}
**Quadrature**: Numerical integration - the process of approximating definite integrals using weighted sums of function values.
:::

**Root finding reveals where physics balances:**
- **Stellar radius**: Where pressure gradient exactly balances gravity ($dP/dr = -\rho g$)
- **Lagrange points**: Where spacecraft can orbit "motionless" relative to two bodies
- **Event horizons**: Where escape velocity equals the speed of light
- **Virial equilibrium**: Where kinetic energy balances potential energy in clusters
- **Photon spheres**: Where light can orbit in perfect circles around black holes

**Integration measures the universe:**
- **Stellar luminosity**: $L = \int_0^{\infty} 4\pi r^2 F(r) dr$ (integrating flux over surface)
- **Galaxy mass**: $M = \int_0^{\infty} 4\pi r^2 \rho(r) dr$ (integrating density profile)
- **Cosmological distances**: $d = c \int_0^z \frac{dz'}{H(z')}$ (integrating through expanding space)
- **Gravitational potential**: $\Phi = -G \int \frac{\rho(\vec{r'})}{|\vec{r}-\vec{r'}|} d^3r'$
- **Column density**: $N = \int_{-\infty}^{\infty} n(s) ds$ (integrating along line of sight)

**The computational challenge:**
These problems resist analytical solution. A star's structure involves coupled differential equations with pressure, temperature, and composition all varying with radius. Galaxy potentials require integrating over billions of stars. We need numerical methods that are:
- **Robust**: Work even with poor initial guesses
- **Efficient**: Converge quickly to required accuracy
- **Stable**: Don't accumulate errors over many iterations
- **Adaptive**: Handle both smooth and rapidly-varying functions

:::{admonition} Course Philosophy Reminder
:class: important
We build every algorithm from first principles. You'll understand not just HOW methods work, but WHY they were invented, WHEN they fail, and HOW to diagnose problems. No black boxes!
:::

This submodule provides the mathematical rigor and physical intuition to solve equilibrium problems and measure integrated quantities throughout astrophysics. These methods will be essential for your N-body orbits (Project 2), radiative transfer (Project 3), and Bayesian inference (Project 4).

---

## Bridge from Submodule 1: From Derivatives to Equilibria

You've mastered computing derivatives despite finite precision limitations. You discovered that the optimal step size balances truncation error ($\propto h^p$) against round-off error ($\propto \epsilon/h$). But derivatives tell us rates of change - what about finding where things *don't* change?

Consider the trajectory of a projectile. The derivative $dy/dt$ tells us the vertical velocity. But where does the projectile reach maximum height? Where $dy/dt = 0$. This is a root-finding problem! Similarly, integrating $dy/dt$ over time gives us the total vertical displacement - a quadrature problem.

The same error analysis principles from Submodule 1 apply here:
- **Root finding**: We iterate until $|f(x)| < \epsilon$, but how many iterations? What if we never converge?
- **Integration**: We approximate $\int f(x)dx$ with finite sums, but how many points? What spacing?

The key insight: both root finding and integration are *inverse problems* to differentiation. This submodule completes your toolkit for transforming between rates, accumulations, and equilibria.

---

## Part 1: Root Finding - Where Physics Reaches Equilibrium

### The Fundamental Problem

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

### Building Intuition: The Geometry of Root Finding

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

### Method 1: Bisection - The Reliable Workhorse

#### The Mathematical Foundation

:::{margin}
**Intermediate Value Theorem**: If $f$ is continuous on $[a,b]$ and $k$ is any value between $f(a)$ and $f(b)$, then there exists at least one $c \in (a,b)$ where $f(c) = k$.
:::

The bisection method is based on the **Intermediate Value Theorem**: 

> If $f$ is continuous on $[a,b]$ and $f(a) \cdot f(b) < 0$, then there exists at least one root $r \in (a,b)$ where $f(r) = 0$.

The algorithm repeatedly halves the interval, keeping the half that contains the root.

#### Visual Intuition

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

#### Algorithm Analysis

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

#### Iterations Required

To achieve error $< \epsilon$, we need:

$$\frac{|b_0 - a_0|}{2^{n+1}} < \epsilon$$

Solving for $n$:

$$n > \log_2\left(\frac{|b_0 - a_0|}{\epsilon}\right) - 1$$

For example, to find a root in $[0, 1]$ to 10 decimal places ($\epsilon = 10^{-10}$):

$$n > \log_2(10^{10}) - 1 \approx 33.2 - 1 = 32.2$$

So we need 33 iterations - slow but guaranteed!

#### Pseudocode

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

### Method 2: Newton-Raphson - The Speed Demon

#### The Geometric Insight

:::{margin}
**Tangent line approximation**: Replacing a curve with its tangent line at a point, valid for small deviations.
:::

Newton's method uses calculus to accelerate convergence. The key insight: near any point, a smooth function looks approximately linear. We can follow the **tangent line** to where it crosses the axis, getting much closer to the root in a single step.

#### Mathematical Derivation

Starting at point $x_n$, the tangent line to $f(x)$ has equation:

$$y - f(x_n) = f'(x_n)(x - x_n)$$

This line crosses the x-axis (where $y = 0$) at:

$$0 - f(x_n) = f'(x_n)(x_{n+1} - x_n)$$

Solving for $x_{n+1}$:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

This is the Newton-Raphson iteration formula.

#### Convergence Analysis

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

#### When Newton Fails

Despite its speed, Newton's method can fail spectacularly:

1. **Zero derivative**: If $f'(x_n) = 0$, the tangent is horizontal and never crosses the axis
2. **Poor initial guess**: May diverge or converge to wrong root
3. **Cycles**: Can oscillate between points without converging
4. **Non-smooth functions**: Requires continuous derivative

#### Visual of Newton's Method

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

#### Example of Newton's Method Cycling

For $f(x) = x^3 - 2x + 2$, starting near $x_0 = 0$:
- $x_1 \approx 1$
- $x_2 \approx 0$ 
- $x_3 \approx 1$ (cycling begins!)

This happens because the function has a special symmetry that causes the tangent lines to create a cycle.

#### Pseudocode

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

### Method 3: Secant - The Practical Compromise

#### Motivation: No Derivatives Required

:::{margin}
**Finite difference approximation**: Estimating derivatives using function values at nearby points.
:::

Newton's method requires $f'(x)$, but what if:
- The derivative is expensive to compute?
- We only have $f(x)$ as a black box?
- The function comes from experimental data?

The secant method approximates the derivative using a **finite difference**:

$$f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}$$

#### The Iteration Formula

Substituting this approximation into Newton's formula:

$$x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

Geometrically, we're replacing the tangent line with a secant line through two points.

#### Visual of Secant Method

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

#### Convergence Analysis

The convergence of the secant method is more complex to analyze. Through detailed analysis (involving difference equations), one can show that the order of convergence is:

$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$$

This is the golden ratio! The error satisfies:

$$|e_{n+1}| \approx C|e_n|^{\phi}$$

:::{margin}
**Superlinear convergence**: Faster than linear but slower than quadratic.
:::

This **superlinear convergence** is slower than Newton's quadratic rate but faster than bisection's linear rate.

#### Comparison of Convergence Rates

| Method | Order | Error Reduction | Digits Gained/Iteration |
|--------|-------|----------------|------------------------|
| Bisection | 1.0 | $e_{n+1} = \frac{1}{2}e_n$ | 0.30 (log₁₀ 2) |
| Secant | 1.618 | $e_{n+1} \approx Ce_n^{1.618}$ | ~0.62 (when e < 1) |
| Newton | 2.0 | $e_{n+1} \approx Ce_n^2$ | Doubles each iteration |

#### Pseudocode

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

### Hybrid Methods: Best of All Worlds

In practice, we often combine methods to leverage their strengths:

#### Brent's Method

Combines bisection's reliability with inverse quadratic interpolation's speed:
- Maintains a bracketing interval like bisection
- Tries fast interpolation when possible
- Falls back to bisection if interpolation fails

#### Practical Strategy

```
1. Start with bisection to bracket the root
2. Switch to Newton/Secant once close enough
3. Fall back to bisection if Newton diverges
```

---

### Debugging Root Finding

Common root-finding bugs and their diagnosis:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Bisection won't start | Same sign at endpoints | Scan for sign changes first |
| Newton diverges | Poor initial guess or $f'(x) \approx 0$ | Use bisection first to bracket |
| Secant oscillates | Nearly parallel secant lines | Check $\|f(x_1) - f(x_0)\|$ threshold |
| Wrong root found | Multiple roots in region | Narrow search interval |
| Infinite loop | Tolerance too small | Use $\epsilon_{machine} \times \|x\|$ |
| No convergence | Discontinuous function | Check function continuity |

#### Practical Initial Bracketing

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

### Worked Example: Kepler's Equation

The most famous root-finding problem in astronomy is Kepler's equation, which relates mean anomaly $M$ (proportional to time) to eccentric anomaly $E$ (related to position):

$$E - e\sin(E) = M$$

where $e$ is the orbital eccentricity. Given $M$ and $e$, solve for $E$.

#### Why This Is Hard

- Transcendental equation (no algebraic solution)
- Must be solved millions of times in orbit propagation
- High eccentricity makes convergence difficult

#### Newton's Method Applied

Define $f(E) = E - e\sin(E) - M$

Then $f'(E) = 1 - e\cos(E)$

Newton iteration:
$$E_{n+1} = E_n - \frac{E_n - e\sin(E_n) - M}{1 - e\cos(E_n)}$$

#### Initial Guess Strategy

Good initial guesses are crucial:
- For $e < 0.8$: Use $E_0 = M$
- For $e \geq 0.8$: Use $E_0 = \pi$

#### Convergence Analysis

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

### Worked Example: Photon Sphere Around a Black Hole

For a Schwarzschild black hole, photons can orbit at radius $r$ where the effective potential has a maximum. This occurs where:

$$\frac{d}{dr}\left[\left(1 - \frac{2GM}{rc^2}\right)\left(1 + \frac{L^2}{r^2}\right)\right] = 0$$

Simplifying for circular orbits gives:
$$r^3 - 3GMr^2/c^2 = 0$$

The non-trivial solution is the photon sphere radius:
$$r_{photon} = \frac{3GM}{c^2} = 1.5 r_s$$

where $r_s = 2GM/c^2$ is the Schwarzschild radius.

This is where light can orbit the black hole in unstable circular paths!

---

### 📊 Conceptual Checkpoint

Before moving to integration, ensure you understand:
- [ ] Why bisection always converges but Newton might not
- [ ] How convergence order affects iteration count
- [ ] When to use each method based on problem characteristics
- [ ] How to diagnose convergence failures
- [ ] The trade-off between reliability and speed

---

## Part 2: Quadrature - From Photon Counts to Dark Matter Halos

### The Fundamental Challenge

While differentiation is a local operation (needing only nearby points), integration is global - we must consider the entire domain. **Quadrature** transforms the continuous integral:

$$I = \int_a^b f(x) dx$$

into a discrete sum:

$$I \approx \sum_{i=0}^n w_i f(x_i)$$

The art lies in choosing:
- The points $x_i$ (where to sample)
- The weights $w_i$ (how much each sample contributes)
- The number $n$ (balancing accuracy vs computation)

### Physical Motivation: Why Astronomers Integrate

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

### Building Integration Methods: From Rectangles to Optimality

#### The Riemann Sum Foundation

The definition of the Riemann integral suggests the simplest approximation:

$$\int_a^b f(x)dx = \lim_{n \to \infty} \sum_{i=0}^{n-1} f(x_i^*) \Delta x_i$$

where $x_i^* \in [x_i, x_{i+1}]$ and $\Delta x_i = x_{i+1} - x_i$.

Different choices of $x_i^*$ give different methods:
- Left endpoint: Rectangle rule (left)
- Right endpoint: Rectangle rule (right)
- Midpoint: Midpoint rule
- Average of endpoints: Trapezoidal rule

---

### Method 1: Rectangle Rule - The Starting Point

#### Mathematical Formulation

For uniform spacing $h = (b-a)/n$:

$$I \approx h \sum_{i=0}^{n-1} f(a + ih) \quad \text{(left rectangle)}$$

#### Error Analysis via Taylor Series

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

#### Why Not to Use Rectangle Rule

- Only first-order accurate
- Systematically over/underestimates for monotonic functions
- Poor for oscillatory functions
- Mainly pedagogical value

---

### Method 2: Trapezoidal Rule - The Workhorse

#### Geometric Intuition

Instead of rectangles, connect consecutive points with straight lines, creating trapezoids.

#### Mathematical Formulation

The area of a trapezoid with parallel sides $f(x_i)$ and $f(x_{i+1})$ and height $h$ is:

$$A_i = \frac{h}{2}[f(x_i) + f(x_{i+1})]$$

Summing over all intervals:

$$I \approx \sum_{i=0}^{n-1} \frac{h}{2}[f(x_i) + f(x_{i+1})] = h\left[\frac{f(a)}{2} + \sum_{i=1}^{n-1} f(x_i) + \frac{f(b)}{2}\right]$$

#### Error Analysis

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

#### Pseudocode

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

### Method 3: Simpson's Rule - Parabolic Perfection

#### The Key Insight

Instead of linear interpolation (trapezoids), use quadratic interpolation (parabolas) through consecutive triplets of points.

#### Derivation via Interpolation

For three equally-spaced points $x_0, x_1, x_2$ with $x_1 = x_0 + h$ and $x_2 = x_0 + 2h$, the unique parabola through $(x_i, f(x_i))$ is:

$$p(x) = f(x_0)\frac{(x-x_1)(x-x_2)}{2h^2} - f(x_1)\frac{(x-x_0)(x-x_2)}{h^2} + f(x_2)\frac{(x-x_0)(x-x_1)}{2h^2}$$

Integrating this parabola from $x_0$ to $x_2$:

$$\int_{x_0}^{x_2} p(x)dx = \frac{h}{3}[f(x_0) + 4f(x_1) + f(x_2)]$$

These are the famous Simpson weights: 1, 4, 1.

#### Composite Simpson's Rule

For $n$ intervals (must be even):

$$I \approx \frac{h}{3}\left[f(a) + 4\sum_{i=1,3,5...}^{n-1} f(x_i) + 2\sum_{i=2,4,6...}^{n-2} f(x_i) + f(b)\right]$$

#### Error Analysis

Through careful Taylor series analysis, the local error for Simpson's rule on interval $[x_i, x_{i+2}]$ is:

$$E_{local} = -\frac{h^5}{90}f^{(4)}(\xi)$$

:::{margin}
**Fourth-order method**: Error decreases as $h^4$ - remarkably accurate for smooth functions!
:::

Global error: $E_{total} = -\frac{(b-a)h^4}{180}f^{(4)}(\xi)$

Simpson's rule is **fourth-order accurate** despite using only parabolas! This "superconvergence" occurs because the method is exact for cubic polynomials. Since the error term involves $f^{(4)}$, and the fourth derivative of any cubic is zero, Simpson's rule integrates cubics exactly.

#### When Simpson's Rule Shines

- Smooth functions with continuous fourth derivative
- Periodic functions
- When high accuracy is needed with moderate $n$

#### Pseudocode

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

#### Visual Comparison: Trapezoids vs Parabolas

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

#### Convergence Visualization

```{mermaid}
graph LR
    subgraph "Rectangle Rule O(h)"
        R1[n=10<br/>Error: 0.1] --> R2[n=20<br/>Error: 0.05]
        R2 --> R3[n=40<br/>Error: 0.025]
    end
    
    subgraph "Trapezoidal O(h²)"
        T1[n=10<br/>Error: 0.01] --> T2[n=20<br/>Error: 0.0025]
        T2 --> T3[n=40<br/>Error: 0.000625]
    end
    
    subgraph "Simpson's O(h⁴)"
        S1[n=10<br/>Error: 0.0001] --> S2[n=20<br/>Error: 0.0000063]
        S2 --> S3[n=40<br/>Error: 0.00000039]
    end
    
    style R1 fill:#ffebee
    style T1 fill:#fff9c4
    style S1 fill:#c8e6c9
```

:::{admonition} Check Your Understanding
:class: question
1. Why must n be even for Simpson's rule?
2. What happens if f(x) is exactly a cubic polynomial?
3. How does the error scale if we double n?
4. When would Simpson's rule perform poorly?
:::

---

### Method 4: Gaussian Quadrature - Optimal Point Placement

#### The Revolutionary Idea

All previous methods used equally-spaced points. But what if we could choose both the points AND weights optimally?

**Gauss's insight**: For $n$ points, we can make the method exact for all polynomials up to degree $2n-1$ by choosing the right locations!

#### Mathematical Foundation

We want to find points $x_i$ and weights $w_i$ such that:

$$\int_{-1}^{1} p(x)dx = \sum_{i=1}^n w_i p(x_i)$$

is exact for all polynomials $p(x)$ of degree $\leq 2n-1$.

:::{margin}
**Legendre polynomials**: A sequence of orthogonal polynomials on $[-1,1]$ with important properties for numerical integration.
:::

The optimal points are the roots of the $n$-th Legendre polynomial $P_n(x)$. This works because Legendre polynomials are orthogonal, meaning:

$$\int_{-1}^{1} P_n(x) P_m(x) dx = 0 \text{ for } n \neq m$$

This orthogonality property ensures that the Gauss points optimally sample the function.

#### Example: 2-Point Gaussian Quadrature

For $n=2$, we need exactness for polynomials up to degree 3.

Setting up equations for monomials $1, x, x^2, x^3$:
- $\int_{-1}^{1} 1\,dx = 2 = w_1 + w_2$
- $\int_{-1}^{1} x\,dx = 0 = w_1x_1 + w_2x_2$
- $\int_{-1}^{1} x^2\,dx = \frac{2}{3} = w_1x_1^2 + w_2x_2^2$
- $\int_{-1}^{1} x^3\,dx = 0 = w_1x_1^3 + w_2x_2^3$

Solution: $x_1 = -\frac{1}{\sqrt{3}}$, $x_2 = \frac{1}{\sqrt{3}}$, $w_1 = w_2 = 1$

#### Transformation to General Intervals

For interval $[a,b]$, transform from $[-1,1]$:

$$x = \frac{b-a}{2}t + \frac{a+b}{2}$$

$$\int_a^b f(x)dx = \frac{b-a}{2}\int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right)dt$$

#### Common Gaussian Quadrature Points and Weights

| n | Gauss Points | Weights | Exact for degree |
|---|-------------|---------|------------------|
| 2 | $\pm 0.57735$ | 1.00000, 1.00000 | 3 |
| 3 | $0.00000, \pm 0.77460$ | 0.88889, 0.55556, 0.55556 | 5 |
| 4 | $\pm 0.33998, \pm 0.86114$ | 0.65215, 0.65215, 0.34785, 0.34785 | 7 |

#### Why Gaussian Quadrature is Magical

- Optimal accuracy for given number of function evaluations
- 3 Gauss points often beats 100 equally-spaced points!
- Exact for polynomials up to degree $2n-1$
- Foundation for spectral methods in advanced simulations

---

### Method 5: Monte Carlo Integration - When Dimensions Explode

#### The Curse of Dimensionality

:::{margin}
**Curse of dimensionality**: Exponential growth of computational cost with dimension for grid-based methods.
:::

For a $d$-dimensional integral with $n$ points per dimension, deterministic methods need $n^d$ evaluations. This becomes impossible for $d > 5$.

#### The Monte Carlo Solution

Randomly sample $N$ points $\vec{x}_i$ in the domain $\Omega$:

$$I = \int_\Omega f(\vec{x})d\vec{x} \approx \frac{V(\Omega)}{N}\sum_{i=1}^N f(\vec{x}_i)$$

where $V(\Omega)$ is the volume of the domain.

#### Error Analysis

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

#### When Monte Carlo Wins

| Dimension | Grid Points for 1% Error | Monte Carlo Points for 1% Error |
|-----------|-------------------------|--------------------------------|
| 1 | 100 | 10,000 |
| 2 | 10,000 | 10,000 |
| 3 | 1,000,000 | 10,000 |
| 5 | 10^10 | 10,000 |
| 10 | 10^20 | 10,000 |

Above ~4 dimensions, Monte Carlo becomes superior!

#### Monte Carlo Visual Representation

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

#### Pseudocode

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

### Adaptive Methods and Advanced Topics

#### Adaptive Quadrature

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

#### Richardson Extrapolation and Romberg Integration

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

#### Quasi-Monte Carlo

For better random sampling than pure Monte Carlo:
- **Sobol sequences**: Low-discrepancy sequences that fill space more uniformly
- **Halton sequences**: Simple quasi-random sequences
- **Latin hypercube sampling**: Ensures all regions are sampled

These methods achieve error $O((\log N)^d/N)$ vs Monte Carlo's $O(1/\sqrt{N})$.

#### Handling Singularities

For integrands with singularities:
1. **Removable**: Transform variables to eliminate singularity
2. **Integrable**: Use specialized quadrature (e.g., Gauss-Chebyshev for $1/\sqrt{x}$)
3. **Non-integrable**: Regularize or use principal value

### Real Application: Galaxy Luminosity from Spectrum

Astronomers measure galaxy spectra as discrete samples at wavelengths $\lambda_i$ with fluxes $F_i$. The total luminosity requires integration:

$$L = 4\pi d^2 \int_{\lambda_{min}}^{\lambda_{max}} F_\lambda d\lambda$$

#### Challenges

1. **Irregular wavelength spacing** (different instruments, resolutions)
2. **Noise** in flux measurements
3. **Missing data** (atmospheric absorption, detector gaps)
4. **Emission lines** (narrow features needing resolution)

#### Solution Strategy

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

### Choosing the Right Integration Method

#### Decision Tree

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

#### Method Comparison Summary

| Method | Order | Pros | Cons | Best For |
|--------|-------|------|------|----------|
| Rectangle | $O(h)$ | Simple | Inaccurate | Teaching only |
| Trapezoid | $O(h^2)$ | Robust, simple | Moderate accuracy | Experimental data |
| Simpson | $O(h^4)$ | Very accurate | Needs smooth f, even n | Smooth functions |
| Gaussian | $O(h^{2n})$ | Optimal accuracy | Complex setup | When you choose points |
| Monte Carlo | $O(N^{-1/2})$ | Dimension-independent | Slow convergence | High dimensions |

---

### 🛠 Debugging Integration

Common integration bugs and their diagnosis:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Error doesn't decrease with n | Wrong implementation | Check weights, indices |
| Error plateaus at ~10^-14 | Round-off dominance | Use fewer points or higher precision |
| Oscillating convergence | Under-resolving oscillations | Increase n or use adaptive methods |
| Simpson gives odd results | Odd n | Ensure even intervals |
| Negative result for positive f | Overflow in sum | Use Kahan summation |

---

### 📊 Conceptual Checkpoint
Can you:
- [ ] Explain why Simpson's needs smooth functions?
- [ ] Predict which method works for noisy data?
- [ ] Determine when Monte Carlo beats deterministic methods?
- [ ] Choose integration method based on function properties?

### Additional Peer Instruction Question
*Think individually, then discuss:*

"You need to integrate a galaxy's spectral energy distribution with 50 narrow emission lines. Which method?"
- A) Rectangle rule with fine spacing
- B) Trapezoidal with adaptive spacing  
- C) Simpson's with uniform spacing
- D) Gaussian quadrature at line centers

*Answer: B - Trapezoidal with adaptive spacing handles irregular features robustly while capturing narrow lines. Simpson's might miss lines between grid points, and Gaussian quadrature needs smooth functions.*

### 🤔 Metacognitive Reflection
*Reflect on these connections:*

1. **How does quadrature connect to root finding?**
   - Both approximate continuous math discretely
   - Integration is "anti-derivative" - root finding often uses integrals

2. **Why does Monte Carlo work in high dimensions?**
   - Deterministic methods need exponentially many points
   - Random sampling explores efficiently

3. **Connection to Project 3 (MCRT)?**
   - Photon paths are high-dimensional integrals
   - Monte Carlo naturally handles complex geometries

---

## Synthesis: The Deep Connections

### Concept Map: Relationships Between Methods

```{mermaid}
graph TB
    subgraph "Root Finding"
        RF[Root Finding f(x)=0]
        Bis[Bisection<br/>Linear O(n)]
        Newt[Newton<br/>Quadratic O(n²)]
        Sec[Secant<br/>Superlinear O(n^1.618)]
        
        RF --> Bis
        RF --> Newt
        RF --> Sec
    end
    
    subgraph "Integration"
        INT[Integration ∫f(x)dx]
        Rect[Rectangle<br/>O(h)]
        Trap[Trapezoidal<br/>O(h²)]
        Simp[Simpson<br/>O(h⁴)]
        Gauss[Gaussian<br/>O(h^2n)]
        MC[Monte Carlo<br/>O(N^-1/2)]
        
        INT --> Rect
        INT --> Trap
        INT --> Simp
        INT --> Gauss
        INT --> MC
    end
    
    subgraph "Connections"
        Error[Error Analysis]
        Taylor[Taylor Series]
        Interp[Interpolation]
        
        Taylor -.-> Newt
        Taylor -.-> Simp
        Interp -.-> Sec
        Interp -.-> Trap
        Error -.-> RF
        Error -.-> INT
    end
    
    style RF fill:#e3f2fd
    style INT fill:#f3e5f5
    style Error fill:#fff9c4
```

### Root Finding ↔ Integration

These are inverse operations in many ways:

1. **Fundamental Theorem of Calculus**: If $F(x) = \int_a^x f(t)dt$, then finding where $F(x) = c$ is a root-finding problem

2. **Fixed points as integrals**: The equation $x = g(x)$ can be written as finding roots of $f(x) = x - g(x) = 0$

3. **Both are iterative**: Root finding iterates points, integration iterates over intervals

4. **Optimization connection**: Finding minima requires $f'(x) = 0$ (root finding), while the minimum value involves integration

### Error Analysis Principles

Both root finding and integration follow the same error framework from Submodule 1:

1. **Truncation error**: From approximating the true function
   - Root finding: Convergence order describes error reduction rate (linear, superlinear, quadratic)
   - Integration: Accuracy order describes error vs step size ($O(h)$, $O(h^2)$, $O(h^4)$)

2. **Round-off error**: From finite precision
   - Both limited by machine epsilon ($\epsilon \approx 2.2 \times 10^{-16}$)
   - Both have optimal problem sizes where total error is minimized

3. **Conditioning**: How errors amplify
   - Root finding: Condition number $\kappa = \frac{1}{|f'(r)|}$ - small derivative means ill-conditioned
   - Integration: Depends on function variation - oscillatory functions are ill-conditioned

:::{margin}
**Condition number**: A measure of how sensitive a problem is to small changes in input. Large condition numbers indicate ill-conditioned problems where errors amplify significantly.
:::

### Method Selection Flowchart

```{mermaid}
flowchart TD
    Start([Problem to Solve]) --> Type{Root Finding<br/>or Integration?}
    
    Type -->|Root Finding| RF[Root Finding Methods]
    Type -->|Integration| INT[Integration Methods]
    
    RF --> RFCheck{Have<br/>Derivative?}
    RFCheck -->|Yes| RFDeriv{Good Initial<br/>Guess?}
    RFCheck -->|No| RFNoDeriv{Need<br/>Guaranteed<br/>Convergence?}
    
    RFDeriv -->|Yes| Newton[Newton-Raphson<br/>Quadratic Conv.]
    RFDeriv -->|No| RFBracket{Can<br/>Bracket?}
    
    RFNoDeriv -->|Yes| Bisection[Bisection<br/>Linear Conv.]
    RFNoDeriv -->|No| Secant[Secant Method<br/>Superlinear Conv.]
    
    RFBracket -->|Yes| Hybrid[Start Bisection<br/>Switch to Newton]
    RFBracket -->|No| MultiStart[Multiple Starting<br/>Points]
    
    INT --> Dim{Dimension?}
    Dim -->|d ≤ 4| LowDim[Low Dimensional]
    Dim -->|d > 4| HighDim[Monte Carlo<br/>O(N^-1/2)]
    
    LowDim --> Smooth{Function<br/>Smooth?}
    Smooth -->|C^4 continuous| Simpson[Simpson's Rule<br/>O(h^4)]
    Smooth -->|C^2 continuous| Trap[Trapezoidal<br/>O(h^2)]
    Smooth -->|Noisy/Discrete| TrapRobust[Trapezoidal<br/>Robust]
    
    style Start fill:#e1f5fe
    style Newton fill:#c8e6c9
    style Bisection fill:#fff9c4
    style Simpson fill:#c8e6c9
    style HighDim fill:#ffccbc
```

### Computational Complexity

| Problem | Method | Function Evaluations | Convergence |
|---------|--------|---------------------|-------------|
| Root finding | Bisection | $\log_2(\epsilon^{-1})$ | Linear |
| Root finding | Newton | $\log_2(\log_2(\epsilon^{-1}))$ | Quadratic |
| Integration 1D | Trapezoid | $\epsilon^{-1/2}$ | $O(h^2)$ |
| Integration 1D | Simpson | $\epsilon^{-1/4}$ | $O(h^4)$ |
| Integration nD | Monte Carlo | $\epsilon^{-2}$ | $O(N^{-1/2})$ |

---

## Connections to Course Projects

### Immediate Applications (Project 2: N-body)

**Root Finding Applications:**
- Find perihelion/aphelion: Solve $\frac{dr}{dt} = 0$
- Determine collision times: Solve $|\vec{r}_1 - \vec{r}_2| = R_1 + R_2$
- Find circular orbit radius: Solve $F_{gravity} = F_{centrifugal}$

**Integration Applications:**
- Calculate orbital period: $P = \oint \frac{dt}{d\theta} d\theta$
- Compute system energy: $E = \int (T + V) dt$
- Find center of mass: $\vec{R}_{cm} = \frac{1}{M}\int \vec{r} dm$

### Future Project Connections

**Project 3 (Monte Carlo Radiative Transfer):**
- Monte Carlo integration for photon paths
- Root finding for optical depth boundaries
- Integration of emission/absorption coefficients

**Project 4 (Bayesian/MCMC):**
- Integration of posterior distributions
- Root finding for credible intervals
- Numerical normalization of probabilities

**Project 5 (Gaussian Processes):**
- Gaussian quadrature for kernel integrals
- Root finding for hyperparameter optimization
- Integration for marginal likelihood

**Final Project (Neural Networks):**
- Integration in loss functions
- Root finding in optimization (finding minima)
- Monte Carlo for stochastic gradient descent

---

## Practice Problems

### Root Finding Challenges

1. **Binary Star Separation**: A binary system has orbital period $P$. Find the separation $a$ using Kepler's third law:
   $$P^2 = \frac{4\pi^2 a^3}{G(M_1 + M_2)}$$

2. **Stellar Radius**: Find where pressure balances gravity in a polytrope:
   $$\frac{d}{dr}\left(r^2 \frac{dP}{dr}\right) = -4\pi r^2 \rho G \frac{M(r)}{r^2}$$

3. **Escape Velocity**: Find the radius where orbital velocity equals escape velocity (photon sphere for black holes)

### Integration Challenges

1. **Blackbody Luminosity**: Integrate Planck function:
   $$L = \int_0^{\infty} \frac{2\pi hc^2}{\lambda^5} \frac{1}{e^{hc/\lambda kT} - 1} d\lambda$$

2. **Galaxy Profile**: Integrate NFW profile to find enclosed mass:
   $$M(<r) = \int_0^r 4\pi r'^2 \frac{\rho_0}{(r'/r_s)(1 + r'/r_s)^2} dr'$$

3. **Limb Darkening**: Integrate stellar intensity over visible disk:
   $$F = \int_{disk} I(\mu) dA$$ where $\mu = \cos\theta$

---

## Summary and Key Takeaways

You've mastered the static problems of computational astrophysics:

### Root Finding
- **Three core methods** with different trade-offs:
  - Bisection: Slow but guaranteed (linear convergence)
  - Newton: Fast but fragile (quadratic convergence)  
  - Secant: Practical compromise (superlinear convergence)
- **Method selection** depends on derivatives, smoothness, and initial guess quality
- **Hybrid approaches** combine reliability with speed

### Numerical Integration
- **Order matters**: Simpson ($h^4$) vastly outperforms trapezoid ($h^2$) for smooth functions
- **Gaussian quadrature** achieves optimal accuracy by choosing points wisely
- **Monte Carlo** conquers high dimensions through randomness
- **Match method to problem**: Smoothness, dimension, and noise determine best approach

### Universal Principles
- **Error analysis** guides algorithm selection and parameter choice
- **Convergence rates** determine computational cost
- **Physics constraints** inform numerical decisions
- **No single best method** - context determines optimal approach

:::{admonition} Final Wisdom
:class: important
"In computational astrophysics, knowing WHEN each method fails is as important as knowing HOW it works. Always verify convergence, check error scaling, and validate against known solutions."
:::

---

## Bridge to Submodule 3: From Statics to Dynamics

You now command the tools for finding equilibria and measuring integrated quantities. But the universe is not static - it evolves! Next, we'll make time flow through differential equations.

In Submodule 3, you'll discover:
- How Euler's method fails catastrophically for orbits
- Why energy conservation requires symplectic integrators
- When higher-order doesn't mean better
- How to diagnose instabilities before they explode

The root finding and integration methods you've mastered become building blocks for the grand challenge: simulating the universe's evolution through time.

*Next: Submodule 3 - ODE Methods & Conservation*