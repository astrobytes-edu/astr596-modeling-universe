# SUBMODULE 2: STATIC PROBLEMS & QUADRATURE
*Week 3 - "Finding Equilibria and Measuring the Universe"*

## Learning Objectives
By the end of this submodule, you will:
- Implement root-finding algorithms for astrophysical equilibria
- Select appropriate methods based on problem characteristics
- Apply numerical integration to measure cosmic quantities
- Connect quadrature methods to Monte Carlo techniques

---

## Part 3: Root Finding - Where Physics Reaches Equilibrium

### Physical Motivation

The universe is full of equilibrium points where forces balance:
- **Lagrange points**: Gravitational + centrifugal forces cancel
- **Stellar structure**: Pressure gradient balances gravity
- **Photon spheres**: Light orbits in perfect circles around black holes
- **Hydrostatic equilibrium**: Where stars stop contracting

Finding these points means solving $f(x) = 0$ - but analytically impossible for most real problems!

### The Three Core Methods

#### Method 1: Bisection - Slow but Guaranteed
```python
def bisection(f, a, b, tol=1e-10):
    """Find root by binary search - always works!"""
    # Requirement: f(a) and f(b) have opposite signs
    assert f(a) * f(b) < 0, "Need sign change!"
    
    while (b - a) > tol:
        mid = (a + b) / 2  # Midpoint
        if f(mid) == 0: return mid
        if f(a) * f(mid) < 0:
            b = mid  # Root in left half
        else:
            a = mid  # Root in right half
    return (a + b) / 2
```
**Convergence**: Linear - error halves each iteration
**Pros**: Always converges if root exists
**Cons**: Slow (~50 iterations for machine precision)

#### Method 2: Newton-Raphson - Fast When It Works
```python
def newton(f, df, x0, tol=1e-10, max_iter=50):
    """Follow the tangent line to zero"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol: return x
        
        dfx = df(x)  # Need derivative
        if abs(dfx) < 1e-14: 
            break  # Derivative too small!
        
        x = x - fx/dfx  # Newton update
    return x
```
**Convergence**: Quadratic - digits double each iteration!
**Pros**: Blazing fast near root
**Cons**: Needs derivative, can diverge, sensitive to initial guess

#### Method 3: Secant - The Practical Compromise
```python
def secant(f, x0, x1, tol=1e-10):
    """Approximate derivative using two points"""
    for i in range(50):
        f0, f1 = f(x0), f(x1)
        if abs(f1) < tol: return x1
        
        # Approximate derivative: f'≈(f1-f0)/(x1-x0)
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x_new
    return x1
```
**Convergence**: Superlinear (~1.618 golden ratio!)
**Pros**: No derivative needed
**Cons**: Needs two initial points

### Worked Example: Kepler's Equation

The most famous root-finding problem in astronomy!

```python
# Kepler's Equation: E - e*sin(E) = M
# E = eccentric anomaly (what we want)
# e = eccentricity, M = mean anomaly (what we know)

def kepler_newton(M, e, tol=1e-10):
    """Solve Kepler's equation for eccentric anomaly"""
    E = M  # Good initial guess for small e
    
    for _ in range(10):  # Rarely needs more
        f = E - e*np.sin(E) - M
        df = 1 - e*np.cos(E)
        
        E_new = E - f/df
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    
    return E

# Test: Mars orbit (e = 0.0934)
M = np.pi/4  # 45 degrees mean anomaly
E = kepler_newton(M, 0.0934)
print(f"Eccentric anomaly: {np.degrees(E):.2f}°")
```

### Guided Practice: Lagrange Points
```python
# TODO: Find L1 point between Earth and Sun
def lagrange_L1(m1, m2, r):
    """
    Find L1 where gravitational and centrifugal balance
    Equation: GM₁/(x-r₁)² = GM₂/(r₂-x)² + ω²x
    """
    # Hints:
    # 1. L1 is between the two masses
    # 2. Start with x ≈ r * (m2/(3*m1))^(1/3)
    # 3. Use Newton's method
    # 4. Watch for singularities at mass positions!
    pass  # Your implementation here
```

### Independent Practice: Chandrasekhar Mass
Find the maximum mass of a white dwarf where electron degeneracy pressure balances gravity. The equation to solve:

$$\frac{GM}{R^2} = \frac{\hbar c}{m_e} \left(\frac{3\pi^2 N}{V}\right)^{1/3}$$

### ⚠️ Common Misconception Alert
> **"Newton's method is always best because it's fastest"**
> 
> **FALSE! Newton can spectacularly fail when:**
> - Derivative near zero (shoots off to infinity)
> - Multiple roots nearby (oscillates between them)
> - Poor initial guess (diverges)
> - Function not smooth (derivative undefined)
> 
> **Example**: Try Newton on $f(x) = x^{1/3}$ starting at $x = 1$

### When to Use Which Method

| Scenario | Best Method | Why |
|----------|------------|-----|
| Know root is bracketed | Bisection | Guaranteed convergence |
| Have good initial guess | Newton | Quadratic convergence |
| Expensive derivatives | Secant | No derivatives needed |
| Multiple roots | Bisection first | Isolate each root |
| Noisy function | Bisection | Robust to noise |
| Need all roots | Scan + any method | Systematic search |

### 🤝 Peer Instruction Question
*Think individually, then discuss:*

"You're finding where a satellite's orbital period equals Earth's rotation (geosynchronous orbit). Which method?"
- A) Bisection - guaranteed to work
- B) Newton - fastest convergence
- C) Secant - good compromise
- D) Depends on what information we have

*Answer: If we know physics well, Newton with period formula derivative. Otherwise, bisection between LEO and Moon.*

---

## Part 4: Quadrature - From Photon Counts to Dark Matter Halos

### Physical Motivation

Astronomy is fundamentally about integrating light:
- **Stellar luminosity**: $L = \int_0^{\infty} F_{\lambda} d\lambda$ (integrate spectrum)
- **Galaxy mass**: $M = \int_0^R 4\pi r^2 \rho(r) dr$ (integrate density)
- **Gravitational potential**: $\Phi = -G \int \frac{\rho(\vec{r'})}{|\vec{r}-\vec{r'}|} d^3r'$
- **Cosmic ray flux**: $F = \int E^{-\gamma} dE$ (power law spectrum)

But we only have discrete samples! How do we integrate?

### Building Up Integration Methods

#### Rectangle Rule (Don't Use!)
```python
def rectangle(f, a, b, n):
    """Simplest but terrible - O(h) error"""
    h = (b - a) / n
    return h * sum(f(a + i*h) for i in range(n))
```

#### Trapezoidal Rule - The Workhorse
```python
def trapezoid(f, a, b, n):
    """Connect points with straight lines - O(h²)"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    # Weight endpoints by 1/2
    return h * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])
```

#### Simpson's Rule - When Smoothness Pays Off
```python
def simpson(f, a, b, n):
    """Fit parabolas through triplets - O(h⁴)"""
    if n % 2: n += 1  # Need even number
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    # Weights: 1, 4, 2, 4, 2, ..., 4, 1
    return h/3 * (y[0] + 4*sum(y[1::2]) + 2*sum(y[2:-1:2]) + y[-1])
```

### Dimensional Analysis Bridge

**Key Principle**: Sample at ~10× the highest frequency component

```python
def choose_integration_points(physics):
    """Let physics guide numerical choices"""
    if physics == "spectrum_with_lines":
        # Need to resolve narrowest line
        n_points = 10 * (lambda_max / line_width)
    elif physics == "orbital_average":
        # Need to sample orbit adequately
        n_points = 100  # Per orbit
    elif physics == "density_profile":
        # More points where gradient steep
        # Use adaptive quadrature
        pass
    return n_points
```

### Gaussian Quadrature - Optimal Point Placement

Instead of evenly spaced points, choose optimal locations!

```python
# Gauss-Legendre quadrature example
# For integral from -1 to 1, n=2 points:
def gauss_2point(f):
    """Two strategically placed points beat many uniform ones!"""
    x1, x2 = -1/np.sqrt(3), 1/np.sqrt(3)  # Optimal points
    w1, w2 = 1.0, 1.0  # Weights
    return w1*f(x1) + w2*f(x2)
    # This is exact for polynomials up to degree 3!
```

### Monte Carlo Integration - When Dimensions Explode

For high dimensions, random sampling beats systematic!

```python
def monte_carlo_integrate(f, bounds, n_samples=10000):
    """Randomly sample the integration domain"""
    dim = len(bounds)
    volume = np.prod([b[1]-b[0] for b in bounds])
    
    # Random points in domain
    points = np.random.rand(n_samples, dim)
    for i, (low, high) in enumerate(bounds):
        points[:, i] = low + points[:, i]*(high-low)
    
    # Average value × volume
    values = [f(p) for p in points]
    return volume * np.mean(values), volume * np.std(values)/np.sqrt(n_samples)
```

**Error scales as $1/\sqrt{N}$ regardless of dimension!**

### Computational Complexity Comparison

| Method | Function Evals | Error Order | Best For | Warning |
|--------|---------------|-------------|----------|---------|
| Rectangle | n | $O(h)$ | Never use! | Teaching only |
| Trapezoid | n+1 | $O(h^2)$ | Experimental data | Assumes linear between points |
| Simpson | n+1 | $O(h^4)$ | Smooth functions | Needs smooth derivatives |
| Gaussian | n optimized | $O(h^{2n})$ | When you choose points | Complex setup |
| Monte Carlo | random n | $O(1/\sqrt{n})$ | d > 4 dimensions | Slow convergence |

### Real Astronomy Application: Measuring Galaxy Luminosity
```python
def galaxy_luminosity(wavelengths, fluxes):
    """
    Integrate spectral energy distribution
    Real challenge: irregular wavelength sampling!
    """
    # Sort by wavelength
    idx = np.argsort(wavelengths)
    wl = wavelengths[idx]
    fl = fluxes[idx]
    
    # Trapezoid rule for irregular spacing
    L = 0
    for i in range(len(wl)-1):
        dw = wl[i+1] - wl[i]
        L += 0.5 * (fl[i] + fl[i+1]) * dw
    
    return L * 4 * np.pi * distance**2  # Total luminosity
```

### 🐛 Debugging Challenge
```python
# BUG HUNT: Why does this integral give wrong answer?
def buggy_integrate(f, a, b, n=100):
    """Student's attempted Simpson's rule"""
    h = (b - a) / n
    total = f(a) + f(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            total += 2 * f(x)  # Bug 1: Wrong multiplier
        else:
            total += 4 * f(x)
    
    return total * h / 3  # Bug 2: Forgot to check n is even

# Fixes needed:
# 1. Even indices get factor 2 (except endpoints)
# 2. Must ensure n is even for Simpson's
```

### 📊 Conceptual Checkpoint
Can you:
- [ ] Explain why Simpson's needs smooth functions?
- [ ] Predict which method works for noisy data?
- [ ] Determine when Monte Carlo beats deterministic methods?
- [ ] Choose integration method based on function properties?

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

## Synthesis: Convergence Study

Let's compare all methods on a real problem:

```python
def compare_integration_methods():
    """Compare convergence rates empirically"""
    # Test function: stellar spectrum with emission line
    def spectrum(wavelength):
        continuum = 1.0 / wavelength**2  # Blackbody-ish
        line = 10 * np.exp(-((wavelength-656.3)/1)**2)  # H-alpha
        return continuum + line
    
    true_integral = 42.7  # "Known" from high-res integration
    n_values = [10, 20, 40, 80, 160]
    
    for n in n_values:
        rect = rectangle(spectrum, 400, 800, n)
        trap = trapezoid(spectrum, 400, 800, n)
        simp = simpson(spectrum, 400, 800, n)
        mc, mc_err = monte_carlo_integrate(
            lambda x: spectrum(x[0]), [(400, 800)], n*10
        )
        
        print(f"n={n:3d}: Rect={abs(rect-true_integral):.2e}, "
              f"Trap={abs(trap-true_integral):.2e}, "
              f"Simp={abs(simp-true_integral):.2e}, "
              f"MC={abs(mc-true_integral):.2e}±{mc_err:.2e}")
```

---

## Connections to Course Projects

### Immediate Application (Project 2)
- Root finding: Determine orbital perihelion/aphelion
- Integration: Calculate orbital period from radius integral

### Future Applications
- **Project 3**: Monte Carlo integration for radiative transfer
- **Project 4**: Integrate posterior distributions
- **Project 5**: Gaussian quadrature for kernel integrals
- **Final**: Numerical integration in loss functions

---

## Practice Problems

1. **Binary Period**: Find orbital radius for desired period (root finding)
2. **Spectrum Integration**: Compare methods on blackbody + lines
3. **Viral Equilibrium**: Find radius where kinetic = potential energy
4. **Power Law Integration**: Handle singularities in cosmic ray spectra

---

## Summary

You've mastered finding where physics balances and measuring integrated quantities:
- Root finding: Three methods with different trade-offs
- Quadrature: From simple trapezoids to optimal Gaussian points
- Monte Carlo: The high-dimensional savior
- Method selection: Match algorithm to problem characteristics

Next submodule: We'll make time flow through differential equations, discovering why energy conservation requires special methods!