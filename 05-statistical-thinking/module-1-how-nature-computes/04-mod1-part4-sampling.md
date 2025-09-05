---
title: "Part 4: Random Sampling - From Theory to Computation"
subtitle: "How Nature Computes | Statistical Thinking Module 1 | ASTR 596"
---

## Navigation
[← Part 3: Moments](03-part3-moments.md) | [Module 2a Home](00-part0-overview.md) | [Part 5: Summary →](05-part5-synthesis.md)

---

## Learning Outcomes

By the end of Part 4, you will be able to:

- [ ] **Implement** the inverse transform method to sample from analytical distributions including exponentials and power laws
- [ ] **Apply** the Cumulative Distribution Function (CDF) concept to transform uniform random numbers into any desired distribution
- [ ] **Sample** from complex astrophysical distributions like the Kroupa IMF using piecewise techniques
- [ ] **Design** rejection sampling algorithms for distributions without analytical inverse transforms
- [ ] **Generate** realistic initial conditions for N-body simulations including mass, position, and velocity distributions

---

## 3.1 Why Random Sampling Matters

**Priority: 🔴 Essential**
Theory tells us distributions exist. But for simulations, we need to generate samples. The challenge: computers only produce uniform random numbers [0,1]. How do we transform these into complex astrophysical distributions?

This is where theory meets computation. Random sampling bridges the gap between:

- Statistical understanding (Parts 1-2)
- Computational modeling (your projects)

```{mermaid}
flowchart TD
    A[Uniform Random 0 to 1] --> B[Choose Method]
    B --> C[Inverse Transform]
    B --> D[Rejection Sampling]
    B --> E[MCMC]
    C --> F[Exponential]
    C --> G[Power Law]
    D --> H[Complex PDFs]
    E --> I[High Dimensions]
    style A fill:#f9f,stroke:#333,stroke-width:2px
```

:::{admonition} 🌟 The More You Know: From Solitaire to Nuclear Weapons - The Monte Carlo Origin Story
:class: info, dropdown

In 1946, mathematician Stanislaw Ulam was recovering from brain surgery at Los Angeles' Cedars of Lebanon Hospital. To pass the time during his convalescence, he played endless games of Canfield solitaire. Being a mathematician, he naturally wondered: what's the probability of winning?

He tried to calculate it using combinatorial analysis, but the problem was intractable – too many possible card arrangements and decision trees. Then came the insight that would revolutionize computational physics: instead of trying to solve it analytically, why not just play thousands of games and count the wins?

Back at Los Alamos, Ulam shared this idea with his colleague John von Neumann. They were working on nuclear weapon design, specifically trying to understand neutron diffusion—how neutrons travel through fissile material, sometimes causing more fissions, sometimes escaping. The equations were impossibly complex, but Ulam's solitaire insight applied perfectly: instead of solving the equations, simulate thousands of random neutron paths and count the outcomes!

They needed a code name for this classified work. Nicholas Metropolis suggested "Monte Carlo" after the famous casino in Monaco, inspired by an uncle who would borrow money from relatives to gamble there. The name was perfect – this method was essentially gambling with mathematics, rolling dice millions of times to get statistical answers.

The first Monte Carlo calculations were performed on ENIAC in 1948. It took the early computer weeks to run simulations that your laptop could do in seconds. But it worked, providing crucial insights for the hydrogen bomb design.

The irony is beautiful: a method developed for the most destructive weapon ever created now helps us understand everything from protein folding to galaxy formation. Your Project 3 radiative transfer code uses the same fundamental technique that emerged from Ulam's hospital bed solitaire games. When you launch random photon packets through your simulated atmosphere, you're using the intellectual descendant of those first neutron transport calculations.

Today, Monte Carlo methods are everywhere:

- Wall Street uses them for options pricing
- Drug companies use them for clinical trial design  
- Climate scientists use them for weather prediction
- Netflix uses them for recommendation algorithms

All because a mathematician recovering from brain surgery asked a simple question about solitaire. Sometimes the most powerful ideas come not from solving equations, but from admitting they're too hard and finding a cleverer way.
:::

::::{admonition} 💻 Hands-On Problem: Monte Carlo Estimation of $\pi$
:class: note

**Your Challenge**: Write a function that estimates π using Monte Carlo sampling.

**The Setup**:

- Consider a circle of radius 1 inscribed in a square with side length 2 (from -1 to 1)
- Area of square: 4
- Area of circle: π
- If you randomly throw darts at the square, the fraction landing inside the circle should be π/4

**Your Task**:

1. Write a function `estimate_pi(n_samples)` that:
   - Generates `n_samples` random points in the square [-1,1] × [-1,1]
   - Counts how many fall inside the unit circle (x² + y² ≤ 1)
   - Returns an estimate of π
   
2. Test your function with n = 100, 1000, 10000, 100000
3. Calculate the error for each sample size
4. What do you notice about how the error changes with n?

**Hint**: Use `np.random.uniform(-1, 1, n_samples)` to generate random coordinates.

:::{admonition} Solution
:class: tip, dropdown

```python
import numpy as np

def estimate_pi_monte_carlo(n_samples):
    """
    Estimate π using Monte Carlo sampling.
    
    Method: Generate random points in [-1,1] x [-1,1].
    Check if they fall inside unit circle (x² + y² ≤ 1).
    π/4 = (points inside circle) / (total points)
    """
    # Generate random points in square [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Check if points are inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    
    # Estimate π
    # π/4 = fraction inside circle, so π = 4 * fraction
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate

# Test with increasing sample sizes
print(f"True value: π = {np.pi:.10f}")
print("-" * 40)

for n in [100, 1000, 10000, 100000, 1000000]:
    pi_est = estimate_pi_monte_carlo(n)
    error = abs(pi_est - np.pi)
    relative_error = error / np.pi * 100
    print(f"N = {n:7d}: π ≈ {pi_est:.6f}, error = {error:.6f} ({relative_error:.2f}%)")
```

:::

**Expected Output**:

```bash
True value: π = 3.1415926536
----------------------------------------
N =     100: π ≈ 3.160000, error = 0.018407 (0.59%)
N =    1000: π ≈ 3.132000, error = 0.009593 (0.31%)
N =   10000: π ≈ 3.141200, error = 0.000393 (0.01%)
N =  100000: π ≈ 3.142520, error = 0.000927 (0.03%)
N = 1000000: π ≈ 3.141532, error = 0.000061 (0.00%)
```

**What You Should Observe**:

1. The estimate gets more accurate as n increases
2. Error roughly decreases as $1/\sqrt{n}$ (to get 10× better accuracy, need 100× more samples!)
3. Even with 1 million samples, we only get 3-4 decimal places of accuracy
4. The method is simple but computationally expensive for high precision

**Extension**: Try plotting the error vs n on a log-log plot. You should see a slope of -0.5, confirming the $1/\sqrt{n}$ scaling!
::::

:::{important} 🔑 What We Just Learned
Monte Carlo methods converge as 1/$\sqrt{N}$:

- Want $10×$ better accuracy? Need $100×$ more samples!
- This scaling is universal - same for integrating stellar opacities or galaxy luminosity functions
- The method works in any dimension (crucial for high-dimensional integrals)
- Random sampling turns geometry problems into counting problems
:::

:::{figure} figures/11_monte_carlo_pi_estimation.png
:name: fig-monte-carlo-pi-estimation
:width: 100%
**Monte Carlo Method: Estimating π Through Random Sampling.** This comprehensive demonstration shows the foundation technique for all computational astrophysics - transforming complex integrals into simple counting problems using random sampling. **Top-left**: Monte Carlo dartboard method with 1,000 samples shows random points (teal inside circle, rose outside) thrown at a square containing a unit circle. The π estimate comes from counting: π ≈ 4 × (fraction of points inside circle). **Top-right**: High density sampling with 10,000 points reveals the method's statistical nature - more samples give better coverage of the true circle boundary, with accuracy annotation showing typical ~0.5% error. **Bottom-left**: Error scaling analysis demonstrates the fundamental 1/√N convergence rate that governs all Monte Carlo methods, comparing measured errors (blue circles) against theoretical scaling (dashed rose line) across 6 orders of magnitude in sample size. **Bottom-right**: Convergence to true value shows how π estimates (teal line) approach the exact value (rose horizontal line) as N increases, with expected ±1σ statistical bounds (gray shaded region). This illustrates the universal trade-off in computational astrophysics: statistical precision scales as 1/√N, meaning 100× more computation yields only 10× better accuracy. The same technique that estimates π also computes stellar opacities, galaxy luminosity functions, and radiative transfer - making this the foundation of modern computational astrophysics.
:::

## 4.2 The Cumulative Distribution Function (CDF) and Inverse Transform Sampling

:::{margin}
**Cumulative Distribution Function (CDF)**
For a random variable X with PDF f(x), the CDF is F(x) = P(X ≤ x), giving the probability that X takes a value less than or equal to x. Always monotonically increasing from 0 to 1.
:::

**Priority: 🔴 Essential**
The key to sampling from any distribution lies in understanding the **cumulative distribution function** (CDF) and how it enables the inverse transform method—the most fundamental technique for converting uniform random numbers into samples from any distribution.

### Understanding the CDF

The CDF is defined as:

$\boxed{F(x) = P(X \leq x) = \int_{-\infty}^{x} f(x') dx'}$

While the probability density function (PDF) f(x) tells us the relative likelihood at each point, the CDF F(x) tells us the accumulated probability up to that point. This accumulation is what makes sampling possible.

To understand why this is so powerful, consider what the CDF actually represents. If you have a PDF f(x) that describes the distribution of stellar masses, then F(m) answers the question: "What fraction of stars have mass less than or equal to m?"

For example, in a stellar population:

- $F(0.5 M_\odot)$ = 0.3 means 30% of stars have mass $\le 0.5 M_\odot$
- $F(1.0 M_\odot)$ = 0.6 means 60% of stars have mass $\le 1.0 M_\odot$  
- $F(∞) = 1.0$ means 100% of stars have some finite mass

**Key properties of CDFs**:

- **Always non-decreasing**: $F(x₁) ≤ F(x₂) if x₁ < x₂$
- **Bounded**: $F(-∞) = 0$ and $F(∞) = 1$
- **Continuous from the right**: Important for discrete distributions
- **Derivative gives PDF**: $dF/dx = f(x)$ for continuous distributions

### The Inverse Transform Method

The crucial insight is that the CDF transforms any distribution—no matter how complex—into a uniform distribution on [0,1]. This provides the bridge between uniform random numbers (which computers generate) and any distribution we want.

:::{admonition} 🎯 The Inverse Transform Algorithm
:class: important

To sample from a distribution with CDF F(x):

1. Generate $u \sim \text{Uniform}(0,1)$
2. Solve $F(x) = u \text{ for } x$
3. The solution $x = F⁻¹(u)$ follows your desired distribution

**Why it works**: If $U$ is uniformly distributed on $[0,1]$, then $F⁻¹(U)$ has the distribution $f(x).$ The CDF essentially "encodes" your distribution in a way that uniform random numbers can decode.
:::

**Visual intuition**: Imagine the CDF as a transformation that "stretches" the uniform distribution. Regions where the PDF is large (high probability density) correspond to steep sections of the CDF. When we invert, these steep sections get "compressed" back, allocating more samples to high-probability regions.

#### Example: Exponential Distribution

Before tackling power laws, let's see how this works for the exponential distribution (which describes photon path lengths, radioactive decay, and time between stellar collisions):

**PDF**: $f(x) = \lambda e^{-\lambda x}$ for x ≥ 0

**CDF**: $F(x) = \int_0^x \lambda e^{-\lambda x'} dx' = 1 - e^{-\lambda x}$

**Inverse**: Solve $u = 1 - e^{-\lambda x}$ for x:

- $e^{-\lambda x} = 1 - u$
- $x = -\frac{1}{\lambda}\ln(1-u)$

Since u and (1-u) are both uniform on [0,1], we can simplify to:
$x = -\frac{1}{\lambda}\ln(u)$

```python
# Sample exponential distribution (e.g., photon mean free paths)
def sample_exponential(lambda_param, n_samples):
    u = np.random.uniform(0, 1, n_samples)
    return -np.log(u) / lambda_param

# Example: mean free path of 1 pc
mfp = 1.0  # pc
lambda_param = 1.0 / mfp
path_lengths = sample_exponential(lambda_param, 10000)
print(f"Mean path: {np.mean(path_lengths):.2f} pc")
print(f"Theory: {mfp:.2f} pc")
```

### Power Law Distributions: The Foundation of Astrophysics

Power law distributions appear everywhere in astronomy:

- Stellar initial mass function (IMF)
- Galaxy luminosity functions
- Cosmic ray energy spectrum
- Size distribution of asteroids and dust grains

:::{admonition} 📚 Example: Sampling from a Power Law Distribution
:class: note

Consider a power law PDF: $p(x) \propto x^{-\alpha}$ for $x \in [x_{\min}, x_{\max}]$

This is the foundation for understanding the Kroupa IMF, which is a broken power law with three different α values in different mass ranges.
:::

**Deriving the sampling formula**:

1. **Normalize the PDF**: 
   $f(x) = \frac{(\alpha-1) x^{-\alpha}}{x_{\min}^{1-\alpha} - x_{\max}^{1-\alpha}}$ 
   (for α ≠ 1)

2. **Compute the CDF**:
   $F(x) = \int_{x_{\min}}^x f(x')dx' = \frac{x^{1-\alpha} - x_{\min}^{1-\alpha}}{x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha}}$

3. **Invert to get sampling formula**:
   Set F(x) = u and solve for x:
   $\boxed{x = \left[x_{\min}^{1-\alpha} + u(x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha})\right]^{1/(1-\alpha)}}$

**Special case α = 1**: The integral of 1/x is ln(x), so:
$\boxed{x = x_{\min} \left(\frac{x_{\max}}{x_{\min}}\right)^u}$

:::{figure} figures/13_power_law_distribution_sampling.png
:name: fig-power-law-sampling
:width: 100%

**Sample Size Determines Accuracy in Power Law Distributions.** The Salpeter Initial Mass Function (α=2.35) demonstrates how statistical convergence works in practice. Left panel (N=100): Small samples show noisy histograms with poor fits to the theoretical power law—individual random fluctuations dominate. Center panel (N=1,000): Medium samples begin to reveal the underlying distribution structure with better agreement. Right panel (N=10,000): Large samples produce smooth histograms that closely match theory. Statistics boxes show quantitative metrics: mean mass, median mass, and fraction below 1 M☉. This convergence behavior is universal across all Monte Carlo methods—larger samples always yield more accurate results, following the fundamental 1/√N error scaling we learned in Section 4.1.
:::

### Implementation and Visualization

The figure above shows our power law sampling in action. Here's how to implement it yourself:

```python
import numpy as np
import matplotlib.pyplot as plt

def sample_simple_power_law(alpha, x_min, x_max, n_samples):
    """
    Sample from a power law distribution p(x) ∝ x^(-alpha).
    
    This demonstrates the inverse transform method for power laws.
    Special case: alpha = 1 requires logarithmic sampling.
    
    Parameters:
    -----------
    alpha : float
        Power law exponent (positive for decreasing function)
    x_min, x_max : float
        Range of the distribution
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    samples : ndarray
        Random samples from the power law
    """
    u = np.random.uniform(0, 1, n_samples)
    
    if abs(alpha - 1.0) < 1e-10:
        # Special case: p(x) ∝ x^(-1)
        # CDF integral gives logarithmic form
        samples = x_min * (x_max/x_min)**u
    else:
        # General case: p(x) ∝ x^(-alpha)
        # Apply the inverse transform formula we derived
        samples = (x_min**(1-alpha) + u*(x_max**(1-alpha) - x_min**(1-alpha)))**(1/(1-alpha))
    
    return samples

# Example: Sample stellar masses with different power laws
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Different IMF slopes (Salpeter has alpha = 2.35)
alphas = [1.3, 2.3, 3.5]
x_min, x_max = 0.1, 100  # Solar masses

for ax, alpha in zip(axes, alphas):
    # Generate samples
    samples = sample_simple_power_law(alpha, x_min, x_max, 10000)
    
    # Plot histogram
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)
    ax.hist(samples, bins=bins, alpha=0.7, density=True, label='Samples')
    
    # Overlay theoretical distribution
    x_theory = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    # Normalized power law
    if alpha != 1:
        norm = (1-alpha)/(x_max**(1-alpha) - x_min**(1-alpha))
        p_theory = abs(norm) * x_theory**(-alpha)  # abs() for alpha < 1
    else:
        norm = 1/np.log(x_max/x_min)
        p_theory = norm / x_theory
    
    ax.plot(x_theory, p_theory, 'r-', linewidth=2, label='Theory')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Mass $[M_☉])
    ax.set_ylabel('Probability Density')
    ax.set_title(rf'$\alpha = ${alpha}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_mass = np.mean(samples)
    median_mass = np.median(samples)
    text_str = rf'Mean: {mean_mass:.1f} $M_☉ + '\n' + rf'Median: {median_mass:.1f} $M_☉
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Power Law Sampling with Different Slopes', fontsize=14)
plt.tight_layout()
plt.show()

# Show how the distribution changes with alpha
print("Effect of power law slope α on stellar mass distribution:")
print(f"{'α':<5} {'Mean [M_☉]':<12} {'Median [M_☉]':<12} {'% below 1 M_☉':<15}")
print("-" * 50)

for alpha in alphas:
    samples = sample_simple_power_law(alpha, x_min, x_max, 10000)
    mean_mass = np.mean(samples)
    median_mass = np.median(samples)
    frac_below_1 = np.sum(samples < 1.0) / len(samples) * 100
    print(f"{alpha:<5.1f} {mean_mass:<12.2f} {median_mass:<12.2f} {frac_below_1:<15.1f}")
```

:::{admonition} 🎯 Conceptual Understanding
:class: important

Notice how the power law slope α dramatically affects the distribution:

- **α < 2**: Mean dominated by massive objects ("top-heavy")
  - Most of the total mass is in high-mass stars
  - Rare but massive objects dominate

- **α = 2**: Special transition point
  - Every logarithmic mass decade contributes equally to total mass
  - log-uniform mass distribution

- **α > 2**: Mean dominated by low-mass objects ("bottom-heavy")
  - Most of the total mass is in low-mass stars
  - Numerous small objects dominate

**Project 2: Kroupa Initial Mass Function (IMF)**
The real Kroupa IMF uses $α = 2.3$ for high masses ($M > 0.5 M_☉$), ensuring that most stellar mass is in low-mass stars. This is why the galaxy is full of red dwarfs! The IMF becomes even steeper $(α = 1.3)$ for intermediate masses $(0.08 M_☉ < M \le 0.5 M_☉)$ and flatter $(α = 0.3)$ for brown dwarfs ($(0.01 M_☉ < M \le 0.08 M_☉)$), creating a complex broken power law.

$\xi(m) \propto \begin{cases}
m^{-0.3} & 0.01 < m/M_\odot < 0.08 \\
m^{-1.3} & 0.08 < m/M_\odot < 0.5 \\
m^{-2.3} & 0.5 < m/M_\odot < 150
\end{cases}$

For power law $f(m) \propto m^{-\alpha}$ on $[m_{\min}, m_{\max}]$:

**Sampling formula** (for $\alpha \neq 1$):
$\boxed{m = \left[m_{\min}^{1-\alpha} + u(m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha})\right]^{1/(1-\alpha)}}$

For $\alpha = 1$ (logarithmic):
$\boxed{m = m_{\min} \left(\frac{m_{\max}}{m_{\min}}\right)^u}$

For Project 2, you'll implement this broken power law with three segments, ensuring continuity at the break points. The principle is the same—apply the inverse transform formula piecewise—but you'll need to:

1. Calculate normalization constants for continuity
2. Choose which segment to sample from based on their relative probabilities
3. Apply the appropriate inverse transform for that segment

:::

### Why the Inverse Transform is Powerful

The inverse transform method is the foundation of Monte Carlo sampling because:

1. **It's exact**: Every uniform random number produces exactly one sample from your distribution
2. **It's efficient**: 100% of random numbers are used (no rejections)
3. **It's deterministic**: The same uniform input always gives the same output (useful for debugging)
4. **It works in any dimension**: Can be extended to multivariate distributions

However, it requires that you can:

- Compute the CDF analytically (or numerically)
- Invert the CDF (analytically or numerically)

When these conditions aren't met, you'll need other methods like rejection sampling (Section 3.6) or Markov Chain Monte Carlo (Project 4).

:::{important} 🔑 Key Takeaways

- The CDF F(x) transforms any distribution to uniform [0,1]
- Inverse transform: x = F⁻¹(u) where u ~ Uniform(0,1)
- Power laws have analytical inverse: $x = [x_{\min}^{1-\alpha} + u(x_{\max}^{1-\alpha} - x_{\min}^{1-\alpha})]^{1/(1-\alpha)}$
- The slope α determines whether the distribution is top-heavy or bottom-heavy
- This method is exact and efficient when you can compute F⁻¹
:::

**Example**: Exponential distribution (photon path lengths!)

- PDF: $f(x) = \lambda e^{-\lambda x}$
- CDF: $F(x) = 1 - e^{-\lambda x}$
- Inverse: $x = -\frac{1}{\lambda}\ln(1-u)$

**Simple implementation:**

```python
# Sample exponential distribution (e.g., photon path lengths)
def sample_exponential(lambda_param, n_samples):
    u = np.random.uniform(0, 1, n_samples)
    # Inverse CDF transform
    return -np.log(1 - u) / lambda_param

# Example: mean free path of 1 pc
mfp = 1.0  # pc
lambda_param = 1.0 / mfp
path_lengths = sample_exponential(lambda_param, 10000)
print(f"Mean path: {np.mean(path_lengths):.2f} pc")
print(f"Theory: {mfp:.2f} pc")
```

### 4.6 Rejection Sampling

**Priority: 🟡 Standard Path**
When inverse transform fails, use rejection sampling:

:::{admonition} 🎯 Rejection Algorithm
:class: important

To sample from PDF $f(x)$:

1. Find envelope $M \geq \max(f(x))$
2. Generate $x \sim U(a,b)$, $y \sim U(0,M)$
3. Accept if $y \leq f(x)$
4. Otherwise reject, repeat

Efficiency = (area under curve)/(box area)
:::

Example: Throw darts at rectangle, keep those under curve.

```python
# Example: Sample from arbitrary distribution
def rejection_sample(f, x_min, x_max, f_max, n_samples):
    """Sample using rejection method"""
    samples = []
    n_tries = 0

    while len(samples) < n_samples:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, f_max)
        n_tries += 1

        if y <= f(x):
            samples.append(x)

    efficiency = n_samples / n_tries
    print(f"Efficiency: {efficiency:.1%}")
    return np.array(samples)
```

:::{figure} figures/12_random_sampling_methods.png
:name: fig-random-sampling-methods
:width: 100%
**Random Sampling Methods: From Uniform to Any Distribution.** This comprehensive comparison demonstrates the two fundamental Monte Carlo sampling techniques that bridge the gap between uniform computer-generated random numbers and complex astrophysical distributions. **Left panels**: Inverse Transform Method applied to exponential distribution shows (top) the theoretical exponential CDF F(x) = 1 - e^(-λx) mapping uniform [0,1] inputs to exponentially distributed outputs, with the dashed line illustrating how uniform input u=0.5 transforms to x=0.35 via the inverse CDF. The (bottom) histogram of 10,000 samples perfectly matches the theoretical exponential PDF f(x) = λe^(-λx) with λ=2. **Right panels**: Rejection Sampling Method for quadratic distribution demonstrates (top) the "dart-throwing" approach where uniform (x,y) points fill a bounding box, with accepted points (blue, lying under the curve) separated from rejected points (red, above the curve). The (bottom) histogram shows excellent agreement between accepted samples and the theoretical quadratic PDF f(x) = 3x². This figure illustrates why inverse transform is preferred when analytical CDFs exist (100% efficiency, exact sampling) while rejection sampling handles arbitrary distributions at the cost of computational overhead through rejections.
:::

## 4.7 Spatial Distributions: The Plummer Sphere

**Priority: 🔴 Essential for Project 2**
Now that you can sample stellar masses from the Kroupa IMF, you need to place these stars in space. The Plummer sphere is a standard density profile for globular clusters and stellar systems.

### The Plummer Profile

The Plummer sphere has density profile:
$$\rho(r) = \frac{3M_{total}}{4\pi a^3} \left(1 + \frac{r^2}{a^2}\right)^{-5/2}$$

where:

- $M_{total}$ is the total cluster mass
- $a$ is the Plummer radius (scale length)
- $r$ is the distance from center

This creates a smooth, centrally concentrated distribution that avoids the infinite density at r=0 that plagues some other profiles.

### The Sampling Challenge

You can't sample directly from $\rho(r)$ because density isn't probability! The key insight: the probability of finding a star between radius $r$ and $r+dr$ is proportional to the **mass** in that shell, not the density.

$$dP \propto \rho(r) \times \text{(volume of shell)} \propto \rho(r) \times 4\pi r^2 dr$$

So the radial PDF is:
$$f(r) \propto r^2 \rho(r) = r^2 \left(1 + \frac{r^2}{a^2}\right)^{-5/2}$$

#### Approach 1: Mass Profile Method

The enclosed mass within radius $r$ is:
$$M(<r) = M_{total} \frac{r^3}{(r^2 + a^2)^{3/2}}$$

The cumulative mass fraction (which IS a CDF!) is:
$$F(r) = \frac{M(<r)}{M_{total}} = \frac{r^3}{(r^2 + a^2)^{3/2}}$$

To sample:

1. Generate $u \sim U(0,1)$ (this represents a mass fraction)
2. Solve $u = \frac{r^3}{(r^2 + a^2)^{3/2}}$ for $r$
3. Place the star at radius $r$ with random angular coordinates

:::{admonition} 🎯 Hint: Solving for r
:class: hint

Let $s = r^2/a^2$. Then the equation becomes:
$$u = \frac{s^{3/2}}{(1 + s)^{3/2}} = \left(\frac{s}{1+s}\right)^{3/2}$$

Can you solve for $s$, then get $r$?
:::

#### Approach 2: Rejection Sampling in 3D

Alternatively, you could:

1. Generate random positions in a cube containing your cluster
2. Accept/reject based on the density at that position
3. But this becomes inefficient for large clusters!

#### From Radius to 3D Positions

Once you have radius $r$:

1. Generate random direction on sphere:
   - $\theta \sim U(0, \pi)$ (polar angle)
   - $\phi \sim U(0, 2\pi)$ (azimuthal angle)
2. Convert to Cartesian:
   - $x = r \sin\theta \cos\phi$
   - $y = r \sin\theta \sin\phi$  
   - $z = r \cos\theta$

Or use the simpler method: generate a random unit vector and scale by $r$.

#### Verification

Your sampled positions should reproduce:

1. The density profile $\rho(r)$ when binned radially
2. The enclosed mass profile $M(<r)$
3. The half-mass radius $r_{half} \approx 1.3a$ for Plummer

:::{figure} figures/14_plummer_sphere_spatial_sampling.png
:name: fig-plummer-sphere-sampling
:width: 100%

**Plummer Sphere Sampling: From 1D CDF to 3D Stellar Positions.** This comprehensive demonstration shows how inverse transform sampling creates realistic star cluster initial conditions using 10,000 stellar positions. **Top-left**: Face-on 2D projection shows the centrally concentrated structure with color-coded radial distances (bright yellow = center, dark purple = outer regions, colorbar spans 0-4 scale radii). Reference circles mark 1a, 2a, and 3a. **Top-right**: Density profile verification on log-log scale compares our sampled data points (blue) against the theoretical Plummer profile (red line), showing excellent agreement across radius range 0.1-4.0a with normalized density values up to 1.0. **Bottom-left**: Cumulative mass profile demonstrates that our inverse transform sampling correctly reproduces the theoretical CDF (red curve), with the half-mass radius marked at r₁/₂ = 1.3a where 50% of stars lie within this radius. **Bottom-right**: Edge-on view reveals the 3D spherical structure, confirming that our sampling method produces realistic spatial distributions for N-body simulations. This validates that the inverse transform method perfectly converts uniform random numbers into physically meaningful 3D stellar positions following the Plummer density law.
:::

:::{admonition} 🌟 Why Plummer for Globular Clusters?
:class: note, dropdown

The Plummer sphere, developed by H.C. Plummer in 1911 for fitting star clusters, has several advantages:

1. **Finite central density**: Unlike the singular isothermal sphere (SIS) model.
2. **Finite total mass**: Unlike the Navarro–Frenk–White (NFW) profile that extends to infinity, used to describe a spatial mass distribution of dark matter fitted to dark matter halos.
3. **Analytical everything**: Density, potential, distribution function all have closed forms.
4. **Stable**: Can be realized as equilibrium of self-gravitating system.

Real globular clusters are more complex (mass segregation, tidal truncation, rotation), but the Plummer profile is an excellent starting point that captures the essential physics: central concentration with smooth decline.
:::

:::{important} 🔑 Key Implementation Tips

- Don't sample from density directly - sample from mass!
- The radial CDF is the **enclosed** mass fraction
- Check your profile by plotting both $\rho(r)$ and $M(<r)$ (1D profiles).
- For Project 2, combine this with Kroupa IMF masses and virial velocities.
:::

### Part 4 Synthesis: Theory to Computation

:::{admonition} 🎯 What We Just Learned
:class: important

**Random sampling bridges theory and simulation**:

1. **CDFs** map any distribution to [0,1]
2. **Inverse transform** samples from analytical distributions
3. **Power laws** (IMF) use piecewise sampling with special $\alpha=1$ case
4. **Density profiles** sample from mass not density
5. **Rejection sampling** handles complex cases

You now have everything for Project 2:

- Realistic masses (Kroupa IMF)
- Spatial structure (Plummer)
- Velocities (virial equilibrium)

This creates self-consistent initial conditions ready for N-body integration!
:::

---

## Navigation

[← Part 3: Moments](03-part3-moments.md) | [Module 2a Home](00-part0-overview.md) | [Part 5: Summary →](05-part5-synthesis.md)