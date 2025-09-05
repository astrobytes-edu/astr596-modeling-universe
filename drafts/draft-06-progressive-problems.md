---
title: "Progressive Problems"
subtitle: "Statistical Foundations - How Nature Computes | ASTR 596: Modeling the Universe"
---

## Navigation

[← Part 5: Summary](05-part5-synthesis.md) | [Module 2a Home](00-part0-overview.md)

---

## Progressive Problem Set: From Statistics to Astrophysics

These problems progressively build your skills from basic statistical concepts to realistic astrophysical simulations. Each problem reinforces key concepts from the module while preparing you for your projects.

---

### Problem 1: Temperature from Velocity Distributions 🔴 Essential

**Concepts**: Temperature as distribution parameter, variance, Maxwell-Boltzmann

#### Part A: Basic Understanding

Given a collection of hydrogen atoms with velocities (in cm/s):

```python
velocities = np.array([3.2e5, -1.8e5, 4.1e5, -2.9e5, 1.5e5, -3.7e5, 2.8e5, -0.9e5, 3.5e5, -2.2e5])
```

1. Calculate the mean velocity. What does this tell you physically?
2. Calculate the velocity variance.
3. Using $\langle v_x^2 \rangle = k_B T / m_H$, determine the temperature.
4. Explain why you can't define temperature with just one velocity measurement.

#### Part B: Distribution Analysis

```python
def generate_maxwell_boltzmann(T, N=10000):
    """Generate N velocities from Maxwell-Boltzmann at temperature T"""
    m_H = 1.67e-24  # g
    k_B = 1.38e-16  # erg/K
    sigma = np.sqrt(k_B * T / m_H)
    return np.random.normal(0, sigma, N)
```

1. Generate distributions at T = 100 K, 1000 K, and 10000 K
2. Plot histograms of all three on the same axes (use transparency)
3. Calculate and compare the fraction of atoms with |v| > 10 km/s for each temperature
4. Verify that $\langle v_x^2 \rangle = k_B T / m_H$ holds for each distribution

#### Part C: Conceptual Challenge

A student claims: "I measured one hydrogen atom moving at 5000 m/s, so its temperature is 1000 K."

Write a paragraph explaining what's wrong with this statement, using concepts from the module.

---

### Problem 2: Central Limit Theorem in Action 🔴 Essential

**Concepts**: CLT, convergence to Gaussian, error scaling

#### Part A: Building Intuition

Start with a highly non-Gaussian distribution:

```python
def exponential_random(N):
    """Generate N samples from exponential distribution"""
    return -np.log(np.random.uniform(0, 1, N))
```

1. Generate 10,000 exponential random numbers and plot the histogram
2. Now create sums of n exponential random variables for n = 1, 2, 5, 10, 30, 100
3. Plot histograms of these sums (normalized). What do you observe?
4. For n=30, overlay a Gaussian with matching mean and variance. How good is the fit?

#### Part B: Astrophysical Application

A dust grain experiences random collisions from gas molecules.

```python
def simulate_dust_grain(n_collisions, collision_strength=1.0):
    """Simulate momentum kicks to a dust grain"""
    # Each collision imparts random momentum in random direction
    kicks = collision_strength * np.random.randn(n_collisions)
    return np.sum(kicks)
```

1. Simulate 1000 dust grains, each experiencing 10 collisions. Plot the distribution of final momenta.
2. Repeat for 100 and 1000 collisions per grain. How does the distribution change?
3. Verify that the width scales as $\sqrt{n}$ where n is the number of collisions
4. Explain why Brownian motion leads to Gaussian velocity distributions

#### Part C: Breaking the CLT

The CLT fails when distributions have infinite variance. Consider the Cauchy distribution:

```python
def cauchy_random(N):
    """Generate Cauchy-distributed random numbers"""
    return np.tan(np.pi * (np.random.uniform(0, 1, N) - 0.5))
```

1. Generate sums of 1, 10, 100, 1000 Cauchy random variables
2. Do the sums converge to a Gaussian? Why not?
3. Where might you encounter such "heavy-tailed" distributions in astrophysics?

---

### Problem 3: Maximum Entropy and Distribution Selection 🟡 Standard

**Concepts**: Maximum entropy, constraints, natural distributions

#### Part A: Deriving Distributions

Use the maximum entropy principle to derive distributions given constraints:

1. **No constraints** (except normalization): Show that maximum entropy gives uniform distribution
2. **Fixed mean energy**: Show this leads to exponential distribution
3. **Fixed mean and variance**: Show this leads to Gaussian distribution

#### Part B: Computational Verification (25 min)

```python
def entropy(samples, bins=50):
    """Calculate entropy of a distribution from samples"""
    hist, edges = np.histogram(samples, bins=bins, density=True)
    # Avoid log(0)
    hist = hist[hist > 0]
    dx = edges[1] - edges[0]
    return -np.sum(hist * np.log(hist) * dx)
```

1. Generate three distributions with the same mean (μ=5) and variance (σ²=4):
   - Gaussian
   - Uniform on appropriate interval
   - Two-point distribution (values chosen to match mean/variance)

2. Calculate the entropy of each. Which has maximum entropy?

3. Create a truncated exponential (exponential limited to [0, 10]). How does its entropy compare to the standard exponential with same mean?

#### Part C: Astrophysical Context

Explain using maximum entropy why:

1. Stellar velocities in a relaxed cluster follow Maxwell-Boltzmann
2. Photon energies from a blackbody follow Planck distribution
3. Why don't all distributions in nature maximize entropy?

---

### Problem 4: Monte Carlo Integration and Error Scaling 🔴 Essential

**Concepts**: Monte Carlo methods, error propagation, $1/\sqrt{N}$ scaling

#### Part A: Volume of a Hypersphere

Calculate the volume of a unit sphere in different dimensions using Monte Carlo:

```python
def monte_carlo_sphere_volume(dimension, n_samples):
    """Calculate volume of unit sphere in d dimensions"""
    # Generate random points in [-1,1]^d cube
    points = np.random.uniform(-1, 1, (n_samples, dimension))
    # Check if inside sphere
    distances = np.sum(points**2, axis=1)
    inside = distances <= 1.0
    # Volume = fraction_inside * cube_volume
    cube_volume = 2**dimension
    return np.sum(inside) / n_samples * cube_volume
```

1. Calculate volumes for d = 2, 3, 4, 5 dimensions with n = 10⁶ samples
2. Compare to analytical: $V_d = \pi^{d/2} / \Gamma(d/2 + 1)$
3. Plot error vs. n for n = 10², 10³, 10⁴, 10⁵, 10⁶ on log-log axes
4. Verify the $1/\sqrt{N}$ scaling. What's the slope on your log-log plot?

#### Part B: Stellar Luminosity Function

Estimate the total luminosity of a stellar population:

```python
def stellar_luminosity(mass):
    """Main sequence mass-luminosity relation"""
    if mass < 0.43:
        return 0.23 * mass**2.3
    elif mass < 2:
        return mass**4
    elif mass < 20:
        return 1.5 * mass**3.5
    else:
        return 3200 * mass  # Approximate for high mass
```

1. Sample 10,000 stellar masses from Salpeter IMF: $\xi(m) \propto m^{-2.35}$ for $m \in [0.5, 100] M_☉$
2. Calculate total luminosity and its Monte Carlo error
3. How many samples needed for 1% accuracy? 0.1% accuracy?
4. Compare efficiency to uniform grid integration

#### Part C: Importance Sampling

Instead of uniform sampling, sample more where the integrand is large:

1. Modify your luminosity calculation to sample more massive stars preferentially
2. Weight samples appropriately to get unbiased estimate
3. Compare convergence rate to uniform sampling
4. When is importance sampling most beneficial?

---

### Problem 5: Power Law Sampling and the IMF 🔴 Essential

**Concepts**: Inverse transform, power laws, Kroupa IMF

#### Part A: Single Power Law

Implement sampling from a single power law:

```python
def sample_power_law(alpha, m_min, m_max, N):
    """Sample N masses from power law p(m) ∝ m^(-alpha)"""
    # Your code here
    pass
```

1. Implement the inverse transform method (handle α=1 special case)
2. Test with Salpeter IMF (α=2.35) from 0.1 to 100 M☉
3. Verify your sampling by plotting histogram with theoretical curve
4. Calculate mean mass and compare to analytical expectation

#### Part B: Broken Power Law - Kroupa IMF (30 min)

The Kroupa IMF has three segments:

$$\xi(m) \propto \begin{cases}
m^{-0.3} & 0.01 < m/M_\odot < 0.08 \\
m^{-1.3} & 0.08 < m/M_\odot < 0.5 \\
m^{-2.3} & 0.5 < m/M_\odot < 150
\end{cases}$$

```python
def sample_kroupa_imf(N):
    """Sample N stellar masses from Kroupa IMF"""
    # Calculate relative probabilities of each segment
    # Choose segment based on these probabilities
    # Apply appropriate inverse transform
    # Your code here
    pass
```

1. Implement the full Kroupa IMF sampler
2. Generate 10,000 stars and plot the mass distribution
3. Calculate: total mass, number ratios in each segment, mean mass per segment
4. What fraction of total mass is in stars above 10 M☉?

#### Part C: Star Cluster Assembly
Build a realistic star cluster:

1. Sample masses until total reaches 10⁴ M☉
2. How many stars do you get? What's the most massive star?
3. Run 100 realizations. Plot distribution of maximum stellar mass
4. Explain why small clusters rarely have O-stars

---

### Problem 6: Correlation and Covariance in Stellar Systems 🟡 Standard

**Concepts**: Correlation, covariance, independence

#### Part A: Velocity Correlations
In a stellar stream, velocities are correlated:

```python
def generate_stream_velocities(N, correlation):
    """Generate correlated velocity components"""
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    return np.random.multivariate_normal(mean, cov, N)
```

1. Generate velocities with ρ = 0, 0.5, 0.9
2. Plot v_x vs v_y scatter plots for each
3. Calculate sample correlation and verify it matches input
4. How does correlation affect the velocity ellipsoid shape?

#### Part B: Breaking Independence
Compare independent vs. correlated systems:

1. **Independent**: Generate $v_x$ and $v_y$ independently from Gaussian$(0, σ)$
2. **Correlated**: Use multivariate Gaussian with correlation ρ
3. Calculate kinetic energy distribution for both
4. Show that mean energy is same but variance differs
5. Derive the relationship between energy variance and velocity correlation

#### Part C: Autocorrelation in Time Series
Stellar brightness varies with autocorrelation:

```python
def generate_correlated_timeseries(N, tau):
    """AR(1) process with correlation time tau"""
    x = np.zeros(N)
    x[0] = np.random.randn()
    for i in range(1, N):
        x[i] = np.exp(-1/tau) * x[i-1] + np.sqrt(1 - np.exp(-2/tau)) * np.random.randn()
    return x
```

1. Generate time series with τ = 1, 10, 100
2. Calculate and plot autocorrelation functions
3. Estimate correlation time from the data
4. How does this relate to effective sample size in MCMC?

---

### Problem 7: Building a Complete Star Cluster 🟢 Advanced

**Concepts**: Synthesis of all techniques

#### The Challenge
Create a self-consistent Plummer sphere star cluster:

```python
def create_star_cluster(M_total, a_plummer):
    """
    Create a realistic star cluster

    Parameters:
    -----------
    M_total : float
        Total cluster mass in solar masses
    a_plummer : float
        Plummer radius in parsecs

    Returns:
    --------
    Dictionary with masses, positions, velocities
    """
    # Your comprehensive implementation here
    pass
```

**Requirements:**
1. Sample stellar masses from Kroupa IMF until reaching M_total
2. Sample positions from Plummer sphere density profile
3. Assign velocities from Maxwell-Boltzmann with position-dependent dispersion
4. Ensure virial equilibrium: $2K + U = 0$
5. Add 10% binary fraction with appropriate orbital parameters

**Verify your cluster:**
1. Plot mass function - does it match Kroupa?
2. Plot density profile - does it match Plummer?
3. Check virial ratio
4. Plot velocity dispersion vs. radius
5. Calculate half-mass radius and crossing time

### Bonus Challenges:
- Add mass segregation (massive stars more centrally concentrated)
- Include primordial binaries with period distribution
- Add rotation to the cluster
- Implement anisotropic velocity dispersion

---

### Problem 8: Error Analysis Laboratory 🟡 Standard

**Concepts**: Error propagation, Monte Carlo errors, correlated uncertainties

#### Part A: Propagating Uncertainties
Given measurements with uncertainties:
- Stellar radius: R = 1.5 ± 0.1 R☉
- Temperature: T = 6000 ± 200 K

Calculate luminosity $L = 4\pi R^2 \sigma T^4$ with uncertainty:

1. Use error propagation formula (partial derivatives)
2. Use Monte Carlo sampling (10,000 samples)
3. Compare the two methods
4. Now add correlation: ρ(R,T) = 0.6. How does this change the error?

#### Part B: Bootstrap Error Estimation
Estimate uncertainty in power law fits:

```python
def bootstrap_power_law_fit(data, n_bootstrap=1000):
    """Estimate uncertainty in fitted power law exponent"""
    # Your implementation
    pass
```

1. Generate synthetic data from known power law with noise
2. Fit power law and estimate α
3. Use bootstrap to get confidence intervals
4. How does uncertainty depend on sample size?

### Part C: Systematic vs. Statistical Errors
Compare error sources in stellar mass estimates:

1. Statistical: Poisson noise in photon counts
2. Systematic: Uncertainty in mass-luminosity relation
3. Which dominates for bright stars? Faint stars?
4. How would you design observations to minimize total error?

---

## Solutions Guide

### Conceptual Checkpoints

After completing these problems, you should understand:

✅ Temperature characterizes distribution width, not individual particles
✅ CLT explains why we see Gaussians everywhere
✅ Maximum entropy gives least-biased distributions
✅ Monte Carlo errors scale as $1/\sqrt{N}$
✅ Inverse transform enables sampling from any analytical distribution
✅ Correlation affects error propagation and effective sample sizes
✅ Complex systems require synthesizing all these concepts

### Common Pitfalls to Avoid

⚠️ Don't confuse intensive $(T, P)$ with extensive $(E, M)$ properties
⚠️ Remember: correlation ≠ causation, and independence → zero correlation (but not vice versa)
⚠️ Power law sampling needs special treatment when α = 1
⚠️ Monte Carlo is inefficient in high dimensions without importance sampling
⚠️ Always check normalization when implementing probability distributions

### Connection to Projects

These problems directly prepare you for:
- **Project 2**: Problems 5 & 7 (N-body initial conditions)
- **Project 3**: Problem 4 (Monte Carlo radiative transfer)
- **Project 4**: Problem 6 (MCMC autocorrelation)
- **Project 5**: Problem 6 (GP covariance functions)
- **Final Project**: All concepts (neural network initialization and training)

---

## Navigation
[← Part 5: Summary](05-part5-synthesis.md) | [Module 2a Home](00-part0-overview.md)