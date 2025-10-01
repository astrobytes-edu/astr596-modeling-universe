---
title: "Statistical Inference: From Photons to Cosmological Parameters"
subtitle: "Measuring the Universe with Uncertainty | Statistical Inference Module | ASTR 596"
---

:::{epigraph}
"The Universe is not only queerer than we suppose, but queerer than we can suppose. The only way to understand it is through the language of probability."
-- Paraphrased from J.B.S. Haldane, adapted for cosmology
:::

## Learning Outcomes

By the end of this module, you will be able to:

- [ ] **Map** astronomical observations (photons) to physical parameters (Ωₘ, h) through statistical inference
- [ ] **Explain** why correlations between measurements require covariance matrices and linear algebra
- [ ] **Construct** likelihood functions that encode your physics models
- [ ] **Translate** prior astronomical knowledge into probability distributions
- [ ] **Implement** the Metropolis-Hastings MCMC algorithm from scratch
- [ ] **Diagnose** convergence using trace plots, R̂ statistics, and effective sample size
- [ ] **Extract** cosmological parameters from Type Ia supernovae data
- [ ] **Interpret** the "banana-shaped" posterior as parameter degeneracy
- [ ] **Connect** every step to how the accelerating Universe was discovered

---

## Week 1: The Architecture of Astronomical Inference

### 1.1 The Fundamental Problem: What Are We Actually Measuring?

**Priority: 🔴 Essential**

:::{margin}
**Parameter (θ)**
The actual physical properties of astronomical objects that exist independent of our measurements. Examples: stellar mass M☉, distance d, dark energy density Ωₘ. These are what we want to know.

**Data (D)**
What we actually observe: photon counts, spectral line positions, apparent magnitudes. Always incomplete, always noisy, never directly measuring what we want.

**Model/Likelihood P(D|θ)**
The physics connecting parameters to observations. How a star of mass M produces a spectrum. How distance affects brightness. This IS your astrophysics knowledge.
:::

**Physical intuition**: You point a telescope at a supernova. Photons that traveled billions of years hit your CCD. Each photon carries information about the explosion, the intervening dust, and the expansion history of the Universe. Your job: extract the cosmological parameters from these messenger particles. This is statistical inference.

Every astronomical measurement is an inference problem. We never directly measure what we care about. We measure photons, and from those photons, we infer:
- How hot is that star? (from its color)
- How far is that galaxy? (from its brightness)
- What is the Universe made of? (from many supernovae)

:::{admonition} 🎯 The Conceptual Mapping
:class: important

**Statistical Concept → Astronomical Reality**

- **Parameters (θ)** → The Universe's actual properties
  - Ωₘ (matter density), h (Hubble constant), stellar masses, distances
  
- **Data (D)** → Your observations
  - CCD counts, spectra, light curves, always with noise
  
- **Likelihood P(D|θ)** → Your physics model
  - Stellar evolution, cosmological distances, dust extinction laws
  
- **Prior P(θ)** → Previous astronomical knowledge  
  - The IMF, the cosmic distance ladder, laboratory physics
  
- **Posterior P(θ|D)** → What you learn
  - Updated knowledge after incorporating new observations

This isn't abstract statistics—it's how we do astronomy!
:::

### 1.2 One Supernova, Many Uncertainties

**Priority: 🔴 Essential**

Let's start with ONE supernova observation to see why inference is necessary:

```python
# Real data from SN2011fe (the Pinwheel Galaxy supernova)
SN2011fe = {
    'redshift': 0.081,          # Spectroscopic measurement
    'magnitude': 16.89,          # Apparent brightness
    'magnitude_error': 0.07,     # Measurement uncertainty
    'host_galaxy': 'M101'        
}

# What we want to know:
# - How far away is it? (depends on cosmology!)
# - What's its intrinsic brightness? (varies by ~0.15 mag)
# - How much dust extinction? (unknown)
# - What does this tell us about Ωₘ and h?
```

**The chain of inference**:
1. **Observed brightness** → Photon statistics (Poisson noise)
2. **Intrinsic brightness** → SNe Ia have scatter (~0.15 mag even after standardization)
3. **Distance** → Depends on cosmological parameters (Ωₘ, h, ΩΛ)
4. **Dust** → Could dim the supernova (A_V uncertainty)
5. **Calibration** → Instrument response varies

**The revelation**: That simple magnitude measurement depends on stellar physics AND dust physics AND cosmology. We need to untangle all of this—that's statistical inference!

:::{important} 💡 What We Just Learned
A single supernova magnitude measurement is influenced by at least five different sources of uncertainty. Statistical inference is the mathematical framework for untangling these effects to extract the cosmological parameters we actually care about.
:::

---

### 1.3 Why Everything Depends on Everything: The Need for Linear Algebra

**Priority: 🔴 Essential**

:::{margin}
**Covariance**
Measure of how two variables change together. Positive means they increase together, negative means one increases when the other decreases. Off-diagonal elements in covariance matrix.

**Correlation**
Normalized covariance (-1 to +1). Tells you the strength of linear relationship. Nothing in astronomy is truly independent!

**Systematic Error**
Errors that affect multiple measurements in the same way. Example: All observations from the same night have correlated calibration errors.
:::

**Physical intuition**: Imagine you observe 10 supernovae with the same telescope on the same night. The weather changes slightly, affecting the seeing for ALL observations. Now their measurements aren't independent—they share systematic errors. This correlation must be tracked, or you'll underestimate your uncertainties.

### Nothing is Independent in Astronomy!

Real correlations you must handle:
- **Dust**: Affects both color (E(B-V)) AND brightness (A_V)
- **Metallicity**: Changes both stellar evolution AND supernova brightness
- **Calibration**: Affects ALL measurements from an instrument
- **Distance ladder**: Each rung's uncertainty propagates forward

### Building the Covariance Matrix

Start simple with 2 supernovae:

```python
import numpy as np

# Measurement uncertainties for 2 SNe
sigma1 = 0.07  # SN1 magnitude error
sigma2 = 0.09  # SN2 magnitude error

# Correlation from shared systematic (same instrument)
rho = 0.3  # 30% correlation

# Build 2×2 covariance matrix
C = np.array([[sigma1**2,        rho*sigma1*sigma2],
              [rho*sigma1*sigma2, sigma2**2]])

print("Covariance matrix:")
print(C)
print(f"\nDiagonal (individual variances): {np.diag(C)}")
print(f"Off-diagonal (correlation): {C[0,1]}")
```

Scale to the real problem (31 supernovae):
```python
# The JLA dataset covariance matrix
C_full = load_jla_covariance()  # 31×31 = 961 numbers!

# What's in this matrix?
# - Diagonal: Individual measurement uncertainties
# - Off-diagonal: Correlations from:
#   * Calibration systematics
#   * Dust modeling
#   * K-corrections
#   * Peculiar velocity uncertainties

# Why we need linear algebra:
C_inv = np.linalg.inv(C_full)  # Appears in χ²
eigenvals = np.linalg.eigvals(C_full)  # Independent modes
det_C = np.linalg.det(C_full)  # "Volume" of uncertainty
```

:::{admonition} 🔬 Why Linear Algebra Is Essential
:class: important

The likelihood with correlated errors:
$$\chi^2 = (\vec{d} - \vec{m})^T C^{-1} (\vec{d} - \vec{m})$$

Where:
- $\vec{d}$ = data vector (31 measurements)
- $\vec{m}$ = model vector (31 predictions)
- $C^{-1}$ = inverse covariance matrix

**Without linear algebra**: Can't handle correlations → Wrong uncertainties → Wrong cosmology!

**With linear algebra**: Proper error propagation → Correct parameter constraints → Nobel Prize!
:::

---

### 1.4 Bayesian Thinking as Astronomical Reasoning

**Priority: 🔴 Essential**

**Physical intuition**: The cosmic distance ladder isn't just a measurement technique—it's Bayesian inference in action! Each rung uses the posterior from the previous rung as its prior. This is how we bootstrap from parallax (parsecs) to the edge of the observable Universe (gigaparsecs).

### The Distance Ladder as Cascading Bayesian Inference

```python
# Step 1: Parallax (nearby stars)
def parallax_inference():
    prior = "Stars in Pleiades are at same distance"
    likelihood = "Gaia parallax measurements"  
    posterior = "Distance = 136.2 ± 1.2 pc"
    return posterior

# Step 2: Cepheids (nearby galaxies)
def cepheid_inference():
    prior = parallax_inference()  # Use Step 1 posterior!
    likelihood = "Period-Luminosity relation"
    posterior = "Distance to LMC = 49.97 ± 0.19 kpc"
    return posterior

# Step 3: SNe Ia (cosmological distances)
def supernova_inference():
    prior = cepheid_inference()  # Use Step 2 posterior!
    likelihood = "Phillips relation (brightness-decline)"
    posterior = "H₀ = 73.04 ± 1.04 km/s/Mpc"
    return posterior

# Each step's posterior becomes the next step's prior!
```

### Bayes' Theorem: The Engine of Learning

The fundamental equation of astronomical inference:

$$\boxed{P(\text{Universe}|\text{photons}) = \frac{P(\text{photons}|\text{Universe}) \times P(\text{Universe})}{P(\text{photons})}}$$

Breaking this down for our supernova cosmology:

$$\underbrace{P(\Omega_m, h | \text{SNe data})}_{\text{What we want}} \propto \overbrace{P(\text{SNe data} | \Omega_m, h)}^{\text{Physics model}} \times \overbrace{P(\Omega_m, h)}^{\text{Prior knowledge}}$$

### Real Astronomical Priors

Priors aren't arbitrary—they're accumulated astronomical knowledge:

```python
def cosmological_prior(omega_m, h):
    """Prior based on decades of astronomy"""
    
    # Physical constraints
    if omega_m < 0 or omega_m > 1:
        return 0  # Density fraction must be 0-1
    
    # Previous measurements
    if h < 0.5 or h > 0.9:
        return 0  # Decades of measurements constrain h
    
    # CMB constraints (if including)
    # Planck gives Ωₘ = 0.315 ± 0.007
    # Could use Gaussian prior if combining datasets
    
    return 1  # Uniform within bounds
```

:::{important} 💡 What We Just Learned
Bayesian inference isn't abstract philosophy—it's how astronomy works! The cosmic distance ladder, combining multiple datasets, and even simple error propagation are all Bayesian. Every astronomical measurement updates our knowledge: Prior × Likelihood = Posterior.
:::

---

### 1.5 The Curse of Dimensionality: Why We Need MCMC

**Priority: 🔴 Essential**

**Physical intuition**: Imagine trying to map dark matter in a galaxy cluster. Each galaxy has 6 phase space coordinates, there are 1000 galaxies, plus cluster parameters. That's ~6000 dimensions! Grid search would need 10^6000 evaluations—more than atoms in the Universe. MCMC solves this by exploring intelligently.

### Grid Search: Works for 2D, Fails for Reality

```python
import matplotlib.pyplot as plt

def grid_search_scaling(n_params, grid_points_per_param=100):
    """Show why grid search fails with dimensionality"""
    
    total_evaluations = grid_points_per_param ** n_params
    
    print(f"Parameters: {n_params}")
    print(f"Grid points per parameter: {grid_points_per_param}")
    print(f"Total evaluations needed: {total_evaluations:.2e}")
    
    # Time estimate (assume 1ms per evaluation)
    time_seconds = total_evaluations * 1e-3
    
    if time_seconds < 60:
        print(f"Time needed: {time_seconds:.1f} seconds")
    elif time_seconds < 3600:
        print(f"Time needed: {time_seconds/60:.1f} minutes")
    elif time_seconds < 86400:
        print(f"Time needed: {time_seconds/3600:.1f} hours")
    elif time_seconds < 31536000:
        print(f"Time needed: {time_seconds/86400:.1f} days")
    else:
        print(f"Time needed: {time_seconds/31536000:.2e} years")
    print("-" * 40)

# Our simple cosmology problem
grid_search_scaling(2)  # Just Ωₘ and h

# Real supernova problem  
grid_search_scaling(5)  # Add stretch, color, dust

# Full hierarchical model
grid_search_scaling(20)  # Population parameters

# Output:
# Parameters: 2 → 10,000 evaluations → 10 seconds ✓
# Parameters: 5 → 10 billion evaluations → 115 days ✗
# Parameters: 20 → 10^40 evaluations → 10^29 years ✗✗✗
```

### The MCMC Solution: Explore, Don't Enumerate

MCMC doesn't try to evaluate everywhere—it explores high-probability regions:

```python
# Pseudo-code for the key insight
def mcmc_exploration(n_steps=10000):
    """Explore parameter space intelligently"""
    
    current = random_start()
    chain = []
    
    for step in range(n_steps):
        # Propose new location
        proposed = current + small_random_jump()
        
        # Only evaluate probability RATIO (no normalization needed!)
        ratio = posterior(proposed) / posterior(current)
        
        # Accept/reject based on ratio
        if ratio > random():
            current = proposed
        
        chain.append(current)
    
    return chain

# 10,000 evaluations works for 2D or 2000D!
# The key: We explore where probability is high
```

---

## Week 2: MCMC—Exploring the Space of Possible Universes

### 2.1 The Metropolis Algorithm: Your First MCMC

**Priority: 🔴 Essential**

:::{margin}
**Detailed Balance**
The condition that ensures MCMC converges to the right distribution. Flow into any state equals flow out at equilibrium.

**Acceptance Ratio**
Probability of accepting a proposed move. Too high = not exploring. Too low = stuck. Goldilocks zone: 20-40%.

**Burn-in**
Initial samples discarded while chain "forgets" its starting point and finds high-probability regions.
:::

**Physical intuition**: Think of MCMC like a photon random-walking out of the Sun. It doesn't know the "right" direction, but by taking many random steps with probabilities determined by the local density, it eventually escapes. Similarly, MCMC random-walks through parameter space, spending more time in high-probability regions.

### The Complete Algorithm in 20 Lines

```python
import numpy as np

def metropolis_hastings(log_posterior, initial, n_steps=10000, step_size=0.1):
    """
    The Metropolis-Hastings algorithm - foundation of modern cosmology!
    
    Parameters:
    - log_posterior: Function that returns log(P(θ|D))
    - initial: Starting point in parameter space
    - n_steps: Number of MCMC steps
    - step_size: Standard deviation of proposal distribution
    """
    
    # Initialize
    current = initial
    chain = np.zeros((n_steps, len(initial)))
    n_accepted = 0
    
    for i in range(n_steps):
        # Propose new position (random walk)
        proposed = current + np.random.normal(0, step_size, size=len(current))
        
        # Calculate acceptance ratio (in log space for stability)
        log_ratio = log_posterior(proposed) - log_posterior(current)
        
        # Accept or reject
        if log_ratio > np.log(np.random.uniform()):
            current = proposed
            n_accepted += 1
        
        # Record current position
        chain[i] = current
    
    print(f"Acceptance rate: {n_accepted/n_steps:.2%}")
    return chain
```

### The Magic: Why This Works

The key insight—we only need the ratio:

$$\alpha = \min\left(1, \frac{P(\theta_{\text{new}}|D)}{P(\theta_{\text{current}}|D)}\right)$$

This means:
- ✅ No need for the normalizing constant P(D)
- ✅ Automatically spends more time in high-probability regions
- ✅ Guaranteed to converge to true posterior (given enough steps)

:::{admonition} 🎯 Implementation Best Practices
:class: important

1. **Always work in log space** to avoid numerical underflow
2. **Tune step size** for 20-40% acceptance rate
3. **Run multiple chains** from different starting points
4. **Remove burn-in** (typically first 10-20% of samples)
5. **Check convergence** before trusting results
:::

---

### 2.2 Building Your Cosmology Inference Engine

**Priority: 🔴 Essential**

Let's build the actual code for supernova cosmology:

```python
import numpy as np
from scipy.integrate import quad

class SupernovaCosmology:
    """Complete inference engine for SNe Ia cosmology"""
    
    def __init__(self, data_file='jla_mub.txt', cov_file='jla_mub_covmatrix.txt'):
        """Load JLA supernova data and covariance matrix"""
        
        # Load 31 supernovae: (redshift, distance_modulus, error)
        self.data = np.loadtxt(data_file)
        self.z = self.data[:, 0]
        self.mu_obs = self.data[:, 1]
        
        # Load 31×31 covariance matrix (includes systematics)
        self.cov = np.loadtxt(cov_file).reshape(31, 31)
        self.cov_inv = np.linalg.inv(self.cov)
        
        print(f"Loaded {len(self.z)} supernovae")
        print(f"Covariance matrix condition number: {np.linalg.cond(self.cov):.2e}")
    
    def luminosity_distance(self, z, omega_m, h):
        """
        Calculate luminosity distance using ΛCDM cosmology.
        Uses Pen (1999) fitting formula for speed.
        """
        # For flat universe: Ωₘ + ΩΛ = 1
        omega_lambda = 1 - omega_m
        
        # Pen's fitting formula (accurate to 0.4%)
        def eta(a, omega_m):
            s = ((1 - omega_m) / omega_m) ** (1/3)
            return 2 * np.sqrt(s**3 + 1) * (
                a**(-4) - 0.1540 * s * a**(-3) + 
                0.4304 * s**2 * a**(-2) + 0.19097 * s**3 * a**(-1) +
                0.066941 * s**4
            ) ** (-1/8)
        
        # Calculate luminosity distance
        d_L = (2997.92458 * (1 + z) / h) * (eta(1, omega_m) - eta(1/(1+z), omega_m))
        
        return d_L
    
    def distance_modulus_model(self, omega_m, h):
        """Calculate theoretical distance modulus for all supernovae"""
        
        mu_model = np.zeros(len(self.z))
        for i, z in enumerate(self.z):
            d_L = self.luminosity_distance(z, omega_m, h)
            mu_model[i] = 5 * np.log10(d_L) + 25
        
        return mu_model
    
    def log_likelihood(self, theta):
        """
        Calculate log-likelihood with full covariance matrix.
        This is where physics meets statistics!
        """
        omega_m, h = theta
        
        # Model prediction
        mu_model = self.distance_modulus_model(omega_m, h)
        
        # Residual
        residual = self.mu_obs - mu_model
        
        # Chi-squared with covariance
        chi2 = residual @ self.cov_inv @ residual
        
        return -0.5 * chi2
    
    def log_prior(self, theta):
        """Prior based on physical constraints"""
        omega_m, h = theta
        
        # Physical bounds
        if 0 < omega_m < 1 and 0.5 < h < 0.9:
            return 0.0  # Log of uniform prior
        else:
            return -np.inf  # Zero probability outside bounds
    
    def log_posterior(self, theta):
        """Posterior = Likelihood × Prior (in log space)"""
        
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta)

# Run the inference!
cosmo = SupernovaCosmology()

# Starting point (reasonable guess)
initial = np.array([0.3, 0.7])  # Ωₘ, h

# Run MCMC
chain = metropolis_hastings(
    cosmo.log_posterior,
    initial,
    n_steps=50000,
    step_size=0.01
)

# Remove burn-in (first 20%)
chain = chain[10000:]

# Results
omega_m_mean = np.mean(chain[:, 0])
omega_m_std = np.std(chain[:, 0])
h_mean = np.mean(chain[:, 1])
h_std = np.std(chain[:, 1])

print(f"\nResults:")
print(f"Ωₘ = {omega_m_mean:.3f} ± {omega_m_std:.3f}")
print(f"h = {h_mean:.3f} ± {h_std:.3f}")
print(f"Dark energy fraction: {(1-omega_m_mean)*100:.1f}%")
```

---

### 2.3 Convergence: Have We Explored Enough?

**Priority: 🔴 Essential**

**Physical intuition**: Convergence is like multiple astronomers independently measuring the same star. If they all get similar answers, you trust the result. If not, someone needs more data. MCMC chains are your "independent astronomers"—when they agree, you've converged.

### Visual Diagnostics: Pattern Recognition

```python
import matplotlib.pyplot as plt

def plot_convergence_diagnostics(chains, labels=['Ωₘ', 'h']):
    """
    Standard convergence plots for MCMC chains.
    This is what you'll see in every cosmology paper!
    """
    
    n_chains, n_steps, n_params = chains.shape
    
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 4*n_params))
    
    for i, label in enumerate(labels):
        # Trace plots (left column)
        ax_trace = axes[i, 0] if n_params > 1 else axes[0]
        for j in range(n_chains):
            ax_trace.plot(chains[j, :, i], alpha=0.7, linewidth=0.5)
        ax_trace.set_ylabel(label)
        ax_trace.set_xlabel('Step')
        ax_trace.set_title(f'Trace Plot - {label}')
        
        # Density plots (right column)
        ax_dens = axes[i, 1] if n_params > 1 else axes[1]
        for j in range(n_chains):
            ax_dens.hist(chains[j, :, i], bins=50, alpha=0.5, density=True)
        ax_dens.set_xlabel(label)
        ax_dens.set_ylabel('Density')
        ax_dens.set_title(f'Posterior - {label}')
    
    plt.tight_layout()
    return fig

# Run multiple chains
n_chains = 4
chains = np.zeros((n_chains, 50000, 2))

for i in range(n_chains):
    # Different random starting points
    initial = np.array([
        np.random.uniform(0.2, 0.4),  # Ωₘ
        np.random.uniform(0.65, 0.75)  # h
    ])
    
    chains[i] = metropolis_hastings(
        cosmo.log_posterior,
        initial,
        n_steps=50000,
        step_size=0.01
    )

# Plot diagnostics
plot_convergence_diagnostics(chains[:, 10000:])  # After burn-in
```

### Quantitative Convergence: The Gelman-Rubin Statistic

```python
def gelman_rubin(chains):
    """
    Calculate Gelman-Rubin R̂ statistic.
    R̂ < 1.01 indicates convergence.
    
    This is THE standard convergence metric in cosmology.
    """
    
    n_chains, n_steps, n_params = chains.shape
    
    R_hat = np.zeros(n_params)
    
    for param in range(n_params):
        # Within-chain variance
        W = np.mean([np.var(chains[c, :, param]) for c in range(n_chains)])
        
        # Between-chain variance
        chain_means = [np.mean(chains[c, :, param]) for c in range(n_chains)]
        B = n_steps * np.var(chain_means)
        
        # Pooled variance estimate
        V = ((n_steps - 1) * W + B) / n_steps
        
        # R̂ statistic
        R_hat[param] = np.sqrt(V / W)
    
    return R_hat

# Check convergence
R_hat = gelman_rubin(chains[:, 10000:])  # After burn-in
print(f"Gelman-Rubin R̂:")
print(f"  Ωₘ: {R_hat[0]:.4f}")
print(f"  h:  {R_hat[1]:.4f}")

if np.all(R_hat < 1.01):
    print("✓ Chains have converged!")
else:
    print("✗ Need more samples")
```

:::{important} 💡 What We Just Learned
Convergence isn't optional—it's essential! Visual inspection (trace plots) gives intuition, but quantitative metrics (R̂ < 1.01) give confidence. Multiple chains from different starts are like independent experiments confirming your result.
:::

---

### 2.4 The Degenerate Banana: Understanding Parameter Correlations

**Priority: 🔴 Essential**

**Physical intuition**: In cosmology, different combinations of parameters can produce nearly the same observations. Higher Ωₘ with lower h can give similar distances as lower Ωₘ with higher h. This degeneracy creates the famous "banana-shaped" posterior—a signature of cosmological inference!

### Visualizing the Cosmological Degeneracy

```python
import corner

def plot_cosmological_posterior(chain, truths=None):
    """
    Create a corner plot—the standard visualization in cosmology papers.
    Shows both marginal distributions and correlations.
    """
    
    # Labels with units
    labels = [r"$\Omega_m$", r"$h$"]
    
    # Create corner plot
    fig = corner.corner(
        chain,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.84],  # 1-sigma bounds
        plot_contours=True,
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        levels=[0.68, 0.95],  # 1-sigma and 2-sigma contours
        color='blue',
        truth_color='red',
        title_fmt='.3f'
    )
    
    # Add title
    fig.suptitle("Supernova Cosmology Constraints", fontsize=16)
    
    return fig

# Combine all chains after burn-in
combined_chain = np.vstack([chains[i, 10000:] for i in range(n_chains)])

# Plot the famous banana!
fig = plot_cosmological_posterior(
    combined_chain,
    truths=[0.315, 0.674]  # Planck values for comparison
)

# Calculate correlation
correlation = np.corrcoef(combined_chain.T)[0, 1]
print(f"\nParameter correlation: {correlation:.3f}")
print("Strong negative correlation creates the 'banana' shape!")
```

### Understanding the Banana

The diagonal degeneracy tells us physics:

```python
# Why the banana exists
def explain_degeneracy():
    """
    The cosmological degeneracy arises because:
    Distance ∝ (Ωₘ, h) in a degenerate way
    """
    
    # Similar distances from different parameters
    omega_m_1, h_1 = 0.25, 0.72  # Lower matter, higher Hubble
    omega_m_2, h_2 = 0.35, 0.68  # Higher matter, lower Hubble
    
    z_test = 0.5
    d_L_1 = cosmo.luminosity_distance(z_test, omega_m_1, h_1)
    d_L_2 = cosmo.luminosity_distance(z_test, omega_m_2, h_2)
    
    print(f"Parameters 1: Ωₘ={omega_m_1}, h={h_1} → d_L={d_L_1:.2f} Mpc")
    print(f"Parameters 2: Ωₘ={omega_m_2}, h={h_2} → d_L={d_L_2:.2f} Mpc")
    print(f"Difference: {abs(d_L_1-d_L_2)/d_L_1*100:.1f}%")
    print("\nDifferent cosmologies give nearly the same distance!")
    print("This is why we need many supernovae at different redshifts")

explain_degeneracy()
```

---

### 2.5 Advanced Topics: When Metropolis Isn't Enough

**Priority: 🟡 Important**

:::{margin}
**Hamiltonian Monte Carlo**
Uses gradient information to take larger, more efficient steps. Like throwing a ball on the posterior surface rather than random walking.

**Effective Sample Size**
Number of independent samples after accounting for autocorrelation. 1000 correlated samples might equal only 100 independent ones.

**Parallel Tempering**
Run chains at different "temperatures" to jump between modes in multimodal distributions.
:::

### Preview: Hamiltonian Monte Carlo

The future of MCMC—using gradients for efficiency:

```python
def hmc_concept():
    """
    HMC uses physics simulation for efficient sampling.
    Think of it as rolling a ball on the posterior landscape.
    """
    
    # Metropolis: Random walk (slow)
    # Steps are random, many rejections
    # Autocorrelation is high
    # Scales poorly with dimension
    
    # HMC: Hamiltonian dynamics (fast)
    # Uses gradients to guide exploration
    # Larger steps with high acceptance
    # Scales well to high dimensions
    
    comparison = """
    | Method      | 2D Problem | 100D Problem |
    |-------------|------------|--------------|
    | Metropolis  | 1,000      | 1,000,000    |
    | HMC         | 100        | 1,000        |
    
    Samples needed for same effective sample size
    """
    
    print(comparison)
    print("\nFor Project 4: Implement Metropolis first,")
    print("then try HMC for extra credit!")

hmc_concept()
```

---

## Module Synthesis: From Photons to Dark Energy

:::{admonition} 🎯 What You've Accomplished
:class: important

You've built the complete pipeline that discovered the accelerating Universe:

1. **Understood the inference problem**: From photons to parameters
2. **Handled correlations**: Full covariance matrix with linear algebra
3. **Encoded physics**: Likelihood = cosmological distance calculations
4. **Incorporated prior knowledge**: Reasonable parameter bounds
5. **Implemented MCMC**: Explored the parameter space efficiently
6. **Diagnosed convergence**: Multiple chains, R̂ statistic
7. **Interpreted the banana**: Understood parameter degeneracies
8. **Measured dark energy**: ~70% of the Universe!

**This isn't a toy problem**—this is exactly how the 2011 Nobel Prize in Physics was won!
:::

### The Big Picture Insights

1. **Parameters = Physical Reality**
   - We're not fitting abstract numbers
   - Ωₘ and h describe the actual Universe
   - Uncertainty quantification is part of the measurement

2. **Everything Affects Everything**
   - No measurement is independent
   - Covariance matrices capture correlations
   - Linear algebra is essential, not optional

3. **Priors Encode Knowledge**
   - Not arbitrary philosophical choices
   - Centuries of astronomical observations
   - The cosmic distance ladder is prior information

4. **MCMC Maps Possibility Space**
   - Not approximating—exploring
   - Each chain maps allowed universes
   - Convergence means thorough exploration

5. **The Universe is 70% Dark Energy**
   - Your analysis confirms this
   - With proper uncertainty quantification
   - Using the same methods as professional cosmologists

### Connections to Your Other Projects

- **Project 3 (Monte Carlo)**: Random sampling to solve problems
- **Project 4 (MCMC)**: This module directly prepares you!
- **Project 5 (Gaussian Processes)**: Another Bayesian inference method
- **All future research**: Every measurement needs uncertainty quantification

---

## Additional Resources and Extensions

### Debugging Checklist

When your MCMC isn't working:

1. **Check priors**: Are parameters within allowed bounds?
2. **Check likelihood**: Test with known parameters
3. **Check step size**: Acceptance rate should be 20-40%
4. **Check convergence**: R̂ < 1.01 for all parameters
5. **Check linear algebra**: Is covariance matrix invertible?

### Validation Tests

Before trusting results:

```python
def validation_suite():
    """Essential tests for any MCMC implementation"""
    
    tests = {
        "Recovery test": "Generate fake data, recover known parameters",
        "Gaussian test": "Sample from known Gaussian, check mean/covariance",
        "Prior test": "Sample with flat likelihood, recover prior",
        "Literature test": "Results roughly match published values"
    }
    
    return tests
```

### Going Further

- **Hierarchical models**: Population-level inference
- **Model comparison**: Bayes factors, evidence calculation
- **Advanced samplers**: NUTS, ensemble samplers
- **Real research**: Apply to your own data!

---

**Remember**: Every number in every astronomy paper comes with uncertainty. Now you know how those uncertainties are calculated, what they mean, and how to produce them yourself. You're not just learning statistics—you're learning how we measure the Universe!
