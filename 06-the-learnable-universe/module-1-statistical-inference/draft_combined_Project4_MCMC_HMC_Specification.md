# Project 4: Building a Bayesian Inference Engine
## Measuring the Universe with Type Ia Supernovae

> *"The most incomprehensible thing about the universe is that it is comprehensible."*  
> — Albert Einstein
>
> *"All models are wrong, but some are useful."*  
> — George Box

---

## 🎯 Learning Objectives

By completing this project, you will:

1. **LO3**: Implement Bayesian inference algorithms from scratch (Metropolis-Hastings MCMC, Hamiltonian Monte Carlo)
2. **LO4**: Develop diagnostic tools for assessing convergence and sampling efficiency
3. **LO5**: Design professional-quality scientific software with modular architecture
4. **LO7**: Apply statistical methods to real astrophysical data (measuring cosmological parameters)
5. **LO9**: Connect computational methods to physical intuition (Hamiltonian dynamics → sampling)

**Timeline**: 3 weeks (~30-40 hours)  
**Deliverable**: Python package `bayesian_inference` with modular structure  
**AI Integration**: Phase 2 (strategic use after baseline implementation works)

---

## Part 1: The Detective Story — Why We're Here

### The 1998 Revolution

In 1998, two independent teams made one of the most shocking discoveries in the history of cosmology: **the universe is accelerating**. Not just expanding (we've known that since Hubble in 1929), but *speeding up* its expansion. This discovery earned Saul Perlmutter, Brian Schmidt, and Adam Riess the 2011 Nobel Prize in Physics.

Their tool? **Type Ia Supernovae** — the deaths of white dwarf stars that explode with remarkably consistent brightness. Because we know their intrinsic luminosity, we can measure how far away they are by how dim they appear. And because we can measure their redshift (how fast they're receding), we can map the expansion history of the universe.

> **Connection to Module 1**  
> This is measurement theory in action: **Data** (supernova brightness & redshift) + **Model** (cosmological distance-redshift relation) + **Uncertainty** (measurement errors) = **Inference** (cosmological parameters).

### The Inverse Problem

We observe ~700 supernovae at various distances (redshifts z). Each gives us a distance modulus μ (a logarithmic measure of distance). Our forward model predicts μ(z; Ωₘ, h, Ωᵥ) based on cosmological parameters:

- **Ωₘ**: Matter density (dark matter + baryons)
- **Ωᵥ**: Dark energy density (cosmological constant)  
- **h**: Hubble constant in units of 100 km/s/Mpc (H₀ = 100h)

The **inverse problem**: Given the data {(zᵢ, μᵢ)} with uncertainties, what are the parameters?

This is a classic **Bayesian inference** problem. We have:
- **Data**: 31 redshift bins with measured distance moduli
- **Unknowns**: 2-3 cosmological parameters
- **Goal**: Quantify our belief about the parameters given the data

### Why MCMC? The Curse of Dimensionality

We want the **posterior distribution** p(Ωₘ, h | data). This tells us not just the "best fit" values, but the full probability distribution — including uncertainties, correlations, and degeneracies.

In 2D-3D parameter spaces, we could just evaluate the posterior on a grid. But:
- 100 grid points per dimension
- 3 dimensions → 100³ = 1,000,000 evaluations
- Each evaluation requires computing D_L, matrix operations, exponentials...
- Add more parameters? Gets exponentially worse (hence "curse")

**Enter Markov Chain Monte Carlo**: Instead of exhaustively exploring parameter space, we *sample* from the posterior distribution. The magic: we get accurate estimates of means, variances, and credible intervals using only thousands of samples, not millions.

> **💡 Tip**  
> You've already done MCMC! In Project 3 (radiative transfer), every time a photon scattered, you drew a random direction from a distribution. That's sampling. MCMC does this in parameter space instead of physical space.

---

## Part 2: Bayesian Foundations — The Conceptual Framework

Your lectures will cover this in depth. Here's the essential framework for implementation.

### 🔴 Bayes' Theorem (Essential)

$$
p(\theta | D, M) = \frac{p(D | \theta, M) \cdot p(\theta | M)}{p(D | M)}
$$

In words:
$$
\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
$$

- **Posterior** p(θ|D,M): Our updated belief about parameters θ after seeing data D
- **Likelihood** p(D|θ,M): Probability of observing data given parameters (your forward model)
- **Prior** p(θ|M): Our belief about parameters before seeing data
- **Evidence** p(D|M): Normalizing constant (we'll ignore it — here's why)

> **💡 The Evidence: Why We Don't Care**  
> The evidence p(D|M) = ∫ p(D|θ,M) p(θ|M) dθ is a nightmare integral over all parameter space. But notice: it doesn't depend on θ. It's just a constant!
>
> For parameter inference, we only need the **unnormalized posterior**:
> $$
> p(\theta | D) \propto p(D | \theta) \cdot p(\theta)
> $$
>
> MCMC samples from this unnormalized distribution. The normalization happens automatically through the Markov chain dynamics.

### 🔴 The Likelihood Function (Essential)

For Gaussian errors with covariance matrix **C**, the likelihood is:

$$
\mathcal{L}(\theta) = p(D | \theta) = \frac{1}{\sqrt{(2\pi)^n \det \mathbf{C}}} \exp\left(-\frac{1}{2} \mathbf{r}^T \mathbf{C}^{-1} \mathbf{r}\right)
$$

where **r** is the residual vector: r_i = μᵢ^obs - μᵢ^th(zᵢ; θ).

In practice, we work with the **log-likelihood**:

$$
\ln \mathcal{L} = -\frac{1}{2} \mathbf{r}^T \mathbf{C}^{-1} \mathbf{r} - \frac{1}{2} \ln \det \mathbf{C} - \frac{n}{2} \ln(2\pi)
$$

The last two terms are constants (don't depend on θ), so for MCMC we only need:

$$
\ln \mathcal{L} = -\frac{1}{2} \sum_{i,j} r_i \, C^{-1}_{ij} \, r_j
$$

> **Connection to Module 1: Chi-Squared**  
> This is χ² = **r**^T **C**^(-1) **r** ! The Gaussian likelihood is just exp(-χ²/2). Maximum likelihood → minimum χ².

### 🟡 Markov Chains and Detailed Balance (Important)

A **Markov chain** is a sequence of states {θ₀, θ₁, θ₂, ...} where each state depends only on the previous state:

$$
p(\theta_{n+1} | \theta_n, \theta_{n-1}, ..., \theta_0) = p(\theta_{n+1} | \theta_n)
$$

This is a **memoryless random walk** through parameter space.

**Detailed balance** ensures the chain converges to the target distribution π(θ) (our posterior):

$$
\pi(\theta) \, T(\theta' | \theta) = \pi(\theta') \, T(\theta | \theta')
$$

where T(θ'|θ) is the **transition probability** (probability of moving from θ to θ').

In words: The rate of moving from θ → θ' equals the rate of moving from θ' → θ in equilibrium. This guarantees the chain "forgets" its starting point and converges to the stationary distribution π(θ).

> **⚠️ Connection to Module 1: Ergodicity**  
> Detailed balance + ergodicity (ability to reach any state from any other state) → **time averages equal ensemble averages**. After burn-in, the chain samples from p(θ|D).
>
> This is the same ergodic principle from statistical mechanics in Module 3!

---

## Part 3: The Forward Model — Cosmological Distance-Redshift Relation

Before we can do inference, we need to predict what we should observe given parameters.

### 🔴 Distance Modulus (Essential)

The **distance modulus** μ relates apparent magnitude m to absolute magnitude M:

$$
\mu = m - M = 5 \log_{10}\left(\frac{D_L}{\text{10 pc}}\right)
$$

For D_L in Mpc:

$$
\mu = 25 - 5 \log_{10} h + 5 \log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)
$$

where D_L^* ≡ D_L(h=1) is the luminosity distance computed with h factored out.

> **Note**  
> We're cheating slightly here. In reality, M and h are degenerate — we calibrate M from low-redshift SNe where we assume Hubble's law. For this project, we'll treat them as independent. This is addressed in real analyses with more sophisticated techniques.

### 🔴 Luminosity Distance (Essential)

The **luminosity distance** in a flat universe (Ω ≡ Ωₘ + Ωᵥ = 1) is:

$$
D_L(z) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m(1+z')^3 + \Omega_v}}
$$

This integral has no closed form, so we have two options:

#### Option 1: Numerical Integration (Recommended for learning)

Use `scipy.integrate.quad`:

```python
def comoving_distance_integral(z, Omega_m, Omega_v=None):
    """Integrand for comoving distance."""
    if Omega_v is None:
        Omega_v = 1.0 - Omega_m  # Flat universe
    return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + Omega_v)

def luminosity_distance(z, Omega_m, h, Omega_v=None):
    """Compute luminosity distance in Mpc."""
    from scipy.integrate import quad
    c_km_s = 299792.458  # Speed of light in km/s
    H0 = 100.0 * h  # Hubble constant in km/s/Mpc
    
    # Integrate from 0 to z
    integral, _ = quad(comoving_distance_integral, 0, z, args=(Omega_m, Omega_v))
    
    D_L = (c_km_s / H0) * (1 + z) * integral
    return D_L
```

#### Option 2: Pen's Fitting Formula (Faster, flat universe only)

From U.-L. Pen, ApJS, 120, 49 (1999) — accurate to <0.4% for 0.2 ≤ Ωₘ ≤ 1:

$$
D_L(z) = \frac{c}{H_0}(1+z)\left[\eta(1, \Omega_m) - \eta\left(\frac{1}{1+z}, \Omega_m\right)\right]
$$

where:

$$
\eta(a, \Omega_m) = 2\sqrt{s^3 + 1} \left[\frac{1}{a^4} - 0.1540\frac{s}{a^3} + 0.4304\frac{s^2}{a^2} + 0.19097\frac{s^3}{a} + 0.066941 s^4\right]^{-1/8}
$$

and $s^3 \equiv (1-\Omega_m)/\Omega_m$.

```python
def luminosity_distance_pen(z, Omega_m, h):
    """Pen's fitting formula for flat universe."""
    c_km_s = 299792.458
    H0 = 100.0 * h
    
    s3 = (1.0 - Omega_m) / Omega_m
    s = s3**(1.0/3.0)
    
    def eta(a, s):
        term = (1.0/a**4 - 0.1540*s/a**3 + 0.4304*s**2/a**2 + 
                0.19097*s**3/a + 0.066941*s**4)
        return 2.0 * np.sqrt(s**3 + 1) * term**(-1.0/8.0)
    
    a_today = 1.0
    a_then = 1.0 / (1.0 + z)
    
    D_L = (c_km_s / H0) * (1 + z) * (eta(a_today, s) - eta(a_then, s))
    return D_L
```

> **💡 Implementation Choice**  
> Start with Pen's formula (faster, fewer dependencies). If you extend to non-flat universes (extra credit), switch to numerical integration.

### 🔴 The Likelihood Implementation (Essential)

```python
def log_likelihood(theta, data_z, data_mu, cov_matrix_inv):
    """
    Compute log-likelihood for cosmological parameters.
    
    Parameters
    ----------
    theta : array-like
        Parameters [Omega_m, h] or [Omega_m, h, Omega_v]
    data_z : array
        Observed redshifts (31 bins)
    data_mu : array  
        Observed distance moduli (31 bins)
    cov_matrix_inv : array
        Inverse of 31x31 covariance matrix
    
    Returns
    -------
    float
        Log-likelihood value
    """
    Omega_m, h = theta[0], theta[1]
    Omega_v = 1.0 - Omega_m if len(theta) == 2 else theta[2]
    
    # Compute theoretical distance moduli
    mu_theory = np.zeros_like(data_mu)
    for i, z in enumerate(data_z):
        D_L = luminosity_distance(z, Omega_m, h, Omega_v)
        mu_theory[i] = 25.0 - 5.0*np.log10(h) + 5.0*np.log10(D_L)
    
    # Compute residuals
    residuals = data_mu - mu_theory
    
    # Chi-squared: r^T C^{-1} r
    chi2 = residuals @ cov_matrix_inv @ residuals
    
    return -0.5 * chi2
```

---

## Part 4: Metropolis-Hastings MCMC — The Core Algorithm

This is the heart of your Bayesian inference engine. You're building this from scratch.

### 🔴 The Algorithm (Essential)

The **Metropolis-Hastings algorithm** generates samples from the posterior distribution:

```
1. Start with initial parameters θ₀
2. For iteration i = 1, 2, 3, ... until convergence:
   a. Propose new parameters: θ* ~ Q(θ*|θᵢ₋₁)
   b. Compute acceptance probability:
      α = min(1, [p(θ*|D) / p(θᵢ₋₁|D)] × [Q(θᵢ₋₁|θ*) / Q(θ*|θᵢ₋₁)])
   c. Draw u ~ Uniform(0,1)
   d. If u < α:  accept θᵢ = θ*
      else:       reject θᵢ = θᵢ₋₁
3. After burn-in, use {θᵢ} as samples from p(θ|D)
```

**Key insight**: The ratio [Q(θᵢ₋₁|θ*) / Q(θ*|θᵢ₋₁)] is called the **Hastings ratio**. For symmetric proposals (like Gaussian random walk), Q(θ'|θ) = Q(θ|θ'), so this term = 1 and we get the simpler **Metropolis algorithm**.

> **⚠️ Why This Works (Mathematical Magic)**  
> This algorithm satisfies **detailed balance** with the posterior as the stationary distribution. The acceptance probability is chosen precisely so that:
>
> $$
> p(\theta|D) \cdot T(\theta'|\theta) = p(\theta'|D) \cdot T(\theta|\theta')
> $$
>
> This means the chain converges to p(θ|D) regardless of where you start! (After burn-in.)

### 🔴 Proposal Distributions (Essential)

The proposal distribution Q(θ*|θ) determines how you step through parameter space. Common choices:

#### Gaussian Random Walk (Recommended)

$$
\theta^* = \theta + \mathcal{N}(0, \Sigma)
$$

where Σ is the proposal covariance matrix.

```python
def propose_gaussian(theta_current, proposal_cov):
    """
    Propose new parameters from Gaussian random walk.
    
    Parameters
    ----------
    theta_current : array
        Current parameter values
    proposal_cov : array
        Proposal covariance matrix (ndim x ndim)
    
    Returns
    -------
    array
        Proposed parameter values
    """
    return np.random.multivariate_normal(theta_current, proposal_cov)
```

#### Uniform Top-Hat (Simpler, less efficient)

$$
\theta^* \sim \text{Uniform}(\theta - \Delta\theta/2, \theta + \Delta\theta/2)
$$

```python
def propose_uniform(theta_current, width):
    """
    Propose new parameters from uniform distribution.
    
    Parameters
    ----------
    theta_current : array
        Current parameter values
    width : float or array
        Width of uniform proposal (per parameter)
    
    Returns
    -------
    array
        Proposed parameter values
    """
    return theta_current + width * (np.random.rand(len(theta_current)) - 0.5)
```

> **💡 Tip**  
> Start with Gaussian. It's more efficient and generalizes better to correlated parameters.

### 🔴 Priors (Essential)

For this project, use **uniform (flat) priors** with physical bounds:

```python
def log_prior(theta):
    """
    Compute log-prior for cosmological parameters.
    
    Parameters
    ----------
    theta : array
        Parameters [Omega_m, h] or [Omega_m, h, Omega_v]
    
    Returns
    -------
    float
        Log-prior value (0 if in bounds, -inf if out of bounds)
    """
    Omega_m, h = theta[0], theta[1]
    
    # Physical bounds
    if Omega_m < 0 or Omega_m > 1:
        return -np.inf
    if h < 0 or h > 2:
        return -np.inf
    
    # For non-flat: check Omega_v
    if len(theta) == 3:
        Omega_v = theta[2]
        if Omega_v < 0 or Omega_v > 2:
            return -np.inf
    
    return 0.0  # log(1) = 0 for uniform prior
```

> **💡 Extra Credit: Informative Priors**  
> The Planck satellite measured H₀ = 67.4 ± 0.5 km/s/Mpc (so h = 0.674 ± 0.005). Add a Gaussian prior on h and see how it affects your constraints!
>
> **Question**: Does this resolve the "Hubble tension" with local measurements (h ~ 0.73)?

### 🔴 The Complete MCMC Loop (Essential)

Here's the skeleton for your main MCMC function:

```python
def run_mcmc(log_likelihood_func, log_prior_func, theta_init, 
             proposal_cov, n_steps, data):
    """
    Run Metropolis-Hastings MCMC.
    
    Parameters
    ----------
    log_likelihood_func : callable
        Function computing log-likelihood
    log_prior_func : callable
        Function computing log-prior
    theta_init : array
        Initial parameter values
    proposal_cov : array
        Proposal covariance matrix
    n_steps : int
        Number of MCMC steps
    data : dict
        Dictionary with 'z', 'mu', 'cov_inv'
    
    Returns
    -------
    chain : array (n_steps, n_params)
        MCMC chain samples
    acceptance_rate : float
        Fraction of proposals accepted
    log_post_chain : array (n_steps,)
        Log-posterior values
    """
    n_params = len(theta_init)
    chain = np.zeros((n_steps, n_params))
    log_post_chain = np.zeros(n_steps)
    n_accepted = 0
    
    # Initialize
    theta_current = theta_init.copy()
    log_prior_current = log_prior_func(theta_current)
    log_like_current = log_likelihood_func(theta_current, data['z'], 
                                           data['mu'], data['cov_inv'])
    log_post_current = log_prior_current + log_like_current
    
    for i in range(n_steps):
        # Propose new parameters
        theta_proposal = propose_gaussian(theta_current, proposal_cov)
        
        # Compute posterior for proposal
        log_prior_proposal = log_prior_func(theta_proposal)
        
        if not np.isfinite(log_prior_proposal):
            # Proposal outside prior bounds - reject immediately
            log_post_proposal = -np.inf
        else:
            log_like_proposal = log_likelihood_func(theta_proposal, data['z'],
                                                   data['mu'], data['cov_inv'])
            log_post_proposal = log_prior_proposal + log_like_proposal
        
        # Compute acceptance probability (in log space)
        log_alpha = log_post_proposal - log_post_current
        
        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            # Accept
            theta_current = theta_proposal
            log_post_current = log_post_proposal
            n_accepted += 1
        # else: reject, keep theta_current
        
        # Store sample
        chain[i] = theta_current
        log_post_chain[i] = log_post_current
    
    acceptance_rate = n_accepted / n_steps
    
    return chain, acceptance_rate, log_post_chain
```

> **⚠️ Implementation Detail: Log-Space Arithmetic**  
> **Always work in log-space!** Posterior values can be tiny (like 10^-300), which underflows to zero. But log-posteriors are manageable numbers.
>
> Acceptance criterion: α = p(θ*|D) / p(θ|D)  
> In log-space: log α = log p(θ*|D) - log p(θ|D)  
> Accept if: log(u) < log α

---

## Part 5: Diagnostics and Tuning — Making MCMC Work

Getting MCMC to work well is an art. Here's how to diagnose problems and tune your sampler.

### 🔴 Trace Plots (Essential)

Plot parameter values vs. iteration number:

```python
import matplotlib.pyplot as plt

def plot_trace(chain, param_names, burn_in=0):
    """
    Plot trace plots for all parameters.
    
    Parameters
    ----------
    chain : array (n_steps, n_params)
        MCMC chain
    param_names : list of str
        Parameter names for labels
    burn_in : int
        Number of burn-in steps to mark
    """
    n_params = chain.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params), sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(chain[:, i], alpha=0.7, lw=0.5)
        if burn_in > 0:
            ax.axvline(burn_in, color='red', ls='--', label='Burn-in')
        ax.set_ylabel(name, fontsize=12)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Iteration', fontsize=12)
    axes[0].legend()
    plt.tight_layout()
    return fig
```

**What to look for**:
- ✅ **Good**: "Fuzzy caterpillar" — no trends, just random fluctuations around mean
- ❌ **Bad**: Trends (not converged), stuck values (proposal too small), huge jumps (proposal too large)

### 🔴 Burn-In (Essential)

The chain needs time to "forget" its initial condition and converge to the stationary distribution. **Burn-in** is the initial portion you discard.

**How to determine burn-in**:
1. Run chain for N steps
2. Plot trace — where does it stabilize?
3. Discard first ~20-50% of steps
4. Check: Do different starting points converge to same distribution?

```python
def remove_burn_in(chain, burn_in_fraction=0.2):
    """
    Remove burn-in samples from chain.
    
    Parameters
    ----------
    chain : array (n_steps, n_params)
        MCMC chain
    burn_in_fraction : float
        Fraction of chain to discard (default: 0.2)
    
    Returns
    -------
    array
        Chain after burn-in
    """
    n_steps = chain.shape[0]
    burn_in_steps = int(n_steps * burn_in_fraction)
    return chain[burn_in_steps:]
```

### 🔴 Acceptance Rate (Essential)

The acceptance rate tells you if your proposal distribution is well-tuned.

**Rules of thumb**:
- **Too low** (< 10%): Proposals too large, mostly rejecting → chain stuck
- **Too high** (> 60%): Proposals too small, accepting everything → slow exploration
- **Optimal**: 20-40% for moderate-dimensional problems

```python
def tune_proposal_scale(log_likelihood_func, log_prior_func, theta_init,
                        base_cov, data, n_test_steps=1000):
    """
    Find optimal proposal scale by testing different widths.
    
    Parameters
    ----------
    ... (same as run_mcmc)
    base_cov : array
        Base covariance matrix (will be scaled)
    n_test_steps : int
        Number of steps for each test
    
    Returns
    -------
    best_scale : float
        Optimal scaling factor
    acceptance_rates : dict
        Acceptance rates for each scale tested
    """
    scales = np.logspace(-2, 1, 20)  # Test scales from 0.01 to 10
    acceptance_rates = {}
    
    for scale in scales:
        proposal_cov = scale**2 * base_cov
        _, acc_rate, _ = run_mcmc(log_likelihood_func, log_prior_func, 
                                  theta_init, proposal_cov, n_test_steps, data)
        acceptance_rates[scale] = acc_rate
    
    # Find scale with acceptance rate closest to 0.3 (30%)
    best_scale = min(scales, key=lambda s: abs(acceptance_rates[s] - 0.3))
    
    return best_scale, acceptance_rates
```

**Visualization**:
```python
def plot_acceptance_vs_scale(scales, acceptance_rates):
    """Plot acceptance rate vs proposal scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(list(scales), list(acceptance_rates.values()), 'o-')
    ax.axhline(0.234, color='green', ls='--', label='Optimal (23.4%)')
    ax.axhspan(0.2, 0.4, alpha=0.2, color='green', label='Good range')
    ax.set_xlabel('Proposal Scale', fontsize=12)
    ax.set_ylabel('Acceptance Rate', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()
    return fig
```

> **The More You Know: Optimal Acceptance Rate**  
> The theoretical optimal acceptance rate is 23.4% for infinite dimensions (Roberts & Rosenthal 2001). For 1-2 dimensions, ~44% is better. For your 2-parameter case, aim for 30-40%.
>
> **Intuition**: You want to explore efficiently. Too conservative (high acceptance) means tiny steps. Too aggressive (low acceptance) means wasting time rejecting. The sweet spot balances exploration vs. exploitation.

### 🟡 Autocorrelation and Effective Sample Size (Important)

MCMC samples are **correlated** — consecutive samples are not independent. The **autocorrelation** quantifies this.

**Autocorrelation function** for lag k:

$$
\rho_k = \frac{\text{Cov}(\theta_i, \theta_{i+k})}{\text{Var}(\theta)}
$$

```python
def autocorrelation(chain_1d, max_lag=None):
    """
    Compute autocorrelation function for a 1D chain.
    
    Parameters
    ----------
    chain_1d : array
        1D array of parameter samples
    max_lag : int, optional
        Maximum lag to compute (default: len(chain)//2)
    
    Returns
    -------
    acf : array
        Autocorrelation function
    """
    if max_lag is None:
        max_lag = len(chain_1d) // 2
    
    mean = np.mean(chain_1d)
    var = np.var(chain_1d)
    
    acf = np.zeros(max_lag)
    for k in range(max_lag):
        acf[k] = np.mean((chain_1d[:-k or None] - mean) * 
                         (chain_1d[k:] - mean)) / var
    
    return acf
```

**Integrated autocorrelation time** τ:

$$
\tau = 1 + 2\sum_{k=1}^{\infty} \rho_k
$$

This tells you how many steps between "independent" samples.

**Effective sample size**:

$$
N_{\text{eff}} = \frac{N}{\tau}
$$

If you run 10,000 steps but τ = 50, you only have 200 independent samples!

```python
def effective_sample_size(chain_1d):
    """Compute effective sample size."""
    acf = autocorrelation(chain_1d)
    
    # Sum until ACF becomes negligible (< 0.05)
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] < 0.05:
            break
        tau += 2.0 * acf[k]
    
    n_eff = len(chain_1d) / tau
    return n_eff, tau
```

> **💡 Tip**  
> If N_eff < 1000, run your chain longer! You need enough independent samples for reliable statistics.

### 🟡 Gelman-Rubin Convergence Diagnostic (Important)

**The problem**: How do you know your chain has converged?

**The solution**: Run multiple chains from different starting points. If they all converge to the same distribution, you're good!

The **Gelman-Rubin statistic** R-hat compares between-chain and within-chain variance:

$$
\hat{R} = \sqrt{\frac{\text{Var}_{\text{total}}}{\text{Var}_{\text{within}}}}
$$

- **R-hat ≈ 1.0**: Converged ✅
- **R-hat > 1.1**: Not converged ❌

```python
def gelman_rubin(chains):
    """
    Compute Gelman-Rubin R-hat statistic.
    
    Parameters
    ----------
    chains : list of arrays
        List of MCMC chains, each shape (n_steps, n_params)
        All chains must have same length and number of parameters
    
    Returns
    -------
    r_hat : array (n_params,)
        R-hat statistic for each parameter
    """
    n_chains = len(chains)
    n_steps, n_params = chains[0].shape
    
    # Compute chain means
    chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
    
    # Overall mean
    overall_mean = np.mean(chain_means, axis=0)
    
    # Between-chain variance
    B = n_steps / (n_chains - 1) * np.sum((chain_means - overall_mean)**2, axis=0)
    
    # Within-chain variance
    W = np.mean([np.var(chain, axis=0, ddof=1) for chain in chains], axis=0)
    
    # Variance estimate
    var_plus = (1 - 1/n_steps) * W + B / n_steps
    
    # R-hat
    r_hat = np.sqrt(var_plus / W)
    
    return r_hat
```

**Usage**:
```python
# Run 4 chains from different starting points
chains = []
for i in range(4):
    theta_init = [np.random.uniform(0.1, 0.5), np.random.uniform(0.5, 1.0)]
    chain, _, _ = run_mcmc(..., theta_init=theta_init, ...)
    chains.append(remove_burn_in(chain))

# Check convergence
r_hat = gelman_rubin(chains)
print(f"R-hat for Omega_m: {r_hat[0]:.4f}")
print(f"R-hat for h: {r_hat[1]:.4f}")
```

### 🔴 Corner Plots (Essential)

The **corner plot** (also called "triangle plot") shows all 1D and 2D marginal distributions. This is the standard visualization for MCMC results.

Use the excellent `corner` package: https://github.com/dfm/corner.py

```python
import corner

def plot_corner(chain, param_names, truths=None):
    """
    Create corner plot of posterior samples.
    
    Parameters
    ----------
    chain : array (n_samples, n_params)
        MCMC samples after burn-in
    param_names : list of str
        Parameter names for labels
    truths : array, optional
        True parameter values to mark
    
    Returns
    -------
    figure
    """
    fig = corner.corner(
        chain,
        labels=param_names,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],  # 1-sigma credible intervals
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        smooth=1.0
    )
    return fig
```

**Interpreting corner plots**:
- **Diagonal**: 1D marginal distributions (posterior for each parameter)
- **Off-diagonal**: 2D joint distributions (correlations/degeneracies)
- **Contours**: 1σ and 2σ credible regions
- **Elliptical contours**: Parameters are correlated (e.g., Ωₘ and h have degeneracy in distance predictions)

> **Connection to Module 1: Marginalization**  
> The 1D histograms are **marginal distributions** — integrating out all other parameters:
>
> $$
> p(\Omega_m | D) = \int p(\Omega_m, h | D) \, dh
> $$
>
> MCMC does this marginalization automatically! Just histogram one parameter and ignore the others.

---

## Part 6: Hamiltonian Monte Carlo — Physics-Inspired Sampling

> **⚠️ Connection to Project 2 & Module 3**  
> **You've already built the key component!** The leapfrog integrator from Project 2 (N-body) is exactly what HMC uses. The only difference: instead of integrating real particles in real space, you're integrating fictitious particles in *parameter space*.
>
> This is Hamilton's equations from Module 3 in action!

### 🟡 Why HMC? The Problem with Random Walk MCMC (Important)

Metropolis-Hastings is a **random walk** — it diffuses through parameter space. This is inefficient in high dimensions:

- Takes O(N²) steps to traverse N-dimensional space
- Gets stuck in narrow valleys (strong correlations)
- Wastes time with rejected proposals

**HMC** uses gradient information to make *informed* proposals. Instead of random walk, it *flows along contours of constant posterior probability*.

**Intuition**: Imagine the negative log-posterior as a potential energy landscape. Drop a ball (with momentum) and let it roll along the valley. It efficiently explores the distribution!

### 🟡 The Physical Analogy (Important)

Define a **Hamiltonian** (total energy):

$$
H(\theta, p) = U(\theta) + K(p)
$$

where:
- **θ**: Position in parameter space (e.g., Ωₘ, h)
- **p**: Fictitious momentum variables
- **U(θ) = -log p(θ|D)**: Potential energy (negative log-posterior)
- **K(p) = p²/(2m)**: Kinetic energy (momentum)

**Hamilton's equations** (from Module 3!):

$$
\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = \frac{p}{m}
$$

$$
\frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = -\frac{\partial U}{\partial \theta}
$$

**Key property**: H is conserved along trajectories (energy conservation).

This means:
- Proposals explore level sets of H
- High acceptance rate (because H(θ*, p*) ≈ H(θ, p))
- Efficient exploration (gradient information guides direction)

> **⚠️ The Profound Insight**  
> Classical mechanics → statistical inference!
>
> The same math that describes planetary orbits also describes how to sample from probability distributions. Hamilton didn't know his equations would power Bayesian inference 200 years later.
>
> **This is why Module 3 matters.** Phase space isn't just for stars — it's the foundation of modern computational statistics.

### 🔴 The Leapfrog Integrator (Essential — You Already Have This!)

The **leapfrog algorithm** is a symplectic integrator that preserves the Hamiltonian structure:

```python
def leapfrog_step(theta, p, grad_U, epsilon, m=1.0):
    """
    Single leapfrog step for Hamiltonian dynamics.
    
    Parameters
    ----------
    theta : array
        Current position (parameters)
    p : array
        Current momentum
    grad_U : callable
        Function that returns gradient of potential ∂U/∂θ
    epsilon : float
        Step size
    m : float
        Mass (usually 1.0)
    
    Returns
    -------
    theta_new : array
        Updated position
    p_new : array
        Updated momentum
    """
    # Half step in momentum
    p_half = p - 0.5 * epsilon * grad_U(theta)
    
    # Full step in position
    theta_new = theta + epsilon * p_half / m
    
    # Half step in momentum
    p_new = p_half - 0.5 * epsilon * grad_U(theta_new)
    
    return theta_new, p_new

def leapfrog_trajectory(theta_init, p_init, grad_U, epsilon, n_steps, m=1.0):
    """
    Integrate full trajectory with multiple leapfrog steps.
    
    Parameters
    ----------
    theta_init : array
        Initial position
    p_init : array
        Initial momentum
    grad_U : callable
        Gradient of potential
    epsilon : float
        Step size
    n_steps : int
        Number of leapfrog steps
    m : float
        Mass
    
    Returns
    -------
    theta_final : array
        Final position after n_steps
    p_final : array
        Final momentum
    """
    theta = theta_init.copy()
    p = p_init.copy()
    
    for _ in range(n_steps):
        theta, p = leapfrog_step(theta, p, grad_U, epsilon, m)
    
    return theta, p
```

> **💡 Connection to Project 2**  
> **This is identical to your N-body integrator!** Same leapfrog structure:
> 1. Half-kick in velocity
> 2. Full drift in position  
> 3. Half-kick in velocity
>
> The only change: forces F → gradients ∂U/∂θ.
>
> **Reuse your code!** Just adapt the force calculation.

### 🔴 Computing Gradients (Essential)

You need ∂U/∂θ where U(θ) = -log p(θ|D).

For Gaussian likelihood with flat prior:

$$
U(\theta) = \frac{1}{2} \mathbf{r}^T \mathbf{C}^{-1} \mathbf{r}
$$

$$
\frac{\partial U}{\partial \theta_i} = \sum_{j,k} \frac{\partial r_j}{\partial \theta_i} C^{-1}_{jk} r_k
$$

where $r_j = \mu_j^{\text{obs}} - \mu_j^{\text{th}}(\theta)$.

**The tricky part**: Computing $\partial \mu^{\text{th}}/\partial \theta_i$ requires differentiating D_L(z; θ).

**Two approaches**:

#### Option 1: Numerical Derivatives (Simple)

```python
def grad_U_numerical(theta, data, epsilon=1e-5):
    """
    Compute gradient of U using finite differences.
    
    Parameters
    ----------
    theta : array
        Parameters [Omega_m, h]
    data : dict
        Data dictionary
    epsilon : float
        Finite difference step size
    
    Returns
    -------
    grad : array
        Gradient ∂U/∂θ
    """
    n_params = len(theta)
    grad = np.zeros(n_params)
    
    U_current = -log_likelihood(theta, data['z'], data['mu'], data['cov_inv'])
    
    for i in range(n_params):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        U_plus = -log_likelihood(theta_plus, data['z'], data['mu'], data['cov_inv'])
        
        grad[i] = (U_plus - U_current) / epsilon
    
    return grad
```

#### Option 2: Automatic Differentiation (Professional)

Use `jax` for automatic differentiation (preview of Final Project!):

```python
import jax
import jax.numpy as jnp

# Define likelihood in JAX
@jax.jit
def U_jax(theta, z, mu_obs, cov_inv):
    """Potential energy U = -log p(θ|D) in JAX."""
    Omega_m, h = theta[0], theta[1]
    
    # Compute D_L for all redshifts (vectorized)
    D_L = luminosity_distance_jax(z, Omega_m, h)
    mu_theory = 25.0 - 5.0*jnp.log10(h) + 5.0*jnp.log10(D_L)
    
    residuals = mu_obs - mu_theory
    chi2 = residuals @ cov_inv @ residuals
    
    return 0.5 * chi2

# Gradient is automatic!
grad_U_jax = jax.grad(U_jax)
```

> **💡 Tip**  
> Start with numerical derivatives (easier to debug). Switch to JAX if you have time — it's 100x faster and teaches you tools for the Final Project!

### 🔴 The HMC Algorithm (Essential)

```python
def run_hmc(U_func, grad_U_func, theta_init, epsilon, L, n_samples, m=1.0):
    """
    Run Hamiltonian Monte Carlo.
    
    Parameters
    ----------
    U_func : callable
        Potential energy function U(θ) = -log p(θ|D)
    grad_U_func : callable
        Gradient of potential ∂U/∂θ
    theta_init : array
        Initial parameters
    epsilon : float
        Leapfrog step size
    L : int
        Number of leapfrog steps per proposal
    n_samples : int
        Number of HMC samples to generate
    m : float
        Mass (default 1.0)
    
    Returns
    -------
    chain : array (n_samples, n_params)
        HMC samples
    acceptance_rate : float
        Acceptance rate
    """
    n_params = len(theta_init)
    chain = np.zeros((n_samples, n_params))
    n_accepted = 0
    
    theta_current = theta_init.copy()
    U_current = U_func(theta_current)
    
    for i in range(n_samples):
        # Sample momentum from standard normal
        p_current = np.random.randn(n_params) * np.sqrt(m)
        K_current = 0.5 * np.sum(p_current**2) / m
        H_current = U_current + K_current
        
        # Leapfrog integration
        theta_proposal, p_proposal = leapfrog_trajectory(
            theta_current, p_current, grad_U_func, epsilon, L, m
        )
        
        # Evaluate Hamiltonian at proposal
        U_proposal = U_func(theta_proposal)
        K_proposal = 0.5 * np.sum(p_proposal**2) / m
        H_proposal = U_proposal + K_proposal
        
        # Metropolis acceptance (H should be conserved)
        log_alpha = -(H_proposal - H_current)  # Note: H = -log p, so flip sign
        
        if np.log(np.random.rand()) < log_alpha:
            # Accept
            theta_current = theta_proposal
            U_current = U_proposal
            n_accepted += 1
        # else: reject, keep theta_current
        
        chain[i] = theta_current
    
    acceptance_rate = n_accepted / n_samples
    
    return chain, acceptance_rate
```

> **⚠️ Tuning HMC Parameters**  
> HMC has two hyperparameters to tune:
>
> 1. **ε (epsilon)**: Step size for leapfrog
>    - Too large: Poor energy conservation, low acceptance
>    - Too small: Waste computation, barely move
>    - **Rule of thumb**: Tune for ~65% acceptance (higher than MCMC!)
>
> 2. **L**: Number of leapfrog steps per sample
>    - Too few: Behaves like MCMC (random walk)
>    - Too many: Waste computation (retracing steps)
>    - **Rule of thumb**: L ~ diameter of posterior / ε
>
> **Strategy**: Run short test chains, plot energy violations |ΔH|, adjust ε to keep |ΔH| < 1.

### 🟡 Comparing MCMC vs HMC (Important)

After implementing both methods, you should compare:

1. **Effective sample size** per unit time
2. **Autocorrelation length** (should be shorter for HMC)
3. **Convergence speed** (plot posterior means vs. iteration)
4. **Computational cost** per sample

```python
def compare_samplers(mcmc_chain, hmc_chain, param_names):
    """
    Compare MCMC and HMC performance.
    
    Parameters
    ----------
    mcmc_chain : array
        MCMC samples (after burn-in)
    hmc_chain : array
        HMC samples (after burn-in)
    param_names : list of str
        Parameter names
    
    Returns
    -------
    comparison : dict
        Dictionary with performance metrics
    """
    comparison = {}
    
    for i, name in enumerate(param_names):
        # Effective sample sizes
        mcmc_ess, mcmc_tau = effective_sample_size(mcmc_chain[:, i])
        hmc_ess, hmc_tau = effective_sample_size(hmc_chain[:, i])
        
        comparison[name] = {
            'MCMC_ESS': mcmc_ess,
            'MCMC_tau': mcmc_tau,
            'HMC_ESS': hmc_ess,
            'HMC_tau': hmc_tau,
            'ESS_ratio': hmc_ess / mcmc_ess,
            'tau_ratio': mcmc_tau / hmc_tau
        }
    
    return comparison
```

**Expected results**:
- HMC should have **higher ESS** (more independent samples)
- HMC should have **lower τ** (faster decorrelation)
- HMC is **more expensive per sample** (gradient + L leapfrog steps)
- Net result: HMC is typically **2-10x more efficient** for moderate dimensions

> **The More You Know: When HMC Shines**  
> HMC is most beneficial when:
> - Parameters are strongly correlated (elongated posteriors)
> - Dimensionality is moderate (d ~ 10-100)
> - Gradients are cheap to compute
>
> HMC struggles when:
> - Likelihood is stochastic or noisy (breaks energy conservation)
> - Posterior is multimodal (gets trapped in one mode)
> - Gradients are expensive (e.g., black-box simulations)
>
> For your 2D cosmology problem, HMC is overkill but pedagogically valuable!

---

## Part 7: Code Structure and Professional Practice

Your deliverable is a **Python package** `bayesian_inference` with clean, modular design.

### 🔴 Required Package Structure (Essential)

```
bayesian_inference/
│
├── README.md                 # Package description, installation, usage
├── requirements.txt          # Dependencies (numpy, scipy, matplotlib, corner)
├── setup.py                  # Installation script
│
├── bayesian_inference/       # Main package directory
│   ├── __init__.py          # Package initialization
│   │
│   ├── cosmology.py         # Cosmological model
│   │   ├── luminosity_distance()
│   │   ├── distance_modulus()
│   │   └── load_jla_data()
│   │
│   ├── likelihood.py        # Likelihood and prior functions
│   │   ├── log_likelihood()
│   │   ├── log_prior()
│   │   └── log_posterior()
│   │
│   ├── mcmc.py              # MCMC sampler
│   │   ├── propose_gaussian()
│   │   ├── propose_uniform()
│   │   └── run_mcmc()
│   │
│   ├── hmc.py               # HMC sampler (25% of grade)
│   │   ├── leapfrog_step()
│   │   ├── leapfrog_trajectory()
│   │   └── run_hmc()
│   │
│   ├── diagnostics.py       # Convergence diagnostics
│   │   ├── autocorrelation()
│   │   ├── effective_sample_size()
│   │   ├── gelman_rubin()
│   │   └── plot_trace()
│   │
│   └── visualization.py     # Plotting utilities
│       ├── plot_corner()
│       ├── plot_acceptance_vs_scale()
│       └── plot_comparison()
│
├── tests/                   # Unit tests (optional but encouraged!)
│   ├── test_cosmology.py
│   ├── test_mcmc.py
│   └── test_hmc.py
│
├── scripts/                 # Example usage scripts
│   ├── run_mcmc.py          # Run MCMC on SNe data
│   ├── run_hmc.py           # Run HMC on SNe data
│   └── compare_samplers.py  # Compare MCMC vs HMC
│
└── results/                 # Output directory for plots and chains
    ├── chains/              # Saved MCMC/HMC chains
    ├── figures/             # Generated plots
    └── analysis.txt         # Summary statistics
```

### 🔴 Example Usage Script (Essential)

Your package should be usable like this:

```python
#!/usr/bin/env python
"""
Example: Run MCMC on JLA supernova data.
"""
import numpy as np
from bayesian_inference import cosmology, likelihood, mcmc, diagnostics, visualization

# Load data
data = cosmology.load_jla_data('jla_mub.txt', 'jla_mub_covmatrix.txt')

# Initial parameters: [Omega_m, h]
theta_init = np.array([0.3, 0.7])

# Proposal covariance (start with scaled identity)
proposal_cov = np.diag([0.01, 0.01])**2

# Run MCMC
print("Running MCMC...")
chain, acc_rate, log_post = mcmc.run_mcmc(
    log_likelihood_func=likelihood.log_likelihood,
    log_prior_func=likelihood.log_prior,
    theta_init=theta_init,
    proposal_cov=proposal_cov,
    n_steps=50000,
    data=data
)

print(f"Acceptance rate: {acc_rate:.3f}")

# Remove burn-in
chain_burned = diagnostics.remove_burn_in(chain, burn_in_fraction=0.2)

# Diagnostics
diagnostics.plot_trace(chain, ['Omega_m', 'h'], burn_in=10000)

# Results
visualization.plot_corner(chain_burned, ['$\\Omega_m$', '$h$'])

# Summary statistics
means = np.mean(chain_burned, axis=0)
stds = np.std(chain_burned, axis=0)

print("\nResults:")
print(f"Omega_m = {means[0]:.4f} ± {stds[0]:.4f}")
print(f"h = {means[1]:.4f} ± {stds[1]:.4f}")

# Check convergence
ess_om, tau_om = diagnostics.effective_sample_size(chain_burned[:, 0])
ess_h, tau_h = diagnostics.effective_sample_size(chain_burned[:, 1])

print(f"\nEffective sample sizes:")
print(f"Omega_m: {ess_om:.0f} (tau = {tau_om:.1f})")
print(f"h: {ess_h:.0f} (tau = {tau_h:.1f})")
```

### 🔴 Documentation Standards (Essential)

Every function must have a **NumPy-style docstring**:

```python
def luminosity_distance(z, Omega_m, h, Omega_v=None):
    """
    Compute luminosity distance for given cosmology.
    
    Uses Pen's fitting formula for flat universes, numerical integration
    for non-flat universes.
    
    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate D_L
    Omega_m : float
        Matter density parameter (0 < Omega_m < 1)
    h : float
        Reduced Hubble constant (H_0 = 100h km/s/Mpc)
    Omega_v : float, optional
        Dark energy density parameter. If None, assumes flat universe
        (Omega_v = 1 - Omega_m)
    
    Returns
    -------
    D_L : float or array-like
        Luminosity distance in Mpc
    
    Examples
    --------
    >>> D_L = luminosity_distance(1.0, 0.3, 0.7)
    >>> print(f"D_L(z=1) = {D_L:.1f} Mpc")
    D_L(z=1) = 6627.4 Mpc
    
    Notes
    -----
    Accurate to 0.4% for 0.2 ≤ Omega_m ≤ 1.0 in flat universes [1]_.
    
    References
    ----------
    .. [1] Pen, U.-L. 1999, ApJS, 120, 49
    """
    # Implementation here...
```

### 🟡 Testing (Important but Optional)

If you have time, write unit tests:

```python
# tests/test_cosmology.py
import numpy as np
from bayesian_inference.cosmology import luminosity_distance

def test_luminosity_distance_low_z():
    """Test D_L at low redshift (should approach Hubble law)."""
    z = 0.01
    Omega_m = 0.3
    h = 0.7
    
    D_L = luminosity_distance(z, Omega_m, h)
    
    # At low z: D_L ≈ (c/H_0) * z = 3000h^-1 * z Mpc
    D_L_expected = 3000.0 / h * z
    
    assert np.abs(D_L - D_L_expected) / D_L_expected < 0.01, \
        f"D_L = {D_L:.1f} Mpc, expected {D_L_expected:.1f} Mpc"

def test_luminosity_distance_flat_vs_numerical():
    """Test Pen's formula against numerical integration."""
    z = np.linspace(0.1, 1.0, 10)
    Omega_m = 0.3
    h = 0.7
    
    D_L_pen = luminosity_distance(z, Omega_m, h, method='pen')
    D_L_num = luminosity_distance(z, Omega_m, h, method='numerical')
    
    rel_error = np.abs(D_L_pen - D_L_num) / D_L_num
    
    assert np.all(rel_error < 0.004), \
        f"Max relative error: {np.max(rel_error):.4f} (should be < 0.4%)"
```

Run tests with: `python -m pytest tests/`

---

## Part 8: Deliverables and Assessment

### 🔴 What to Submit (Essential)

1. **GitHub repository** with your `bayesian_inference` package
   - Follow the directory structure above
   - Include `README.md` with installation and usage instructions
   - Include `requirements.txt`

2. **Required Plots** (save to `results/figures/`):
   - `trace_mcmc.png`: Trace plots for both parameters (MCMC)
   - `trace_hmc.png`: Trace plots (HMC)
   - `corner_mcmc.png`: Corner plot (MCMC)
   - `corner_hmc.png`: Corner plot (HMC)
   - `acceptance_vs_scale.png`: Tuning curve
   - `comparison.png`: Side-by-side comparison of MCMC vs HMC posteriors
   - `autocorrelation.png`: ACF for both samplers

3. **Analysis Report** (`results/analysis.txt`):
   - Parameter estimates with uncertainties (both samplers)
   - Acceptance rates
   - Effective sample sizes and autocorrelation times
   - Comparison of MCMC vs HMC efficiency
   - Interpretation: What do your results say about our Universe?

4. **Example Scripts** (`scripts/`):
   - `run_mcmc.py`: Complete MCMC analysis
   - `run_hmc.py`: Complete HMC analysis
   - `compare_samplers.py`: Performance comparison

### 🔴 Assessment Rubric (Essential)

**Total: 100 points**

| Component | Points | Criteria |
|-----------|--------|----------|
| **MCMC Implementation** | **40** | |
| - Core algorithm | 15 | Correct Metropolis-Hastings, proper acceptance probability |
| - Proposal distributions | 5 | Both Gaussian and uniform implemented |
| - Likelihood/Prior | 10 | Correct log-likelihood with covariance matrix |
| - Convergence | 10 | Reasonable burn-in, tuned proposals, converged chains |
| **HMC Implementation** | **25** | |
| - Leapfrog integrator | 10 | Correct symplectic integration (can adapt from Project 2) |
| - Gradient computation | 8 | Numerical or automatic differentiation |
| - Full HMC algorithm | 7 | Correct momentum sampling and acceptance |
| **Diagnostics** | **15** | |
| - Trace plots | 3 | Clear visualization, burn-in marked |
| - Acceptance rate | 3 | Computed and reported, tuning demonstrated |
| - Autocorrelation | 4 | ACF computed, ESS calculated |
| - Convergence tests | 5 | Gelman-Rubin or equivalent |
| **Visualization** | **10** | |
| - Corner plots | 5 | Proper marginals, contours, labels |
| - Comparison plots | 5 | MCMC vs HMC clearly compared |
| **Code Quality** | **10** | |
| - Structure | 3 | Modular design, proper package organization |
| - Documentation | 3 | All functions documented (docstrings) |
| - Style | 2 | PEP8 compliant, readable |
| - Reproducibility | 2 | Clear README, runs without errors |

**Extra Credit Opportunities** (+10 points max):
- Non-flat universe extension (Ωᵥ as 3rd parameter): +5 points
- Importance sampling with Planck prior: +3 points
- Unit tests: +2 points
- JAX implementation with autodiff: +5 points
- Advanced HMC features (No-U-Turn Sampler): +5 points

### 🟡 Common Pitfalls to Avoid (Important)

> **⚠️ Things that will tank your grade:**
>
> 1. **Log-space arithmetic errors**: Always work in log-space! `p = exp(log_p)` will underflow.
>
> 2. **Ignoring burn-in**: Reporting results from the full chain including burn-in gives wrong answers.
>
> 3. **Poor convergence**: Acceptance rate outside 10-60%, R-hat > 1.1, visible trends in trace plots.
>
> 4. **Hard-coded values**: Magic numbers everywhere instead of named constants.
>
> 5. **No error handling**: Code crashes on edge cases (negative Ωₘ, h=0, etc.).
>
> 6. **Ignoring correlations**: Using proposal_cov = diagonal when parameters are clearly correlated.
>
> 7. **Wrong covariance matrix**: Using C instead of C^(-1) in likelihood.
>
> 8. **Forgetting the (1+z) factor** in D_L — very common mistake!

---

## Part 9: Connections to the Bigger Picture

### Connection to Module 1: Statistical Foundations

Every element of this project rests on Module 1:

- **CLT**: Why Gaussian likelihood works (averaging many SNe per bin)
- **Sampling Distributions**: MCMC generates samples from p(θ|D)
- **Ergodicity**: Time averages (MCMC chain) = ensemble averages (posterior)
- **Moments**: You're estimating mean and covariance of posterior
- **Maximum Entropy**: Flat prior = maximum ignorance given bounds

> **💡 Synthesis Question**  
> **Reflection**: How is MCMC sampling from p(θ|D) analogous to molecular dynamics sampling from the Boltzmann distribution in Module 3?
>
> **Answer**: They're mathematically identical! Replace E → -log p, T → 1, and you get the same exponential distribution. This is why statistical mechanics and statistical inference use the same mathematics.

### Connection to Module 3: Phase Space and Hamiltonian Dynamics

HMC is **literally** Hamilton's equations:

- Parameter space (Ωₘ, h) ↔ Position space (x, y)
- Fictitious momentum p ↔ Physical momentum
- Negative log-posterior U(θ) ↔ Gravitational potential Φ(x)
- Leapfrog integrator ↔ Your N-body code from Project 2

The symplectic structure of Hamiltonian dynamics ensures:
1. **Energy conservation** → High HMC acceptance rate
2. **Volume preservation** → Correct sampling density
3. **Time-reversibility** → Detailed balance

> **⚠️ The Unity of Physics and Statistics**  
> Hamilton formulated his equations in 1833 to describe planetary motion. Metropolis invented MCMC in 1953 for statistical mechanics simulations. Duane combined them in 1987 to create HMC.
>
> **Same math, different contexts.** This is the power of physics — the structures you learn apply everywhere from atoms to galaxies to probability distributions.

### Connection to Project 3: Monte Carlo Methods

Both projects use Monte Carlo sampling, but in different spaces:

| Project 3 (Radiative Transfer) | Project 4 (MCMC) |
|--------------------------------|------------------|
| Sample photon paths | Sample parameter values |
| Physical space (x, y, z) | Parameter space (Ωₘ, h) |
| Rejection sampling (τ, scattering) | Metropolis acceptance |
| Estimate physical quantities (flux, spectrum) | Estimate statistical quantities (mean, std) |
| Convergence: More photons → less noise | Convergence: Longer chain → better posterior |

**Common principle**: When integrals are intractable, sample instead!

---

## Part 10: The Professional Path — After the Glass Box

Once you've built MCMC/HMC from scratch, here's what the professionals use:

### Industry-Standard Tools

1. **emcee** (Foreman-Mackey et al. 2013)
   - Affine-invariant ensemble sampler
   - Better than single-chain MCMC (no tuning needed!)
   - Used in ~1000 astronomy papers
   - GitHub: https://github.com/dfm/emcee

2. **PyMC** (formerly PyMC3)
   - Full Bayesian modeling framework
   - Automatic HMC tuning (NUTS algorithm)
   - Includes HMC, NUTS, variational inference
   - Website: https://www.pymc.io

3. **numpyro** (NumPyro)
   - HMC/NUTS in JAX (fast + autodiff)
   - GPU acceleration
   - Preview of your Final Project tools!
   - GitHub: https://github.com/pyro-ppl/numpyro

4. **Stan**
   - The gold standard for Bayesian inference
   - Automatic differentiation, NUTS, diagnostics
   - C++ backend (very fast)
   - Website: https://mc-stan.org

> **💡 Tip**  
> After this project, you can use these tools **intelligently**. You understand:
> - Why MCMC converges (detailed balance)
> - How to diagnose problems (R-hat, ESS, trace plots)
> - Why HMC is faster (gradients, less random walk)
> - When methods fail (multimodal posteriors, stochastic likelihoods)
>
> **You're not a black-box user anymore.** You're a computational scientist who happens to use powerful tools.

### Example: Solving This Problem with `emcee`

```python
import emcee

# Same likelihood function you wrote
def log_probability(theta, z, mu, cov_inv):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, cov_inv)

# Ensemble sampler (32 walkers, 2 dimensions)
nwalkers = 32
ndim = 2
p0 = np.random.randn(nwalkers, ndim) * 0.1 + [0.3, 0.7]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                args=(data['z'], data['mu'], data['cov_inv']))

# Run MCMC
sampler.run_mcmc(p0, 5000, progress=True)

# Get chains
chain = sampler.get_chain(discard=1000, flat=True)

# That's it! emcee handles tuning, parallelization, everything.
```

**Compare to your code**: You did all this manually! `emcee` automates it, but you understand what it's doing under the hood.

---

## Part 11: Scientific Context and Implications

### What Are We Measuring?

Your MCMC inference is determining the **contents of the Universe**:

- **Ωₘ ≈ 0.3**: About 30% of the universe is matter (mostly dark matter)
- **Ωᵥ ≈ 0.7**: About 70% is dark energy (cosmological constant Λ)
- **h ≈ 0.7**: Hubble constant H₀ ≈ 70 km/s/Mpc

**Historical values** (1998 discovery):
- Perlmutter et al.: Ωₘ = 0.28 ± 0.08, ΩΛ = 0.72
- Riess et al.: Ωₘ = 0.24 ± 0.09, ΩΛ = 0.76

**Modern values** (Planck 2018 + SNe):
- Ωₘ = 0.315 ± 0.007
- ΩΛ = 0.685 ± 0.007
- h = 0.674 ± 0.005 (but see "Hubble tension" below!)

### The Hubble Tension

There's a **crisis in cosmology**! Two methods give different H₀:

1. **Early universe** (CMB, Planck): H₀ = 67.4 ± 0.5 km/s/Mpc
2. **Late universe** (SNe, Cepheids): H₀ = 73.0 ± 1.0 km/s/Mpc

This is a **5σ discrepancy** — not measurement error! Possible explanations:
- Systematic errors in distance ladder
- New physics (early dark energy, extra relativistic species)
- Breakdown of ΛCDM model

**Your analysis** uses JLA (2014 data). Your h should be around 0.7, intermediate between the two camps. This is an **active area of research**!

### Why Type Ia SNe Work

Type Ia SNe are "standard candles" because:

1. **Formation mechanism**: White dwarf accretes matter until it hits Chandrasekhar limit (~1.4 M☉), then explodes
2. **Physics is universal**: Same mass → same explosion energy → same peak luminosity
3. **Empirical corrections**: Light curve shape and color can standardize further

**Limitations**:
- Evolution (early universe SNe may differ)
- Environment effects (metallicity, dust)
- Sample selection (Malmquist bias)
- Systematic uncertainties dominate at high precision

> **The More You Know: Nobel Prize Physics**  
> The 2011 Physics Nobel Prize citation:
>
> *"For the discovery of the accelerating expansion of the Universe through observations of distant supernovae"*
>
> This discovery was **completely unexpected**. Most cosmologists thought the universe was either decelerating (if dense) or coasting (if flat). Acceleration implies:
> - Either: A cosmological constant (vacuum energy)
> - Or: New physics (quintessence, modified gravity)
>
> We still don't know what dark energy is! This is one of the biggest open questions in physics. Your MCMC code is analyzing data from Nobel Prize-winning science.

---

## Part 12: Timeline and Milestones

**Week 1: Foundation** (MCMC Core)
- ✅ Implement forward model (D_L, likelihood)
- ✅ Write Metropolis-Hastings sampler
- ✅ Get a chain running (even if poorly tuned)
- ✅ Implement trace plots
- **Checkpoint**: Show Dr. Rosen a trace plot (even if it looks terrible!)

**Week 2: Refinement** (Diagnostics & Tuning)
- ✅ Tune proposal distribution (acceptance rate 20-40%)
- ✅ Implement burn-in removal
- ✅ Compute autocorrelation and ESS
- ✅ Generate corner plots
- ✅ Get converged chains with R-hat < 1.1
- **Checkpoint**: Show corner plot with reasonable posteriors

**Week 3: Extension** (HMC & Comparison)
- ✅ Implement leapfrog integrator (adapt from Project 2)
- ✅ Compute gradients (numerical or autodiff)
- ✅ Write full HMC sampler
- ✅ Compare MCMC vs HMC performance
- ✅ Polish code, documentation, README
- **Deliverable**: Complete package + analysis

> **💡 Tip**  
> **Don't get stuck on perfect tuning in Week 1!** Get something working, then iterate. Premature optimization is the root of all evil.

---

## Resources and References

### Essential Reading

1. **Hogg & Foreman-Mackey (2018)**: "Data analysis recipes: Using Markov Chain Monte Carlo"  
   arXiv:1710.06068 — Best practical guide to MCMC

2. **Betancourt (2017)**: "A Conceptual Introduction to Hamiltonian Monte Carlo"  
   arXiv:1701.02434 — Geometric intuition for HMC

3. **Foreman-Mackey et al. (2013)**: "emcee: The MCMC Hammer"  
   PASP, 125, 306 — The ensemble sampler paper

### Data

- **JLA Sample**: http://supernovae.in2p3.fr/sdss_snls_jla/
  - `jla_mub.txt`: 31 binned (z, μ) measurements
  - `jla_mub_covmatrix.txt`: 31×31 covariance matrix

- **Pantheon+ Sample** (newer, optional): https://github.com/PantheonPlusSH0ES/
  - ~1700 individual SNe (unbinned)
  - More complex systematics

### Software

- **Corner plots**: https://github.com/dfm/corner.py  
  `pip install corner`

- **emcee** (for comparison): https://github.com/dfm/emcee  
  `pip install emcee`

- **JAX** (optional, for autodiff): https://github.com/google/jax  
  `pip install jax jaxlib`

### Cosmology Resources

- **Ned Wright's Cosmology Calculator**: http://www.astro.ucla.edu/~wright/CosmoCalc.html  
  Check your D_L calculations

- **Planck 2018 Results**: https://arxiv.org/abs/1807.06209  
  Modern cosmological parameters

---

## Final Thoughts

> **⚠️ What You're Really Learning**  
> This project isn't just about measuring Ωₘ and h. You're learning:
>
> 1. **Computational Statistics**: How to sample from complex distributions
> 2. **Algorithm Design**: Balancing exploration vs. exploitation
> 3. **Convergence Theory**: Markov chains, ergodicity, detailed balance
> 4. **Professional Software**: Modular design, documentation, testing
> 5. **Scientific Inference**: From data → uncertainty → conclusions
> 6. **Physical Intuition**: Phase space dynamics → statistical sampling
>
> These skills transfer beyond astronomy:
> - ML training is stochastic optimization (related to MCMC)
> - Finance uses Monte Carlo for option pricing
> - Epidemiology uses Bayesian inference for disease modeling
> - Climate science uses ensemble methods for uncertainty
>
> **You're building a foundation for computational science broadly.**

> **💡 The Glass-Box Philosophy in Action**  
> After this project, when someone says "I ran MCMC on my data," you'll know:
> - What algorithm they used (probably Metropolis-Hastings or HMC)
> - How it works (random walk with acceptance probability)
> - What can go wrong (poor convergence, long autocorrelation)
> - How to diagnose problems (trace plots, R-hat, ESS)
> - When to use alternatives (HMC for gradients, ensemble for efficiency)
>
> **That's the glass-box difference.** You're not just a tool-user. You're a tool-builder who understands the machinery.

---

## Getting Started — Your First Steps

1. **Clone the starter repo** (if provided) or create the package structure
2. **Download the JLA data** files
3. **Implement `cosmology.py`** — get D_L working and tested
4. **Write a simple script** that computes μ_th for fixed parameters
5. **Compare to published values** (sanity check!)
6. **Implement likelihood** and test with known parameters
7. **Write minimal MCMC loop** (10 lines of code is enough to start!)
8. **Get ONE parameter working** (fix h=0.7, only vary Ωₘ)
9. **Add second parameter** (now vary both)
10. **Iterate, tune, diagnose, improve!**

> **💡 Tip**  
> **Start simple!** Don't try to build the perfect package on day 1. Get something running, then refactor. Premature organization is as bad as premature optimization.

**Good luck! You're about to measure the contents of the Universe using algorithms you built from scratch. That's pretty cool.** 🌌

---

**Questions? Stuck? Confused?**  
- Check the course discussion board
- Office hours: [insert times]
- Remember: Struggling is part of learning. After 30 minutes stuck, it's time for strategic AI assistance (Phase 2!)
