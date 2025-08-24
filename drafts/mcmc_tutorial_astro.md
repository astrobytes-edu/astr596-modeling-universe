# MCMC and Bayesian Inference for Astronomers: From Probability to Parameter Estimation

## Prerequisites Check

This tutorial introduces fundamental concepts in probability and statistics. We'll build everything from the ground up!

### Mathematical Prerequisites
You should be comfortable with:
- **Basic calculus**: Derivatives, integrals, partial derivatives
- **Linear algebra**: Matrix multiplication, solving linear systems
- **Python programming**: Loops, functions, NumPy arrays

### Notation Convention
Throughout this document:
- $P(A)$ denotes the probability of event $A$
- $P(A|B)$ denotes the conditional probability of $A$ given $B$
- $p(x)$ denotes a probability density function (PDF) for continuous variable $x$
- $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ denotes a column vector
- $\sim$ means "is distributed as" (e.g., $x \sim \mathcal{N}(\mu, \sigma^2)$)
- $\propto$ means "proportional to" (equal up to a constant)
- $\mathbb{E}[X]$ denotes the expected value (mean) of $X$
- $\text{Var}[X]$ denotes the variance of $X$

---

## Introduction: Why Bayesian Inference and MCMC Matter in Astronomy

Imagine you're studying a newly discovered exoplanet. Your radial velocity measurements suggest a planetary mass of 1.5 Jupiter masses, but there's uncertainty from measurement noise, stellar activity, and incomplete orbital coverage. How do you:
1. Quantify your uncertainty about the planet's mass?
2. Incorporate prior knowledge (e.g., planetary mass distributions)?
3. Handle correlations between parameters (mass, period, eccentricity)?
4. Make decisions based on incomplete data?

This is where Bayesian inference shines. Unlike traditional "best-fit" approaches that give you a single answer, Bayesian methods provide the full **probability distribution** of your parameters given the data.

But here's the challenge: for realistic problems with many parameters, computing these distributions requires integrals in high dimensions - often impossible to solve analytically. This is where Markov Chain Monte Carlo (MCMC) comes to the rescue, allowing us to sample from complex probability distributions without computing intractable integrals.

In this tutorial, we'll build up from basic probability theory to implementing MCMC for real astronomical problems. By the end, you'll be able to:
- Think probabilistically about data and models
- Use Bayes' theorem to update beliefs with new data
- Implement MCMC algorithms from scratch
- Apply these tools to stellar dynamics and N-body simulations

Let's begin with the fundamentals!

---

## Part 1: Probability Foundations

### Basic Probability Concepts

**Probability** quantifies uncertainty. For an event $A$:
$$0 \leq P(A) \leq 1$$

where $P(A) = 0$ means impossible and $P(A) = 1$ means certain.

### Discrete vs Continuous Probabilities

**Discrete**: Probability mass function (PMF)
- Example: Number of planets in a system
- $P(N = k)$ = probability of exactly $k$ planets
- Must sum to 1: $\sum_k P(N = k) = 1$

**Continuous**: Probability density function (PDF)
- Example: Stellar mass
- $p(m)$ = probability density at mass $m$
- Must integrate to 1: $\int_{-\infty}^{\infty} p(m) dm = 1$
- Probability of range: $P(a < m < b) = \int_a^b p(m) dm$

### 🌟 Astronomical Example: Stellar Mass Distribution

The initial mass function (IMF) for stars follows approximately:
$$p(m) \propto m^{-2.35} \quad \text{for } 0.5 < m < 100 \, M_\odot$$

To normalize:
$$\int_{0.5}^{100} c \cdot m^{-2.35} dm = 1$$

Solving: $c \approx 0.73$

📝 **Checkpoint 1**: If the IMF is $p(m) \propto m^{-2.35}$, what's the probability a randomly selected star has mass between 1 and 2 $M_\odot$?  
*Answer: $P(1 < m < 2) = \int_1^2 0.73 m^{-2.35} dm \approx 0.18$ or 18%*

### Joint, Marginal, and Conditional Probabilities

**Joint Probability**: Probability of multiple events occurring together
$$P(A \text{ and } B) = P(A, B)$$

**Marginal Probability**: Probability of one event, summing over all possibilities of others
$$P(A) = \sum_B P(A, B)$$

For continuous variables:
$$p(x) = \int p(x, y) dy$$

**Conditional Probability**: Probability of $A$ given that $B$ occurred
$$P(A|B) = \frac{P(A, B)}{P(B)}$$

### Independent vs Dependent Variables

**Independent**: Knowledge of one doesn't affect the other
$$P(A, B) = P(A) \cdot P(B)$$
$$P(A|B) = P(A)$$

**Dependent**: Knowledge of one changes probabilities for the other
$$P(A, B) \neq P(A) \cdot P(B)$$

### 🌟 Example: Binary Star Systems

Consider a stellar cluster where:
- $P(\text{star is binary}) = 0.3$
- $P(\text{star is massive} | \text{binary}) = 0.4$
- $P(\text{star is massive} | \text{single}) = 0.2$

These are **dependent** because binarity affects mass probability.

Joint probability of massive binary:
$$P(\text{massive, binary}) = P(\text{massive}|\text{binary}) \cdot P(\text{binary}) = 0.4 \times 0.3 = 0.12$$

Total probability of massive star (marginalizing):
$$P(\text{massive}) = P(\text{massive}|\text{binary})P(\text{binary}) + P(\text{massive}|\text{single})P(\text{single})$$
$$= 0.4 \times 0.3 + 0.2 \times 0.7 = 0.26$$

📝 **Checkpoint 2**: If 30% of stars are binaries and massive stars are twice as likely to be in binaries, what's $P(\text{binary}|\text{massive})$?  
*Answer: Using Bayes' theorem (coming next!), $P(\text{binary}|\text{massive}) = 0.4 \times 0.3 / 0.26 \approx 0.46$*

---

## Part 2: Bayes' Theorem - The Foundation of Bayesian Inference

### Deriving Bayes' Theorem

Start with the definition of conditional probability:
$$P(A|B) = \frac{P(A, B)}{P(B)} \quad \text{and} \quad P(B|A) = \frac{P(A, B)}{P(A)}$$

Since $P(A, B) = P(B, A)$:
$$P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

Rearranging gives **Bayes' Theorem**:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Bayes' Theorem for Parameter Estimation

In the context of data analysis:
$$p(\theta | D) = \frac{p(D | \theta) \cdot p(\theta)}{p(D)}$$

where:
- $\theta$: Parameters we want to estimate
- $D$: Observed data
- $p(\theta | D)$: **Posterior** - what we want (parameter probability given data)
- $p(D | \theta)$: **Likelihood** - probability of data given parameters
- $p(\theta)$: **Prior** - initial belief about parameters
- $p(D)$: **Evidence** - normalization constant

Often we write:
$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

### 🌟 Complete Example: Estimating Star Formation Rate

A galaxy's H-α luminosity depends on its star formation rate (SFR). We observe:
- H-α flux: $F_{obs} = 3.2 \pm 0.5 \times 10^{-15}$ erg/s/cm²
- Distance: $d = 10$ Mpc

**Model**: $F_{model}(\text{SFR}) = \frac{\text{SFR} \cdot k}{4\pi d^2}$

where $k = 10^{41}$ erg/s per $M_\odot$/yr (calibration constant).

**Likelihood** (assuming Gaussian errors):
$$p(F_{obs} | \text{SFR}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(F_{obs} - F_{model}(\text{SFR}))^2}{2\sigma^2}\right)$$

**Prior** (log-normal, typical for SFRs):
$$p(\text{SFR}) = \frac{1}{\text{SFR} \cdot \sqrt{2\pi \cdot 0.5^2}} \exp\left(-\frac{(\ln(\text{SFR}) - \ln(1))^2}{2 \cdot 0.5^2}\right)$$

**Posterior**:
$$p(\text{SFR} | F_{obs}) \propto p(F_{obs} | \text{SFR}) \cdot p(\text{SFR})$$

Computing this gives most probable SFR ≈ 2.1 $M_\odot$/yr with 68% credible interval [1.6, 2.7].

📝 **Checkpoint 3**: Why do we use a log-normal prior for star formation rate instead of a uniform prior?  
*Answer: SFR is positive-definite and spans orders of magnitude (0.01 to 1000 $M_\odot$/yr), making log-normal more physically appropriate than uniform*

---

## Part 3: Linear Regression - A Gateway to Machine Learning

### The Classical (Frequentist) Approach

Given data points $(x_i, y_i)$ with $i = 1, ..., n$, find the line:
$$y = mx + b$$

that minimizes squared errors:
$$\chi^2 = \sum_{i=1}^n (y_i - mx_i - b)^2$$

Taking derivatives and setting to zero:
$$\frac{\partial \chi^2}{\partial m} = -2\sum_i x_i(y_i - mx_i - b) = 0$$
$$\frac{\partial \chi^2}{\partial b} = -2\sum_i (y_i - mx_i - b) = 0$$

Solving gives:
$$m = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$$
$$b = \frac{\sum y_i - m\sum x_i}{n}$$

### The Bayesian Approach

Now let's think probabilistically!

**Likelihood** (assuming Gaussian noise with variance $\sigma^2$):
$$p(\mathbf{y} | m, b, \sigma^2, \mathbf{x}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - mx_i - b)^2}{2\sigma^2}\right)$$

**Prior** (weakly informative):
$$p(m) \sim \mathcal{N}(0, 100) \quad p(b) \sim \mathcal{N}(0, 100) \quad p(\sigma) \sim \text{Half-Cauchy}(0, 5)$$

**Posterior**:
$$p(m, b, \sigma | \mathbf{y}, \mathbf{x}) \propto p(\mathbf{y} | m, b, \sigma, \mathbf{x}) \cdot p(m) \cdot p(b) \cdot p(\sigma)$$

This posterior is a 3D distribution - we get uncertainty estimates for all parameters!

### 🌟 Astronomical Example: Period-Luminosity Relation

For Cepheid variables, brightness relates to pulsation period:
$$M_V = a \log_{10}(P) + b$$

Given observations of 10 Cepheids:

| Period (days) | $M_V$ (mag) | $\sigma$ (mag) |
|--------------|-------------|----------------|
| 3.2 | -2.81 | 0.15 |
| 5.4 | -3.35 | 0.12 |
| 10.1 | -4.02 | 0.18 |
| ... | ... | ... |

**Frequentist result**: $a = -2.43 \pm 0.08$, $b = -1.32 \pm 0.05$

**Bayesian result**: Full posterior distribution showing:
- Most probable: $a = -2.43$, $b = -1.32$ (same as frequentist)
- But also: 15% probability that $a < -2.5$ (important for distance ladder!)
- Parameter correlation: $\text{Cov}[a, b] = -0.02$

### Matrix Formulation for Multiple Linear Regression

For model $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}$ where:
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$: observations
- $X \in \mathbb{R}^{n \times p}$: design matrix
- $\boldsymbol{\beta} \in \mathbb{R}^{p \times 1}$: parameters
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$: noise

**Maximum likelihood solution**:
$$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$$

**Bayesian posterior** (with conjugate prior $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$):
$$p(\boldsymbol{\beta} | \mathbf{y}) \sim \mathcal{N}(\boldsymbol{\mu}_{\beta}, \boldsymbol{\Sigma}_{\beta})$$

where:
$$\boldsymbol{\Sigma}_{\beta} = (\frac{1}{\sigma^2}X^T X + \frac{1}{\tau^2}I)^{-1}$$
$$\boldsymbol{\mu}_{\beta} = \boldsymbol{\Sigma}_{\beta} \frac{1}{\sigma^2} X^T \mathbf{y}$$

📝 **Checkpoint 4**: In Bayesian linear regression, what happens to the posterior as we get more data (n → ∞)?  
*Answer: The likelihood dominates the prior, and the posterior approaches the frequentist MLE solution*

---

## Part 4: Why We Need MCMC

### The Challenge: Intractable Integrals

For most real problems, we need to compute:

**Posterior normalization**:
$$p(D) = \int p(D|\theta) p(\theta) d\theta$$

**Posterior expectations**:
$$\mathbb{E}[\theta|D] = \int \theta \cdot p(\theta|D) d\theta$$

**Marginal posteriors** (integrating out nuisance parameters):
$$p(\theta_1|D) = \int p(\theta_1, \theta_2, ..., \theta_n|D) d\theta_2...d\theta_n$$

For a 10-parameter model, numerical integration on a 100-point grid per dimension requires $100^{10} = 10^{20}$ evaluations!

### The Solution: Monte Carlo Integration

Instead of computing integrals, draw samples from the distribution:

$$\mathbb{E}[f(\theta)] = \int f(\theta) p(\theta) d\theta \approx \frac{1}{N} \sum_{i=1}^N f(\theta_i)$$

where $\theta_i \sim p(\theta)$

But how do we sample from complex distributions? Enter MCMC!

### Markov Chains: The Key Insight

A **Markov chain** is a sequence where each state depends only on the previous state:
$$p(\theta_{t+1} | \theta_t, \theta_{t-1}, ..., \theta_0) = p(\theta_{t+1} | \theta_t)$$

The brilliant idea: Design a Markov chain whose **stationary distribution** is our target posterior $p(\theta|D)$.

### 🌟 Intuitive Analogy: Exploring a Mountain Range

Imagine you're dropped on a mountain range at night with only a dim flashlight:
- You can only see the local terrain (evaluate probability locally)
- You want to explore according to elevation (sample from distribution)
- Random walk with clever rules → eventually explores properly
- Higher elevations (probability) visited more often

This is exactly what MCMC does in parameter space!

---

## Part 5: The Metropolis-Hastings Algorithm

### The Algorithm

The Metropolis-Hastings algorithm generates samples from any distribution $p(\theta)$:

1. **Initialize**: Start at $\theta_0$
2. **Propose**: Generate candidate $\theta^* \sim q(\theta^* | \theta_t)$
3. **Compute acceptance ratio**:
   $$\alpha = \min\left(1, \frac{p(\theta^*) q(\theta_t | \theta^*)}{p(\theta_t) q(\theta^* | \theta_t)}\right)$$
4. **Accept/Reject**:
   - Generate $u \sim \text{Uniform}(0, 1)$
   - If $u < \alpha$: accept, set $\theta_{t+1} = \theta^*$
   - Else: reject, set $\theta_{t+1} = \theta_t$
5. **Repeat** steps 2-4

### Special Case: Metropolis Algorithm

If the proposal is symmetric: $q(\theta^* | \theta_t) = q(\theta_t | \theta^*)$

The acceptance ratio simplifies to:
$$\alpha = \min\left(1, \frac{p(\theta^*)}{p(\theta_t)}\right)$$

Common symmetric proposal: $\theta^* = \theta_t + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

### 🌟 Complete Example: Fitting a Spectral Line

Observe a spectral line with Gaussian profile:
$$f(\lambda) = A \exp\left(-\frac{(\lambda - \lambda_0)^2}{2\sigma_{\lambda}^2}\right) + B$$

where:
- $A$: amplitude
- $\lambda_0$: central wavelength
- $\sigma_{\lambda}$: line width
- $B$: continuum level

**Data**: 50 flux measurements with Gaussian noise

**Implementation**:
```python
def log_posterior(theta, data, wavelengths):
    A, lambda_0, sigma_lambda, B = theta
    
    # Prior (log probabilities)
    if A < 0 or sigma_lambda < 0:
        return -np.inf
    log_prior = -0.5*(A/10)**2 - 0.5*((lambda_0-5000)/5)**2
    
    # Likelihood
    model = A*np.exp(-(wavelengths-lambda_0)**2/(2*sigma_lambda**2)) + B
    residuals = data - model
    log_likelihood = -0.5*np.sum(residuals**2/noise_var)
    
    return log_prior + log_likelihood

# Metropolis algorithm
def metropolis(log_post, theta_init, n_samples, step_size):
    samples = np.zeros((n_samples, len(theta_init)))
    theta = theta_init
    n_accepted = 0
    
    for i in range(n_samples):
        # Propose
        theta_prop = theta + np.random.normal(0, step_size, len(theta))
        
        # Accept/reject
        log_alpha = log_post(theta_prop) - log_post(theta)
        if np.log(np.random.rand()) < log_alpha:
            theta = theta_prop
            n_accepted += 1
        
        samples[i] = theta
    
    print(f"Acceptance rate: {n_accepted/n_samples:.2%}")
    return samples
```

📝 **Checkpoint 5**: If the acceptance rate is 5%, what should you do to the step size?  
*Answer: Decrease it - too many rejections mean the proposals are too ambitious. Aim for 20-50% acceptance.*

### Why It Works: Detailed Balance

The key property ensuring convergence is **detailed balance**:
$$p(\theta) P(\theta \to \theta') = p(\theta') P(\theta' \to \theta)$$

This means probability flow from $\theta$ to $\theta'$ equals flow back.

For Metropolis-Hastings:
- $P(\theta \to \theta') = q(\theta'|\theta) \cdot \alpha(\theta, \theta')$
- Detailed balance can be proven algebraically

### Convergence Diagnostics

How do we know when the chain has converged?

**1. Trace Plots**: Parameter values vs iteration should look like noise around a constant

**2. Autocorrelation**: Measure correlation between $\theta_t$ and $\theta_{t+k}$
$$\rho_k = \frac{\text{Cov}[\theta_t, \theta_{t+k}]}{\text{Var}[\theta]}$$

Effective sample size: $N_{eff} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$

**3. Gelman-Rubin Statistic**: Compare variance within and between multiple chains
$$\hat{R} = \sqrt{\frac{\text{Var}_{between} + \text{Var}_{within}}{\text{Var}_{within}}}$$

Want $\hat{R} < 1.1$ for convergence.

⚠️ **Advanced Note**: The first samples are influenced by initialization. Always discard "burn-in" period (typically first 10-50% of samples).

---

## Part 6: Advanced MCMC Methods

### Gibbs Sampling

When we can sample from conditional distributions:

Instead of updating all parameters at once, update one at a time:
1. Sample $\theta_1^{(t+1)} \sim p(\theta_1 | \theta_2^{(t)}, \theta_3^{(t)}, ..., D)$
2. Sample $\theta_2^{(t+1)} \sim p(\theta_2 | \theta_1^{(t+1)}, \theta_3^{(t)}, ..., D)$
3. Continue for all parameters

No accept/reject step needed!

### Hamiltonian Monte Carlo (HMC)

Uses gradient information to make better proposals:
- Introduce momentum variables $\mathbf{p}$
- Simulate Hamiltonian dynamics: $H(\theta, \mathbf{p}) = U(\theta) + K(\mathbf{p})$
- Where $U(\theta) = -\log p(\theta|D)$ (potential energy)
- And $K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T M^{-1} \mathbf{p}$ (kinetic energy)

Advantages:
- Explores more efficiently
- Better for high dimensions
- Less correlation between samples

### Parallel Tempering

Run multiple chains at different "temperatures":
$$p_{\beta}(\theta) \propto [p(\theta|D)]^{\beta}$$

- $\beta = 1$: target distribution
- $\beta < 1$: flattened distribution (easier to explore)
- Occasionally swap states between chains

Helps escape local modes!

### 🌟 Astronomical Application: Multi-Modal Posteriors

Binary star orbital solutions often have multiple modes (e.g., $\omega$ and $\omega + 180°$).

Standard MCMC gets stuck in one mode. Parallel tempering explores both:
- Chain 1 ($\beta = 1.0$): Sharp posterior
- Chain 2 ($\beta = 0.5$): Smoothed, can jump between modes
- Chain 3 ($\beta = 0.1$): Nearly flat, explores widely

---

## Part 7: Applications to Stellar Dynamics

### Example 1: Mass Estimation from Stellar Orbits

**Problem**: Estimate central black hole mass from stellar orbits

**Data**: Positions $(x_i, y_i)$ and velocities $(v_{x,i}, v_{y,i})$ for 20 stars

**Model**: Stars orbit in Keplerian potential
$$v^2 = \frac{GM_{BH}}{r}$$

**Parameters**:
- $M_{BH}$: Black hole mass
- $(x_0, y_0)$: Black hole position
- $d$: Distance to system

**Likelihood** (assuming Gaussian errors):
$$p(\text{data} | M_{BH}, x_0, y_0, d) = \prod_i \mathcal{N}(v_{obs,i} | v_{model,i}, \sigma_v^2)$$

**Implementation challenges**:
- Strong correlations between $M_{BH}$ and $d$
- Need good proposal distribution
- Use logarithmic parameters: $\log M_{BH}$

### Example 2: Star Cluster IMF from Photometry

**Problem**: Determine initial mass function from observed luminosity function

**Complications**:
- Mass-luminosity relation uncertain
- Binaries contaminate sample
- Completeness varies with magnitude

**Hierarchical Bayesian Model**:
$$\text{True masses} \sim \text{IMF}(\alpha)$$
$$\text{Binary fraction} \sim \text{Beta}(a, b)$$
$$\text{Observed luminosity} = f(\text{mass}, \text{age}, \text{metallicity}) + \text{noise}$$

Parameters: $\alpha$ (IMF slope), binary parameters, completeness function

MCMC explores joint posterior of all parameters simultaneously!

### Example 3: Dynamical Modeling of Globular Clusters

**Observable**: Velocity dispersion profile $\sigma(r)$

**Model**: Jeans equation for spherical system
$$\frac{d(\rho \sigma_r^2)}{dr} + \frac{2\beta \rho \sigma_r^2}{r} = -\rho \frac{GM(r)}{r^2}$$

**Parameters**:
- Mass profile: King model with $W_0$, $r_c$
- Anisotropy: $\beta(r)$
- Distance and inclination

**Challenge**: Degeneracies between mass and anisotropy

**Solution**: MCMC explores all degeneracies, giving honest uncertainties

📝 **Checkpoint 6**: Why is MCMC particularly valuable for the mass-anisotropy degeneracy?  
*Answer: It naturally explores the curved degeneracy valley in parameter space, giving the full range of viable solutions rather than just a "best fit"*

---

## Part 8: Practical Implementation Guide

### Choosing Priors

**Types of Priors**:

1. **Uniform (flat)**: $p(\theta) = \text{const}$ for $\theta \in [a, b]$
   - Use when: Truly ignorant about parameter
   - Warning: Not invariant under transformation!

2. **Gaussian**: $p(\theta) \sim \mathcal{N}(\mu, \sigma^2)$
   - Use when: Have previous measurement
   - Good for: Location parameters

3. **Log-normal**: $\log \theta \sim \mathcal{N}(\mu, \sigma^2)$
   - Use when: Parameter is positive, spans orders of magnitude
   - Good for: Masses, luminosities, rates

4. **Jeffreys**: $p(\theta) \propto 1/\theta$
   - Use when: Want scale-invariant prior
   - Good for: Scale parameters

### 🌟 Prior Selection Example

For stellar mass estimation:
- **Bad**: Uniform(0, 1000) - gives huge prior weight to massive stars
- **Better**: Log-uniform(0.08, 150) - equal weight per decade
- **Best**: Use known IMF as prior!

### Proposal Distribution Tuning

**Adaptive Metropolis**: Adjust proposal covariance during burn-in
```python
def adaptive_metropolis(log_post, theta_init, n_samples):
    d = len(theta_init)
    samples = np.zeros((n_samples, d))
    
    # Initial proposal covariance
    C = np.eye(d) * 0.1
    
    # Adaptation parameters
    adapt_interval = 100
    target_accept = 0.234  # Optimal for high dimensions
    
    for i in range(n_samples):
        # Propose
        theta_prop = theta + np.random.multivariate_normal(np.zeros(d), C)
        
        # Accept/reject (standard Metropolis)
        # ...
        
        # Adapt covariance
        if i % adapt_interval == 0 and i < n_samples // 2:
            emp_cov = np.cov(samples[max(0,i-1000):i].T)
            C = 2.38**2 / d * emp_cov + 1e-6 * np.eye(d)
    
    return samples
```

### Convergence Assessment Workflow

1. **Run multiple chains** with different initializations
2. **Plot traces** - should look like "hairy caterpillars"
3. **Check $\hat{R}$** - want < 1.1 for all parameters
4. **Compute effective sample size** - want > 100 per parameter
5. **Plot autocorrelation** - should decay quickly
6. **Compare marginal distributions** between chains

### Common Pitfalls and Solutions

**Problem 1**: Chain stuck in local mode
- *Solution*: Use multiple chains, parallel tempering

**Problem 2**: Very slow mixing (high autocorrelation)
- *Solution*: Reparameterize, use HMC, or thin samples

**Problem 3**: Parameters on very different scales
- *Solution*: Work in log space or standardize

**Problem 4**: Numerical overflow/underflow
- *Solution*: Work with log probabilities throughout

### Visualization Best Practices

```python
def plot_mcmc_results(samples, labels):
    """Corner plot showing all parameters"""
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                # Diagonal: marginal distribution
                ax.hist(samples[:, i], 50, density=True)
            elif i > j:
                # Lower triangle: 2D density
                ax.hist2d(samples[:, j], samples[:, i], 50)
            else:
                ax.set_visible(False)
    
    return fig
```

---

## Part 9: Connection to Machine Learning

### Bayesian Linear Regression as Foundation for ML

Linear regression with basis functions:
$$y = \sum_{j=1}^M w_j \phi_j(\mathbf{x}) + \epsilon$$

This is a **linear model** in parameters $w_j$, but can capture non-linear relationships through basis functions $\phi_j$.

Examples:
- Polynomial: $\phi_j(x) = x^j$
- Fourier: $\phi_j(x) = \sin(jx), \cos(jx)$  
- Radial basis: $\phi_j(x) = \exp(-(x-\mu_j)^2/2\sigma^2)$

This leads directly to:
- **Gaussian Processes**: Infinite basis functions (next tutorial!)
- **Neural Networks**: Learned basis functions

### MCMC for Neural Network Training

Bayesian neural networks use MCMC to sample weight distributions:
- Each weight has a distribution, not a point estimate
- Predictions integrate over weight uncertainty
- Natural regularization through priors

### The Bigger Picture: Probabilistic Thinking

MCMC teaches us to think probabilistically:
- **Uncertainty is fundamental** - embrace it, don't hide it
- **Prior knowledge matters** - use it wisely
- **Model comparison** - via evidence $p(D)$
- **Decision making** - integrate over uncertainty

This probabilistic framework underlies:
- Gaussian Processes (distributions over functions)
- Bayesian neural networks
- Probabilistic programming
- Modern machine learning

---

## Quick Reference Summary

### Key Probability Rules

| Rule | Formula |
|------|---------|
| Sum Rule | $P(A) = \sum_B P(A, B)$ |
| Product Rule | $P(A, B) = P(A\|B)P(B)$ |
| Bayes' Theorem | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ |
| Marginalization | $p(x) = \int p(x,y) dy$ |

### MCMC Algorithms

| Algorithm | When to Use | Key Feature |
|-----------|------------|-------------|
| Metropolis | Simple posteriors | Symmetric proposals |
| Metropolis-Hastings | Asymmetric proposals | General purpose |
| Gibbs | Conditional distributions known | No rejection |
| HMC | High dimensions | Uses gradients |
| Parallel Tempering | Multi-modal | Explores modes |

### Convergence Diagnostics

| Diagnostic | Target Value | Interpretation |
|------------|--------------|----------------|
| $\hat{R}$ | < 1.1 | Between/within chain variance |
| $N_{eff}$ | > 100 | Effective independent samples |
| Acceptance Rate | 20-50% | Proposal tuning quality |
| Autocorrelation | Quick decay | Sample independence |

### Common Priors

| Prior | Use Case | Formula |
|-------|----------|---------|
| Uniform | Bounded, uninformative | $p(\theta) = 1/(b-a)$ |
| Gaussian | Previous measurement | $p(\theta) \propto \exp(-\theta^2/2\sigma^2)$ |
| Log-normal | Positive, wide range | $p(\log\theta) \sim \mathcal{N}(\mu, \sigma^2)$ |
| Jeffreys | Scale parameter | $p(\theta) \propto 1/\theta$ |

---

## Conclusion: The Power of Probabilistic Inference

You now have the tools to:
- Think probabilistically about data and models
- Use Bayes' theorem to update beliefs with evidence
- Implement MCMC to explore complex parameter spaces
- Quantify uncertainty in all your estimates
- Make informed decisions under uncertainty

For your N-body simulations, MCMC enables:
- **Parameter estimation**: Initial conditions from observations
- **Model comparison**: Different dynamical models
- **Uncertainty propagation**: How observational errors affect conclusions
- **Missing data**: Inference with incomplete observations

The journey from basic probability through Bayes' theorem to MCMC has equipped you with a fundamental tool of modern astronomy. Every major astronomical discovery now involves Bayesian inference:
- Exoplanet masses and orbits
- Cosmological parameters
- Gravitational wave source properties
- Dark matter distribution

As you move forward to Gaussian Processes and Neural Networks, remember that they build on these probabilistic foundations. GPs are Bayesian regression with infinite basis functions. Neural networks can be viewed through a Bayesian lens, with MCMC providing uncertainty in predictions.

The key insight: **Embrace uncertainty rather than hiding it**. In astronomy, where we observe the universe from a single vantage point with limited data, honest uncertainty quantification through Bayesian inference and MCMC is not just useful—it's essential.

---

## Final Checkpoints

📝 **Final Challenge 1**: You observe a star with periodic radial velocity variations. Design a Bayesian model to determine if it's due to a planet or stellar activity. What parameters would you include? What priors?

📝 **Final Challenge 2**: Your MCMC chain for a 10-parameter model has $\hat{R} = 1.3$ for one parameter and 1.05 for others. The trace plot shows the problematic parameter occasionally jumping between two values. What's happening and how would you fix it?

📝 **Final Challenge 3**: You're fitting a power law $y = Ax^{\alpha}$ to data spanning 6 orders of magnitude in both x and y. How would you parameterize this for MCMC and what priors would you use?

*Think about these as you prepare for Gaussian Processes, where we'll extend these ideas to infinite-dimensional parameter spaces!*