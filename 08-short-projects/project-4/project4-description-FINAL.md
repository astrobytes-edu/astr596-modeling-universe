# Project 4: Building a Universal Inference Engine

**ASTR 596: Modeling the Universe**
**Instructor:** Dr. Anna Rosen
**Due**: Monday, November 10, 2025, 11:59 PM  
**AI Policy**: Phase 2 (strategic use after baseline implementation works)

> *"The most important questions of life are indeed, for the most part, really only problems of probability."*  
> — Pierre-Simon Laplace, *Théorie Analytique des Probabilités* (1812)

> *"It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts."*  
> — Arthur Conan Doyle, *A Scandal in Bohemia*

---

## AI Usage Policy Reminder

This project operates under **Phase 2 AI Policy**: You may use AI tools strategically after your baseline implementation works. **You must document all AI usage:**

1. **In code**: Include a 3-line header comment in any file where AI assisted:

   ```python
   # AI Usage: [Tool name] assisted with [specific task]
   # Date: [Date]
   # Extent: [Brief description of what was AI-generated vs. your work]
   ```

2. **In Growth Memo**: Include an "AI Usage Reflection" section discussing how AI tools helped or hindered your learning

See the course AI policy document for complete guidelines.

---

## The Big Picture: What You're Really Building

This project is about building **general-purpose inference machinery**, not just analyzing one dataset. You're creating tools that work for *any* scientific model, *any* data, *any* dimensionality.

**The Core Insight**: Most scientific problems share the same structure:

1. You have a **forward model** that predicts observations from parameters
2. You have **data** (noisy measurements of those observations)
3. You need to **infer** the parameters that best explain the data
4. You need to **quantify uncertainty** (not just a best fit, but a full probability distribution)

Whether you're:

- Measuring cosmological parameters from supernovae (this project)
- Characterizing exoplanet properties from transit light curves
- Inferring dark matter halo profiles from galaxy kinematics
- Fitting stellar population models to galaxy spectra
- Determining black hole masses from AGN variability

**The inference machinery is the same.** Only the forward model and likelihood function change.

### What Makes This Powerful: Modularity

You'll build a **two-layer architecture**:

**Layer 1: Universal Samplers** (general, reusable)

- `mcmc.py`: Metropolis-Hastings algorithm
- `hmc.py`: Hamiltonian Monte Carlo
- `diagnostics.py`: Convergence tests, autocorrelation, Gelman-Rubin

These modules know nothing about cosmology. They just need:

- A function that computes `log_posterior(theta)`
- Initial parameter values
- Proposal/step size hyperparameters

**Layer 2: Problem-Specific Components** (this project)

- `cosmology.py`: Forward model ($D_L(z; \Omega_m, h)$ → $\mu$)
- `likelihood.py`: Log-likelihood for supernova data

The beauty: **Tomorrow you can swap in a different likelihood (exoplanets, stellar dynamics, whatever) and your samplers still work.** This is the glass-box philosophy in action — you're not learning "how to analyze supernovae," you're learning "how to build inference engines."

---

## Connection to Your Learning Journey

This project synthesizes everything you've learned:

**Module 1 - How the Universe Computes:**

- Statistical thinking: parameters are random variables, posteriors are distributions
- Sampling theory: We can't compute high-dimensional integrals, so we sample
- The CLT: Averages over MCMC samples converge to true posterior means
- Ergodicity: A single long Markov chain explores the same distribution as the ensemble

**Module 5 - Statistical Inference:**

- Forward and inverse problems: Predict data from parameters, infer parameters from data
- Bayes' theorem: $p(\theta|D) \propto p(D|\theta)p(\theta)$
- Metropolis-Hastings: Build a Markov chain that explores the posterior
- Detailed balance: The mathematical guarantee that MCMC works
- Diagnostics: How to know when convergence is achieved
- Advanced methods: Hamiltonian Monte Carlo uses gradients to explore parameter space efficiently

**Project 2: N-body Simulation**

- Leapfrog integration: Symplectic, time-reversible, conserves energy
- **Key insight**: HMC is just N-body in parameter space!
  - Particles → Parameters
  - Potential energy → Negative log-posterior
  - Kinetic energy → Momentum (auxiliary variables)

Everything connects. This isn't isolated knowledge—it's a unified computational framework for science.

---

## The Scientific Application: Measuring Dark Energy

To make this concrete, you'll apply your inference engine to **Type Ia supernova cosmology**—the 2011 Nobel Prize-winning discovery of cosmic acceleration.

**Read first**: [Scientific Background Document](./project4-scientific-background.md)

That document explains:

- Why Type Ia supernovae are "standard candles"
- What $\Omega_m$, $\Omega_\Lambda$, and $h$ mean physically
- The distance-redshift relation in Big Bang cosmology
- The 1998 discovery that the universe is accelerating

**Your task**: Use Bayesian inference (MCMC and HMC) to infer $(\Omega_m, h)$ from real supernova data, assuming a flat universe ($\Omega_m + \Omega_\Lambda = 1$).

This is the same analysis that revealed dark energy dominates the cosmos. You're not just practicing methods—you're doing real science with Nobel Prize data.

---

## Project Structure and Workflow

**Important**: The parts below are not strictly sequential. As you implement your MCMC sampler (Part 2), you'll need to develop diagnostic tools (Part 3) in parallel to assess convergence and tune your algorithm. Think of this project as iterative: build, test, diagnose, refine. By the time you move to HMC (Part 4), you'll have a robust testing framework that you can immediately reuse—same diagnostic functions, different sampler.

---

## Part 1: The Forward Model and Likelihood

### 1.1: Forward Model Implementation

**Module**: `cosmology.py`

You need to compute the theoretical distance modulus $\mu(z; \Omega_m, h)$ for a flat universe.

**Required functions**:

- `luminosity_distance(z, Omega_m, h)`
- `distance_modulus(z, Omega_m, h)`

**Mathematical Background**:

For a flat universe ($\Omega_\Lambda = 1 - \Omega_m$), the luminosity distance is:

$$D_L(z) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}$$

where:
$$E(z) = \sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}$$

and $H_0 = 100h$ km s$^{-1}$ Mpc$^{-1}$, $c = 299{,}792.458\ \text{km s}^{-1}$.

The distance modulus is:

$$\mu = 25 - 5\log_{10}(h) + 5\log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)$$

where $D_L^* \equiv D_L(h=1)$ is the luminosity distance computed with $h=1$.

**Implementation approaches**:

1. **Numerical integration**: Use `scipy.integrate.quad` to evaluate the integral directly
2. **Fitting formula**: Use the Pen (1999) approximation (accurate to 0.4% for $0.2 \leq \Omega_m \leq 1$):

$$D_L(z) = \frac{c}{H_0}(1+z)\left[\eta(1, \Omega_m) - \eta\left(\frac{1}{1+z}, \Omega_m\right)\right]$$

where:
$$\eta(a, \Omega_m) = 2\sqrt{s^3 + 1}\left[\frac{1}{a^4} - 0.1540\frac{s}{a^3} + 0.4304\frac{s^2}{a^2} + 0.19097\frac{s^3}{a} + 0.066941s^4\right]^{-1/8}$$

and $s^3 \equiv (1-\Omega_m)/\Omega_m$.

**What to do**: Implement **both** methods. The fitting formula is much faster (no numerical integration), but you should validate both approaches against each other to ensure they agree within 0.4% or better. Then experiment with timing: which method is fast enough for MCMC where you'll call this function thousands of times? You'll likely choose the approximation for production runs, but understanding both builds intuition about the trade-offs between accuracy and speed.

**Validation**: Your implementations should agree with Ned Wright's Cosmology Calculator to within ~0.1%. For the worked example in the Scientific Background ($z=0.5$, $\Omega_m=0.3$, $h=0.7$), you should get $D_L^* \approx 1{,}983$ Mpc and $\mu \approx 42.26$ mag (±0.02).

---

### 1.2: Likelihood Function

**Module**: `likelihood.py`

**Required functions**:

- `log_likelihood(theta, z_data, mu_data, C_factor)`
- `log_prior(theta)`
- `log_posterior(theta, z_data, mu_data, C_factor)`

**Mathematical Background**:

The log-likelihood for Gaussian errors with covariance $\mathbf{C}$ is:

$$\ln \mathcal{L}(\theta) = -\frac{1}{2}\mathbf{r}^\top \mathbf{C}^{-1} \mathbf{r}$$

where the residual vector is:
$$r_i = \mu_i^{\text{obs}} - \mu_i^{\text{theory}}(z_i; \theta)$$

and $\theta = (\Omega_m, h)$.

**Note on constants**: The full Gaussian log-likelihood includes normalization terms $-\frac{1}{2}\ln|\mathbf{C}| - \frac{n}{2}\ln(2\pi)$. Since $\mathbf{C}$ is the fixed data covariance (independent of model parameters $\theta$), these are parameter-independent constants that cancel in Metropolis-Hastings and HMC acceptance ratios. We omit them for computational efficiency.

**Implementation notes - CRITICAL**:

**Do NOT invert the covariance matrix.** Instead, use **Cholesky decomposition** for numerical stability and efficiency:

```python
from scipy.linalg import cho_factor, cho_solve

# One-time setup (do this once, outside your likelihood function):
C_factor = cho_factor(C)  # Returns (c, lower) tuple

# In your likelihood function (reuse C_factor):
# To compute C^{-1} @ r efficiently:
C_inv_r = cho_solve(C_factor, r)
chi_squared = np.dot(r, C_inv_r)
log_like = -0.5 * chi_squared
```

**Why Cholesky?**

- **Stability**: Avoids numerical issues from explicit matrix inversion
- **Speed**: $O(n³)$ for decomposition (done once), $O(n²)$ per solve (done thousands of times)
- **Professional practice**: This is how modern inference codes handle covariance matrices

**Prior specification**:

Use flat (uniform) priors with physically motivated bounds:

- $\Omega_m \in [0.0, 1.2]$: Matter density can't be negative; upper bound allows for slightly matter-dominated universes while avoiding unphysical values.
- $h \in [0.5, 0.9]$: Encompasses the range between Planck ($h \approx 0.67$) and SH0ES ($h \approx 0.73$) measurements.

Return `-np.inf` for parameters outside these bounds, return `1` for parameters inside since the constant cancels out in the acceptance ratio.

**Justification**: These bounds are informed by independent observations (CMB, BAO, local distance ladder) but are broad enough to let the supernova data speak. You can justify different bounds in your memo if you have good physical reasons.

---

### 1.3: Data Loading

**Data files** (in same directory as this document):

- [`jla_mub.txt`](./jla_mub.txt): 31 data points $(z_i, \mu_i)$
- [`jla_mub_covmatrix.txt`](./jla_mub_covmatrix.txt): 31×31 covariance matrix (961 numbers in row-major order)

Implement data loading functions to read these files and prepare them for your likelihood function. Compute the Cholesky factorization once during data loading, then pass the factored form to your likelihood function.

---

## Part 2: Metropolis-Hastings MCMC

**Module**: `mcmc.py`

This is **completely general** — it knows nothing about cosmology and should work for any inference problem.

**Required function**:

- `metropolis_hastings(log_prob_fn, theta_init, proposal_cov, n_steps, args=())`

**Algorithm**:

1. Initialize: $\theta^{(0)} = \theta_{\text{init}}$
2. For $i = 1, 2, \ldots, N$:
   - Propose: $\theta^* \sim \mathcal{N}(\theta^{(i-1)}, \Sigma_{\text{prop}})$ (multivariate normal proposal)
   - Compute log-probabilities: $\ln p(\theta^*)$ and $\ln p(\theta^{(i-1)})$
   - Compute acceptance ratio: $\alpha = \min\left(1, \exp[\ln p(\theta^*) - \ln p(\theta^{(i-1)})]\right)$
   - Draw $u \sim \text{Uniform}(0, 1)$
   - If $u < \alpha$: accept $\theta^{(i)} = \theta^*$
   - Else: reject $\theta^{(i)} = \theta^{(i-1)}$
3. Return chain and acceptance rate

**Returns**:

- Array of samples (including burn-in)
- Acceptance rate

**Tuning**: Target acceptance rate 20-50%. Adjust proposal covariance to achieve this.

See the [Appendix] for more information on tuning your MCMC proposal distribution.

### The Multivariate Proposal Distribution

For the proposal distribution, you can use either:

- `np.random.default_rng().multivariate_normal(mean, cov)` - Modern NumPy random number generator (PCG64), recommended for reproducible chains (see <https://numpy.org/devdocs/reference/random/generated/numpy.random.Generator.multivariate_normal.html>).
- `np.random.multivariate_normal(mean, cov)` - Legacy global RNG, simpler but less control over reproducibility
- [scipy.stats.multivariate_normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) - More features (pdf, logpdf), but slight object overhead

For MCMC where you're drawing thousands of proposals, the NumPy options are typically faster. We recommend using `np.random.default_rng()` for better reproducibility and independent random streams across chains.

---

## Part 3: Convergence Diagnostics

**Module**: `diagnostics.py`

**Why build these now**: As you develop and tune your MCMC sampler, you need tools to assess whether it's working. These diagnostic functions are universal—they'll work for MCMC, HMC, and any other sampler you build in the future. Develop these tools alongside your MCMC implementation in Part 2, then reuse them for HMC in Part 4.

MCMC samples are correlated. You need to assess convergence and effective sample size.

**Required functions**:

- `compute_autocorrelation(chain)`
- `effective_sample_size(chain)`
- `gelman_rubin(chains)` (for multiple chains)

**Mathematical Background**:

**Autocorrelation**: Measures correlation between samples separated by lag $k$:
$$\rho_k = \frac{\mathbb{E}[(\theta_t - \bar{\theta})(\theta_{t+k} - \bar{\theta})]}{\text{Var}(\theta)}$$

**Effective Sample Size**:
$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{K}\rho_k}$$

where $K$ is chosen such that $\rho_k$ becomes negligible.

**Gelman-Rubin Statistic (split-$\hat{R}$)**: Compares within-chain to between-chain variance. For $m$ chains of length $n$, compute the potential scale reduction factor:
$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where $\hat{V}$ combines within-chain and between-chain variance, and $W$ is the average within-chain variance. Modern implementations use "split-$\hat{R}$" which splits each chain in half to increase the number of chains for more robust diagnostics.

Convergence is achieved when $\hat{R} < 1.1$ for all parameters.

See Module 5 Part 4 for implementation details and formulas.

---

## Part 4: Hamiltonian Monte Carlo

**Module**: `hmc.py`

**Before you start**: Make sure your MCMC (Part 2) is working and you have diagnostic tools (Part 3) functioning. You'll use the same `diagnostics.py` functions to assess your HMC chains—this demonstrates the power of modular design.

### 4.1: The Connection to N-body Simulation

HMC solves Hamilton's equations in parameter space. This is the same physics you implemented in Project 2!

**The Hamiltonian**:
$$H(\theta, p) = U(\theta) + K(p)$$

where:

- $U(\theta) = -\ln p(\theta|D)$ is the "potential energy" (negative log-posterior)
- $K(p) = \frac{1}{2}p^\top M^{-1} p$ is the "kinetic energy" (momentum term)
- $M$ is a "mass matrix" (typically identity or inverse covariance)

**Hamilton's equations**:
$$\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = M^{-1}p$$
$$\frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = -\nabla U(\theta) = \nabla \ln p(\theta|D)$$

These are exactly analogous to position and momentum evolution in N-body dynamics. **You can repurpose your leapfrog integrator from Project 2.**

---

### 4.2: Computing Gradients

HMC requires $\nabla_\theta \ln p(\theta|D)$. Two approaches:

**Finite differences**:
$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + h\mathbf{e}_i) - f(\theta - h\mathbf{e}_i)}{2h}$$

Use $h \sim 10^{-5}$ to $10^{-7}$.

**Automatic differentiation** (see Extensions for JAX):
More accurate and efficient, but requires learning JAX basics.

Implement gradient computation using at least one of these methods.

---

### 4.3: The HMC Algorithm

**Required function**:

- `hamiltonian_monte_carlo(log_prob_fn, grad_log_prob_fn, theta_init, epsilon, L, n_steps, args=())`

**Algorithm**:

1. Initialize: $\theta^{(0)} = \theta_{\text{init}}$
2. For $i = 1, 2, \ldots, N$:
   - Sample momentum: $p \sim \mathcal{N}(0, M)$
   - Set $(\theta, p) = (\theta^{(i-1)}, p)$
   - Simulate Hamiltonian dynamics for $L$ leapfrog steps:
     - For $j = 1, 2, \ldots, L$:
       - Half-step momentum: $p \leftarrow p - \frac{\epsilon}{2}\nabla U(\theta)$
       - Full-step position: $\theta \leftarrow \theta + \epsilon M^{-1}p$
       - Half-step momentum: $p \leftarrow p - \frac{\epsilon}{2}\nabla U(\theta)$
   - Compute energy change: $\Delta H = H(\theta_{\text{new}}, p_{\text{new}}) - H(\theta^{(i-1)}, p_{\text{init}})$
   - Compute acceptance probability: $\alpha = \min(1, \exp(-\Delta H))$
   - Draw $u \sim \text{Uniform}(0, 1)$
   - If $u < \alpha$: accept $\theta^{(i)} = \theta_{\text{new}}$
   - Else: reject $\theta^{(i)} = \theta^{(i-1)}$
3. Return chain and acceptance rate

**CRITICAL NOTE ON SIGNS**: Since $U(\theta) = -\ln p(\theta|D)$ and Hamilton's equations give $dp/dt = -\nabla U$, the leapfrog momentum updates use **subtraction**:
- $p \leftarrow p - \frac{\epsilon}{2}\nabla U(\theta)$

This is opposite from N-body simulation where you had gravitational acceleration entering with a negative sign. Here, $\nabla U$ points "uphill" in the negative log-posterior, so we subtract it to move "downhill" toward higher probability regions.

**Returns**:

- Array of samples
- Acceptance rate

**Tuning**:

- Step size $\epsilon$: Start with 0.01-0.1, adjust until energy is approximately conserved ($|\Delta H| \sim 0.1$-$1$)
- Trajectory length $L$: Start with 10-20 steps
- Target acceptance: 60-80% (higher than Metropolis-Hastings!)

**Resources**: Betancourt (2017), "A Conceptual Introduction to Hamiltonian Monte Carlo" ([arXiv:1701.02434](https://arxiv.org/abs/1701.02434))

---

## Part 5: Extensions (Choose 1, or Propose Your Own)

You must complete **one extension** beyond the baseline MCMC/HMC implementation. Discuss with the instructor if you want to propose something not listed here.

### Extension Option 1: JAX for Automatic Differentiation

Replace finite-difference gradients with automatic differentiation using JAX.

**Why**: JAX provides exact gradients (no approximation error) and is typically much faster than finite differences for gradient computation.

**Key JAX function**: `jax.grad()` computes gradients automatically. Example usage:

```python
import jax
grad_fn = jax.grad(your_function)
gradient = grad_fn(theta)
```

See the [JAX documentation](https://docs.jax.dev/en/latest/beginner_guide.html#beginner-guide) for details on making your functions JAX-compatible.

**Resources**:

- [JAX Beginner's Guide](https://docs.jax.dev/en/latest/beginner_guide.html#beginner-guide)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)

**What to implement**: JAX-compatible versions of your likelihood and gradient functions. Compare computational performance (wall-clock time) and accuracy to finite differences. Quantify the speedup and discuss when autodiff is worth the additional dependency.

---

### Extension Option 2: Non-Flat Universe

Relax the flatness assumption: let $\Omega_m$, $\Omega_\Lambda$, $h$ all vary independently (3 parameters instead of 2).

**Why**: Tests whether data alone can constrain spatial curvature $\Omega_K = 1 - \Omega_m - \Omega_\Lambda$.

**What changes**:

- Full luminosity distance formula (numerical integration required)
- 3-parameter MCMC/HMC (slower convergence)
- Prior ranges: $\Omega_m \in [0, 2]$, $\Omega_\Lambda \in [0, 2]$, $h \in [0.5, 0.9]$

**Physics**: 
$$D_L(z) = \frac{c(1+z)}{H_0\sqrt{|\Omega_K|}} S_K\left(\sqrt{|\Omega_K|} \int_0^z \frac{dz'}{E(z')}\right)$$

where $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda + \Omega_K(1+z)^2}$ and:

$$S_K(x) = \begin{cases}
\sinh(x) & \Omega_K > 0 \text{ (open)} \\
x & \Omega_K = 0 \text{ (flat)} \\
\sin(x) & \Omega_K < 0 \text{ (closed)}
\end{cases}$$

Compare your $\Omega_K$ constraint to Planck+BAO's measurement: $\Omega_K = 0.001 \pm 0.002$ (consistent with flat).

---

### Extension Option 3: No-U-Turn Sampler (NUTS)

Standard HMC requires manual tuning of trajectory length $L$. NUTS adapts $L$ dynamically by detecting when the trajectory starts doubling back.

**Why**: State-of-the-art method used in Stan and PyMC.

**What to implement**: Research the NUTS algorithm, implement it, compare performance to fixed-$L$ HMC.

**Resources**: Hoffman & Gelman (2014), "The No-U-Turn Sampler" ([arXiv:1111.4246](https://arxiv.org/abs/1111.4246))

---

### Extension Option 4: Importance Sampling with Planck Prior

Weight MCMC samples by an informative Gaussian prior on $h$ from Planck CMB measurements: $h \sim \mathcal{N}(0.674, 0.005^2)$.

**Why**: Demonstrates combining information from multiple datasets without re-running chains.

**What to implement**: Reweight posterior samples, compare results with/without prior. How much do constraints on $\Omega_m$ tighten?

---

### Extension Option 5: Propose Your Own

Have an idea? Want to explore something specific? Talk to the instructor. Possibilities include:
- Parallel tempering for multimodal posteriors
- Affine-invariant ensemble samplers (emcee algorithm)
- Comparison to nested sampling
- Bayesian model comparison (flat vs. non-flat)

---

## Part 6: Analysis and Results

### 6.1: Running Your Chains

**For both MCMC and HMC**:

1. Run multiple independent chains (at least 3-4) with different random seeds
2. Discard burn-in (first 20% is typical)
3. Check convergence using Gelman-Rubin (split-$\hat{R} < 1.1$)
4. Compute effective sample size
5. Pool chains for final posterior analysis

**Tuning guidance**:
- **MCMC**: Adjust proposal covariance until acceptance ~30%
- **HMC**: Adjust $\epsilon$ and $L$ until acceptance ~70% and energy approximately conserved

---

### 6.2: Interpreting Your Results

**Parameter estimates**:
- Compute mean and standard deviation for $\Omega_m$ and $h$
- Report credible intervals (16th, 50th, 84th percentiles)
- Compute correlation coefficient between parameters

**Comparison to literature**:
- Betoule et al. (2014) JLA: $\Omega_m = 0.295 \pm 0.034$ (flat, $h=0.7$)
- Planck 2018: $\Omega_m = 0.315 \pm 0.007$, $h = 0.674 \pm 0.005$
- SH0ES: $h = 0.730 \pm 0.010$ (Hubble Tension!)

Are your results consistent? How do uncertainties compare?

**Physical interpretation**:
- What does $\Omega_m \approx 0.3$ mean? (30% matter, 70% dark energy)
- Why are $\Omega_m$ and $h$ correlated?
- Where does your $h$ fall in the Hubble Tension debate?

---

### 6.3: Efficiency Comparison (MCMC vs. HMC)

**Metrics to compare**:
1. Effective samples per second: ESS / wall-clock time
2. Autocorrelation length: How many steps between independent samples?
3. Convergence speed: How quickly does $\hat{R} \to 1$?

**Expected**: HMC should be 5-20× more efficient than random-walk Metropolis-Hastings.

**Why?** HMC uses gradient information to propose distant points in high-probability regions, while MCMC random walks diffuse slowly.

---

## Part 7: Deliverables

### 7.1: Code Repository

Submit a well-organized code repository with the following structure:

```bash
project4/
├── src/
│   ├── cosmology.py
│   ├── likelihood.py
│   ├── mcmc.py
│   ├── hmc.py
│   └── diagnostics.py
├── scripts/
│   ├── run_mcmc.py
│   ├── run_hmc.py
│   └── analyze_results.py
├── tests/
│   ├── test_cosmology.py
│   ├── test_mcmc.py
│   └── test_hmc.py
├── outputs/
│   ├── chains/
│   └── figures/
├── README.md
└── requirements.txt
```

**Validation tests**: Include tests that verify:
- Forward model accuracy: 
  * $D_L(0) = 0$ (boundary condition)
  * Numerical integration vs. Pen approximation agree within 0.4% on a grid of $z\in[0,1.3]$ (e.g., 0, 0.1, …, 1.3)
  * Worked example: $\mu(z=0.5, \Omega_m=0.3, h=0.7) \approx 42.26 \pm 0.02$ mag
- MCMC Gaussian sanity check: Sample a known 2D Gaussian and compare sample mean/covariance to ground truth; target 20-50% acceptance after tuning
- HMC energy conservation: $\Delta H$ distribution centered near 0, acceptance 60-80% when tuned
- Gradient correctness: finite differences vs. analytic (when applicable), included in `test_hmc.py`

**README**: Document installation, usage, and how to reproduce your results.

---

### 7.2: Research Memo

Write a research memo (format and length at your discretion, typically 5-10 pages) that presents your analysis. Your memo should:

- **Compare MCMC vs. HMC performance**: Efficiency, convergence, practical considerations
- **Present cosmological parameter estimates**: Values, uncertainties, correlations
- **Interpret results physically**: What do these numbers mean for the universe?
- **Compare to literature**: How do your results relate to published values?
- **Discuss your extension**: What did you learn? How did it enhance the baseline analysis?

**Style**: This is a technical research memo — more formal than a lab report, but more concise than a journal article. Write for a peer who understands MCMC fundamentals but wants to learn your specific implementation choices (forward model approach, tuning methods, gradient computation) and scientific findings. Include technical details needed for reproducibility (chain lengths, acceptance rates, convergence metrics) and maintain formal scientific tone.

**Figures**: Include publication-quality plots (see Part 8 below). Each figure must have a detailed caption that enables standalone understanding: state what is plotted, include key technical details (e.g., "4 chains of 20,000 samples, 20% burn-in discarded"), and summarize the main takeaway. Captions are where you practice technical writing precision.

---

### 7.3: Growth Memo

Submit a **Growth Memo** (1-2 pages) reflecting on your learning process. Address:

- **Conceptual breakthroughs**: What "clicked" for you during this project?
- **Challenges overcome**: What was hardest? How did you work through it?
- **AI usage reflection**: How did AI tools help or hinder your learning? When was AI most/least useful?
- **Debugging strategies**: What debugging approaches proved most effective?
- **Connections made**: How did this project connect to other courses, research, or your broader learning?
- **Next steps**: What would you explore next if you had more time?

This memo is about **metacognition**—thinking about your thinking. Be honest and specific. See the course Growth Memo template for detailed guidance.

---

## Part 8: Visualization Requirements

Generate the following plots for **both MCMC and HMC** (where applicable):

### Required Plots:

1. **Trace plots**: Parameter values vs. iteration for each chain
   - Shows mixing, burn-in, stationarity
   - One plot per parameter, all chains overlaid with different colors
   - Include axis labels with units, legend identifying chains

2. **Autocorrelation plots**: $\rho_k$ vs. lag $k$ for each parameter
   - Shows how quickly samples become independent
   - Compare MCMC vs. HMC autocorrelation length explicitly
   - Mark where autocorrelation drops below 0.1 or 0.05

3. **Corner plot**: Joint and marginal posterior distributions
   - Use `corner.py` package if desired: `pip install corner` ([documentation](https://corner.readthedocs.io/))
   - Shows parameter correlations and credible intervals (16th, 50th, 84th percentiles)
   - Overlay MCMC and HMC results to compare (use transparency/different colors)

4. **Data vs. model**: $\mu(z)$ with observations and posterior predictions
   - Plot JLA data points with error bars
   - Overplot 50-100 random posterior samples as thin lines (shows uncertainty)
   - Include both MCMC and HMC posteriors (different colors/styles)
   - Add legend, axis labels with units

5. **Diagnostics**:
   - Gelman-Rubin (split-$\hat{R}$) vs. iteration (convergence monitoring)
   - Energy conservation for HMC: histogram of $\Delta H$ (should be centered near 0)
   - Acceptance rate vs. proposal scale/step size (for tuning demonstration)

6. **Efficiency comparison**: 
   - Bar chart or table comparing ESS/second for MCMC vs. HMC
   - Illustrative figure showing why HMC explores parameter space more efficiently (e.g., overlaid trace plots showing faster mixing)

**Caption requirements**: Each figure needs a caption with:
- What is plotted (axes, quantities)
- Technical parameters (number of chains, samples, burn-in discarded, acceptance rate)
- Key takeaway (what should reader learn from this figure?)

**Quality standards**: All plots should be publication-ready with:
- Clear labels, legible font sizes (12pt minimum for axes)
- Consistent color scheme across figures
- High resolution (300 DPI for raster images, or vector formats like PDF/SVG)
- Units clearly specified
- See course Matplotlib guidelines for detailed checklist

---

## Part 9: Tips and Debugging

### Common Issues:

**MCMC stuck or not mixing**:
- Check: Is log-posterior finite at initial point?
- Check: Is proposal covariance appropriate scale?
- Try: Different initial values, adjust proposal

**HMC rejecting everything**:
- Check: Is energy conserved? Plot $\Delta H$ distribution (should be centered near 0)
- Check: Are gradients correct? Test against finite differences
- Try: Reduce step size $\epsilon$ until $|\Delta H| < 1$

**Parameters hitting boundaries**:
- Check: Are prior bounds reasonable?
- Check: Is forward model correct? Validate numerical integration vs. Pen approximation and spot-check against Ned Wright's calculator
- Consider: Are you in a local mode? Try different initializations

**Chains not converging** ($\hat{R} > 1.1$):
- Run longer (more samples)
- Check for coding bugs (are you implementing the acceptance rule correctly? does the proposal distribution have the right structure?)
- Consider: Are there multimodal posteriors? (Shouldn't be for this problem, but good to check)

### Computational Efficiency:

- **Vectorize**: Compute $\mu$ for all 31 redshifts at once using NumPy arrays
- **Use Cholesky**: Factor covariance matrix once with `cho_factor`, reuse with `cho_solve` — much faster and more stable than inverting
- **Profile**: Use `cProfile` to find bottlenecks: `python -m cProfile -s cumtime your_script.py`
- **Parallelize**: Run multiple chains on different cores (optional, using `multiprocessing` or `joblib`)

---

## Appendix: Tuning MCMC Proposal Distribution

### Tuning Your MCMC Sampler: A Practical Guide

Getting good performance from Metropolis-Hastings requires tuning the proposal covariance $\Sigma_{\text{prop}}$. The goal is to find a proposal scale that balances exploration (proposing new regions) with efficiency (not wasting time on rejected proposals).

#### The Target: Acceptance Rate

**Aim for acceptance rate $r \in [0.2, 0.5]$** (20-50%)

**Why this range?**
- Too high (>50%): Steps too small → slow exploration, high autocorrelation
- Too low (<20%): Steps too large → many rejections, wasted computation
- Just right (20-50%): Efficient exploration of parameter space

#### Step 1: Choose a Reasonable Starting Point

Initialize your chain near physically plausible values: $\theta^{(0)} = (\Omega_m, h) \approx (0.30, 0.70)$

#### Step 2: Set Initial Proposal Scale

Start with a **diagonal** covariance matrix where each diagonal element reflects a small percentage of the parameter's typical scale:

$$\Sigma_{\text{prop}}^{(0)} = \begin{pmatrix} \sigma_{\Omega_m}^2 & 0 \\ 0 & \sigma_h^2 \end{pmatrix}$$

Reasonable starting values: $\sigma_{\Omega_m} \approx 0.05$ (5% of $\Omega_m$) and $\sigma_h \approx 0.02$ (2-3% of $h$).

**Physical intuition**: You're proposing to move by a few percent of each parameter's meaningful range per step.

#### Step 3: Iterate to Find Good Scale

The tuning process is iterative:

1. **Run a short test chain** (500-1000 steps)
2. **Compute acceptance rate** $r = N_{\text{accepted}} / N_{\text{total}}$
3. **Adjust proposal scale**:
   - If $r < 0.20$: Scale is too large → **shrink** $\Sigma_{\text{prop}}$ by factor $\gamma^2$ with $\gamma \in [0.6, 0.9]$
   - If $r > 0.50$: Scale is too small → **grow** $\Sigma_{\text{prop}}$ by factor $\gamma^2$ with $\gamma \in [1.1, 1.5]$
   - If $r \in [0.20, 0.50]$: You're in the target range → proceed to Step 4

**Mathematically**: $\Sigma_{\text{prop}}^{\text{new}} = \gamma^2 \Sigma_{\text{prop}}^{\text{old}}$

**Repeat** this adjustment process until acceptance falls in the target range.

#### Step 4 (Optional): Capture Parameter Correlations

Once you have reasonable acceptance with a diagonal proposal, you can optionally improve efficiency by accounting for correlations between $\Omega_m$ and $h$.

**The idea**: If parameters are correlated in the posterior, proposing along the correlation structure is more efficient than axis-aligned steps.

**Method**:
1. Run a longer warm-up chain (e.g., 5,000-10,000 steps) with your tuned diagonal $\Sigma_{\text{prop}}$
2. Discard burn-in (first 20-30% of samples)
3. Compute the **empirical covariance** $\widehat{\Sigma}$ from the warm-up samples
4. Use **Gelman's optimal scaling**:

$$\Sigma_{\text{prop}} = \frac{2.38^2}{d} \widehat{\Sigma}$$

where $d$ is the parameter dimension (here $d=2$).

**Why $2.38^2/d$?** This comes from asymptotic theory showing this scaling gives optimal acceptance (~23%) for multivariate Gaussian target distributions (Roberts, Gelman & Gilks 1997).

**Optional refinement** (if your warm-up sample is small or noisy): Apply shrinkage to regularize the off-diagonal terms:

$$\Sigma_{\text{prop}} = \frac{2.38^2}{d}\left[(1-\lambda)\widehat{\Sigma} + \lambda \cdot \text{diag}(\widehat{\Sigma})\right]$$

with $\lambda \in [0.05, 0.2]$. This interpolates between the full empirical covariance and a diagonal version.

#### Step 5: Run Production Chains

With your tuned $\Sigma_{\text{prop}}$ (either diagonal or with correlations), run multiple independent chains:
- Use different random seeds for each chain
- Keep $\Sigma_{\text{prop}}$ **fixed** (no more adaptation)
- Run long enough to ensure convergence (assess with diagnostics)

Monitor that acceptance rates remain in the 20-50% range across all chains.

#### Step 6: Verify Convergence

Use your diagnostic functions to ensure chains have converged:
- **Gelman-Rubin**: Split-$\hat{R} < 1.1$ for all parameters
- **Effective Sample Size**: ESS > 1000 effective samples per parameter
- **Trace plots**: Should show good mixing (fuzzy caterpillars, not sticky trends)
- **Autocorrelation**: Should decay to near-zero within reasonable lag

---

### Quick Tuning Workflow

```markdown
Initialize:  θ⁽⁰⁾ = (0.30, 0.70)
             Σ_prop = diag(0.05², 0.02²)

Test loop:   Run 500-1000 steps
             Compute acceptance rate r
             If r < 20%:  Shrink Σ_prop (multiply by 0.7²)
             If r > 50%:  Grow Σ_prop (multiply by 1.3²)
             If r ∈ [20%, 50%]: Exit loop

Optional:    Run warm-up (5000+ steps)
             Compute empirical covariance
             Apply 2.38²/d scaling

Production:  Run multiple chains (fixed Σ_prop)
             Monitor diagnostics
             Verify convergence
```

---

### Important Considerations

**Diagonal vs. correlated proposals**: For this 2-parameter problem, a well-tuned diagonal proposal is often sufficient. The correlation capture (Step 4) can improve efficiency but isn't strictly necessary for convergence.

**Hard boundaries**: Your flat priors have hard bounds ($\Omega_m \in [0, 1.2]$, $h \in [0.5, 0.9]$). Proposals outside these bounds return `-np.inf` log-posterior and are automatically rejected. Near boundaries, your effective acceptance rate will be lower—this is expected and not a problem.

**When to stop tuning**: You've tuned sufficiently when you achieve:
- Acceptance rate 20-50% ✓
- Good mixing in trace plots ✓  
- Split-$\hat{R} < 1.1$ across all parameters ✓
- ESS > 1000 effective samples ✓

**Don't over-tune**: Perfect tuning isn't necessary. MCMC is remarkably robust — even suboptimal proposals will eventually converge (just more slowly). Get "good enough" and move on.

---

### Computational Implementation Notes

You'll need to implement:
- A way to track accepted vs. proposed steps
- Functions to compute acceptance rate from your chain
- Logic to scale your proposal covariance (matrix multiplication by scalar)
- (Optional) Functions to compute empirical covariance from samples
- (Optional) Matrix operations for the shrinkage formula

Think about how to structure your tuning workflow: Will you build an automated tuning function, or tune interactively? How will you store and update $\Sigma_{\text{prop}}$ between tuning iterations?

---

*"Science is not about building a body of known 'facts'. It is a method for asking awkward questions and subjecting them to a reality-check, thus avoiding the human tendency to fallaciously reason from assumptions to conclusions."*  
— Terry Pratchett
