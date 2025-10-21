# Project 4: Building a Universal Inference Engine

**ASTR 596: Modeling the Universe**
**Instructor:** Dr. Anna Rosen
**Due**: Friday, November 7, 2025, 11:59 PM  
**AI Policy**: Phase 2 (strategic use after baseline implementation works)

> *"The most important questions of life are indeed, for the most part, really only problems of probability."*  
> — Pierre-Simon Laplace, *Théorie Analytique des Probabilités* (1812)

> *"It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts."*  
> — Arthur Conan Doyle, *A Scandal in Bohemia*

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

and $H_0 = 100h$ km/s/Mpc, $c = 3 \times 10^5$ km/s.

The distance modulus is:

$$\mu = 25 - 5\log_{10}(h) + 5\log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)$$

where $D_L^* \equiv D_L(h=1)$ is the luminosity distance with $h$ factored out.

**Implementation approaches**:

1. **Numerical integration**: Use `scipy.integrate.quad` to evaluate the integral directly
2. **Fitting formula**: Use the Pen (1999) approximation (accurate to 0.4% for $0.2 \leq \Omega_m \leq 1$):

$$D_L(z) = \frac{c}{H_0}(1+z)\left[\eta(1, \Omega_m) - \eta\left(\frac{1}{1+z}, \Omega_m\right)\right]$$

where:
$$\eta(a, \Omega_m) = 2\sqrt{s^3 + 1}\left[\frac{1}{a^4} - 0.1540\frac{s}{a^3} + 0.4304\frac{s^2}{a^2} + 0.19097\frac{s^3}{a} + 0.066941s^4\right]^{-1/8}$$

and $s^3 \equiv (1-\Omega_m)/\Omega_m$.

**What to do**: Implement **both** methods. The fitting formula is much faster (no numerical integration), but you should validate both approaches against each other to ensure they agree. Then experiment with timing: which method is fast enough for MCMC where you'll call this function thousands of times? You'll likely choose the approximation for production runs, but understanding both builds intuition about the trade-offs between accuracy and speed.

---

### 1.2: Likelihood Function

**Module**: `likelihood.py`

**Required functions**:

- `log_likelihood(theta, z_data, mu_data, Cinv)`
- `log_prior(theta)`
- `log_posterior(theta, z_data, mu_data, Cinv)`

**Mathematical Background**:

The log-likelihood for Gaussian errors with covariance $\mathbf{C}$ is:

$$\ln \mathcal{L}(\theta) = -\frac{1}{2}\sum_{i,j=1}^{n} r_i\, [\mathbf{C}^{-1}]_{ij}\, r_j$$

where the residual vector is:
$$r_i = \mu_i^{\text{obs}} - \mu_i^{\text{theory}}(z_i; \theta)$$

and $\theta = (\Omega_m, h)$.

**Implementation notes**:

- Invert $\mathbf{C}$ once at the start using `np.linalg.inv(C)`, then reuse `Cinv`
- Use flat (uniform) priors with physical bounds (think about reasonable ranges from the Scientific Background)
- Return `-np.inf` for parameters outside prior bounds

---

### 1.3: Data Loading

**Data files** (in same directory as this document):

- [`jla_mub.txt`](./jla_mub.txt): 31 data points $(z_i, \mu_i)$
- [`jla_mub_covmatrix.txt`](./jla_mub_covmatrix.txt): 31×31 covariance matrix (961 numbers in row-major order)

Implement data loading functions to read these files and prepare them for your likelihood function.

---

## Part 2: Metropolis-Hastings MCMC

**Module**: `mcmc.py`

This is **completely general**—it knows nothing about cosmology and should work for any inference problem.

**Required function**:

- `metropolis_hastings(log_prob_fn, theta_init, proposal_cov, n_steps, args=())`

**Algorithm**:

1. Initialize: $\theta^{(0)} = \theta_{\text{init}}$
2. For $i = 1, 2, \ldots, N$:
   - Propose: $\theta^* \sim \mathcal{N}(\theta^{(i-1)}, \Sigma_{\text{prop}})$
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

**Gelman-Rubin Statistic**: Compares within-chain to between-chain variance. For $m$ chains of length $n$:
$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

Convergence achieved when $\hat{R} < 1.1$.

See Module 5 Part 4 for implementation details.

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
- $K(p) = \frac{1}{2}p^T M^{-1} p$ is the "kinetic energy" (momentum term)
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
       - Half-step momentum: $p \leftarrow p + \frac{\epsilon}{2}\nabla U(\theta)$
       - Full-step position: $\theta \leftarrow \theta + \epsilon M^{-1}p$
       - Half-step momentum: $p \leftarrow p + \frac{\epsilon}{2}\nabla U(\theta)$
   - Compute energy change: $\Delta H = H(\theta_{\text{new}}, p_{\text{new}}) - H(\theta^{(i-1)}, p_{\text{init}})$
   - Compute acceptance probability: $\alpha = \min(1, \exp(-\Delta H))$
   - Draw $u \sim \text{Uniform}(0, 1)$
   - If $u < \alpha$: accept $\theta^{(i)} = \theta_{\text{new}}$
   - Else: reject $\theta^{(i)} = \theta^{(i-1)}$
3. Return chain and acceptance rate

**Returns**:

- Array of samples
- Acceptance rate

**Tuning**:

- Step size $\epsilon$: Start with 0.01-0.1, adjust until energy is approximately conserved ($|\Delta H| \sim 0.1$-$1$)
- Trajectory length $L$: Start with 10-20 steps
- Target acceptance: 60-80% (higher than Metropolis-Hastings!)

**Resources**: Betancourt (2017), "A Conceptual Introduction to
Hamiltonian Monte Carlo" ([arXiv:1701.02434](https://arxiv.org/abs/1701.02434))

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
- Prior ranges: $0 < \Omega_m < 2$, $0 < \Omega_\Lambda < 2$

**Physics**: 
$$D_L(z) = \frac{c(1+z)}{H_0\sqrt{|\Omega_K|}} S_k\left(\sqrt{|\Omega_K|} \int_0^z \frac{dz'}{E(z')}\right)$$

where $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda + \Omega_K(1+z)^2}$ and:

$$S_k(x) = \begin{cases}
\sin(x) & \Omega_K > 0 \text{ (closed)} \\
x & \Omega_K = 0 \text{ (flat)} \\
\sinh(x) & \Omega_K < 0 \text{ (open)}
\end{cases}$$

Compare your $\Omega_K$ constraint to Planck's $|\Omega_K| < 0.001$.

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
3. Check convergence using Gelman-Rubin ($\hat{R} < 1.1$)
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
- Forward model accuracy: numerical integration vs. Pen approximation (should agree to ~0.4% or better). You can also spot-check against Ned Wright's calculator, but validating your two implementations against each other is more important.
- MCMC detailed balance (acceptance ratio symmetry)
- HMC energy conservation ($|\Delta H|$ distribution)
- Gradient correctness (finite diff vs. analytic when possible)

**README**: Document installation, usage, and how to reproduce your results.

---

### 7.2: Research Memo

Write a research memo (format and length at your discretion) that presents your analysis. Your memo should:

- **Compare MCMC vs. HMC performance**: Efficiency, convergence, practical considerations
- **Present cosmological parameter estimates**: Values, uncertainties, correlations
- **Interpret results physically**: What do these numbers mean for the universe?
- **Compare to literature**: How do your results relate to published values?
- **Discuss your extension**: What did you learn? How did it enhance the baseline analysis?

**Style:** This is a technical research memo — more formal than a lab report, but more concise than a journal article. Write for a peer who understands MCMC fundamentals but wants to learn your specific implementation choices (forward model approach, tuning methods, gradient computation) and scientific findings. Include technical details needed for reproducibility (chain lengths, acceptance rates, convergence metrics) and maintain formal scientific tone. 

**Figures**: Include publication-quality plots (see Part 8 below). Each figure must have a detailed caption that enables standalone understanding: state what is plotted, include key technical details (e.g., "4 chains of 20,000 samples, 20% burn-in discarded"), and summarize the main takeaway. Captions are where you practice technical writing precision.

---

## Part 8: Visualization Requirements

Generate the following plots for **both MCMC and HMC** (where applicable):

### Required Plots:

1. **Trace plots**: Parameter values vs. iteration for each chain
   - Shows mixing, burn-in, stationarity
   - One plot per parameter, all chains overlaid

2. **Autocorrelation plots**: $\rho_k$ vs. lag $k$ for each parameter
   - Shows how quickly samples become independent
   - Compare MCMC vs. HMC autocorrelation length

3. **Corner plot**: Joint and marginal posterior distributions
   - Use `corner.py` package if desired: `pip install corner` ([documentation](https://corner.readthedocs.io/))
   - Shows parameter correlations and credible intervals
   - Overlay MCMC and HMC results to compare

4. **Data vs. model**: $\mu(z)$ with observations and posterior predictions
   - Plot JLA data points with error bars
   - Overplot 50-100 random posterior samples as thin lines (shows uncertainty)
   - Include both MCMC and HMC posteriors

5. **Diagnostics**:
   - Gelman-Rubin $\hat{R}$ vs. iteration (convergence monitoring)
   - Energy conservation for HMC: histogram of $\Delta H$
   - Acceptance rate vs. proposal scale (for tuning demonstration)

6. **Efficiency comparison**: 
   - Bar chart or table comparing ESS/second for MCMC vs. HMC
   - Illustrative figure showing why HMC explores parameter space more efficiently

**Tip**: All plots should have clear labels, legends, and be publication-ready. Consider using a consistent color scheme and style.

---

## Part 9: Tips and Debugging

### Common Issues:

**MCMC stuck or not mixing**:
- Check: Is log-posterior finite at initial point?
- Check: Is proposal covariance appropriate scale?
- Try: Different initial values, adjust proposal

**HMC rejecting everything**:
- Check: Is energy conserved? Plot $\Delta H$ distribution
- Check: Are gradients correct? Test against finite differences
- Try: Reduce step size $\epsilon$

**Parameters hitting boundaries**:
- Check: Are prior bounds reasonable?
- Check: Is forward model correct? Validate numerical integration vs. Pen approximation (and optionally spot-check against Ned Wright's calculator)
- Consider: Are you in a local mode? Try different initializations

**Chains not converging** ($\hat{R} > 1.1$):
- Run longer (more samples)
- Check for coding bugs (detailed balance violated?)
- Consider: Are there multimodal posteriors? (Shouldn't be for this problem)

### Computational Efficiency:

- **Vectorize**: Compute $\mu$ for all 31 redshifts at once
- **Cache**: Invert covariance matrix once, reuse `Cinv`
- **Profile**: Use `cProfile` to find bottlenecks
- **Parallelize**: Run multiple chains on different cores (optional)

---

*"Science is not about building a body of known 'facts'. It is a method for asking awkward questions and subjecting them to a reality-check, thus avoiding the human tendency to fallaciously reason from assumptions to conclusions."*  
— Terry Pratchett
