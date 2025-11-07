---
title: "Part II: Gaussian Processes for Scientific Emulation"
subtitle: "Why the Universe Looks the Way It Does | The Learnable Universe Module 3 | ASTR 596"
---

> "A Gaussian process is a distribution over functions, where any finite collection of function values has a joint Gaussian distribution."
> 
> — Carl Edward Rasmussen & Christopher K. I. Williams, *Gaussian Processes for Machine Learning*

> "The practical question is not whether to use a model, but which model to use."
> 
> — George E. P. Box

---

## Learning Objectives

By the end of this module, you will be able to:

1. **LO1 - Conceptual Understanding**: Explain Gaussian Processes as probability distributions over functions and connect this to Bayesian inference from Module 5
2. **LO2 - Mathematical Foundation**: Derive the GP predictive distribution from first principles using multivariate Gaussian conditioning
3. **LO3 - Kernel Design**: Understand how kernel functions encode prior beliefs about smoothness, periodicity, and structure in physical systems
4. **LO4 - Practical Implementation**: Implement GP regression from scratch in JAX and use it to emulate expensive astrophysical simulations
5. **LO5 - Uncertainty Quantification**: Interpret and validate GP predictive uncertainty, distinguishing epistemic from aleatoric uncertainty
6. **LO6 - Method Limitations**: Recognize when GPs fail (high dimensions, non-smooth functions, computational constraints) and articulate why
7. **LO7 - Scientific Application**: Apply GPs to surrogate modeling of star cluster evolution, connecting to your N-body simulations from Projects 2 and 5

---

## The Big Picture: Why Gaussian Processes?

### The Fundamental Problem

You've spent this semester building computational models from first principles:

- **Project 2**: N-body simulations of star cluster dynamics
- **Project 3**: Monte Carlo radiative transfer simulations  
- **Project 4**: MCMC sampling of posterior distributions
- **Project 5**: Fast JAX implementations of these algorithms

Each of these simulators encodes physics: gravity, radiation, statistical mechanics and cosmology. But they share a common challenge: **they're expensive to run**.

Consider your N-body simulation:

- Input: Initial conditions (masses, positions, velocities, virial ratio, concentration)
- Output: Evolutionary trajectory (core radius, velocity dispersion, bound mass fraction over time)
- Cost: Minutes to hours per simulation

Now suppose you want to:

1. **Explore parameter space**: Run 10,000 simulations with different initial conditions
2. **Infer initial conditions**: Use Bayesian inference (Project 4 style) to find ICs that match observations
3. **Optimize designs**: Find initial conditions that maximize cluster lifetime

Each requires evaluating your simulator thousands or millions of times. Even with JAX's speed, this is prohibitive.

**The solution**: Build a *surrogate model* — a fast approximation that captures the input-output relationship learned from a modest number of expensive simulations.

:::{admonition} Connection to Module 5: Bayesian Inference
:class: note

In Project 4, you used MCMC to sample posterior distributions: $p(\theta | \mathcal{D})$. Each MCMC step required evaluating the likelihood $p(\mathcal{D} | \theta)$, which was cheap (just a probability calculation).

But what if evaluating the likelihood requires *running a simulation*? Then MCMC becomes intractable. GPs offer a solution: replace the expensive simulator with a fast emulator, then run MCMC on the emulator. This is called **Bayesian optimization** or **simulation-based inference**.
:::

### What is a Gaussian Process?

At its core, a Gaussian Process (GP) is a **probability distribution over functions**. Just as a Gaussian distribution describes uncertainty over scalars or vectors, a GP describes uncertainty over *entire functions* $f: \mathbb{R}^D \to \mathbb{R}$.

**Intuition**: Imagine you have a function $f(x)$ that you don't know completely, but you have some observations $(x_i, y_i)$ where $y_i = f(x_i) + \epsilon_i$. A GP gives you:
1. A **mean function** $\mu(x)$: Your best guess for $f(x)$ at any point $x$
2. A **variance function** $\sigma^2(x)$: Your uncertainty about $f(x)$ at point $x$
3. A **covariance function** $k(x, x')$: How much knowing $f(x)$ tells you about $f(x')$

This is exactly the Bayesian framework from Module 5:
- **Prior**: GP before seeing data (smooth functions? periodic? what lengthscale?)
- **Likelihood**: How observations relate to true function (measurement noise)
- **Posterior**: GP after conditioning on data (updated beliefs about $f$)

```{admonition} The More You Know: GPs in Machine Learning History
:class: tip, dropdown

Gaussian Processes have a rich history:
- **1940s-60s**: Developed in geostatistics as "Kriging" (named after Danie Krige, South African mining engineer)
- **1990s**: Formalized in ML by Neal, Williams, Rasmussen
- **2000s**: Applied to hyperparameter optimization (Bayesian optimization)
- **2010s**: Scaled to large datasets via sparse approximations and GPU acceleration
- **2020s**: Used in simulation-based inference, physics-informed models, and neural network theory

In astrophysics, GPs have been used for:
- Cosmological parameter inference (Planck mission)
- Exoplanet detection (Gaussian process regression on stellar variability)
- Supernova light curve interpolation
- Surrogate modeling of expensive simulations (exactly what you'll do!)
```

---

## 🔴 Part 1: From Gaussian Distributions to Gaussian Processes

### Reviewing Multivariate Gaussians

Before we tackle infinite-dimensional function spaces, let's revisit what you know from Module 1: the **multivariate Gaussian distribution**.

A random vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ follows a multivariate Gaussian if its probability density is:

$$
p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^n$: Mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$: Covariance matrix (symmetric, positive definite)

**Key properties** (recall from Module 1):
1. **Marginals are Gaussian**: If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then any subset of components is also Gaussian
2. **Conditionals are Gaussian**: Given some observations, the distribution over unobserved variables is Gaussian
3. **Linear transformations are Gaussian**: If $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$, then $\mathbf{y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$

These properties make Gaussians **analytically tractable** — we can compute posteriors, marginals, and predictions in closed form. This is why GPs are so powerful!

```{admonition} Connection to Module 1: Central Limit Theorem
:class: note

Remember the CLT from Module 1? It stated that sums of independent random variables converge to Gaussian distributions. This is why Gaussians appear everywhere in nature: any quantity influenced by many small, independent effects tends toward Gaussian.

GPs extend this: if a function is the result of many small, independent contributions (think basis function expansion), then our uncertainty about that function is Gaussian. This is the **function-space view** of GPs.
```

### Gaussian Conditioning: The Core of GP Regression

The most important property for GPs is **Gaussian conditioning**. Suppose we partition a Gaussian random vector into observed and unobserved parts:

$$
\begin{bmatrix} \mathbf{f}_1 \\ \mathbf{f}_2 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix} \right)
$$

If we **observe** $\mathbf{f}_1 = \mathbf{y}_1$, what is the distribution of the unobserved $\mathbf{f}_2$?

**Theorem (Gaussian Conditioning)**:

$$
p(\mathbf{f}_2 | \mathbf{f}_1 = \mathbf{y}_1) = \mathcal{N}(\mathbf{f}_2 | \boldsymbol{\mu}_{2|1}, \boldsymbol{\Sigma}_{2|1})
$$

where:

$$
\begin{align}
\boldsymbol{\mu}_{2|1} &= \boldsymbol{\mu}_2 + \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1} (\mathbf{y}_1 - \boldsymbol{\mu}_1) \\
\boldsymbol{\Sigma}_{2|1} &= \boldsymbol{\Sigma}_{22} - \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1} \boldsymbol{\Sigma}_{12}
\end{align}
$$

**Interpretation**:
- $\boldsymbol{\mu}_{2|1}$: Posterior mean (best estimate of $\mathbf{f}_2$ given observations)
- $\boldsymbol{\Sigma}_{2|1}$: Posterior covariance (remaining uncertainty after seeing data)
- $\boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1}$: Regression weights (how much each observation influences prediction)

This formula is **the entire machinery of GP regression**! Everything else is just specifying the prior covariance structure.

```{admonition} Derivation: Gaussian Conditioning (Optional Deep Dive)
:class: tip, dropdown

To derive the conditional distribution, we complete the square in the joint Gaussian density. Starting with:

$$
p(\mathbf{f}_1, \mathbf{f}_2) = \mathcal{N}\left( \begin{bmatrix} \mathbf{f}_1 \\ \mathbf{f}_2 \end{bmatrix} \Bigg| \begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix} \right)
$$

The conditional $p(\mathbf{f}_2 | \mathbf{f}_1)$ is proportional to the joint (treating $\mathbf{f}_1$ as fixed). Using the matrix inversion lemma (Woodbury identity), we can show:

$$
(\boldsymbol{\Sigma})^{-1} = \begin{bmatrix} \boldsymbol{\Lambda}_{11} & \boldsymbol{\Lambda}_{12} \\ \boldsymbol{\Lambda}_{21} & \boldsymbol{\Lambda}_{22} \end{bmatrix}
$$

where $\boldsymbol{\Lambda}_{22} = (\boldsymbol{\Sigma}_{22} - \boldsymbol{\Sigma}_{21}\boldsymbol{\Sigma}_{11}^{-1}\boldsymbol{\Sigma}_{12})^{-1}$ is the **precision matrix** of the conditional.

Completing the square in the exponential and extracting terms involving $\mathbf{f}_2$ yields the conditional mean and covariance formulas above. See Chapter 2 of Bishop's *Pattern Recognition and Machine Learning* for the full derivation.
```

### From Finite to Infinite Dimensions: The Function-Space View

Now comes the conceptual leap: **extend from finite to infinite dimensions**.

Instead of a vector $\mathbf{f} = [f(x_1), f(x_2), \ldots, f(x_n)]^T$ at $n$ specific points, consider the **entire function** $f: \mathcal{X} \to \mathbb{R}$ where $\mathcal{X}$ is your input space (could be $\mathbb{R}^D$, could be more exotic).

**Definition: Gaussian Process**

A **Gaussian Process** is a collection of random variables, any finite number of which have a joint Gaussian distribution.

More formally: $f \sim \mathcal{GP}(m, k)$ means that for any finite collection of points $\mathbf{x} = \{x_1, \ldots, x_n\}$, the function values have a joint Gaussian distribution:

$$
\mathbf{f} = \begin{bmatrix} f(x_1) \\ f(x_2) \\ \vdots \\ f(x_n) \end{bmatrix} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})
$$

where:
- $m: \mathcal{X} \to \mathbb{R}$ is the **mean function**: $[\mathbf{m}]_i = m(x_i)$
- $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is the **covariance function** (or **kernel**): $[\mathbf{K}]_{ij} = k(x_i, x_j)$

**Why is this useful?**

1. **Priors over functions**: The GP encodes your prior beliefs about what functions are plausible (smooth? periodic? what lengthscale?)
2. **Bayesian inference**: Observing data at some points conditions the GP, giving a posterior distribution over functions
3. **Prediction with uncertainty**: At any new point $x_*$, the posterior gives both a mean prediction and an uncertainty estimate
4. **Tractability**: Despite being infinite-dimensional, we only ever work with finite subsets (the points we care about), which are multivariate Gaussians

```{admonition} The Philosophical Depth: Priors Over Functions
:class: important

This is a profound shift in thinking from Module 5. In Bayesian parameter inference, you had:
- **Parameters**: Finite-dimensional $\theta \in \mathbb{R}^p$
- **Prior**: $p(\theta)$ - beliefs about parameter values
- **Posterior**: $p(\theta | \mathcal{D})$ - updated beliefs

With GPs, you have:
- **Function**: Infinite-dimensional $f: \mathcal{X} \to \mathbb{R}$
- **Prior**: $\mathcal{GP}(m, k)$ - beliefs about function properties (smoothness, structure)
- **Posterior**: Another GP - updated beliefs

This is called **non-parametric Bayesian inference** because you're not inferring a finite set of parameters, but rather an entire function. The "parameters" (kernel hyperparameters) control the structure of function space, not the function itself.

In a sense, every function is a "hypothesis" about the data-generating process, and the GP posterior is a probability distribution over these hypotheses. This is Bayesian inference at its most general!
```

---

## 🔴 Part 2: Covariance Functions (Kernels)

The **kernel** $k(x, x')$ is the heart of a GP. It encodes **all your prior knowledge** about the function you're modeling:
- How smooth is it?
- Is it periodic?
- What's the typical lengthscale of variation?
- Are there local vs global patterns?

### Properties of Valid Kernels

For $k$ to be a valid covariance function, it must produce **positive definite** covariance matrices for any set of points. That is, for any $\{x_1, \ldots, x_n\}$ and any vector $\mathbf{a} \in \mathbb{R}^n$:

$$
\sum_{i=1}^n \sum_{j=1}^n a_i a_j k(x_i, x_j) \geq 0
$$

**Why?** Because $\mathbf{K}$ must be a valid covariance matrix, and covariance matrices are positive semi-definite by definition: $\text{Var}(\mathbf{a}^T \mathbf{f}) = \mathbf{a}^T \mathbf{K} \mathbf{a} \geq 0$.

This is a strong constraint! Not all functions $k(x, x')$ are valid kernels. Fortunately, there are:
1. **Building blocks**: A few fundamental kernels that are provably valid
2. **Composition rules**: Ways to combine valid kernels to create more complex valid kernels

```{admonition} Connection to Module 3: Correlation Functions
:class: note

In Module 3 (stellar dynamics), you encountered **two-point correlation functions** $\xi(r)$ that describe spatial clustering of stars. These are closely related to kernels!

A correlation function describes how the density at two points is related: $\langle \rho(x) \rho(x') \rangle \propto 1 + \xi(|x - x'|)$. This is exactly the covariance structure that a kernel encodes.

In fact, many kernels are **stationary** (depend only on $|x - x'|$) just like correlation functions. The mathematical machinery is the same: both describe how information propagates through space.
```

### The Squared Exponential (RBF) Kernel

The most commonly used kernel is the **Radial Basis Function** (RBF) or **Squared Exponential** kernel:

$$
k_{\text{SE}}(x, x') = \sigma_f^2 \exp\left( -\frac{\|x - x'\|^2}{2\ell^2} \right)
$$

**Parameters**:
- $\sigma_f^2$: **Output variance** - controls the typical amplitude of function variations
- $\ell$: **Lengthscale** - controls how quickly correlations decay with distance

**Properties**:
- **Stationary**: Depends only on $|x - x'|$ (translation invariant)
- **Isotropic**: Depends only on Euclidean distance (rotation invariant)
- **Infinitely differentiable**: Functions drawn from a GP with SE kernel are *infinitely smooth* (all derivatives exist)
- **Short-range**: Correlations decay exponentially, so distant points are nearly independent

**Interpretation**: Two function values $f(x)$ and $f(x')$ are highly correlated if $|x - x'| \ll \ell$, and nearly uncorrelated if $|x - x'| \gg \ell$. The lengthscale $\ell$ defines what "nearby" means for your problem.

**When to use it**: 
- Smooth underlying functions (no sharp discontinuities)
- No strong prior knowledge about structure
- Default choice for exploration

**Limitations**:
- Too smooth for some real functions (e.g., functions with kinks or discontinuities)
- All lengthscales are the same (might want different smoothness in different directions)

```{admonition} The More You Know: Visualization of Kernels
:class: tip, dropdown

The kernel $k(x, x')$ defines a *covariance matrix* over function values. To visualize it:

1. **Covariance matrix**: For points $x_1, \ldots, x_n$, plot the matrix $[\mathbf{K}]_{ij} = k(x_i, x_j)$. This shows which function values are correlated.

2. **Sample functions**: Draw samples from $f \sim \mathcal{GP}(0, k)$ by:
   - Choose evaluation points $x_1, \ldots, x_n$
   - Compute covariance matrix $\mathbf{K}$
   - Sample $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$ (multivariate Gaussian)
   - Plot $(x_i, f_i)$ pairs

Different kernels produce characteristically different function samples:
- SE kernel: Very smooth, infinitely differentiable
- Matérn kernel: Controllable smoothness (can be rough)
- Periodic kernel: Repeating patterns

You'll implement this visualization in your project!
```

### The Matérn Kernel Family

The **Matérn kernel** generalizes the SE kernel to allow functions with finite differentiability:

$$
k_{\text{Matérn}}(x, x') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \sqrt{2\nu} \frac{|x - x'|}{\ell} \right)^\nu K_\nu\left( \sqrt{2\nu} \frac{|x - x'|}{\ell} \right)
$$

where:
- $\nu > 0$: **Smoothness parameter** - controls differentiability
- $K_\nu$: Modified Bessel function of the second kind
- $\Gamma$: Gamma function

**Key special cases**:
- $\nu = 1/2$: **Exponential kernel** (continuous but not differentiable)
  $$k(x, x') = \sigma_f^2 \exp\left( -\frac{|x - x'|}{\ell} \right)$$
  
- $\nu = 3/2$: **Once differentiable**
  $$k(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{3}|x - x'|}{\ell}\right) \exp\left( -\frac{\sqrt{3}|x - x'|}{\ell} \right)$$
  
- $\nu = 5/2$: **Twice differentiable**
  $$k(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{5}|x - x'|}{\ell} + \frac{5|x - x'|^2}{3\ell^2}\right) \exp\left( -\frac{\sqrt{5}|x - x'|}{\ell} \right)$$
  
- $\nu \to \infty$: Converges to **SE kernel** (infinitely differentiable)

**Why is this useful?** Real physical processes are often **not infinitely smooth**. For example:
- Star cluster core radius might have kinks due to encounters
- Stellar mass functions have features at characteristic masses
- Radiative transfer solutions can have discontinuities at optically thick/thin boundaries

The Matérn kernel lets you match the smoothness of your prior to the physics!

**Practical advice**: In most applications, $\nu = 5/2$ is a good default. It's more flexible than SE but not so rough that functions are jagged. The formulas are also simpler (no Bessel functions to compute).

```{admonition} Connection to Physics: Roughness and Timescales
:class: note

The smoothness parameter $\nu$ has a physical interpretation. Consider a time-series $f(t)$ representing some physical quantity (e.g., stellar luminosity):

- $\nu = 1/2$: **Ornstein-Uhlenbeck process** - like Brownian motion (white noise integrated once). This is the solution to $df/dt = -f/\tau + \sigma \xi(t)$ where $\xi$ is white noise. Physical example: damped random walks.

- $\nu = 3/2$: **Velocity is continuous** but acceleration can be discontinuous. Physical example: particle motion with impulsive forces (like gravitational encounters in clusters!).

- $\nu = 5/2$: **Acceleration is continuous** but jerk can be discontinuous. Physical example: smooth orbital dynamics with occasional perturbations.

- $\nu \to \infty$: **Infinitely smooth** - all derivatives exist. Physical example: Kepler orbits, analytical solutions to smooth potentials.

So choosing $\nu$ is actually choosing what timescales of variability you expect in the physics!
```

### Periodic Kernels

If your function has **periodic structure** (e.g., orbital motion, stellar pulsations, seasonal effects), you can encode this with a **periodic kernel**:

$$
k_{\text{periodic}}(x, x') = \sigma_f^2 \exp\left( -\frac{2\sin^2(\pi |x - x'| / p)}{\ell^2} \right)
$$

where:
- $p$: **Period** - repeats every distance $p$
- $\ell$: **Lengthscale** - controls smoothness within each period

**Properties**:
- Exactly periodic: $k(x, x') = k(x + np, x' + mp)$ for any integers $n, m$
- Smooth within periods (controlled by $\ell$)
- Can combine with other kernels for quasi-periodic behavior

**Astrophysical examples**:
- Exoplanet detection: stellar variability has periodic component from rotation
- Cepheid variables: periodic light curves with varying amplitude
- Binary star systems: orbital modulation

**Limitation**: Exactly periodic, which real data rarely is. Solution: **combine** with aperiodic kernels!

### Combining Kernels: Sum and Product Rules

You can build complex kernels from simple ones using:

**1. Addition (Sum Rule)**:
$$k_{\text{sum}}(x, x') = k_1(x, x') + k_2(x, x')$$

**Interpretation**: Functions drawn from $\mathcal{GP}(0, k_{\text{sum}})$ can be decomposed as $f = f_1 + f_2$ where $f_i \sim \mathcal{GP}(0, k_i)$ independently.

**Use case**: Multiple sources of variation at different lengthscales
- Example: $k_{\text{SE}}(\ell_{\text{short}}) + k_{\text{SE}}(\ell_{\text{long}})$ models both local wiggles and global trends

**2. Multiplication (Product Rule)**:
$$k_{\text{prod}}(x, x') = k_1(x, x') \cdot k_2(x, x')$$

**Interpretation**: Creates modulated patterns - one kernel modulates the amplitude of another.

**Use case**: Quasi-periodic functions (periodicity with varying amplitude)
- Example: $k_{\text{periodic}} \times k_{\text{SE}}$ models periodicity that drifts or decays

**Common composite kernels for astrophysics**:

1. **Trend + wiggles**:
   $$k = k_{\text{linear}}(x, x') + k_{\text{SE}}(x, x'; \ell_{\text{short}})$$
   - Linear trend (long timescale) + short-timescale variations
   - Example: Stellar evolution with short-term variability

2. **Multi-scale structure**:
   $$k = \sum_{i=1}^3 k_{\text{SE}}(x, x'; \ell_i)$$
   - Multiple lengthscales (large, medium, small)
   - Example: Galaxy clustering (large-scale structure + groups + individual galaxies)

3. **Decaying periodicity**:
   $$k = k_{\text{periodic}}(x, x'; p) \times k_{\text{SE}}(x, x'; \ell_{\text{decay}})$$
   - Periodic oscillations with exponentially decaying amplitude
   - Example: Damped pulsations in stars

```{admonition} Warning: Kernel Complexity vs Data
:class: warning

There's a trade-off between kernel complexity and data requirements:
- **Simple kernels** (e.g., single SE): Few parameters, easy to fit with limited data
- **Complex kernels** (e.g., sum of 5 components): Many parameters, risk overfitting

**Rule of thumb**: You need roughly 10-20 data points per kernel hyperparameter to fit reliably. With 100 training simulations, stick to kernels with ~5-10 hyperparameters total.

For your cluster evolution project, start with:
- Single SE kernel: 2 parameters ($\sigma_f^2$, $\ell$) per input dimension
- If 5D input space: ~10-15 parameters total (one lengthscale per dimension)
- Total data: 100-500 simulations → sufficient for this complexity

Don't over-engineer the kernel until you have evidence from the data that you need more complexity!
```

### Automatic Relevance Determination (ARD)

For multi-dimensional inputs $\mathbf{x} = [x^{(1)}, x^{(2)}, \ldots, x^{(D)}]^T$, you can use **different lengthscales** for each dimension:

$$
k_{\text{ARD}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left( -\frac{1}{2} \sum_{d=1}^D \frac{(x^{(d)} - x'^{(d)})^2}{\ell_d^2} \right)
$$

This is called **Automatic Relevance Determination** because the lengthscales $\ell_d$ automatically determine which dimensions are *relevant*:
- Large $\ell_d$: Dimension $d$ doesn't affect the output much (smooth, nearly constant)
- Small $\ell_d$: Dimension $d$ is important (output varies quickly with $x^{(d)}$)

**For your cluster evolution project**:
- Inputs: $N_{\star}$, $M_{\text{total}}$, $Q_{\text{virial}}$, concentration $c$, ...
- ARD kernel learns: "Core radius depends strongly on concentration ($\ell_c$ small) but weakly on total mass ($\ell_M$ large)"
- This gives **scientific insight** about which initial conditions matter most!

---

## 🔴 Part 3: Gaussian Process Regression

Now we have the tools to do **GP regression**: Given observations, predict function values at new points with uncertainty estimates.

### The Setup

**Observed data**: 
- Inputs: $\mathbf{X} = \{x_1, \ldots, x_n\}$ (e.g., initial conditions for $n$ simulations)
- Outputs: $\mathbf{y} = [y_1, \ldots, y_n]^T$ (e.g., core radius at $t=100$ Myr)
- Noise model: $y_i = f(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$

**Goal**: Predict $f_*$ at new input $x_*$ (with uncertainty)

**Prior**: $f \sim \mathcal{GP}(m, k)$ with mean function $m$ (often $m \equiv 0$) and kernel $k$

### Step 1: Joint Distribution (Prior)

Before seeing data, the joint distribution of observed function values $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^T$ and test function value $f_* = f(x_*)$ is:

$$
\begin{bmatrix} \mathbf{f} \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m} \\ m_* \end{bmatrix}, \begin{bmatrix} \mathbf{K} & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix} \right)
$$

where:
- $[\mathbf{K}]_{ij} = k(x_i, x_j)$: Covariance between training points
- $[\mathbf{k}_*]_i = k(x_i, x_*)$: Covariance between training and test point
- $k_{**} = k(x_*, x_*)$: Prior variance at test point (usually $\sigma_f^2$)

### Step 2: Account for Noise

We don't observe $f$ directly; we observe $\mathbf{y} = \mathbf{f} + \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 \mathbf{I})$.

The joint distribution of observations and test point is:

$$
\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m} \\ m_* \end{bmatrix}, \begin{bmatrix} \mathbf{K} + \sigma_n^2 \mathbf{I} & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix} \right)
$$

**Key insight**: Noise adds to the diagonal of the covariance matrix: $\mathbf{K} \to \mathbf{K} + \sigma_n^2 \mathbf{I}$. This makes the matrix better conditioned (more numerically stable) and encodes uncertainty in observations.

### Step 3: Condition on Observations (Posterior)

Using the Gaussian conditioning formula from Part 1, the **posterior predictive distribution** is:

$$
p(f_* | x_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(f_* | \mu_*, \sigma_*^2)
$$

where:

$$
\begin{align}
\mu_* &= m_* + \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m}) \\
\sigma_*^2 &= k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
\end{align}
$$

**This is GP regression!** Two simple formulas that give you:
1. **Mean prediction** $\mu_*$: Best estimate of $f(x_*)$ given data
2. **Uncertainty** $\sigma_*^2$: How confident you are in that prediction

Let's unpack what these formulas *mean*:

### Understanding the Predictive Mean

$$
\mu_* = m_* + \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m})
$$

**Decomposition**:
- $m_*$: Prior mean (your guess before seeing data)
- $\mathbf{y} - \mathbf{m}$: Residuals (how much data deviates from prior)
- $(\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1}$: Weights residuals by noise and covariances
- $\mathbf{k}_*^T$: Projects weighted residuals to test point using kernel similarities

**Interpretation**: The prediction is a **weighted average** of observed values, where weights depend on:
1. **Similarity**: How correlated is $x_*$ with each training point? (via $\mathbf{k}_*$)
2. **Reliability**: How noisy are the observations? (via $\sigma_n^2$)
3. **Internal consistency**: How correlated are training points with each other? (via $\mathbf{K}$)

**Limiting cases**:
- If $x_*$ is very close to some $x_i$ (so $k(x_i, x_*) \approx \sigma_f^2$) and noise is low: $\mu_* \approx y_i$ (trust nearby observation)
- If $x_*$ is far from all training data (so $\mathbf{k}_* \approx \mathbf{0}$): $\mu_* \approx m_*$ (revert to prior)
- If observations are very noisy ($\sigma_n^2 \gg \sigma_f^2$): predictions are smoothed out, don't trust individual points

### Understanding the Predictive Variance

$$
\sigma_*^2 = k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

**Decomposition**:
- $k_{**}$: Prior variance at $x_*$ (uncertainty before seeing data, usually $\sigma_f^2$)
- $\mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$: **Reduction in variance** due to observations

**Key properties**:
1. **Always positive**: The subtracted term is a quadratic form with positive definite matrix, so $\sigma_*^2 \geq 0$
2. **Variance reduction**: Observations can only *reduce* uncertainty, never increase it: $\sigma_*^2 \leq k_{**}$
3. **Independent of $\mathbf{y}$**: Uncertainty depends only on *where* you observed, not *what* you observed! (This is special to Gaussian models)

**Interpretation**:
- Near training data: $\mathbf{k}_*$ is large → large variance reduction → low uncertainty
- Far from training data: $\mathbf{k}_*$ is small → small variance reduction → high uncertainty (revert to prior)
- Observation noise: If $\sigma_n^2$ is large, observations provide less information → less variance reduction

**This is exactly the Bayesian principle**: Uncertainty decreases where you have data, and increases where you extrapolate.

```{admonition} Connection to Module 5: Bayesian Inference
:class: note

Compare GP regression to parameter inference from Project 4:

**Parameter Inference**:
- Prior: $p(\theta)$
- Likelihood: $p(\mathcal{D} | \theta)$
- Posterior: $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta) p(\theta)$

**GP Regression**:
- Prior: $p(f) = \mathcal{GP}(m, k)$ - distribution over functions
- Likelihood: $p(\mathbf{y} | f) = \prod_i \mathcal{N}(y_i | f(x_i), \sigma_n^2)$ - noise model
- Posterior: $p(f | \mathbf{y}) = \mathcal{GP}(\mu_*, \sigma_*^2)$ - updated distribution over functions

The structure is *identical*! GP regression is just Bayesian inference in function space. The difference is that everything is Gaussian, so we can compute the posterior analytically instead of using MCMC.

This is why GPs are sometimes called "non-parametric Bayesian methods" - you're doing Bayesian inference, but over infinite-dimensional function spaces rather than finite parameter vectors.
```

### Computing Predictions: The Algorithm

Here's the complete algorithm for GP regression:

**Training** (fit hyperparameters - covered in next section):
1. Choose kernel $k$ and mean $m$
2. Optimize hyperparameters $\theta = \{\sigma_f^2, \ell, \sigma_n^2, \ldots\}$ by maximizing marginal likelihood

**Prediction** (given trained GP):
1. Compute covariance matrix: $\mathbf{K} = k(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I}$
2. Compute Cholesky decomposition: $\mathbf{K} = \mathbf{L}\mathbf{L}^T$ where $\mathbf{L}$ is lower triangular
3. Solve for weights: $\boldsymbol{\alpha} = \mathbf{L}^T \backslash (\mathbf{L} \backslash (\mathbf{y} - \mathbf{m}))$ (two triangular solves)
4. For each test point $x_*$:
   - Compute $\mathbf{k}_* = k(\mathbf{X}, x_*)$
   - Mean: $\mu_* = m_* + \mathbf{k}_*^T \boldsymbol{\alpha}$
   - Solve: $\mathbf{v} = \mathbf{L} \backslash \mathbf{k}_*$
   - Variance: $\sigma_*^2 = k_{**} - \mathbf{v}^T \mathbf{v}$

**Computational complexity**:
- Training (Cholesky): $O(n^3)$ for $n$ training points
- Prediction (per test point): $O(n)$ after precomputation

**Why Cholesky?** Instead of computing $\mathbf{K}^{-1}$ explicitly (numerically unstable!), we:
1. Factorize $\mathbf{K} = \mathbf{L}\mathbf{L}^T$ (stable)
2. Solve triangular systems (stable and fast)

This is the **standard approach** and what you'll implement in your project.

```{admonition} The More You Know: Numerical Stability
:class: tip, dropdown

Computing $\mathbf{K}^{-1}$ directly via `numpy.linalg.inv` is **bad practice** for several reasons:

1. **Numerical instability**: Inversion amplifies numerical errors, especially if $\mathbf{K}$ is poorly conditioned
2. **Unnecessary work**: We never need $\mathbf{K}^{-1}$ itself, only products $\mathbf{K}^{-1} \mathbf{v}$
3. **Lower accuracy**: Direct inversion has roughly 2x the error of triangular solves

**Best practice**:
```python
# Bad: Direct inversion
K_inv = np.linalg.inv(K)
alpha = K_inv @ y

# Good: Cholesky factorization + triangular solves
L = np.linalg.cholesky(K)
alpha = scipy.linalg.cho_solve((L, True), y)
```

The Cholesky approach is:
- More stable (better conditioning)
- Faster (reuse $\mathbf{L}$ for multiple solves)
- Standard in all GP libraries (GPy, GPFlow, GPyTorch, GPJax)

In JAX, you'll use `jax.numpy.linalg.cholesky` and `jax.scipy.linalg.solve_triangular`.
```

---

## 🔴 Part 4: Hyperparameter Optimization

So far we've assumed kernel hyperparameters $\theta = \{\sigma_f^2, \ell, \sigma_n^2, \ldots\}$ are known. In practice, we must **learn them from data**.

This is another level of inference: Given data, what hyperparameters make it most likely?

### The Marginal Likelihood (Evidence)

The **marginal likelihood** (or **evidence**) is the probability of the data given the hyperparameters, *marginalizing over* all possible functions:

$$
p(\mathbf{y} | \mathbf{X}, \theta) = \int p(\mathbf{y} | f, \mathbf{X}) p(f | \mathbf{X}, \theta) \, df
$$

**Why is this useful?**
- It's a **model selection criterion**: Higher marginal likelihood = hyperparameters that better explain the data
- Unlike maximum likelihood over $f$, it **automatically penalizes complexity** (Bayesian Occam's razor)
- For GPs, we can compute it **analytically**! (No MCMC needed)

For a GP with Gaussian noise, the marginal likelihood is:

$$
\log p(\mathbf{y} | \mathbf{X}, \theta) = -\frac{1}{2} (\mathbf{y} - \mathbf{m})^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}| - \frac{n}{2} \log(2\pi)
$$

**Decomposition** (three terms):

1. **Data fit**: $-\frac{1}{2} (\mathbf{y} - \mathbf{m})^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m})$
   - How well does the GP fit the observations?
   - Similar to negative squared error weighted by covariance
   - Reward: fits data well → high likelihood

2. **Complexity penalty**: $-\frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}|$
   - Log determinant penalizes large covariances (overly flexible models)
   - Automatic Occam's razor: simpler models are favored unless data demands complexity
   - Penalty: too flexible → low likelihood

3. **Normalization**: $-\frac{n}{2} \log(2\pi)$
   - Just a constant, doesn't affect optimization

**The beauty**: These two terms balance automatically!
- Too small $\ell$ (very wiggly functions): Good data fit, but huge complexity penalty (large $|\mathbf{K}|$)
- Too large $\ell$ (nearly constant functions): Low complexity, but poor data fit
- Optimum: Trade-off between fit and simplicity

```{admonition} Connection to Module 5: Model Evidence
:class: note

In Bayesian model comparison (Module 5), you compared models $M_1$ vs $M_2$ using **Bayes factors**:

$$
\frac{p(M_1 | \mathcal{D})}{p(M_2 | \mathcal{D})} = \frac{p(\mathcal{D} | M_1)}{p(\mathcal{D} | M_2)} \times \frac{p(M_1)}{p(M_2)}
$$

where $p(\mathcal{D} | M_i)$ is the **model evidence** (marginal likelihood).

For GPs, each choice of hyperparameters $\theta$ defines a different model. The marginal likelihood $p(\mathbf{y} | \mathbf{X}, \theta)$ is exactly the evidence for that model!

By optimizing hyperparameters via marginal likelihood, you're doing **empirical Bayes**: Using the data to choose your prior (the GP with best hyperparameters). This is a pragmatic compromise:
- Fully Bayesian: Put prior on $\theta$, integrate over it (expensive, requires MCMC)
- Empirical Bayes: Optimize $\theta$ via marginal likelihood (fast, analytic)

For most applications, empirical Bayes works well. If you have strong prior knowledge about hyperparameters, you can add priors and use MCMC (that's the full hierarchical Bayesian approach).
```

### Optimization via Gradient Ascent

To find optimal hyperparameters, we **maximize the log marginal likelihood**:

$$
\theta^* = \arg\max_\theta \log p(\mathbf{y} | \mathbf{X}, \theta)
$$

The gradient with respect to hyperparameters is:

$$
\frac{\partial \log p(\mathbf{y} | \mathbf{X}, \theta)}{\partial \theta_j} = \frac{1}{2} (\mathbf{y} - \mathbf{m})^T \mathbf{K}^{-1} \frac{\partial \mathbf{K}}{\partial \theta_j} \mathbf{K}^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \text{tr}\left( \mathbf{K}^{-1} \frac{\partial \mathbf{K}}{\partial \theta_j} \right)
$$

where $\mathbf{K} = \mathbf{K} + \sigma_n^2 \mathbf{I}$ for brevity.

**Don't panic!** You won't compute this by hand. In JAX, we use **automatic differentiation**:

```python
import jax
import jax.numpy as jnp

def log_marginal_likelihood(theta, X, y):
    """Compute log p(y | X, theta)"""
    K = kernel(X, X, theta) + theta['sigma_n']**2 * jnp.eye(len(X))
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.solve_triangular(L, y, lower=True)
    alpha = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)
    
    data_fit = -0.5 * jnp.dot(y, alpha)
    complexity = -jnp.sum(jnp.log(jnp.diag(L)))
    constant = -0.5 * len(y) * jnp.log(2 * jnp.pi)
    
    return data_fit + complexity + constant

# Automatic differentiation!
grad_log_ml = jax.grad(log_marginal_likelihood)
```

**Optimization algorithm**:
1. Initialize hyperparameters (e.g., from heuristics or random)
2. Repeat until convergence:
   - Compute gradient $\nabla_\theta \log p(\mathbf{y} | \mathbf{X}, \theta)$
   - Update: $\theta \gets \theta + \eta \nabla_\theta \log p(\mathbf{y} | \mathbf{X}, \theta)$
3. Return optimized $\theta^*$

**Practical considerations**:
- **Multiple initializations**: Log marginal likelihood is **non-convex** (multiple local optima). Try several random initializations, pick best.
- **Constrained optimization**: Hyperparameters must be positive ($\sigma_f^2 > 0$, $\ell > 0$). Use log-transformations: optimize $\log \theta$ instead of $\theta$.
- **Optimizer choice**: L-BFGS-B (quasi-Newton) works well. In JAX, use `optax` (Adam, SGD) or `scipy.optimize`.

```{admonition} The More You Know: Bayesian Optimization of Hyperparameters
:class: tip, dropdown

For a fully Bayesian treatment, you'd put a **prior** on hyperparameters $p(\theta)$ and compute:

$$
p(f_* | x_*, \mathbf{X}, \mathbf{y}) = \int p(f_* | x_*, \mathbf{X}, \mathbf{y}, \theta) p(\theta | \mathbf{X}, \mathbf{y}) \, d\theta
$$

This **marginalizes over hyperparameter uncertainty**, giving more robust predictions (especially with limited data).

**Approaches**:
1. **MCMC**: Sample $\theta$ using Hamiltonian Monte Carlo (HMC from Project 4!)
2. **Variational inference**: Approximate $p(\theta | \mathbf{X}, \mathbf{y})$ with a simpler distribution
3. **Laplace approximation**: Gaussian approximation around MAP estimate

This adds computational cost (need to sample or optimize over $\theta$ distribution), but can be worth it for:
- Small datasets (hyperparameter uncertainty is large)
- High-stakes decisions (want robust uncertainty estimates)
- Model selection (comparing different kernels)

For your project, **empirical Bayes** (maximizing marginal likelihood) is sufficient and standard. If you have extra time, implementing MCMC over hyperparameters would be an excellent extension!
```

### Hyperparameter Initialization Heuristics

Random initialization often fails for GPs. Here are **physics-informed heuristics**:

**Output variance** $\sigma_f^2$:
```python
sigma_f = jnp.std(y)  # Sample standard deviation of outputs
```
**Reasoning**: The output variance should match the scale of variation in your data.

**Lengthscale** $\ell$:
```python
# Median distance between points
distances = jnp.pdist(X)  # Pairwise distances
ell = jnp.median(distances)
```
**Reasoning**: If lengthscale is much smaller than typical spacing, nearby points look uncorrelated (overfitting). If much larger, all points look identical (underfitting). Median distance is a reasonable middle ground.

**Noise variance** $\sigma_n^2$:
```python
# Estimate from nearby point differences (if you have them)
# Or use a small fraction of output variance
sigma_n = 0.1 * sigma_f
```
**Reasoning**: Start with modest noise (10% of signal). If data is actually noisy, optimization will increase it.

**For ARD kernels** (per-dimension lengthscales):
```python
for d in range(D):
    ell[d] = jnp.std(X[:, d])  # Standard deviation in dimension d
```
**Reasoning**: Lengthscale should be comparable to the range of variation in each dimension. Using standard deviation ensures dimensionless units.

These heuristics give a good starting point, then optimization refines them.

---

## 🔴 Part 5: Practical Implementation in JAX

Now let's connect theory to code. You'll implement GP regression from scratch in JAX, learning both the math and the computational tricks.

### Why JAX for GPs?

**JAX advantages**:
1. **Automatic differentiation**: Compute gradients of marginal likelihood for free
2. **JIT compilation**: Makes GP code fast (critical for optimization loops)
3. **Vectorization**: `vmap` for computing kernels over batches
4. **GPU acceleration**: Move to GPU with one line (helpful for large $n$)
5. **Functional paradigm**: Pure functions → easy to reason about, compose

**JAX challenges**:
1. **Immutability**: No in-place operations (but easier to parallelize)
2. **Pure functions**: No side effects (but easier to optimize)
3. **Fixed shapes**: Can't change array sizes inside JIT (but there are workarounds)

Overall, JAX is **ideal for GPs**. The functional style matches the mathematical structure, and autodiff eliminates tedious gradient derivations.

### Core JAX Tools for GPs

Let's introduce the key JAX/ecosystem packages you'll use:

#### 1. **JAX Core** (`jax.numpy`, `jax.scipy`)

**Basic operations**:
```python
import jax.numpy as jnp
import jax.scipy as jsp

# Matrix operations (like numpy)
K = jnp.array([[1.0, 0.5], [0.5, 1.0]])
L = jnp.linalg.cholesky(K)  # Cholesky decomposition

# Solve linear system
alpha = jsp.linalg.solve_triangular(L, y, lower=True)
```

**Key differences from NumPy**:
- **Immutable arrays**: `x = x.at[0].set(5)` instead of `x[0] = 5`
- **Explicit random keys**: `key = jax.random.PRNGKey(0); jax.random.normal(key, shape)`
- **Device arrays**: Data lives on CPU/GPU, operations respect that

#### 2. **Automatic Differentiation** (`jax.grad`, `jax.value_and_grad`)

**Compute gradients**:
```python
import jax

# Define function
def f(x):
    return jnp.sum(x**2)

# Get gradient function
grad_f = jax.grad(f)

# Evaluate
x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))  # [2.0, 4.0, 6.0]
```

**For optimization** (get both value and gradient):
```python
def loss(params):
    # ... compute loss ...
    return loss_value

loss_and_grad = jax.value_and_grad(loss)
loss_val, loss_grad = loss_and_grad(params)
```

**For GPs**: This is how you'll optimize hyperparameters! Define `log_marginal_likelihood(theta, X, y)`, then:
```python
grad_fn = jax.grad(log_marginal_likelihood, argnums=0)  # Gradient w.r.t. theta
```

#### 3. **JIT Compilation** (`jax.jit`)

**Speed up functions**:
```python
@jax.jit
def predict(X_train, y_train, X_test, theta):
    """Fast GP prediction (compiled once, reused)"""
    K = kernel(X_train, X_train, theta)
    K_s = kernel(X_train, X_test, theta)
    # ... rest of prediction ...
    return mu, sigma

# First call: compiles (slow)
# Subsequent calls: uses compiled version (fast!)
```

**When to JIT**:
- Inner loops (called many times)
- Complex computations (matrix operations, reductions)
- Not for top-level scripts (compilation overhead)

**For GPs**: JIT your `predict` function, your `log_marginal_likelihood`, and your kernel evaluations.

#### 4. **Vectorization** (`jax.vmap`)

**Batch operations**:
```python
# Compute kernel for all pairs of points
def kernel_single(x, x_prime, theta):
    """Kernel for single pair"""
    return theta['sigma_f']**2 * jnp.exp(-jnp.sum((x - x_prime)**2) / (2 * theta['ell']**2))

# Vectorize over rows and columns
kernel_matrix = jax.vmap(
    jax.vmap(kernel_single, in_axes=(None, 0, None)),  # Over x_prime
    in_axes=(0, None, None)  # Over x
)

# Now: K = kernel_matrix(X, X, theta) computes full matrix efficiently
```

**For GPs**: Use `vmap` to compute covariance matrices, evaluate predictions at multiple test points, etc.

#### 5. **Optax** (Optimization library)

**Gradient-based optimization**:
```python
import optax

# Define optimizer
optimizer = optax.adam(learning_rate=0.01)

# Initialize
params = initialize_params()
opt_state = optimizer.init(params)

# Optimization loop
for step in range(1000):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss_val:.4f}")
```

**For GPs**: Optimize hyperparameters by minimizing negative log marginal likelihood.

**Common optimizers**:
- `optax.adam`: Adaptive learning rates, good default
- `optax.sgd`: Stochastic gradient descent with momentum
- `optax.lbfgs`: Quasi-Newton (requires `jaxopt` library)

#### 6. **Equinox** (Neural network library)

You'll use Equinox for the neural network part of the project (Week 3-4), but it's worth mentioning now:

```python
import equinox as eqx

class GaussianProcess(eqx.Module):
    """GP as an Equinox module"""
    kernel: eqx.Module
    mean_function: eqx.Module
    noise: float
    
    def __call__(self, X_train, y_train, X_test):
        # ... GP prediction logic ...
        return mu, sigma
```

**Why Equinox?**
- **Pytree-based**: Models are JAX pytrees (easy to JIT, vmap, differentiate)
- **Functional**: No hidden state, just pure functions
- **Composable**: Modules are just callables, easy to combine
- **Type-safe**: Uses Python type hints (better IDE support)

You can implement GPs purely in JAX, but Equinox makes it cleaner if you want modularity (e.g., swapping kernels).

### A Minimal GP Implementation

Here's a complete, working GP implementation in JAX:

```python
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, vmap

class SimpleGP:
    def __init__(self, kernel_fn, mean_fn=None):
        self.kernel = kernel_fn
        self.mean = mean_fn if mean_fn is not None else lambda x: jnp.zeros(x.shape[0])
    
    @jit
    def fit(self, X, y, noise):
        """Precompute training data covariance and weights"""
        n = X.shape[0]
        K = self.kernel(X, X) + noise**2 * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        
        m = self.mean(X)
        alpha = jsp.linalg.solve_triangular(L, y - m, lower=True)
        alpha = jsp.linalg.solve_triangular(L.T, alpha, lower=False)
        
        return {'L': L, 'alpha': alpha, 'X': X}
    
    @jit
    def predict(self, state, X_test):
        """Predict at test points"""
        K_s = self.kernel(state['X'], X_test)  # (n_train, n_test)
        
        # Mean
        m_test = self.mean(X_test)
        mu = m_test + K_s.T @ state['alpha']
        
        # Variance
        v = jsp.linalg.solve_triangular(state['L'], K_s, lower=True)
        K_ss = self.kernel(X_test, X_test)
        cov = K_ss - v.T @ v
        sigma = jnp.sqrt(jnp.diag(cov))
        
        return mu, sigma

# Example kernel
@jit
def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """Squared exponential kernel"""
    X1 = X1 / lengthscale
    X2 = X2 / lengthscale
    
    # Efficient computation: ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x·x'
    X1_sq = jnp.sum(X1**2, axis=1, keepdims=True)
    X2_sq = jnp.sum(X2**2, axis=1, keepdims=True)
    dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
    
    return variance * jnp.exp(-0.5 * dist_sq)

# Usage
gp = SimpleGP(kernel_fn=lambda X1, X2: rbf_kernel(X1, X2, lengthscale=1.0))
state = gp.fit(X_train, y_train, noise=0.1)
mu, sigma = gp.predict(state, X_test)
```

**What this code does**:
1. **`fit`**: Computes Cholesky decomposition and weights $\boldsymbol{\alpha}$ (training)
2. **`predict`**: Uses conditioning formulas to predict mean and variance (inference)
3. **JIT everywhere**: Functions compile once, then run fast

**Your task in the project**: Extend this to:
- Multiple kernels (Matérn, periodic, composite)
- Hyperparameter optimization (maximize marginal likelihood)
- Multi-dimensional inputs (ARD kernels)
- Uncertainty visualization and calibration

---

## 🟡 Part 6: Applying GPs to Star Cluster Evolution

Now let's connect this to your specific project: **emulating N-body simulations of star clusters**.

### The Scientific Problem

From Project 2 and Project 5, you have a **simulator**:

**Input** (initial conditions): 
- $N_\star$: Number of stars
- $M_{\text{tot}}$: Total cluster mass [$M_\odot$]
- $Q_{\text{virial}}$: Virial ratio (ratio of kinetic to potential energy)
- $c$: King concentration parameter
- (Optional) IMF parameters, mass segregation, etc.

**Output** (evolutionary trajectory):
- $r_{\text{core}}(t)$: Core radius [pc]
- $\sigma_v(t)$: Velocity dispersion [km/s]
- $f_{\text{bound}}(t)$: Bound mass fraction
- (Optional) Lagrange radii, energy, anisotropy, etc.

**Cost**: Each simulation takes **minutes to hours**

**Goal**: Build a GP emulator that:
1. Takes initial conditions as input
2. Predicts outputs at arbitrary times
3. Provides uncertainty estimates
4. Enables fast parameter space exploration

### Phase 1: Scalar Predictions at Fixed Time

Start simple: Predict **one quantity at one time**.

**Example**: Given ICs $\mathbf{x} = (N_\star, M_{\text{tot}}, Q, c)$, predict $r_{\text{core}}(t=100 \text{ Myr})$.

**Training data**: Run $n$ simulations with different ICs, record $r_{\text{core}}$ at 100 Myr:
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n \quad \text{where} \quad y_i = r_{\text{core}}(t=100 | \mathbf{x}_i)$$

**GP setup**:
- Input dimension: $D = 4$ (or more with IMF parameters)
- Output dimension: $1$ (scalar)
- Kernel: ARD-SE (different lengthscale per dimension)
- Mean function: Zero (or constant estimated from data)

**Implementation steps**:

1. **Preprocess inputs**: Standardize to zero mean, unit variance
   ```python
   X_mean = jnp.mean(X_train, axis=0)
   X_std = jnp.std(X_train, axis=0)
   X_train_norm = (X_train - X_mean) / X_std
   ```
   
2. **Initialize hyperparameters**:
   ```python
   theta = {
       'sigma_f': jnp.std(y_train),
       'lengthscales': jnp.ones(D),  # Start with unity (after standardization)
       'sigma_n': 0.1 * jnp.std(y_train)
   }
   ```
   
3. **Optimize via marginal likelihood**:
   ```python
   def neg_log_marginal_likelihood(theta, X, y):
       return -log_marginal_likelihood(theta, X, y)
   
   # Optimize (e.g., using optax)
   theta_opt = optimize(neg_log_marginal_likelihood, theta, X_train_norm, y_train)
   ```
   
4. **Make predictions**:
   ```python
   mu, sigma = gp.predict(theta_opt, X_train_norm, y_train, X_test_norm)
   ```

5. **Validate**:
   - Plot predictions vs true values on test set
   - Check uncertainty calibration (see next section)
   - Compute RMSE, log-likelihood

**Expected results**:
- RMSE on test set: ~5-10% of typical $r_{\text{core}}$ values (depending on noise)
- Uncertainty bands cover ~68% of test data (if GP is well-calibrated)
- Learned lengthscales reveal which ICs matter most

```{admonition} Scientific Interpretation: What Do Lengthscales Tell You?
:class: note

After fitting, examine the learned lengthscales $\ell_d$ for each input dimension:

**Small lengthscale** ($\ell_d \ll 1$ after standardization):
- Output is sensitive to dimension $d$
- Small changes in $d$ cause large changes in output
- This dimension is **important** for determining $r_{\text{core}}$

**Large lengthscale** ($\ell_d \gg 1$):
- Output barely varies with dimension $d$
- This dimension is **less relevant**
- Could potentially remove it (dimensionality reduction)

**Example findings** (hypothetical):
- $\ell_{Q} \approx 0.3$: Virial ratio strongly affects core radius (expected! Initial energy balance matters)
- $\ell_c \approx 0.5$: Concentration matters (denser clusters have smaller cores)
- $\ell_M \approx 3.0$: Total mass has weak effect at fixed $N$ (perhaps core radius is more about relaxation time)
- $\ell_N \approx 1.5$: Moderate effect (two-body relaxation scales with $N$)

This is **data-driven astrophysics**: The GP learns from simulations which physics dominates!
```

### Phase 2: Time Series (Where GPs Struggle)

Now try: Predict **full temporal evolution** $r_{\text{core}}(t)$ for $t \in [0, 200]$ Myr.

**Approach 1: Time as Input**

Augment input: $\mathbf{x} = (N_\star, M_{\text{tot}}, Q, c, t)$ (5D input)

**Training data**: Each simulation gives multiple $(x, t, r(t))$ triplets:
$$\mathcal{D} = \{(\mathbf{x}_i, t_j, r_{ij})\}$$

**Problem**: GP doesn't naturally understand temporal structure. Two issues:

1. **Independent predictions**: $r(t)$ and $r(t + \Delta t)$ are treated as independent observations, even though they're from the same trajectory. The kernel needs to learn temporal smoothness from data.

2. **Extrapolation fails**: GP trained on $t \in [0, 200]$ can't reliably extrapolate to $t = 300$ Myr. It reverts to prior mean with large uncertainty.

**Workaround**: Use specialized kernels:
- Product kernel: $k(\mathbf{x}, \mathbf{x}') \cdot k_{\text{time}}(t, t')$ where $k_{\text{time}}$ is smooth
- Separate lengthscales: $\ell_t$ (temporal) vs $\ell_x$ (spatial)

But this is **ad hoc** and doesn't capture true dynamics.

**Approach 2: Multi-Output GP**

Treat each time slice as a separate output: $\mathbf{y} = [r(t_1), r(t_2), \ldots, r(t_T)]^T$

**Training data**: Each simulation gives a vector of outputs:
$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$$

**Problem**: Covariance matrix explodes!
- Single-output GP: $\mathbf{K} \in \mathbb{R}^{n \times n}$
- Multi-output GP: $\mathbf{K} \in \mathbb{R}^{nT \times nT}$ (block covariance matrix)
- For $n=500$ sims, $T=100$ timesteps: $\mathbf{K}$ has 2.5 billion entries!

**Cost**: 
- Memory: $O(n^2 T^2)$
- Computation: $O(n^3 T^3)$ for Cholesky

Even on GPU, this is intractable for $n > 100$, $T > 50$.

**Workarounds**:
- **Sparse GPs**: Inducing points reduce complexity to $O(nm^2)$ where $m \ll n$
- **Structured kernels**: Kronecker factorization if covariance is separable
- **GP approximations**: Variational inference, spectral methods

But these add complexity and approximations.

### The GP Limitation: Motivation for Neural Networks

This is the **pedagogical turning point**! Students should conclude:

**GPs excel at**:
- Low-dimensional problems ($D \lesssim 10$)
- Smooth functions (infinitely differentiable or controlled smoothness)
- Scalar or low-dimensional outputs
- Small-to-medium data ($n \lesssim 1000$)
- Uncertainty quantification (built-in)
- Interpretability (kernel structure + hyperparameters)

**GPs struggle with**:
- High-dimensional outputs (time series, images)
- Complex temporal/sequential structure
- Large datasets ($n \gtrsim 10000$)
- Non-smooth functions (discontinuities, phase transitions)
- Extrapolation far from training data

**This motivates neural networks**:
- Can handle high-dimensional outputs (via latent representations)
- Natural for sequences (RNNs, Transformers)
- Scale to large data (stochastic gradient descent)
- Can learn complex nonlinear patterns
- But: lose uncertainty quantification (need ensembles) and interpretability

**The project arc**:
1. **Week 1**: Build GP, apply to scalars → see success
2. **Week 2**: Try GP on time series → see limitations
3. **Week 3-4**: Build Neural ODE / RNN / etc. → overcome limitations, but face new trade-offs

This is **learning by doing**: experiencing the method boundaries firsthand!

---

## 🟡 Part 7: Uncertainty Quantification and Validation

One of GPs' superpowers is **principled uncertainty quantification**. But how do we validate these uncertainty estimates?

### Types of Uncertainty

**1. Epistemic Uncertainty** (reducible):
- Uncertainty due to lack of knowledge
- "We haven't observed this region of input space"
- Reduces as you collect more training data
- Captured by GP predictive variance: $\sigma_*^2 = k_{**} - \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{k}_*$

**2. Aleatoric Uncertainty** (irreducible):
- Uncertainty inherent in the system (noise)
- "Simulations with identical ICs give slightly different results" (stochastic)
- Does not reduce with more data
- Captured by noise parameter: $\sigma_n^2$

**GP predictive distribution** combines both:
$$p(y_* | x_*, \mathcal{D}) = \mathcal{N}(y_* | \mu_*, \sigma_*^2 + \sigma_n^2)$$

where:
- $\sigma_*^2$: Epistemic (model uncertainty)
- $\sigma_n^2$: Aleatoric (observation noise)
- $\sigma_*^2 + \sigma_n^2$: Total predictive uncertainty

**For noiseless functions** (deterministic simulators): $\sigma_n^2 \approx 0$, so only epistemic uncertainty matters. But in practice, we add small $\sigma_n$ for numerical stability (jitter).

```{admonition} Connection to Module 5: Posterior Uncertainty
:class: note

In Bayesian parameter inference (Project 4), posterior uncertainty $\text{Var}[\theta | \mathcal{D}]$ reflected:
- **Epistemic**: Don't have enough data to pin down $\theta$ precisely
- **Not aleatoric**: The parameters themselves aren't noisy (they're fixed, just unknown)

For GPs:
- **Posterior over functions**: $\mathcal{GP}(\mu_*, \sigma_*^2)$ is epistemic (function uncertainty)
- **Observation noise**: $\sigma_n^2$ is aleatoric (data uncertainty)

In both cases, more data reduces epistemic uncertainty but doesn't affect aleatoric. This distinction is fundamental to scientific reasoning: What can we learn (epistemic) vs what is inherently random (aleatoric)?
```

### Uncertainty Calibration

A well-calibrated GP should have:
- **68% of test data** within 1-sigma ($\mu_* \pm \sigma_*$)
- **95% of test data** within 2-sigma ($\mu_* \pm 2\sigma_*$)

**Calibration plot**:
```python
def calibration_plot(y_test, mu, sigma):
    """Check if uncertainties are well-calibrated"""
    # Compute z-scores
    z = (y_test - mu) / sigma
    
    # Expected: z ~ N(0, 1) if calibrated
    # Plot histogram of z vs standard normal
    import matplotlib.pyplot as plt
    plt.hist(z, bins=30, density=True, alpha=0.6, label='Empirical')
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1)')
    plt.legend()
    plt.xlabel('z-score')
    plt.ylabel('Density')
    plt.title('Calibration Check')
```

**Interpretation**:
- Histogram matches N(0,1): **Well-calibrated** ✓
- Histogram narrower: **Overconfident** (uncertainties too small) ✗
- Histogram wider: **Underconfident** (uncertainties too large) ✗

**Common failure modes**:

1. **Overconfident**: $\sigma_n^2$ too small, or kernel is too flexible
   - Fix: Increase noise, regularize hyperparameters
   
2. **Underconfident**: $\sigma_n^2$ too large, or kernel is too rigid
   - Fix: Decrease noise, allow more flexible kernel

3. **Biased**: Mean predictions systematically high/low
   - Fix: Use non-zero mean function, or transform outputs

### Coverage Diagnostics

Another calibration check: **prediction intervals**.

For each confidence level $\alpha$:
$$P(y_* \in [\mu_* - z_\alpha \sigma_*, \mu_* + z_\alpha \sigma_*]) = \alpha$$

**Expected coverage**:
| Confidence | $z_\alpha$ | Expected coverage |
|------------|-----------|-------------------|
| 68% | 1.0 | 68% of test data |
| 95% | 1.96 | 95% of test data |
| 99% | 2.58 | 99% of test data |

**Empirical coverage**: Count how many test points fall within intervals
```python
def coverage(y_test, mu, sigma, z_alpha=1.0):
    """Fraction of test data within z_alpha standard deviations"""
    lower = mu - z_alpha * sigma
    upper = mu + z_alpha * sigma
    in_interval = (y_test >= lower) & (y_test <= upper)
    return jnp.mean(in_interval)

# Check multiple confidence levels
for alpha, z in [(0.68, 1.0), (0.95, 1.96), (0.99, 2.58)]:
    emp_coverage = coverage(y_test, mu, sigma, z)
    print(f"{100*alpha:.0f}% interval: {100*emp_coverage:.1f}% actual coverage")
```

**Ideal result**: Empirical coverage matches expected within ~5% (statistical noise)

### Sharpness vs Calibration Trade-off

Two competing goals for uncertainty estimates:

1. **Calibration**: Coverage matches nominal levels (correct on average)
2. **Sharpness**: Intervals are as narrow as possible (informative)

**Trivial solutions**:
- $\sigma_* = \infty$ for all points: Perfect calibration (always right), but useless (no information)
- $\sigma_* = 0$ for all points: Perfect sharpness, but terrible calibration (overly confident)

**Good GP**: Balances both! Narrow uncertainties *and* correct coverage.

**Metric**: **Continuous Ranked Probability Score (CRPS)**
$$\text{CRPS} = \int_{-\infty}^\infty (F(y) - \mathbb{1}_{y \geq y_*})^2 \, dy$$
where $F$ is predicted CDF, $y_*$ is true value.

Lower CRPS = better (sharp + calibrated). For Gaussians:
$$\text{CRPS}(y_*, \mu, \sigma) = \sigma \left[ \frac{z}{\sqrt{\pi}} - 2\phi(z) - z(2\Phi(z) - 1) \right]$$
where $z = (y_* - \mu)/\sigma$, $\phi$ is standard normal PDF, $\Phi$ is CDF.

**In your project**: Report CRPS on test set as a single metric combining accuracy and uncertainty quality.

---

## 🟡 Part 8: When GPs Fail (and What To Do About It)

Understanding limitations is as important as understanding capabilities. Here's what breaks GPs:

### 1. High-Dimensional Inputs ($D \gg 10$)

**Problem**: "Curse of dimensionality"
- Volume of space grows exponentially with $D$
- Training points become sparse (typical distance grows as $\sqrt{D}$)
- Need exponentially more data to cover space

**Manifestation**:
- Uncertainty doesn't decrease much (all points look "far" from data)
- Extrapolation everywhere (never really "interpolating")
- Lengthscale optimization becomes ill-conditioned

**Mitigations**:
1. **Dimensionality reduction**: PCA on inputs, keep top principal components
2. **ARD kernel**: Learn which dimensions matter, effectively reduce dimensionality
3. **Feature engineering**: Combine correlated inputs into summary statistics
4. **Additive models**: $f(\mathbf{x}) = \sum_d f_d(x_d)$ (assumes additive structure)

**For your project**: With $D \leq 8$ (N, M, Q, c, IMF params), you're safe. But good to be aware!

### 2. High-Dimensional Outputs

**Problem**: Multi-output GPs scale poorly
- Covariance matrix: $O((nT)^2)$ memory, $O((nT)^3)$ time for $T$ outputs
- Difficult to model correlations between outputs

**Manifestation**:
- Can't fit time series (100 timesteps = 100 outputs)
- Memory errors or extremely slow training

**Mitigations**:
1. **Output separately**: Train one GP per output (ignores correlations)
2. **Sparse GPs**: Induce sparsity via low-rank approximations
3. **Latent variable models**: Project outputs to low-dimensional space
4. **Neural networks**: Natural for high-dimensional outputs (this is your project transition!)

**For your project**: This is exactly the limitation you'll hit! Time series have $T \sim 100$ outputs → GP struggles → motivates Neural ODE.

### 3. Large Datasets ($n \gg 1000$)

**Problem**: Cubic scaling with data size
- Cholesky: $O(n^3)$ operations
- Memory: $O(n^2)$ for covariance matrix

**Manifestation**:
- Hours to train on $n = 10000$ points
- Out-of-memory errors for $n > 50000$

**Mitigations**:
1. **Sparse GPs**: Inducing points (variational sparse GPs)
2. **Local GPs**: Partition space, fit separate GPs to regions
3. **Stochastic gradient descent**: Subsample data, mini-batch training
4. **Neural networks**: Scale linearly $O(n)$ with SGD

**For your project**: With $n = 500$ simulations, you're fine. But if you had millions of observations (e.g., Gaia catalog), GPs wouldn't work.

### 4. Non-Stationary Functions

**Problem**: Standard kernels (SE, Matérn) are stationary (depend only on $|x - x'|$)
- Assume same smoothness everywhere
- Can't capture functions that vary differently in different regions

**Example**: Cluster core radius
- Smooth evolution during main sequence ($t < 100$ Myr)
- Rapid collapse during post-MS ($t \sim 100-120$ Myr) when massive stars leave
- Slow expansion after core bounce ($t > 120$ Myr)

Different smoothness at different times! But SE kernel assumes same lengthscale everywhere.

**Mitigations**:
1. **Input warping**: Transform inputs to make function more stationary
2. **Non-stationary kernels**: Gibbs kernel (input-dependent lengthscale), deep GPs
3. **Piecewise GPs**: Detect regime changes, fit separate GPs
4. **Neural networks**: Can learn spatially varying smoothness naturally

### 5. Discontinuities and Phase Transitions

**Problem**: GP kernels produce smooth functions (at least continuous)
- Can't represent sharp jumps
- Will smooth over discontinuities (Gibbs phenomenon-like oscillations)

**Astrophysical examples**:
- Ionization fronts (sharp boundary between ionized/neutral gas)
- Tidal disruption (cluster suddenly loses mass when passing close to galaxy)
- Two-body encounters (sudden velocity kicks)

**Manifestation**:
- Poor fit near discontinuities
- Over-smoothing, wiggles
- Underestimated uncertainty at jumps

**Mitigations**:
1. **Discontinuous kernels**: Rarely used, hard to implement
2. **Transform outputs**: Smooth transformed variable (e.g., log if exponential jumps)
3. **Mixture models**: Separate GPs for different regimes, with jump probabilities
4. **Physics-informed NNs**: Encode discontinuity conditions as constraints

**For your project**: Star cluster evolution is mostly smooth (thank gravity!), so this shouldn't be an issue. But be aware if you see strange behavior.

### 6. Extrapolation Beyond Training Range

**Problem**: GPs revert to prior mean when extrapolating
- Outside training data: $\mathbf{k}_* \approx \mathbf{0}$ → $\mu_* \approx m$, $\sigma_* \approx \sigma_f$
- No physical constraints guide extrapolation

**Manifestation**:
- Predictions flatten to constant (prior mean) outside training range
- Uncertainty maxes out at prior variance
- No ability to use physics to extrapolate

**Example**: 
- Train on clusters with $N \in [100, 1000]$
- Predict for $N = 5000$: GP says "I don't know" (large uncertainty, mean prediction = prior)
- But physics *does* constrain: relaxation time $t_{\text{relax}} \propto N$ (could encode this!)

**Mitigations**:
1. **Physics-informed mean functions**: Encode known scaling laws
2. **Physics-informed kernels**: Build in symmetries, conservation laws
3. **Hybrid models**: GP learns residuals from physics-based model
4. **Neural networks with physics**: PINNs, Neural ODEs (Project Week 3-4!)

**For your project**: Test this explicitly! Train on subset of parameter space, extrapolate, compare to true simulations. This demonstrates GP limitations and motivates NN extensions.

```{admonition} The Profound Limitation: GPs Don't "Know Physics"
:class: important

GPs are **purely data-driven**. They learn patterns from examples, but don't understand:
- Conservation laws (energy, momentum, angular momentum)
- Scaling relations (dimensional analysis)
- Differential equations governing dynamics
- Symmetries and invariances

This is both:
- **Strength**: Work for any function, no physics modeling needed, model-agnostic
- **Weakness**: Can't extrapolate using physics, need data everywhere, no built-in constraints

**The frontier of ML for science**: Combine data-driven flexibility (GPs, NNs) with physics knowledge (conservation laws, ODEs, symmetries). This is **physics-informed machine learning**, and it's what your project explores in Weeks 3-4!

Examples:
- **Neural ODEs**: Learn dynamics $df/dt = F_\theta(f, t)$, integrate with physics-based solver
- **PINNs**: Loss function includes physics residuals (how well do predictions satisfy PDEs?)
- **Hamiltonian NNs**: Constrain architecture to preserve energy
- **Equivariant NNs**: Build in symmetries (rotation, translation, permutation)

This is the future of computational astrophysics: ML that respects physics!
```

---

## ⚪ Part 9: Advanced Topics (Optional Extensions)

These are beyond the scope of your core project, but excellent for ambitious students:

### Multi-Fidelity GPs

**Idea**: You have simulators at different accuracy levels
- **High-fidelity**: Expensive, accurate (e.g., $N=10000$ star N-body)
- **Low-fidelity**: Cheap, approximate (e.g., $N=1000$ star N-body, or analytical model)

**Goal**: Use lots of cheap simulations + few expensive simulations to predict expensive outputs everywhere.

**Method**: Hierarchical GP
$$f_{\text{high}}(\mathbf{x}) = \rho \cdot f_{\text{low}}(\mathbf{x}) + \delta(\mathbf{x})$$
where $\rho$ is correlation, $\delta$ is discrepancy (another GP).

**Benefit**: Orders of magnitude fewer expensive simulations needed.

**Astrophysical applications**:
- Cosmological simulations (dark matter only vs full baryonic physics)
- Stellar evolution (simple vs detailed nuclear networks)
- N-body (few particles vs many)

### Deep Gaussian Processes

**Idea**: Compose GPs in layers (like deep neural networks)
$$f = f_L \circ f_{L-1} \circ \cdots \circ f_1$$
where each $f_\ell$ is a GP.

**Why**: Single-layer GPs have limited expressivity. Composing them increases flexibility (can learn hierarchical features).

**Challenge**: Inference is hard (posterior is no longer Gaussian). Need variational approximations.

**When useful**: Complex, hierarchical functions that single GPs can't capture.

### GP Classification

**What you've learned**: GP *regression* (continuous outputs)

**Extension**: GP *classification* (discrete outputs)
- Predict category instead of number
- E.g., "Will this cluster survive 1 Gyr?" (yes/no)

**Method**: Use GP to model latent function, squash through sigmoid
$$p(y=1 | \mathbf{x}) = \sigma(f(\mathbf{x})) \quad \text{where} \quad f \sim \mathcal{GP}(m, k)$$

**Challenge**: Likelihood is no longer Gaussian → no closed-form posterior. Need approximations (Laplace, EP, MCMC).

**Astrophysical applications**:
- Star/galaxy classification
- Supernova type prediction
- Cluster survival probability

### Gaussian Processes for Inverse Problems

**Setup**: You observe some quantity $\mathbf{d}$ that depends on hidden function $f$:
$$\mathbf{d} = \mathcal{F}(f) + \boldsymbol{\epsilon}$$
where $\mathcal{F}$ is a forward model (e.g., integral operator, differential equation).

**Goal**: Infer $f$ from $\mathbf{d}$ (inverse problem)

**GP approach**: Put GP prior on $f$, compute posterior given data
$$p(f | \mathbf{d}) \propto p(\mathbf{d} | f) p(f)$$

**Example**: Inferring stellar mass distribution from observed velocity dispersion profile.

**Why GPs**: Smoothness prior helps regularize ill-posed inverse problems.

---

## 📊 Part 10: Software Packages and Ecosystem

You'll implement GPs from scratch to understand the internals, but here are professional libraries for reference:

### JAX Ecosystem

**1. GPJax** (https://github.com/JaxGaussianProcesses/GPJax)
- Modern GP library in JAX
- Supports exact GPs, sparse GPs, multi-output, classification
- Clean functional API, compatible with Optax, Flax, Equinox
- Great for research, actively developed

**2. tinygp** (https://github.com/dfm/tinygp)
- Minimal GP library by Dan Foreman-Mackey (astronomer!)
- Emphasizes simplicity and composability
- Good for quick prototyping
- Used in exoplanet research (stellar variability modeling)

### Other Ecosystems (for reference)

**PyTorch**:
- **GPyTorch**: Scalable GPs with GPU support, active development
- **BoTorch**: Bayesian optimization (uses GPyTorch under the hood)

**TensorFlow**:
- **GPFlow**: Feature-rich, variational sparse GPs
- **TensorFlow Probability**: Includes GP layers

**Standalone**:
- **scikit-learn**: Basic GP regression (`sklearn.gaussian_process`)
- **GPy**: Mature Python GP library (Sheffield ML group)

**For your project**: Implement from scratch in pure JAX initially. Then optionally compare to GPJax or tinygp.

### Learning Resources

**Books**:
1. *Gaussian Processes for Machine Learning* - Rasmussen & Williams (2006)
   - **The** canonical reference, freely available online
   - Chapters 1-5 cover everything you need
   
2. *Machine Learning: A Probabilistic Perspective* - Kevin Murphy (2012)
   - Chapter 15 on GPs, connects to broader ML

**Papers**:
1. "Gaussian Processes for Big Data" - Hensman et al. (2013)
   - Sparse GP approximations
   
2. "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" - Gardner et al. (2018)
   - Modern computational techniques

**Online**:
1. distill.pub: "A Visual Exploration of Gaussian Processes"
   - Interactive visualizations
   
2. Martin Krasser's blog: https://krasserm.github.io/2018/03/19/gaussian-processes/
   - Clear explanations with code

---

## 🎯 Part 11: Summary and Looking Ahead

### What You've Learned

**Conceptual**:
- GPs as probability distributions over functions
- Connection to Bayesian inference (Module 5)
- Kernels encode prior beliefs about function structure
- Uncertainty quantification is built-in and interpretable

**Mathematical**:
- Multivariate Gaussian conditioning (the core formula)
- Kernel properties and composition rules
- Marginal likelihood for hyperparameter optimization
- Predictive distributions and uncertainty decomposition

**Computational**:
- Efficient implementation via Cholesky decomposition
- Automatic differentiation for gradient-based optimization
- JAX tools: jit, grad, vmap for fast GP code
- Numerical stability considerations

**Scientific**:
- GPs excel at surrogate modeling for expensive simulations
- Uncertainty calibration and validation methods
- Understanding when GPs fail (limitations)
- Connecting ML methods to astrophysical problems

### What's Next: Neural Networks

**GPs are excellent but limited**:
✅ Small data, smooth functions, uncertainty quantification
❌ High-dimensional outputs, large data, complex structure

**Neural networks complement GPs**:
✅ Scale to large data, handle time series, learn complex patterns
❌ Harder to train, less interpretable, uncertainty requires extra work

**Your project arc**:
1. **Week 1** (Done after this lecture!): Implement GP, apply to scalar predictions
2. **Week 2**: Push GPs to limits (time series), document failures
3. **Week 3-4**: Build Neural ODE / RNN to overcome limitations

**The synthesis**: You'll learn when to use each method:
- **GPs**: Exploration phase, limited data, need uncertainty, interpret physics
- **NNs**: Exploitation phase, lots of data, need speed/flexibility, production systems
- **Hybrid**: GP + NN (e.g., NN learns mean, GP models residuals)

### Connections to Course Themes

This module brings together everything you've learned:

**Module 1 (Statistics)**: 
- Gaussian distributions and CLT underpin GPs
- Sampling and probability distributions

**Module 2-3 (Stellar/Phase Space Dynamics)**:
- Kernels analogous to correlation functions
- Emulating dynamical systems

**Module 4 (Radiative Transfer)**:
- Surrogate models for expensive forward models
- Uncertainty propagation

**Module 5 (Bayesian Inference)**:
- GPs ARE Bayesian inference in function space
- Hyperparameter optimization = empirical Bayes
- Marginal likelihood = model evidence

**Module 6 Part 1 (JAX)**:
- Functional programming enables clean GP code
- Autodiff eliminates tedious gradient derivations
- JIT makes GPs fast

**The Big Picture**: You're learning to build **fast, accurate, uncertainty-aware surrogate models** for expensive physical simulations. This is a frontier skill in computational science, combining:
- Deep physics understanding (what to simulate)
- Computational prowess (how to simulate efficiently)
- Statistical sophistication (how to learn from simulations)
- Software engineering (how to build reliable tools)

You're becoming **computational astrophysicists** who can bridge theory, simulation, observation, and machine learning. This is the future of the field!

---

## 📝 Assignment Preview: Project Milestone 1

**Due**: End of Week 1 (Final Project)

**Deliverable**: Working GP implementation + validation report

**Specific tasks**:
1. Implement GP regression from scratch in JAX (kernel, prediction, hyperparameter optimization)
2. Apply to scalar predictions from your JAX N-body simulations
3. Validate uncertainty calibration (coverage, CRPS)
4. Interpret learned hyperparameters (what physics did the GP discover?)
5. Document failure modes (what happens when you extrapolate? predict time series?)

**Success criteria**:
- [ ] Code runs without errors, is well-documented
- [ ] Test RMSE < 10% of typical output values
- [ ] Uncertainty is calibrated (coverage within 5% of nominal)
- [ ] Hyperparameters are scientifically interpreted
- [ ] Limitations are clearly articulated (sets up Week 2)

**Starter code structure** (next section) will guide you, but you'll make design decisions!

---

## 🚀 Starter Code Structure

Here's the minimal file structure for your GP module. You'll fill in the details!

```
project-final/
├── src/
│   └── gp/
│       ├── __init__.py
│       ├── kernels.py          # Kernel functions
│       ├── mean_functions.py   # Mean functions (start with zero)
│       ├── gp.py              # Core GP class
│       ├── optimization.py     # Hyperparameter optimization
│       └── utils.py           # Helper functions
├── tests/
│   ├── test_kernels.py        # Unit tests for kernels
│   ├── test_gp.py             # Integration tests
│   └── test_validation.py     # Uncertainty calibration tests
├── notebooks/
│   ├── 01_kernel_exploration.ipynb   # Visualize kernels
│   ├── 02_gp_toy_example.ipynb       # 1D regression
│   └── 03_cluster_emulation.ipynb    # Your science application
├── data/
│   ├── training_data.h5       # Your N-body simulation results
│   └── test_data.h5
├── pyproject.toml             # Package configuration
└── README.md
```

**Key files to implement**:

### `kernels.py` - Kernel Functions

```python
"""Kernel functions for Gaussian processes."""

import jax.numpy as jnp
from jax import jit
from typing import Dict

@jit
def rbf_kernel(X1: jnp.ndarray, X2: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """
    Radial Basis Function (Squared Exponential) kernel.
    
    Args:
        X1: Array of shape (n1, d)
        X2: Array of shape (n2, d)
        params: Dictionary with 'lengthscale' and 'variance'
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # TODO: Implement efficient squared distance computation
    # TODO: Return variance * exp(-0.5 * squared_dist / lengthscale**2)
    raise NotImplementedError

@jit
def matern52_kernel(X1: jnp.ndarray, X2: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Matérn 5/2 kernel."""
    # TODO: Implement
    raise NotImplementedError

# TODO: Add more kernels (periodic, ARD, composite)
```

### `gp.py` - Core GP Class

```python
"""Gaussian Process regression."""

import jax.numpy as jnp
from jax import jit
from typing import Callable, Dict, Tuple

class GaussianProcess:
    """Gaussian Process for regression."""
    
    def __init__(self, kernel: Callable, mean_fn: Callable = None):
        """
        Args:
            kernel: Kernel function k(X1, X2, params) -> K
            mean_fn: Mean function m(X) -> mean (default: zero)
        """
        self.kernel = kernel
        self.mean_fn = mean_fn if mean_fn is not None else lambda X: jnp.zeros(X.shape[0])
    
    @jit
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, params: Dict) -> Dict:
        """
        Precompute quantities for prediction.
        
        Args:
            X: Training inputs (n, d)
            y: Training outputs (n,)
            params: Kernel + noise parameters
        
        Returns:
            State dictionary for prediction
        """
        # TODO: 
        # 1. Compute K = kernel(X, X) + noise^2 * I
        # 2. Compute Cholesky decomposition L
        # 3. Solve for alpha = K^{-1} (y - mean(X))
        # 4. Return {'L': L, 'alpha': alpha, 'X': X, 'params': params}
        raise NotImplementedError
    
    @jit
    def predict(self, state: Dict, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict at test points.
        
        Args:
            state: Fitted GP state from fit()
            X_test: Test inputs (n_test, d)
        
        Returns:
            mu: Predictive mean (n_test,)
            sigma: Predictive std deviation (n_test,)
        """
        # TODO:
        # 1. Compute k_s = kernel(X_train, X_test)
        # 2. Compute mu = mean(X_test) + k_s.T @ alpha
        # 3. Compute sigma via v = L \ k_s; sigma^2 = k_ss - v.T @ v
        raise NotImplementedError
    
    def log_marginal_likelihood(self, X: jnp.ndarray, y: jnp.ndarray, params: Dict) -> float:
        """
        Compute log p(y | X, params).
        
        This is what you'll optimize to find best hyperparameters!
        """
        # TODO: Implement the three terms (data fit + complexity + constant)
        raise NotImplementedError
```

### `optimization.py` - Hyperparameter Optimization

```python
"""Hyperparameter optimization for GPs."""

import jax
import optax
from typing import Dict, Callable

def optimize_hyperparameters(
    gp: GaussianProcess,
    X: jnp.ndarray,
    y: jnp.ndarray,
    initial_params: Dict,
    n_steps: int = 1000,
    learning_rate: float = 0.01
) -> Dict:
    """
    Optimize GP hyperparameters via gradient ascent on marginal likelihood.
    
    Args:
        gp: GaussianProcess instance
        X, y: Training data
        initial_params: Starting hyperparameters
        n_steps: Optimization iterations
        learning_rate: Step size
    
    Returns:
        Optimized hyperparameters
    """
    # TODO:
    # 1. Define loss = -log_marginal_likelihood
    # 2. Use optax to optimize (Adam or L-BFGS)
    # 3. Transform parameters to enforce positivity (use log-space)
    # 4. Track loss over iterations (for debugging)
    # 5. Return best parameters
    raise NotImplementedError

def initialize_hyperparameters(X: jnp.ndarray, y: jnp.ndarray) -> Dict:
    """Heuristic initialization (see lecture notes!)"""
    # TODO: Implement heuristics for lengthscale, variance, noise
    raise NotImplementedError
```

### `utils.py` - Validation and Diagnostics

```python
"""Utilities for GP validation."""

import jax.numpy as jnp
from typing import Tuple

def calibration_metrics(y_true: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> Dict:
    """
    Compute uncertainty calibration metrics.
    
    Returns:
        Dictionary with RMSE, NLL, CRPS, coverage at different levels
    """
    # TODO: Implement
    raise NotImplementedError

def coverage(y_true: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, z: float = 1.0) -> float:
    """Fraction of data within z standard deviations."""
    # TODO: Implement
    raise NotImplementedError

def plot_predictions_1d(X_train, y_train, X_test, mu, sigma, y_test=None):
    """Visualization helper for 1D functions."""
    # TODO: Plot mean, confidence bands, training data, optional test data
    raise NotImplementedError
```

---

## 🎓 Self-Assessment Questions

Test your understanding before starting the project:

**Conceptual**:
1. Explain GP regression to someone who knows Bayesian inference but not GPs. How is it different from parameter inference?
2. Why do GPs provide uncertainty estimates naturally, while neural networks don't?
3. What does the lengthscale hyperparameter $\ell$ control physically?
4. When would you use a Matérn 5/2 kernel instead of SE?
5. Explain the marginal likelihood "Occam's razor" in your own words.

**Mathematical**:
1. Derive the GP predictive mean formula from Gaussian conditioning. Show each step.
2. Why does the predictive variance not depend on observed values $\mathbf{y}$?
3. Prove that $k(x, x') = k_1(x, x') + k_2(x, x')$ is a valid kernel if $k_1, k_2$ are valid.
4. What happens to predictions as noise $\sigma_n^2 \to 0$? As $\sigma_n^2 \to \infty$?
5. Compute the gradient $\partial k_{\text{SE}}/\partial \ell$ (used in hyperparameter optimization).

**Computational**:
1. Why use Cholesky instead of direct inversion for $\mathbf{K}^{-1}$?
2. What is the computational complexity of: (a) training a GP, (b) predicting at $m$ test points?
3. How would you implement ARD (per-dimension lengthscales) in JAX?
4. Explain what `jax.jit` does and when to use it for GPs.
5. How would you parallelize GP predictions at many test points?

**Scientific**:
1. You fit a GP to cluster core radius vs initial conditions. Which IC (N, M, Q, c) do you expect to have the *smallest* lengthscale? Why?
2. Your GP has poor coverage (only 50% of test data within 1-sigma). What are two possible causes and fixes?
3. You want to predict cluster evolution out to 500 Myr, but trained on [0, 200] Myr. What happens? How could you improve this?
4. When would you use a GP vs a neural network for emulating simulations?
5. Propose a physics-informed mean function for cluster core radius evolution.

**Answers**: Think through these yourself first! Discuss with classmates. Come to office hours if stuck.

---

## 📚 Key Takeaways

```{admonition} The Core Ideas
:class: important

1. **GPs are distributions over functions** - Bayesian inference in infinite dimensions
2. **Kernels encode prior beliefs** - Smoothness, periodicity, structure
3. **Conditioning gives predictions** - One formula, many applications
4. **Uncertainty is built-in** - Epistemic + aleatoric, calibration crucial
5. **Hyperparameters learn from data** - Marginal likelihood balances fit and complexity
6. **JAX makes it fast** - Autodiff, JIT, vectorization
7. **GPs have limits** - High dimensions, large data, complex outputs
8. **NNs complement GPs** - Different strengths, can combine

**The Philosophical Point**: Machine learning is not just "black boxes that work." It's principled probabilistic inference with mathematical foundations. GPs show this beautifully: every aspect (kernel, hyperparameters, predictions, uncertainty) has clear interpretation and connection to statistics and physics.

**The Practical Point**: You now have a powerful tool for surrogate modeling. Use it to:
- Speed up parameter space exploration (10000x faster than simulations)
- Quantify uncertainty in predictions (know when to trust your emulator)
- Discover patterns in data (which parameters matter?)
- Bridge simulations and observations (infer ICs from data)

This is computational astrophysics in 2025!
```

---

*This material will be the foundation for your Final Project Milestone 1. Come to lecture prepared to ask questions! We'll work through examples and build intuition before you implement your own GP.*

*Next lecture: We'll introduce the neural network architectures you'll use in Weeks 3-4 to overcome GP limitations.*

**Now: Go implement a Gaussian Process and emulate the universe!** 🌌