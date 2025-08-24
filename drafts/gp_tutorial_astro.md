# Gaussian Processes for Astronomers: From Theory to Stellar Applications

## Prerequisites Check

Before we begin exploring Gaussian Processes, let's review the mathematical tools we'll need. Don't worry if some concepts are rusty - we'll build everything from the ground up!

### Mathematical Prerequisites
You should be familiar with:
- **Probability basics**: Mean, variance, probability distributions
- **Linear algebra**: Matrix multiplication, inverse, eigenvalues
- **Multivariate calculus**: Partial derivatives, gradients
- **Basic statistics**: Linear regression, least squares

### Notation Convention
Throughout this document:
- Vectors are **column vectors** by default: $\mathbf{x} \in \mathbb{R}^{n \times 1}$
- $\mathbf{x}^T$ denotes the transpose (converting column to row vector)
- $\mathcal{N}(\mu, \sigma^2)$ denotes a normal distribution with mean $\mu$ and variance $\sigma^2$
- $\mathbf{X}$ (capital) denotes a matrix or a collection of vectors
- $\mathbb{E}[\cdot]$ denotes expectation (average)
- $\text{Var}[\cdot]$ denotes variance
- $\text{Cov}[\cdot, \cdot]$ denotes covariance

---

## Introduction: Why Gaussian Processes for Astronomy?

Imagine you're studying the radial velocity curve of a star with an exoplanet. You have observations at irregular times with different measurement uncertainties. You need to:
1. Interpolate between observations to find the velocity at any time
2. Quantify your uncertainty at each point
3. Detect periodic signals buried in noise
4. Avoid overfitting sparse data

Traditional approaches like polynomial fitting or splines give you a single "best fit" curve. But what if you could have a **probability distribution over all possible curves** that fit your data? This is exactly what Gaussian Processes provide.

Gaussian Processes (GPs) are particularly powerful in astronomy because they:
- **Quantify uncertainty**: Every prediction comes with error bars
- **Handle irregular sampling**: Perfect for typical astronomical observations
- **Incorporate prior knowledge**: Through kernel selection
- **Avoid overfitting**: Naturally regularized by prior assumptions
- **Work with small datasets**: Unlike neural networks that need thousands of examples

In the context of your N-body simulations, GPs can:
- Emulate expensive simulations (predict outcomes without running full N-body)
- Learn correlations in phase space
- Model stellar density fields
- Interpolate between discrete particle positions to get smooth fields

Let's build the mathematical framework from first principles.

---

## Part 1: From Gaussians to Gaussian Processes

### The Gaussian Distribution - Our Foundation

A Gaussian (normal) distribution is completely characterized by its mean and variance:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where:
- $\mu$ is the mean (center of the distribution)
- $\sigma^2$ is the variance (spread of the distribution)
- $\sigma$ is the standard deviation

### The Multivariate Gaussian

For a vector $\mathbf{x} \in \mathbb{R}^n$, the multivariate Gaussian is:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^{n \times 1}$ is the mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$ is the covariance matrix
- $|\boldsymbol{\Sigma}|$ is the determinant of $\boldsymbol{\Sigma}$

We write this as: $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

### 🌟 Astronomical Example: Stellar Velocities

Consider measuring the 3D velocity of a star. Let $\mathbf{v} = [v_x, v_y, v_z]^T$ be the velocity vector. If measurements have uncertainties and are correlated:

$$\mathbf{v} \sim \mathcal{N}\left(\begin{bmatrix} 10 \\ -5 \\ 2 \end{bmatrix}, \begin{bmatrix} 4 & 1 & 0 \\ 1 & 3 & 0.5 \\ 0 & 0.5 & 2 \end{bmatrix}\right)$$

The off-diagonal terms in $\boldsymbol{\Sigma}$ represent correlations between velocity components.

📝 **Checkpoint 1**: If $v_x$ and $v_y$ are independent, what would $\Sigma_{12}$ equal?  
*Answer: $\Sigma_{12} = 0$ (no correlation between independent variables)*

### Key Properties of Multivariate Gaussians

**1. Marginalization**: If we observe only some components, the rest remain Gaussian.

**2. Conditioning**: If we observe some components, the conditional distribution of the others is also Gaussian.

For a Gaussian vector partitioned as:
$$\begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{bmatrix}\right)$$

The conditional distribution is:
$$\mathbf{x}_1 | \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

where:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

These equations are the heart of GP predictions!

### The Leap to Infinite Dimensions

Here's the key insight: a Gaussian Process is simply an **infinite-dimensional Gaussian distribution**. Instead of a distribution over vectors, it's a distribution over functions.

**Definition**: A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

We write: $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$

where:
- $m(\mathbf{x})$ is the mean function
- $k(\mathbf{x}, \mathbf{x}')$ is the covariance function (kernel)

---

## Part 2: Gaussian Process Regression

### The Regression Problem

Given:
- Training inputs: $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n]^T \in \mathbb{R}^{n \times d}$
- Training outputs: $\mathbf{y} = [y_1, y_2, ..., y_n]^T \in \mathbb{R}^{n \times 1}$
- Test inputs: $\mathbf{X}_* = [\mathbf{x}_{*1}, \mathbf{x}_{*2}, ..., \mathbf{x}_{*m}]^T \in \mathbb{R}^{m \times d}$

Goal: Predict $\mathbf{f}_*$ at test points with uncertainty.

### The Prior

Before seeing any data, we assume:
$$f(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))$$

(We often set the mean to zero for simplicity; non-zero means can be handled by preprocessing.)

### The Likelihood

We observe noisy measurements:
$$y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2)$$

where $\sigma_n^2$ is the noise variance.

### The Posterior (The Key Result!)

Given the joint Gaussian:
$$\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K + \sigma_n^2 I & K_* \\ K_*^T & K_{**} \end{bmatrix}\right)$$

where:
- $K \in \mathbb{R}^{n \times n}$ with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ (training covariances)
- $K_* \in \mathbb{R}^{n \times m}$ with $[K_*]_{ij} = k(\mathbf{x}_i, \mathbf{x}_{*j})$ (train-test covariances)
- $K_{**} \in \mathbb{R}^{m \times m}$ with $[K_{**}]_{ij} = k(\mathbf{x}_{*i}, \mathbf{x}_{*j})$ (test covariances)

Using the conditioning formulas from Part 1:

**Posterior Mean** (our prediction):
$$\boldsymbol{\mu}_* = K_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

**Posterior Covariance** (our uncertainty):
$$\boldsymbol{\Sigma}_* = K_{**} - K_*^T (K + \sigma_n^2 I)^{-1} K_*$$

**Posterior Variance** (uncertainty at each test point):
$$\sigma_*^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

where $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), ..., k(\mathbf{x}_n, \mathbf{x}_*)]^T$

### 🌟 Complete Example: Interpolating Stellar Brightness

Let's interpolate the brightness of a variable star from 3 observations.

**Data:**
- Times: $t = [0, 1, 3]$ days
- Brightness: $y = [10.0, 10.5, 9.8]$ magnitudes
- Measurement noise: $\sigma_n = 0.1$ mag

**Kernel** (RBF/Gaussian kernel):
$$k(t, t') = \sigma_f^2 \exp\left(-\frac{(t-t')^2}{2\ell^2}\right)$$

with $\sigma_f = 1.0$ (signal variance) and $\ell = 2.0$ (length scale).

**Step 1:** Compute training covariance matrix:
$$K = \begin{bmatrix} k(0,0) & k(0,1) & k(0,3) \\ k(1,0) & k(1,1) & k(1,3) \\ k(3,0) & k(3,1) & k(3,3) \end{bmatrix} = \begin{bmatrix} 1.0 & 0.882 & 0.325 \\ 0.882 & 1.0 & 0.606 \\ 0.325 & 0.606 & 1.0 \end{bmatrix}$$

**Step 2:** Add noise:
$$K + \sigma_n^2 I = \begin{bmatrix} 1.01 & 0.882 & 0.325 \\ 0.882 & 1.01 & 0.606 \\ 0.325 & 0.606 & 1.01 \end{bmatrix}$$

**Step 3:** Predict at $t_* = 2.0$:

Compute $\mathbf{k}_* = [k(0,2), k(1,2), k(3,2)]^T = [0.606, 0.882, 0.882]^T$

Mean prediction:
$$\mu_* = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y} = 10.21 \text{ mag}$$

Variance:
$$\sigma_*^2 = 1.0 - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_* = 0.18$$

So our prediction is $10.21 \pm 0.42$ mag (using $\sigma_* = \sqrt{0.18}$).

📝 **Checkpoint 2**: What happens to the uncertainty $\sigma_*$ as we move far from training points?  
*Answer: It approaches $\sigma_f = 1.0$, reverting to the prior uncertainty*

---

## Part 3: Covariance Functions (Kernels)

### The Role of the Kernel

The kernel encodes our assumptions about the function:
- Smoothness
- Periodicity
- Length scales
- Amplitude

It must be a valid covariance function (positive semi-definite).

### Common Kernels and Their Properties

#### Radial Basis Function (RBF) / Squared Exponential

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)$$

For multi-dimensional inputs:
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_{i=1}^d \frac{(x_i-x'_i)^2}{\ell_i^2}\right)$$

**Properties:**
- Infinitely differentiable (very smooth functions)
- $\sigma_f^2$: variance (function amplitude)
- $\ell$: length scale (how quickly correlation decays with distance)

**When to use:** Smooth phenomena like gravitational potentials, density fields

#### Matérn Kernel

$$k_{\nu}(r) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}\frac{r}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{r}{\ell}\right)$$

where $r = |\mathbf{x} - \mathbf{x}'|$, $K_\nu$ is a modified Bessel function.

Special cases:
- $\nu = 1/2$: Exponential kernel (rough functions)
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable
- $\nu \to \infty$: RBF kernel

**When to use:** When you want to control smoothness explicitly

#### Periodic Kernel

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)$$

where $p$ is the period.

**When to use:** Stellar rotation, orbital phenomena, pulsating variables

#### Linear Kernel

$$k(\mathbf{x}, \mathbf{x}') = \sigma_b^2 + \sigma_v^2 (\mathbf{x} - \mathbf{c})^T(\mathbf{x}' - \mathbf{c})$$

**When to use:** When you expect linear trends

### 🌟 Astronomical Example: Designing Kernels for Light Curves

For a star with both periodic variability and long-term trends:

$$k = k_{\text{periodic}} + k_{\text{RBF}} + k_{\text{linear}}$$

This captures:
- Short-term periodic variations (rotation, pulsation)
- Medium-term smooth changes (spots evolving)
- Long-term linear trends (evolutionary changes)

### Combining Kernels

Kernels can be combined:
- **Addition**: $k_1 + k_2$ (independent processes)
- **Multiplication**: $k_1 \times k_2$ (interaction between processes)

📝 **Checkpoint 3**: What kernel would you use for a function that's periodic in time but with amplitude that decays with distance from Earth?  
*Answer: $k(t,r,t',r') = k_{\text{periodic}}(t,t') \times k_{\text{RBF}}(r,r')$*

### 🎨 Visual Description: Kernel Effects

*[Imagine plots showing:]*
- RBF with small $\ell$: Wiggly functions with rapid variation
- RBF with large $\ell$: Smooth, slowly varying functions
- Periodic kernel: Regular oscillations
- Matérn-1/2: Rough, non-differentiable paths
- Sum of kernels: Complex behavior combining multiple patterns

---

## Part 4: Hyperparameter Learning

### The Marginal Likelihood

Kernels have hyperparameters $\boldsymbol{\theta}$ (like $\ell$, $\sigma_f$, $\sigma_n$). We can learn these from data using the marginal likelihood:

$$\log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T K_{\boldsymbol{\theta}}^{-1}\mathbf{y} - \frac{1}{2}\log|K_{\boldsymbol{\theta}}| - \frac{n}{2}\log(2\pi)$$

where $K_{\boldsymbol{\theta}} = K + \sigma_n^2 I$ with kernel parameters $\boldsymbol{\theta}$.

This has three terms:
1. **Data fit**: $-\frac{1}{2}\mathbf{y}^T K_{\boldsymbol{\theta}}^{-1}\mathbf{y}$
2. **Complexity penalty**: $-\frac{1}{2}\log|K_{\boldsymbol{\theta}}|$
3. **Normalization**: $-\frac{n}{2}\log(2\pi)$

### Computing Gradients

For optimization, we need:
$$\frac{\partial}{\partial \theta_j} \log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\theta}) = \frac{1}{2}\text{tr}\left((K^{-1}\mathbf{y}\mathbf{y}^T K^{-1} - K^{-1})\frac{\partial K}{\partial \theta_j}\right)$$

Let's compute this for the RBF length scale:
$$\frac{\partial K_{ij}}{\partial \ell} = K_{ij} \cdot \frac{(x_i - x_j)^2}{\ell^3}$$

### 🌟 Example: Learning the Period of a Variable Star

Given noisy observations of a periodic variable:

1. Start with initial guess: $p_0 = 1.0$ day
2. Compute marginal likelihood
3. Gradient ascent on $p$: $p \leftarrow p + \eta \frac{\partial \log p(\mathbf{y})}{\partial p}$
4. Repeat until convergence

The learned period maximizes the probability of the observed data.

📝 **Checkpoint 4**: Why does the complexity penalty $-\frac{1}{2}\log|K|$ prevent overfitting?  
*Answer: Smaller length scales make K more "flexible" but increase |K|, penalizing complexity*

⚠️ **Advanced Note**: The marginal likelihood can have multiple local maxima. It's often good to try several initializations or use prior knowledge to constrain the search.

---

## Part 5: Computational Considerations

### The Computational Challenge

GP regression requires:
- Computing $(K + \sigma_n^2 I)^{-1}\mathbf{y}$: $O(n^3)$ time
- Storing $K$: $O(n^2)$ memory

For large datasets (n > 10,000), this becomes prohibitive.

### Efficient Implementation via Cholesky Decomposition

Instead of directly inverting, use Cholesky decomposition:

$K + \sigma_n^2 I = LL^T$ where $L$ is lower triangular.

Then solve:
1. $L\boldsymbol{\alpha} = \mathbf{y}$ (forward substitution)
2. $L^T\boldsymbol{\beta} = \boldsymbol{\alpha}$ (back substitution)

Now $\boldsymbol{\beta} = (K + \sigma_n^2 I)^{-1}\mathbf{y}$

This is more stable and efficient than direct inversion.

### Sparse Approximations

For large datasets, approximate with m << n inducing points:

**Subset of Regressors (SoR):**
$$\mu_* \approx K_{*u}K_{uu}^{-1}\mathbf{y}_u$$

where u indexes a subset of training points.

**Low-Rank Approximations:**
$$K \approx K_{nu}K_{uu}^{-1}K_{un}$$

This reduces complexity to $O(nm^2)$.

### 🌟 Astronomical Application: Gaussian Process Emulation

For expensive N-body simulations:

1. Run full simulations at m "inducing" points in parameter space
2. Use GP to predict outcomes at new parameters
3. Computational savings: $O(n^{11/3})$ → $O(m^3)$ where m << n

Example parameter space for globular clusters:
- Initial mass: $10^4 - 10^6 M_\odot$
- Concentration: $c = 0.5 - 2.5$
- Binary fraction: $0 - 0.5$

Run 100 simulations, emulate millions of configurations!

---

## Part 6: Advanced Topics

### GP Classification

For binary outcomes (e.g., "cluster undergoes core collapse" vs "remains stable"):

Use a link function: $p(y=1|\mathbf{x}) = \Phi(f(\mathbf{x}))$

where $\Phi$ is the probit or logistic function and $f \sim \mathcal{GP}$.

The posterior is no longer Gaussian - use Laplace approximation or MCMC.

### Multi-Output GPs

For vector-valued functions (e.g., 3D velocity fields):

$$\begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ f_3(\mathbf{x}) \end{bmatrix} \sim \mathcal{GP}\left(\mathbf{0}, \begin{bmatrix} k_{11} & k_{12} & k_{13} \\ k_{21} & k_{22} & k_{23} \\ k_{31} & k_{32} & k_{33} \end{bmatrix}\right)$$

Cross-covariances $k_{ij}$ capture correlations between outputs.

### Deep Gaussian Processes

Stack GPs hierarchically:
$$f_L = \text{GP}_L(f_{L-1}), \quad f_{L-1} = \text{GP}_{L-1}(f_{L-2}), \quad ..., \quad f_1 = \text{GP}_1(\mathbf{x})$$

This allows learning of feature representations, similar to deep neural networks but with uncertainty quantification.

⚠️ **Advanced Box: Spectral Mixture Kernels**
For complex patterns, use:
$$k(x,x') = \sum_{q=1}^Q \sigma_q^2 \exp\left(-\frac{(x-x')^2}{2\ell_q^2}\right)\cos(2\pi w_q(x-x'))$$
This can approximate any stationary kernel via mixture of Gaussians in frequency space.

---

## Part 7: Applications to Stellar Dynamics

### Example 1: Smoothing N-Body Density Fields

**Problem**: Convert discrete particle positions to smooth density field.

**GP Solution**:
- Input: 3D position $\mathbf{r}$
- Output: Density $\rho(\mathbf{r})$
- Training: Particle positions with kernel density estimates
- Kernel: RBF with length scale ~ mean inter-particle spacing

**Implementation**:
```python
# Pseudo-code
kernel = RBF(length_scale=r_h/sqrt(N))
gp = GaussianProcess(kernel=kernel)
gp.fit(particle_positions, local_densities)
smooth_density = gp.predict(grid_points)
```

### Example 2: Learning Dynamical Mappings

**Problem**: Predict cluster half-mass radius evolution without running full simulation.

**Setup**:
- Input: Initial conditions $(M_0, r_{h,0}, c_0, ...)$
- Output: $r_h(t)$ at multiple times
- Training: 100 full N-body runs
- Kernel: RBF with different length scales per dimension (ARD)

**Advantages over Neural Networks**:
- Works with just 100 training runs (NN might need thousands)
- Provides uncertainty bounds on predictions
- No architecture selection needed

### Example 3: Orbit Classification in Phase Space

**Problem**: Identify chaotic vs regular orbits in galactic potential.

**Features**:
- Energy $E$
- Angular momentum components $L_x, L_y, L_z$
- Orbit shape parameters

**GP Classifier**:
- Prior: Zero mean GP
- Kernel: RBF + Linear (captures both local and global patterns)
- Link function: Logistic

**Result**: Probability map of chaos in phase space with uncertainty.

📝 **Checkpoint 5**: For a cluster with 10^6 particles, what's the computational complexity of direct GP vs inducing point approximation with 1000 points?  
*Answer: Direct: O((10^6)^3) = O(10^18), Approximation: O(10^6 × 1000^2) = O(10^12), speedup of 10^6!*

---

## Part 8: Practical Implementation Guide

### Choosing Mean Functions

While we often use zero mean, sometimes a parametric mean helps:

$$m(\mathbf{x}) = \boldsymbol{\beta}^T \boldsymbol{\phi}(\mathbf{x})$$

where $\boldsymbol{\phi}$ are basis functions.

Example for stellar evolution:
$$m(t) = \beta_0 + \beta_1 t + \beta_2 \log(t)$$

captures power-law evolution.

### Kernel Selection Strategy

1. **Start simple**: RBF with single length scale
2. **Check residuals**: Look for patterns
3. **Add complexity**: 
   - Periodic patterns → Add periodic kernel
   - Different scales → Use multiple length scales
   - Non-stationarity → Consider non-stationary kernels
4. **Validate**: Use held-out data or cross-validation

### Numerical Stability Tips

**Problem**: $K + \sigma_n^2 I$ becomes ill-conditioned

**Solutions**:
- Add jitter: $K + (\sigma_n^2 + \epsilon)I$ with $\epsilon \sim 10^{-6}$
- Standardize inputs: Zero mean, unit variance
- Work in log-space for positive parameters
- Use stable Cholesky updates for sequential data

### Common Pitfalls and Solutions

**Pitfall 1**: Optimization gets stuck in bad local maximum
- *Solution*: Multiple random restarts, or use prior knowledge

**Pitfall 2**: Predictions explode far from data
- *Solution*: Use bounded kernels or add trend function

**Pitfall 3**: Too slow for real-time predictions
- *Solution*: Pre-compute $(K + \sigma_n^2 I)^{-1}\mathbf{y}$, use sparse methods

### 🌟 Complete Code Structure (Pseudo-code)

```python
class GaussianProcess:
    def __init__(self, kernel, noise=1e-6):
        self.kernel = kernel
        self.noise = noise
        
    def fit(self, X, y):
        # Compute kernel matrix
        K = self.kernel(X, X)
        K_noise = K + self.noise * eye(len(X))
        
        # Cholesky decomposition
        self.L = cholesky(K_noise)
        
        # Solve for alpha
        self.alpha = solve_triangular(self.L.T, 
                      solve_triangular(self.L, y))
        
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        # Compute cross-covariance
        K_star = self.kernel(self.X_train, X_test)
        
        # Mean prediction
        mu = K_star.T @ self.alpha
        
        # Variance
        v = solve_triangular(self.L, K_star)
        var = self.kernel(X_test, X_test) - v.T @ v
        
        return mu, sqrt(diag(var))
```

---

## Quick Reference Summary

### Key Equations

| Component | Formula |
|-----------|---------|
| GP Prior | $f \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ |
| Posterior Mean | $\boldsymbol{\mu}_* = K_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$ |
| Posterior Variance | $\sigma_*^2 = k_{**} - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$ |
| Marginal Likelihood | $\log p(\mathbf{y}) = -\frac{1}{2}\mathbf{y}^T K^{-1}\mathbf{y} - \frac{1}{2}\log\|K\| - \frac{n}{2}\log(2\pi)$ |

### Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| RBF | $k = \sigma_f^2 \exp(-\frac{r^2}{2\ell^2})$ | Smooth functions |
| Matérn-3/2 | $k = \sigma_f^2(1 + \frac{\sqrt{3}r}{\ell})\exp(-\frac{\sqrt{3}r}{\ell})$ | Once-differentiable |
| Periodic | $k = \sigma_f^2 \exp(-\frac{2\sin^2(\pi r/p)}{\ell^2})$ | Periodic phenomena |
| Linear | $k = \sigma_b^2 + \sigma_v^2 \mathbf{x}^T\mathbf{x}'$ | Linear trends |

### Computational Complexity

| Operation | Standard GP | Sparse GP (m inducing) |
|-----------|------------|------------------------|
| Training | $O(n^3)$ | $O(nm^2)$ |
| Prediction | $O(n^2)$ | $O(m^2)$ |
| Memory | $O(n^2)$ | $O(nm)$ |

---

## Conclusion: The Power of Gaussian Processes

You now have the mathematical foundation to:
- Build probabilistic models that quantify uncertainty
- Interpolate and extrapolate with confidence bounds
- Learn complex patterns from limited data
- Design kernels that encode physical knowledge
- Efficiently emulate expensive simulations

Gaussian Processes offer unique advantages for astronomical applications:
- **Small data regime**: Unlike neural networks, GPs work well with dozens of examples
- **Uncertainty quantification**: Every prediction comes with error bars
- **Interpretability**: Kernels have physical meaning
- **Flexibility**: No architecture selection - just choose appropriate kernel

For your N-body projects, GPs can:
- Emulate expensive simulations across parameter space
- Smooth discrete particle data into continuous fields
- Learn dynamical mappings with uncertainty
- Classify orbital types probabilistically

The journey from Gaussian distributions to processes that can model stellar dynamics shows the elegant power of probabilistic thinking. Combined with the neural networks you'll learn next, you'll have both probabilistic (GP) and deterministic (NN) tools for tackling complex dynamical systems.

Remember the key insight: **GPs are distributions over functions**. When you need not just a prediction but a measure of confidence in that prediction, GPs are your tool of choice.

---

## Final Checkpoints

📝 **Final Challenge 1**: You observe a star's radial velocity at 5 epochs with 0.5 m/s uncertainties. Design a GP to predict the velocity at intermediate times and detect a potential 10-day periodic planet signal. What kernel would you use?

📝 **Final Challenge 2**: You've run 50 expensive N-body simulations of globular clusters with different initial conditions. How would you use a GP to predict outcomes for 10,000 new initial conditions? What would be your input features and kernel choice?

📝 **Final Challenge 3**: Your GP predictions have huge uncertainty between training points. What are three things you could try to improve this?

*Answers to consider:*
1. *Periodic + RBF kernel to capture planet signal + stellar noise*
2. *Inputs: mass, concentration, binary fraction, etc. Kernel: RBF with ARD to learn relevant length scales*
3. *Add more training data, decrease length scale, or add a mean function to capture trends*

Now you're ready to combine GPs with neural networks in JAX for powerful hybrid models!