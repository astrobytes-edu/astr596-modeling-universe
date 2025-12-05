---
title: "Part II: Gaussian Processes - Implementation and Practice"
subtitle: "The Complete Emulation Workflow | Module 7: The Learnable Universe | ASTR 596"
---

> "In theory, theory and practice are the same. In practice, they are not."
>
> — Attributed to Yogi Berra (and many others)

---

## Overview

**Prerequisites**: Complete [Part I: GP Theory](02a-gp-theory.md) before proceeding. You should understand:

- What a Gaussian Process is (distribution over functions)
- Gaussian conditioning (how predictions work)
- Kernels (encoding smoothness and structure)
- When to use GPs vs other methods

**In Part II, you'll learn**:

- The complete emulation workflow from simulations to predictions
- How to train GPs by optimizing marginal likelihood
- Numerical implementation techniques in JAX
- Validation and diagnostics (calibration, OOD detection)
- How to use emulators for scientific applications

---

## Quick Recap: What You Learned in Part I

### The Core Ideas

**Gaussian Processes are**:

- Probability distributions over **functions** $f: \mathbb{R}^D \to \mathbb{R$
- Specified by mean function $m(\mathbf{x})$ and kernel $k(\mathbf{x}, \mathbf{x}')$
- A Bayesian approach to function learning with **built-in uncertainty**

**The Prediction Formulas**:

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ and GP prior $f \sim \mathcal{GP}(m, k)$:

$$
\boxed{
\begin{align}
\mu(\mathbf{x}_*) &= m(\mathbf{x}_*) + \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m}) \\[0.5em]
\sigma^2(\mathbf{x}_*) &= k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
\end{align}
}
$$

where:

- $\mathbf{K}$: kernel matrix with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
- $\mathbf{k}_*$: kernel vector with $[\mathbf{k}_*]_i = k(\mathbf{x}_i, \mathbf{x}_*)$
- $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$
- $\sigma_n^2$: observation noise variance

**The Key Kernels**:

1. **Squared Exponential (SE)**: $k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)$ (infinitely smooth)
2. **Matérn-5/2**: Twice differentiable, more realistic smoothness
3. **ARD (Automatic Relevance Determination)**: Per-dimension lengthscales $\ell_d$

**When to Use GPs**:

- ✅ Low-dimensional inputs ($D < 20$)
- ✅ Limited training data ($N < 5000$)
- ✅ Smooth physics
- ✅ Uncertainty quantification critical
- ✅ Interpretability matters

**Now let's build one!**

---

## The Emulation Workflow: From Simulations to Predictions

Now that you understand GPs conceptually, let's walk through the complete workflow you'll implement in Part 2.

### Step 1: Generate Training Data

**Goal**: Sample initial condition space intelligently to get informative training set.

:::{admonition} Pre-Training Checklist: Invariances & Normalization
:class: important

**Before generating training data, ensure your inputs respect physical symmetries:**

1. **Translational invariance**:
   - ❌ Don't use absolute positions of stars
   - ✅ Use relative features (e.g., center-of-mass quantities, density profile parameters)

2. **Rotational invariance**:
   - ❌ Don't use raw velocity vectors
   - ✅ Use scalar quantities (virial ratio $Q$, angular momentum magnitude)

3. **Scale invariance** (where appropriate):
   - ❌ Don't mix dimensional quantities with different scales
   - ✅ Use dimensionless ratios (e.g., $R_{\rm core}/R_h$, $Q = 2K/|W|$)
   - ✅ Scale radii by half-mass radius, velocities by virial velocity

4. **Input normalization**:
   - Z-score normalize: $(x - \mu_x) / \sigma_x$ for each dimension
   - For parameters spanning orders of magnitude (e.g., $N \in [500, 2000]$), use $\log N$ first
   - Store normalization constants for denormalizing predictions

5. **Test invariance** (after training):
   - Rotate/translate initial conditions → predictions should be identical
   - Scale system → predictions should scale appropriately
   - If not invariant, your features are wrong!

**Why this matters**:

- Reduces effective dimensionality (GP doesn't waste capacity learning translations)
- Ensures emulator works for any cluster orientation/position
- Matches physics: cluster evolution doesn't depend on where you put origin!

**For your N-body emulation**:

- ✅ Use $(Q, N, a)$ — all intrinsic properties
- ✅ Consider $\log N$ for scaling behavior; $M_{\rm tot}$ is derived from $N$ using fixed Kroupa IMF
- ✅ Normalize inputs: Z-score each dimension after any log-transforms
- ✅ Outputs already intrinsic (bound fraction, $R_{\rm core}$)
:::

**Naive approach**: Random sampling

- Draw $(Q, N, a)$ uniformly from allowed ranges
- Problem: Might cluster samples, miss important regions

**Better approach**: Latin Hypercube Sampling (LHS)

- Stratified sampling ensuring coverage in each dimension
- Maximizes "space-filling" property
- Standard in experimental design

**Best approach**: Active learning (advanced)

- Start with LHS
- Iteratively add simulations where GP is most uncertain
- Adaptively refine emulator

**For your project**: Provided dataset uses LHS. You'll add 50 more simulations where you choose (tests if you can identify gaps).

### Step 2: Train the GP

**Goal**: Find hyperparameters $\boldsymbol{\theta} = (\sigma_f^2, \ell, \sigma_n^2)$ that best explain data.

**Approach**: Maximize **marginal likelihood** (evidence):

$$
\log p(\mathbf{y} | X, \boldsymbol{\theta}) = -\frac{1}{2} \mathbf{y}^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}| - \frac{N}{2} \log(2\pi)
$$

**Three terms**:

1. **Data fit**: $-\frac{1}{2} \mathbf{y}^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$ (how well does GP explain data?)
2. **Complexity penalty**: $-\frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}|$ (Occam's razor—prefer simpler explanations)
3. **Normalization**: $-\frac{N}{2} \log(2\pi)$ (constant, doesn't affect optimization)

**Interpretation** (Occam's Razor):

- Small $\ell$ → Very flexible GP (can fit any data) → High complexity penalty
- Large $\ell$ → Very rigid GP (smooth functions only) → Low complexity penalty, but poor data fit
- Marginal likelihood automatically trades off fit vs complexity
- **No manual regularization needed!** You still optimize hyperparameters by maximizing the marginal likelihood, but this automatic Occam's razor avoids the manual tuning of regularization parameters common in neural networks

**Optimization**:

- Use gradient-based optimizer (L-BFGS, Adam)
- JAX computes gradients automatically: `jax.grad(log_marginal_likelihood)`
- Often use log-space for hyperparameters (ensures positivity): $\tilde{\ell} = \log \ell$

**Initialization heuristics** (guess starting point):

- $\ell$: Typical distance between data points, or fraction of input range (e.g., 0.1 × data range)
- $\sigma_f^2$: Variance of outputs $\text{Var}[\mathbf{y}]$
- $\sigma_n^2$: Estimate from simulation variability or set to small value (1% of signal variance)

### Step 3: Make Predictions

**Goal**: Given new initial conditions $\mathbf{x}_*$, predict bound fraction $\mu(\mathbf{x}_*)$ with uncertainty $\sigma(\mathbf{x}_*)$.

**Algorithm** (from conditioning formulas):

1. Compute kernel vectors:
   - $\mathbf{k}_* = k(X_{\rm train}, \mathbf{x}_*)$ (training-test covariances)
   - $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$ (test-test covariance, usually 1)

2. Compute predictive mean:
   $$\mu(\mathbf{x}_*) = m(\mathbf{x}_*) + \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m})$$

   **Note**: We typically standardize training targets to zero mean ($\mathbf{m} = 0$) during training, then de-standardize predictions for reporting. This simplifies computation while preserving generality.

3. Compute predictive variance:
   $$\sigma^2(\mathbf{x}_*) = k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

**Efficient implementation** (precompute during training):

**Step 1: Cholesky factorization**
$$\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I} = \mathbf{L} \mathbf{L}^T$$

where $\mathbf{L}$ is lower-triangular (Cholesky factor).

**Step 2: Precompute regression weights**
Solve $\mathbf{L} \boldsymbol{\beta} = \mathbf{y}$, then solve $\mathbf{L}^T \boldsymbol{\alpha} = \boldsymbol{\beta}$ to get:
$$\boldsymbol{\alpha} = \mathbf{K}_y^{-1} \mathbf{y}$$

**Step 3: Fast prediction (whitened form).**

For any test point $\mathbf{x}_*$:

$$
\boxed{
\begin{align}
\mathbf{v} &= \mathbf{L}^{-1} \mathbf{k}_* \quad \text{(solve triangular system)} \\[0.5em]
\mu(\mathbf{x}_*) &= \mathbf{k}_*^T \boldsymbol{\alpha} \quad \text{(dot product)} \\[0.5em]
\sigma^2(\mathbf{x}_*) &= k_{**} - \|\mathbf{v}\|^2 \quad \text{(variance reduction)}
\end{align}
}
$$

**Why this works**:

- Never compute $\mathbf{K}_y^{-1}$ explicitly (numerically unstable!)
- Stable: Cholesky preserves positive-definiteness

**Computational complexity summary**:

| Operation | Complexity | When Needed | Your Project ($N=250, M=10^6$) |
|-----------|-----------|-------------|--------------------------------|
| **Training** (one-time) | $O(N^3)$ | Cholesky + hyperparameter optimization | Fast for $N \sim 250$ on modern CPUs/GPUs (seconds to minutes for optimization) |
| **Marginal predictions** | $O(MN)$ | Mean/variance at $M$ independent points (typical) | ~seconds for $M=10^6$ |
| **Joint covariance** | $O(NM^2)$ | Full predictive covariance for $M$ test points via $M$ triangular solves (rarely needed) | Use batches: $M_{\rm batch} \sim 10^4$ |
| **Joint sampling** | $O(M^2 N + M^3)$ | Drawing correlated samples for visualization | Only for small $M < 1000$ |

**Key insight**: Marginal predictions scale linearly with $M$ → You can predict at millions of test points efficiently!

**For your project**: Training is instant ($N=250$), predictions are fast. Focus on getting the kernel right, not optimizing speed.

**Numerical stability**:

- Never compute $(\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1}$ explicitly!
- Use Cholesky factorization + triangular solves (stable)
- Add small "jitter" $10^{-6}$ to diagonal if matrix near-singular

:::{admonition} JAX-Native GP Numerics: Essential Implementation Tips
:class: important

**Students: Read this carefully before implementing!**

**Suggested imports at top of your Part 2 implementation file**:

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad, value_and_grad
import jax.scipy as jsp
from functools import partial
import optax  # for optimization
```

Then use:

- `jnp.*` for numerical operations (replaces `numpy`)
- `jit` for JIT compilation
- `vmap` for vectorization
- `jsp.linalg.*` for linear algebra (Cholesky, solve, eigh)
- `optax` for optimizers (replaces manual gradient descent)

Make sure JAX is configured for 64-bit: `jax.config.update("jax_enable_x64", True)` at startup.

---

1. **Always use Cholesky + triangular solves, never invert**

   ```python
   # ❌ BAD: K_inv = jnp.linalg.inv(K + noise * I)
   # ✅ GOOD: L = jnp.linalg.cholesky(K + noise * I)
   #          alpha = jsp.linalg.solve_triangular(L.T, jsp.linalg.solve_triangular(L, y, lower=True))
   ```

2. **Add scale-aware jitter for numerical stability**

   Jitter should scale with the matrix magnitude to avoid under/over-regularization:

   ```python
   # Recommended adaptive jitter
   diag_mean = jnp.mean(jnp.diag(K + noise_var * jnp.eye(n)))
   jitter = 1e-6 * diag_mean
   K_y = K + (noise_var + jitter) * jnp.eye(n)

   # If Cholesky fails: escalate jitter
   # jitter = 1e-5 * diag_mean  # try this
   # jitter = 1e-4 * diag_mean  # then this
   # jitter = 1e-3 * diag_mean  # only if desperate
   ```

   **Why scale-aware?**: A diagonal value of 1 needs smaller absolute jitter than a diagonal of 100.

#### Memory-safe batch prediction recipe

For large test sets ($M \gg N$), avoid allocating full $M \times M$ covariance matrices. Instead, predict in batches:

```python
def batch_predict_gp(X_train, y_train, X_test, K_fn, theta, batch_size=10000):
    """
    Predict at test points in batches without storing full covariance.

    Args:
        X_train: (N, D) training inputs
        y_train: (N,) training targets (assumed standardized)
        X_test: (M, D) test inputs
        K_fn: kernel function(x, x_prime, theta) -> scalar
        theta: hyperparameters dict with keys 'sigma_f', 'ell', 'noise_var'
        batch_size: points per batch (tune for your GPU memory)

    Returns:
        mu: (M,) predictive means
        var: (M,) predictive variances
    """
    N, M = X_train.shape[0], X_test.shape[0]

    # Precompute once: Cholesky factorization and alpha
    K = jnp.array([[K_fn(X_train[i], X_train[j], theta)
                    for j in range(N)] for i in range(N)])
    K_y = K + theta['noise_var'] * jnp.eye(N)
    L = jnp.linalg.cholesky(K_y)
    alpha = jsp.linalg.solve_triangular(
        L.T, jsp.linalg.solve_triangular(L, y_train, lower=True))

    mu_list, var_list = [], []
    for b_start in range(0, M, batch_size):
        b_end = min(b_start + batch_size, M)
        X_batch = X_test[b_start:b_end]

        # Kernel vector for this batch (batch_size x N)
        k_star = jnp.array([[K_fn(x, X_train[j], theta)
                             for j in range(N)] for x in X_batch])

        # Means
        mu_batch = k_star @ alpha

        # Variances
        k_star_star = jnp.array([K_fn(x, x, theta) for x in X_batch])
        v = jsp.linalg.solve_triangular(L, k_star.T, lower=True)
        var_batch = k_star_star - jnp.sum(v**2, axis=0)

        mu_list.append(mu_batch)
        var_list.append(var_batch)

    return jnp.concatenate(mu_list), jnp.concatenate(var_list)
```

**Key optimizations**:

- Precompute Cholesky once, reuse for all test batches
- Batch kernel evaluations (vectorize if possible with `vmap`)
- Never form $(\mathbf{K}_y)^{-1}$ or full test covariance

For even faster: vectorize batch processing with `vmap` over test batches.

:::{admonition} Advanced tip: Whitening Reparameterization for Optimizer Stability
:class: tip, dropdown

The standard formulation $\boldsymbol{\alpha} = (\mathbf{K}_y)^{-1} \mathbf{y}$ can have poor conditioning during hyperparameter optimization. An alternative used in many production GPs is the **whitening reparameterization**:

Given Cholesky $\mathbf{K}_y = \mathbf{L}\mathbf{L}^T$, instead of optimizing $\mathbf{y}$ directly, work with $\mathbf{v} = (\mathbf{L}^{-1}) \mathbf{y}$. The marginal likelihood becomes:

$$\log p(\mathbf{y} | X, \boldsymbol{\theta}) = -\frac{1}{2}\|\mathbf{v}\|^2 - \sum_i \log L_{ii} - \frac{N}{2}\log(2\pi)$$

**Benefit**: The Hessian is better-conditioned, so optimization converges faster and more reliably. Gradient descent takes fewer iterations and is less sensitive to initialization.

**Implementation**: Store both $\mathbf{L}$ and $\mathbf{v}$, update in place during optimization. This is an optional speedup; for $N < 500$ (your case), the standard approach is fine.

**Reference**: Matthews et al. (2016) "GPflow: A Gaussian Process Library in TensorFlow" for details.

:::

---

3. **Keep shapes static under `jit`**
   - Pre-allocate arrays inside compiled functions
   - Avoid dynamic shapes (e.g., no `n = X.shape[0]` that varies between calls)

4. **Standardize inputs and outputs**
   - Z-score normalize each input dimension: `(X - mean) / std`
   - For outputs spanning orders of magnitude, normalize or log-transform
   - Store normalization constants, denormalize predictions when reporting

5. **Use log-space for positive hyperparameters (with bounds)**
   - Ensures positivity: $\ell = \exp(\tilde{\ell})$, optimize over $\tilde{\ell} \in \mathbb{R}$
   - Gradients often better behaved in log-space
   - Apply to lengthscales, variances, noise
   - **Tip for robustness**: Bound the log-space parameters to avoid extreme values:
     - Lengthscale: $\log \ell \in [-2, 1]$ (roughly $\ell \in [0.13, 2.7] \times$ input range) — prevents learning scales far outside data
     - Signal variance: $\log \sigma_f^2 \in [-3, 2]$ — prevents excessively small or large signal
     - Noise variance: $\log \sigma_n^2 \in [-8, -1]$ — prevents noise from dominating or disappearing
   - Implement bounded optimization via `optax.clip_by_global_norm` or reparameterize to constrained domain (e.g., sigmoid)

6. **Multiple optimizer restarts**
   - Marginal likelihood is non-convex (many local optima)
   - Try 5-10 random initializations, keep best
   - Use heuristic initialization (see below) as starting point

7. **Precision considerations**
   - Enable 64-bit in JAX: Set environment variable `JAX_ENABLE_X64=True` or use `jax.config.update("jax_enable_x64", True)` at start
   - **Default**: Use `float64` for training and hyperparameter optimization
   - **Speed optimization** (optional): After validating $\mathbf{K}_y$ is well-conditioned, test `float32` for predictions
   - **Matérn-5/2 + float32**: Often stable for $N < 1000$ with proper jitter (Matérn kernels numerically better-behaved than SE)
   - **SE kernel + float32**: More prone to numerical issues near singularities (requires higher jitter or float64)
   - Monitor condition number: `cond(K_y) = max(eig)/min(eig)` should be $< 10^{10}$ for float64, $< 10^{6}$ for float32
   - **Your project**: Stick with float64 for $N \sim 250$ (no performance penalty, guaranteed stability)

8. **Batch predictions for memory safety**
   - For $M \gg N$, predict in batches: `M_batch ~ 10^4`
   - Use `vmap` (vectorized map) for vectorization: `jax.vmap(predict_fn)(X_test_batch)`

9. **Deduplicate training inputs**
   - **Problem**: Near-identical training points cause $\mathbf{K}$ to be nearly singular
   - **Check**: Compute pairwise distances: $\|\mathbf{x}_i - \mathbf{x}_j\|$ for all $i \neq j$
   - **Threshold**: Flag duplicates if distance $< 10^{-8}$ (numerical precision limit)
   - **Causes**: Accidentally running same simulation twice, numerical roundoff in parameter sampling
   - **Symptom**: Cholesky fails with "matrix not positive definite"
   - **Fix**: Remove exact duplicates, or add more jitter to diagonal

**💡 Debugging breadcrumb: If Cholesky fails...**

1. **Increase jitter**: Try `1e-5` or `1e-4` (scale-aware formula above)
2. **Deduplicate inputs**: Check for near-identical training points (distance < `1e-8`)
3. **Standardize features**: Z-score normalize all inputs before training
4. **Try Matérn-5/2** instead of SE (less prone to near-singularity)
5. **Check condition number**: `jnp.linalg.cond(K_y)` should be $< 10^{10}$ for float64
:::

### Step 4: Validate the Emulator

**Goal**: Check that GP predictions are accurate and well-calibrated.

**Metrics**:

1. **Predictive accuracy**:
   - RMSE: $\sqrt{\frac{1}{M} \sum_{i=1}^M (y_i - \mu(\mathbf{x}_i))^2}$ (how close are predictions?)
   - $R^2$ coefficient: fraction of variance explained

2. **Uncertainty calibration**:
   - **Coverage**: What fraction of test points fall within predicted 1-$\sigma$ bands?
     - Should be 68% if calibrated (for Gaussian residuals)
     - **Empirical coverage < nominal** (e.g., 60% within 1σ vs expected 68%) → **Overconfident** (error bars too narrow)
     - **Empirical coverage > nominal** (e.g., 80% within 1σ) → **Underconfident** (error bars too wide)
   - **Calibration plot**: Plot predicted std vs actual error, should align

3. **Negative log-likelihood (NLL)** on test set:
   $$\text{NLL} = -\frac{1}{M} \sum_{i=1}^M \log \mathcal{N}\big(y_i \mid \mu(\mathbf{x}_i), \sigma_*^2(\mathbf{x}_i) + \sigma_n^2\big)$$

   **Note**: Use **total predictive variance** $\sigma_*^2(\mathbf{x}_i) + \sigma_n^2$ for observed targets. For latent-function scoring, use epistemic variance $\sigma_*^2(\mathbf{x}_i)$ only.

   - Jointly measures accuracy + calibration
   - Lower is better

**Diagnostics**:

- **Residual plots**: $(y_i - \mu(\mathbf{x}_i))$ vs $\mathbf{x}_i$ (should be random, no patterns)
- **QQ plots** (quantile–quantile plots): Normalized residuals $(y_i - \mu(\mathbf{x}_i)) / \sigma(\mathbf{x}_i)$ should be $\mathcal{N}(0,1)$
- **Learned hyperparameters**: Do lengthscales make physical sense?
- **Out-of-distribution (OOD) detection**: Flag test inputs with low training similarity:
  $$\text{OOD if } \max_i k(\mathbf{x}_*, \mathbf{x}_i) / \sigma_f^2 < \text{threshold}$$

  **Note**: The threshold (e.g., 0.05) should be validated on your data; it depends on learned hyperparameters ($\sigma_f^2$, $\ell$). Cross-validate by checking whether OOD-flagged points have poor predictive accuracy.

  **Interpretation**: Point $\mathbf{x}_*$ has weak correlation with ALL training points → extrapolation!

  **Action items**:

  - Annotate predictions as **low-trust** in plots (different color/marker)
  - Report OOD fraction in test set as quality metric
  - Consider running additional simulations in OOD regions before making scientific claims
  - **Rule**: Never publish conclusions based solely on OOD predictions without validation

  **For your project**: If >10% of test set is OOD, your training data doesn't cover parameter space adequately.

**Cross-validation**:

- Leave-one-out: Train on $N-1$ points, predict left-out point, repeat
- $k$-fold: Split data into $k$ chunks, train on $k-1$, test on 1, repeat
- Checks: Does GP generalize? Or just memorizing training data?

### Step 5: Use the Emulator for Science

**Now the payoff! You have a fast, accurate, uncertainty-aware emulator. Use it for:**

**Application 1: Parameter Space Exploration.**

- Evaluate $\mu(\mathbf{x})$ on dense grid (millions of points, takes seconds)
- Visualize: Which combinations of $(Q, N, a)$ lead to survival vs dissolution?
- Identify interesting regions for further study

**Application 2: Optimization.**

- Find $\mathbf{x}^* = \arg\max_{\mathbf{x}} \mu(\mathbf{x})$ (e.g., maximize cluster lifetime)
- Use gradient-based optimizer (JAX gives gradients!)
- Or: Bayesian optimization (sample where acquisition function is high)

**Application 3: Bayesian Parameter Inference.**

- Observe real cluster (e.g., Pleiades): $(R_{\rm core, obs}, \sigma_{v, obs}, M_{\rm obs})$
- Likelihood: $p(\text{data} | \mathbf{x}) = \mathcal{N}(\text{data} | \mu_{\rm emu}(\mathbf{x}), \sigma^2_{\rm emu}(\mathbf{x}))$
- Run MCMC (you know how!) using emulator for likelihood
- Infer: What initial conditions produced observed cluster?

:::{admonition} Critical: Using Emulator in MCMC Likelihoods
:class: danger

**Common mistake**: Double-counting or forgetting emulator uncertainty in inference!

When using your emulator inside MCMC, the total observational uncertainty is:

$$
\boxed{\Sigma_{\text{total}}(\mathbf{x}) = \Sigma_{\text{obs}} + \sigma_*^2(\mathbf{x})}
$$

where:

- $\Sigma_{\text{obs}}$: Measurement uncertainty from observations (telescope noise, etc.)
- $\sigma_*^2(\mathbf{x})$: Emulator epistemic uncertainty (from GP)

**The likelihood is**:
$$
p(\mathbf{y}_{\text{obs}} | \mathbf{x}) = \mathcal{N}(\mathbf{y}_{\text{obs}} | \mu_{\text{emu}}(\mathbf{x}), \Sigma_{\text{total}}(\mathbf{x}))
$$

**NOT**: $\mathcal{N}(\mathbf{y}_{\text{obs}} | \mu_{\text{emu}}(\mathbf{x}), \Sigma_{\text{obs}})$ ❌ (ignores emulator error!)

**NOT**: $\mathcal{N}(\mathbf{y}_{\text{obs}} | \mu_{\text{emu}}(\mathbf{x}), \sigma_*^2(\mathbf{x}))$ ❌ (ignores observation noise!)

**Why this matters**:

- Ignoring emulator uncertainty → overconfident posteriors (claims too much precision)
- Double-counting → underconfident posteriors (claims too little)
- Correct accounting → honest uncertainty quantification

**In practice**:

```python
# Emulator prediction
mu_emu, var_emu = gp.predict(theta)  # var_emu = sigma_*^2

# Total variance
var_total = var_obs + var_emu  # Add in quadrature

# Likelihood
log_likelihood = -0.5 * (y_obs - mu_emu)**2 / var_total - 0.5 * jnp.log(2 * jnp.pi * var_total)
```

**Connection to Module 5**: This is Bayesian error propagation—combining uncertainties from multiple sources (observations + model).
:::

**Application 4: Experimental Design.**

- GP uncertainty tells you: "I'm uncertain here, run more simulations!"
- Active learning: Iteratively add simulations to maximize information gain
  - **Exploration vs Exploitation**: Greedy variance sampling (below) maximizes exploration by filling uncertain regions, but may miss important physical boundaries. More sophisticated methods (expected improvement, mutual information) balance exploration with exploitation (reducing uncertainty where predictions matter most). For your project, simple variance sampling works well; for research applications, consider hybrid strategies that weight uncertainty by physical relevance.
  - **Greedy variance sampling**: Pick $\mathbf{x}_{\text{new}} = \arg\max_{\mathbf{x}} \sigma_*^2(\mathbf{x})$ (easiest to implement, purely exploratory)
  - **Batch-greedy**: Pick $K$ points by iterating predictive updates or approximating with farthest-point in ARD-scaled space
- Minimize computational cost to achieve target accuracy

:::{admonition} Physics-Aware Target Transformations
:class: warning

**Problem**: GPs are Gaussian, so predictions can be negative or > 1. But physics constrains some outputs:

- Bound fraction $\in [0, 1]$ (can't have 120% of stars bound!)
- Radii, masses, times $> 0$ (can't be negative!)

**Solutions**:

**For bounded outputs** $y \in [0, 1]$ (e.g., bound fraction):

Use **logit transformation**:
$$
g = \text{logit}(y) = \log\left(\frac{y}{1-y}\right), \quad y = \text{sigmoid}(g) = \frac{1}{1 + e^{-g}}
$$

- Train GP on $g \in (-\infty, \infty)$ (unbounded)
- Predictions: $g_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$
- Transform back: $y_* = \text{sigmoid}(g_*)$ (now $\in [0,1]$!)
- Uncertainty: Propagate via Monte Carlo or delta method

**For positive outputs** $y > 0$ (e.g., core radius, time to core collapse):

Use **log transformation**:
$$
g = \log(y), \quad y = \exp(g)
$$

- Train GP on $\log(y)$ (unbounded)
- Predictions: $g_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$
- Transform back for **median** prediction: $y_* = \exp(\mu_*)$ (often reported as point estimate because it's the 50% quantile)
- Transform back for **mean** prediction (log-normal): $\mathbb{E}[y_*] = \exp(\mu_* + \tfrac{1}{2}\sigma_*^2)$ (mathematically correct mean, but heavily skewed toward large values if $\sigma_*^2$ is large)
- **Recommendation**: Report median for interpretability; use mean for Bayesian inference with cost functions symmetric in log-space
- **Important**: Transform-then-average $\neq$ average-then-transform! Report which you use.
- Bonus: Models multiplicative noise naturally

**For multi-output with mixed constraints**:

- Transform each output appropriately before training
- Train separate GPs (or multi-output GP with appropriate likelihoods)

**Why this matters**:

- Prevents unphysical predictions in plots/reports
- Turns multiplicative errors into additive (log) or symmetric errors (logit)
- GP assumptions (Gaussian noise) better match transformed space

**Your N-body emulation**:

- Bound fraction: Use logit transform
- Core radius $R_{\rm core}$: Use log transform
- Velocity dispersion $\sigma_v$: Use log transform
- Number of stars $N$ (if input): Already should be $\log N$ for scaling
:::

:::{admonition} Connection to Frontier Research: DESI Cosmology
:class: note

The Dark Energy Spectroscopic Instrument (DESI) measures Baryon Acoustic Oscillations (BAO) in galaxy clustering to constrain dark energy.

**Problem**: Comparing observations to theory requires:

- Input: Cosmological parameters $(\Omega_m, \Omega_b, h, \sigma_8, n_s, w_0, w_a)$ ($D=7$)
- Output: BAO correlation function $\xi(r)$ at many scales
- Simulator: N-body + hydrodynamics (hours per realization)
- Need: $10^5$-$10^6$ evaluations for MCMC

**Solution**: GP emulator!

- Train on ~1000 simulations (one-time cost)
- Emulate $\xi(r)$ at all scales jointly
- MCMC uses emulator: 10,000× speedup
- **Result**: DESI 2024 achieved percent-level constraints on dark energy equation of state, enabled by GP emulation (numbers evolving with each data release; see latest DESI publications for current status)

**Your project is the same workflow**: Expensive simulation → GP emulator → Bayesian inference. You're learning cutting-edge research methods!
:::

:::{admonition} Conceptual Checkpoint #6: The Workflow
:class: warning

Before Part 2, make sure you understand the full pipeline:

1. **Training data**: Why 250 simulations? Could you do it with 50? With 1000? What changes?

2. **Hyperparameter optimization**: Why maximize marginal likelihood instead of just minimizing prediction error on training set? (Hint: Overfitting!)

3. **Prediction cost**: For $M$ test points, marginal predictions (mean and variance at each point independently) are $O(M N)$ operations after precomputing Cholesky. If you need the full joint covariance for $M$ points, that's $O(M^2 N)$. For $N = 250$ and $M = 10^6$, compute in batches (e.g., $M_{\text{batch}} \sim 10^4$) to stay memory-safe.

4. **Extrapolation danger**: If all training data has $Q \in [0.3, 0.7]$ (all subvirial), should you trust predictions at $Q = 0.9$ (approaching virial equilibrium at $Q=1$)? The GP will extrapolate with increasing uncertainty, but physics may change qualitatively outside the training regime! Only trust OOD predictions if validated with new simulations in that regime. How does GP uncertainty indicate this risk?

5. **Multi-fidelity**: Suppose you have 200 expensive sims ($N = 2000$ particles) and 1000 cheap sims ($N = 500$ particles). Can you use both? (Yes! Multi-fidelity GPs. Advanced topic.)

6. **Failure modes**: When would this workflow fail? What if cluster evolution is chaotic (sensitive to tiny changes in IC)? What if $D = 50$ instead of 3?

Discuss these with your team!
:::

---

## 🎓 What We've Learned: Implementation Summary

### The Complete Workflow

1. **Generate training data** (LHS sampling)
2. **Train GP** (maximize marginal likelihood)
3. **Make predictions** (Gaussian conditioning)
4. **Validate** (coverage, calibration)
5. **Use for science** (exploration, inference, optimization)

### Connection to Everything You've Learned

**Module 1 (Statistics)**:

- GPs extend multivariate Gaussians to infinite dimensions
- Central Limit Theorem: Why Gaussian assumptions are reasonable
- Moments: GP mean and covariance are first two moments

**Module 2-3 (Stellar Systems)**:

- N-body simulations define functions $f(\text{ICs}) = \text{cluster properties}$
- Emulation lets you explore phase space without full dynamics
- Lengthscales relate to characteristic scales (dynamical time, relaxation time)

**Module 4 (Radiative Transfer)**:

- Monte Carlo RT is expensive (like N-body)
- Could emulate emergent spectrum vs stellar parameters
- Same workflow applies!

**Module 5 (Bayesian Inference)**:

- GPs are Bayesian: Prior (kernel) + Likelihood (data) = Posterior (predictions)
- MCMC becomes tractable with emulator replacing expensive likelihood
- Marginal likelihood for hyperparameters = evidence for model selection

**Project 5 (JAX)**:

- Automatic differentiation gives gradients for optimization
- JIT compilation makes GP fast
- Vectorization (`vmap`) enables efficient prediction at many test points
- GPU acceleration makes $N=1000$ feasible

**Next Lecture (Neural Networks)**:

- NNs are alternative emulators (different tradeoffs)
- Can combine: Deep kernel learning, neural processes
- Compare empirically: Which is better for your problem?

### The Philosophical Point

Machine learning is not magic. It's **principled probabilistic inference with mathematical foundations**.

GPs exemplify this:

- Every component has clear interpretation (kernel, hyperparameters, predictions, uncertainty)
- Connections to statistics, physics, computation
- Uncertainty quantification is first-class citizen (not afterthought)
- Hyperparameters have physical meaning (not just numbers to tune)

**When you implement a GP, you're not using a black box—you're applying a century of mathematics (Kolmogorov, Wiener, Rasmussen) to solve a 21st-century astrophysics problem.**

This is computational astrophysics in 2025!

---

## 🔮 Your Final Project

**Next Lecture** (Part 2: Implementation):

- Build GP regression from scratch in JAX
- Apply to N-body emulation (live demo)
- Hyperparameter optimization (gradients via autodiff)
- Validation and diagnostics
- Comparison to neural networks (preview)

**Your Final Project** (Weeks 1-3):

**Week 1: Core Emulation.**

- Fit GP to provided training data
- Predict bound fraction and core radius
- Validate uncertainty calibration
- Deliverable: Working GP emulator

**Week 2: Neural Network Comparison.**

- Build NN emulator (MLP in JAX)
- Compare: Accuracy, uncertainty, computational cost
- Analyze: When is each better?
- Deliverable: Comparative analysis

**Week 3: Extension.**

- Choose one:
  - Time series emulation (predict full trajectories)
  - Active learning (adaptive simulation placement)
  - Multi-fidelity (combine cheap + expensive sims)
  - Physics-informed constraints
- Deliverable: Extended emulator + documentation

**Success Criteria**:

1. Emulator predicts well ($R^2 > 0.9$ on test set)
2. Uncertainty is calibrated (68% coverage at 1-$\sigma$)
3. Can explain: When to use GP vs NN?
4. Code is modular (easy to swap kernels, add data)
5. Documentation is research-grade (hand to collaborator, they can use it)

:::{admonition} Expectations: Glass-Box Philosophy
:class: important

**In Part 2, you will NOT receive**:

- Complete code templates with TODOs
- Step-by-step instructions for every line
- Scaffolding that tells you exactly what to compute

**You WILL receive**:

- Mathematical equations (GP predictive formulas, marginal likelihood)
- Conceptual guidance ("avoid computing $\mathbf{K}^{-1}$ explicitly—why?")
- Physical intuition ("what should lengthscale be? Think about data range")
- Numerical advice ("use Cholesky, not inverse")

**Your job**: Translate math → algorithm → code

This tests whether you truly understand GPs or just memorized formulas. Can you:

- Look at Eq. (2.25) and figure out what to compute?
- Recognize that $\mathbf{K}^{-1} \mathbf{y}$ means "solve $\mathbf{K} \boldsymbol{\alpha} = \mathbf{y}$"?
- Implement optimization loop using `optax` based on conceptual description?
- Debug when Cholesky fails ("matrix not positive definite")?

**This is how research works**: You read papers, understand the math, implement it yourself. No TODOs in Nature papers!

If you can't translate the equations from today's lecture into code, you don't understand GPs deeply enough yet. That's what Part 2 is for—working through it together, building from first principles.
:::

---

## 📚 Preparation for Next Lecture

**Before Part 2, you should**:

1. **Review these notes**: Make sure you understand every conceptual checkpoint

2. **Mathematical preparation**: Can you derive the GP predictive formulas? Try it on paper.

3. **Computational thinking**:
   - How would you implement $k_{\text{SE}}(\mathbf{x}, \mathbf{x}')$ in JAX?
   - How to compute $\mathbf{K}$ for $N$ training points efficiently?
   - What's the shape of all the matrices involved?

4. **Read (optional but recommended)**:
   - Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch. 2.1-2.3
   - MacKay, *Information Theory, Inference, and Learning Algorithms*, Ch. 45

5. **Physical intuition**:
   - For your N-body sims, which parameter do you expect matters most for bound fraction?
   - Sketch what you think the emulator function looks like: $f(Q)$ holding others fixed
   - What should lengthscales be approximately? (Guess based on parameter ranges)

6. **Deliverables checklist** (for your final project report):
   - Report **RMSE**, **$R^2$**, **NLL** (with total variance $\sigma_*^2 + \sigma_n^2$), and **1σ/2σ coverage**
   - Include **one residual plot** and **one QQ plot** (quantile–quantile)
   - Include an **OOD similarity histogram** (flag extrapolation regions)
   - **Discuss ARD lengthscales physically**: After training, list $\ell_Q, \ell_N, \ell_a$ and write one sentence per parameter interpreting physics:
     - Example: "$\ell_Q$ small → bound fraction highly sensitive to virial ratio"
     - Example: "$\ell_N$ large → bound fraction insensitive to particle number for $N > 500$"
     - Example: "$\ell_a$ moderate → cluster scale directly affects tidal stripping timescale"

7. **Questions**: Write down anything confusing. We'll address in Part 2!

---

## 📖 Glossary of Key Terms

**ARD (Automatic Relevance Determination)**: Kernel with per-dimension lengthscales $\ell_d$, allowing the GP to learn which input dimensions matter most. Small $\ell_d$ → high sensitivity to dimension $d$, large $\ell_d$ → low sensitivity (GP effectively ignores that dimension).

**LHS (Latin Hypercube Sampling)**: Stratified sampling method ensuring uniform coverage in each dimension separately. Better than random sampling for space-filling experimental designs. Widely used for training surrogate models.

**OOD (Out-of-Distribution)**: Test points far from training data, where the emulator extrapolates and uncertainty is high. Detect via kernel similarity: $\max_i k(\mathbf{x}_*, \mathbf{x}_i) / \sigma_f^2 < \text{threshold}$ (e.g., 0.05; validate empirically for your problem as threshold depends on learned hyperparameters).

**ICM/LMC (Intrinsic Coregionalization Model / Linear Model of Coregionalization)**: Multi-output GP methods that model correlations between outputs via cross-covariance functions. Beyond scope of this course; see Álvarez et al. (2012) for details.

**Epistemic Uncertainty**: Uncertainty about the latent function $f$ (reducible with more training data). From GP: $\sigma_*^2(\mathbf{x})$. Also called "model uncertainty" or "knowledge uncertainty."

**Aleatoric Uncertainty**: Irreducible noise in observations (measurement error, simulation stochasticity). From likelihood: $\sigma_n^2$. Also called "data uncertainty."

**Marginal Likelihood (Evidence)**: $p(\mathbf{y} | X, \boldsymbol{\theta})$, the probability of observing training data given hyperparameters, integrating over all possible functions. Used for Type-II maximum likelihood hyperparameter optimization. Implements Occam's razor automatically.

**Kernel (Covariance Function)**: $k(\mathbf{x}, \mathbf{x}')$, measures correlation between function values at inputs $\mathbf{x}$ and $\mathbf{x}'$. Encodes smoothness assumptions and prior beliefs about function structure.

**Lengthscale ($\ell$)**: Characteristic distance in input space over which function values remain correlated. Small $\ell$ → rapid variation (wiggly functions), large $\ell$ → smooth variation (slowly varying functions). Units: same as input dimensions.

**Signal Variance ($\sigma_f^2$)**: Prior variance of function values (overall amplitude of variation). Units: output$^2$.

**Noise Variance ($\sigma_n^2$)**: Observation noise level (aleatoric uncertainty). Units: output$^2$.

**Cholesky Factorization**: Decomposition $\mathbf{K}_y = \mathbf{L} \mathbf{L}^T$ where $\mathbf{L}$ is lower-triangular. Numerically stable way to solve linear systems and compute determinants. Essential for GP implementation.

**Jitter**: Small constant added to kernel matrix diagonal to ensure numerical stability: $\mathbf{K}_y = \mathbf{K} + (\sigma_n^2 + \epsilon_{\rm jitter}) \mathbf{I}$. Prevents Cholesky failures from near-singular matrices. Typical values: $\epsilon_{\rm jitter} \sim 10^{-6}$ (scale-aware).

**SE (Squared Exponential) Kernel**: Also called RBF (Radial Basis Function). Assumes infinitely smooth functions. Often too smooth for real physics.

**Matérn Kernel**: Family of kernels with tunable smoothness $\nu$. Matérn-5/2 is recommended default (twice differentiable, realistic for physics).

**PSD (Positive Semi-Definite)**: Property required for valid kernels. Ensures covariance matrices have non-negative eigenvalues (variances can't be negative).

**MCMC (Markov Chain Monte Carlo)**: Sampling method for complex posterior distributions (from Module 5). GPs enable MCMC-based inference by providing fast, differentiable emulators of expensive simulators.

**JAX**: Just-In-Time compiled array computing library for Python with automatic differentiation. Makes GP implementation fast and enables gradient-based hyperparameter optimization.

**JIT (Just-In-Time Compilation)**: Compilation technique that speeds up Python code to near-C performance. Use `@jit` decorator in JAX.

**vmap (Vectorized Map)**: JAX function for automatic vectorization. Maps a function over batch dimension efficiently (avoids Python loops).

---

**Now you're ready to implement!** 🚀

*"The best way to understand a Gaussian Process is to implement one."* — Every computational scientist ever
