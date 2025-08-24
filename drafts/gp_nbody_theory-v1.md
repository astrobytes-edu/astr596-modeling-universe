# Gaussian Processes for N-body Emulation: From First Principles

## Part I: The Fundamental Problem

### Why We Need Emulation

Imagine you've built a beautiful N-body simulation that evolves a star cluster over billions of years. Running this simulation with 1000 stars takes perhaps 30 minutes of compute time. Now suppose you want to understand how the cluster's evolution depends on its initial conditions: the number of stars, their mass distribution, the cluster's concentration, its initial virial ratio. To explore just 10 values for each of 4 parameters would require 10^4 = 10,000 simulations, taking roughly 200 days of continuous computation. This is computationally infeasible.

The key insight is that the mapping from initial conditions to outcomes is likely to be smooth. If a cluster with 500 stars has a relaxation time of 2 Gyr, then a cluster with 501 stars will have a very similar relaxation time. This smoothness suggests we shouldn't need to run all possible simulations—we should be able to predict intermediate values by learning the underlying function that maps inputs to outputs.

This is where Gaussian Processes enter: they provide a principled, probabilistic way to learn this function from a limited number of simulation runs.

## Part II: What is a Gaussian Process?

### The Core Definition

A Gaussian Process is a collection of random variables, any finite subset of which follows a multivariate Gaussian distribution. More intuitively, a GP is a probability distribution over functions. Instead of having a probability distribution over numbers (like a normal distribution) or over vectors (like a multivariate normal), a GP gives us a probability distribution over entire functions.

To understand this deeply, let's build up from familiar concepts.

### From Gaussian to Multivariate Gaussian

You're familiar with a Gaussian distribution for a single variable:
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

This describes our uncertainty about a single number. For multiple correlated variables, we have the multivariate Gaussian:
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Here, $\boldsymbol{\mu}$ is a vector of means and $\Sigma$ is the covariance matrix that captures how the variables relate to each other. The key insight is that $\Sigma_{ij} = \text{Cov}(x_i, x_j)$ tells us how knowing $x_i$ informs us about $x_j$.

### From Finite to Infinite: The Function View

Now for the conceptual leap: imagine we want to describe a function $f(x)$ probabilistically. We could discretize the function, evaluating it at points $x_1, x_2, ..., x_n$, giving us values $f_1, f_2, ..., f_n$. If these values follow a multivariate Gaussian distribution, we can write:
$$[f_1, f_2, ..., f_n]^T \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$$

where $\mathbf{K}$ is a covariance matrix. As we let $n \to \infty$, considering the function at infinitely many points, we arrive at a Gaussian Process. The GP is completely specified by:
- A mean function: $m(x) = \mathbb{E}[f(x)]$
- A covariance function (kernel): $k(x, x') = \text{Cov}(f(x), f(x'))$

We write this as:
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

## Part III: The Kernel - Heart of the Gaussian Process

### What the Kernel Represents

The kernel function $k(x, x')$ is the crucial component that encodes our assumptions about the function we're learning. It defines the covariance between function values at different inputs. If $k(x, x')$ is large, then when $f(x)$ is above its mean, $f(x')$ is likely to be above its mean too.

### Properties of Valid Kernels

Not every function can be a kernel. Valid kernels must produce positive semi-definite covariance matrices for any set of inputs. This ensures the resulting distribution is a valid probability distribution. The most common kernel for smooth functions is the Radial Basis Function (RBF) or Squared Exponential kernel:

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

Here:
- $\sigma_f^2$ is the signal variance (how much the function varies)
- $\ell$ is the length scale (how quickly the function changes)

This kernel encodes the assumption that nearby inputs will have similar outputs, with the similarity decaying smoothly with distance.

### Why the Kernel Matters for N-body Emulation

For N-body simulations, the kernel encodes our physical intuition. We expect that:
- Similar initial conditions produce similar evolutionary outcomes (smoothness)
- The effect of changing one parameter might depend on others (interaction)
- Some parameters might matter more than others (different length scales)

We might use an Automatic Relevance Determination (ARD) kernel:
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\sum_{d=1}^D \frac{(x_d - x'_d)^2}{2\ell_d^2}\right)$$

This assigns different length scales $\ell_d$ to each input dimension, automatically learning which parameters most strongly influence the simulation outcomes.

## Part IV: Gaussian Process Regression

### The Setup

Suppose we've run our N-body simulation $n$ times with different initial conditions. We have:
- Training inputs: $\mathbf{X} = [\mathbf{x}_1, ..., \mathbf{x}_n]^T$ (each $\mathbf{x}_i$ contains initial conditions like mass, concentration, etc.)
- Training outputs: $\mathbf{y} = [y_1, ..., y_n]^T$ (each $y_i$ is an outcome like relaxation time)

We want to predict the output $y_*$ at a new input $\mathbf{x}_*$ without running the simulation.

### The Prior Distribution

Before seeing any data, our GP prior says that function values at any collection of points follow a multivariate Gaussian:
$$\begin{bmatrix} \mathbf{y} \\ y_* \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mathbf{m} \\ m_* \end{bmatrix}, \begin{bmatrix} \mathbf{K} & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix}\right)$$

where:
- $\mathbf{K}$ is the $n \times n$ covariance matrix with $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
- $\mathbf{k}_* = [k(\mathbf{x}_1, \mathbf{x}_*), ..., k(\mathbf{x}_n, \mathbf{x}_*)]^T$
- $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$

### Conditioning on Observations

The magic of GPs is that conditioning a multivariate Gaussian on observed values gives another Gaussian. After observing $\mathbf{y}$, the posterior distribution for $y_*$ is:

$$p(y_* | \mathbf{y}, \mathbf{X}, \mathbf{x}_*) = \mathcal{N}(\mu_*, \sigma_*^2)$$

with:
$$\mu_* = m_* + \mathbf{k}_*^T \mathbf{K}^{-1}(\mathbf{y} - \mathbf{m})$$
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T \mathbf{K}^{-1} \mathbf{k}_*$$

These equations have beautiful interpretations:
- The mean prediction $\mu_*$ is the prior mean plus a linear combination of the observed deviations from their prior means, weighted by the covariances
- The variance $\sigma_*^2$ is the prior variance minus the reduction in uncertainty from the observations

### Including Observation Noise

Real simulations have numerical errors, so we model:
$$y_i = f(\mathbf{x}_i) + \epsilon_i$$
where $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$

This modifies our covariance matrix:
$$\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$$

The noise term prevents overfitting and ensures numerical stability by making the covariance matrix better conditioned.

## Part V: Why GPs are Perfect for N-body Emulation

### Quantified Uncertainty

Unlike other interpolation methods, GPs provide not just predictions but also uncertainty estimates. When predicting the relaxation time for untested initial conditions, we get both the expected value and our confidence in that prediction. This is crucial for understanding where we might need to run additional simulations.

### Optimal Interpolation

GPs provide the Best Linear Unbiased Predictor (BLUP) under the assumption that the function follows a Gaussian process with the specified kernel. This means that given our smoothness assumptions, the GP gives the optimal interpolation in a precise mathematical sense.

### Automatic Complexity Control

Through the marginal likelihood:
$$p(\mathbf{y}|\mathbf{X}) = \int p(\mathbf{y}|f, \mathbf{X})p(f)df$$

we can optimize hyperparameters (length scales, signal variance) by maximizing:
$$\log p(\mathbf{y}|\mathbf{X}) = -\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{n}{2}\log(2\pi)$$

This automatically balances model complexity with data fit, implementing Occam's razor.

## Part VI: The N-body Emulation Pipeline

### Step 1: Choosing the Parameter Space

For N-body simulations, we identify key initial conditions that determine evolution:
- Number of stars ($N$)
- Total mass ($M$)
- Concentration parameter ($c$)
- Initial virial ratio ($Q$)
- Mass segregation parameter ($S$)

Each simulation run provides a mapping from $\mathbf{x} = [N, M, c, Q, S]$ to outcomes like $y = T_{\text{relax}}$ (relaxation time).

### Step 2: Design of Experiments

We need to choose which simulations to run. Latin Hypercube Sampling ensures good coverage of the parameter space with minimal runs. For a 5-dimensional parameter space, perhaps 50-100 simulations suffice to build an accurate emulator.

### Step 3: Kernel Design

For N-body emulation, we might use:
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\sum_{d=1}^5 \frac{(x_d - x'_d)^2}{2\ell_d^2}\right) + \sigma_n^2\delta(\mathbf{x}, \mathbf{x}')$$

The ARD structure lets us learn that, perhaps, the number of stars matters more than mass segregation for relaxation time.

### Step 4: Training the Emulator

Given our simulation results $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, we:
1. Standardize inputs and outputs (zero mean, unit variance)
2. Optimize hyperparameters $\theta = [\sigma_f, \ell_1, ..., \ell_5, \sigma_n]$ by maximizing marginal likelihood
3. Compute and store the inverse covariance matrix $\mathbf{K}_y^{-1}$

### Step 5: Making Predictions

For any new initial conditions $\mathbf{x}_*$:
1. Compute covariances with training points: $\mathbf{k}_*$
2. Predict mean: $\mu_* = \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{y}$
3. Predict uncertainty: $\sigma_*^2 = k_{**} - \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{k}_*$

The emulator now replaces expensive simulations with microsecond predictions!

## Part VII: Multi-Output Extension

### The Challenge

N-body simulations produce multiple outputs: relaxation time, core collapse time, escape fraction, binary fraction. We could build separate GPs for each, but they're likely correlated—clusters that relax quickly might also lose stars quickly.

### Linear Model of Coregionalization

We model multiple outputs as:
$$\mathbf{f}(\mathbf{x}) = \mathbf{B} \mathbf{g}(\mathbf{x})$$

where $\mathbf{g}(\mathbf{x})$ are independent GPs and $\mathbf{B}$ is a mixing matrix. This captures output correlations while maintaining computational tractability.

## Part VIII: Active Learning

### Where to Simulate Next?

The GP's uncertainty estimates guide where to run new simulations. We might choose the point with:
- Maximum uncertainty (exploration)
- Maximum expected improvement (exploitation)
- Maximum information gain (optimal design)

This creates an iterative loop:
1. Train GP on current simulations
2. Find point of maximum uncertainty
3. Run simulation at that point
4. Add to training data and repeat

## Part IX: Why This Works

### The Smoothness Prior

Physical systems are generally smooth—small changes in initial conditions lead to small changes in outcomes. GPs encode this through the kernel, making them natural for physical emulation.

### The Bayesian Framework

By treating the simulation as an unknown function and using Bayesian inference, we get principled uncertainty quantification. We know not just our best guess but how confident to be.

### The Data Efficiency

GPs extract maximum information from limited data. With just 50-100 simulations, we can accurately predict outcomes across a continuous parameter space that would require millions of runs to explore exhaustively.

## Part X: Deeper Insights

### Connection to Kriging

GPs are equivalent to Kriging in geostatistics. The same mathematics that predicts gold deposits underground predicts star cluster evolution!

### Connection to Reproducing Kernel Hilbert Spaces

Every kernel defines a Hilbert space of functions. The GP finds the minimum-norm interpolant in this space—the "simplest" function consistent with the data.

### The Representer Theorem

The GP prediction can be written as:
$$f(\mathbf{x}_*) = \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \mathbf{x}_*)$$

The predicted function is a weighted sum of kernels centered at training points—we're essentially learning the "influence" of each training simulation.

## Conclusion: The Power of Probabilistic Emulation

By viewing N-body simulation as sampling from an unknown function and using Gaussian Processes to learn this function, we transform an intractable computational problem into a tractable statistical one. The students' realization that their month-long parameter study can be completed in minutes isn't just a computational trick—it represents a fundamental shift in how we think about simulation-based science. Instead of treating each simulation as isolated, we recognize them as glimpses of an underlying functional relationship that we can learn and exploit.

The beauty is that this same framework applies whether we're emulating N-body dynamics, radiative transfer, or any other expensive simulation. The Gaussian Process provides a universal tool for turning computational physics into statistical learning, opening doors to parameter studies, optimization, and inverse problems that would otherwise remain computationally forbidden.