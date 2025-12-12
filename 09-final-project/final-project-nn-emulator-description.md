# Final Project: From Simulation to Surrogate

**ASTR 596: Modeling the Universe**  
**Instructor:** Dr. Anna Rosen  
**Due:** Thursday, December 18, 2025, 11:59 PM  
**AI Policy:** Phase 3 (Professional integration — use AI tools as a working scientist would, but all code must be your own)

:::{admonition} No Late Submissions
:class: warning
Final projects cannot be accepted after the deadline. This is a hard cutoff due to grade submission timelines.
:::

---

## TL;DR

**What you're building**: A neural network that predicts N-body simulation outcomes, then using it for Bayesian inference.

**The pipeline**:

1. Run 100 N-body simulations with varied initial conditions $(Q_0, a)$
2. Train a neural network to predict summary statistics $(f_{\rm bound}, \sigma_v, r_h)$
3. Use the fast emulator inside `NumPyro` to infer what initial conditions produced a given outcome

**Packages you'll learn**: Equinox (NNs), Optax (optimization), NumPyro (probabilistic programming)

**Time**: ~2.5 weeks | **Deliverables**: Code package + research memo
*Note*: You do not need to submit a growth memo for this final project. Instead, you will submit a growth synthesis reflection separately.

---

## Table of Contents

1. [The Big Picture](#the-big-picture) — What and why
2. [Connection to Your Learning Journey](#connection-to-your-learning-journey) — How this synthesizes the course
3. [The JAX Ecosystem](#the-jax-ecosystem) — Packages you'll use
4. [Part 1: Generate Training Data](#part-1-generate-training-data) — N-body simulations and summary statistics
5. [Part 2: Neural Network Emulator](#part-2-neural-network-emulator) — Build and train the surrogate model
6. [Part 3: Evaluate Your Emulator](#part-3-evaluate-your-emulator) — Accuracy, uncertainty, edge behavior
7. [Part 4: Inference with NumPyro](#part-4-inference-with-numpyro) — Bayesian parameter recovery
8. [Part 5: Package Structure](#part-5-package-structure) — Code organization
9. [Part 6: Deliverables](#part-6-deliverables) — What to submit
10. [Rubric](#rubric) — How you'll be graded
11. [Tips](#tips) — Debugging guidance

---

## The Big Picture

This project brings your semester full circle. You've built simulations from scratch, inference engines from scratch, and transformed your N-body code into a professional JAX package. Now you'll tackle one of the most powerful techniques in modern computational science: **surrogate modeling**. 

**The Problem**: Your N-body simulations are expensive. Each run takes seconds to minutes. But scientific inference requires *thousands* of model evaluations. What if you could train a fast approximation that captures the essential physics?

**The Solution**: Build an *emulator* — a neural network trained on simulation outputs that predicts results for new parameters in milliseconds instead of minutes. Then use that emulator for Bayesian inference with NumPyro. The pipeline closes a loop using tools you've built all semester: your N-body package (Project 5) generates training data, your neural network learns to approximate the simulation, and NumPyro's NUTS sampler (the production version of the HMC you built in Project 4) explores the posterior over initial conditions.

See these slides for an overview of the workflow: [*Neurons to Emulators* (NotebookLM-made slides)](./Neurons_to_Emulators_NotebookLM_slides.pdf)

### The Scientific Question

Given the final state of a star cluster — its bound mass fraction, velocity dispersion, spatial extent — can we infer what initial conditions produced it? This is an *inverse problem*, the same class as inferring cosmological parameters from supernovae (Project 4), recovering dark matter distributions from stellar kinematics, or constraining protoplanetary disk properties from exoplanet populations.

---

## Connection to Your Learning Journey

This project synthesizes everything:

**Module 1 — Statistical Foundations:**

- Training data as samples from parameter space
- Summary statistics compress high-dimensional simulation outputs into meaningful numbers

**Module 3 — Stellar Dynamics:**

- The virial theorem: $2K + W = 0$ at equilibrium (where $W < 0$ is gravitational potential energy and $K$ is kinetic energy).
- Virial ratio $Q = 2K/|W|$ characterizes dynamical state.
- Relaxation, mass segregation, and cluster evolution.

**Module 5 — Bayesian Inference:**

- Inverse problems: observations → parameters
- You built MCMC/HMC from scratch; now you'll use NumPyro's production implementation

**Module 6 — JAX:**

- Automatic differentiation enables gradient-based optimization
- JIT compilation makes emulator evaluation fast enough for inference

**Module 7 — Machine Learning:**

- Neural networks as universal function approximators
- Training via gradient descent with Optax

**Connection to Projects 2 & 5:**

- Your N-body code generates the ground truth
- Your JAX package provides the infrastructure

---

## The JAX Ecosystem

You'll learn three professional JAX packages:

| Package | Purpose | Documentation |
|---------|---------|---------------|
| **Equinox** | Neural networks as PyTrees | <https://docs.kidger.site/equinox/> |
| **Optax** | Gradient-based optimization | <https://optax.readthedocs.io/> |
| **NumPyro** | Probabilistic programming | <https://num.pyro.ai/> |

All three follow JAX conventions: pure functions, explicit state, composable transformations. **Read the "Getting Started" tutorials for each.** Learning new tools from documentation is a core professional skill.

---

## Part 1: Generate Training Data

### 1.1 The Emulation Task

Train a neural network to predict **summary statistics** of N-body simulations as a function of **initial condition parameters**.

**Input parameters** (what you vary):

| Parameter | Symbol | Definition | Range |
|-----------|--------|------------|-------|
| Initial virial ratio | $Q_0$ | $Q = \tfrac{2K}{|W|}$ | 0.5 – 1.5 |
| Plummer scale radius | $a$ | Sets initial concentration | 50 – 200 AU |

The virial ratio characterizes the dynamical state:

- $Q < 1$: Subvirial ("cold")—system will collapse
- $Q = 1$: Virial equilibrium
- $Q > 1$: Supervirial ("hot")—system will expand and dissolve

By spanning $Q \in [0.5, 1.5]$, you explore systems that collapse violently, settle to equilibrium, and fly apart. This produces interesting variation in the outputs your emulator must learn.

**Output predictions** (what you emulate):

| Statistic | Symbol | Definition |
|-----------|--------|------------|
| Bound mass fraction | $f_{\rm bound}$ | Mass in bound particles / total mass |
| Velocity dispersion | $\sigma_v$ | Mass-weighted RMS speed in COM frame |
| Half-mass radius | $r_h$ | Radius enclosing half the bound mass |

:::{admonition} Why These Quantities?
:class: note

These are fundamental diagnostics of stellar dynamics:

- $f_{\rm bound}$ measures cluster survival vs. dissolution
- $\sigma_v$ connects to total energy via the virial theorem
- $r_h$ characterizes spatial structure and concentration

Together they encode the dynamical state. An emulator that predicts these accurately has learned real physics, not just interpolation.
:::

### 1.2 Fixed Simulation Parameters

Hold these constant across all training runs:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $N$ | 200 | Large enough for dynamics, fast enough to generate many runs |
| IMF | Kroupa | Realistic mass spectrum enables mass segregation, preferential ejection |
| Integrator | Leapfrog | Symplectic (your Project 2/5 code) |
| Timestep | Constant $\Delta t$ | Required for symplectic structure (see below) |
| Softening | $\epsilon \sim 0.1 \times a / N^{1/3}$ | ~10% of mean interparticle spacing |

A reasonable starting point is $\epsilon \sim 0.1 \times a / N^{1/3}$, i.e., roughly 10% of the mean interparticle spacing. Use your Project 2 experiments to choose a value that avoids close-encounter divergences without over-smoothing the dynamics.

**Simulation Duration**

Run each simulation long enough for the system to dynamically evolve but not so long that every supervirial system has completely dissolved. A good choice is **10 crossing times**.

The **crossing time** is the characteristic timescale for a star to traverse the cluster:

$$t_{\rm cross} = \frac{R}{\sigma_v} \sim \frac{a}{\sigma_v}$$

where $R$ is a characteristic radius (use your Plummer scale radius $a$) and $\sigma_v$ is the velocity dispersion. For a virialized Plummer sphere, $\sigma_v \sim \sqrt{GM/a}$, giving:

$$t_{\rm cross} \sim \sqrt{\frac{a^3}{GM}}$$

This is essentially the dynamical time: the timescale over which gravitational dynamics operate. The scaling $t_{\rm cross} \propto a^{3/2} M^{-1/2}$ is what matters; compute it from your initial conditions.

After $\sim 10 \, t_{\rm cross}$:

- Subvirial systems ($Q < 1$) have collapsed and re-virialized
- Systems near equilibrium ($Q \approx 1$) remain bound with modest evolution
- Supervirial systems ($Q > 1$) have expanded, with unbound stars escaping

You can estimate $t_{\rm cross}$ from your initial conditions and use $t_{\rm final} = 10 \times t_{\rm cross}$. Alternatively, pick a fixed duration in years that spans this range for your parameter space—just be consistent across all runs.

:::{admonition} Why Constant Timestep?
:class: important

Symplectic integrators like Leapfrog have a special property: they exactly conserve a *modified* Hamiltonian $\tilde{H} = H + \mathcal{O}(\Delta t^2)$. This is why your Project 2 simulations showed bounded energy oscillation rather than secular drift.

**But this only works if $\Delta t$ is constant.** Adaptive timestepping breaks the symplectic structure, and you lose the energy conservation guarantee. For this project, use constant $\Delta t$ throughout each simulation.
:::

:::{admonition} Choosing Your Timestep
:class: note

**Starting point:** $\Delta t \sim 0.01 \times t_{\rm cross}$

**Goal:** Qualitatively correct dynamics with **bounded** energy error.

Run a single test simulation (e.g., $Q_0 = 1.0$, $a = 100$ AU) and check how the total energy behaves over the full integration (10 crossing times). For a well-chosen timestep with Leapfrog you should see **oscillatory** energy error (no secular drift) with amplitude

$$\max_t \left|\frac{E(t) - E_0}{E_0}\right| \;\lesssim\; 10^{-4}$$

If you see secular drift or much larger oscillations, reduce $\Delta t$ by a factor of 2 and rerun. Repeat until the error is $\sim 10^{-4}$ and oscillatory.

**Do not** chase machine-precision energy conservation. For this project, dynamics that are physically sensible and numerically stable at the $\sim 10^{-4}$ level are sufficient. Once you find a good $\Delta t$, freeze it and reuse for your full training set.
:::

### 1.3 Sampling Parameter Space

When training an emulator, the design of your training set matters. You need samples that **cover the parameter space efficiently**—not clustered in one region, not leaving gaps where the emulator must extrapolate blindly.

**Why not uniform random sampling?**

Random samples from a uniform distribution tend to clump and leave holes, especially in higher dimensions. With only 100 samples in 2D, random sampling might accidentally cluster points in one corner, leaving another corner with no training data. Your emulator would perform poorly there.

**Latin Hypercube Sampling (LHS)** solves this by ensuring *stratified* coverage. The idea:

1. Divide each parameter's range into $N$ equal bins (where $N$ is your sample size)
2. Place exactly one sample in each bin along each dimension
3. Randomly shuffle which bins pair together across dimensions

The result: every "row" and "column" of parameter space contains exactly one sample. No clumping, no gaps, guaranteed coverage.

For 2D with $N = 100$ samples:

- Divide $Q_0 \in [0.5, 1.5]$ into 100 bins of width 0.01
- Divide $a \in [50, 200]$ into 100 bins of width 1.5 AU
- Each bin in $Q_0$ gets exactly one sample; same for $a$
- The pairing is randomized to avoid diagonal patterns

LHS is standard practice for computer experiments and surrogate modeling. It provides better space coverage than random sampling with the same number of points—crucial when each simulation is expensive.

**Implementation**: `scipy.stats.qmc.LatinHypercube`

**Dataset sizes**:

- Training set: 100 simulations
- Test set: 20 simulations (held out — never used in training)

:::{admonition} Practical Tip
:class: tip
Start by generating a smaller training set (e.g., 30–40 simulations) to debug your pipeline end-to-end. Once everything works—summary stats look right, NN trains, inference runs—scale up to 80–100 simulations for your final results.
:::

### 1.4 Computing Summary Statistics

Extract summary statistics from the final state of each simulation. A particle $i$ is **gravitationally bound** if its total specific energy is negative:

$$\varepsilon_i = \frac{1}{2}|\mathbf{v}_i - \mathbf{v}_{\rm COM}|^2 + \phi(\mathbf{r}_i) < 0$$

where $\mathbf{v}_{\rm COM}$ is the center-of-mass velocity of the bound particles, and $\phi(\mathbf{r}_i)$ is the gravitational potential at particle $i$'s position due to all other particles (use your softened potential).

**Bound mass fraction**:
$$f_{\rm bound} = \frac{\sum_{i \in \text{bound}} m_i}{\sum_i m_i}$$

**Velocity dispersion** (mass-weighted RMS speed in the COM frame, bound particles only):
$$\sigma_v^2 = \frac{\sum_{i \in \text{bound}} m_i \, |\mathbf{v}_i - \mathbf{v}_{\rm COM}|^2}{\sum_{i \in \text{bound}} m_i}$$

*Note:* First compute $\mathbf{v}_{\rm COM}$ for the bound population, then compute $\sigma_v$ using velocities relative to that COM. This removes bulk motion and isolates the internal velocity dispersion.

**Half-mass radius**: Sort bound particles by distance from the center of mass. Find the radius $r_h$ enclosing half of the *bound* mass (i.e., the radius within which the cumulative bound mass equals $\frac{1}{2}\sum_{i \in \text{bound}} m_i$).

**Expected output ranges** (approximate, for your parameter space):

| Output | Typical Range | Units |
|--------|---------------|-------|
| $f_{\rm bound}$ | 0.3 – 1.0 | dimensionless |
| $\sigma_v$ | 0.5 – 5 | AU/yr |
| $r_h$ | 20 – 150 | AU |

Use these to sanity-check your summary statistics. Values far outside these ranges may indicate a bug or an extreme outcome; inspect outliers carefully.

:::{admonition} Why Mass-Weighted?
:class: note

With a Kroupa IMF, your cluster spans a wide mass range. Mass-weighted statistics track where the *mass* is, not just particle counts. This is more physical and connects to real observables (luminosity-weighted quantities in observed clusters trace massive stars).
:::

:::{admonition} Handling Stochasticity
:class: warning

With N=200 and a Kroupa IMF, each simulation has a *different mass realization*—even with identical $(Q_0, a)$, the randomly-drawn stellar masses vary. This introduces scatter in your outputs that isn't due to $(Q_0, a)$.

**Your options:**

1. **Accept it as noise** (recommended): The emulator learns the *mean* relationship, and the scatter contributes to irreducible uncertainty. This is realistic—real clusters have stochastic IMF sampling too.

2. **Fix the random seed for mass sampling**: Every simulation uses the same 200 masses, only IC parameters vary. Cleaner for emulation, but less physical.

3. **Run multiple realizations per $(Q_0, a)$**: Average the outputs. More expensive, diminishing returns.

For this project, option 1 is fine. Your ensemble uncertainty will naturally capture some of this scatter.
:::

:::{admonition} Checkpoint
:class: important

Before proceeding:
- [ ] Training and test datasets saved to disk
- [ ] Summary statistics tested on a known configuration
- [ ] Diagnostic plots: $Q_0$ vs $f_{\rm bound}$ shows expected trend (higher $Q_0$ → lower $f_{\rm bound}$)
:::

---

## Part 2: Neural Network Emulator

### 2.1 What Is an Emulator?

An **emulator** (or **surrogate model**) is a fast approximation to an expensive computation. Your N-body simulation defines a function:

$$\mathbf{y} = f_{\rm sim}(\boldsymbol{\theta})$$

where $\boldsymbol{\theta} = (Q_0, a)$ are initial conditions and $\mathbf{y} = (f_{\rm bound}, \sigma_v, r_h)$ are outputs. Each evaluation of $f_{\rm sim}$ requires running a full simulation—seconds to minutes of compute.

An emulator $\hat{f}$ learns to approximate this mapping:

$$\hat{f}(\boldsymbol{\theta}) \approx f_{\rm sim}(\boldsymbol{\theta})$$

Once trained, $\hat{f}$ evaluates in microseconds. This speedup—often $10^4$ to $10^6\times$—is what makes Bayesian inference tractable. MCMC needs thousands of likelihood evaluations; with an emulator, that's seconds instead of days.

Emulators are ubiquitous in modern computational science: cosmologists emulate power spectra, climate scientists emulate Earth system models, engineers emulate CFD simulations. The workflow you're learning transfers directly.

### 2.2 Why Neural Networks?

Neural networks are **universal function approximators**: given enough parameters, they can approximate any continuous function to arbitrary accuracy. This is the Universal Approximation Theorem, and it's why NNs work for such diverse problems.

For emulation, NNs offer practical advantages:

- **Flexibility**: No need to specify functional form—the network learns it from data
- **Scalability**: Training cost scales linearly with dataset size (unlike Gaussian Processes, which scale as $N^3$)
- **Speed**: Inference is just matrix multiplications—trivially parallelizable on GPUs

The tradeoff: NNs are interpolators with no built-in uncertainty quantification. They can fail silently when extrapolating. We'll address this with ensembles.

### 2.3 The Multi-Layer Perceptron

The simplest neural network architecture is the **multi-layer perceptron (MLP)**—layers of linear transformations interleaved with nonlinear activations.

For input $\mathbf{x} \in \mathbb{R}^{d_{\rm in}}$, a single hidden layer computes:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$

where $\mathbf{W}_1$ is a weight matrix, $\mathbf{b}_1$ is a bias vector, and $\sigma$ is a nonlinear activation function. The output layer then computes:

$$\hat{\mathbf{y}} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

Stacking multiple hidden layers creates a "deep" network capable of learning hierarchical representations.

**Activation functions** introduce nonlinearity—without them, stacked linear layers would collapse to a single linear transformation. The **ReLU** (Rectified Linear Unit) is the standard choice:

$$\text{ReLU}(x) = \max(0, x)$$

It's simple, computationally cheap, and works well in practice.

### 2.4 Training as Optimization

Training a neural network means finding weights $\mathbf{W}, \mathbf{b}$ that minimize a **loss function** measuring prediction error. For regression, the standard choice is mean squared error:

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{y}}_i - \mathbf{y}_i^{\rm true}\|^2$$

This is an optimization problem—find the parameters that minimize $\mathcal{L}$. Sound familiar? It's the same structure as finding the maximum of a posterior (Project 4), just minimizing loss instead of maximizing probability.

**Gradient descent** iteratively updates parameters in the direction that reduces loss:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \, \nabla_{\mathbf{W}} \mathcal{L}$$

where $\eta$ is the **learning rate**. The gradient $\nabla_{\mathbf{W}} \mathcal{L}$ tells us how to adjust each weight to reduce error. JAX computes these gradients automatically via automatic differentiation—the same machinery that powered your HMC gradients.

**Adam** is a popular optimizer that adapts the learning rate for each parameter based on gradient history. It's the default choice for neural network training and is available in Optax.

### 2.5 Your Architecture

Build an MLP using Equinox:

- **Input**: 2 normalized features $(Q_0, a)$
- **Hidden layers**: 2 layers, 64 units each, ReLU activations
- **Output**: 3 predictions $(f_{\rm bound}, \sigma_v, r_h)$, normalized

This architecture is likely overkill for a 2D → 3D mapping, but it gives you room to experiment. Start here; you can simplify if training is fast and accurate.

**Resource**: Equinox documentation, "Getting Started" → "Your first neural network"

### 2.6 Why Normalize?

Neural networks train poorly when input features have vastly different scales. If $Q_0 \in [0.5, 1.5]$ but $a \in [50, 200]$, the network's gradients will be dominated by $a$, and learning will be slow and unstable.

**Standardization** transforms each feature to zero mean and unit variance:

$$\tilde{x} = \frac{x - \mu_x}{\sigma_x}$$

where $\mu_x$ and $\sigma_x$ are the mean and standard deviation computed from your **training set only**.

Apply the same transformation (using training set statistics) to:
- Training inputs and outputs
- Test inputs and outputs  
- Any new predictions

To recover physical units from normalized predictions:

$$x = \tilde{x} \cdot \sigma_x + \mu_x$$

### 2.7 Training in Practice

Use Optax for gradient-based optimization. Adam is a reasonable default optimizer.

**Track**: Loss vs. epoch—should decrease and plateau.

**Hyperparameters**:
- Learning rate: start with $10^{-3}$
- Epochs: 500–1000 typical
- Batch size: full-batch is fine for ~100 samples

**Convergence criteria**: Training is "converged" when:
1. Loss has plateaued (not decreasing meaningfully over ~100 epochs)
2. Loss is small relative to data variance—aim for MSE $\lesssim 0.01$ on normalized data
3. No NaN or exploding values

If loss plateaus at a high value, try: lower learning rate, different initialization, more hidden units, or check your data for issues.

### 2.8 Uncertainty Estimation

A single neural network provides point estimates with no uncertainty quantification. For scientific applications, we need uncertainty.

**Ensemble method**: Train $M$ networks (3–5 is sufficient) with different random initializations. For any input $\mathbf{x}$:

$$\hat{y}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} f_m(\mathbf{x}) \quad \text{(mean prediction)}$$

$$\sigma_y(\mathbf{x}) = \sqrt{\frac{1}{M} \sum_{m=1}^{M} \left(f_m(\mathbf{x}) - \hat{y}(\mathbf{x})\right)^2} \quad \text{(uncertainty)}$$

The spread across ensemble members estimates **epistemic uncertainty**—uncertainty due to limited training data.

### 2.9 Code Skeleton

These are templates, not prescriptions—you may adapt function names and internal structure, as long as your package provides equivalent functionality.

Your `emulator.py` should contain something like:

```python
import jax.numpy as jnp
import equinox as eqx
import optax

class NNEmulator(eqx.Module):
    # TODO: Define layers as attributes
    # Hint: eqx.nn.Linear, eqx.nn.MLP, or build your own
    
    def __init__(self, key):
        # TODO: Initialize layers
        pass
    
    def __call__(self, x):
        # TODO: Forward pass
        # x: (2,) array of [Q0, a] (normalized)
        # returns: (3,) array of [f_bound, sigma_v, r_h] (normalized)
        pass

def mse_loss(model, x, y):
    # TODO: Compute mean squared error
    pass

def train_step(model, opt_state, optimizer, x, y):
    # TODO: Single gradient descent step
    # Hint: use eqx.filter_grad
    pass

def train_ensemble(key, x_train, y_train, n_models=5):
    # TODO: Train multiple models with different initializations
    pass

def predict_ensemble(models, x):
    # TODO: Return mean and std across ensemble
    pass
```

Study the Equinox documentation—the "Getting Started" and "Train an MLP" examples show exactly this pattern.

:::{admonition} Checkpoint
:class: important

Before proceeding:
- [ ] Training loss converges
- [ ] Test set predictions reasonable (see Part 3)
- [ ] Ensemble produces meaningful spread
:::

---

## Part 3: Evaluate Your Emulator

### 3.1 Accuracy Metrics

On your held-out test set, compute for each output:

**Mean Absolute Error**:
$$\text{MAE} = \frac{1}{N_{\rm test}} \sum_{i=1}^{N_{\rm test}} |y_i^{\rm true} - y_i^{\rm pred}|$$

**Root Mean Square Error**:
$$\text{RMSE} = \sqrt{\frac{1}{N_{\rm test}} \sum_{i=1}^{N_{\rm test}} (y_i^{\rm true} - y_i^{\rm pred})^2}$$

### 3.2 Predicted vs. True

Scatter plots of predicted vs. true values. Points should cluster along $y = x$. Systematic deviations indicate model bias.

### 3.3 Uncertainty Behavior

Plot predictions along a 1D slice: fix $a = 125$ AU, vary $Q_0$ from 0.5 to 1.5. Show ensemble mean ± 2σ.

Does uncertainty increase where training data is sparse?

### 3.4 Behavior at Edges

Neural networks are interpolators—they work best *within* the training distribution. Examine how your emulator behaves near the edges of your parameter space:

1. Compare test-set errors for points near $(Q_0 = 0.5, a = 50)$ vs. points near the center $(Q_0 = 1.0, a = 125)$
2. Does the ensemble uncertainty increase near the edges?
3. What happens if you evaluate *slightly* outside your training range (e.g., $Q_0 = 0.45$)?

Discuss: How would you know if your emulator is being asked to extrapolate dangerously?

---

## Part 4: Inference with NumPyro

### 4.1 The Inverse Problem

**Given**: Summary statistics from a held-out simulation with known true parameters

**Infer**: What initial conditions $(Q_0, a)$ produced these statistics?

Instead of running expensive N-body simulations inside your inference loop, you call your fast emulator. This is the payoff: thousands of posterior evaluations in seconds, not days.

### 4.2 What Is Probabilistic Programming?

**Probabilistic programming** is a paradigm where you write a *generative model*—code that describes how data is generated from parameters—and the framework automatically performs inference.

The idea: instead of deriving Bayes' theorem by hand and implementing samplers yourself (Project 4), you declare:
1. What parameters exist and their prior distributions
2. How parameters generate predictions (your forward model)
3. How predictions relate to observations (the likelihood)

The probabilistic programming framework handles the rest: computing posteriors, running MCMC, diagnostics.

**NumPyro** is a JAX-native probabilistic programming library. It's fast (JIT-compiled), composable (works with other JAX code), and uses the same NUTS sampler you understand from HMC—just a production-quality implementation with automatic tuning.

### 4.3 The Generative Model

Think of your model as a story of how data comes to exist:

1. **Nature chooses parameters**: $(Q_0, a)$ drawn from some prior distribution
2. **Physics happens**: The cluster evolves according to N-body dynamics
3. **We measure outcomes**: Summary statistics $(f_{\rm bound}, \sigma_v, r_h)$, with some uncertainty

In NumPyro, you write this story as code:

```python
def model(observed_data):
    # Step 1: Prior beliefs about parameters
    Q0 = sample("Q0", Uniform(0.5, 1.5))
    a = sample("a", Uniform(50, 200))
    
    # Step 2: Forward model (your emulator replaces N-body sim)
    predicted = emulator(Q0, a)
    
    # Step 3: Likelihood—how observations relate to predictions
    sample("obs", Normal(predicted, sigma_obs), obs=observed_data)
```

NumPyro's NUTS sampler then explores the posterior $p(Q_0, a \,|\, \text{observed data})$.

### 4.4 Why NUTS?

NUTS (No-U-Turn Sampler) is an adaptive variant of HMC that automatically tunes the trajectory length. You built HMC in Project 4—NUTS is the same idea, but:
- Automatically determines how many leapfrog steps to take
- Adapts step size during warmup
- Handles a wide range of posteriors without manual tuning

For this problem with 2 continuous parameters and a smooth posterior, NUTS is ideal.

### 4.5 Observation Uncertainty

You must specify $\sigma_{\rm obs}$ for each output. But what does this represent?

In real applications, $\sigma_{\rm obs}$ is **measurement uncertainty**—noise in your observations. Here, you're using synthetic "observations" from simulations, so there's no measurement noise in the traditional sense.

**For this project**, treat $\sigma_{\rm obs}$ as representing the **combined uncertainty** from:
1. **Emulator error**: Your NN doesn't perfectly predict the simulation
2. **Stochastic variation**: Different IMF realizations produce different outputs even for the same $(Q_0, a)$

**Practical guidance**: Use your emulator's test-set RMSE for each output as $\sigma_{\rm obs}$. This is a reasonable approximation—it captures how well your emulator predicts unseen simulations.

If $\sigma_{\rm obs}$ is too small, the posterior will be overconfident and may miss the true parameters. If too large, the posterior will be uninformative. Your test-set RMSE is a principled starting point.

### 4.6 Validation

Pick a simulation from your test set with known $(Q_0^{\rm true}, a^{\rm true})$. Use its summary statistics as "observations." Run NUTS.

**Check**: Do the 95% credible intervals contain the true values?

Create a corner plot showing joint and marginal posteriors. Mark the true parameter values.

**Study the NumPyro tutorials**:
- "Getting Started"
- "Bayesian Regression" (closest to your problem structure)

### 4.7 Code Skeleton

Similarly, your `inference.py` should contain something like:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def make_model(emulator, sigma_obs):
    """
    Returns a NumPyro model function for inference.
    
    Args:
        emulator: trained NN emulator (or ensemble predict function)
        sigma_obs: (3,) array of observation uncertainties [sigma_f, sigma_v, sigma_r]
    """
    def model(observed_stats):
        # TODO: Sample Q0 from prior (Uniform over your training range)
        # TODO: Sample a from prior (Uniform over your training range)
        
        # TODO: Call emulator to get predicted stats
        # (Remember to normalize inputs and unnormalize outputs!)
        
        # TODO: Define likelihood 
        # Hint: numpyro.sample("obs", dist.Normal(...), obs=observed_stats)
        pass
    
    return model

def run_inference(model, observed_stats, num_warmup=500, num_samples=2000):
    # TODO: Set up NUTS sampler
    # TODO: Run MCMC
    # TODO: Return samples
    pass
```

The NumPyro "Bayesian Regression" tutorial shows this exact pattern—priors, forward model, likelihood with `obs=`.

:::{admonition} Checkpoint
:class: important

Before finalizing:
- [ ] NumPyro runs without errors
- [ ] Posteriors are sensible (not flat, not delta functions)
- [ ] True parameters recovered within credible intervals
:::

---

## Part 5: Package Structure

Organize as an installable package:

```
nbody_emulator/
├── pyproject.toml
├── README.md
├── src/
│   └── nbody_emulator/
│       ├── __init__.py
│       ├── data.py           # Summary stats, data I/O
│       ├── emulator.py       # NN definition, training, prediction
│       ├── inference.py      # NumPyro model
│       └── utils.py          # Normalization, helpers
├── scripts/
│   ├── generate_data.py
│   ├── train_emulator.py
│   └── run_inference.py
├── tests/
│   └── test_emulator.py
└── outputs/
    ├── data/
    └── figures/
```

**Dependencies** (in `pyproject.toml`):
- jax, jaxlib
- equinox, optax, numpyro
- numpy, scipy, matplotlib
- Your N-body package from Project 5 (install locally with `pip install -e /path/to/your/nbody_package`)

**Tests**: Basic sanity checks—normalization roundtrip, correct output shapes, no NaN.

---

## Part 6: Deliverables

**Submission**: Push your final code to your GitHub Classroom repository by the deadline.

### 6.1 Code Repository

- Package structure as above
- README with installation and usage
- Scripts that reproduce your results
- Test suite

### 6.2 Research Memo

Include in your repository as `memo.pdf` or `memo.md`. Address:

**Introduction**: The scientific question and your approach.

**Methods**: Training data design, architecture choices, challenges encountered.

**Emulator Evaluation**: Accuracy metrics (table), predicted vs. true (figure), uncertainty and edge behavior.

**Inference Results**: Posterior corner plot, parameter recovery assessment.

### 6.3 Required Figures

1. Training data distribution in $(Q_0, a)$ space
2. Training loss vs. epoch
3. Predicted vs. true (3-panel, one per output)
4. Uncertainty slice (mean ± 2σ along 1D cut)
5. Posterior corner plot with true values marked

---

## Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Emulator Pipeline** | 45% | Training data (LHS, summary stats), NN converges, test accuracy, uncertainty estimation |
| **NumPyro Inference** | 25% | Working pipeline, parameter recovery within credible intervals |
| **Code Quality** | 10% | Package structure, documentation, tests |
| **Research Memo** | 20% | Clear presentation, figures, thoughtful interpretation |

---

## Tips

**Learning new packages**: Start with official tutorials. Run examples before customizing. Read error messages carefully.

**NN won't train**: Check normalization. Lower learning rate. Look for NaN in data.

**NumPyro slow**: Is emulator JIT-compiled? Are priors reasonable?

**Posterior misses true values**: Is $\sigma_{\rm obs}$ too small (overconfident)? Is emulator accurate in that region?

---

## The Bigger Picture

This project teaches the workflow of modern computational science:

1. Expensive simulations → ground truth
2. Machine learning → fast surrogate
3. Probabilistic inference → knowledge from data

The JAX ecosystem you're learning represents the frontier of scientific ML. You're ahead of the curve.
