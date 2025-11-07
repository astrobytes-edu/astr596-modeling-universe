# Introduction to Machine Learning for Astrophysics

```{epigraph}
"The purpose of computing is insight, not numbers."

-- Richard Hamming
```

## Learning Objectives

By the end of this module, you will be able to:

1. **Articulate** what machine learning is and how it differs from (and connects to) traditional scientific computing
2. **Identify** when machine learning is appropriate for astrophysical problems versus when physics-based simulation is better
3. **Understand** the fundamental learning problem: balancing model complexity with data limitations
4. **Connect** machine learning to statistical inference (Module 5) and function approximation (Module 1)
5. **Distinguish** between supervised, unsupervised, and reinforcement learning paradigms
6. **Recognize** common pitfalls: overfitting, underfitting, and the bias-variance tradeoff
7. **Design** validation strategies to assess model generalization
8. **Appreciate** the philosophical shift from "physics first" to "data first" approaches in modern astrophysics

```{admonition} Where We Are in the Course
:class: note

This module represents the culmination of everything you've learned:

**The Statistical Foundation** (Modules 1-4):
- **Module 1**: Probability, distributions, moments, sampling → *statistical description of uncertainty*
- **Module 2**: Statistical mechanics, Boltzmann equation → *physics from statistics*
- **Module 3**: Phase space, dynamics, N-body → *simulating the universe*
- **Module 4**: Radiative transfer, Monte Carlo → *photons and computation*

**The Inferential Framework** (Module 5):
- Bayesian inference: $p(\theta \,|\, \mathbf{D})$ → *learning from data*
- MCMC: sampling complex posteriors → *computational inference*
- Model comparison: evidence and Bayes factors → *choosing between models*

**The Machine Learning Synthesis** (This Module):
- **Part 1** (This): What is machine learning? The learning problem
- **Part 2**: Gaussian Processes → learning *functions* not parameters
- **Part 3**: Neural Networks → learning *dynamics* from data

We're moving from **understanding the universe through physics** to **learning the universe from data**.
```

---

## 🔴 Part 1: What Is Machine Learning?

### The Traditional Scientific Method

You've spent this entire semester doing **physics-based** computational astrophysics:

1. **Write down equations**: $\frac{d\mathbf{v}_i}{dt} = \sum_{j \neq i} \frac{Gm_j(\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}$
2. **Solve numerically**: Runge-Kutta, leapfrog, adaptive timesteps
3. **Analyze results**: Measure observables, compare to theory
4. **Iterate**: Refine physics, improve numerics, explore parameter space

This is **model-driven science**: we start with physical laws and derive predictions.

**Strengths**:
- ✅ Interpretable (we understand every step)
- ✅ Generalizable (physics is universal)
- ✅ Predictive (can extrapolate beyond training regime)
- ✅ Satisfying (we *understand* why things happen)

**Limitations**:
- ❌ Computationally expensive (Project 2: minutes per simulation)
- ❌ Doesn't scale to complex systems (turbulence, galaxy formation)
- ❌ Requires known physics (what about dark matter?)
- ❌ Parameter space exploration is prohibitive (need 10,000+ simulations)

### The Machine Learning Paradigm

Machine learning inverts the traditional approach:

**Instead of**: Physics equations → Simulation → Predictions

**We do**: Data → Learning algorithm → Predictions

```{admonition} The Profound Shift
:class: tip

Traditional science asks: **"Given these physical laws, what will happen?"**

Machine learning asks: **"Given these observations, what patterns exist?"**

This isn't replacing physics—it's *complementing* it. We use ML when:
1. Physics is too expensive to simulate directly
2. Physics is unknown or incomplete
3. Data is abundant but complex
4. We need fast predictions for exploration/optimization
```

### A Concrete Example: Your Final Project

**The Problem**: Predict how a star cluster evolves from $t=0$ to $t=200$ Myr.

**Physics-based approach** (Projects 2 & 5):
```
Initial conditions → N-body equations → Integrate 200 Myr → r_core(t)
                    (Computationally expensive!)
```

**Machine learning approach** (Final Project):
```
Many N-body simulations → Learn patterns → Instant predictions
(Upfront cost, then fast!)
```

**The key insight**: We're not *replacing* physics—we're using N-body simulations to *train* a fast surrogate model.

This is **physics-informed machine learning**: combining physical knowledge with data-driven learning.

---

## 🔴 Part 2: The Learning Problem - Generalization from Data

### What Does It Mean to "Learn"?

Suppose we have training data:
$$
\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}
$$

where:
- $\mathbf{x}_i$ are inputs (e.g., initial cluster conditions)
- $y_i$ are outputs (e.g., core radius at $t=100$ Myr)

**Goal**: Find a function $f$ such that for **new, unseen** inputs $\mathbf{x}_*$:
$$
f(\mathbf{x}_*) \approx y_*
$$

**This is the learning problem**: generalize from observed data to unobserved cases.

```{admonition} Connection to Module 1: Function Approximation
:class: note

Remember basis function expansions?
$$
f(x) = \sum_{i=1}^{M} w_i \phi_i(x)
$$

Machine learning is function approximation where:
1. The basis functions $\phi_i$ are learned from data (not chosen a priori)
2. We care about **generalization** to new data (not just fitting training data)
3. We quantify **uncertainty** in predictions

This connects directly to your work on Fourier series, Legendre polynomials, etc. in Module 1!
```

### The Fundamental Challenge: Bias-Variance Tradeoff

Consider fitting a polynomial to data:

**Underfitting** (high bias):
$$
f(x) = w_0 + w_1 x \quad \text{(linear model)}
$$
- Too simple to capture true pattern
- Poor performance on *both* training and test data
- **Bias**: model assumptions are wrong

**Overfitting** (high variance):
$$
f(x) = \sum_{i=0}^{100} w_i x^i \quad \text{(100th degree polynomial)}
$$
- Fits training data perfectly (even noise!)
- Poor performance on *test* data
- **Variance**: model is too sensitive to training data

**Just right**:
$$
f(x) = \sum_{i=0}^{5} w_i x^i \quad \text{(modest degree polynomial)}
$$
- Captures true pattern without memorizing noise
- Good performance on *both* training and test data

```{figure} #
:name: fig-bias-variance

**Figure 1**: The bias-variance tradeoff.
- **Left**: Underfit model (high bias) - too simple, doesn't capture curvature
- **Middle**: Well-fit model - captures trend, robust to noise
- **Right**: Overfit model (high variance) - memorizes noise, wiggles unrealistically
```

**The mathematical formulation**:

Expected prediction error decomposes as:
$$
\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{sensitivity to data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
$$

**Key insight**: There's an optimal model complexity that balances bias and variance!

### Training, Validation, and Test Sets

**The gold standard**: Split data into three sets:

1. **Training set** (60-80%): Fit model parameters
2. **Validation set** (10-20%): Tune hyperparameters (model complexity, regularization)
3. **Test set** (10-20%): Final evaluation (never used during development!)

**Why three sets?**
- Training: Learns patterns
- Validation: Prevents overfitting to training data
- Test: Unbiased estimate of generalization performance

```{admonition} Critical Mistake to Avoid
:class: warning

**Never** use the test set for:
- Choosing model architecture
- Tuning hyperparameters
- Deciding when to stop training
- Any decision whatsoever!

The test set is **sacred**—touch it only once at the very end.

If you tune your model based on test performance, you're effectively training on the test set (indirectly). This gives you an overly optimistic estimate of how well your model will generalize.
```

### Cross-Validation: Making the Most of Limited Data

When data is scarce (e.g., 200 expensive N-body simulations), splitting into train/val/test is wasteful.

**K-fold cross-validation**:
1. Divide data into $K$ folds (typically $K=5$ or $K=10$)
2. For each fold $k$:
   - Train on all folds except $k$
   - Validate on fold $k$
3. Average performance across all $K$ folds

This uses all data for both training and validation (at different times).

```python
import jax.numpy as jnp

def k_fold_cross_validation(X, y, model_fn, k=5):
    """K-fold cross-validation
    
    Args:
        X: Input data (N, d)
        y: Outputs (N,)
        model_fn: Function that trains and evaluates model
        k: Number of folds
        
    Returns:
        scores: Validation scores for each fold
    """
    N = len(X)
    fold_size = N // k
    scores = []
    
    for i in range(k):
        # Split into train and validation
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        # Validation fold
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        
        # Training folds (everything else)
        X_train = jnp.concatenate([X[:val_start], X[val_end:]])
        y_train = jnp.concatenate([y[:val_start], y[val_end:]])
        
        # Train and evaluate
        score = model_fn(X_train, y_train, X_val, y_val)
        scores.append(score)
    
    return jnp.array(scores)

# Usage
mean_score = jnp.mean(scores)
std_score = jnp.std(scores)
print(f"CV Score: {mean_score:.3f} ± {std_score:.3f}")
```

---

## 🟡 Part 3: Types of Machine Learning

### Supervised Learning: Learning from Labeled Examples

**Setup**: We have input-output pairs $\{(\mathbf{x}_i, y_i)\}$

**Goal**: Learn mapping $f: \mathbf{x} \to y$

**Two subtypes**:

1. **Regression**: Predict continuous values
   - Example: Initial conditions → core radius
   - Loss: Mean Squared Error (MSE)

2. **Classification**: Predict discrete categories  
   - Example: Galaxy image → [spiral, elliptical, irregular]
   - Loss: Cross-entropy

**Astrophysical applications**:
- Photometric redshifts (colors → distance)
- Supernova classification (light curve → type Ia vs core-collapse)
- Exoplanet detection (light curve → planet or not)
- **Your project**: Initial conditions → cluster evolution

```{admonition} Connection to Module 5: Maximum Likelihood
:class: note

Supervised learning is **maximum likelihood estimation**!

If we assume Gaussian noise: $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

Then the likelihood is:
$$
p(\mathbf{y} \,|\, \mathbf{X}, \boldsymbol{\theta}) = \prod_{i=1}^{N} \mathcal{N}(y_i \,|\, f_{\boldsymbol{\theta}}(\mathbf{x}_i), \sigma^2)
$$

Maximizing likelihood = minimizing negative log-likelihood:
$$
-\log p(\mathbf{y} \,|\, \mathbf{X}, \boldsymbol{\theta}) = \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i))^2 + \text{const}
$$

**This is exactly Mean Squared Error!**

So training neural networks is just maximum likelihood estimation under the hood. Everything connects back to Bayesian inference from Module 5.
```

### Unsupervised Learning: Finding Structure Without Labels

**Setup**: We only have inputs $\{\mathbf{x}_i\}$ (no outputs!)

**Goal**: Discover hidden structure, patterns, or groupings

**Common tasks**:

1. **Clustering**: Group similar objects
   - K-means, hierarchical clustering
   - Example: Group galaxies by properties (without pre-defined labels)

2. **Dimensionality reduction**: Find low-dimensional representation
   - PCA, t-SNE, autoencoders
   - Example: Compress 1000-dimensional galaxy spectra to 10 principal components

3. **Density estimation**: Learn probability distribution
   - Example: What's the distribution of stellar masses in clusters?

**Astrophysical applications**:
- Discovering new classes of objects (quasars, gamma-ray bursts)
- Anomaly detection (unusual supernovae, transients)
- Data compression (large survey data)

```{admonition} Example: Unsupervised vs Supervised
:class: tip

**Supervised**: "Here are 1000 galaxies labeled as spiral or elliptical. Learn to classify new galaxies."

**Unsupervised**: "Here are 1000 unlabeled galaxies. Do they naturally group into categories?"

Unsupervised learning can *discover* categories we didn't know existed!
```

### Reinforcement Learning: Learning from Interaction

**Setup**: Agent interacts with environment, receives rewards

**Goal**: Learn policy (strategy) that maximizes cumulative reward

**Components**:
- State $s$: Current situation
- Action $a$: What agent can do
- Reward $r$: Feedback signal
- Policy $\pi(a|s)$: Strategy for choosing actions

**Astrophysical applications** (less common but emerging):
- Optimizing telescope scheduling
- Adaptive optics control
- Gravitational wave detector tuning

**Not the focus of this course** (supervised learning is most relevant for emulation).

---

## 🔴 Part 4: Why Machine Learning for Astrophysics?

### The Data Deluge

Astronomy is drowning in data:

| Survey/Instrument | Data Volume | Time to Analyze Manually |
|------------------|-------------|-------------------------|
| Sloan Digital Sky Survey (SDSS) | 200 million objects | Centuries |
| Large Synoptic Survey Telescope (LSST) | 30 TB/night | Impossible |
| Square Kilometer Array (SKA) | 160 TB/second | Absolutely impossible |
| Gaia | 1 billion stars | Many lifetimes |

**Traditional approach**: Manually inspect each object, classify, measure properties

**Machine learning approach**: Train algorithms to do this automatically

### The Complexity Challenge

Some astrophysical systems are too complex for analytic solutions:

**Simple** (we have equations):
- Two-body problem: Solved exactly (Kepler orbits)
- Linear perturbations: Analytic solutions exist
- Spherical symmetry: Reduces to 1D ODEs

**Complex** (need simulation):
- N-body problem ($N > 2$): No closed-form solution
- Turbulence: Highly nonlinear, chaotic
- Galaxy formation: Multi-scale, multi-physics

**Very complex** (even simulation is hard):
- Cosmological structure formation: Box size vs resolution tradeoff
- Stellar interiors with magnetic fields: MHD is expensive
- Radiative transfer in 3D: Photon transport is costly

**Machine learning solution**: Learn simplified models from expensive simulations

### The Emulation Use Case: Your Final Project

This is the **primary motivation** for your final project:

**The Problem**:
- N-body simulation: 1 minute per run
- Want to explore 5D parameter space: Need ~10,000 runs
- Total time: ~1 week of continuous computing
- Then if we update physics? Start over!

**The ML Solution**:
1. Run 500 simulations (8 hours)
2. Train surrogate model (1 hour)
3. Make 10,000 predictions (seconds)
4. **Total: <10 hours instead of 1 week**

**The benefit**: 100× speedup enables:
- Parameter space exploration
- Uncertainty quantification
- Optimization (find best-fit parameters)
- Real-time analysis

```{admonition} The Tradeoff
:class: important

**Physics simulation**:
- ✅ Exact (within numerical precision)
- ✅ Interpretable (we understand every force)
- ✅ Generalizes (physics is universal)
- ❌ Slow (minutes to hours per run)

**ML emulator**:
- ✅ Fast (milliseconds per prediction)
- ✅ Smooth (learns continuous functions)
- ⚠️ Approximate (prediction error ~1-10%)
- ⚠️ Interpolates (not extrapolates) well
- ❌ Black box (harder to interpret)

The goal isn't to replace physics—it's to **accelerate exploration** while staying honest about uncertainties.
```

### When NOT to Use Machine Learning

ML is powerful but not always appropriate:

❌ **Don't use ML when**:
1. **Physics is cheap**: If simulation takes seconds, no need for emulation
2. **Data is scarce**: <100 training examples → physics priors are better
3. **Interpretability is critical**: Need to understand mechanism, not just predict
4. **Extrapolation is required**: ML fails outside training distribution
5. **Uncertainty quantification is essential**: Standard NNs don't provide this

✅ **Use ML when**:
1. **Physics is expensive**: Emulation saves time
2. **Data is abundant**: Enough examples to learn patterns
3. **Patterns are complex**: Too nonlinear for simple models
4. **Speed matters**: Need real-time or interactive predictions
5. **You validate carefully**: Test generalization thoroughly

---

## 🟡 Part 5: The Spectrum of Models - From Physics to Data

### A Taxonomy of Approaches

```{figure} #
:name: fig-model-spectrum

**Figure 2**: The spectrum from purely physics-based to purely data-driven models.
```

**Level 1: Pure Physics** (Projects 1-5)
- Start from first principles ($F = ma$)
- Solve equations numerically
- No learning from data
- Example: Your N-body code

**Level 2: Physics-Informed ML** (Final project, Residual Networks)
- Use physics as scaffold
- Learn corrections from data
- Example: $f_{\text{total}} = f_{\text{Henon}} + f_{\text{NN}}$
- Best of both worlds!

**Level 3: Constrained ML** (Physics-Informed Neural Networks)
- Encode physics in loss functions
- Network is flexible but respects constraints
- Example: Energy conservation penalty

**Level 4: Pure ML** (Standard Neural Networks)
- Learn directly from data
- No explicit physics
- Example: Image recognition, time series prediction

**Level 5: End-to-End Learning** (Deep Learning on Raw Data)
- Learn representations + predictions jointly
- Example: Galaxy morphology from pixels

```{admonition} The Astrophysics Sweet Spot
:class: tip

Most astrophysical applications sit at **Levels 2-3**:
- We have strong physical priors (conservation laws, symmetries)
- Data is expensive but available
- We want speed + interpretability

Pure data-driven ML (Levels 4-5) works when:
- Physics is unknown or extremely complex
- Massive datasets available (images, spectra)
- Task is pattern recognition rather than understanding
```

### Example: Photometric Redshifts

**The Problem**: Estimate galaxy distance from broad-band colors (no spectroscopy)

**Level 1 (Pure Physics)**: 
- Model galaxy spectral energy distributions (SEDs)
- Compute expected colors at each redshift
- Find best-fit template
- **Pro**: Interpretable. **Con**: Templates may not match real galaxies

**Level 2-3 (Physics-Informed ML)**:
- Train on galaxies with known redshifts (spectroscopy)
- Use physics-motivated features (colors, emission lines)
- Constrain predictions to be positive, ordered
- **Pro**: More accurate. **Con**: Needs training data

**Level 4 (Pure ML)**:
- Neural network: colors → redshift
- No physics assumptions
- **Pro**: Very accurate. **Con**: Black box, hard to diagnose failures

**Current best practice**: Hybrid approaches (Levels 2-3)

---

## 🔴 Part 6: The Learning Algorithm Landscape

### Classical Machine Learning (Pre-Deep Learning)

**Linear Models**:
- Linear regression: $y = \mathbf{w}^T \mathbf{x} + b$
- Logistic regression: $p(y=1) = \sigma(\mathbf{w}^T \mathbf{x} + b)$
- **Pro**: Fast, interpretable. **Con**: Limited expressivity

**Tree-Based Methods**:
- Decision trees: Binary splits on features
- Random forests: Ensemble of trees
- Gradient boosting (XGBoost, LightGBM)
- **Pro**: Handles nonlinearity, feature importance. **Con**: Can overfit

**Support Vector Machines (SVMs)**:
- Find maximum-margin hyperplane
- Kernel trick for nonlinearity
- **Pro**: Good for small data. **Con**: Doesn't scale to large datasets

**Gaussian Processes** (Part 2 of this module):
- Bayesian approach to function learning
- Automatic uncertainty quantification
- **Pro**: Principled, interpretable. **Con**: $O(N^3)$ scaling

### Deep Learning (Modern ML)

**Neural Networks** (Part 3 of this module):
- Multilayer perceptrons (MLPs)
- Convolutional networks (CNNs) for images
- Recurrent networks (RNNs, GRUs) for sequences
- Transformers for sequences
- **Pro**: Universal approximators, scalable. **Con**: Data-hungry, black-box

**Specialized Architectures**:
- Neural ODEs: Learn continuous dynamics
- Graph neural networks: Data with graph structure
- Physics-informed neural networks: Encode PDEs
- **Pro**: Incorporate domain knowledge. **Con**: More complex to train

**The Modern Paradigm**:
- Deep learning dominates when data is abundant (millions of examples)
- Classical ML still competitive for small datasets (<10,000 examples)
- Your project uses **both**: GPs for scalars, NNs for time series

---

## 🟡 Part 7: Evaluation and Validation

### Metrics for Regression

For continuous predictions $\hat{y}_i$ vs true values $y_i$:

**Mean Squared Error (MSE)**:
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$
- Units: squared output units
- Penalizes large errors heavily

**Root Mean Squared Error (RMSE)**:
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$
- Units: same as output
- Easier to interpret than MSE

**Mean Absolute Error (MAE)**:
$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$
- More robust to outliers than MSE

**R² Score** (Coefficient of Determination):
$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: No better than predicting mean
- $R^2 < 0$: Worse than predicting mean!

```python
import jax.numpy as jnp

def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    # Residuals
    residuals = y_true - y_pred
    
    # MSE and RMSE
    mse = jnp.mean(residuals**2)
    rmse = jnp.sqrt(mse)
    
    # MAE
    mae = jnp.mean(jnp.abs(residuals))
    
    # R^2
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
```

### Metrics for Classification

For discrete predictions (categories):

**Accuracy**:
$$
\text{Accuracy} = \frac{\text{Number correct}}{\text{Total predictions}}
$$

**Precision and Recall**:
- Precision: Of predicted positives, how many are correct?
- Recall: Of true positives, how many did we find?

**F1 Score**: Harmonic mean of precision and recall

**Confusion Matrix**: Table of true vs predicted labels

### Visualizing Model Performance

**Residual Plots**:
```
Plot: y_true - y_pred vs y_pred
Good: Random scatter around zero
Bad: Systematic patterns (model bias)
```

**Prediction Plots**:
```
Plot: y_pred vs y_true
Good: Points on diagonal
Bad: Systematic deviations
```

**Learning Curves**:
```
Plot: Training loss and validation loss vs epoch
Good: Both decrease, converge
Bad: Validation loss increases (overfitting!)
```

---

## 🔴 Part 8: Regularization - Preventing Overfitting

### The Overfitting Problem

Complex models can **memorize** training data instead of learning patterns.

**Symptoms**:
- Training error very low, test error high
- Model performs well on training data, poorly on new data
- Predictions are overly sensitive to noise

**Solution**: **Regularization** - constrain model complexity

### L2 Regularization (Weight Decay)

Add penalty for large weights:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_i w_i^2
$$

**Effect**: Prefers simpler models with smaller weights

**Interpretation**: This is a **Bayesian prior** on weights!
$$
p(w_i) = \mathcal{N}(0, \sigma^2) \quad \Rightarrow \quad -\log p(\mathbf{w}) \propto \sum_i w_i^2
$$

**Connection to Module 5**: Regularization = MAP estimation with Gaussian prior!

### L1 Regularization (Lasso)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|
$$

**Effect**: Drives some weights to exactly zero (sparse models)

**Use case**: Feature selection (which inputs matter?)

### Early Stopping

Monitor validation loss during training:
- If validation loss stops improving for $N$ epochs → stop training
- Don't wait until convergence on training data

**Why it works**: Prevents model from overfitting to training noise

### Dropout (for Neural Networks)

During training, randomly "drop" (set to zero) some neurons with probability $p$

**Effect**: Forces network to learn robust features (can't rely on any single neuron)

**At test time**: Use all neurons, scale activations by $(1-p)$

```{admonition} The Regularization Principle
:class: important

**Occam's Razor for Machine Learning**:

Among models that fit the data equally well, prefer the simplest.

Regularization mathematically encodes this principle:
- L2: Prefer small weights (smooth functions)
- L1: Prefer sparse weights (few active features)
- Early stopping: Prefer early solutions (less complex)
- Dropout: Prefer distributed representations (robust)

All of these reduce model complexity → better generalization!
```

---

## 🟡 Part 9: The Philosophical Shift - From Understanding to Prediction

### The Traditional Physics Mindset

As a physicist, you've been trained to ask:
- **Why** does this happen?
- **What are the fundamental laws**?
- **Can we derive this from first principles**?

This is the **explanatory** paradigm: science as understanding mechanism.

**Example**: Newton didn't just predict planetary motion—he explained it via gravity.

### The Machine Learning Mindset

Machine learning often asks:
- **What** will happen?
- **How accurately** can we predict?
- **What patterns exist** in the data?

This is the **predictive** paradigm: science as pattern recognition.

**Example**: A neural network might predict galaxy morphology accurately without understanding how galaxies form.

```{admonition} Are These in Conflict?
:class: tip

**No!** They're complementary.

**Physics gives us**:
- Fundamental understanding
- Generalization beyond training data
- Confidence in extrapolation

**Machine learning gives us**:
- Speed (predictions in milliseconds)
- Ability to handle complexity
- Discovery of unexpected patterns

**Best approach**: Combine both!
- Use physics to constrain ML models
- Use ML to accelerate physics simulations
- Use ML to discover new physics (anomaly detection)
```

### The Role of Interpretability

**The interpretability spectrum**:

**Most interpretable**:
1. Linear models ($y = w_1 x_1 + w_2 x_2 + \ldots$)
   - Every coefficient has clear meaning
2. Decision trees
   - Can visualize decision path
3. Gaussian Processes (Part 2)
   - Lengthscales tell us which features matter
4. Shallow neural networks
   - Can visualize feature importance

**Least interpretable**:
5. Deep neural networks (Part 3)
   - Millions of parameters, no clear interpretation

**The tradeoff**: Often accuracy ↔ interpretability

**For astrophysics**:
- Interpretability matters when testing theories
- Prediction matters when enabling observations
- Choose based on goal!

### Scientific Integrity in the ML Era

**Challenges**:
1. **Publication bias**: Only reporting successful models
2. **Data leakage**: Accidentally using test data during development
3. **Overfitting to benchmarks**: Tuning until test score is good
4. **Black-box predictions**: Can't explain why model failed

**Best practices**:
1. **Pre-register analysis plan**: Decide methodology before seeing results
2. **Hold out test set**: Touch it only once!
3. **Report all experiments**: Including failures
4. **Validate physically**: Do predictions make sense?
5. **Open science**: Share code and data

```{admonition} The Glass-Box Philosophy in ML
:class: important

This course emphasizes **understanding before using**:

**Week 1-2 (GPs)**: Build from scratch
- Implement kernels, Cholesky decomposition, conditioning
- Understand *why* GPs work, not just *how* to use them

**Week 3-4 (NNs)**: Build from scratch
- Implement MLPs, backprop, training loops
- Understand gradient flow, loss landscapes

**Why this matters**:
- You'll know when models fail (and why)
- You'll choose architectures wisely
- You'll debug effectively
- You'll use libraries responsibly

**Analogy**: You wouldn't use an N-body code without understanding Newton's laws. Same principle applies to ML!
```

---

## 🔴 Part 10: Looking Ahead - The Module Structure

### Part 2: Gaussian Processes

**What you'll learn**:
- GPs as distributions over functions (infinite-dimensional Gaussians)
- Kernels encode physical assumptions (smoothness, periodicity)
- Exact Bayesian inference (no MCMC needed!)
- Uncertainty quantification for free

**What you'll build**:
- GP regression from scratch
- Hyperparameter optimization
- Application to scalar predictions ($\mathbf{x} \to r_{\text{core}}$)

**What you'll discover**:
- Where GPs fail (high-dimensional outputs, time series)
- Why we need something more powerful...

### Part 3: Neural Networks

**What you'll learn**:
- Universal approximation theorem
- Backpropagation and gradient descent
- Neural ODEs for continuous dynamics
- RNNs/GRUs for sequential data
- Physics-informed loss functions

**What you'll build**:
- Feedforward networks (MLPs)
- Neural ODE for cluster evolution
- Recurrent network for time series
- Full training pipelines with validation

**What you'll discover**:
- Trade-offs: expressivity vs interpretability
- When to use GPs vs NNs vs hybrid approaches

### The Final Project Arc

**Week 1**: Train GP on scalar outputs
- Learn: GP theory, kernel design, uncertainty
- Succeed: Predict $r_{\text{core}}(t=100)$ with error bars
- Discover: GPs struggle with time series

**Week 2**: Demonstrate GP limitations
- Experiment: Try GP on full trajectory
- Observe: Slow training, uncorrelated predictions, wide uncertainty
- Conclude: Need architecture that understands temporal structure

**Week 3-4**: Implement neural networks
- Build: MLP, GRU, Neural ODE
- Train: Learn full trajectories
- Compare: GP vs NN—when to use each?

**Deliverable**: Comprehensive analysis answering:
- **When should you use GP?** (Scalar outputs, uncertainty critical, data scarce)
- **When should you use NN?** (Time series, complex patterns, speed matters)
- **What about hybrid approaches?** (Best of both worlds!)

---

## Conceptual Checkpoints

Before moving to Parts 2 and 3, reflect on these questions:

1. **Generalization**: You have 500 N-body simulations. How do you know your ML model will work on the 501st simulation you haven't run yet?

2. **Bias-Variance**: A simple linear model has high bias but low variance. A 100-layer neural network has low bias but high variance. For 200 training samples, which would you choose? Why?

3. **Cross-validation**: Why do we need a separate validation set? Can't we just look at training error?

4. **Physics vs ML**: Your physics-based N-body code gives exact answers (within numerical precision). Your ML emulator gives approximate answers (±5% error). Why would you ever use the emulator?

5. **Interpretability**: A Gaussian Process tells you that the lengthscale for the $N$ parameter is small (meaning $N$ strongly affects core radius). A neural network predicts accurately but doesn't tell you why. Which is more useful for science?

6. **Overfitting**: You train a model that achieves 99.9% accuracy on training data but only 60% accuracy on test data. What went wrong? How would you fix it?

7. **Regularization**: Explain why adding a penalty $\lambda \sum_i w_i^2$ to the loss function is equivalent to putting a Gaussian prior on weights in Bayesian inference (Module 5 connection!).

8. **Connection to Module 1**: How is machine learning related to function approximation with basis functions? What makes ML different from choosing Fourier or Legendre basis functions?

---

## Further Reading

### Foundational Machine Learning

1. **James, Witten, Hastie, Tibshirani** (2013): *An Introduction to Statistical Learning*
   - Excellent introduction, light on math
   - Free PDF: https://www.statlearning.com/

2. **Bishop** (2006): *Pattern Recognition and Machine Learning*
   - More mathematical, comprehensive
   - Covers Bayesian perspective

3. **Murphy** (2012): *Machine Learning: A Probabilistic Perspective*
   - Connects ML to probabilistic inference
   - Great for physics backgrounds

### ML for Physical Sciences

1. **Mehta et al.** (2019): "A high-bias, low-variance introduction to Machine Learning for physicists"
   - Physics Today review article
   - Excellent overview for scientists
   - https://arxiv.org/abs/1803.08823

2. **Cranmer et al.** (2020): "The frontier of simulation-based inference"
   - Connects ML to statistical inference
   - Relevant for emulation

3. **Carleo et al.** (2019): "Machine learning and the physical sciences"
   - Reviews of Modern Physics
   - Comprehensive survey

### Astrophysics Applications

1. **Baron** (2019): "Machine Learning in Astronomy: A Practical Overview"
   - Survey of ML applications in astronomy

2. **Ntampaka et al.** (2019): "The Role of Machine Learning in Astrophysics"
   - Philosophical and practical perspectives

3. **Fluke & Jacobs** (2020): "Surveying the reach and maturity of machine learning and artificial intelligence in astronomy"
   - State of the field

### Gaussian Processes (Preview of Part 2)

1. **Rasmussen & Williams** (2006): *Gaussian Processes for Machine Learning*
   - The canonical reference
   - Free PDF: http://www.gaussianprocess.org/gpml/

### Deep Learning (Preview of Part 3)

1. **Goodfellow, Bengio, Courville** (2016): *Deep Learning*
   - Comprehensive textbook
   - Free online: https://www.deeplearningbook.org/

---

## What's Next

You now understand:
- ✅ What machine learning is (and isn't)
- ✅ The fundamental learning problem (generalization)
- ✅ Types of ML (supervised, unsupervised, reinforcement)
- ✅ Why ML matters for astrophysics (speed, complexity, data)
- ✅ How to evaluate models (metrics, validation, regularization)
- ✅ The philosophical shift (explanation vs prediction)

**In Part 2**, you'll learn:
- Gaussian Processes: Bayesian inference in function space
- How to build GP regression from scratch
- When GPs work (and when they don't)

**In Part 3**, you'll learn:
- Neural Networks: Universal function approximators
- How to build NNs from scratch (MLPs, Neural ODEs, RNNs)
- When NNs work better than GPs

**Together**, these three parts will give you:
- Deep understanding of modern ML methods
- Ability to choose appropriate tools for astrophysical problems
- Skills to build and validate emulators for expensive simulations
- Critical perspective on the role of ML in science

---

```{admonition} Final Thought: The Computational Astrophysicist's Toolkit
:class: important

At the beginning of this course, you had:
- **Theory**: Equations of motion, statistical mechanics, radiative transfer
- **Numerics**: Integrators, Monte Carlo methods, discretization schemes

Now you're adding:
- **Statistics**: Bayesian inference (Module 5)
- **Machine Learning**: GPs and NNs (this module)

These aren't separate domains—they're deeply connected:

$$
\text{Physics} \xrightarrow{\text{simulations}} \text{Data} \xrightarrow{\text{ML}} \text{Fast predictions} \xrightarrow{\text{inference}} \text{Scientific insights}
$$

You're not just a programmer or a physicist or a statistician.

You're a **computational scientist** who uses all these tools synergistically to understand the universe.

**Welcome to the 21st century of astrophysics.** 🌌
```
