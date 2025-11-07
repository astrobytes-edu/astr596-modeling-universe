# Uncertainty Quantification for Scientific Machine Learning

```{epigraph}
"It is better to be vaguely right than exactly wrong."

-- Carveth Read
```

## Learning Objectives

By the end of this module, you will be able to:

1. **Distinguish** between epistemic uncertainty (model uncertainty) and aleatoric uncertainty (data noise)
2. **Implement** ensemble methods to quantify neural network uncertainty in JAX/Equinox
3. **Apply** Monte Carlo Dropout as a Bayesian approximation for uncertainty estimation
4. **Evaluate** uncertainty calibration using reliability diagrams and coverage metrics
5. **Compare** GP uncertainty (automatic, principled) with NN uncertainty (ensemble-based, approximate)
6. **Design** uncertainty-aware loss functions that incorporate prediction confidence
7. **Validate** that your error bars are honest and scientifically meaningful
8. **Communicate** uncertainty appropriately in scientific contexts

```{admonition} Why This Module Is Critical
:class: important

**The fundamental difference between engineering and science:**

- **Engineering**: "Does it work?" → Accuracy matters most
- **Science**: "How confident are we?" → Uncertainty matters equally

In your final project, you'll compare:
- **GP predictions**: $r_{\text{core}} = 1.2 \pm 0.1$ pc (automatic uncertainty!)
- **NN predictions**: $r_{\text{core}} = 1.2$ pc (no uncertainty... yet)

This module teaches you how to get error bars from neural networks, so you can make **fair, scientific comparisons** between methods.
```

---

## 🔴 Part 1: Why Uncertainty Matters in Science

### The Problem with Point Predictions

Standard neural network training gives you a single prediction:
$$
\hat{y} = f_{\boldsymbol{\theta}}(\mathbf{x})
$$

**What this tells you**: "My best guess is $\hat{y}$"

**What this doesn't tell you**:
- How confident am I?
- Should I trust this prediction?
- Where do I need more training data?
- Is this interpolation or extrapolation?

### A Cautionary Tale: The Overconfident Model

**Scenario**: You train a neural network to predict core radius from initial conditions.

**Training data**: Clusters with $N \in [100, 1000]$ stars

**Test case**: Cluster with $N = 5000$ stars (extrapolation!)

**NN prediction**: $r_{\text{core}} = 0.8$ pc (looks reasonable!)

**True value**: $r_{\text{core}} = 2.5$ pc (factor of 3 wrong!)

**The problem**: The NN had no way to say "I don't know, this is outside my training distribution."

**With uncertainty quantification**:
- NN would predict: $r_{\text{core}} = 0.8 \pm 2.0$ pc (huge error bars!)
- You'd know not to trust this prediction
- You'd know where to collect more training data

```{admonition} Connection to Module 5: Bayesian Inference
:class: note

In Module 5 (Project 4), you learned Bayesian inference:
$$
p(\theta \,|\, \mathcal{D}) = \frac{p(\mathcal{D} \,|\, \theta) p(\theta)}{p(\mathcal{D})}
$$

The **posterior** $p(\theta \,|\, \mathcal{D})$ gives you uncertainty about parameters.

For predictions, we want the **posterior predictive distribution**:
$$
p(y_* \,|\, \mathbf{x}_*, \mathcal{D}) = \int p(y_* \,|\, \mathbf{x}_*, \theta) p(\theta \,|\, \mathcal{D}) \, d\theta
$$

**This is a distribution over predictions**, not a single value!

- **Gaussian Processes** compute this exactly (conjugacy)
- **Neural Networks** approximate it (ensembles, dropout)
```

### Two Types of Uncertainty

**1. Epistemic Uncertainty** (Model Uncertainty)
- Uncertainty about the **model itself**
- "I don't have enough data to know the right function"
- **Reducible**: Collect more training data → decreases

**Example**: 
- Few training points → High epistemic uncertainty
- Many training points → Low epistemic uncertainty

**2. Aleatoric Uncertainty** (Data Noise)
- Uncertainty inherent in the **data**
- "Even with perfect model, observations are noisy"
- **Irreducible**: More data doesn't help (it's inherent noise)

**Example**:
- N-body simulations have stochastic relaxation
- Even identical initial conditions → slightly different outcomes
- This is aleatoric uncertainty

```{figure} #
:name: fig-uncertainty-types

**Figure 1**: Epistemic vs Aleatoric Uncertainty
- **Left**: Few training points (circles) → Wide epistemic uncertainty (shaded region)
- **Right**: Many training points → Narrow epistemic uncertainty, but aleatoric noise remains
```

**For your final project**:
- **Epistemic**: Do we have enough N-body simulations to learn the pattern?
- **Aleatoric**: Even with infinite data, relaxation is stochastic

**Why distinguish them?**
- Epistemic → Tells us where to collect more data
- Aleatoric → Tells us fundamental limits of predictability

---

## 🔴 Part 2: Ensemble Methods - The Workhorse Approach

### The Core Idea

Train **multiple models** with different random initializations:
$$
f_{\boldsymbol{\theta}_1}(\mathbf{x}), \, f_{\boldsymbol{\theta}_2}(\mathbf{x}), \, \ldots, \, f_{\boldsymbol{\theta}_M}(\mathbf{x})
$$

**Ensemble prediction** (mean):
$$
\bar{f}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} f_{\boldsymbol{\theta}_m}(\mathbf{x})
$$

**Ensemble uncertainty** (standard deviation):
$$
\sigma_{\text{ensemble}}(\mathbf{x}) = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} \left(f_{\boldsymbol{\theta}_m}(\mathbf{x}) - \bar{f}(\mathbf{x})\right)^2}
$$

**Interpretation**: 
- If all models agree → Low uncertainty
- If models disagree → High uncertainty

```{admonition} Why This Works
:class: tip

**Neural networks are not unique!**

Given the same data, different initializations converge to different local minima. These different solutions represent **plausible explanations** of the data.

The spread of predictions across the ensemble captures **model uncertainty** (epistemic).

**Analogy**: Imagine asking 5 different scientists to fit a model to data. If they all agree, you're confident. If they disagree, you're uncertain.
```

### Implementation in JAX/Equinox

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List

class NeuralNetworkEnsemble(eqx.Module):
    """Ensemble of neural networks for uncertainty quantification
    
    Args:
        models: List of trained models
    """
    models: List[eqx.Module]
    
    def __init__(self, models):
        self.models = models
    
    def predict_with_uncertainty(self, x):
        """Predict with epistemic uncertainty
        
        Args:
            x: Input (d,) or batch (N, d)
            
        Returns:
            mean: Ensemble mean prediction
            std: Ensemble standard deviation (epistemic uncertainty)
        """
        # Check if batch or single input
        is_batch = x.ndim > 1
        
        if is_batch:
            # Batch prediction: each model predicts for all samples
            predictions = jnp.stack([
                jax.vmap(model)(x) for model in self.models
            ])  # (n_models, batch_size, output_dim)
        else:
            # Single prediction
            predictions = jnp.stack([
                model(x) for model in self.models
            ])  # (n_models, output_dim)
        
        # Ensemble statistics
        mean = jnp.mean(predictions, axis=0)
        std = jnp.std(predictions, axis=0)
        
        return mean, std


def train_ensemble(
    model_fn,
    X_train, y_train,
    X_val, y_val,
    n_models=5,
    n_epochs=1000,
    batch_size=32,
    learning_rate=1e-3,
    base_seed=0
):
    """Train ensemble of models with different initializations
    
    Args:
        model_fn: Function that creates a new model given a random key
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_models: Number of models in ensemble
        n_epochs: Training epochs per model
        batch_size: Batch size
        learning_rate: Learning rate
        base_seed: Base random seed
        
    Returns:
        ensemble: Trained ensemble
    """
    import optax
    
    trained_models = []
    
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")
        
        # Create model with unique initialization
        key = jax.random.PRNGKey(base_seed + i)
        model = model_fn(key)
        
        # Train model
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        def loss_fn(model, x, y):
            y_pred = jax.vmap(model)(x)
            return jnp.mean((y - y_pred)**2)
        
        @eqx.filter_jit
        def train_step(model, opt_state, x, y):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
        
        # Training loop
        best_val_loss = jnp.inf
        patience_counter = 0
        patience = 50
        
        for epoch in range(n_epochs):
            # Shuffle and batch
            key = jax.random.PRNGKey(epoch + i * 1000)
            perm = jax.random.permutation(key, len(X_train))
            
            epoch_loss = 0.0
            n_batches = 0
            
            for j in range(0, len(X_train), batch_size):
                batch_idx = perm[j:j+batch_size]
                x_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
                epoch_loss += loss
                n_batches += 1
            
            # Validation
            val_loss = loss_fn(model, X_val, y_val)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: train_loss={epoch_loss/n_batches:.4f}, val_loss={val_loss:.4f}")
        
        trained_models.append(best_model)
        print(f"  Best val loss: {best_val_loss:.4f}\n")
    
    return NeuralNetworkEnsemble(trained_models)


# Example usage
if __name__ == "__main__":
    from module6_part3_nns import MLP  # From Part 3
    
    # Generate dummy data
    key = jax.random.PRNGKey(42)
    X_train = jax.random.normal(key, (500, 5))
    y_train = jnp.sin(jnp.sum(X_train, axis=1, keepdims=True))
    X_val = jax.random.normal(key, (100, 5))
    y_val = jnp.sin(jnp.sum(X_val, axis=1, keepdims=True))
    
    # Model factory
    def create_mlp(key):
        return MLP(in_size=5, out_size=1, width=64, depth=3, key=key)
    
    # Train ensemble
    ensemble = train_ensemble(
        create_mlp,
        X_train, y_train,
        X_val, y_val,
        n_models=5,
        n_epochs=500
    )
    
    # Predict with uncertainty
    x_test = jax.random.normal(key, (5,))
    mean, std = ensemble.predict_with_uncertainty(x_test)
    print(f"Prediction: {mean[0]:.3f} ± {std[0]:.3f}")
```

### Bootstrapping: Ensembles with Data Variation

**Standard ensembles**: Same data, different initializations (epistemic only)

**Bootstrap ensembles**: Different data subsets + different initializations (epistemic + some aleatoric)

**Bootstrap procedure**:
1. For each model $m$:
   - Sample $N$ training points **with replacement**
   - Train model on this bootstrapped dataset
2. Ensemble predictions capture both model and data uncertainty

```python
def bootstrap_sample(X, y, key):
    """Sample with replacement"""
    n = len(X)
    indices = jax.random.choice(key, n, shape=(n,), replace=True)
    return X[indices], y[indices]

def train_bootstrap_ensemble(
    model_fn,
    X_train, y_train,
    X_val, y_val,
    n_models=5,
    **train_kwargs
):
    """Train ensemble with bootstrap sampling"""
    trained_models = []
    
    for i in range(n_models):
        # Bootstrap sample
        key = jax.random.PRNGKey(i)
        X_boot, y_boot = bootstrap_sample(X_train, y_train, key)
        
        # Train on bootstrapped data
        # (Use train_ensemble code but with X_boot, y_boot)
        # ... training logic ...
        
        trained_models.append(model)
    
    return NeuralNetworkEnsemble(trained_models)
```

**When to use bootstrap**:
- When you want to capture both epistemic and aleatoric uncertainty
- When training data is limited (bootstrap effectively increases diversity)
- When you suspect data sampling affects results

---

## 🟡 Part 3: Monte Carlo Dropout - Fast Uncertainty Approximation

### The Bayesian Interpretation of Dropout

**Standard dropout** (during training):
- Randomly drop neurons with probability $p$
- Forces network to learn robust features

**Monte Carlo dropout** (at test time):
- Keep dropout **active** during prediction
- Multiple forward passes → distribution of predictions

**Theoretical foundation** (Gal & Ghahramani, 2016):
- Dropout approximates **variational inference** in Bayesian neural networks
- Each dropout mask samples from approximate posterior $q(\boldsymbol{\theta})$

```{admonition} Why This is Profound
:class: tip

**Standard approach**: Dropout is a regularization trick

**Bayesian interpretation**: Dropout is approximate Bayesian inference!

$$
p(\boldsymbol{\theta} \,|\, \mathcal{D}) \approx q(\boldsymbol{\theta})
$$

where $q(\boldsymbol{\theta})$ is implicitly defined by the dropout mask distribution.

**Practical benefit**: Get Bayesian uncertainty without MCMC!
```

### Implementation in Equinox

```python
import jax
import jax.numpy as jnp
import equinox as eqx

class MLPWithDropout(eqx.Module):
    """MLP with dropout for MC uncertainty estimation"""
    layers: list
    dropouts: list
    activation: callable
    
    def __init__(
        self, 
        in_size, 
        out_size, 
        width, 
        depth, 
        dropout_rate=0.1,
        activation=jax.nn.relu,
        *, 
        key
    ):
        keys = jax.random.split(key, depth + 1)
        
        self.layers = []
        self.dropouts = []
        
        # Input layer
        self.layers.append(eqx.nn.Linear(in_size, width, key=keys[0]))
        self.dropouts.append(eqx.nn.Dropout(p=dropout_rate))
        
        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
            self.dropouts.append(eqx.nn.Dropout(p=dropout_rate))
        
        # Output layer (no dropout)
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))
        
        self.activation = activation
    
    def __call__(self, x, *, key=None):
        """Forward pass with optional dropout
        
        Args:
            x: Input
            key: Random key for dropout (if None, no dropout)
        """
        if key is not None:
            keys = jax.random.split(key, len(self.dropouts))
        else:
            keys = [None] * len(self.dropouts)
        
        # Hidden layers with activation and dropout
        for layer, dropout, dropout_key in zip(self.layers[:-1], self.dropouts, keys):
            x = layer(x)
            x = self.activation(x)
            if dropout_key is not None:
                x = dropout(x, key=dropout_key)
        
        # Output layer (no activation, no dropout)
        x = self.layers[-1](x)
        return x


def mc_dropout_predict(model, x, n_samples=100, *, key):
    """Monte Carlo dropout prediction with uncertainty
    
    Args:
        model: Model with dropout
        x: Input (d,) or batch (N, d)
        n_samples: Number of MC samples
        key: Random key
        
    Returns:
        mean: Mean prediction
        std: Standard deviation (epistemic uncertainty)
    """
    keys = jax.random.split(key, n_samples)
    
    # Check if batch
    is_batch = x.ndim > 1
    
    if is_batch:
        # Multiple MC samples for each input in batch
        predictions = jnp.stack([
            jax.vmap(lambda xi: model(xi, key=k))(x)
            for k in keys
        ])  # (n_samples, batch_size, output_dim)
    else:
        # Multiple MC samples for single input
        predictions = jnp.stack([
            model(x, key=k) for k in keys
        ])  # (n_samples, output_dim)
    
    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0)
    
    return mean, std


# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    
    # Create model with dropout
    model = MLPWithDropout(
        in_size=5,
        out_size=1,
        width=64,
        depth=3,
        dropout_rate=0.2,
        key=key
    )
    
    # Train model (with dropout active)
    # ... training code similar to ensemble ...
    
    # Predict with MC dropout
    x_test = jax.random.normal(key, (5,))
    mean, std = mc_dropout_predict(model, x_test, n_samples=100, key=key)
    
    print(f"MC Dropout Prediction: {mean[0]:.3f} ± {std[0]:.3f}")
```

### Ensemble vs MC Dropout

| Aspect | Ensemble | MC Dropout |
|--------|----------|------------|
| **Training cost** | High (train $M$ models) | Low (train 1 model) |
| **Inference cost** | Low (1 forward pass/model) | Medium ($T$ forward passes) |
| **Uncertainty type** | Epistemic | Epistemic (approximate) |
| **Theoretical foundation** | Bootstrap / Bayesian model averaging | Variational inference |
| **Accuracy** | Generally better | Good approximation |
| **Memory** | High ($M$ models in memory) | Low (1 model) |

**Recommendation**:
- **Ensemble**: When you have compute budget and want best uncertainty estimates
- **MC Dropout**: When you need fast uncertainty at inference time

**For your final project**:
- Use ensembles (5-10 models) for primary results
- Can compare with MC dropout as ablation study

---

## 🔴 Part 4: Calibration - Are Your Error Bars Honest?

### The Calibration Problem

**Well-calibrated uncertainty**:
- If you predict $y = 10 \pm 2$ (95% interval: [6, 14])
- Then 95% of true values should fall in [6, 14]

**Poorly calibrated**:
- **Overconfident**: Error bars too small (true values often outside)
- **Underconfident**: Error bars too large (true values always inside)

```{admonition} The Scientific Imperative
:class: important

**Calibrated uncertainty is essential for science:**

If your error bars are wrong:
- ❌ Can't properly weight observations
- ❌ Can't combine predictions with other measurements
- ❌ Can't make reliable scientific conclusions
- ❌ Can't know when to trust predictions

**Example**: If you claim 95% confidence intervals but only 60% of true values fall inside, you're systematically overconfident. This breaks downstream inference!
```

### Calibration Metrics

**Expected Calibration Error (ECE)**:
1. Sort predictions by confidence
2. Bin into $K$ bins (e.g., $K=10$)
3. For each bin: compare predicted confidence vs empirical accuracy
4. Weighted average of differences

**Practical calibration for regression**:

For each prediction $\hat{y}_i \pm \sigma_i$, compute standardized residual:
$$
z_i = \frac{y_i - \hat{y}_i}{\sigma_i}
$$

**If well-calibrated**: $z_i \sim \mathcal{N}(0, 1)$

**Tests**:
- Mean: $\bar{z} \approx 0$ (no systematic bias)
- Std: $\text{std}(z) \approx 1$ (correct variance)
- Shapiro-Wilk: Test if $z$ is normally distributed
- QQ-plot: Visual check of normality

### Implementation

```python
import jax.numpy as jnp
from scipy import stats

def compute_calibration_metrics(y_true, y_pred, y_std):
    """Compute calibration metrics for regression
    
    Args:
        y_true: True values (N,)
        y_pred: Predicted means (N,)
        y_std: Predicted standard deviations (N,)
        
    Returns:
        metrics: Dictionary of calibration metrics
    """
    # Standardized residuals
    z = (y_true - y_pred) / y_std
    
    # Basic statistics
    mean_z = float(jnp.mean(z))
    std_z = float(jnp.std(z))
    
    # Coverage at different confidence levels
    coverage_68 = float(jnp.mean(jnp.abs(z) < 1.0))  # 68% (1 sigma)
    coverage_95 = float(jnp.mean(jnp.abs(z) < 1.96))  # 95% (2 sigma)
    coverage_99 = float(jnp.mean(jnp.abs(z) < 2.58))  # 99% (3 sigma)
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(z)
    
    return {
        "mean_z": mean_z,  # Should be ~0
        "std_z": std_z,    # Should be ~1
        "coverage_68": coverage_68,  # Should be ~0.68
        "coverage_95": coverage_95,  # Should be ~0.95
        "coverage_99": coverage_99,  # Should be ~0.99
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_p_value": float(shapiro_p),
        "is_normal": shapiro_p > 0.05
    }


def plot_calibration(y_true, y_pred, y_std):
    """Visualize calibration
    
    Creates:
    1. Reliability diagram (predicted vs empirical coverage)
    2. QQ-plot (standardized residuals vs normal)
    3. Histogram of standardized residuals
    """
    import matplotlib.pyplot as plt
    
    z = (y_true - y_pred) / y_std
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Reliability diagram
    confidence_levels = jnp.linspace(0, 3, 30)  # 0 to 3 sigma
    empirical_coverage = []
    
    for conf in confidence_levels:
        coverage = jnp.mean(jnp.abs(z) < conf)
        empirical_coverage.append(coverage)
    
    # Theoretical coverage for normal distribution
    theoretical_coverage = 2 * stats.norm.cdf(confidence_levels) - 1
    
    axes[0].plot(confidence_levels, empirical_coverage, 'b-', label='Empirical')
    axes[0].plot(confidence_levels, theoretical_coverage, 'r--', label='Ideal (Normal)')
    axes[0].set_xlabel('Confidence Level (σ)')
    axes[0].set_ylabel('Empirical Coverage')
    axes[0].set_title('Reliability Diagram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. QQ-plot
    stats.probplot(z, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Histogram
    axes[2].hist(z, bins=30, density=True, alpha=0.7, edgecolor='black')
    x_range = jnp.linspace(-4, 4, 100)
    axes[2].plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2, label='N(0,1)')
    axes[2].set_xlabel('Standardized Residuals (z)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Distribution of Standardized Residuals')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    key = jax.random.PRNGKey(42)
    n_test = 200
    
    y_true = jax.random.normal(key, (n_test,))
    y_pred = y_true + 0.1 * jax.random.normal(key, (n_test,))  # Small bias
    y_std = 1.0 + 0.2 * jax.random.uniform(key, (n_test,))  # Varying uncertainty
    
    # Compute metrics
    metrics = compute_calibration_metrics(y_true, y_pred, y_std)
    
    print("Calibration Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Visualize
    fig = plot_calibration(y_true, y_pred, y_std)
    plt.show()
```

### Calibration Strategies

If your model is **overconfident** (error bars too small):

1. **Temperature scaling**: Scale uncertainties by factor $T > 1$
   $$
   \sigma_{\text{calibrated}} = T \cdot \sigma_{\text{predicted}}
   $$
   Choose $T$ on validation set to match empirical coverage

2. **Train with calibration loss**:
   $$
   \mathcal{L} = \text{MSE} + \lambda \left(\frac{1}{N}\sum_i z_i^2 - 1\right)^2
   $$
   Penalizes deviation from unit variance of standardized residuals

3. **Increase ensemble size**: More models → better uncertainty estimates

If your model is **underconfident** (error bars too large):

1. **Reduce regularization**: Less dropout, smaller weight decay
2. **Use smaller ensemble**: Fewer models may give tighter estimates
3. **Check for dataset shift**: Validation set may be harder than test set

---

## 🟡 Part 5: Comparing GP and NN Uncertainty

### Side-by-Side Comparison

Let's predict core radius and compare uncertainties:

```python
import jax
import jax.numpy as jnp

def compare_gp_nn_uncertainty(
    X_test,
    y_test,
    gp_model,
    nn_ensemble
):
    """Compare GP vs NN uncertainty on same test set
    
    Args:
        X_test: Test inputs (N, d)
        y_test: Test outputs (N,)
        gp_model: Trained GP (from Part 2)
        nn_ensemble: Trained NN ensemble (from this part)
        
    Returns:
        comparison: Dictionary with metrics
    """
    # GP predictions
    gp_mean, gp_std = gp_model.predict(X_test)
    
    # NN predictions
    nn_mean, nn_std = nn_ensemble.predict_with_uncertainty(X_test)
    
    # Compute metrics for both
    gp_metrics = compute_calibration_metrics(y_test, gp_mean, gp_std)
    nn_metrics = compute_calibration_metrics(y_test, nn_mean, nn_std)
    
    # Prediction accuracy
    gp_rmse = float(jnp.sqrt(jnp.mean((y_test - gp_mean)**2)))
    nn_rmse = float(jnp.sqrt(jnp.mean((y_test - nn_mean)**2)))
    
    # Average uncertainty
    gp_avg_unc = float(jnp.mean(gp_std))
    nn_avg_unc = float(jnp.mean(nn_std))
    
    # Negative log-likelihood (accounts for both accuracy and calibration)
    gp_nll = float(jnp.mean(0.5 * ((y_test - gp_mean) / gp_std)**2 + jnp.log(gp_std)))
    nn_nll = float(jnp.mean(0.5 * ((y_test - nn_mean) / nn_std)**2 + jnp.log(nn_std)))
    
    return {
        "gp": {
            "rmse": gp_rmse,
            "avg_uncertainty": gp_avg_unc,
            "nll": gp_nll,
            "calibration": gp_metrics
        },
        "nn": {
            "rmse": nn_rmse,
            "avg_uncertainty": nn_avg_unc,
            "nll": nn_nll,
            "calibration": nn_metrics
        }
    }


# Example output (hypothetical):
"""
Comparison Results:

GP:
  RMSE: 0.15 pc
  Avg Uncertainty: 0.12 pc
  Negative Log-Likelihood: -0.23
  Calibration:
    mean_z: 0.02 (excellent!)
    std_z: 0.98 (excellent!)
    coverage_95: 0.94 (near ideal)

NN Ensemble:
  RMSE: 0.12 pc (better!)
  Avg Uncertainty: 0.18 pc (larger)
  Negative Log-Likelihood: -0.15 (worse)
  Calibration:
    mean_z: -0.05 (small bias)
    std_z: 1.15 (slightly overconfident)
    coverage_95: 0.89 (undercovers!)

Interpretation:
- NN is more accurate (lower RMSE)
- But NN uncertainty is less calibrated
- GP gives more honest error bars
- For science, might prefer GP despite lower accuracy!
"""
```

### Key Insights for Your Final Project

**Gaussian Processes**:
- ✅ Automatic uncertainty (no ensembles needed)
- ✅ Well-calibrated by construction (Bayesian posterior)
- ✅ Uncertainty increases far from training data (honest extrapolation)
- ⚠️ Only works for scalar outputs (your Week 1 results)

**Neural Network Ensembles**:
- ✅ Can handle time series (your Week 3-4 results)
- ✅ Often more accurate predictions (lower RMSE)
- ⚠️ Uncertainty may be miscalibrated (need to check!)
- ⚠️ Requires training multiple models (computational cost)
- ⚠️ May be overconfident in extrapolation

**Your Comparative Analysis Should Include**:
1. **Accuracy**: RMSE, MAE, R² for both methods
2. **Calibration**: Standardized residuals, coverage, reliability diagrams
3. **Computational cost**: Training time, inference time
4. **Interpretability**: GP lengthscales vs NN black box
5. **Use case recommendation**: When to use each method

---

## 🔴 Part 6: Uncertainty-Aware Training

### Incorporating Uncertainty in Loss Functions

**Standard loss** (MSE):
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**Uncertainty-aware loss** (negative log-likelihood):
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[\frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} + \log \sigma_i\right]
$$

**Interpretation**:
- First term: Penalizes prediction error (weighted by uncertainty)
- Second term: Penalizes large uncertainties (prevents trivial solution $\sigma_i \to \infty$)

**Why this works**: This is the negative log-likelihood under Gaussian noise!

### Predicting Aleatoric Uncertainty

Train network to output **both mean and variance**:
$$
(\mu, \sigma^2) = f_{\boldsymbol{\theta}}(\mathbf{x})
$$

**Architecture**:
```python
class NetworkWithUncertainty(eqx.Module):
    """Network that predicts mean and aleatoric uncertainty"""
    shared_layers: eqx.nn.MLP
    mean_head: eqx.nn.Linear
    log_var_head: eqx.nn.Linear  # Predict log(σ²) for numerical stability
    
    def __init__(self, in_size, hidden_size, out_size, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Shared feature extractor
        self.shared_layers = eqx.nn.MLP(
            in_size=in_size,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=3,
            activation=jax.nn.relu,
            key=key1
        )
        
        # Separate heads for mean and variance
        self.mean_head = eqx.nn.Linear(hidden_size, out_size, key=key2)
        self.log_var_head = eqx.nn.Linear(hidden_size, out_size, key=key3)
    
    def __call__(self, x):
        """Predict mean and variance
        
        Returns:
            mean: Predicted mean (out_size,)
            var: Predicted variance (out_size,)
        """
        features = self.shared_layers(x)
        
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        var = jnp.exp(log_var)  # Ensure positive
        
        return mean, var


def gaussian_nll_loss(model, x, y):
    """Negative log-likelihood loss
    
    Args:
        model: NetworkWithUncertainty
        x: Inputs (batch_size, in_size)
        y: Targets (batch_size, out_size)
    """
    def single_loss(xi, yi):
        mean, var = model(xi)
        # NLL = 0.5 * log(2π) + 0.5 * log(var) + 0.5 * (y - mean)^2 / var
        # Omitting constant 0.5 * log(2π)
        nll = 0.5 * jnp.log(var) + 0.5 * (yi - mean)**2 / var
        return jnp.sum(nll)  # Sum over output dimensions
    
    return jnp.mean(jax.vmap(single_loss)(x, y))
```

**Now your network outputs**:
- $\mu$: Best prediction (epistemic captured by ensembles)
- $\sigma^2$: Aleatoric uncertainty (data noise, irreducible)

**Total uncertainty** (for ensemble):
$$
\sigma_{\text{total}}^2 = \underbrace{\sigma_{\text{epistemic}}^2}_{\text{ensemble variance}} + \underbrace{\bar{\sigma}_{\text{aleatoric}}^2}_{\text{mean predicted variance}}
$$

---

## 🟡 Part 7: Practical Considerations for Your Final Project

### Computational Budget vs Uncertainty Quality

**Ensemble sizes**:
- $M = 5$: Minimum for reasonable uncertainty
- $M = 10$: Good balance (recommended)
- $M = 20$: Diminishing returns
- $M > 50$: Rarely worth it

**Rule of thumb**: Uncertainty converges as $\sim 1/\sqrt{M}$

**MC Dropout samples**:
- $T = 50$: Fast, rough uncertainty
- $T = 100$: Balanced (recommended)
- $T = 500$: High quality, slower

### When to Use Which Method

**Use Ensembles when**:
- ✅ You have computational resources
- ✅ You need best uncertainty estimates
- ✅ Final production model (worth the cost)
- ✅ Your final project report (primary method)

**Use MC Dropout when**:
- ✅ Quick prototyping
- ✅ Limited memory (can't store many models)
- ✅ Real-time applications
- ✅ Ablation study in your project

**Use Predicted Variance when**:
- ✅ Aleatoric uncertainty matters (e.g., noisy simulations)
- ✅ Combined with ensembles for total uncertainty
- ✅ You want to learn spatially-varying noise

### Reporting Uncertainty in Your Final Project

**For each method (GP, MLP, GRU, Neural ODE), report**:

1. **Prediction accuracy**: RMSE, MAE, R²
2. **Calibration metrics**: Mean/std of standardized residuals, coverage
3. **Uncertainty statistics**: Mean uncertainty, min/max uncertainty
4. **Visualizations**:
   - Predictions with error bars vs true values
   - Reliability diagrams
   - Uncertainty vs distance to training data

**Example visualization**:
```python
def plot_predictions_with_uncertainty(y_true, y_pred, y_std, title="Predictions"):
    """Plot predictions with error bars"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Predicted vs True
    axes[0].errorbar(y_true, y_pred, yerr=2*y_std, fmt='o', alpha=0.5, capsize=3)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', label='Perfect')
    axes[0].set_xlabel('True Value')
    axes[0].set_ylabel('Predicted Value')
    axes[0].set_title(f'{title}: Predictions with 95% CI')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals vs Uncertainty
    residuals = jnp.abs(y_true - y_pred)
    axes[1].scatter(y_std, residuals, alpha=0.5)
    axes[1].plot([0, y_std.max()], [0, 2*y_std.max()], 'r--', label='2σ line')
    axes[1].set_xlabel('Predicted Uncertainty (σ)')
    axes[1].set_ylabel('Absolute Residual')
    axes[1].set_title('Uncertainty vs Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

---

## 🔴 Part 8: Advanced Topics

### Conformal Prediction

**Problem**: Ensembles give uncertainty, but not guaranteed coverage

**Solution**: Conformal prediction provides **distribution-free** coverage guarantees

**Key idea**:
1. Compute residuals on calibration set: $|y_i - \hat{y}_i|$
2. Find quantile $q_\alpha$ such that $\alpha$ fraction of residuals are below $q_\alpha$
3. For new prediction: $\hat{y} \pm q_\alpha$ has coverage $\geq 1-\alpha$

**Advantage**: Works for **any** model (even black box!)

```python
def conformal_prediction_interval(y_cal, y_pred_cal, y_pred_test, alpha=0.05):
    """Compute conformal prediction intervals
    
    Args:
        y_cal: True values on calibration set (N_cal,)
        y_pred_cal: Predictions on calibration set (N_cal,)
        y_pred_test: Predictions on test set (N_test,)
        alpha: Desired error rate (0.05 for 95% intervals)
        
    Returns:
        lower: Lower bounds (N_test,)
        upper: Upper bounds (N_test,)
    """
    # Compute absolute residuals on calibration set
    residuals = jnp.abs(y_cal - y_pred_cal)
    
    # Find (1-alpha) quantile
    n = len(residuals)
    q_level = jnp.ceil((n + 1) * (1 - alpha)) / n
    q = jnp.quantile(residuals, q_level)
    
    # Prediction intervals
    lower = y_pred_test - q
    upper = y_pred_test + q
    
    return lower, upper
```

**When to use**: When you need guaranteed coverage (e.g., safety-critical applications)

### Bayesian Neural Networks

**Full Bayesian approach**: Put prior on **all weights**
$$
p(\mathbf{W} \,|\, \mathcal{D}) \propto p(\mathcal{D} \,|\, \mathbf{W}) p(\mathbf{W})
$$

**Challenge**: Posterior is intractable (millions of parameters!)

**Solutions**:
1. **Variational inference**: Approximate posterior with tractable family
2. **MCMC**: Sample weights (expensive!)
3. **Laplace approximation**: Gaussian approximation around MAP estimate

**Practical note**: These are advanced techniques beyond this course, but worth knowing they exist.

---

## Conceptual Checkpoints

1. **Epistemic vs Aleatoric**: Your GP predicts $r_{\text{core}} = 1.0 \pm 0.2$ pc. You collect 1000 more training simulations. Will the uncertainty decrease? Which type of uncertainty does GP primarily capture?

2. **Ensemble Size**: You train ensembles of size $M = 2, 5, 10, 20, 50$. Plot (mentally) how uncertainty estimates change. At what point do you see diminishing returns?

3. **Calibration**: Your NN ensemble predicts 95% confidence intervals, but when you check, only 70% of true values fall inside. Is your model overconfident or underconfident? How would you fix it?

4. **Extrapolation**: You train on $N \in [100, 1000]$ and test on $N = 5000$. Your GP gives $r_{\text{core}} = 1.0 \pm 2.0$ pc (huge uncertainty). Your NN ensemble gives $r_{\text{core}} = 0.8 \pm 0.1$ pc (tight uncertainty). Which should you trust?

5. **MC Dropout**: Explain why keeping dropout active at test time approximates Bayesian inference. How is this different from standard dropout as regularization?

6. **Comparison**: For your final project, you need to compare GP vs NN fairly. What metrics should you use? Why isn't RMSE alone sufficient?

7. **Connection to Module 5**: How is ensemble uncertainty related to the posterior distribution $p(\theta \,|\, \mathcal{D})$ from Bayesian inference? What role do different initializations play?

---

## Summary and Looking Forward

### What You've Learned

You now understand:
- ✅ **Why** uncertainty matters for science (trustworthiness, data collection)
- ✅ **Two types**: Epistemic (model) vs aleatoric (data noise)
- ✅ **Ensemble methods**: Train multiple models, capture disagreement
- ✅ **MC Dropout**: Fast Bayesian approximation
- ✅ **Calibration**: Checking if error bars are honest
- ✅ **GP vs NN**: Automatic vs approximate uncertainty
- ✅ **Practical implementation**: Complete JAX/Equinox code

### For Your Final Project

**Week 1-2 (GP)**:
- Train GP on scalar outputs ($r_{\text{core}}$ at fixed time)
- Get uncertainty for free (posterior variance)
- Observe: Well-calibrated, honest error bars
- Document: Uncertainty increases far from training data

**Week 3-4 (NN)**:
- Train NN ensembles on time series
- Implement uncertainty quantification
- Check calibration (reliability diagrams)
- Compare: GP vs NN uncertainty

**Your Report Should Include**:

1. **Uncertainty Analysis**:
   - Calibration plots for all methods
   - Coverage statistics (68%, 95%, 99%)
   - Discussion of when predictions are trustworthy

2. **Method Comparison**:
   - Not just "NN is 10% more accurate"
   - But "NN is more accurate but GP is better calibrated"
   - Trade-offs: accuracy vs uncertainty quality

3. **Scientific Recommendations**:
   - When to use GP (scalar outputs, need calibration)
   - When to use NN (time series, can validate carefully)
   - How to combine (GP for scalars, NN for dynamics)

---

```{admonition} Final Thought: Honest Uncertainty in Science
:class: important

**The difference between prediction and science:**

- **Prediction**: "The answer is 1.23"
- **Science**: "The answer is 1.23 ± 0.15, calibrated on held-out data"

**Why this matters**:

In engineering, being wrong is costly.
In science, being overconfident is fatal.

If you claim narrow error bars that don't reflect true uncertainty:
- Other scientists can't properly combine your results with theirs
- Downstream inference (parameter estimation, model selection) breaks
- You can't identify where more data is needed

**Your final project demonstrates this principle**:
- GPs give honest uncertainty (sometimes large!)
- NNs can be more accurate but less calibrated
- The choice depends on scientific goals

**As computational astrophysicists, we have a responsibility**:

Not just to predict accurately, but to quantify our ignorance honestly.

That's what makes machine learning *scientific* machine learning. 🌌
```
