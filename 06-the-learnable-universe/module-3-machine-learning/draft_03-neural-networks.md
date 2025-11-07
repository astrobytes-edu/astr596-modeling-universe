# Module 6, Part 3: Neural Networks for Physical Dynamics

> "The question is not whether intelligent machines can have emotions, but whether machines can be intelligent without emotions."
> 
> — Marvin Minsky

> "A neural network is a differentiable program. The key insight is that we can learn the program from data."
> 
> — Yann LeCun

---

## Learning Objectives

By the end of this module, you will be able to:

1. **LO1 - Fundamental Understanding**: Explain neural networks as universal function approximators and understand how they differ from Gaussian Processes in expressivity and uncertainty
2. **LO2 - Architecture Design**: Design appropriate neural network architectures for different physical problems (sequences vs continuous dynamics vs distributions)
3. **LO3 - Temporal Modeling**: Implement and compare methods for modeling time evolution: Neural ODEs, RNNs, and Transformers
4. **LO4 - Physics Integration**: Incorporate physical constraints into neural networks through architecture design, loss functions, and hybrid models
5. **LO5 - Training Strategies**: Train neural networks effectively using JAX/Equinox, understanding optimization challenges, hyperparameter tuning, and debugging strategies
6. **LO6 - Uncertainty Quantification**: Implement ensemble methods to recover uncertainty estimates that neural networks don't provide naturally
7. **LO7 - Comparative Analysis**: Evaluate trade-offs between GP and NN approaches for emulating star cluster evolution
8. **LO8 - Production ML**: Build production-quality ML models using modern software engineering practices (testing, validation, reproducibility)

---

## The Big Picture: From GPs to Neural Networks

### Where We Left Off: GP Limitations

In the previous module, you built Gaussian Process emulators for your star cluster simulations. GPs excelled at:

✅ **Scalar predictions** - Core radius at t=100 Myr given initial conditions  
✅ **Uncertainty quantification** - Built-in epistemic uncertainty  
✅ **Data efficiency** - Work well with 100-500 training simulations  
✅ **Interpretability** - Kernel hyperparameters reveal which ICs matter  

But GPs struggled with:

❌ **Time series** - 100 timesteps = 100-dimensional output space  
❌ **Computational scaling** - $O(n^3)$ training, $O(n^2)$ memory  
❌ **Extrapolation** - Revert to prior outside training data  
❌ **Non-smooth functions** - Kernel smoothness assumptions break  

**The fundamental issue**: GPs treat each output dimension independently (or with simple correlations). They don't understand **sequential structure** or **dynamical evolution**.

### What Neural Networks Bring

Neural networks offer complementary strengths:

✅ **High-dimensional outputs** - Natural for time series, images, distributions  
✅ **Scalability** - $O(n)$ per training step with mini-batching  
✅ **Flexibility** - Can learn arbitrary nonlinear patterns  
✅ **Sequential modeling** - RNNs, Transformers understand temporal structure  
✅ **Physics integration** - Can encode ODEs, conservation laws, symmetries  

But with trade-offs:

⚠️ **No built-in uncertainty** - Need ensembles or Bayesian approaches  
⚠️ **Data hungry** - Typically need 1000s+ of training examples  
⚠️ **Harder to train** - Local minima, hyperparameter sensitivity, overfitting  
⚠️ **Less interpretable** - "Black boxes" (though improving)  

```{admonition} The Philosophical Shift: Learning Structure vs Learning Functions
:class: important

**Gaussian Processes**: Learn a *distribution over functions* $f: \mathbb{R}^D \to \mathbb{R}$ directly. The prior (kernel) encodes smoothness, but no other structure.

**Neural Networks**: Learn the *structure* of the data-generating process:
- **RNNs**: Model sequential dependencies - $h_t = f(h_{t-1}, t)$
- **Neural ODEs**: Model continuous dynamics - $df/dt = g(f, t)$
- **Transformers**: Model attention patterns - "which past times matter?"

This is deeper! Instead of approximating $f$, you're learning the **mechanism** that generates $f$. This is closer to how physics works: we don't memorize solutions, we solve differential equations.

**The connection to your course**: In Modules 2-4, you solved ODEs (stellar structure, phase space evolution, radiative transfer). Neural networks let you *learn* those ODEs from data when analytical solutions don't exist!
```

### Your Project Arc: GP → Limitation → NN

**Week 1**: Built GP emulator  
- Predicted $r_{\text{core}}(t=100 \text{ Myr})$ from ICs  
- Achieved ~5-10% RMSE, calibrated uncertainty  
- Learned which ICs matter most (concentration > virial ratio > mass)  

**Week 2**: Pushed GPs to breaking point  
- Tried predicting full trajectory $r_{\text{core}}(t)$ for $t \in [0, 200]$ Myr  
- Multi-output GP: memory explosion ($50k \times 50k$ covariance matrix)  
- Time-as-input GP: poor temporal structure, extrapolation fails  
- **Conclusion**: Need methods designed for sequences!  

**Week 3-4**: Neural network solutions  
- Choose architecture based on physical problem structure  
- Implement from scratch in JAX/Equinox (glass-box philosophy)  
- Compare to GP: when to use each?  
- **Goal**: Fast, accurate emulator for full cluster evolution  

---

## 🔴 Part 1: Neural Network Foundations

Before diving into sophisticated architectures, let's build from first principles.

### The Perceptron: Building Block

A **perceptron** is the simplest neural unit:

$$
y = \phi\left( \sum_{i=1}^d w_i x_i + b \right) = \phi(\mathbf{w}^T \mathbf{x} + b)
$$

where:
- $\mathbf{x} \in \mathbb{R}^d$: Input features
- $\mathbf{w} \in \mathbb{R}^d$: Weights (learnable parameters)
- $b \in \mathbb{R}$: Bias (learnable offset)
- $\phi$: **Activation function** (nonlinearity)

**Without $\phi$**: Just linear regression! $y = \mathbf{w}^T \mathbf{x} + b$ can only fit hyperplanes.

**With $\phi$**: Can approximate nonlinear functions (if you stack many perceptrons).

**Common activation functions**:

1. **ReLU** (Rectified Linear Unit): $\phi(z) = \max(0, z)$
   - Most common in modern deep learning
   - Computationally cheap, good gradients
   - "Neurons" turn on/off (sparse activations)

2. **Tanh**: $\phi(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
   - Smooth, zero-centered
   - Gradients vanish for large $|z|$
   - Good for RNNs (historical reasons)

3. **Sigmoid**: $\phi(z) = \frac{1}{1 + e^{-z}}$
   - Outputs in $(0, 1)$ (good for probabilities)
   - Severe vanishing gradients
   - Rarely used in hidden layers

4. **Softplus**: $\phi(z) = \log(1 + e^z)$
   - Smooth approximation to ReLU
   - Useful when smoothness matters (e.g., Neural ODEs)

5. **GELU** (Gaussian Error Linear Unit): $\phi(z) = z \Phi(z)$ where $\Phi$ is Gaussian CDF
   - Used in Transformers (GPT, BERT)
   - Stochastic interpretation (dropout connection)

**For your project**: Start with ReLU (simple, effective). Try Softplus for Neural ODEs (smoothness helps ODE solvers). Experiment with others if interested.

```{admonition} Connection to Module 1: The Role of Nonlinearity
:class: note

Remember from Module 1: the Central Limit Theorem states that sums of independent random variables converge to Gaussians (linear combinations).

**Key insight**: If neural networks were just linear operations ($\mathbf{W}\mathbf{x}$), then no matter how many layers you stack:
$$y = \mathbf{W}_L \mathbf{W}_{L-1} \cdots \mathbf{W}_1 \mathbf{x} = \mathbf{W}_{\text{effective}} \mathbf{x}$$
Just one big linear transformation! Can only fit hyperplanes, same as linear regression.

**Activation functions** break this:
$$y = \mathbf{W}_L \phi(\mathbf{W}_{L-1} \phi(\cdots \phi(\mathbf{W}_1 \mathbf{x})))$$
Now you have *nonlinear* composition. This is what gives neural networks their power: they can approximate arbitrarily complex nonlinear functions.

**The Universal Approximation Theorem** (Cybenko 1989, Hornik 1991): A neural network with just *one hidden layer* and *enough neurons* can approximate any continuous function on a compact domain to arbitrary accuracy.

But: "Enough neurons" might be exponentially many! Deep networks (many layers, fewer neurons per layer) are more efficient.
```

### Multi-Layer Perceptrons (MLPs)

Stack perceptrons in layers:

**Input layer**: $\mathbf{x} \in \mathbb{R}^{d_0}$ (your data)

**Hidden layer 1**: 
$$\mathbf{h}_1 = \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^{d_1}$$

**Hidden layer 2**:
$$\mathbf{h}_2 = \phi(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \in \mathbb{R}^{d_2}$$

**...**

**Output layer**:
$$\mathbf{y} = \mathbf{W}_L \mathbf{h}_{L-1} + \mathbf{b}_L \in \mathbb{R}^{d_L}$$

(Often no activation on output for regression)

**Parameters**: $\theta = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \ldots, \mathbf{W}_L, \mathbf{b}_L\}$

**Forward pass**: Compute $\mathbf{y} = f_\theta(\mathbf{x})$ - just matrix multiplies and element-wise nonlinearities!

**What do hidden layers learn?**
- **Layer 1**: Simple features (edges, gradients in images; linear combinations in tabular data)
- **Layer 2**: Combinations of features (corners, textures; nonlinear interactions)
- **Layer L-1**: High-level abstractions (objects in images; physical regimes in scientific data)
- **Layer L**: Map abstractions to outputs

**Depth vs width trade-off**:
- **Deep (many layers, narrow)**: More expressive, harder to train, better for complex hierarchies
- **Wide (few layers, many neurons)**: Easier to train, more parameters, Universal Approximation with 1 layer
- **Modern consensus**: Depth is more important than width for most tasks

**For your project**: Start with 2-3 hidden layers of 64-128 neurons each. Adjust based on overfitting/underfitting.

### Training: Loss Functions and Backpropagation

**Goal**: Find parameters $\theta$ that minimize prediction error on training data:

$$
\theta^* = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \mathcal{L}(y_i, f_\theta(x_i)) + \lambda \mathcal{R}(\theta)
$$

where:
- $\mathcal{L}$: **Loss function** (measures error)
- $\mathcal{R}$: **Regularization** (prevents overfitting, often $L^2$ norm $\|\theta\|^2$)
- $\lambda$: Regularization strength

**Common loss functions**:

1. **Mean Squared Error (MSE)** - for regression:
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2$$
   - Penalizes large errors heavily (quadratic)
   - Assumes Gaussian noise
   - What you'll use for cluster evolution!

2. **Mean Absolute Error (MAE)** - robust regression:
   $$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^n |y_i - f_\theta(x_i)|$$
   - Less sensitive to outliers (linear penalty)
   - Harder to optimize (non-smooth at 0)

3. **Huber Loss** - compromise:
   $$\mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$
   - Quadratic for small errors, linear for large
   - Best of both worlds

4. **Log-likelihood** - probabilistic:
   $$\mathcal{L}_{\text{NLL}} = -\log p(y | f_\theta(x))$$
   - If you predict distribution parameters (e.g., $\mu_\theta(x), \sigma_\theta(x)$ for Gaussian)
   - Maximum likelihood estimation
   - Connects to Bayesian inference!

**Backpropagation**: Compute gradients $\nabla_\theta \mathcal{L}$ efficiently via chain rule.

**Manual derivation** (one hidden layer example):

$$
\mathcal{L} = (y - \mathbf{w}_2^T \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))^2
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = -2(y - \hat{y}) \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = -2(y - \hat{y}) \mathbf{w}_2^T \circ \phi'(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \mathbf{x}^T
$$

where $\circ$ is element-wise product (Hadamard).

**With JAX**: You never compute these by hand!

```python
import jax
import jax.numpy as jnp

def loss(params, x, y):
    y_pred = network(params, x)
    return jnp.mean((y - y_pred)**2)

# Automatic differentiation!
grad_loss = jax.grad(loss, argnums=0)  # Gradient w.r.t. params

# Usage in training loop
grads = grad_loss(params, x_batch, y_batch)
```

**This is the magic of JAX**: Backpropagation is automatic. You just write the forward pass, JAX handles the rest.

```{admonition} Connection to Module 5: Maximum Likelihood Estimation
:class: note

Training a neural network with MSE loss is *exactly* maximum likelihood estimation under Gaussian noise!

**Probabilistic interpretation**:
$$p(y | x, \theta) = \mathcal{N}(y | f_\theta(x), \sigma^2)$$

**Negative log-likelihood**:
$$-\log p(y | x, \theta) = \frac{1}{2\sigma^2}(y - f_\theta(x))^2 + \text{const}$$

**Minimizing NLL** = **Minimizing MSE** (ignoring constant and $\sigma^2$ scale).

So when you train a NN with MSE, you're implicitly assuming:
1. Your model outputs the mean of a Gaussian
2. The variance is constant (homoscedastic noise)
3. Observations are independent

**Extension**: Predict both $\mu_\theta(x)$ and $\sigma_\theta(x)$ (heteroscedastic noise):
$$\mathcal{L} = -\log \mathcal{N}(y | \mu_\theta(x), \sigma_\theta^2(x)) = \frac{(y - \mu_\theta(x))^2}{2\sigma_\theta^2(x)} + \frac{1}{2}\log \sigma_\theta^2(x)$$

This penalizes both inaccuracy and overconfidence! This is one way to get uncertainty from NNs.
```

### Optimization: Gradient Descent and Variants

**Vanilla Gradient Descent**:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the learning rate.

**Problem**: Slow, gets stuck in local minima, sensitive to $\eta$.

**Stochastic Gradient Descent (SGD)**:
Instead of using full dataset, use random **mini-batch**:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{batch}}(\theta_t)$$

**Advantages**:
- Much faster per step (don't need full dataset)
- Stochasticity helps escape local minima
- Scales to huge datasets (millions of examples)

**Disadvantage**: Noisy gradients, needs learning rate tuning.

**Modern optimizers** (all available in Optax):

1. **SGD with Momentum**:
   $$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla_\theta \mathcal{L}_t$$
   $$\theta_{t+1} = \theta_t - \eta \mathbf{v}_{t+1}$$
   - Accumulates velocity (exponential moving average of gradients)
   - Dampens oscillations, accelerates in consistent directions
   - $\beta \approx 0.9$ typical

2. **Adam** (Adaptive Moment Estimation):
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}_t$$
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}_t)^2$$
   $$\theta_{t+1} = \theta_t - \eta \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}$$
   - Adapts learning rate per parameter
   - Very robust, minimal tuning needed
   - **Default choice** for most problems
   - $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ typical

3. **AdamW** (Adam with Weight Decay):
   - Adam + proper $L^2$ regularization
   - Better generalization than Adam
   - **Recommended** for production

4. **Learning Rate Schedules**:
   - **Constant**: $\eta_t = \eta_0$ (simple, but often suboptimal)
   - **Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$ (drop every $k$ steps)
   - **Exponential**: $\eta_t = \eta_0 e^{-\lambda t}$
   - **Cosine annealing**: $\eta_t = \eta_0 \cos\left(\frac{\pi t}{T}\right)$ (smooth decay)
   - **Warm-up + decay**: Start small, increase, then decay (very effective!)

**For your project**: Use **AdamW** with **cosine annealing**. This is the modern best practice and requires minimal tuning.

```python
import optax

# Learning rate schedule: warm-up + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)

# Optimizer
optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)

# Initialize
opt_state = optimizer.init(params)

# Training step
grads = jax.grad(loss_fn)(params, batch)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

### Overfitting, Underfitting, and Regularization

**The fundamental trade-off**:

- **Underfitting**: Model too simple, high training error, high test error
  - Not enough capacity to capture patterns
  - Solution: More layers, more neurons, train longer

- **Overfitting**: Model too complex, low training error, high test error
  - Memorizes training data, doesn't generalize
  - Solution: Regularization!

**Regularization techniques**:

1. **Weight Decay** ($L^2$ regularization):
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_i \theta_i^2$$
   - Penalizes large weights
   - Encourages simpler functions (smoother)
   - $\lambda \sim 10^{-4}$ typical

2. **Dropout**:
   - During training: Randomly set neurons to 0 with probability $p$ (typically 0.1-0.5)
   - During testing: Use all neurons, scale by $(1-p)$
   - Forces network to be robust (can't rely on specific neurons)
   - Equivalent to training an ensemble!

3. **Early Stopping**:
   - Monitor validation loss during training
   - Stop when validation loss stops improving
   - Simple, effective, always use it!

4. **Data Augmentation**:
   - Create variations of training data (rotate, scale, add noise)
   - Increases effective dataset size
   - Especially important for small datasets

5. **Batch Normalization**:
   - Normalize activations within each mini-batch
   - Reduces internal covariate shift
   - Acts as regularization (noise from batch statistics)
   - Makes training more stable

**For your project**: Use weight decay (via AdamW), early stopping, and potentially dropout. Batch norm less critical for small MLPs but important for deep networks.

```{admonition} The More You Know: Why Does Dropout Work?
:class: tip, dropdown

**Intuition**: Dropout trains an *ensemble* of neural networks!

Each training step, you're randomly selecting a subset of neurons (a "thinned network"). Over many steps, you train many different sub-networks that share parameters.

At test time, using all neurons approximates *averaging* over all possible sub-networks. This is ensemble learning!

**Theoretical connection**: Dropout approximates a Bayesian neural network where weights have Gaussian priors. The uncertainty in which neurons fire corresponds to posterior uncertainty over weights.

**Practical benefit**: One network trained with dropout ≈ ensemble of many networks → better generalization + implicit uncertainty quantification.

For your project, if you use dropout in an ensemble (train multiple networks with dropout), you get *nested ensembles* - even better uncertainty estimates!
```

---

## 🔴 Part 2: Recurrent Neural Networks for Sequences

Now we move beyond feedforward networks to architectures designed for **sequences**: time series, text, trajectories.

### The Sequential Problem

Your star cluster evolution is a **sequence**:
$$r_{\text{core}}(t_0), r_{\text{core}}(t_1), r_{\text{core}}(t_2), \ldots, r_{\text{core}}(t_T)$$

**Key properties**:
1. **Temporal dependencies**: $r(t+1)$ depends on $r(t)$, not independent
2. **Variable length**: Different simulations might run for different durations
3. **Causal structure**: Future doesn't influence past

**Feedforward MLPs** ignore this structure:
- Treat each timestep independently
- Fixed input/output dimensions (can't handle variable length)
- No memory of past states

**Recurrent Neural Networks** (RNNs) are designed for sequences.

### Vanilla RNN Architecture

**Idea**: Maintain a **hidden state** $\mathbf{h}_t$ that summarizes past information.

**Recurrence relation**:
$$
\mathbf{h}_t = \phi(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

**Parameters**:
- $\mathbf{W}_{hh}$: Hidden-to-hidden (recurrence weights)
- $\mathbf{W}_{xh}$: Input-to-hidden
- $\mathbf{W}_{hy}$: Hidden-to-output
- $\mathbf{b}_h, \mathbf{b}_y$: Biases

**Key features**:
- Same weights $\mathbf{W}$ applied at every timestep (parameter sharing)
- Hidden state $\mathbf{h}_t$ carries information from all previous timesteps
- Can process variable-length sequences
- Output at each timestep (or just final output)

**For cluster evolution**:
- Input: Initial conditions $\mathbf{x}$ + current time $t$
- Hidden state: Compressed representation of trajectory so far
- Output: $r_{\text{core}}(t)$, $\sigma_v(t)$, $f_{\text{bound}}(t)$

**Unrolling through time**:
```
x_0 → [RNN] → h_0 → y_0
      ↓
x_1 → [RNN] → h_1 → y_1
      ↓
x_2 → [RNN] → h_2 → y_2
      ↓
      ...
```

Each $[RNN]$ box uses the *same* parameters!

### Training RNNs: Backpropagation Through Time (BPTT)

**Forward pass**: Unroll RNN for $T$ steps, compute loss:
$$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(y_t, \hat{y}_t)$$

**Backward pass**: Compute gradients by chain rule through time:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}}$$

**Problem**: Gradients flow backward through $T$ steps:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_0} = \prod_{t=1}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \prod_{t=1}^T \mathbf{W}_{hh}^T \text{diag}(\phi'(\cdot))$$

If eigenvalues of $\mathbf{W}_{hh}$ are:
- **< 1**: Product decays exponentially → **vanishing gradients** (can't learn long-term dependencies)
- **> 1**: Product explodes exponentially → **exploding gradients** (unstable training)

**Vanishing gradients**: Most severe problem. After 10-20 steps, gradient becomes negligible. RNN can't learn dependencies longer than ~10 timesteps.

**Solution 1: Gradient clipping** (for exploding)
```python
# Clip gradient norms
grads = jax.grad(loss_fn)(params, batch)
grads, _ = optax.clip_by_global_norm(max_norm=1.0)(grads, None)
```

**Solution 2: Better architectures** (for vanishing) → **LSTM** and **GRU**!

```{admonition} Connection to Module 3: Phase Space Evolution
:class: note

RNN hidden states $\mathbf{h}_t$ are analogous to **phase space coordinates** from Module 3!

**Phase space**: $(q, p)$ fully describes system state, evolves according to Hamiltonian: $\dot{q} = \partial H/\partial p$, $\dot{p} = -\partial H/\partial q$

**RNN hidden state**: $\mathbf{h}_t$ fully describes history, evolves according to learned dynamics: $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$

**Key parallel**:
- Phase space: Compact representation of system (6N dimensions for N particles)
- Hidden state: Compact representation of sequence (often 64-512 dimensions)

Both obey **Markov property**: Given current state, future independent of past. $\mathbf{h}_t$ is sufficient statistic for prediction!

**Difference**: Phase space is constructed from physics (position + momentum). Hidden state is *learned* from data. The RNN discovers an effective "phase space" for cluster evolution!
```

### LSTM (Long Short-Term Memory)

**Motivation**: Solve vanishing gradients by adding **gating mechanisms** that control information flow.

**Architecture**: Replace simple hidden state with **cell state** $\mathbf{c}_t$ and **hidden state** $\mathbf{h}_t$.

**Gates** (all element-wise, values in $[0, 1]$ via sigmoid):

1. **Forget gate**: What to remove from memory?
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

2. **Input gate**: What new information to add?
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

3. **Cell update**: Candidate new information
   $$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

4. **Output gate**: What to output from memory?
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**State updates**:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\odot$ is element-wise product.

**Why this works**:
- Cell state $\mathbf{c}_t$ has **linear path** backward through time (just element-wise multiplies)
- Gradients flow easily through cell state (no vanishing!)
- Gates learn *what* to remember/forget/output adaptively
- Can capture dependencies over 100s of timesteps

**Intuition**: LSTM is like a computer with memory (cell state) and control logic (gates). Gates are learned programs!

### GRU (Gated Recurrent Unit)

**Motivation**: LSTM works great but has many parameters (4 weight matrices). Can we simplify?

**Architecture**: Combine cell and hidden state, fewer gates.

**Gates**:

1. **Update gate**: How much to update hidden state?
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

2. **Reset gate**: How much to forget previous hidden state?
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**State update**:
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**Interpolation**: $\mathbf{h}_t$ is weighted average of old state $\mathbf{h}_{t-1}$ and new state $\tilde{\mathbf{h}}_t$.

**Advantages over LSTM**:
- Fewer parameters (faster training, less overfitting)
- Often similar performance to LSTM
- Simpler implementation

**Disadvantages**:
- Slightly less expressive
- No separate cell state (might limit very long-term memory)

**Consensus**: Try both! GRU is a good default (simpler), use LSTM if you need maximum capacity.

### RNN Architectures for Cluster Evolution

**Three paradigms**:

1. **Many-to-One**: Sequence → Single prediction
   - Input: Full trajectory ICs + times $[t_1, \ldots, t_T]$
   - Output: Final state $r_{\text{core}}(t_T)$
   - Use case: "Will cluster survive?" or "Final mass?"

2. **Many-to-Many (Synced)**: Sequence → Sequence (same length)
   - Input: ICs + times $[t_1, \ldots, t_T]$
   - Output: Full trajectory $[r_1, \ldots, r_T]$
   - **This is what you'll use!** Predict evolution at each timestep
   - Loss: MSE summed over all timesteps

3. **Many-to-Many (Async)**: Encoder-Decoder
   - Encoder RNN: Compress input sequence → fixed vector
   - Decoder RNN: Expand vector → output sequence
   - Use case: Variable length input/output (e.g., summarization)

**Implementation for your project**:

```python
import equinox as eqx
import jax.numpy as jnp

class ClusterRNN(eqx.Module):
    """RNN for star cluster evolution prediction."""
    
    # Encoder: ICs → initial hidden state
    encoder: eqx.nn.Linear
    
    # RNN cell: h_t, x_t → h_{t+1}
    rnn_cell: eqx.nn.GRUCell  # or LSTMCell
    
    # Decoder: h_t → outputs
    decoder: eqx.nn.Linear
    
    hidden_size: int
    
    def __init__(self, input_size, output_size, hidden_size, *, key):
        keys = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        
        self.encoder = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.rnn_cell = eqx.nn.GRUCell(1, hidden_size, key=keys[1])  # Input: just time
        self.decoder = eqx.nn.Linear(hidden_size, output_size, key=keys[2])
    
    def __call__(self, initial_conditions, times):
        """
        Args:
            initial_conditions: (batch, input_size) - ICs
            times: (T,) - timesteps to predict
        
        Returns:
            outputs: (batch, T, output_size) - predictions at each time
        """
        # Encode ICs to initial hidden state
        h = jax.nn.tanh(eqx.filter_vmap(self.encoder)(initial_conditions))
        
        # Unroll RNN over time
        def step(h, t):
            # Input at this timestep: just the time value
            x_t = jnp.array([[t]])  # Shape: (1, 1)
            h_next = self.rnn_cell(x_t, h)
            output = self.decoder(h_next)
            return h_next, output
        
        # Scan over timesteps
        _, outputs = jax.lax.scan(step, h, times)
        return outputs
```

**Training strategy**:
1. Load training data: $(ICs_i, \{r_i(t_j)\}_{j=1}^T)$ for $i=1,\ldots,n$
2. Forward pass: Predict trajectories
3. Loss: MSE between predicted and true trajectories (sum over time)
4. Backprop through time (BPTT)
5. Update weights via Adam

**Advantages**:
- Natural for sequences
- Learns temporal dependencies
- Can handle variable-length trajectories (stop early if cluster disrupts)

**Challenges**:
- Harder to train than feedforward (vanishing gradients, even with GRU/LSTM)
- Sequential (can't parallelize over time)
- May struggle with very long sequences (>100 steps)

---

## 🔴 Part 3: Neural ODEs for Continuous Dynamics

RNNs model discrete-time sequences. But physical systems evolve **continuously** in time! Enter **Neural ODEs**.

### The Core Idea: Learn the Dynamics

**Physics perspective**: Star cluster evolution satisfies differential equations:
$$\frac{d\mathbf{r}}{dt} = \mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = \mathbf{a}(\mathbf{r}, \mathbf{v}, t)$$

where $\mathbf{a}$ comes from gravitational forces, two-body relaxation, stellar evolution, etc.

**Neural ODE perspective**: Instead of solving these equations analytically (impossible!), *learn* an effective dynamics:
$$\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z}, t)$$

where:
- $\mathbf{z}(t)$: Latent state (compressed representation of full cluster)
- $f_\theta$: Neural network (learns effective forces/evolution)
- $\theta$: Parameters (weights of NN)

**Given initial state** $\mathbf{z}(0)$, **integrate ODE** to get $\mathbf{z}(t)$ for any $t$:
$$\mathbf{z}(t) = \mathbf{z}(0) + \int_0^t f_\theta(\mathbf{z}(s), s) \, ds$$

**Then decode** to observables:
$$r_{\text{core}}(t) = g_\theta(\mathbf{z}(t))$$

**Why this is powerful**:
1. **Continuous time**: Can query at any $t$, not just discrete timesteps
2. **Physics-inspired**: Directly models dynamics (rates of change)
3. **Implicit smoothness**: ODE solutions are smooth (differentable)
4. **Efficient**: Adaptive timestep solvers (use finer steps when dynamics change fast)
5. **Memory efficient**: Don't need to store intermediate states during training

```{admonition} Connection to Modules 2-4: You've Solved ODEs All Semester!
:class: note

**Module 2 (Stellar Structure)**: 
$$\frac{dP}{dr} = -\frac{GM(r)\rho}{r^2}, \quad \frac{dM}{dr} = 4\pi r^2 \rho$$
Solved via integration (Runge-Kutta, etc.)

**Module 3 (Phase Space)**:
$$\frac{d\mathbf{r}}{dt} = \frac{\partial H}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{r}}$$
Hamilton's equations govern dynamics

**Module 4 (Radiative Transfer)**:
$$\frac{dI_\nu}{ds} = -\kappa_\nu I_\nu + j_\nu$$
Integrating along rays

**Neural ODEs**: Same idea, but $f$ is *learned* not derived!
$$\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z}, t)$$

When physics is too complex for analytical $f$ (turbulence, N-body relaxation, multi-scale phenomena), let a neural network approximate it!

This bridges **physics-based simulation** and **data-driven learning**. You get the best of both: interpretability of ODEs + flexibility of neural networks.
```

### Neural ODE Architecture

**Three components**:

1. **Encoder**: Map observables to latent state
   $$\mathbf{z}_0 = \text{Encoder}_\theta(ICs) \quad \in \mathbb{R}^d$$
   - Typical: 1-2 layer MLP
   - $d$ = latent dimension (e.g., 64-128)

2. **Dynamics Network**: Learn evolution
   $$f_\theta(\mathbf{z}, t) = \text{MLP}([\mathbf{z}, t]) \quad \in \mathbb{R}^d$$
   - Input: Current state $\mathbf{z}$ + time $t$
   - Output: Time derivative $d\mathbf{z}/dt$
   - Typical: 2-4 layer MLP with 64-256 neurons
   - Activation: Softplus or Tanh (smoother than ReLU for ODE solvers)

3. **Decoder**: Map latent state to observables
   $$\mathbf{y}(t) = \text{Decoder}_\theta(\mathbf{z}(t)) \quad \in \mathbb{R}^p$$
   - Typical: 1-2 layer MLP
   - $p$ = number of outputs (e.g., 3 for $r_{\text{core}}, \sigma_v, f_{\text{bound}}$)

**Forward pass**:
1. Encode: $\mathbf{z}_0 = \text{Encoder}(ICs)$
2. Integrate: $\mathbf{z}(t) = \text{ODESolve}(f_\theta, \mathbf{z}_0, [0, t])$
3. Decode: $\mathbf{y}(t) = \text{Decoder}(\mathbf{z}(t))$

**Training**: Standard supervised learning!
- Loss: MSE between predicted and true trajectories
- Backprop: Through ODE solver (adjoint method - see below)
- Optimizer: Adam/AdamW

### The Adjoint Method: Backpropagating Through ODE Solvers

**Challenge**: How do you compute $\partial \mathcal{L}/\partial \theta$ when $\mathcal{L}$ depends on $\mathbf{z}(T)$ which is obtained by solving an ODE?

**Naive approach**: 
1. Store all intermediate ODE solver states
2. Backprop through each solver step

**Problem**: 
- Memory scales with number of solver steps (could be 1000s!)
- Very slow

**Adjoint method** (Chen et al. 2018):

Define **adjoint state** $\mathbf{a}(t) = \partial \mathcal{L}/\partial \mathbf{z}(t)$.

**Key insight**: $\mathbf{a}(t)$ itself satisfies an ODE (backward in time):
$$\frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \frac{\partial f_\theta}{\partial \mathbf{z}}$$

**Algorithm**:
1. Forward pass: Solve $d\mathbf{z}/dt = f_\theta(\mathbf{z}, t)$ from $t=0$ to $t=T$
2. Compute loss $\mathcal{L}(\mathbf{z}(T))$ and $\mathbf{a}(T) = \partial \mathcal{L}/\partial \mathbf{z}(T)$
3. Backward pass: Solve adjoint ODE backward from $t=T$ to $t=0$
4. Gradient: $\partial \mathcal{L}/\partial \theta = -\int_T^0 \mathbf{a}^T \frac{\partial f_\theta}{\partial \theta} dt$

**Advantages**:
- Memory: $O(1)$ (only need current state, not all intermediate states!)
- Accurate: Same accuracy as forward ODE solve
- Flexible: Works with any ODE solver (Euler, RK4, adaptive, etc.)

**In practice** (with JAX + Diffrax):
```python
import diffrax

# Define dynamics
def f(t, z, args):
    """Neural network dynamics."""
    params = args['params']
    return dynamics_network(params, z, t)

# Solve ODE (forward + backward via autodiff)
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(f),
    solver=diffrax.Dopri5(),  # Adaptive Runge-Kutta
    t0=0.0,
    t1=t_final,
    dt0=0.01,
    y0=z0,
    args={'params': params},
    saveat=diffrax.SaveAt(ts=times),
)

# Loss
loss = jnp.mean((solution.ys - y_true)**2)

# Gradient (autodiff handles adjoint method!)
grad_loss = jax.grad(loss)(params)
```

**JAX + Diffrax handles the adjoint method automatically!** You just write forward ODE, backprop "just works."

```{admonition} The More You Know: Why Adjoint Method Works
:class: tip, dropdown

**Calculus of variations**: If $\mathcal{L}$ depends on trajectory $\mathbf{z}(t)$ which satisfies ODE constraint $d\mathbf{z}/dt - f(\mathbf{z}, t) = 0$, then:

**Lagrangian**:
$$\mathcal{L}_{\text{aug}} = \mathcal{L}(\mathbf{z}(T)) + \int_0^T \mathbf{a}(t)^T \left[\frac{d\mathbf{z}}{dt} - f(\mathbf{z}, t)\right] dt$$

where $\mathbf{a}(t)$ is **Lagrange multiplier** (adjoint state).

**Stationarity**: $\delta \mathcal{L}_{\text{aug}}/\delta \mathbf{z} = 0$ gives adjoint ODE:
$$\frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \frac{\partial f}{\partial \mathbf{z}}$$

with boundary condition $\mathbf{a}(T) = \partial \mathcal{L}/\partial \mathbf{z}(T)$.

**Then**: $\partial \mathcal{L}/\partial \theta = \int_0^T \mathbf{a}^T \frac{\partial f}{\partial \theta} dt$

This is **optimal control theory** applied to neural networks! Same math that powers spacecraft trajectories and economic optimization.

**Reference**: Pontryagin's Maximum Principle (1962) - foundational result in optimal control.
```

### Implementation in JAX + Equinox + Diffrax

Here's a complete Neural ODE for cluster evolution:

```python
import equinox as eqx
import diffrax
import jax
import jax.numpy as jnp

class ClusterNeuralODE(eqx.Module):
    """Neural ODE for star cluster evolution."""
    
    encoder: eqx.nn.MLP
    dynamics: eqx.nn.MLP
    decoder: eqx.nn.MLP
    
    def __init__(self, input_dim, latent_dim, output_dim, *, key):
        keys = jax.random.split(key, 3)
        
        # Encoder: ICs → latent state
        self.encoder = eqx.nn.MLP(
            in_size=input_dim,
            out_size=latent_dim,
            width_size=128,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[0]
        )
        
        # Dynamics: (latent, time) → d(latent)/dt
        self.dynamics = eqx.nn.MLP(
            in_size=latent_dim + 1,  # z + t
            out_size=latent_dim,
            width_size=128,
            depth=3,
            activation=jax.nn.softplus,  # Smooth for ODE solver
            key=keys[1]
        )
        
        # Decoder: latent → observables
        self.decoder = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=output_dim,
            width_size=64,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[2]
        )
    
    def dynamics_fn(self, t, z, args):
        """ODE right-hand side: dz/dt = f(z, t)"""
        # Concatenate z and t
        zt = jnp.concatenate([z, jnp.array([t])])
        return self.dynamics(zt)
    
    def __call__(self, initial_conditions, times):
        """
        Predict cluster evolution.
        
        Args:
            initial_conditions: (input_dim,) - ICs
            times: (T,) - times to evaluate
        
        Returns:
            outputs: (T, output_dim) - predictions
        """
        # Encode to latent state
        z0 = self.encoder(initial_conditions)
        
        # Solve ODE
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.dynamics_fn),
            solver=diffrax.Dopri5(),  # Adaptive RK
            t0=times[0],
            t1=times[-1],
            dt0=0.1,
            y0=z0,
            saveat=diffrax.SaveAt(ts=times),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        )
        
        # Decode latent states to observables
        outputs = jax.vmap(self.decoder)(solution.ys)
        return outputs

# Usage
key = jax.random.PRNGKey(0)
model = ClusterNeuralODE(
    input_dim=5,   # N, M, Q, c, IMF_slope
    latent_dim=64,
    output_dim=3,  # r_core, sigma_v, f_bound
    key=key
)

# Predict
times = jnp.linspace(0, 200, 100)  # 0-200 Myr, 100 timesteps
predictions = model(initial_conditions, times)
```

**Training loop**:
```python
import optax

@eqx.filter_jit
def loss_fn(model, ics, times, y_true):
    """MSE loss over trajectory."""
    y_pred = jax.vmap(model, in_axes=(0, None))(ics, times)
    return jnp.mean((y_pred - y_true)**2)

@eqx.filter_jit
def train_step(model, opt_state, ics, times, y_true):
    """Single training step."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, ics, times, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Optimizer
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

# Training loop
for epoch in range(num_epochs):
    for batch_ics, batch_times, batch_y in dataloader:
        model, opt_state, loss = train_step(model, opt_state, batch_ics, batch_times, batch_y)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
```

### Advantages of Neural ODEs

1. **Continuous time**: Predict at *any* time, not just training timesteps
2. **Adaptive computation**: ODE solver uses finer steps when dynamics change rapidly
3. **Memory efficient**: Adjoint method is $O(1)$ memory
4. **Physics-inspired**: Directly models dynamics (aligns with how physics works!)
5. **Smooth predictions**: ODE solutions are differentiable

### Challenges and Solutions

**Challenge 1: Slow training**
- ODE solves are expensive (many function evaluations)
- Solution: Use faster solvers (lower-order methods), or train on shorter sequences first

**Challenge 2: Numerical instability**
- Stiff ODEs can cause solver failures
- Solution: Use implicit solvers (more stable but slower), or regularize dynamics network

**Challenge 3: Local minima**
- Complex loss landscape
- Solution: Good initialization (pretrain encoder/decoder separately), learning rate schedules

**Challenge 4: Overfitting**
- Can memorize training trajectories
- Solution: Weight decay, dropout, data augmentation (add noise to ICs)

**For your project**: Start simple (2-layer dynamics network), then increase complexity. Monitor ODE solver failures (Diffrax will warn you).

---

## 🔴 Part 4: Transformers for Attention Over Time

The most recent revolution in deep learning: **attention mechanisms** and **Transformers**.

### The Attention Idea

**Problem with RNNs**: Information from early timesteps must flow through all hidden states to reach later timesteps. This creates a bottleneck.

**Attention**: Directly connect any output timestep to *all* input timesteps. Learn which inputs are relevant for each output.

**Query-Key-Value framework**:
- **Query** (Q): What am I looking for?
- **Key** (K): What information do I have?
- **Value** (V): What is that information?

**Attention weights**: How much does query $q$ attend to key $k$?
$$\alpha_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_j \exp(q_i^T k_j / \sqrt{d})}$$

where $d$ is dimension (scaling for numerical stability).

**Attention output**: Weighted sum of values
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

**Intuition**: Each output position "asks questions" (queries) and gets answers weighted by relevance (keys) with content (values).

### Self-Attention for Sequences

For sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T$:

1. **Project** to Q, K, V:
   $$Q = \mathbf{X} W_Q, \quad K = \mathbf{X} W_K, \quad V = \mathbf{X} W_V$$

2. **Compute attention**:
   $$\text{Attention}(\mathbf{X}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Properties**:
- **Parallel**: Computes all outputs simultaneously (unlike RNN's sequential)
- **Long-range**: Any position can attend to any other (no vanishing gradients!)
- **Permutation invariant** (without positional encoding)

### Transformer Architecture

**Full Transformer layer**:

1. **Multi-head self-attention**:
   - Run $h$ attention heads in parallel (learn different patterns)
   - Concatenate + project

2. **Feed-forward network** (MLP applied to each position):
   $$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

3. **Residual connections** + **Layer normalization**:
   $$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{Attention}(\mathbf{x}))$$
   $$\mathbf{x}'' = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

**Stack multiple layers** (6-12 typical) for deep Transformer.

### Positional Encoding

**Problem**: Attention is permutation-invariant. Doesn't know *order* of sequence!

**Solution**: Add positional information to inputs.

**Sinusoidal encoding** (original Transformer):
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

**Learned encoding**: Treat positions as learnable embeddings.

**For time series**: Directly include time as a feature!
$$\mathbf{x}_t' = [\mathbf{x}_t, t]$$

### Transformers for Cluster Evolution

**Architecture**:

1. **Encoder**:
   - Input: ICs
   - Output: Context embedding

2. **Time queries**:
   - Input: Times $[t_1, \ldots, t_T]$ (as embeddings)
   - Cross-attention to context

3. **Decoder**:
   - Input: Attended time embeddings
   - Output: Predictions $[y_1, \ldots, y_T]$

**Advantages**:
- **Parallel**: Predict all timesteps simultaneously (fast!)
- **Flexible**: Can query arbitrary times (like Neural ODE)
- **Attention patterns**: Interpretable (which past times matter for prediction?)

**Disadvantages**:
- **More parameters**: Typically need more data than RNN/Neural ODE
- **Quadratic complexity**: $O(T^2)$ attention for sequence length $T$
- **Less physics-inspired**: No explicit dynamics model

**Implementation sketch**:
```python
class ClusterTransformer(eqx.Module):
    """Transformer for cluster evolution."""
    
    ic_encoder: eqx.nn.MLP
    time_embedding: eqx.nn.Embedding
    transformer_layers: list
    output_projection: eqx.nn.Linear
    
    def __init__(self, ...):
        # Initialize components
        pass
    
    def __call__(self, ics, times):
        # Encode ICs
        context = self.ic_encoder(ics)
        
        # Embed times
        time_queries = self.time_embedding(times)
        
        # Transformer layers with cross-attention
        for layer in self.transformer_layers:
            time_queries = layer(time_queries, context)
        
        # Project to outputs
        predictions = self.output_projection(time_queries)
        return predictions
```

**When to use Transformers**:
- Long sequences (>100 timesteps)
- Need parallelism (have GPU)
- Data is plentiful (1000+ training examples)
- Want interpretable attention weights

**When NOT to use**:
- Small datasets (<500 examples) → overfit easily
- Need strong physics constraints → hard to incorporate
- Sequence length varies wildly → inefficient padding

---

## 🟡 Part 5: ResNets and Physics Priors

Sometimes we *do* have physics knowledge! Instead of learning from scratch, we can **incorporate prior knowledge**.

### Residual Networks (ResNets)

**Standard network**: $\mathbf{y} = f(\mathbf{x})$

**ResNet**: $\mathbf{y} = \mathbf{x} + f(\mathbf{x})$

**Key idea**: Learn the *residual* (correction), not the full function.

**Why this helps**:
1. **Easier optimization**: Identity mapping is easy to learn ($f \approx 0$)
2. **Gradient flow**: Gradients flow through identity skip connection (helps deep networks)
3. **Interpretable**: $f$ is "what's missing from identity"

### Physics-Informed ResNets

**Setup**: You have an approximate physics model $f_{\text{physics}}(x)$ (analytical, simplified, etc.)

**Hybrid model**: 
$$\mathbf{y} = f_{\text{physics}}(\mathbf{x}) + f_{\theta}(\mathbf{x})$$

where $f_\theta$ is a neural network learning **corrections**.

**For cluster evolution**:

**Analytical baseline**: Use simplified dynamics (e.g., Henon's self-similar collapse model, isothermal sphere)

$$r_{\text{core}}^{\text{analytic}}(t) = r_0 \left(1 - \frac{t}{t_{\text{collapse}}}\right)^{1/2}$$

**Neural correction**:
$$r_{\text{core}}^{\text{true}}(t) = r_{\text{core}}^{\text{analytic}}(t) + \text{NN}_\theta(ICs, t)$$

**Benefits**:
1. **Better extrapolation**: Physics baseline guides predictions outside training data
2. **Data efficiency**: NN only learns deviations (smaller, easier to fit)
3. **Interpretability**: Residuals tell you what physics is missing!
4. **Faster training**: Network starts from good initialization

**Example residuals you might find**:
- **Two-body relaxation**: Analytic model assumes smooth potential, misses discrete encounters
- **Stellar evolution**: Mass loss from evolved stars not in simple model
- **Tidal effects**: External field effects if baseline is isolated cluster

```{admonition} Connection to Module 2: Approximate vs Exact Solutions
:class: note

In Module 2 (stellar structure), you used:
- **Polytropes**: Approximate solutions assuming $P \propto \rho^\gamma$
- **Lane-Emden equation**: Exact solution for polytropes
- **Real stars**: Deviations due to detailed EOS, opacity, nuclear reactions

**Physics-informed NN**: Same philosophy!
- **Polytrope/simple model**: Physics baseline $f_{\text{physics}}$
- **Real star**: True solution $f_{\text{true}}$
- **Neural network**: Learns residual $f_{\text{true}} - f_{\text{physics}}$

This is **scientific ML**: Combine domain knowledge (physics) with data-driven learning (NN). Best of both worlds!
```

### Implementation

```python
class PhysicsInformedResNet(eqx.Module):
    """ResNet with physics baseline."""
    
    correction_network: eqx.nn.MLP
    
    def __init__(self, ...):
        self.correction_network = eqx.nn.MLP(...)
    
    def physics_baseline(self, ics, times):
        """Analytical approximation (e.g., self-similar collapse)."""
        N, M, Q, c = ics
        t_relax = self.compute_relaxation_time(N, M)
        t_collapse = c * t_relax
        
        # Self-similar core collapse (Henon 1961)
        r_core = jnp.where(
            times < t_collapse,
            1.0 - (times / t_collapse)**0.5,  # Collapse
            0.1 * (1.0 + (times - t_collapse) / t_relax)**0.5  # Re-expansion
        )
        return r_core
    
    def __call__(self, ics, times):
        """Prediction = physics + learned correction."""
        baseline = self.physics_baseline(ics, times)
        correction = self.correction_network(jnp.concatenate([ics, times]))
        return baseline + correction
```

**Training**: Standard supervised learning. The physics baseline is fixed (not trained), only correction network parameters are learned.

---

## 🟡 Part 6: Physics-Informed Neural Networks (PINNs)

Take physics integration further: **enforce physical laws as constraints** in the loss function.

### The PINN Framework

**Idea**: If $f(x, t)$ satisfies PDE $\mathcal{N}[f] = 0$, train NN to satisfy this PDE!

**Loss function**:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

where:
- $\mathcal{L}_{\text{data}} = \|f_\theta(x_i, t_i) - y_i\|^2$ (fit observations)
- $\mathcal{L}_{\text{physics}} = \|\mathcal{N}[f_\theta]\|^2$ (satisfy PDE)

**Example**: Poisson equation for gravitational potential
$$\nabla^2 \Phi = 4\pi G \rho$$

**PINN loss**:
$$\mathcal{L}_{\text{physics}} = \left\|\frac{\partial^2 \Phi_\theta}{\partial x^2} + \frac{\partial^2 \Phi_\theta}{\partial y^2} + \frac{\partial^2 \Phi_\theta}{\partial z^2} - 4\pi G \rho\right\|^2$$

**How to compute $\partial^2 \Phi_\theta / \partial x^2$?** Automatic differentiation!

```python
def poisson_residual(phi_network, x, y, z, rho):
    """Compute PDE residual for Poisson equation."""
    def phi(xyz):
        return phi_network(xyz)
    
    # First derivatives
    grad_phi = jax.grad(phi)(jnp.array([x, y, z]))
    
    # Second derivatives (Laplacian)
    laplacian = jax.grad(lambda xyz: jnp.sum(jax.grad(phi)(xyz)))(jnp.array([x, y, z]))
    laplacian = jnp.trace(jax.hessian(phi)(jnp.array([x, y, z])))
    
    # PDE residual
    residual = laplacian - 4 * jnp.pi * G * rho
    return residual**2
```

**For cluster evolution**:

**Conservation laws** (energy, momentum, mass):
$$\frac{dE}{dt} = 0, \quad \frac{d\mathbf{P}}{dt} = 0, \quad \frac{dM}{dt} = 0$$

**PINN loss**:
```python
def physics_loss(model, ics, times):
    """Enforce conservation laws."""
    predictions = model(ics, times)
    
    # Energy conservation: dE/dt ≈ 0
    energy = compute_energy(predictions)
    energy_derivative = jnp.gradient(energy, times)
    loss_energy = jnp.mean(energy_derivative**2)
    
    # Mass conservation: dM/dt = 0
    mass = predictions['bound_mass']
    mass_derivative = jnp.gradient(mass, times)
    loss_mass = jnp.mean(mass_derivative**2)
    
    return loss_energy + loss_mass
```

### Advantages of PINNs

1. **Physics guarantees**: Hard constraints (exactly satisfy PDEs)
2. **Data efficiency**: Can train with fewer examples (physics fills in gaps)
3. **Extrapolation**: Physics guides predictions outside training data
4. **Interpretability**: Know *why* predictions are physical (not black box)

### Challenges

1. **Balancing loss terms**: How to weight $\lambda$ (data vs physics)?
2. **Stiff PDEs**: Hard to optimize (conflicting gradients)
3. **Requires differentiable physics**: Need to write PDE in differentiable form
4. **Computational cost**: Higher-order derivatives are expensive

**For your project**: PINNs are advanced. If you have time, try enforcing energy conservation or virial theorem. Otherwise, stick to RNN/Neural ODE/ResNet.

---

## 🟡 Part 7: Uncertainty Quantification for Neural Networks

Unlike GPs, NNs don't provide uncertainty naturally. But we can recover it!

### Method 1: Ensemble

**Train $K$ networks** with different initializations:
$$f_1, f_2, \ldots, f_K$$

**Prediction**:
- **Mean**: $\mu(x) = \frac{1}{K} \sum_{k=1}^K f_k(x)$
- **Std**: $\sigma(x) = \sqrt{\frac{1}{K} \sum_{k=1}^K (f_k(x) - \mu(x))^2}$

**Advantages**:
- Simple to implement (just train multiple times!)
- Captures model uncertainty (epistemic)
- Often improves accuracy (ensemble averaging reduces variance)

**Disadvantages**:
- $K \times$ training cost
- $K \times$ inference cost (can parallelize)

**Typical**: $K = 5-10$ for good uncertainty estimates.

### Method 2: Monte Carlo Dropout

**Use dropout at test time** (not just training!):

1. Train network with dropout (e.g., $p=0.1$)
2. At test time: Run $K$ forward passes with dropout enabled
3. Average predictions, compute standard deviation

**Approximates Bayesian inference**: Each dropout mask is a sample from posterior over weights.

**Advantages**:
- Single network (cheap training)
- Calibrated uncertainty (especially if tuned)

**Disadvantages**:
- Requires dropout in architecture
- Approximate (not true Bayesian posterior)

### Method 3: Quantile Regression

**Predict multiple quantiles** instead of just mean:

$$\text{NN output} = [\hat{y}_{0.05}, \hat{y}_{0.50}, \hat{y}_{0.95}]$$

**Train with quantile loss**:
$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

**Prediction intervals**: $[\hat{y}_{0.05}, \hat{y}_{0.95}]$ gives 90% interval.

**Advantages**:
- Direct prediction of intervals
- Captures aleatoric uncertainty
- Single network

**Disadvantages**:
- Doesn't capture epistemic uncertainty
- Need to choose quantiles in advance

### Method 4: Bayesian Neural Networks (Advanced)

**Put priors on weights**: $p(\theta)$

**Posterior**: $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta) p(\theta)$

**Prediction**: Marginalize over posterior
$$p(y | x, \mathcal{D}) = \int p(y | x, \theta) p(\theta | \mathcal{D}) d\theta$$

**Approximate inference**:
- **Variational inference**: Approximate posterior with Gaussian
- **MCMC**: Sample weights using HMC (connect to Project 4!)
- **Laplace approximation**: Gaussian around MAP estimate

**Advantages**:
- Principled Bayesian uncertainty
- Connects to GP (GP is infinite-width BNN!)

**Disadvantages**:
- Hard to implement (need specialized libraries)
- Expensive inference
- Hyperparameter choices (prior specification)

**For your project**: Use **ensembles** (simplest, effective). If time permits, try MC Dropout.

---

## 🔴 Part 8: Comparing Methods - When to Use What?

You now have many tools! How to choose?

### Decision Tree

**Q1: Do you need uncertainty quantification?**
- Yes → GP (if low-dim output) or NN Ensemble
- No → NN (single model)

**Q2: Is output low-dimensional (< 10)?**
- Yes → Try GP first (data-efficient, interpretable)
- No → Use NN

**Q3: Is your function smooth?**
- Very smooth → GP with SE kernel or Neural ODE
- Moderately smooth → GP with Matérn or RNN
- Non-smooth → NN (flexible)

**Q4: Do you have physics knowledge?**
- Strong (equations) → PINN or Hybrid model
- Moderate (approximate solution) → ResNet with physics baseline
- None → Standard NN

**Q5: Is data plentiful?**
- < 100 examples → GP
- 100-1000 examples → GP or NN
- > 1000 examples → NN (scalable)

**Q6: Do you need continuous-time predictions?**
- Yes → Neural ODE or Transformer
- No → RNN is simpler

**Q7: Are sequences very long (>100 steps)?**
- Yes → Transformer (parallel) or Neural ODE
- No → RNN/GRU works fine

### Summary Table

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **GP** | Uncertainty, data-efficient, interpretable | Scalability, high-dim output | <1000 data, low-dim out |
| **RNN/GRU** | Sequences, moderate data | Sequential (slow), vanishing grads | Discrete time series |
| **LSTM** | Long sequences, memory | More parameters, complexity | Very long dependencies |
| **Neural ODE** | Continuous, physics-inspired, memory-efficient | Slow training, local minima | Smooth dynamics |
| **Transformer** | Parallel, long-range, interpretable attention | Data-hungry, quadratic complexity | Long sequences, big data |
| **ResNet + Physics** | Extrapolation, interpretable, data-efficient | Need good baseline | Have approximate model |
| **PINN** | Physics guarantees, data-efficient | Hard to tune, requires differentiable physics | Strong physics knowledge |

### Your Project Strategy

**Week 1**: GP for scalars
- Fastest to implement
- Baseline for comparison
- Learn which ICs matter

**Week 2**: GP for time series → discover limitations
- Multi-output explodes
- Motivates NNs

**Week 3-4**: Choose NN architecture based on priorities:

**Priority 1: Simplicity & Speed** → **RNN/GRU**
- Standard, well-understood
- Easiest to implement and debug
- Good starting point

**Priority 2: Accuracy & Physics** → **Neural ODE**
- Most elegant for continuous dynamics
- Best aligns with physics of cluster evolution
- Recommended for ambitious students

**Priority 3: Flexibility & Scalability** → **Transformer**
- Modern architecture
- Good for very long sequences
- If you have lots of data

**Priority 4: Interpretability** → **ResNet with Physics Prior**
- Learn what physics is missing
- Best extrapolation
- Scientific insights

**My recommendation**: Start with **GRU** (simple, works). If that succeeds, try **Neural ODE** (elegant). If you have strong physics baseline, try **ResNet**.

---

## 🔴 Part 9: JAX + Equinox Ecosystem Deep Dive

You've seen code snippets. Now let's understand the full ecosystem.

### Equinox: Neural Networks as PyTrees

**Philosophy**: Models are just **callable PyTrees** (nested dicts/lists/tuples of arrays).

**Key abstractions**:

1. **`eqx.Module`**: Base class for models
   ```python
   class MyModel(eqx.Module):
       layers: list
       activation: callable
       
       def __call__(self, x):
           for layer in self.layers:
               x = self.activation(layer(x))
           return x
   ```

2. **Automatic PyTree registration**: Equinox makes modules JAX-compatible
   - Can JIT compile
   - Can differentiate
   - Can serialize/deserialize

3. **Filtering**: Separate trainable vs non-trainable
   ```python
   # Get only trainable parameters
   params = eqx.filter(model, eqx.is_inexact_array)
   
   # Update only trainable parameters
   model = eqx.apply_updates(model, updates)
   ```

**Built-in layers**:
- `eqx.nn.Linear`: Fully connected
- `eqx.nn.MLP`: Multi-layer perceptron
- `eqx.nn.Conv2d`: Convolution (if you need images)
- `eqx.nn.GRUCell`, `eqx.nn.LSTMCell`: RNN cells
- `eqx.nn.Dropout`: Dropout layer
- Many more!

### Optax: Optimization Library

**Composable optimizers**:
```python
import optax

# Chain multiple transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),       # Gradient clipping
    optax.adamw(learning_rate=1e-3),       # Adam with weight decay
    optax.apply_if_finite(optax.sgd(1e-3)) # Only apply if grads finite
)
```

**Learning rate schedules**:
```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-5
)
optimizer = optax.adam(learning_rate=schedule)
```

**Common optimizers**:
- `optax.sgd`: Stochastic gradient descent (with optional momentum)
- `optax.adam`: Adam
- `optax.adamw`: Adam with decoupled weight decay (recommended!)
- `optax.adagrad`: Adaptive gradients
- `optax.rmsprop`: RMSProp

### Diffrax: ODE/SDE Solvers

**Solving ODEs**:
```python
import diffrax

# Define ODE
def f(t, y, args):
    return -y  # dy/dt = -y

# Solve
solution = diffrax.diffeqsolve(
    terms=diffrax.ODETerm(f),
    solver=diffrax.Dopri5(),  # Adaptive Runge-Kutta 4/5
    t0=0.0,
    t1=10.0,
    dt0=0.1,
    y0=1.0,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10, 100)),
)

y_values = solution.ys  # Solution at specified times
```

**Available solvers**:
- `Euler()`: Simple, fast, low accuracy
- `Heun()`: 2nd order Runge-Kutta
- `Dopri5()`: Adaptive RK 4/5 (default, good choice)
- `Dopri8()`: Higher accuracy (for smooth problems)
- `ImplicitEuler()`: For stiff equations
- `Kvaerno5()`: Implicit solver (very stiff)

**Stepsize control**:
```python
stepsize_controller = diffrax.PIDController(
    rtol=1e-5,  # Relative tolerance
    atol=1e-7,  # Absolute tolerance
)
```

**Adjoint method** (for backprop):
```python
solution = diffrax.diffeqsolve(
    ...,
    adjoint=diffrax.RecursiveCheckpointAdjoint(),  # Memory-efficient backprop
)
```

### Putting It Together: Training Loop Template

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple

@eqx.filter_jit
def loss_fn(model, x_batch, y_batch):
    """Compute loss on batch."""
    y_pred = jax.vmap(model)(x_batch)
    return jnp.mean((y_pred - y_batch)**2)

@eqx.filter_jit
def train_step(model, opt_state, x_batch, y_batch):
    """Single training step with gradient update."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@eqx.filter_jit
def eval_step(model, x_batch, y_batch):
    """Evaluation step (no gradients)."""
    loss = loss_fn(model, x_batch, y_batch)
    return loss

# Initialize
key = jax.random.PRNGKey(0)
model = MyModel(key=key)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

# Training loop
for epoch in range(num_epochs):
    # Training
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
        train_loss += loss
    
    # Validation
    val_loss = 0.0
    for x_batch, y_batch in val_loader:
        loss = eval_step(model, x_batch, y_batch)
        val_loss += loss
    
    # Logging
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    # Early stopping (if validation loss stops improving)
    if early_stopping_criterion_met:
        break

# Save model
eqx.tree_serialise_leaves("model.eqx", model)

# Load model
model = eqx.tree_deserialise_leaves("model.eqx", model)
```

---

## ⚪ Part 10: Advanced Topics and Extensions

For ambitious students or future exploration.

### Graph Neural Networks for N-Body

**Idea**: Represent cluster as graph:
- Nodes = particles
- Edges = pairwise interactions

**GNN**: Message-passing updates
$$\mathbf{h}_i^{(l+1)} = \phi\left(\mathbf{h}_i^{(l)}, \sum_{j \in \mathcal{N}(i)} \psi(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij})\right)$$

**Benefits**:
- Permutation invariant (particle order doesn't matter)
- Learns effective interactions
- Can incorporate physics (edges = gravitational forces)

**Libraries**: `jraph` (JAX), `jax-md`

### Normalizing Flows for Distributions

**Idea**: Learn evolving phase space distribution $f(\mathbf{x}, \mathbf{v}, t)$

**Flow**: Invertible transformation
$$\mathbf{z} \sim p_0(\mathbf{z}) \quad \Rightarrow \quad \mathbf{x} = T_\theta(\mathbf{z})$$

**Training**: Maximum likelihood via change-of-variables
$$\log p(\mathbf{x}) = \log p_0(T_\theta^{-1}(\mathbf{x})) + \log \left|\det \frac{\partial T_\theta^{-1}}{\partial \mathbf{x}}\right|$$

**For clusters**: Learn $f(x, v, t)$ as normalizing flow conditioned on $t$.

**Libraries**: `surflows`, `flowjax`

### Amortized Inference

**Inverse problem**: Given final cluster state, infer initial conditions.

**Standard approach**: Run MCMC for each observation (expensive!)

**Amortized**: Train NN to directly output posterior
$$q_\theta(\text{ICs} | \text{observed state})$$

**Training**: Simulation-based inference (generate many IC → state pairs, train NN)

**Benefit**: One forward pass → full posterior (no MCMC needed!)

**Libraries**: `sbi` (simulation-based inference)

### Hamiltonian Neural Networks

**Idea**: Parameterize Hamiltonian $H_\theta(q, p)$, enforce Hamilton's equations

$$\dot{q} = \frac{\partial H_\theta}{\partial p}, \quad \dot{p} = -\frac{\partial H_\theta}{\partial q}$$

**Benefits**:
- Energy conservation built-in
- Symplectic structure preserved
- Long-term stability

**For clusters**: Total energy $E = K + U$ (kinetic + potential)

---

## 📊 Part 11: Debugging and Best Practices

Neural networks are finicky! Here's how to succeed.

### Common Failure Modes

**1. Loss not decreasing**
- Check: Gradient flow (use `jax.debug.print` to inspect gradients)
- Solution: Lower learning rate, check for NaNs, simplify architecture

**2. Training loss decreases, validation loss increases**
- Problem: Overfitting
- Solution: More data, weight decay, dropout, early stopping

**3. Both losses high**
- Problem: Underfitting
- Solution: More capacity (layers/neurons), train longer, check data quality

**4. NaN or Inf in loss**
- Problem: Numerical instability (exploding gradients, bad initialization)
- Solution: Gradient clipping, lower learning rate, better initialization

**5. ODE solver failures** (for Neural ODEs)
- Problem: Stiff dynamics, bad dynamics network
- Solution: Lower tolerance, use implicit solver, regularize dynamics

### Debugging Checklist

1. **Start simple**: Overfit on single batch first (should get loss ~0)
2. **Visualize**: Plot predictions vs truth throughout training
3. **Monitor**: Track loss, gradients, weight norms
4. **Ablate**: Remove components to isolate issues
5. **Compare**: Benchmark against baseline (linear model, GP)

### Hyperparameter Tuning

**Priority order**:
1. **Learning rate**: Most important! Try $[10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}]$
2. **Architecture**: Depth (2-6 layers), width (64-256 neurons)
3. **Regularization**: Weight decay $[10^{-5}, 10^{-4}, 10^{-3}]$, dropout $[0.1, 0.3, 0.5]$
4. **Batch size**: 32-256 (larger = stabler gradients but slower updates)
5. **Optimizer**: AdamW is usually best

**Systematic search**:
- Start with defaults (learning rate $10^{-3}$, 3 layers × 128 neurons, AdamW)
- Grid search learning rate first
- Then adjust architecture
- Finally tune regularization

**Rule of thumb**: Spend 80% of time on data quality, 15% on architecture, 5% on hyperparameters.

---

## 🎯 Part 12: Summary and Project Roadmap

### What You've Learned

**Architectures**:
- MLPs: Universal function approximators
- RNNs/GRUs/LSTMs: Sequential data with memory
- Neural ODEs: Continuous dynamics, physics-inspired
- Transformers: Attention over time, parallel
- ResNets: Physics priors, corrections to baselines
- PINNs: Hard physics constraints in loss

**Training**:
- Loss functions (MSE, MAE, quantile)
- Optimizers (Adam, AdamW, schedules)
- Regularization (weight decay, dropout, early stopping)
- Backpropagation (autodiff handles it!)

**JAX Ecosystem**:
- Equinox: Neural networks as PyTrees
- Optax: Composable optimizers
- Diffrax: ODE solvers with adjoint backprop

**Uncertainty**:
- Ensembles: Train multiple models
- MC Dropout: Stochastic inference
- Quantile regression: Direct interval prediction

**Comparison**:
- When to use GP vs NN
- Trade-offs: data efficiency, scalability, uncertainty, interpretability

### Your Project Roadmap (Weeks 3-4)

**Week 3, Day 1-2: Choose Architecture**
- Review Week 2 GP limitations analysis
- Decide: RNN? Neural ODE? Transformer? ResNet?
- Justify choice in project proposal

**Week 3, Day 3-5: Implementation**
- Implement architecture from scratch in Equinox
- Start with small version (2 layers, 64 neurons)
- Get training loop working, visualize predictions

**Week 3, Day 6-7: Debug and Tune**
- Overfit on small dataset (sanity check)
- Tune hyperparameters (learning rate, architecture size)
- Monitor training/validation curves

**Week 4, Day 1-3: Scale and Optimize**
- Train on full dataset
- Implement uncertainty quantification (ensemble or MC Dropout)
- Compare to GP from Week 1

**Week 4, Day 4-5: Analysis**
- Quantitative: RMSE, coverage, CRPS
- Qualitative: When does NN succeed/fail? Comparison to GP?
- Physics: What did NN learn? Interpretable?

**Week 4, Day 6-7: Final Report**
- Write comparative analysis (GP vs NN)
- Create visualizations (predictions, uncertainty, attention weights if Transformer)
- Prepare 10-minute presentation

### Success Criteria

**Minimum viable project**:
- [ ] Working NN implementation (trains without errors)
- [ ] Predicts time series (qualitatively reasonable)
- [ ] Comparison to GP (quantitative metrics)
- [ ] Documented code (installable package)

**Good project**:
- [ ] All above +
- [ ] Uncertainty quantification (ensemble or MC Dropout)
- [ ] Ablation studies (architecture choices, hyperparameters)
- [ ] Physical validation (energy conservation, etc.)
- [ ] Clear scientific insights (what did NN learn?)

**Excellent project**:
- [ ] All above +
- [ ] Multiple architectures compared (RNN vs Neural ODE vs Transformer)
- [ ] Physics integration (ResNet or PINN)
- [ ] Novel analysis (attention patterns, learned dynamics, extrapolation)
- [ ] Publishable-quality figures and writing

---

## 📝 Starter Code Structure

```
project-final/
├── src/
│   ├── gp/                    # Week 1 (complete)
│   │   └── ...
│   └── nn/                    # Weeks 3-4 (you implement!)
│       ├── __init__.py
│       ├── architectures/
│       │   ├── mlp.py         # Basic feedforward
│       │   ├── rnn.py         # RNN/GRU/LSTM
│       │   ├── neural_ode.py  # Neural ODE
│       │   ├── transformer.py # Transformer
│       │   └── resnet.py      # ResNet with physics
│       ├── training.py        # Training loops, loss functions
│       ├── uncertainty.py     # Ensemble, MC Dropout
│       └── utils.py           # Metrics, visualization
├── notebooks/
│   ├── 04_nn_exploration.ipynb      # Week 3: Try architectures
│   ├── 05_training_comparison.ipynb # Week 3: Training
│   └── 06_final_analysis.ipynb      # Week 4: GP vs NN
├── tests/
│   ├── test_architectures.py
│   └── test_training.py
└── docs/
    ├── week3_progress.md      # Checkpoint
    └── final_report.md        # Week 4 deliverable
```

**Key files to implement**:

### `architectures/rnn.py`

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class ClusterGRU(eqx.Module):
    """GRU for cluster evolution."""
    
    encoder: eqx.nn.MLP
    gru_cell: eqx.nn.GRUCell
    decoder: eqx.nn.MLP
    hidden_size: int
    
    def __init__(self, input_size, hidden_size, output_size, *, key):
        # TODO: Initialize encoder, GRU cell, decoder
        raise NotImplementedError
    
    def __call__(self, initial_conditions, times):
        # TODO: Encode ICs → h_0
        # TODO: Scan GRU over times
        # TODO: Decode hidden states → predictions
        raise NotImplementedError
```

### `architectures/neural_ode.py`

```python
class ClusterNeuralODE(eqx.Module):
    """Neural ODE for cluster evolution."""
    
    encoder: eqx.nn.MLP
    dynamics: eqx.nn.MLP
    decoder: eqx.nn.MLP
    
    def __init__(self, input_dim, latent_dim, output_dim, *, key):
        # TODO: Initialize components
        raise NotImplementedError
    
    def dynamics_fn(self, t, z, args):
        """ODE right-hand side."""
        # TODO: Return dz/dt = f(z, t)
        raise NotImplementedError
    
    def __call__(self, initial_conditions, times):
        # TODO: Encode → z_0
        # TODO: Solve ODE with Diffrax
        # TODO: Decode → predictions
        raise NotImplementedError
```

### `training.py`

```python
import optax
import equinox as eqx

def train_model(model, train_data, val_data, config):
    """
    Train neural network with early stopping.
    
    Args:
        model: Equinox model
        train_data: (X_train, times, y_train)
        val_data: (X_val, times, y_val)
        config: Training config (learning rate, epochs, etc.)
    
    Returns:
        trained_model, training_history
    """
    # TODO: Implement training loop
    # - Initialize optimizer
    # - Training step (forward, loss, backprop, update)
    # - Validation step
    # - Early stopping
    # - Checkpointing
    raise NotImplementedError

def loss_fn(model, x_batch, times, y_batch):
    """MSE loss over trajectories."""
    # TODO: Implement
    raise NotImplementedError
```

### `uncertainty.py`

```python
def train_ensemble(model_fn, train_data, val_data, config, n_models=5):
    """Train ensemble of models."""
    # TODO: Train n_models with different initializations
    # TODO: Return list of trained models
    raise NotImplementedError

def ensemble_predict(models, x, times):
    """Predict with ensemble (mean + std)."""
    # TODO: Get predictions from all models
    # TODO: Compute mean and std
    raise NotImplementedError
```

---

## 🎓 Self-Assessment Questions

**Conceptual**:
1. Why can't RNNs capture long-term dependencies? How do LSTMs/GRUs solve this?
2. What's the key difference between Neural ODEs and RNNs for modeling dynamics?
3. Explain attention mechanism in your own words. Why is it powerful?
4. How does a physics-informed ResNet differ from a standard ResNet?
5. Why don't neural networks provide uncertainty naturally? How do ensembles help?

**Mathematical**:
1. Derive the gradient of MSE loss w.r.t. weights in a 2-layer MLP
2. Write out the GRU update equations. What do the gates control?
3. For Neural ODE $dz/dt = f(z, t)$, how do you compute $\partial z(T)/\partial \theta$?
4. What is the computational complexity of self-attention? Why is it $O(T^2)$?
5. Prove that residual connections help gradient flow (hint: show $\partial \mathcal{L}/\partial x$ has identity component)

**Computational**:
1. Implement forward pass for 2-layer MLP in JAX (no Equinox). Include activations.
2. How would you implement gradient clipping in a training loop?
3. Write code to compute RMSE and coverage for ensemble predictions
4. What's the difference between `jax.jit` and `eqx.filter_jit`?
5. How do you save and load an Equinox model?

**Scientific**:
1. You're modeling cluster core collapse. Which architecture (RNN/Neural ODE/Transformer) would you choose and why?
2. Your NN predicts negative core radii sometimes. How do you fix this?
3. Training loss is much lower than GP, but validation loss is higher. What's happening?
4. Your Neural ODE solver fails with "max steps reached". What are possible causes?
5. How would you visualize what your Transformer learned (attention patterns)?

---

## 📚 Key Resources

**Papers** (foundational):
1. "Neural Ordinary Differential Equations" - Chen et al. (2018) - NeurIPS Best Paper
2. "Attention Is All You Need" - Vaswani et al. (2017) - Introduced Transformers
3. "Physics-Informed Neural Networks" - Raissi et al. (2019)
4. "Deep Residual Learning for Image Recognition" - He et al. (2015) - ResNets

**Books**:
1. *Deep Learning* - Goodfellow, Bengio, Courville (2016) - Comprehensive textbook
2. *Neural Networks and Deep Learning* - Nielsen (online) - Clear explanations
3. *Dive into Deep Learning* - Zhang et al. (interactive) - Code-first approach

**JAX Resources**:
1. JAX documentation: https://jax.readthedocs.io
2. Equinox docs: https://docs.kidger.site/equinox/
3. Diffrax docs: https://docs.kidger.site/diffrax/
4. Patrick Kidger's blog: https://kidger.site/thoughts/

**Astrophysics + ML**:
1. "Machine Learning for Astrophysics" - Huertas-Company & Lanusse (2023)
2. "Neural Networks for N-Body Simulations" - various recent papers on arXiv
3. Your course! This is frontier research territory.

---

## 🚀 Final Words

You've now learned the full spectrum of modern machine learning methods for scientific computing:

**From GPs to NNs**: Data-efficient, uncertainty-aware methods → Scalable, flexible architectures

**From Black Boxes to Physics-Informed**: Pure learning → Hybrid physics + learning

**From Theory to Practice**: Mathematical foundations → Production JAX code

**The synthesis**: You understand *when* to use *which* method, and *why*. This is the mark of a computational scientist, not just an ML practitioner.

Your final project brings this full circle:
1. **Identify problem** (expensive simulations)
2. **Try simple method** (GP - Week 1)
3. **Understand limitations** (time series - Week 2)
4. **Apply sophisticated solution** (NN - Weeks 3-4)
5. **Compare and contrast** (trade-offs, scientific insights)

This is **research-level scientific computing**. You're not just applying tools—you're evaluating methods, making design decisions, and contributing knowledge about when/how ML can accelerate astrophysics.

**The future**: Methods you learn here (Neural ODEs, Transformers, PINNs) are actively used in:
- Climate modeling (weather prediction, climate emulation)
- Drug discovery (molecular dynamics)
- Cosmology (large-scale structure formation)
- Engineering (fluid dynamics, structural analysis)

You're learning **transferable skills** that span scientific disciplines.

**Now go build something amazing!** 🌌

---

*Next: Week 3 begins - choose your architecture, start implementing, and push the boundaries of what's possible with ML for astrophysics!*

**Remember**: The goal isn't perfection, it's **learning through doing**. Embrace the struggles, debug with curiosity, and celebrate the breakthroughs. This is how computational astrophysicists are forged!
