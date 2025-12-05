---
title: "Module 7: Neural Networks—Learning Functions from Data"
subtitle: "From Neurons to Emulators | The Learnable Universe | ASTR 596"
---

:::{epigraph}
"What I cannot create, I do not understand."

— Richard Feynman (found on his blackboard at the time of his death, 1988)
:::

## Learning Objectives

By the end of this module, you will be able to:

- [ ] **Explain** why neural networks are universal function approximators and what this means for scientific emulation
- [ ] **Construct** a multi-layer perceptron (MLP) architecture from first principles
- [ ] **Derive** how forward propagation computes predictions through layers of transformations
- [ ] **Connect** backpropagation to the automatic differentiation you learned in Module 6
- [ ] **Implement** gradient descent training using loss functions appropriate for regression
- [ ] **Apply** data normalization and recognize why it's essential for neural network training
- [ ] **Use** ensemble methods to quantify predictive uncertainty
- [ ] **Build** neural network emulators in Equinox and train them with Optax
- [ ] **Evaluate** emulator accuracy and understand failure modes
- [ ] **Integrate** trained emulators into NumPyro for Bayesian inference

---

:::{admonition} 🗺️ Your Roadmap Through This Module
:class: note

**Core Question**: How can we train a flexible mathematical function to approximate expensive computations—and when should we trust its predictions?

**The Big Picture**: You've built N-body simulations (Project 2/5) and Bayesian inference engines (Project 4). Now you'll learn to build *fast approximations* to expensive computations. This is the key to the final project: training a neural network to predict N-body simulation outcomes, then using that emulator for inference.

**Section 7.1: The Emulation Problem**
Why we need fast surrogates for expensive simulations. The scientific case.

**Section 7.2: From Linear Regression to Neural Networks**
The intellectual progression: linear models → feature engineering → learned features.

**Section 7.3: The Computational Neuron**
The mathematical building blocks. Neurons, activations, and why nonlinearity matters.

**Section 7.4: Layering Neurons into Networks**
Architecture design: hidden layers, parameter counting, and the expressiveness tradeoff.

**Section 7.5: The Universal Approximation Theorem**
Why neural networks can (in principle) approximate any function. The profound idea and its limits.

**Section 7.6: Forward Propagation**
How input becomes output. The computational graph perspective you know from Module 6.

**Section 7.7: Training as Optimization**
Loss functions, gradient descent, and the connection to likelihood maximization.

**Section 7.8: Weight Initialization—Breaking Symmetry**
Why random initialization matters and how it enables ensembles.

**Section 7.9: Practical Training**
Normalization, learning rates, convergence, multi-output handling, and debugging.

**Section 7.10: Uncertainty via Ensembles**
Why single networks are dangerous, and how ensembles quantify epistemic uncertainty.

**Section 7.11: The JAX Ecosystem—Equinox and Optax**
Professional tools for building and training neural networks.

**Section 7.12: From Emulator to Inference**
Connecting your trained network to NumPyro for Bayesian parameter recovery.

**Section 7.13: Synthesis**
What you've learned and how it all connects.
:::

---

## 7.1: The Emulation Problem

**Priority: 🔴 Essential**

### Why Emulators Matter

:::{margin}
**Emulator** (or **surrogate model**): A fast approximation to an expensive computation. Trained on input-output pairs from the expensive model, then used in place of it.
:::

Consider your N-body simulator. Each run takes seconds to minutes. But scientific inference requires *thousands* of model evaluations—MCMC needs to evaluate the likelihood at every step, optimization needs gradients computed repeatedly. What if we could train a fast approximation that captures the essential physics?

**The computational bottleneck**: Your final project asks you to infer what initial conditions $(Q_0, a)$ produced a given cluster outcome. Using your actual N-body code inside MCMC would require:

$$\underbrace{2000}_{\text{MCMC samples}} \times \underbrace{50}_{\text{leapfrog steps}} \times \underbrace{30\text{ sec}}_{\text{per sim}} = 35 \text{ days}$$

That's *one* inference run. Unacceptable.

**The emulator solution**: Train a neural network on ~100 N-body runs. The network learns to predict outputs from inputs in *microseconds*. Now inference takes minutes, not months.

<!-- VISUAL: Side-by-side comparison diagram
Left panel: "Direct Inference" showing MCMC loop calling N-body simulator (slow)
Right panel: "Emulator-Based Inference" showing MCMC loop calling neural network (fast)
Include approximate timings: 35 days vs 5 minutes
-->

:::{admonition} 📗 Connection to Module 1: Models as Compression
:class: note

Remember Module 1's insight that *models compress information*? Summary statistics compress high-dimensional simulation outputs (particle positions and velocities) into meaningful numbers ($f_{\rm bound}$, $\sigma_v$, $r_h$). An emulator goes further—it compresses the *entire input-output relationship* of your simulator into a set of neural network weights.

This is learned compression: the network discovers which features of the mapping matter, automatically finding a low-dimensional representation of a complex physical process. Your ~4,500 network parameters encode the essential physics of how initial virial ratio and scale radius determine cluster fate.
:::

### Emulators in Modern Astrophysics

This isn't a toy problem. Emulation is transforming computational science:

**Cosmology**: The Dark Energy Survey uses neural network emulators to predict matter power spectra. Full $N$-body simulations take GPU-hours; emulators return predictions in milliseconds, enabling proper Bayesian parameter estimation.

**Stellar evolution**: MESA simulations of stellar interiors are expensive. Emulators predict stellar properties (luminosity, radius, lifetime) from initial conditions, enabling population synthesis studies with millions of virtual stars.

**Galaxy formation**: IllustrisTNG-scale hydrodynamic simulations are computationally prohibitive to run thousands of times. Emulators trained on existing runs enable exploration of the parameter space.

**Your final project**: You're learning the same workflow used in frontier research—just on a tractable problem (star clusters instead of cosmological volumes).

---

## 7.2: From Linear Regression to Neural Networks

**Priority: 🟡 Important**

Before diving into neural network details, let's see how they emerge naturally from ideas you already know.

### The Progression of Function Approximation

:::{margin}
**Feature engineering**: Manually constructing input transformations (like $x^2$, $\log x$, $x_1 x_2$) to capture nonlinear relationships. Traditional approach before neural networks automated this process.
:::

**Level 1: Linear Regression**

The simplest model: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$

You find weights $\mathbf{w}$ and bias $b$ that minimize squared error. This works beautifully when the true relationship is linear—but nature rarely cooperates.

**Level 2: Polynomial/Basis Regression**

To capture nonlinearity, you engineer features: $\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \ldots$

This is still linear in the *parameters* (the $w_i$), but nonlinear in the input. The problem: you must *choose* which features to include. For your emulator, should you use $Q_0^2$? $\log a$? $Q_0 \times a$? You'd have to guess.

**Level 3: Neural Networks**

Here's the key insight: **what if we *learned* the features?**

A neural network with one hidden layer computes:
$$\hat{y} = \mathbf{w}_{\rm out}^T \underbrace{\sigma(\mathbf{W}_{\rm hidden} \mathbf{x} + \mathbf{b}_{\rm hidden})}_{\text{learned features}} + b_{\rm out}$$

The hidden layer creates *learned nonlinear features* of the input. The output layer then does linear regression on these features. Training optimizes both the feature extraction *and* the final regression simultaneously.

<!-- VISUAL: Three-panel progression diagram
Panel 1: Linear regression - straight line through data points
Panel 2: Polynomial regression - curve through data, with explicit x, x², x³ features shown
Panel 3: Neural network - curve through data, with "learned features" as hidden layer
Show that NN subsumes the others as special cases
-->

:::{admonition} The Profound Shift
:class: important

Traditional statistics: Human chooses features → algorithm finds weights
Neural networks: Algorithm finds features *and* weights

This is why neural networks work across such diverse problems—they discover task-appropriate representations automatically. The same architecture that predicts cluster dynamics can (with different trained weights) recognize galaxies in images or denoise spectra.
:::

### Where Does Your Emulator Fit?

Your task: learn the mapping $(Q_0, a) \to (f_{\rm bound}, \sigma_v, r_h)$.

Is this relationship linear? Almost certainly not. The physics of gravitational collapse, violent relaxation, and stellar escape involves complex nonlinear dynamics. You *could* try to engineer features based on virial theorem scaling relations—but why guess when you can learn?

A neural network will discover whatever nonlinear transformations of $(Q_0, a)$ are useful for predicting cluster outcomes. If it turns out linear features suffice, the network can learn that too (it includes linear models as a special case).

---

## 7.3: The Computational Neuron

**Priority: 🔴 Essential**

### Anatomy of a Neuron

:::{margin}
**Artificial neuron**: A computational unit that computes a weighted sum of inputs, adds a bias, and applies a nonlinear activation function. Inspired by (but vastly simpler than) biological neurons.
:::

The artificial neuron is the atomic unit of neural networks. It takes $d$ inputs $x_1, x_2, \ldots, x_d$ and produces one output:

$$a = \sigma\left(\sum_{j=1}^{d} w_j x_j + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

where:
- $\mathbf{w} = (w_1, \ldots, w_d)$ are **weights** controlling how much each input matters
- $b$ is a **bias** allowing the neuron to shift its activation threshold
- $\sigma$ is a nonlinear **activation function**

Think of it as "how strongly should this neuron fire, given these inputs?" The weights determine sensitivity to each input; the bias sets the baseline; the activation function introduces nonlinearity.

:::{margin}
**Pre-activation**: The quantity $z = \mathbf{w}^T \mathbf{x} + b$ before the nonlinearity acts. Represents "how much this neuron wants to activate" before thresholding.
:::

The quantity $z = \mathbf{w}^T \mathbf{x} + b$ is called the **pre-activation**—the neuron's response before the nonlinear squashing. The activation function then decides the actual output: $a = \sigma(z)$.

<!-- VISUAL: Single neuron diagram
Show: multiple inputs x₁, x₂, ..., xₐ entering from left
Weights w₁, w₂, ..., wₐ on each connection
Summation node computing z = Σwᵢxᵢ + b
Activation function σ(z) producing output a
Label all components clearly
-->

### Why Nonlinearity Is Essential

:::{admonition} The Collapse of Linear Networks
:class: important

Without activation functions, a neuron computes a linear transformation: $a = \mathbf{w}^T \mathbf{x} + b$.

Stack two linear layers:
$$\text{Layer 1: } \mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
$$\text{Layer 2: } \mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

Substitute:
$$\mathbf{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \underbrace{(\mathbf{W}_2 \mathbf{W}_1)}_{\text{one matrix}} \mathbf{x} + \underbrace{(\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)}_{\text{one bias}}$$

No matter how many layers, you'd collapse to a single linear transformation. Deep networks would be no more powerful than shallow ones!

The nonlinearity $\sigma$ prevents this collapse. It's what allows networks to represent complex, nonlinear relationships—like how a cluster's bound fraction depends on initial virial ratio through the physics of violent relaxation and escape.
:::

### Activation Functions

Several activation functions are commonly used:

**ReLU (Rectified Linear Unit)**—the modern default:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

Beautifully simple: pass positive values unchanged, zero out negative values. Computationally cheap and works remarkably well. The gradient is 1 for $z > 0$ and 0 for $z < 0$, which helps training (no "vanishing gradients" for positive inputs).

**Sigmoid** (the "Fermi function" you know from statistical mechanics):
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Squashes outputs to $(0, 1)$. Useful for probabilities, but can cause vanishing gradients in deep networks (derivative is small when $|z|$ is large).

**Tanh**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Squashes to $(-1, 1)$. Zero-centered, which can help optimization. Same vanishing gradient concern as sigmoid.

<!-- VISUAL: Activation function comparison plot
Three panels showing ReLU, sigmoid, tanh
Each panel: function plot on left, derivative plot on right
Annotate key features: ReLU's "kink" at 0, sigmoid's saturation, tanh's zero-centering
-->

:::{margin}
**Historical note**: Early neural networks (1980s–90s) used sigmoid and tanh. ReLU was popularized around 2010 and revolutionized deep learning by alleviating the vanishing gradient problem. Sometimes the simplest solutions win.
:::

**For your emulator**: Use ReLU for hidden layers. It's the standard choice for regression and works well in practice.

---

## 7.4: Layering Neurons into Networks

**Priority: 🔴 Essential**

### The Multi-Layer Perceptron

:::{margin}
**Multi-Layer Perceptron (MLP)**: A neural network architecture consisting of an input layer, one or more hidden layers, and an output layer. Also called a "feed-forward" network because information flows in one direction (no loops).
:::

A single neuron has limited expressiveness. The power comes from *layering* neurons into a network:

**Input layer**: Your raw features $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$. Not really "neurons"—just the data entering the network.

**Hidden layer(s)**: Where the magic happens. Each layer transforms its input through weights, biases, and activations. A hidden layer with $n$ neurons computes:

$$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where $\mathbf{W} \in \mathbb{R}^{n \times d_{\text{in}}}$ is the weight matrix, $\mathbf{b} \in \mathbb{R}^n$ is the bias vector, and $\sigma$ applies element-wise.

**Output layer**: Produces final predictions $\hat{\mathbf{y}} \in \mathbb{R}^{d_{\text{out}}}$. For regression, typically no activation (linear output).

### Your Emulator Architecture

For the final project, you'll build:
- **Input**: 2 features $(Q_0, a)$
- **Hidden 1**: 64 neurons, ReLU activation
- **Hidden 2**: 64 neurons, ReLU activation  
- **Output**: 3 predictions $(f_{\text{bound}}, \sigma_v, r_h)$, linear (no activation)

:::{margin}
**Units reminder**: Ensure $\sigma_v$ uses consistent units with your Project 5 code (typically km/s or internal N-body units). The scale radius $a$ and half-mass radius $r_h$ should be in AU.
:::

<!-- VISUAL: Full architecture diagram for the 2-64-64-3 network
Show all layers with neuron counts
Indicate activation functions at each layer
Show example data flow: specific values of Q₀, a flowing through to predictions
Include dimension annotations on weight matrices
-->

```{mermaid}
flowchart LR
    subgraph Input["Input Layer (2)"]
        Q0["Q₀"]
        a["a"]
    end
    
    subgraph H1["Hidden Layer 1\n64 neurons, ReLU"]
        h1["h₁⁽¹⁾"]
        h2["⋮"]
        h3["h₆₄⁽¹⁾"]
    end
    
    subgraph H2["Hidden Layer 2\n64 neurons, ReLU"]
        h4["h₁⁽²⁾"]
        h5["⋮"]
        h6["h₆₄⁽²⁾"]
    end
    
    subgraph Output["Output Layer (3)\nLinear"]
        f["f_bound"]
        s["σᵥ"]
        r["rₕ"]
    end
    
    Q0 --> H1
    a --> H1
    H1 --> H2
    H2 --> Output
```

### Counting Parameters

:::{margin}
**Learnable parameters**: The weights and biases that training optimizes. The "knobs" the network adjusts to fit data.
:::

How many learnable parameters does your network have? This matters for understanding model complexity.

**Layer 1** (input → hidden 1):
- Weight matrix $\mathbf{W}_1 \in \mathbb{R}^{64 \times 2}$: $64 \times 2 = 128$ weights
- Bias vector $\mathbf{b}_1 \in \mathbb{R}^{64}$: 64 biases
- **Subtotal: 192 parameters**

**Layer 2** (hidden 1 → hidden 2):
- Weight matrix $\mathbf{W}_2 \in \mathbb{R}^{64 \times 64}$: $64 \times 64 = 4096$ weights
- Bias vector $\mathbf{b}_2 \in \mathbb{R}^{64}$: 64 biases
- **Subtotal: 4,160 parameters**

**Layer 3** (hidden 2 → output):
- Weight matrix $\mathbf{W}_3 \in \mathbb{R}^{3 \times 64}$: $3 \times 64 = 192$ weights
- Bias vector $\mathbf{b}_3 \in \mathbb{R}^{3}$: 3 biases
- **Subtotal: 195 parameters**

**Total: 4,547 parameters**

:::{admonition} 📗 Connection to Module 1: Degrees of Freedom and Overfitting
:class: warning

With 4,547 parameters and only ~100 training examples, alarm bells should ring. You have ~45× more parameters than data points!

In Module 1, you learned that model complexity must balance against data availability. Too many degrees of freedom and you fit noise, not signal—the model memorizes training data but fails on new inputs.

Why doesn't this doom neural networks? Several factors help:
1. **Implicit regularization**: Gradient descent with early stopping tends to find "simple" solutions
2. **Parameter sharing**: Many weight configurations produce similar functions
3. **Validation**: We monitor test performance to detect overfitting
4. **Normalization**: Keeps optimization well-behaved

Still, with limited data, you should expect some overfitting and use ensembles to quantify uncertainty.
:::

---

## 7.5: The Universal Approximation Theorem

**Priority: 🟡 Important**

### The Theoretical Foundation

:::{margin}
**Universal Approximation Theorem**: First proved by Cybenko (1989) for sigmoid activations; later extended to ReLU and other activations by Hornik, Leshno, and others.
:::

Here's a remarkable mathematical result that justifies using neural networks as flexible function approximators:

**Theorem (Universal Approximation)**: Let $f: K \to \mathbb{R}$ be any continuous function defined on a compact set $K \subset \mathbb{R}^d$. For any $\epsilon > 0$, there exists a neural network $\hat{f}$ with a single hidden layer and *sufficiently many neurons* such that:

$$\sup_{\mathbf{x} \in K} |f(\mathbf{x}) - \hat{f}(\mathbf{x})| < \epsilon$$

In words: any continuous function can be approximated arbitrarily well by a neural network with one hidden layer, if you use enough neurons.

**What this means for emulation**: Your N-body simulator defines a continuous function from initial conditions to summary statistics (assuming the summary statistics vary continuously with inputs, which they do for your problem). The theorem guarantees that *some* neural network can approximate this mapping to any desired accuracy.

### The Intuition: Building Functions from Simple Pieces

How can a network approximate arbitrary functions? 

**For sigmoid activations**: Each neuron creates a "soft step"—a smooth transition from low to high output. By positioning many soft steps at different locations with different heights, you can approximate any shape. It's like building a sculpture from many small clay pieces, each contributing a local bump or dip.

**For ReLU activations**: Each neuron creates a "hinge"—a bend in the function at some location. A network with $n$ ReLU neurons can create a piecewise linear function with up to $n$ hinges. With enough hinges, piecewise linear functions can approximate any continuous curve.

<!-- VISUAL: Universal approximation intuition
Panel 1: A complex target function (smooth curve)
Panel 2: Approximation with 3 ReLU neurons (crude, 3 segments)
Panel 3: Approximation with 10 ReLU neurons (better, 10 segments)
Panel 4: Approximation with 50 ReLU neurons (very good, nearly smooth)
Show how adding neurons improves approximation
-->

### The Crucial Caveats

:::{admonition} What the Theorem Does NOT Guarantee
:class: warning

**1. How many neurons you need**: The theorem says "sufficiently many" but the number could be astronomically large for complex functions. It's an existence theorem, not a construction.

**2. How to find the right weights**: Knowing a good network exists doesn't tell you how to train it. Optimization might get stuck in poor local minima.

**3. That training data suffices**: The theorem assumes you can evaluate the true function anywhere. In practice, you only have finite samples. Generalization from samples to the full function is a separate challenge.

**4. Behavior outside the training region**: The theorem applies on a *compact* set $K$. Outside this region, the network extrapolates with no guarantees.

**Universal approximation ≠ universal learning.** The theorem speaks to expressiveness (what networks *can* represent), not learnability (what we *can find* from data).
:::

---

## 7.6: Forward Propagation

**Priority: 🔴 Essential**

### Computing Predictions

:::{margin}
**Forward propagation** (or **forward pass**): The process of computing a network's output from its input by applying each layer's transformation sequentially.
:::

**Forward propagation** is how we compute predictions: apply each layer's transformation in sequence, feeding outputs forward.

For your 2-hidden-layer network, the forward pass computes:

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \quad \text{(pre-activation, layer 1)}$$
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1) \quad \text{(activation, layer 1)}$$
$$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2 \quad \text{(pre-activation, layer 2)}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{z}_2) \quad \text{(activation, layer 2)}$$
$$\hat{\mathbf{y}} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3 \quad \text{(output, linear)}$$

Each layer takes the previous layer's output, applies a linear transformation (matrix multiply + bias), and passes through an activation (except the final layer).

:::{admonition} Why No Activation on the Output Layer?
:class: note

For regression tasks (predicting continuous values like $f_{\text{bound}}$, $\sigma_v$, $r_h$), the output layer should be **linear—no activation function**.

Activations like sigmoid squash outputs to bounded ranges $(0, 1)$. If your targets can span ranges like $r_h \in [20, 150]$ AU, a sigmoid would artificially constrain predictions. ReLU would prevent negative predictions, which might be fine for $f_{\rm bound} \geq 0$ but problematic if residuals can go negative.

For *classification* tasks, you'd use sigmoid (binary) or softmax (multiclass) to produce probabilities. But your emulator does regression, so keep the output linear.
:::

### The Computational Graph Perspective

:::{margin}
**Computational graph**: A directed acyclic graph representing how computations compose. Nodes are operations or values; edges show data flow. JAX builds these automatically when tracing functions.
:::

Forward propagation defines a **computational graph**—exactly the structure JAX uses for automatic differentiation (Module 6!).

```{mermaid}
flowchart TB
    x["x (input)"] --> z1["z₁ = W₁x + b₁"]
    W1["W₁"] --> z1
    b1["b₁"] --> z1
    z1 --> h1["h₁ = ReLU(z₁)"]
    h1 --> z2["z₂ = W₂h₁ + b₂"]
    W2["W₂"] --> z2
    b2["b₂"] --> z2
    z2 --> h2["h₂ = ReLU(z₂)"]
    h2 --> y["ŷ = W₃h₂ + b₃"]
    W3["W₃"] --> y
    b3["b₃"] --> y
    y --> L["Loss ℒ"]
    ytrue["y_true"] --> L
```

This graph shows exactly how the output $\hat{\mathbf{y}}$ depends on the input $\mathbf{x}$ and all parameters $\boldsymbol{\theta} = (\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_3, \mathbf{b}_3)$.

When we train, we need gradients $\nabla_{\boldsymbol{\theta}} \mathcal{L}$. The computational graph tells JAX exactly how to compute them via the chain rule—that's what `jax.grad` does automatically.

---

## 🔬 Conceptual Checkpoint: Architecture Foundations

Before moving to training, verify your understanding:

:::{admonition} Self-Assessment
:class: important

1. **Nonlinearity**: If you removed all ReLU activations from your network, what class of functions could it represent? Why is this limiting?

2. **Parameter count**: A network has layers of sizes 10 → 50 → 50 → 5. How many total parameters (weights + biases)?

3. **Forward pass**: Given input $\mathbf{x} = [0.5, 1.0]^T$ and first layer weights $\mathbf{W}_1 = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}$ with bias $\mathbf{b}_1 = [0, -1]^T$, compute $\mathbf{h}_1$ (the first hidden layer output, with ReLU).

4. **Architecture choice**: Why does your emulator use ReLU for hidden layers but no activation for the output layer?

::::{dropdown} Check Your Answers
1. Only linear functions $\hat{\mathbf{y}} = \mathbf{A}\mathbf{x} + \mathbf{c}$. Limiting because most physical relationships are nonlinear.

2. Layer 1: $50 \times 10 + 50 = 550$. Layer 2: $50 \times 50 + 50 = 2550$. Layer 3: $5 \times 50 + 5 = 255$. **Total: 3,355 parameters.**

3. $\mathbf{z}_1 = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 0.5 \\ 1.0 \end{pmatrix} + \begin{pmatrix} 0 \\ -1 \end{pmatrix} = \begin{pmatrix} -0.5 \\ 1.0 \end{pmatrix}$. Then $\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1) = \begin{pmatrix} 0 \\ 1.0 \end{pmatrix}$.

4. ReLU provides nonlinearity so hidden layers can learn complex features. Output layer is linear because we're doing regression—predictions should be unconstrained real numbers.
::::
:::

---

## 7.7: Training as Optimization

**Priority: 🔴 Essential**

### The Loss Function

:::{margin}
**Loss function** (or **cost function**, **objective function**): A scalar function measuring how wrong the network's predictions are. Training minimizes this function.
:::

Training requires a measure of "how wrong" the network is. For regression, the standard choice is **mean squared error (MSE)**:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{y}}_i - \mathbf{y}_i^{\text{true}}\|^2$$

where $\boldsymbol{\theta}$ denotes all network parameters, $N$ is the number of training examples, and $\hat{\mathbf{y}}_i = f_{\text{network}}(\mathbf{x}_i; \boldsymbol{\theta})$ is the prediction for input $\mathbf{x}_i$.

The loss is a function of the parameters: different weights produce different predictions, which produce different errors. Training finds parameters that minimize the loss.

:::{admonition} 📗 Connection to Module 5: Maximum Likelihood
:class: note

In Bayesian inference, you maximize the log-posterior (or log-likelihood). In neural network training, you minimize a loss. These are closely related!

**Gaussian likelihood → MSE loss**

Assume observations are the true function plus Gaussian noise:
$$y_i = f_{\text{true}}(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

The log-likelihood is:
$$\ln p(\mathbf{y} | \boldsymbol{\theta}) = -\frac{1}{2\sigma^2} \sum_i (y_i - \hat{y}_i)^2 + \text{const}$$

Maximizing log-likelihood = minimizing $\sum_i (y_i - \hat{y}_i)^2$ = minimizing MSE.

**The deep connection**: Training a neural network with MSE loss is *maximum likelihood estimation* under a Gaussian noise model. You've seen this framework before—now it's driving optimization instead of sampling.
:::

### Gradient Descent

:::{margin}
**Gradient descent**: An iterative optimization algorithm that updates parameters in the direction opposite to the gradient (direction of steepest increase), thereby decreasing the objective function.
:::

We want $\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$. Gradient descent iteratively steps toward the minimum:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

where $\eta > 0$ is the **learning rate**—how large a step we take.

:::{margin}
**Learning rate** ($\eta$): The step size in gradient descent. Controls the tradeoff between speed (large $\eta$) and stability (small $\eta$). Choosing it well is crucial.
:::

**The gradient** $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ points in the direction of steepest *increase* in loss. We move in the *opposite* direction to decrease the loss.

<!-- VISUAL: Gradient descent on a 2D loss landscape
Show contour plot of loss as function of two parameters
Show gradient descent trajectory starting from random point
Annotate: gradient vectors pointing "uphill", steps going "downhill"
Show convergence toward minimum
Include inset showing effect of learning rate: too large (oscillates), too small (slow), just right
-->

:::{admonition} 📗 Connection to Module 5: Optimization vs. Sampling
:class: note

You've now seen three approaches to finding good parameters:

| Approach | What It Finds | Module |
|----------|---------------|--------|
| Analytical | Exact optimum (when possible) | Linear regression |
| Sampling (MCMC) | Full posterior distribution | Module 5, Project 4 |
| Optimization (GD) | Single point estimate (mode) | This module |

**Gradient descent finds one answer**—the parameter values at the (local) minimum. **MCMC explores the full posterior**—the entire distribution of plausible parameters.

For inference, we often want the full posterior. For training emulators, we usually just want a good point estimate. But we'll recover uncertainty via ensembles (Section 7.10).
:::

### Backpropagation = Chain Rule = Autodiff

:::{margin}
**Backpropagation**: The algorithm for computing gradients of the loss with respect to all network parameters. Mathematically, it's the chain rule applied systematically through the computational graph.
:::

**Backpropagation** computes gradients of the loss with respect to all network parameters. The key insight: **it's just the chain rule applied systematically.**

Consider how the loss depends on first-layer weights:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial \mathbf{W}_1}$$

The chain rule propagates gradients *backward* from the loss through each layer to the parameters.

:::{admonition} 📗 Connection to Module 6: You Already Know This!
:class: tip

In Module 6, you learned that JAX computes gradients via automatic differentiation—specifically, reverse-mode autodiff. **Backpropagation is exactly this**: reverse-mode autodiff applied to neural networks.

When you write:
```python
def loss_fn(params):
    pred = model(params, x)
    return jnp.mean((pred - y_true)**2)

gradients = jax.grad(loss_fn)(params)
```

JAX is performing backpropagation automatically. It traces the forward pass to build the computational graph, then applies the chain rule in reverse during `grad()`.

**You don't implement backprop by hand.** JAX does it for you. But understanding what it computes helps you debug and reason about training dynamics.
:::

### The Adam Optimizer

:::{margin}
**Adam** (Adaptive Moment Estimation): An optimizer that adapts learning rates for each parameter based on gradient history. Tracks running averages of gradients (first moment) and squared gradients (second moment).
:::

Plain gradient descent uses a fixed learning rate for all parameters. This can be suboptimal—some parameters benefit from larger steps, others from smaller.

**Adam** adapts the learning rate per parameter based on gradient history:
- **First moment** (exponential moving average of gradients): Which direction have we been moving?
- **Second moment** (exponential moving average of squared gradients): How variable have gradients been?

Parameters with consistently large gradients in the same direction get boosted. Parameters with noisy, inconsistent gradients get damped.

**In practice**: Adam is the default optimizer for neural networks. It works well across problems with minimal tuning. Start with learning rate $\eta = 10^{-3}$.

Optax provides Adam and many other optimizers.

---

## 7.8: Weight Initialization—Breaking Symmetry

**Priority: 🟡 Important**

### Why Initialization Matters

Before training begins, you must set initial values for all weights and biases. This choice profoundly affects training dynamics.

:::{admonition} The Symmetry Problem
:class: warning

**What if you initialize all weights to zero?**

Every neuron in a layer computes the same function (weighted sum of inputs with weights 0 → output 0). They all produce identical outputs. During backprop, they all receive identical gradients. They all update identically. They *stay* identical forever.

You've effectively reduced your 64-neuron layer to a single neuron. The network can't learn complex features because all neurons are copies of each other.

**Random initialization breaks this symmetry.** Different neurons start with different weights, compute different functions, receive different gradients, and specialize to detect different features.
:::

### Scale Matters Too

Even with random initialization, the *scale* of initial weights affects training:

**Weights too large**: Pre-activations $z = \mathbf{W}\mathbf{x} + \mathbf{b}$ become large. For sigmoid/tanh, this saturates the activation (gradient ≈ 0). For ReLU, values may explode through layers.

**Weights too small**: Signals shrink as they pass through layers. By the time they reach the output, everything is near zero. Gradients vanish.

:::{margin}
**Xavier/Glorot initialization**: Initialize weights with variance $\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ where $n_{\text{in}}$, $n_{\text{out}}$ are input/output dimensions of the layer. Keeps signal magnitudes stable across layers.
:::

**Xavier/Glorot initialization** (what Equinox uses by default) sets the right scale:

$$w_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

This keeps the variance of activations roughly constant across layers, enabling stable gradient flow.

### Why Different Seeds → Different Solutions

Neural network loss landscapes are **non-convex**—they have many local minima and saddle points. Gradient descent finds *a* minimum, not *the* global minimum. Which minimum you find depends on where you start.

<!-- VISUAL: Non-convex loss landscape illustration
Show a 1D or 2D loss function with multiple local minima
Show two gradient descent trajectories from different starting points
converging to different local minima
Annotate: "Different initializations → Different solutions"
-->

**Different random seeds produce different initial weights → different optimization trajectories → different final solutions.**

These solutions may all have similar training loss but differ in their predictions, especially in regions with sparse data. This is the foundation for ensemble uncertainty (Section 7.10): train multiple networks with different seeds and measure how much they disagree.

---

## 7.9: Practical Training

**Priority: 🔴 Essential**

### Data Normalization

:::{margin}
**Standardization** (z-score normalization): Transform each feature to zero mean and unit variance: $\tilde{x} = (x - \mu)/\sigma$. Essential for neural network training.
:::

Neural networks are sensitive to input/output scales. If $Q_0 \in [0.5, 1.5]$ but $a \in [50, 200]$, gradients with respect to $a$-related weights will dominate, causing slow and unstable learning.

**Standardization** transforms each feature to zero mean and unit variance:

$$\tilde{x} = \frac{x - \mu_x}{\sigma_x}$$

where $\mu_x$ and $\sigma_x$ are computed from your **training set only**.

:::{admonition} The Cardinal Rule of Normalization
:class: warning

Compute $\mu$ and $\sigma$ from the **training set only**. Apply the *same* transformation (using training statistics) to:

- Training inputs and outputs ✓
- Validation/test inputs and outputs ✓
- Any new data at inference time ✓

**Never** use test set statistics for normalization—that would be data leakage, using information from data you're supposed to predict.
:::

**Denormalization**: To convert predictions back to physical units:
$$x = \tilde{x} \cdot \sigma_x + \mu_x$$

### Multi-Output Normalization

:::{margin}
**Multi-output regression**: Predicting multiple target variables simultaneously. Each output should typically be normalized independently.
:::

Your emulator predicts three quantities: $(f_{\text{bound}}, \sigma_v, r_h)$. The MSE loss sums over all outputs:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[ (\hat{f}_{\text{bound},i} - f_{\text{bound},i})^2 + (\hat{\sigma}_{v,i} - \sigma_{v,i})^2 + (\hat{r}_{h,i} - r_{h,i})^2 \right]$$

**Problem**: If outputs have different scales, this implicitly weights them unequally. If $r_h$ varies from 20–150 AU (range 130) while $f_{\text{bound}}$ varies from 0.3–1.0 (range 0.7), squared errors for $r_h$ are ~35,000× larger. The network will focus on $r_h$ at the expense of $f_{\text{bound}}$.

**Solution**: Normalize each output independently to zero mean, unit variance. Then all three contribute roughly equally to the loss.

**The normalization pattern** (you'll implement this):

```
NORMALIZATION PROCEDURE
=======================

1. COMPUTE STATISTICS (from training set only!)
   μ = mean of training data, per feature    # Shape: (3,) for outputs
   σ = std of training data, per feature     # Shape: (3,) for outputs

2. NORMALIZE any data (train, test, or new):
   x̃ = (x - μ) / σ
   
3. DENORMALIZE predictions back to physical units:
   x = x̃ · σ + μ
```

**Critical**: Compute μ and σ from training data, then use those *same values* everywhere—for normalizing test data, for denormalizing predictions, and inside your NumPyro model. Store these statistics alongside your trained model.

### Hyperparameters

:::{margin}
**Hyperparameter**: A setting that controls training but isn't learned from data. Must be chosen before training begins.

**Epoch**: One complete pass through all training data. If you have 100 training examples and train for 500 epochs, the network sees each example 500 times.
:::

Key hyperparameters for your emulator:

| Hyperparameter | Starting Value | Adjustment Guidance |
|----------------|----------------|---------------------|
| Learning rate | $10^{-3}$ | Decrease by 10× if loss oscillates; increase if loss decreases too slowly |
| Epochs | 500–1000 | Until loss plateaus on validation set |
| Hidden layers | 2 | More layers rarely help for smooth, low-dimensional functions |
| Neurons per layer | 64 | Try 32 or 128 if 64 seems too small/large |
| Batch size | Full batch | With ~100 samples, no need for mini-batches |

### Train/Test Split

With limited data (~100 simulations), use an **80/20 split**: 80 examples for training, 20 for testing. 

:::{admonition} The Test Set Is Sacred
:class: warning

Never use test data for *any* training decisions:
- Don't tune hyperparameters based on test performance
- Don't normalize using test statistics
- Don't stop training based on test loss

The test set exists solely to evaluate your final model. If you peek at it during development, you lose your ability to honestly assess generalization.
:::

### Convergence Criteria

Training is "done" when:
1. **Loss has plateaued**: Not decreasing meaningfully over ~100 epochs
2. **Loss is reasonably small**: MSE $\lesssim 0.01$ on normalized data means typical errors < 10% of standard deviation
3. **No pathologies**: No NaN values, no loss increasing

### The Training Loop

:::{margin}
**Training loop**: The iterative process of computing predictions, calculating loss, computing gradients, and updating parameters. Repeats for many epochs until convergence.
:::

A typical training loop:

```python
# Pseudocode
losses = []
for epoch in range(num_epochs):
    # Forward pass: compute predictions
    predictions = model(params, x_train)
    
    # Compute loss
    loss = jnp.mean((predictions - y_train)**2)
    losses.append(loss)
    
    # Backward pass: compute gradients
    gradients = jax.grad(loss_fn)(params)
    
    # Update parameters
    params = update_with_adam(params, gradients)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")

# ALWAYS plot the loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
```

**Always plot the loss curve.** Its shape reveals problems that printed numbers hide.

<!-- VISUAL: Loss curve diagnostics
Four panels showing different loss curve shapes:
1. "Healthy training" - smooth decrease, then plateau
2. "Learning rate too high" - oscillating wildly
3. "Learning rate too low" - decreasing very slowly
4. "Overfitting" - training loss decreases, validation loss increases
Annotate each with diagnosis and fix
-->

### Debugging Checklist

:::{admonition} When Training Goes Wrong
:class: important

**Before training—verify setup:**
- [ ] Normalization correct? Print `mean(x_norm)` ≈ 0, `std(x_norm)` ≈ 1
- [ ] Shapes match? Print input shape, output shape, prediction shape
- [ ] Forward pass works? Call `model(x)` once, check for NaN, verify output shape
- [ ] Loss computable? Compute loss once, verify it's a reasonable positive number

**During training—monitor health:**
- [ ] Loss decreasing? Should drop significantly in first 10 epochs
- [ ] No NaN? Add `assert not jnp.any(jnp.isnan(loss))`
- [ ] Plot loss curve (don't just print values!)

**After training—validate results:**
- [ ] Training vs. test loss? Large gap suggests overfitting
- [ ] Predicted vs. true plots? Points should cluster along $y = x$
- [ ] Residual patterns? Systematic errors indicate model limitations
- [ ] Predictions sensible? Check a few examples manually

**Common fixes:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss stays constant | Learning rate too small | Increase by 10× |
| Loss oscillates wildly | Learning rate too large | Decrease by 10× |
| Loss becomes NaN | Numerical instability | Lower LR, check for division by zero |
| Loss plateaus high | Architecture too simple, or bug | Add neurons/layers, check code |
| Train loss good, test loss bad | Overfitting | Early stopping, more data, or accept it |
:::

---

## 7.10: Uncertainty via Ensembles

**Priority: 🔴 Essential**

### The Problem with Single Networks

A trained neural network gives **point predictions**—single numbers with no uncertainty. But scientific applications need uncertainty quantification:

- How confident is this prediction?
- Are we extrapolating outside the training distribution?
- How should prediction uncertainty propagate to inference?

A single network provides none of this. Worse, networks can be *confidently wrong*, especially when extrapolating beyond training data.

### Deep Ensembles

:::{margin}
**Ensemble**: A collection of models that make independent predictions. Disagreement among members quantifies uncertainty. For neural networks, different random initializations suffice to create an ensemble.
:::

The simplest approach to neural network uncertainty: **train multiple networks with different random seeds.**

Why does this work? From Section 7.8: different initializations lead to different local minima. These solutions may give similar predictions where training data is dense but diverge where data is sparse.

**Procedure:**
1. Train $M$ networks with different random seeds (typically $M = 3$–5)
2. For any input $\mathbf{x}$:
   - Get predictions from each: $\hat{y}_1(\mathbf{x}), \ldots, \hat{y}_M(\mathbf{x})$
   - **Mean prediction**: $\bar{y}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m(\mathbf{x})$
   - **Uncertainty**: $\sigma_y(\mathbf{x}) = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} (\hat{y}_m(\mathbf{x}) - \bar{y}(\mathbf{x}))^2}$

:::{margin}
**Epistemic uncertainty**: Uncertainty due to limited knowledge—here, limited training data. Reducible with more data. Distinct from **aleatoric uncertainty** (irreducible noise in the process itself).
:::

The spread $\sigma_y$ estimates **epistemic uncertainty**—uncertainty from limited training data.

<!-- VISUAL: Ensemble predictions visualization
Show 1D slice: x-axis is Q₀ (holding a fixed), y-axis is f_bound prediction
Plot: training points as dots
Plot: 5 individual ensemble member predictions as thin lines (slightly different)
Plot: ensemble mean as thick line
Plot: shaded ±2σ uncertainty band
Show uncertainty is larger where training points are sparse
-->

### Interpreting Ensemble Spread

**Uncertainty should be high:**
- Near edges of training distribution (extrapolation begins)
- In gaps between training points
- Where the function is complex/rapidly varying

**Uncertainty should be low:**
- In regions densely covered by training data
- Where all ensemble members agree
- Where the function is simple/slowly varying

:::{admonition} 📗 Connection to Module 5: Epistemic vs. Aleatoric
:class: note

Remember from Module 5:

**Epistemic uncertainty** is reducible with more data. It represents "we're uncertain because we haven't seen enough examples." Ensemble spread captures this—with more training data, ensemble members would agree better.

**Aleatoric uncertainty** is irreducible noise in the process itself. For your emulator, this includes stochastic variation from different IMF realizations—even with identical $(Q_0, a)$, different random stellar masses produce different outcomes.

Ensembles capture epistemic uncertainty but *not* aleatoric uncertainty. The ensemble members all learned from the same training data, so they share the same aleatoric floor. More sophisticated approaches (predicting variance directly, Bayesian neural networks) can capture both, but ensembles are a practical first step.
:::

### Ensemble in Practice

The ensemble workflow is conceptually simple:

```
ENSEMBLE WORKFLOW
=================

Training:
1. Split random key into M independent keys
2. Train M models, each with a different key
3. Store all trained models

Prediction:
1. Get predictions from all M models
2. Stack predictions into array of shape (M, num_outputs)
3. Mean = jnp.mean(predictions, axis=0)
4. Std = jnp.std(predictions, axis=0, ddof=1)  # ddof=1 for unbiased
```

**Implementation hints**:
- `jax.random.split(key, M)` creates M independent keys
- Store models in a Python list
- For single input: `preds = jnp.array([model(x) for model in models])`
- For batched inputs: use `jax.vmap` inside the list comprehension

For your final project, $M = 5$ ensemble members is a good balance between computational cost and uncertainty quality. You'll implement this yourself—the pattern above tells you *what* to compute, not *how* to code it.

---

## 🔬 Conceptual Checkpoint: Training and Uncertainty

Before moving to implementation, verify your understanding:

:::{admonition} Self-Assessment
:class: important

1. **Loss interpretation**: If your MSE loss on normalized data is 0.04, roughly how large are typical prediction errors relative to output standard deviations?

2. **Learning rate**: Describe what happens to the loss curve if the learning rate is (a) too large, (b) too small, (c) well-chosen.

3. **Normalization**: Why must you use *training set* statistics (not test set) when normalizing test data?

4. **Multi-output**: Your emulator predicts three outputs. Before normalization, $r_h$ has 100× larger variance than $f_{\rm bound}$. What problem does this cause, and how do you fix it?

5. **Ensembles**: You train 5 networks on the same data but with different random seeds. Why do they converge to different solutions?

6. **Uncertainty interpretation**: Your ensemble predicts $r_h = 85 \pm 3$ AU for one input and $r_h = 120 \pm 20$ AU for another. What does the larger uncertainty in the second case tell you?

::::{dropdown} Check Your Answers
1. MSE = 0.04 means RMSE = 0.2. On normalized data (std = 1), typical errors are ~20% of one standard deviation—quite good.

2. (a) Too large: loss oscillates wildly, may diverge. (b) Too small: loss decreases very slowly, may not reach good minimum in reasonable time. (c) Well-chosen: smooth decrease, then plateau.

3. Using test statistics would be data leakage—incorporating information from data you're supposed to predict. The model must work with only training-time information.

4. Without normalization, the loss is dominated by $r_h$ (larger squared errors). The network focuses on $r_h$ at the expense of $f_{\rm bound}$. Fix: normalize each output independently to unit variance.

5. The loss landscape has many local minima. Different initial weights lead to different optimization trajectories, ending at different minima with similar loss but different predictions.

6. The second prediction has higher epistemic uncertainty. The ensemble members disagree more, likely because that input region had sparser training data. You should be less confident in the second prediction.
::::
:::

---

## 7.11: The JAX Ecosystem—Equinox and Optax

**Priority: 🔴 Essential**

### Equinox: Neural Networks as PyTrees

:::{margin}
**Equinox**: A JAX library for neural networks by Patrick Kidger. Models are PyTrees (nested containers JAX understands), enabling seamless use of `jax.grad`, `jax.jit`, `jax.vmap`.
:::

Equinox makes neural networks fit naturally into JAX's functional paradigm:

**Models are PyTrees**: Parameters live in a tree structure. JAX transformations (`grad`, `jit`, `vmap`) work seamlessly because they know how to traverse PyTrees.

**Models are callables**: An Equinox model is a function. You call it: `output = model(input)`.

**Explicit state**: No hidden mutable variables. All parameters are explicit. This makes debugging easier and enables JAX's transformations.

### Learning from Documentation

:::{admonition} Your Primary Resources
:class: important

The final project requires you to learn Equinox and Optax from their documentation. This is intentional—**reading documentation and adapting examples is a core professional skill.**

**Equinox** (https://docs.kidger.site/equinox/):
- **"Getting Started"**: How to define an `eqx.Module` with `__init__` and `__call__`
- **"Train a Neural Network"**: Complete training loop pattern you'll adapt

**Optax** (https://optax.readthedocs.io/):
- **"Quick Start"**: How to create optimizers and apply updates

**Your task**: Work through these tutorials *before* starting your emulator. Run the examples. Modify them. Break them and fix them. Then adapt the patterns to your problem.
:::

### Glass-Box Exercise: Manual Forward Pass

Before using any library, make sure you understand what's happening inside:

:::{admonition} 🔧 Build It Yourself First
:class: tip, dropdown

Implement a forward pass manually using only JAX primitives:

```python
import jax.numpy as jnp

def manual_forward(params, x):
    """Forward pass through a 2-64-64-3 MLP.
    
    params: dict with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'
    x: input array of shape (2,)
    Returns: output array of shape (3,)
    """
    # Layer 1: linear transformation + ReLU
    z1 = params['W1'] @ x + params['b1']
    h1 = jnp.maximum(0, z1)
    
    # Layer 2: linear transformation + ReLU
    z2 = params['W2'] @ h1 + params['b2']
    h2 = jnp.maximum(0, z2)
    
    # Output layer: linear only (no activation)
    y = params['W3'] @ h2 + params['b3']
    return y
```

**Verify that JAX can differentiate this:**
```python
def loss_fn(params, x, y_true):
    y_pred = manual_forward(params, x)
    return jnp.mean((y_pred - y_true)**2)

grad_fn = jax.grad(loss_fn)  # This works!
```

This exercise shows that neural networks are just compositions of simple operations. Equinox handles the bookkeeping—but you should understand what it's managing.
:::

### A Teaching Example: Learning $\sin(x)$

To illustrate Equinox patterns *without* solving your final project, let's build a tiny network that learns the sine function. **This is NOT your emulator architecture—adapt the concepts, don't copy the code.**

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# A SIMPLE 1D example - your emulator will differ!
class TinySinNet(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(1, 16, key=k1)  # 1 input
        self.layer2 = eqx.nn.Linear(16, 1, key=k2)  # 1 output
    
    def __call__(self, x):
        x = jax.nn.relu(self.layer1(x))
        return self.layer2(x)
```

:::{admonition} How Your Emulator Differs
:class: warning

TinySinNet has **2 layers** (1→16→1) with only one hidden layer.

Your emulator needs **3 layers** (2→64→64→3):
- 2 inputs, not 1
- Two hidden layers of 64 neurons each, not one layer of 16
- 3 outputs, not 1
- ReLU activation after *each* hidden layer

Adapt the *pattern* (how to define `eqx.Module`, split keys, write `__call__`), not the specific architecture.
:::

**Key patterns to notice** (these transfer to your emulator):

1. **Inherit from `eqx.Module`** — makes your class a PyTree
2. **Declare layers as typed attributes** — Equinox tracks these
3. **Split keys for each layer** — ensures independent initialization
4. **`__call__` defines forward pass** — apply layers and activations

### Essential Equinox Patterns

These small patterns are the building blocks. You'll combine them for your emulator.

**Pattern 1: Creating layers**
```python
# eqx.nn.Linear(in_features, out_features, key=...)
layer = eqx.nn.Linear(64, 32, key=some_key)
```

**Pattern 2: Applying activations**
```python
# Activations are just JAX functions
x = jax.nn.relu(layer(x))  # ReLU
x = jax.nn.tanh(layer(x))  # Tanh (if you needed it)
```

**Pattern 3: Filtering arrays from models**
```python
# Equinox models contain arrays (parameters) and non-arrays (structure)
# Many operations need just the arrays:
params = eqx.filter(model, eqx.is_array)
```

**Pattern 4: Computing gradients**
```python
# eqx.filter_value_and_grad works like jax.value_and_grad
# but handles models with non-array components
loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
```

**Pattern 5: Applying updates**
```python
# After computing updates from optimizer:
new_model = eqx.apply_updates(model, updates)
```

**Pattern 6: JIT-compiling with models**
```python
# Use eqx.filter_jit instead of jax.jit for functions involving eqx.Module
@eqx.filter_jit
def train_step(model, ...):
    ...
```

### The Training Loop Structure

Here's the *structure* of a training loop—you'll fill in the details:

```
TRAINING LOOP PSEUDOCODE
========================

1. INITIALIZE
   - Create model with random key
   - Create optimizer (e.g., optax.adam)
   - Initialize optimizer state from model parameters

2. FOR EACH EPOCH:
   a. FORWARD PASS
      - Compute predictions: pred = model(x_train)  
      - (Use jax.vmap to handle batches)
   
   b. COMPUTE LOSS
      - MSE: loss = mean((pred - y_train)²)
   
   c. BACKWARD PASS  
      - Compute gradients w.r.t. model parameters
      - (Use eqx.filter_value_and_grad)
   
   d. UPDATE
      - Get updates from optimizer
      - Apply updates to model
   
   e. LOG
      - Store loss for plotting
      - Print progress periodically

3. PLOT LOSS CURVE (always!)

4. RETURN trained model
```

**Your task**: Translate this pseudocode into working Equinox/Optax code. The documentation tutorials show exactly how.

### Training an Ensemble: The Idea

For uncertainty quantification, you'll train multiple networks:

```
ENSEMBLE PSEUDOCODE
===================

1. Split your random key into M keys (one per model)

2. FOR EACH key:
   - Train a model using that key
   - (Different key → different initialization → different solution)
   - Store the trained model

3. TO PREDICT with uncertainty:
   - Get prediction from each model
   - Mean = average of predictions
   - Uncertainty = standard deviation of predictions
```

**Implementation hints**:
- `jax.random.split(key, n)` gives you n independent keys
- Store models in a Python list
- For ensemble prediction, iterate over models and stack results with `jnp.array([...])`
- Use `jnp.mean(..., axis=0)` and `jnp.std(..., axis=0, ddof=1)` to aggregate

:::{admonition} What You Must Figure Out
:class: warning

The final project skeleton provides function signatures with `# TODO` comments. You need to:

1. **Define your model class** — How many layers? What dimensions? Where do activations go?

2. **Write the training step** — How do you combine `eqx.filter_value_and_grad`, `optimizer.update`, and `eqx.apply_updates`?

3. **Handle batches** — When do you use `jax.vmap`?

4. **Train the ensemble** — How do you manage multiple models with different initializations?

The Equinox "Train a Neural Network" tutorial shows a complete example. Study it, understand it, then adapt it—don't copy it blindly.
:::

:::{admonition} Saving Your Trained Models
:class: tip

You'll need to save trained models for use in the inference phase. Equinox provides serialization via `eqx.tree_serialise_leaves` and `eqx.tree_deserialise_leaves`. 

**Important**: Save your normalization statistics (mean, std for inputs and outputs) alongside your models—you'll need them for inference. A simple approach is to save everything in a dictionary or use separate files.

See the Equinox documentation on serialization for details.
:::

---

## 7.12: From Emulator to Inference

**Priority: 🔴 Essential**

### The Goal

You've trained a fast emulator. Now use it for Bayesian inference: given observed cluster properties, infer the initial conditions that produced them.

The emulator replaces the expensive N-body simulation inside your inference loop. NumPyro's NUTS sampler will call it thousands of times—this is why speed matters.

### The Conceptual Framework

:::{margin}
**Probabilistic programming**: A paradigm where you write a generative model (how data is produced from parameters), and the framework handles inference automatically.
:::

In NumPyro, you write a **generative model**—code describing how observations arise from parameters. The structure directly mirrors Bayes' theorem:

```
GENERATIVE MODEL STRUCTURE
==========================

1. PRIOR: What do we believe about parameters before seeing data?
   - Sample Q₀ from some distribution (e.g., Uniform over training range)
   - Sample a from some distribution
   
2. FORWARD MODEL: Given parameters, what do we predict?
   - Normalize the sampled (Q₀, a) using TRAINING statistics
   - Pass through your emulator to get predicted (f_bound, σᵥ, rₕ)
   - Denormalize predictions back to physical units
   
3. LIKELIHOOD: How probable are actual observations given predictions?
   - Compare predictions to observations
   - Account for observation uncertainty (σ_obs)
```

**The key insight**: Your emulator serves as the forward model. Instead of running an expensive N-body simulation for each $(Q_0, a)$ the sampler proposes, you call your fast neural network.

### What You Need to Figure Out

:::{admonition} NumPyro Learning Path
:class: important

The final project requires you to build a NumPyro model. Here's what to learn:

**NumPyro documentation** (https://num.pyro.ai/):
- **"Getting Started"**: Basic model structure with `numpyro.sample`
- **"Bayesian Regression"**: Example closest to your task
- **MCMC and NUTS**: How to run inference

**Key NumPyro primitives you'll use**:
- `numpyro.sample("name", distribution)` — sample a parameter from a prior
- `numpyro.sample("obs", distribution, obs=data)` — define likelihood given observed data
- `dist.Uniform(low, high)` — uniform prior
- `dist.Normal(loc, scale)` — Gaussian likelihood
- `NUTS(model)` — the sampler (you know this from Module 5!)
- `MCMC(kernel, num_warmup, num_samples)` — run the chain
:::

### The Normalization Challenge

Your emulator was trained on normalized data. Your NumPyro model must handle this correctly:

```
NORMALIZATION FLOW
==================

Physical parameters (Q₀, a) from prior
          ↓
    Normalize using TRAINING statistics
          ↓
Normalized inputs to emulator
          ↓
    Emulator predicts normalized outputs
          ↓
    Denormalize using TRAINING statistics
          ↓
Physical predictions (f_bound, σᵥ, rₕ)
          ↓
    Compare to physical observations in likelihood
```

:::{admonition} Critical Detail
:class: warning

You must use the **same normalization statistics** (mean and std from training set) that you used during training. Store these when you train your emulator, and pass them to your inference model.

A common bug: accidentally using different statistics, which silently produces wrong predictions.
:::

### Choosing Observation Uncertainty

The likelihood requires $\sigma_{\rm obs}$—how uncertain are our "observations"? For synthetic observations from your simulations, this represents:

1. **Emulator error**: The network doesn't perfectly predict simulations
2. **Stochastic variation**: Different random seeds produce scatter even for fixed $(Q_0, a)$

**Practical approach**: Use your test-set RMSE for each output.

```
COMPUTING σ_obs
===============

1. Get emulator predictions on test set (normalized)
2. Denormalize to physical units
3. Compare to true test values
4. RMSE = sqrt(mean((pred - true)²)) for each output
5. Use these three RMSE values as σ_obs
```

:::{admonition} Uncertainty Calibration
:class: note

If $\sigma_{\rm obs}$ is too small, posteriors will be overconfident and may miss true values. If too large, posteriors will be vague and uninformative.

Test set RMSE is a principled starting point. You can check calibration: do ~95% of your validation posteriors contain the true parameter values within their 95% credible intervals?
:::

### Validation: Parameter Recovery

The critical test of your inference pipeline:

```
PARAMETER RECOVERY TEST
=======================

1. Pick a simulation from your TEST set
   - You know the true (Q₀, a) that generated it
   - You have its summary statistics (f_bound, σᵥ, rₕ)

2. Treat summary statistics as "observations"

3. Run inference to get posterior over (Q₀, a)

4. Check: Do 95% credible intervals contain true values?
   - Compute 2.5th and 97.5th percentiles of posterior samples
   - True value should fall within this range
   
5. Repeat for several test cases
   - If ~95% contain truth, your pipeline is well-calibrated
   - If fewer, σ_obs may be too small (overconfident)
   - If more, σ_obs may be too large (uninformative)
```

**Visualization**: Create corner plots showing the joint posterior over $(Q_0, a)$ with true values marked. The true values should fall within the 95% contours.

<!-- VISUAL: Corner plot example
Show 2D posterior for (Q₀, a) with:
- Marginal distributions on diagonal
- Joint distribution in off-diagonal
- True values marked with crosshairs
- 68% and 95% contours visible
Annotate: "True values should fall within contours"
-->

### Connecting the Pieces

Here's how all the components fit together in your final project:

```
COMPLETE INFERENCE PIPELINE
===========================

┌─────────────────────────────────────────────────────────┐
│  TRAINING PHASE (done before inference)                 │
├─────────────────────────────────────────────────────────┤
│  1. Generate N-body training data                       │
│  2. Compute and store normalization statistics          │
│  3. Train ensemble of emulators                         │
│  4. Evaluate on test set → get σ_obs                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  INFERENCE PHASE                                        │
├─────────────────────────────────────────────────────────┤
│  1. Define NumPyro model:                               │
│     - Priors on (Q₀, a)                                 │
│     - Emulator as forward model (with normalization!)   │
│     - Gaussian likelihood with σ_obs                    │
│                                                         │
│  2. Run NUTS:                                           │
│     - Warmup: adapts step size and mass matrix          │
│     - Sampling: collects posterior samples              │
│                                                         │
│  3. Analyze posterior:                                  │
│     - Summary statistics (mean, std, credible intervals)│
│     - Corner plots                                      │
│     - Parameter recovery validation                     │
└─────────────────────────────────────────────────────────┘
```

:::{admonition} Why This Is Fast
:class: tip

Remember the calculation from Section 7.1? Direct inference with N-body simulations would take ~35 days.

With your emulator:
- NUTS calls the forward model ~100,000 times (warmup + samples × leapfrog steps)
- Each emulator call: ~10 μs (JIT-compiled)
- Total: ~1 second of emulator evaluation

The bottleneck becomes NUTS overhead, not the forward model. Your inference runs in minutes, not months.

**This is why JIT compilation matters**: the `@eqx.filter_jit` decorator on your model makes each call ~1000× faster than uncompiled Python.
:::

---

## 7.13: Synthesis

**Priority: 🔴 Essential**

### The Complete Workflow

You've learned a powerful workflow for modern computational science:

```{mermaid}
flowchart TD
    subgraph Data["1. Generate Training Data"]
        sims["Run N-body simulations\n(Latin Hypercube in Q₀, a)"]
        stats["Extract summary statistics\n(f_bound, σᵥ, rₕ)"]
        sims --> stats
    end
    
    subgraph Prep["2. Data Preparation"]
        split["Train/test split"]
        norm["Normalize to zero mean,\nunit variance"]
        split --> norm
    end
    
    subgraph Arch["3. Design Architecture"]
        mlp["MLP: 2 → 64 → 64 → 3"]
        act["ReLU hidden, linear output"]
    end
    
    subgraph Train["4. Train Network"]
        init["Xavier initialization"]
        opt["Adam optimizer"]
        loss["MSE loss"]
        init --> opt --> loss
    end
    
    subgraph Ens["5. Build Ensemble"]
        multi["Train 5 models\n(different seeds)"]
        uncert["Predict: mean ± std"]
        multi --> uncert
    end
    
    subgraph Eval["6. Evaluate"]
        test["Test set accuracy"]
        plots["Predicted vs. true"]
        edges["Edge behavior"]
    end
    
    subgraph Infer["7. Inference"]
        numpyro["NumPyro model:\nprior → emulator → likelihood"]
        nuts["NUTS sampling"]
        post["Posterior over (Q₀, a)"]
        numpyro --> nuts --> post
    end
    
    Data --> Prep --> Arch --> Train --> Ens --> Eval --> Infer
```

### Connections Across the Course

| Concept | Where You Learned It | How It Appears Here |
|---------|---------------------|---------------------|
| Models as compression | Module 1 | Emulator compresses input→output relationship into weights |
| Moments & summary statistics | Module 1 | $f_{\rm bound}$, $\sigma_v$, $r_h$ compress simulation output |
| Likelihood & MSE | Module 5 | MSE loss = Gaussian likelihood maximization |
| Gradient-based inference | Module 5 (HMC) | Training uses gradients; NUTS uses gradients |
| Automatic differentiation | Module 6 | Backprop = reverse-mode autodiff (JAX handles it) |
| JIT compilation | Module 6 | Essential for fast emulator evaluation in inference |
| Functional programming | Module 6 | Equinox models are pure functions with explicit state |
| N-body dynamics | Module 3, Projects 2 & 5 | The physics your emulator learns to approximate |

### What Makes a Good Emulator?

**Accuracy**: Predictions match simulations in the training region
- Test RMSE should be small relative to output variation
- Predicted vs. true plots cluster along $y = x$

**Calibration**: Uncertainty estimates are meaningful  
- ~95% of true values fall within ±2σ of predictions
- Uncertainty increases where training data is sparse

**Speed**: Fast enough for inference
- Microseconds per prediction (vs. seconds for simulation)
- JIT compilation is essential

**Robustness**: Sensible behavior at boundaries
- No wild predictions just outside training range
- Ensemble uncertainty increases appropriately

### Looking Ahead: The Final Project

You're now ready for the final project. You'll:

1. **Generate training data**: Latin Hypercube sampling in $(Q_0, a)$ space; compute summary statistics from N-body simulations using your Project 5 code

2. **Build and train an emulator**: MLP in Equinox; train ensemble with Optax

3. **Evaluate accuracy**: Test metrics, predicted vs. true plots, uncertainty behavior

4. **Perform inference**: NumPyro model with emulator as forward model; NUTS sampling; corner plots showing parameter recovery

This workflow—expensive simulations → fast surrogates → Bayesian inference—is exactly how frontier research operates. You're learning it from the inside out.

---

## 🔬 Final Conceptual Checkpoint

Before starting the final project, ensure you can answer:

:::{admonition} Comprehensive Self-Assessment
:class: important

**Architecture & Theory:**
1. What is the role of activation functions? What happens without them?
2. What does the Universal Approximation Theorem guarantee? What doesn't it guarantee?
3. How many parameters does a network with layers 5 → 100 → 100 → 3 have?

**Training:**
4. Explain how backpropagation relates to `jax.grad` and Module 6's autodiff.
5. Why must normalization use only training set statistics?
6. Your training loss decreases but validation loss increases after epoch 200. What's happening?
7. What hyperparameters would you adjust if loss oscillates wildly?

**Uncertainty:**
8. Why do different random seeds produce different trained networks?
9. What type of uncertainty do ensembles capture? What don't they capture?
10. Where would you expect ensemble uncertainty to be highest?

**Integration:**
11. In your NumPyro model, what role does the emulator play?
12. How do you choose $\sigma_{\rm obs}$ for the likelihood?
13. How would you check if your inference is working correctly?

**Practical:**
14. What does `eqx.filter_jit` do differently from `jax.jit`?
15. What is the purpose of `jax.vmap(model)` in the training step?
16. Why is the output layer linear (no activation) for your emulator?
:::

---

## Further Reading and Resources

### Essential (Complete Before Final Project)

- **Equinox documentation**: https://docs.kidger.site/equinox/
  - "Getting Started"
  - "Train a Neural Network" example
  
- **Optax documentation**: https://optax.readthedocs.io/
  - "Quick Start" tutorial

- **NumPyro documentation**: https://num.pyro.ai/
  - "Getting Started"
  - "Bayesian Regression" example

### For Deeper Understanding

- **Nielsen, Michael**: *Neural Networks and Deep Learning*
  - Free online: http://neuralnetworksanddeeplearning.com/
  - Excellent intuitive explanations
  
- **Goodfellow, Bengio, Courville**: *Deep Learning* (2016)
  - Free online: https://www.deeplearningbook.org/
  - Chapters 6–8 cover feed-forward networks
  
- **Mehta et al.**: "A high-bias, low-variance introduction to Machine Learning for physicists" (Physics Reports, 2019)
  - https://arxiv.org/abs/1803.08823
  - Machine learning from a physicist's perspective

### Emulation in Astrophysics

- **Cranmer et al.** (2020): "The frontier of simulation-based inference"
  - https://arxiv.org/abs/1911.01429
  - Overview of ML for scientific simulation

- **Alsing et al.** (2019): "Fast likelihood-free cosmology with neural density estimators and active learning"
  - Example of emulator-based inference in cosmology

---

**Now go build something that learns.** 🚀

---

**Next**: Proceed to Final Project, or continue with Gaussian Processes (optional advanced topic)
