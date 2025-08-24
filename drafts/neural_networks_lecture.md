# Chapter 6: Neural Networks - From Function Approximation to Representation Learning

## Prelude: Where We've Been and Where We're Going

Throughout our journey, we've progressively abstracted our understanding of computation. We began with deterministic physics—stars evolving according to known equations. We discovered that numerical methods introduce unavoidable approximations. We embraced randomness through Monte Carlo methods, turning intractable integrals into sampling problems. We learned to quantify uncertainty through Bayesian inference, treating parameters as probability distributions rather than fixed values. With Gaussian Processes, we took the ultimate leap, placing probability distributions over entire functions.

Now we arrive at neural networks, which might seem like a step backward. After all, we're returning to parametric models with fixed architectures. But this apparent regression hides a profound insight: neural networks don't just fit functions—they learn representations. They discover which features matter, how to combine them, and what patterns exist in data that we never explicitly programmed. This is the difference between interpolation and understanding.

---

# Part I: The Fundamental Problem - Why We Need Neural Networks

## 1.1 The Limitations of What We've Built

### The Curse of Dimensionality

Gaussian Processes are mathematically elegant and theoretically optimal, but they face a crushing computational burden. The GP requires inverting an $n \times n$ covariance matrix, which costs $O(n^3)$ operations and $O(n^2)$ memory. For a million data points—common in modern astronomy—this means:

$$\text{Operations} \sim 10^{18} \quad \text{Memory} \sim 10^{12} \text{ bytes} = 1 \text{ TB}$$

Even with sparse approximations, GPs struggle beyond $10^4$ points. Modern surveys like LSST will generate billions of observations. We need methods that scale linearly, not cubically.

### The Feature Engineering Problem

Consider classifying galaxy morphologies from images. With our current toolkit, we'd need to manually extract features: ellipticity, concentration index, color gradients, spiral arm pitch angles. But what if the discriminating features are subtle combinations we haven't thought of? What if different features matter for different galaxy types?

The Gaussian Process assumes we know which kernel to use—that we understand the similarity structure of our problem. But in high-dimensional spaces with complex patterns, designing kernels becomes impossible. We need methods that learn their own similarity measures.

### The Representation Learning Gap

Our N-body simulations generate positions and velocities for thousands of stars. Our MCRT code produces spectra with thousands of wavelength bins. These raw outputs aren't the natural variables for understanding the physics. The relaxation time depends on global properties like density profiles and velocity dispersions. Spectral classifications depend on line ratios and continuum shapes.

Currently, we manually compute these derived quantities—we impose our understanding of what matters. But what if there are better representations we haven't discovered? What if the optimal features for prediction aren't the ones that seem physically intuitive?

## 1.2 The Neural Network Promise

Neural networks address all these limitations simultaneously. They scale to billions of data points, using stochastic gradient descent to avoid matrix inversions. They automatically learn features from raw data, discovering representations we never programmed. They can approximate any continuous function to arbitrary accuracy, given sufficient capacity.

But this power comes with trade-offs. We lose the elegant uncertainty quantification of GPs. We can't easily interpret what the network has learned. Training requires careful tuning and can fail mysteriously. The same flexibility that enables learning can lead to overfitting or underfitting.

Understanding neural networks deeply—not just as black boxes but as mathematical objects with specific properties and limitations—is essential for using them effectively. Let's build this understanding from first principles.

---

# Part II: The Mathematical Foundation

## 2.1 From Linear Models to Nonlinearity

### Starting Simple: Linear Regression Revisited

We've seen linear regression in our MCMC project:

$$y = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^{d} w_i x_i + b$$

where $\mathbf{w} \in \mathbb{R}^d$ are weights, $\mathbf{x} \in \mathbb{R}^d$ are inputs, and $b \in \mathbb{R}$ is bias. This model assumes the output is a linear combination of inputs. We found optimal parameters by minimizing squared error:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{j=1}^{n} (y_j - \mathbf{w}^T\mathbf{x}_j - b)^2$$

The solution has a closed form: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ for appropriate data matrices. But linear models can only represent linear relationships. They can't learn that luminosity scales as mass to the 3.5 power, or that extinction depends exponentially on optical depth.

### The Nonlinear Transformation

The key insight of neural networks is to apply nonlinear transformations to linear combinations. Consider a single "neuron":

$$h = \phi(\mathbf{w}^T\mathbf{x} + b)$$

where $\phi: \mathbb{R} \to \mathbb{R}$ is an activation function. The neuron first computes a linear combination (like linear regression), then applies a nonlinear transformation. Common activation functions include:

**Sigmoid** (historically important, now rarely used):
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This "squashes" any input to the range (0, 1). It's differentiable everywhere with derivative:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

The sigmoid was inspired by biological neurons, which have binary (fire/don't fire) outputs. The smooth approximation enables gradient-based learning.

**Hyperbolic Tangent**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

This maps to (-1, 1) instead of (0, 1), often improving training dynamics by keeping activations centered around zero.

**Rectified Linear Unit (ReLU)** (modern standard):
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Despite the non-differentiability at zero (we set the gradient to 0 or 1 arbitrarily), ReLU works remarkably well. It avoids the "vanishing gradient" problem of sigmoids and is computationally efficient.

### Building Depth: The Multilayer Perceptron

A single nonlinear transformation provides limited expressiveness. The power comes from composing multiple layers:

$$\mathbf{h}^{(1)} = \phi(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \phi(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$y = \mathbf{W}^{(3)}\mathbf{h}^{(2)} + b^{(3)}$$

Here:
- $\mathbf{W}^{(1)} \in \mathbb{R}^{h_1 \times d}$ transforms input to first hidden layer
- $\mathbf{W}^{(2)} \in \mathbb{R}^{h_2 \times h_1}$ transforms first to second hidden layer  
- $\mathbf{W}^{(3)} \in \mathbb{R}^{1 \times h_2}$ transforms final hidden layer to output
- $\phi$ is applied element-wise to each component

This architecture is called a Multilayer Perceptron (MLP) or feedforward neural network. The term "perceptron" is historical—the original perceptron was a linear classifier. "Multilayer" indicates we've stacked multiple transformations. "Feedforward" means information flows in one direction, from input to output.

## 2.2 The Universal Approximation Theorem

### The Remarkable Result

The Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact set to arbitrary accuracy. Mathematically:

For any continuous function $f: K \subset \mathbb{R}^n \to \mathbb{R}$ where $K$ is compact, and any $\epsilon > 0$, there exists a single-hidden-layer network with activation $\phi$ such that:

$$\sup_{\mathbf{x} \in K} |f(\mathbf{x}) - \text{NN}(\mathbf{x})| < \epsilon$$

The proof relies on the Stone-Weierstrass theorem, showing that neural networks are dense in the space of continuous functions. The key requirements are:
- The activation function is continuous and non-polynomial
- We have enough hidden units
- We can set weights and biases appropriately

### Why Depth Matters Anyway

If a single hidden layer suffices for universal approximation, why use deep networks? The theorem says nothing about how many neurons we need. For some functions, the required width grows exponentially with desired accuracy. Deep networks can be exponentially more efficient than shallow ones.

Consider representing a piecewise linear function with $2^n$ pieces. A shallow network needs $O(2^n)$ neurons—one for each piece. A deep network with $n$ layers needs only $O(n)$ neurons—each layer divides the input space in half. This exponential advantage of depth over width is why modern networks have dozens or even hundreds of layers.

Moreover, depth enables hierarchical feature learning. Early layers learn simple features (edges in images), middle layers combine these into complex features (textures, shapes), and late layers represent high-level concepts (objects, categories). This hierarchical structure mirrors how we believe the brain processes information and how physical systems have multiple scales of organization.

---

# Part III: Learning Through Backpropagation

## 3.1 The Credit Assignment Problem

Given a network with millions of parameters and a loss function measuring prediction error, how do we determine which parameters to adjust and by how much? This is the credit assignment problem: which weights deserve "credit" (or blame) for the output?

The solution is backpropagation (Rumelhart et al., 1986), which efficiently computes gradients using the chain rule of calculus. Despite its simplicity, backpropagation is arguably the most important algorithm in machine learning.

### Forward Pass: Computing Predictions

Consider a simple two-layer network:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{h}^{(1)} = \phi(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}$$
$$\hat{y} = \mathbf{z}^{(2)}$$

We've separated the linear transformation ($\mathbf{z}$) from the activation ($\mathbf{h}$) for clarity. Given input $\mathbf{x}$, we compute layer by layer until reaching the output $\hat{y}$.

### The Loss Function

To train the network, we need a loss function measuring prediction quality. For regression with target $y$:

$$L = \frac{1}{2}(y - \hat{y})^2$$

The factor of 1/2 simplifies derivatives. For classification with $C$ classes, we use softmax activation and cross-entropy loss:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
$$L = -\sum_{i=1}^{C} y_i \log p_i$$

where $y_i = 1$ for the correct class and 0 otherwise.

### Backward Pass: Computing Gradients

We need gradients of the loss with respect to all parameters. Starting from the output and working backward:

**Output layer gradient**:
$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

**Pre-activation gradient** (using chain rule):
$$\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{z}^{(2)}} = \hat{y} - y$$

**Weight gradient** for layer 2:
$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot \frac{\partial \mathbf{z}^{(2)}}{\partial \mathbf{W}^{(2)}} = (\hat{y} - y) \cdot \mathbf{h}^{(1)T}$$

**Hidden layer gradient** (this is the key step):
$$\frac{\partial L}{\partial \mathbf{h}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot \frac{\partial \mathbf{z}^{(2)}}{\partial \mathbf{h}^{(1)}} = \mathbf{W}^{(2)T} \cdot \frac{\partial L}{\partial \mathbf{z}^{(2)}}$$

**Pre-activation gradient** for layer 1:
$$\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(1)}} \odot \phi'(\mathbf{z}^{(1)})$$

where $\odot$ denotes element-wise multiplication. For ReLU, $\phi'(z) = 1$ if $z > 0$, else 0.

**Weight gradient** for layer 1:
$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} \cdot \mathbf{x}^T$$

### The Backpropagation Algorithm

The pattern is clear: we propagate gradients backward through the network, using the chain rule at each layer. The algorithm:

1. **Forward pass**: Compute and store all activations
2. **Compute loss**: Evaluate $L$ at the output
3. **Backward pass**: Compute gradients layer by layer, moving backward
4. **Update parameters**: Adjust weights using gradients

The key insight is that we can reuse intermediate gradients. Computing $\partial L/\partial \mathbf{h}^{(i)}$ once lets us compute gradients for all parameters in layer $i$. This reduces complexity from exponential to linear in network depth.

## 3.2 Gradient Descent and Its Variants

### Vanilla Gradient Descent

With gradients in hand, we update parameters to reduce loss:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}$$

where $\eta$ is the learning rate. This moves parameters in the direction of steepest descent. For convex losses (like linear regression), this converges to the global minimum. For neural networks, the loss landscape is highly non-convex, with many local minima and saddle points.

### Stochastic Gradient Descent (SGD)

Computing gradients over all training data is expensive. SGD approximates the gradient using a random subset (minibatch):

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{\partial L_i}{\partial \mathbf{w}}$$

where $\mathcal{B}$ is a minibatch of typically 32-512 examples. The noise from sampling can actually help escape shallow local minima. The variance of the gradient estimate decreases as $1/|\mathcal{B}|$, so larger batches give more stable updates.

### Momentum: Learning from Physics

Recall our Hamiltonian Monte Carlo from the MCMC project. We can view gradient descent as simulating a particle rolling down the loss landscape. Adding momentum helps overcome local bumps:

$$\mathbf{v} \leftarrow \beta \mathbf{v} - \eta \frac{\partial L}{\partial \mathbf{w}}$$
$$\mathbf{w} \leftarrow \mathbf{w} + \mathbf{v}$$

where $\mathbf{v}$ is velocity and $\beta \approx 0.9$ is the momentum coefficient. This accelerates convergence in consistent directions while damping oscillations.

### Adaptive Learning Rates: Adam

Different parameters may need different learning rates. Adam (Kingma & Ba, 2015) adapts rates based on gradient history:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$ 
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Here $m_t$ tracks the mean gradient (momentum), $v_t$ tracks the second moment (adaptive learning rate), and the hat notation corrects for initialization bias. Adam often works well with default parameters ($\beta_1=0.9$, $\beta_2=0.999$, $\eta=0.001$).

---

# Part IV: The Deep Learning Revolution

## 4.1 Why Deep Networks Were Hard to Train

### The Vanishing Gradient Problem

Consider a deep network with sigmoid activations. The gradient for early layers involves products of many terms:

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(n)}} \prod_{i=2}^{n} \frac{\partial \mathbf{h}^{(i)}}{\partial \mathbf{h}^{(i-1)}}$$

For sigmoid, $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. After 10 layers, gradients are multiplied by at most $(0.25)^{10} \approx 10^{-6}$. Early layers receive virtually no gradient signal and stop learning.

### The Exploding Gradient Problem

With poor initialization, gradients can also explode. If weights are too large, activations saturate (reach extreme values), and if they're in the linear region, products of weights grow exponentially. This causes training instability and numerical overflow.

## 4.2 Modern Solutions

### Better Activation Functions

ReLU largely solved the vanishing gradient problem. For positive inputs, $\text{ReLU}'(z) = 1$, so gradients pass through unchanged. This enables training networks with hundreds of layers. Variants like Leaky ReLU address the "dying ReLU" problem where neurons get stuck at zero:

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}$$

with $\alpha \approx 0.01$.

### Careful Initialization

Xavier/Glorot initialization scales weights based on layer size:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

This keeps activation magnitudes roughly constant across layers. For ReLU, He initialization is preferred:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

The factor of 2 accounts for ReLU zeroing half the inputs on average.

### Normalization Techniques

Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations within each minibatch:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{z}_i + \beta$$

where $\mu_B$, $\sigma_B^2$ are batch statistics, and $\gamma$, $\beta$ are learned parameters. This stabilizes training by preventing internal covariate shift—the problem of activation distributions changing during training.

### Residual Connections

ResNets (He et al., 2016) add skip connections that bypass layers:

$$\mathbf{h}^{(l+2)} = \mathbf{h}^{(l)} + F(\mathbf{h}^{(l)})$$

where $F$ represents two layers of transformation. This creates "highways" for gradients to flow backward and enables training networks with thousands of layers. The identity path ensures gradients reach early layers even if $F$ produces small derivatives.

---

# Part V: Regularization and Generalization

## 5.1 The Overfitting Problem

Neural networks with millions of parameters can memorize training data perfectly. A network with more parameters than training examples can achieve zero training error by essentially storing a lookup table. But this memorization doesn't generalize to new data.

The fundamental challenge is the bias-variance tradeoff:
- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)

Simple models have high bias but low variance. Complex models have low bias but high variance. We need the sweet spot that minimizes total error.

## 5.2 Regularization Techniques

### L2 Regularization (Weight Decay)

Add a penalty for large weights to the loss:

$$L_{\text{total}} = L_{\text{data}} + \lambda \sum_{i} ||\mathbf{W}^{(i)}||_2^2$$

This encourages small weights, which create smoother functions. The gradient includes a decay term:

$$\frac{\partial L_{\text{total}}}{\partial \mathbf{W}} = \frac{\partial L_{\text{data}}}{\partial \mathbf{W}} + 2\lambda \mathbf{W}$$

During updates, weights shrink by factor $(1 - 2\eta\lambda)$ each step, hence "weight decay."

### Dropout: Learning Robust Features

Dropout (Srivastava et al., 2014) randomly zeros neurons during training:

$$h_i' = \begin{cases} h_i/p & \text{with probability } p \\ 0 & \text{with probability } 1-p \end{cases}$$

The scaling by $1/p$ maintains expected activation magnitude. At test time, we use all neurons without dropout.

Dropout forces the network to be robust—no single neuron can be relied upon. It's equivalent to training an ensemble of $2^n$ networks (all possible dropout masks) with shared weights. This reduces overfitting dramatically.

### Early Stopping: Implicit Regularization

Monitor validation loss during training. When it stops improving (while training loss continues decreasing), stop training. This prevents the network from memorizing noise in training data.

Early stopping is related to L2 regularization—both limit model complexity. For gradient descent with small learning rate, the number of iterations acts like an inverse regularization parameter.

### Data Augmentation: Expanding the Training Set

For image data, apply random transformations: rotations, shifts, scaling, flips. This creates new training examples that preserve labels. For astronomical images, we might add:
- Realistic noise models
- PSF convolution
- Atmospheric distortion
- Cosmic ray hits

Data augmentation encodes invariances we know should exist. If galaxy classification shouldn't depend on orientation, random rotations ensure the network learns this invariance.

---

# Part VI: Connections to Our Previous Work

## 6.1 Neural Networks as Gaussian Process Approximations

### The Infinite Width Limit

Recall from our GP lectures that as neural network width approaches infinity, the network converges to a Gaussian Process. Consider a single hidden layer network:

$$f(x) = \sum_{i=1}^{H} v_i \phi(w_i^T x + b_i)$$

If we initialize weights $v_i \sim \mathcal{N}(0, \sigma_v^2/H)$, $w_i \sim \mathcal{N}(0, \sigma_w^2 I)$, $b_i \sim \mathcal{N}(0, \sigma_b^2)$, then as $H \to \infty$, the Central Limit Theorem tells us:

$$f(x) \to \mathcal{GP}(0, k(x, x'))$$

where the kernel depends on the activation function. For ReLU:

$$k(x, x') = \frac{\sigma_v^2}{2\pi}||x|| \cdot ||x'|| \left(\sin\theta + (\pi - \theta)\cos\theta\right)$$

where $\theta = \cos^{-1}\left(\frac{x^T x'}{||x|| \cdot ||x'||}\right)$.

This connection reveals that:
- Neural networks are essentially learning finite-dimensional approximations to GPs
- The prior over functions is determined by initialization
- Architecture choices implicitly specify kernel properties

### Why Finite Networks Outperform Their Infinite Limit

If infinite networks are GPs, why use finite networks? The key is that finite networks can adapt their "kernel" through learning. While GPs have fixed kernels, neural networks learn feature representations that effectively change the similarity metric during training.

## 6.2 The Leapfrog Connection: Training Dynamics as Hamiltonian Flow

The training dynamics of neural networks can be viewed through the lens of Hamiltonian mechanics, connecting to our N-body and HMC projects. Define the Hamiltonian:

$$H(\mathbf{w}, \mathbf{v}) = L(\mathbf{w}) + \frac{1}{2}||\mathbf{v}||^2$$

where $L$ is the loss (potential energy) and $\mathbf{v}$ is momentum. Gradient descent with momentum follows approximate Hamiltonian dynamics:

$$\frac{d\mathbf{w}}{dt} = \mathbf{v}$$
$$\frac{d\mathbf{v}}{dt} = -\nabla L(\mathbf{w}) - \gamma \mathbf{v}$$

The damping term $\gamma \mathbf{v}$ violates energy conservation but ensures convergence. This is analogous to friction in physical systems, dissipating energy until the system settles into a minimum.

## 6.3 Automatic Differentiation: The Ultimate Chain Rule

### From Manual to Automatic Gradients

In our MCMC project, we computed likelihood gradients manually. For complex models, this becomes error-prone and tedious. Automatic differentiation (autodiff) computes exact derivatives of any composition of differentiable operations.

There are two modes:

**Forward Mode**: Computes directional derivatives $\nabla f \cdot v$ efficiently. Good when outputs >> inputs.

**Reverse Mode**: Computes gradients $\nabla f$ efficiently. Good when inputs >> outputs (typical for neural networks).

Backpropagation is reverse-mode autodiff applied to neural networks. Modern frameworks like JAX make this transparent—you write the forward pass, and gradients come for free.

### JAX: Functional Programming Meets Neural Networks

JAX combines NumPy's interface with autodiff, JIT compilation, and functional programming. Key concepts:

```python
import jax.numpy as jnp
from jax import grad, jit, vmap

def loss(params, x, y):
    prediction = neural_network(params, x)
    return jnp.mean((prediction - y)**2)

# Automatic gradient
loss_grad = grad(loss)

# Just-in-time compilation for speed
loss_grad_fast = jit(loss_grad)

# Vectorization over batch dimension
batch_loss = vmap(loss, in_axes=(None, 0, 0))
```

JAX treats functions as first-class objects that can be transformed. This functional approach connects to our GP work—both treat functions as fundamental entities rather than implementations.

---

# Part VII: Modern Architectures and Applications

## 7.1 Convolutional Neural Networks: Exploiting Spatial Structure

### The Problem with Fully Connected Networks

For a 256×256 astronomical image, a fully connected layer to 1000 hidden units requires 65 million parameters. This is computationally prohibitive and ignores the spatial structure of images. Nearby pixels are related; distant pixels are largely independent.

### The Convolution Operation

Convolutional layers apply the same small filter across the entire image:

$$h_{ij} = \phi\left(\sum_{m,n} w_{mn} \cdot x_{i+m,j+n} + b\right)$$

where $w$ is typically a 3×3 or 5×5 filter. This has several advantages:
- **Parameter sharing**: Same filter used everywhere
- **Translation equivariance**: Features detected regardless of position
- **Local connectivity**: Each output depends only on nearby inputs

### Building Hierarchical Representations

CNNs stack convolution and pooling layers:
1. **Convolution**: Detect features (edges, textures)
2. **Activation**: Add nonlinearity (ReLU)
3. **Pooling**: Reduce spatial resolution (max or average)
4. **Repeat**: Build hierarchical features

Early layers learn Gabor-like filters (edge detectors). Middle layers combine edges into shapes and textures. Deep layers represent complex patterns and objects.

For astronomical applications:
- Early layers might detect stars and artifacts
- Middle layers identify galaxies, clusters, diffraction spikes
- Late layers classify morphologies or detect rare objects

## 7.2 Recurrent Networks: Processing Sequences

### The Temporal Dimension

Our time-series data—variable star light curves, gravitational wave signals, pulsar timing—has temporal dependencies. Recurrent Neural Networks (RNNs) process sequences by maintaining hidden state:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

The hidden state $h_t$ acts as memory, accumulating information from previous timesteps. The same weights are used at each timestep (weight sharing across time).

### The Vanishing Gradient Problem (Again)

Backpropagation through time unrolls the RNN and applies standard backprop. But gradients must flow through many timesteps:

$$\frac{\partial L}{\partial h_0} = \frac{\partial L}{\partial h_T} \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

For long sequences, gradients vanish or explode. LSTMs (Long Short-Term Memory) solve this with gating mechanisms that create gradient highways, similar to ResNets in the spatial domain.

## 7.3 Transformers: Attention Is All You Need

### The Attention Mechanism

Transformers (Vaswani et al., 2017) replace recurrence with attention—directly modeling relationships between all positions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q$ (queries): What information do we want?
- $K$ (keys): What information is available?
- $V$ (values): The actual information content

This computes weighted averages where weights depend on query-key similarity. Every position can attend to every other position, capturing long-range dependencies.

### Why Transformers Matter for Science

Transformers excel at:
- Finding patterns in irregular time series (like our RR Lyrae light curves)
- Correlating multi-wavelength observations
- Processing tabular data with missing values
- Learning from heterogeneous data sources

The self-attention mechanism discovers which features are relevant for which predictions—automated feature engineering at its finest.

---

# Part VIII: Practical Considerations and Pitfalls

## 8.1 The Debugging Challenge

### Why Neural Networks Are Hard to Debug

Unlike our N-body simulations where energy conservation provides a clear diagnostic, neural networks can fail silently. A network might train without errors but learn nothing useful. Common failure modes:

**Gradient Problems**:
- Vanishing gradients: Network stops learning (check gradient magnitudes)
- Exploding gradients: NaN losses (check for numerical overflow)
- Dead neurons: ReLUs stuck at zero (monitor activation statistics)

**Optimization Issues**:
- Learning rate too high: Loss oscillates or diverges
- Learning rate too low: Extremely slow convergence
- Poor initialization: Network stuck in bad region

**Data Problems**:
- Label noise: Network memorizes errors
- Distribution shift: Training and test data differ
- Imbalanced classes: Network always predicts majority class

### Diagnostic Tools

Monitor these metrics during training:
- Loss curves (training and validation)
- Gradient norms by layer
- Activation distributions
- Weight distributions
- Individual neuron activations

Visualize learned features:
- First layer filters (for CNNs)
- Activation maximization (what inputs maximize specific neurons?)
- Saliency maps (what input regions affect predictions?)

## 8.2 The Interpretation Challenge

### Black Box vs Glass Box

Our GP models provided uncertainty estimates and used interpretable kernels. Neural networks are often black boxes—we know they work but not why. This is problematic for science where understanding matters as much as prediction.

Interpretation techniques include:

**Feature Attribution**:
- Gradient-based: How does each input affect the output?
- Perturbation-based: What happens when we modify inputs?
- Shapley values: Game-theoretic attribution of contributions

**Concept Extraction**:
- What patterns activate specific neurons?
- Can we find human-interpretable concepts in hidden layers?
- Do learned features correspond to known physical quantities?

### The Uncertainty Problem

Neural networks output point predictions without uncertainty. This is dangerous for scientific applications where knowing confidence is crucial. Solutions include:

**Bayesian Neural Networks**: Place distributions over weights instead of point estimates. Computationally expensive but principled.

**Dropout at Test Time**: Monte Carlo dropout approximates Bayesian uncertainty by running multiple forward passes with different dropout masks.

**Ensemble Methods**: Train multiple networks with different initializations. Variance across predictions indicates uncertainty.

**Hybrid Approaches**: Use neural networks to learn features, then apply GPs for uncertainty-aware predictions.

---

# Part IX: Implementing Neural Networks for Astrophysics

## 9.1 From N-body to Neural Networks

Let's design a neural network that learns from our N-body simulations:

**Problem**: Predict relaxation time from initial conditions
**Input**: $\mathbf{x} = [N, M, c, Q, S]$ (5 dimensions)
**Output**: $T_{\text{relax}}$ (scalar)

**Architecture Design**:
```
Input Layer: 5 neurons (one per parameter)
Hidden Layer 1: 64 neurons, ReLU activation
Hidden Layer 2: 32 neurons, ReLU activation  
Hidden Layer 3: 16 neurons, ReLU activation
Output Layer: 1 neuron, linear activation
```

The decreasing width creates a bottleneck that forces the network to learn compressed representations. This is similar to our GP learning relevant length scales, but more flexible.

**Training Considerations**:
- Standardize inputs to zero mean, unit variance
- Log-transform the output (relaxation time spans orders of magnitude)
- Use MSE loss in log space: $L = (\\log T_{\text{pred}} - \\log T_{\text{true}})^2$
- Apply dropout (p=0.2) to hidden layers
- Use Adam optimizer with learning rate 0.001

## 9.2 From MCRT to CNNs

**Problem**: Classify galaxy morphologies from dust-affected images
**Input**: 128×128 pixel images, multiple bands
**Output**: Probability distribution over morphological types

**Architecture**:
```
Conv Layer 1: 32 filters, 3×3, ReLU
MaxPool: 2×2
Conv Layer 2: 64 filters, 3×3, ReLU
MaxPool: 2×2
Conv Layer 3: 128 filters, 3×3, ReLU
GlobalAveragePool
Dense Layer: 256 neurons, ReLU, Dropout(0.5)
Output Layer: C neurons (classes), Softmax
```

The convolutions learn hierarchical features invariant to translation. Pooling provides limited rotation invariance. Global average pooling reduces spatial dimensions while maintaining feature maps.

**Data Augmentation for Astronomy**:
- Random rotations (galaxies have no preferred orientation)
- Random flips (mirror symmetry)
- Realistic noise injection (Poisson + Gaussian)
- PSF convolution variations
- Small shifts (centering uncertainty)

## 9.3 From Time Series to RNNs

**Problem**: Classify variable stars from irregular light curves
**Input**: Time series $(t_i, m_i, \sigma_i)$ with varying length
**Output**: Star type (RR Lyrae, Cepheid, eclipsing binary, etc.)

**Architecture Using LSTM**:
```
Embedding: Project (t, m, σ) to higher dimension
LSTM Layer 1: 128 units
LSTM Layer 2: 64 units
Global Attention: Weight timesteps by importance
Dense Layer: 32 neurons, ReLU
Output Layer: Softmax over classes
```

The LSTM handles irregular sampling naturally. Attention identifies informative observations (e.g., moments of eclipse or peak brightness).

---

# Part X: The Future - Neural Networks and Scientific Discovery

## 10.1 Beyond Supervised Learning

### Unsupervised Learning: Finding Structure

Real astronomical surveys produce far more unlabeled than labeled data. Unsupervised methods discover structure without labels:

**Autoencoders**: Learn compressed representations by reconstructing inputs
$$\mathbf{x} \to \text{Encoder} \to \mathbf{z} \to \text{Decoder} \to \hat{\mathbf{x}}$$

The bottleneck $\mathbf{z}$ captures essential features. For spectra, this might discover principal components. For images, it might separate morphology from viewing angle.

**Variational Autoencoders (VAEs)**: Add probabilistic structure
$$q(z|x) = \mathcal{N}(\mu(x), \sigma(x))$$

The latent space becomes interpretable—nearby points are similar, interpolation is meaningful. VAEs can generate new examples by sampling from the latent distribution.

### Self-Supervised Learning: Creating Labels from Data

Transform unlabeled data into supervised problems:
- Predict masked portions of light curves
- Forecast future observations from past
- Identify which augmentations were applied
- Contrastive learning: distinguish similar from dissimilar pairs

These tasks force the network to learn meaningful representations without human labels.

## 10.2 Neural Networks as Scientific Instruments

### Simulation-Based Inference

Combine neural networks with our simulation expertise:

1. Run simulations across parameter space
2. Train network to predict parameters from observables
3. Apply to real observations
4. Uncertainty from ensemble or Bayesian methods

This is our GP emulation strategy but scales to millions of simulations and complex observables.

### Differentiable Simulations

Implement physical simulations in autodiff frameworks:

```python
def n_body_step(state, params):
    # Implement Leapfrog in JAX
    positions, velocities = state
    forces = compute_forces(positions, params)
    velocities_half = velocities + 0.5 * dt * forces
    positions_new = positions + dt * velocities_half
    forces_new = compute_forces(positions_new, params)
    velocities_new = velocities_half + 0.5 * dt * forces_new
    return (positions_new, velocities_new)

# Gradient of simulation with respect to parameters!
simulation_grad = grad(n_body_simulation, argnums=1)
```

Now we can optimize initial conditions to match observations—inverse problems through gradient descent!

### Learning Physical Laws

Neural networks can discover governing equations from data:

**Neural ODEs**: Learn differential equations
$$\frac{d\mathbf{x}}{dt} = f_{\theta}(\mathbf{x}, t)$$

where $f_{\theta}$ is a neural network. This could discover modifications to gravity from orbital data.

**Symbolic Regression**: Combine neural networks with genetic programming to find analytical expressions. The network guides the search through equation space.

**Physics-Informed Neural Networks (PINNs)**: Include physical constraints in the loss:
$$L = L_{\text{data}} + \lambda L_{\text{physics}}$$

where $L_{\text{physics}}$ penalizes violations of known laws (conservation, symmetries, PDEs).

## 10.3 The Synthesis: Hybrid Models

The future isn't neural networks or physical models—it's both. Consider this architecture for analyzing galaxy clusters:

1. **CNN** extracts features from multi-band images
2. **N-body simulation** provides dynamical prior
3. **MCRT** models dust extinction
4. **Gaussian Process** interpolates sparse spectroscopy
5. **Neural ODE** evolves the system forward
6. **Bayesian layer** quantifies uncertainty

Each component contributes its strength:
- Neural networks: Feature learning, pattern recognition
- Physical models: Interpretability, constraints
- Gaussian Processes: Uncertainty quantification
- Bayesian methods: Principled inference

---

# Conclusion: The Complete Computational Astronomer

## The Journey's End and Beginning

You began this course representing stars as simple objects with mass and luminosity. You've now built systems that can learn representations we never explicitly programmed, discover patterns in data we didn't know existed, and solve inverse problems that seemed computationally intractable.

The progression wasn't arbitrary. Each project addressed limitations of the previous ones:
- Object-oriented programming organized our physical understanding
- N-body dynamics revealed that algorithms affect outcomes profoundly
- Monte Carlo methods turned intractable integrals into sampling problems
- MCMC quantified uncertainty in all measurements
- Gaussian Processes extended uncertainty to entire functions
- Neural networks learned representations automatically

## The Unified Framework

Despite their apparent differences, these methods share deep connections:

**Optimization** appears everywhere:
- Minimizing energy drift (symplectic integration)
- Maximizing likelihood (parameter inference)
- Minimizing prediction error (neural network training)
- Maximizing marginal likelihood (GP hyperparameters)

**The Chain Rule** underlies:
- Force calculations in N-body systems
- Jacobians in variable transformations
- Backpropagation in neural networks
- Automatic differentiation in JAX

**Probabilistic Thinking** evolved from:
- Sampling distributions (Monte Carlo)
- To quantifying uncertainty (Bayesian inference)
- To distributions over functions (Gaussian Processes)
- To learned representations with uncertainty (Bayesian neural networks)

## The Tools You've Mastered

You now possess a complete toolkit for computational discovery:

**For Forward Modeling**:
- Simulate physical systems (N-body dynamics)
- Include realistic observations (MCRT)
- Propagate uncertainties (Monte Carlo)

**For Inverse Problems**:
- Infer parameters (MCMC)
- Learn functions (Gaussian Processes)
- Discover representations (Neural Networks)

**For Scaling**:
- Vectorization (NumPy)
- Automatic differentiation (JAX)
- Stochastic optimization (SGD)
- Distributed computing (batch parallelism)

## The Scientific Impact

These aren't just computational techniques—they're instruments for discovery. With neural networks, you can:

- Find rare objects in petabyte surveys
- Classify millions of galaxies in minutes
- Predict expensive simulations instantly
- Discover unknown correlations in high-dimensional data
- Solve inverse problems that were computationally forbidden

But remember: neural networks are powerful tools, not magic. They require careful design, thorough validation, and healthy skepticism. Always ask:
- What has the network actually learned?
- How does it fail?
- What are the uncertainties?
- Does it respect known physics?

## The Path Forward

This foundation prepares you for whatever comes next:

**If you pursue astronomy**: You can tackle problems at the intersection of theory, simulation, and observation, using modern methods that most astronomers don't yet understand.

**If you enter industry**: You have both the theoretical understanding and practical skills that companies desperately need—people who understand why methods work, not just how to apply them.

**If you continue research**: You're prepared for the convergence of physics and machine learning, where differentiable programming and learned representations transform how we understand nature.

## The Final Lesson

The most important lesson isn't any specific technique—it's the recognition that computational science is about transforming problems. Every method we've studied transforms something intractable into something solvable:

- Continuous differential equations → Discrete timesteps
- Infinite-dimensional integrals → Monte Carlo samples
- Inverse problems → Forward sampling
- Function learning → Parameter optimization
- Feature engineering → Representation learning

You've learned not just to solve problems but to recognize which transformations make problems solvable. This meta-skill—knowing which tool fits which problem and why—is what distinguishes a computational scientist from a programmer.

As you apply these methods to your own research, remember that the universe is not only comprehensible but computable. The same mathematics that keeps planets in stable orbits helps neural networks learn. The same algorithm that samples from probability distributions transports photons through dust. The patterns are everywhere, waiting to be discovered.

Go forth and compute the universe. But more importantly, understand it.