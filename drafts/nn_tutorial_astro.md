# Neural Networks for Astronomers: From Theory to Star Clusters

## Prerequisites Check

Before we begin, let's make sure you're comfortable with the mathematical tools we'll use. Don't worry if you're rusty - we'll explain everything as we go!

### Mathematical Prerequisites
You should be familiar with:
- **Matrix multiplication**: If $A$ is $m \times n$ and $B$ is $n \times p$, then $AB$ is $m \times p$
- **Partial derivatives**: $\frac{\partial f}{\partial x}$ means "how $f$ changes when only $x$ varies"  
- **The chain rule**: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$
- **Vector notation**: Bold symbols like $\mathbf{x}$ represent vectors; regular symbols like $x$ represent scalars

### Notation Convention
Throughout this document:
- Vectors are **column vectors** by default: $\mathbf{x} \in \mathbb{R}^{n \times 1}$
- $\mathbf{x}^T$ denotes the transpose (converting column to row vector)
- $\mathbf{w}^T \mathbf{x}$ is the dot product (scalar result)
- $\mathbf{x} \mathbf{y}^T$ is the outer product (matrix result)
- $\odot$ denotes element-wise (Hadamard) multiplication
- Subscripts index elements: $x_i$ is the $i$-th element of vector $\mathbf{x}$
- Superscripts in parentheses denote layers: $W^{(2)}$ is the weight matrix for layer 2

---

## Introduction: Why Neural Networks for N-Body Dynamics?

Imagine you're studying a globular cluster containing a million stars. You've spent weeks of supercomputer time running N-body simulations to understand how this cluster evolves over billions of years. Now you need to explore thousands of different initial conditions to understand how evolution depends on initial concentration, mass distribution, and environment. Running thousands more simulations would take years.

This is where neural networks become transformative. At their core, neural networks are mathematical structures that learn patterns from data. Once trained on a subset of your expensive simulations, a neural network can predict outcomes for new scenarios in milliseconds rather than weeks.

The mathematical foundation that makes this possible is the **Universal Approximation Theorem**: neural networks can approximate any continuous function to arbitrary accuracy. Since stellar dynamics follows deterministic physical laws (a complex but continuous mapping from initial to final states), neural networks can learn these mappings.

Let's build up the mathematics from first principles, always connecting back to astronomical applications.

---

## Part 1: The Single Neuron - Understanding the Basic Building Block

### The Mathematical Model

A single artificial neuron performs three operations:
1. Takes inputs and computes a weighted sum
2. Adds a bias term
3. Applies a non-linear activation function

Mathematically, for a neuron with $n$ inputs:

**Step 1 - Weighted Sum:**
$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

where:
- $\mathbf{x} = [x_1, x_2, ..., x_n]^T \in \mathbb{R}^{n \times 1}$ is the input vector (column vector)
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T \in \mathbb{R}^{n \times 1}$ is the weight vector (column vector)  
- $b \in \mathbb{R}$ is the bias (scalar)
- $z \in \mathbb{R}$ is the pre-activation or net input (scalar)

**Step 2 - Activation:**
$$y = \sigma(z)$$

where $\sigma$ is the activation function and $y$ is the neuron's output.

### 🌟 Astronomical Example: Core Collapse Predictor

Let's make this concrete. Suppose our neuron predicts whether a star cluster will undergo core collapse:

- $x_1$: Total mass (in units of $10^5 M_\odot$)
- $x_2$: Half-mass radius (in parsecs)  
- $x_3$: Central velocity dispersion (in km/s)
- $x_4$: Galactocentric distance (in kpc)

If we have weights $\mathbf{w} = [2.0, -3.0, 1.5, -0.5]^T$ and bias $b = 1.0$, then for a cluster with $\mathbf{x} = [3.0, 2.0, 10.0, 8.0]^T$:

$$z = 2.0(3.0) + (-3.0)(2.0) + 1.5(10.0) + (-0.5)(8.0) + 1.0 = 12.0$$

The large positive value suggests core collapse is likely (we'll see why after discussing activation functions).

📝 **Checkpoint 1**: Calculate $z$ for a cluster with $\mathbf{x} = [1.0, 5.0, 5.0, 20.0]^T$ using the same weights.  
*Answer: $z = 2.0(1.0) - 3.0(5.0) + 1.5(5.0) - 0.5(20.0) + 1.0 = -14.5$*

### Activation Functions: Adding Non-linearity

Without activation functions, even deep networks can only compute linear transformations. Let me prove this, then introduce the key activation functions.

#### Why We Need Non-linearity (Mathematical Proof)

Consider two layers without activation:
- Layer 1: $\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$
- Layer 2: $\mathbf{z}^{(2)} = W^{(2)}\mathbf{z}^{(1)} + \mathbf{b}^{(2)}$

Substituting:
$$\mathbf{z}^{(2)} = W^{(2)}(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = (W^{(2)}W^{(1)})\mathbf{x} + (W^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)})$$

This is still linear in $\mathbf{x}$! But gravitational dynamics involves $1/r$ potentials, $r^{3/2}$ orbital periods, and exponential instabilities - all non-linear.

#### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Range: $(0, 1)$ - perfect for probabilities
- Smooth and differentiable everywhere
- Saturates for large $|z|$ (outputs near 0 or 1)

**Derivative** (we'll need this for learning):
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

*Derivation:*
$$\sigma'(z) = \frac{d}{dz}\left(\frac{1}{1 + e^{-z}}\right) = \frac{0 - (-e^{-z})}{(1 + e^{-z})^2} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Note that:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$
- $1 - \sigma(z) = \frac{e^{-z}}{1 + e^{-z}}$
- Therefore: $\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$

📝 **Checkpoint 2**: Calculate $\sigma(2.0)$ and $\sigma'(2.0)$.  
*Answer: $\sigma(2.0) = 1/(1+e^{-2}) \approx 0.881$, $\sigma'(2.0) = 0.881(0.119) \approx 0.105$*

#### The ReLU Function

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Properties:**
- Computationally efficient (just a threshold)
- No saturation for positive inputs
- Sparsity (some neurons output exactly 0)

**Derivative:**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

Note: At $z = 0$, the derivative is undefined mathematically, but in practice we set it to 0.

#### Hyperbolic Tangent

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Properties:**
- Range: $(-1, 1)$ - zero-centered unlike sigmoid
- Relationship to sigmoid: $\tanh(z) = 2\sigma(2z) - 1$

**Derivative:**
$$\tanh'(z) = 1 - \tanh^2(z)$$

### 🎨 Visual Description: What These Functions Look Like

*[In a real document, we'd have plots here. Let me describe them:]*

- **Sigmoid**: S-shaped curve, gradually rising from 0 to 1, centered at (0, 0.5)
- **ReLU**: Hockey stick shape - flat at 0 for negative inputs, then linear with slope 1
- **Tanh**: S-shaped like sigmoid but centered at origin, ranging from -1 to 1

---

## Part 2: The Feedforward Network - Building Deep Understanding

### Architecture and Notation

A feedforward network consists of layers of neurons. Let's establish precise notation:

- $L$: Total number of layers (including input and output)
- $\ell \in \{0, 1, ..., L\}$: Layer index (0 is input, $L$ is output)
- $n_\ell$: Number of neurons in layer $\ell$
- $\mathbf{a}^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}$: Activation vector of layer $\ell$ (column vector)
- $\mathbf{z}^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}$: Pre-activation vector of layer $\ell$ (column vector)
- $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$: Weight matrix from layer $\ell-1$ to layer $\ell$
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}$: Bias vector for layer $\ell$ (column vector)

⚠️ **Important**: $W^{(\ell)}_{ij}$ is the weight from neuron $j$ in layer $\ell-1$ to neuron $i$ in layer $\ell$.

### Forward Propagation

Information flows forward through the network:

**Input Layer:**
$$\mathbf{a}^{(0)} = \mathbf{x}$$

**For each layer** $\ell = 1, 2, ..., L$:
$$\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{a}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$$

where $\sigma^{(\ell)}$ is applied element-wise: $[\mathbf{a}^{(\ell)}]_i = \sigma([\mathbf{z}^{(\ell)}]_i)$

### 🌟 Complete Example: Predicting Cluster Evolution

Let's design a small network to predict a cluster's half-mass radius after 1 Gyr.

**Architecture:**
- Input layer ($\ell=0$): 3 neurons
  - $x_1$: Initial mass ($10^5 M_\odot$)
  - $x_2$: Initial radius (parsecs)
  - $x_3$: Velocity dispersion (km/s)
- Hidden layer ($\ell=1$): 2 neurons with ReLU
- Output layer ($\ell=2$): 1 neuron (final radius)

**Weight Matrices and Biases:**
$$W^{(1)} = \begin{bmatrix} 0.5 & -0.3 & 0.2 \\ 0.1 & 0.4 & -0.6 \end{bmatrix}, \quad \mathbf{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$

$$W^{(2)} = \begin{bmatrix} 0.8 & -0.5 \end{bmatrix}, \quad b^{(2)} = 0.3$$

**Forward Pass for** $\mathbf{x} = [2.0, 3.0, 10.0]^T$:

*Layer 1:*
$$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{bmatrix} 0.5 & -0.3 & 0.2 \\ 0.1 & 0.4 & -0.6 \end{bmatrix} \begin{bmatrix} 2.0 \\ 3.0 \\ 10.0 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$

Computing each element:
- $z_1^{(1)} = 0.5(2.0) + (-0.3)(3.0) + 0.2(10.0) + 0.1 = 2.2$
- $z_2^{(1)} = 0.1(2.0) + 0.4(3.0) + (-0.6)(10.0) + (-0.2) = -4.8$

Apply ReLU:
$$\mathbf{a}^{(1)} = \begin{bmatrix} \max(0, 2.2) \\ \max(0, -4.8) \end{bmatrix} = \begin{bmatrix} 2.2 \\ 0 \end{bmatrix}$$

*Layer 2:*
$$z^{(2)} = [0.8, -0.5] \begin{bmatrix} 2.2 \\ 0 \end{bmatrix} + 0.3 = 1.76 + 0 + 0.3 = 2.06$$

Output: $\hat{y} = 2.06$ parsecs

📝 **Checkpoint 3**: What would $\mathbf{a}^{(1)}$ be if we used sigmoid instead of ReLU?  
*Answer: $\mathbf{a}^{(1)} = [\sigma(2.2), \sigma(-4.8)]^T \approx [0.900, 0.008]^T$*

### 🎨 Network Diagram Description

*[This would be a diagram showing:]*
- Three input nodes connected to two hidden nodes
- Each connection labeled with its weight
- Two hidden nodes connected to one output node
- Activation functions shown at each layer

---

## Part 3: Learning Through Backpropagation

### The Loss Function

To train our network, we need to quantify prediction error. For regression problems (predicting continuous values):

**Mean Squared Error (MSE):**
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}||\hat{\mathbf{y}}_i - \mathbf{y}_i||^2$$

where:
- $N$: Number of training examples
- $\mathbf{y}_i$: True value for example $i$
- $\hat{\mathbf{y}}_i$: Predicted value for example $i$
- The factor $\frac{1}{2}$ simplifies derivatives

For a single example with scalar output:
$$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$$

### Gradient Descent

We minimize the loss by adjusting parameters in the direction of steepest descent:

$$\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$$

where $\theta$ represents any parameter (weight or bias) and $\eta$ is the learning rate.

### The Backpropagation Algorithm

Define the error signal for layer $\ell$:
$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}} \in \mathbb{R}^{n_\ell \times 1}$$

This tells us how sensitive the loss is to each pre-activation.

#### Key Equations

**Output Layer Error** (for MSE loss, no activation):
$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

**Output Layer Error** (with activation):
$$\boldsymbol{\delta}^{(L)} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(\mathbf{z}^{(L)})$$

**Hidden Layer Error:**
$$\boldsymbol{\delta}^{(\ell)} = (W^{(\ell+1)})^T \boldsymbol{\delta}^{(\ell+1)} \odot \sigma'(\mathbf{z}^{(\ell)})$$

**Weight Gradients:**
$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} (\mathbf{a}^{(\ell-1)})^T$$

Note: This is an outer product resulting in a matrix of size $n_\ell \times n_{\ell-1}$

**Bias Gradients:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}$$

### Complete Worked Example

Let's continue our 3-2-1 network example. Given:
- Network output: $\hat{y} = 2.06$
- True value: $y = 3.0$
- Loss: $\mathcal{L} = \frac{1}{2}(2.06 - 3.0)^2 = 0.442$

**Backward Pass:**

*Output layer error:*
$$\delta^{(2)} = \hat{y} - y = 2.06 - 3.0 = -0.94$$

*Hidden layer error:*
First, compute ReLU derivatives:
$$\sigma'(\mathbf{z}^{(1)}) = \begin{bmatrix} \mathbb{1}_{z_1^{(1)} > 0} \\ \mathbb{1}_{z_2^{(1)} > 0} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

Then:
$$\boldsymbol{\delta}^{(1)} = (W^{(2)})^T \delta^{(2)} \odot \sigma'(\mathbf{z}^{(1)}) = \begin{bmatrix} 0.8 \\ -0.5 \end{bmatrix} (-0.94) \odot \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

$$= \begin{bmatrix} -0.752 \\ 0.470 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -0.752 \\ 0 \end{bmatrix}$$

*Weight gradients:*

For $W^{(2)}$:
$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)} (\mathbf{a}^{(1)})^T = (-0.94) [2.2, 0] = [-2.068, 0]$$

For $W^{(1)}$:
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} (\mathbf{a}^{(0)})^T = \begin{bmatrix} -0.752 \\ 0 \end{bmatrix} [2.0, 3.0, 10.0]$$

$$= \begin{bmatrix} -1.504 & -2.256 & -7.520 \\ 0 & 0 & 0 \end{bmatrix}$$

Note: The second row is all zeros because the second hidden neuron was "dead" (ReLU output 0).

📝 **Checkpoint 4**: If the learning rate $\eta = 0.1$, what is the new value of $W^{(2)}_{11}$ after one update?  
*Answer: $W^{(2)}_{11, new} = 0.8 - 0.1(-2.068) = 1.007$*

⚠️ **Advanced Note**: The "dead ReLU" problem (neurons that output 0 and stop learning) is why we sometimes use variants like Leaky ReLU: $\max(0.01z, z)$

---

## Part 4: Universal Approximation - The Power of Neural Networks

### Intuitive Understanding

The Universal Approximation Theorem tells us something remarkable: a neural network with just one hidden layer can approximate ANY continuous function to arbitrary accuracy, given enough neurons.

Think of it like this: each neuron in the hidden layer creates a "bump" or "step" in the function landscape. With enough bumps, you can sculpt any shape - just like how any continuous sound wave can be built from enough sine waves (Fourier series).

### 🌟 Astronomical Analogy

Consider approximating the gravitational potential of an irregular galaxy. We could:
1. Use millions of point masses (exact but expensive)
2. Fit with basis functions (multipole expansion)
3. Train a neural network to learn the potential

The neural network approach learns the optimal "basis functions" (neurons) from data!

### Simplified Mathematical Statement

For any continuous function $f$ and desired accuracy $\epsilon > 0$, there exists a neural network with one hidden layer such that:

$$\left| \sum_{i=1}^{N} v_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i) - f(\mathbf{x}) \right| < \epsilon$$

for all inputs $\mathbf{x}$ in the domain.

### The Practical Reality

While theoretically powerful, the theorem has limitations:
- Doesn't tell us how many neurons $N$ we need (could be huge!)
- Doesn't tell us how to find the right weights
- One hidden layer might need exponentially many neurons

This is why we use **deep** networks - they can represent complex functions more efficiently than shallow ones.

📝 **Checkpoint 5**: A network needs $2^{10} = 1024$ neurons to approximate a function with one hidden layer. If going deep reduces this by a factor of 2 per layer, how many neurons would a 4-layer network need?  
*Answer: Layer 1: 512, Layer 2: 256, Layer 3: 128, Layer 4: 64. Total: 960 neurons (less than the shallow network!)*

⚠️ **Advanced Box: Formal Statement**
For the mathematically inclined, the complete formal statement: Let $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ be a non-constant, bounded, continuous function. For any $f \in C([0,1]^m)$ and $\epsilon > 0$, there exist $N \in \mathbb{N}$, $v_i, b_i \in \mathbb{R}$, $\mathbf{w}_i \in \mathbb{R}^m$ such that the function $F(\mathbf{x}) = \sum_{i=1}^N v_i\sigma(\mathbf{w}_i^T\mathbf{x} + b_i)$ satisfies $\sup_{\mathbf{x} \in [0,1]^m}|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon$.

---

## Part 5: Optimization Algorithms

### Stochastic Gradient Descent (SGD)

Instead of computing gradients on all data (expensive!), we use random mini-batches:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{1}{|B|} \sum_{i \in B} \frac{\partial \mathcal{L}_i}{\partial \mathbf{w}}$$

where $B$ is a mini-batch (typically 32-512 examples).

### Momentum

Add physics-inspired momentum to accelerate convergence:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla_\mathbf{w} \mathcal{L}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

Typical value: $\beta = 0.9$

### Adam Optimizer

Adapts learning rate per parameter based on gradient history:

$$\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1) \nabla_\mathbf{w} \mathcal{L}$$
$$\mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1-\beta_2) (\nabla_\mathbf{w} \mathcal{L})^2$$

Bias correction:
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

Update:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## Part 6: Regularization Techniques

### L2 Regularization (Weight Decay)

Add penalty for large weights to the loss:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum_{\ell=1}^{L} ||W^{(\ell)}||_F^2$$

where $||W||_F^2 = \sum_{i,j} W_{ij}^2$ is the Frobenius norm squared.

Effect on gradient:
$$\frac{\partial \mathcal{L}_{total}}{\partial W_{ij}} = \frac{\partial \mathcal{L}_{data}}{\partial W_{ij}} + 2\lambda W_{ij}$$

### Dropout

During training, randomly set neurons to zero with probability $p$:

For each neuron in layer $\ell$:
$$a_j^{(\ell)} = \begin{cases} 
\frac{a_j^{(\ell)}}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}$$

At test time: use all neurons without dropout.

### Batch Normalization

Normalize inputs to each layer:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$\tilde{z}_i = \gamma \hat{z}_i + \beta$$

where $\mu_B$, $\sigma_B^2$ are batch statistics, and $\gamma$, $\beta$ are learned parameters.

---

## Part 7: Applications to Star Cluster Dynamics

### Example 1: Learning Gravitational Potentials

**Goal**: Learn smooth potential $\Phi(\mathbf{r})$ from particle distribution.

**Network Design**:
- Input: 3D position $(x, y, z)$
- Hidden: 4 layers × 128 neurons, tanh activation
- Output: Scalar potential $\Phi$

**Physics-Informed Loss**:
$$\mathcal{L} = \alpha||\Phi_{NN} - \Phi_{true}||^2 + \beta||\nabla\Phi_{NN} + \mathbf{F}_{true}||^2 + \gamma||\nabla^2\Phi_{NN} + 4\pi G\rho||^2$$

Terms enforce:
1. Potential accuracy
2. Force consistency ($\mathbf{F} = -\nabla\Phi$)
3. Poisson equation

### Example 2: Predicting Cluster Evolution

**Input Features** (all normalized):
- Global: mass, radius, concentration
- Kinematic: velocity dispersion, rotation
- Environmental: galactocentric distance

**Architecture**:
- Input → 100 neurons → 50 neurons → 50 neurons → Output
- ReLU activations, dropout rate 0.2

**Training Data**: 10,000 N-body simulations

### Example 3: Chaos Detection

**Goal**: Classify orbital stability in triple star systems.

**Features** (rotation-invariant):
- Period ratios: $P_2/P_1$
- Eccentricities: $e_1, e_2$
- Mass ratios: $m_1/m_3, m_2/m_3$
- Mutual inclination: $i_{mutual}$

**Output**: Probability of chaotic evolution

---

## Part 8: Practical Implementation Guide

### Weight Initialization

**Xavier/Glorot** (for sigmoid/tanh):
$$W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

**He Initialization** (for ReLU):
$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

Note: We sample from a normal distribution with this variance.

### Data Preprocessing

For astronomical data spanning orders of magnitude:

$$x_{normalized} = \frac{\log_{10}(x) - \mu_{\log}}{\sigma_{\log}}$$

Example: Cluster masses from $10^2$ to $10^6 M_\odot$ become roughly $[-2, 2]$ after normalization.

### Training Tips

1. **Start simple**: Linear → Shallow → Deep
2. **Monitor gradients**: Check for vanishing/exploding
3. **Use validation set**: Stop when validation loss increases
4. **Ensemble models**: Average multiple networks for robustness

### Common Debugging Steps

If your network isn't learning:

✓ Can it overfit 10 examples? (If not, there's a bug)  
✓ Are gradients flowing? (Print gradient norms by layer)  
✓ Are inputs normalized? (Check mean ≈ 0, std ≈ 1)  
✓ Is learning rate reasonable? (Try $10^{-4}$ to $10^{-2}$)  
✓ Are there dead neurons? (Check activation statistics)

---

## Quick Reference Summary

### Forward Pass
```
For each layer ℓ = 1 to L:
  z^(ℓ) = W^(ℓ) @ a^(ℓ-1) + b^(ℓ)
  a^(ℓ) = σ(z^(ℓ))
```

### Backward Pass
```
δ^(L) = (ŷ - y) ⊙ σ'(z^(L))
For each layer ℓ = L-1 to 1:
  δ^(ℓ) = W^(ℓ+1)ᵀ @ δ^(ℓ+1) ⊙ σ'(z^(ℓ))
  ∂L/∂W^(ℓ) = δ^(ℓ) @ a^(ℓ-1)ᵀ
  ∂L/∂b^(ℓ) = δ^(ℓ)
```

### Key Equations

| Function | Formula | Derivative |
|----------|---------|------------|
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ |
| ReLU | $\text{ReLU}(z) = \max(0,z)$ | $\text{ReLU}'(z) = \mathbb{1}_{z>0}$ |
| Tanh | $\tanh(z) = (e^z-e^{-z})/(e^z+e^{-z})$ | $\tanh'(z) = 1-\tanh^2(z)$ |
| MSE Loss | $\mathcal{L} = \frac{1}{2N}\sum(\hat{y}_i-y_i)^2$ | $\partial\mathcal{L}/\partial\hat{y}_i = (\hat{y}_i-y_i)/N$ |

### Parameter Updates

| Algorithm | Update Rule |
|-----------|-------------|
| SGD | $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla_\mathbf{w}\mathcal{L}$ |
| Momentum | $\mathbf{v} \leftarrow \beta\mathbf{v} + \eta\nabla_\mathbf{w}\mathcal{L}$; $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{v}$ |
| Adam | See Part 5 for full equations |

---

## Conclusion: Your Neural Network Journey

You now have the mathematical foundation to:
- Understand how neural networks transform inputs to outputs
- Implement backpropagation from scratch
- Apply neural networks to astronomical dynamics problems
- Debug and optimize your networks

The journey from single neurons to networks capable of learning stellar dynamics represents a fundamental shift in computational astrophysics. With these tools and your upcoming JAX implementation, you're ready to accelerate N-body simulations, discover hidden dynamical relationships, and push the boundaries of what's computationally possible in astronomy.

Remember: Neural networks are powerful but not magical. They excel when you:
- Have sufficient quality training data
- Encode appropriate physical constraints
- Choose architectures suited to your problem
- Carefully validate on unseen data

Now go forth and revolutionize computational stellar dynamics!

---

## Final Checkpoints

📝 **Final Challenge 1**: Design a network architecture for predicting whether a planetary system with 3 planets will be stable. What inputs would you use? How many hidden layers?

📝 **Final Challenge 2**: You're training a network to learn the Plummer potential $\Phi(r) = -GM/\sqrt{r^2 + a^2}$. What physics-informed loss terms would you include?

📝 **Final Challenge 3**: Your network has 90% training accuracy but only 60% validation accuracy. What regularization techniques would you try first?

*Think about these as you begin your JAX implementation!*