# The Learnable Universe

**Modules 5 and 6 Draft Outlines.**

## Module 5: From Observations to Inference

### Relevant Projects

- **Project 4**: MCMC/HMC Type 1A Supernovae Cosmology -- Inferring the Universe ✅

## Module 6: From Inference to Intelligence

**Relevant Projects:**

- **Project 5**: GP emulating N-body simulations ✅
- **Final Project**: Neural networks from scratch in JAX ✅

This means Module 6 should:

- Give GPs sufficient depth for Project 5 (emulation focus)
- Build neural networks from scratch for Final Project
- Include Physics-Informed Learning (PINNs, Neural ODEs)

## Draft Outline for Module 6: From Inference to Intelligence

**Part 1: The Learning Framework.**

- Supervised learning as inference
- Loss functions and likelihood
- Bias-variance tradeoff
- Regularization as priors

**Part 2: Gaussian Processes** (Heavy emphasis for Project 5)

- Parametric vs. non-parametric models
- GPs as Bayesian function approximation
- Kernels encode beliefs about smoothness
- Build GP from scratch
- Emulation and surrogate modeling
- **Direct prep for Project 5**: Emulating N-body simulations

**Part 3: Neural Networks Fundamentals** (Prep for Final Project)

- Universal approximation theorem
- Neurons and activation functions
- Network architectures
- Backpropagation from first principles
- Build simple network in JAX from scratch
- **Example**: Star/galaxy classification

### Part 4: Modern Architectures

- Convolutional Neural Networks (CNNs) and inductive biases
- Recurrent Neural Networks (RNNs)
- Transformers
- Attention mechanisms
- When to use GPs vs. NNs

### Part 5: Physics-Informed Learning

- Neural ODEs
- Physics-Informed Neural Networks (PINNs)
- Incorporating physical constraints
- Symmetries and equivariance
- **Examples**: Learning stellar structure, radiative transfer

Perfect! Let me propose revised module structures without timing:

---

**"The Learnable Universe"** section contains two modules with parallel structure:

**Module 5 - "From Observations to Inference"**:

*How we transform photons into knowledge about an uncertain universe.*

**Module 6: "From Inference to Intelligence"**:

*How we build machines that learn patterns we never programmed.*

**Why this parallel structure works:**

- Creates a narrative arc: Observations → Inference → Intelligence
- Both emphasize the JOURNEY (the "from...to..." structure)
- Shows progression from human reasoning to machine learning
- Maintains your big-picture, connecting philosophy

---

## Module 5 - From Observations to Inference

### Part 1: Philosophy of Measurement

- What is a model? (compression, beliefs)
- Forward vs. inverse problems
- The measurement chain

### Part 2: Bayesian Framework

- Probability as extended logic
- Likelihood, Prior, Posterior
- Bayes' theorem from first principles

### Part 3: Computational Reality

- Curse of dimensionality
- Why sampling? (Monte Carlo review from Module 1)
- Introduction to MCMC concept

### Part 4: MCMC in Practice

- Build Metropolis from scratch
- Markov chains and detailed balance
- Convergence diagnostics
- **Example**: Cepheid distances (prep for SNe)

### Part 5: Advanced Topics in MCMC

- Gibbs sampling overview (?)
- Affine-invariant samplers (?)
- Hamiltonian Monte Carlo (essential for Project 4)
- No-U-Turn Sampler (NUTS) overview
- Connection to SNe cosmology project
- When to use which sampler

**Assessment**: Project 4 - SNe Cosmology with MCMC

---

## Module 6 - From Inference to Intelligence

**Part 1: The Learning Framework**

- Supervised learning as inference
- Loss functions and likelihood
- Bias-variance tradeoff (callback to Module 1)
- Regularization as priors (callback to Module 5)

**Part 2: Gaussian Processes**

- GPs as Bayesian function approximation
- Kernels encode beliefs about smoothness
- Build GP from scratch
- **Example**: Emulating expensive simulations
- Project 5 preview

**Part 3: Neural Networks Fundamentals**

- Universal approximation theorem
- Backpropagation from first principles
- Build simple network in JAX
- **Example**: Star/galaxy classification

**Part 4: Modern Architectures**

- CNNs and inductive biases from physics
- Attention mechanisms
- Training strategies and optimization
- When to use GPs vs. NNs
- **Example**: Morphological classification

**Part 5: Physics-Informed Learning** ⭐ NEW

- Neural ODEs (learning differential equations)
- Physics-Informed Neural Networks (PINNs)
- Incorporating physical constraints into loss functions
- Symmetries and equivariance in neural architectures
- **Example**: Learning stellar structure equations OR learning radiative transfer
- **Connection**: Brings course full circle - using ML to solve the physics from Modules 1-4

**Assessment**: Project 5 (GP or NN application) + Final Project (Neural network from scratch in JAX)

---

**Why Part 5 (Physics-Informed Learning) is Perfect:**

1. **Closes the loop**: Students learn physics (Modules 1-4) → learn inference (Module 5) → teach machines to learn physics (Module 6 Part 5)

2. **Cutting edge**: Neural ODEs and PINNs are hot topics in scientific ML

3. **Glass-box philosophy**: Understanding how to incorporate physical constraints shows deep understanding of BOTH physics AND ML

4. **Natural examples**:
   - Neural ODE learning stellar structure from Module 2
   - PINN learning radiative transfer from Module 4
   - Enforcing conservation laws in neural networks

5. **Career relevant**: This is where physics PhDs add unique value in ML - domain knowledge

**Suggested brief outline for Part 5**:

**5.1: Why Physics Matters in ML**
- Incorporating domain knowledge
- Data efficiency through constraints
- Interpretability and physical validity

**5.2: Neural ODEs**
- Learning differential equations from data
- **Example**: Rediscovering stellar structure equations

**5.3: Physics-Informed Neural Networks (PINNs)**
- Loss functions with physical constraints
- Boundary conditions and conservation laws
- **Example**: Learning radiative transfer

**5.4: Symmetries and Equivariance**
- Rotational/translational invariance
- Building physics into architecture
- **Example**: Galaxy morphology with rotational equivariance

Does this structure work for you?
