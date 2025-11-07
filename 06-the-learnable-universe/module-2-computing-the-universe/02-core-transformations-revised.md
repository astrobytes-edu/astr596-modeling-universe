---
title: "Part 2: Core Transformations — JAX's Computational Superpowers"
subtitle: "Automatic Differentiation, JIT Compilation, and Vectorization | Computing the Universe | ASTR 596"
---

**Prerequisites**: [Part 1: Conceptual Foundations](part-01-conceptual-foundations.md) completed

> *"The purpose of computing is insight, not numbers."* — Richard Hamming
>
> *"A language that doesn't affect the way you think about programming is not worth knowing."* — Alan Perlis

---

(learning-outcomes-part2)=
## Learning Outcomes

By the end of Part 2, you will be able to:

- [ ] **Derive** how automatic differentiation computes gradients using the chain rule
- [ ] **Contrast** forward-mode and reverse-mode autodiff and explain when each is efficient
- [ ] **Apply** `jax.grad` to compute numerically exact derivatives of physics functions
- [ ] **Explain** how JIT compilation transforms Python to machine code via XLA
- [ ] **Identify** code patterns that JIT can and cannot optimize
- [ ] **Use** `jax.vmap` to eliminate explicit loops and batch computations
- [ ] **Compose** transformations (jit ∘ vmap ∘ grad) and predict the result
- [ ] **Debug** common JAX errors by recognizing constraint violations

---

(roadmap-part2)=
## Roadmap: From Concepts to Practice

**Priority: 🔴 Essential**

**Part 1 gave you the WHY.** Now we learn the HOW.

You understand:
- ✅ Why functional programming (enables transformations)
- ✅ Why pure functions (enables JIT, autodiff, vmap)
- ✅ Why explicit control flow (enables tracing)
- ✅ What computational graphs are (mental model)

**Part 2 teaches technical mastery**:

1. **`grad`** — Automatic differentiation: How does JAX compute numerically exact derivatives?
2. **`jit`** — JIT compilation: How does XLA turn Python into machine code?
3. **`vmap`** — Vectorization: How do you batch operations without loops?
4. **Composing** — How do transformations combine?

**Structure**: For each transformation, you'll learn:
- **Mathematics** (glass-box: how it works internally)
- **API** (how to use it in practice)
- **When to use** (and when NOT to use)
- **Common errors** (and how to debug them)
- **Physics examples** (gravitational forces, supernova distances, MCMC)

### Project 5 Preview

**In 2 weeks, you'll refactor your Project 2 N-body code using these transformations:**
- Replace force calculation loops with `vmap`
- Add `jit` to compile the integration step
- Use `grad` for exact force computation (optional: compare to your finite differences)
- Package as installable JAX library with tests

**This part teaches the tools. Project 5 makes you use them.**

---

(automatic-differentiation)=

## 2.1: Automatic Differentiation — Computing Numerically Exact Gradients

**Priority: 🔴 Essential**

### The Problem: Finite Differences Don't Scale

**Your Project 2 N-body code** likely computed forces via finite differences (if you implemented adaptive timesteps or force validation):

```python
def compute_forces_finite_diff(positions, masses, h=1e-5):
    """Compute forces via finite differences: F = -∇U ≈ -(U(r+h) - U(r-h))/2h"""
    N, d = positions.shape  # N particles, d=3 dimensions
    forces = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(d):
            # Perturb position of particle i in dimension j
            pos_plus = positions.copy()
            pos_plus[i, j] += h
            pos_minus = positions.copy()
            pos_minus[i, j] -= h
            
            # Central difference
            U_plus = gravitational_potential(pos_plus, masses)
            U_minus = gravitational_potential(pos_minus, masses)
            forces[i, j] = -(U_plus - U_minus) / (2*h)
    
    return forces
    # Cost: 2*N*d potential evaluations (N=100, d=3 → 600 evaluations!)
```

**Problems**:
1. **Expensive**: $O(Nd)$ potential evaluations for $N$ particles in $d$ dimensions
2. **Approximate**: Choosing $h$ is tricky (roundoff vs truncation error)
3. **Doesn't scale**: For $N=1000$, this is 6000 potential evaluations per timestep

**Project 4 (MCMC)** had the same problem for gradient-based sampling:
```python
def grad_log_posterior(theta, h=1e-5):
    """Gradient for HMC via finite differences."""
    d = len(theta)
    grad = np.zeros(d)
    for i in range(d):
        theta_plus = theta.copy()
        theta_plus[i] += h
        theta_minus = theta.copy()
        theta_minus[i] -= h
        grad[i] = (log_posterior(theta_plus) - log_posterior(theta_minus)) / (2*h)
    return grad
    # Cost: 2*d function evaluations (d=3 cosmology parameters → 6 likelihood calculations)
```

**Automatic differentiation solves all three problems:**

```python
import jax

# Your gravitational potential function
def U(positions, masses):
    # ... your code ...
    return total_potential  # scalar

# Get numerically exact gradient function (machine precision, NOT finite differences!)
grad_U = jax.grad(U, argnums=0)  # ∂U/∂positions

# Compute forces
forces = -grad_U(positions, masses)  # Cost: ~3× one potential evaluation
# For N=100: equivalent to ~3 potential calculations (200× faster than finite differences)
# Note: Gradients are exact wherever the function is differentiable; non-smooth operations
# (e.g., abs, jnp.where with discontinuous branches) may need smoothing or subgradients
```

**This section explains HOW this works.**

:::{margin}
**Automatic Differentiation (Autodiff)**
Systematic application of the chain rule at the level of elementary operations. Exact to machine precision, not an approximation.
:::

---

### What Is Automatic Differentiation?

**Automatic differentiation (autodiff)** is NOT:
- ❌ Symbolic differentiation (like SymPy or Mathematica) — manipulates expressions
- ❌ Numerical differentiation (finite differences) — approximate

**Autodiff IS**:
- ✅ Systematic application of the chain rule to computational graphs
- ✅ Exact to machine precision (only limited by floating-point arithmetic)
- ✅ Efficient (typically 2-4× cost of forward evaluation for reverse-mode)

**Key insight**: Every complex function is built from elementary operations ($+$, $\times$, $\sin$, $\exp$, etc.) whose derivatives we know exactly. The chain rule tells us how to combine these.

---

### The Chain Rule: Foundation of Autodiff

**Multivariable chain rule** (what we actually use):

If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$, then:

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
$$

**General form** (for computational graphs):

For a function composed of operations $v_1, v_2, \ldots, v_n$:

$$
\frac{\partial v_n}{\partial v_i} = \sum_{j \in \text{children}(i)} \frac{\partial v_n}{\partial v_j} \frac{\partial v_j}{\partial v_i}
$$

This recursive structure is what autodiff exploits.

---

### Forward-Mode vs Reverse-Mode Autodiff

There are **two ways** to apply the chain rule through a computational graph:

:::{margin}
**Forward-Mode**: Propagates derivatives forward (efficient when outputs ≫ inputs)

**Reverse-Mode**: Propagates derivatives backward (efficient when inputs ≫ outputs)
:::

#### Forward-Mode Autodiff

**Idea**: Propagate derivative **forward** alongside the computation.

**Example**: Compute $f(x) = \sin(x^2)$ and its derivative simultaneously.

**Forward pass**:
1. $v_1 = x$, $\dot{v}_1 = 1$ (input)
2. $v_2 = v_1^2$, $\dot{v}_2 = 2v_1 \cdot \dot{v}_1 = 2x$ (power rule)
3. $v_3 = \sin(v_2)$, $\dot{v}_3 = \cos(v_2) \cdot \dot{v}_2 = 2x\cos(x^2)$ (chain rule)

**Result**: $f(x) = v_3$, $f'(x) = \dot{v}_3 = 2x \cos(x^2)$ ✅ Exact!

**Cost**: One forward pass per input dimension. For $d$ inputs → $d$ forward passes.

**When to use**: Few inputs, many outputs (e.g., Jacobian of $\mathbb{R}^2 \to \mathbb{R}^{1000}$).

---

#### Reverse-Mode Autodiff (Backpropagation)

**This is what JAX uses by default** (and what PyTorch/TensorFlow use for neural networks).

**Idea**: Propagate derivative **backward** from outputs to inputs.

**Example**: Same function $f(x) = \sin(x^2)$, but compute derivative **after** forward pass.

**Forward pass** (build computational graph):
1. $v_1 = x$
2. $v_2 = v_1^2$
3. $v_3 = \sin(v_2)$

**Backward pass** (propagate adjoints $\bar{v}_i = \frac{\partial f}{\partial v_i}$):

:::{margin}
**Adjoint notation**: $\bar{v}_i$ represents $\frac{\partial f}{\partial v_i}$ (sensitivity of output to intermediate value $v_i$)
:::

1. $\bar{v}_3 = 1$ (seed: output's gradient w.r.t. itself)
2. $\bar{v}_2 = \bar{v}_3 \cdot \frac{\partial v_3}{\partial v_2} = 1 \cdot \cos(v_2) = \cos(x^2)$ (chain rule backward)
3. $\bar{v}_1 = \bar{v}_2 \cdot \frac{\partial v_2}{\partial v_1} = \cos(x^2) \cdot 2v_1 = 2x \cos(x^2)$ (chain rule backward)

**Result**: $\frac{\partial f}{\partial x} = \bar{v}_1 = 2x \cos(x^2)$ ✅ Exact!

**Key insight**: Each step multiplies the incoming gradient by the local derivative and passes it backward. This is **exactly the chain rule**, applied automatically.

**Cost**: One backward pass **regardless of input dimension**. Typically 2-4× cost of forward pass.

**When to use**: Many inputs, few outputs (e.g., gradient of scalar loss: $\mathbb{R}^d \to \mathbb{R}$).

---

### Why Reverse-Mode for Physics?

**Most physics/ML applications have this structure**:

$$
f: \mathbb{R}^d \to \mathbb{R} \quad \text{(many parameters → one scalar)}
$$

**Examples from your course**:
- **Gravitational potential**: $(\mathbf{r}_1, \ldots, \mathbf{r}_N) \to U(\mathbf{r})$ (scalar energy)
- **Log-posterior (Project 4)**: $(\Omega_m, \Omega_\Lambda, H_0) \to \log P(D|\theta)$ (scalar)
- **Loss function**: $(w_1, \ldots, w_d) \to \mathcal{L}(w)$ (scalar)

**Cost comparison for $d$ inputs, 1 output**:
- **Forward-mode**: $d$ forward passes
- **Reverse-mode**: 1 forward + 1 backward pass ≈ 3-4 forward passes

**For $d = 1000$** (neural network parameters):
- Forward-mode: 1000 forward passes
- Reverse-mode: 3-4 forward passes
- **Speedup: ~250-300×**

**For $d = 3$ (cosmology parameters, Project 4)**:
- Forward-mode: 3 forward passes
- Reverse-mode: 3-4 forward passes
- **Speedup: ~1× (roughly equal)**

**This is why JAX defaults to reverse-mode autodiff** (`jax.grad` uses backpropagation).

:::{admonition} 💡 Key Takeaway
:class: important

**Reverse-mode autodiff is efficient when you have many inputs → one output.**

For physics: energy functions, likelihoods, loss functions all have this structure.

Cost is ~3-4× one function evaluation, **independent of parameter count**.
:::

---

### Using `jax.grad`: Basic Patterns

#### Pattern 1: Single-Argument Functions

```python
import jax
import jax.numpy as jnp

# Simple function
def f(x):
    return jnp.sum(x**2)

# Get gradient function
grad_f = jax.grad(f)

# Compute gradient
x = jnp.array([1.0, 2.0, 3.0])
gradient = grad_f(x)
print(gradient)  # [2., 4., 6.] = 2*x (exact!)
```

#### Pattern 2: Multi-Argument Functions

```python
# Gravitational potential between two particles
def U_pair(r1, r2, m1=1.0, m2=1.0, G=1.0):
    """Potential energy between two point masses."""
    r_vec = r1 - r2
    r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # Softening
    return -G * m1 * m2 / r

# Gradient w.r.t. first argument (position of particle 1)
grad_U_r1 = jax.grad(U_pair, argnums=0)

# Gradient w.r.t. second argument (position of particle 2)
grad_U_r2 = jax.grad(U_pair, argnums=1)

# Force on particle 1 (F = -∇U)
r1 = jnp.array([0.0, 0.0, 0.0])
r2 = jnp.array([1.0, 0.0, 0.0])
F1 = -grad_U_r1(r1, r2)  # Force on particle 1
print(F1)  # Should point toward r2
```

**The `argnums` parameter** specifies which argument to differentiate with respect to:
- `argnums=0`: gradient w.r.t. first argument
- `argnums=1`: gradient w.r.t. second argument
- `argnums=(0,1)`: gradients w.r.t. both arguments (returns tuple)

#### Pattern 3: Auxiliary Data (Static Arguments)

```python
def log_posterior(theta, data):
    """Cosmology log-posterior (Project 4)."""
    H0, Omega_m, Omega_Lambda = theta
    
    # Prior (Gaussian)
    log_prior = -0.5 * jnp.sum((theta - data['prior_mean'])**2 / data['prior_std']**2)
    
    # Likelihood (chi-squared for supernovae)
    predicted_mu = distance_modulus(data['redshifts'], H0, Omega_m, Omega_Lambda)
    residuals = data['mu_obs'] - predicted_mu
    log_like = -0.5 * jnp.sum((residuals / data['sigma_mu'])**2)
    
    return log_prior + log_like

# Gradient w.r.t. theta (first argument), data is auxiliary
grad_log_posterior = jax.grad(log_posterior, argnums=0)

# Use in HMC
theta = jnp.array([70.0, 0.3, 0.7])
gradient = grad_log_posterior(theta, data)  # data is not differentiated
```

**Key point**: Only arguments specified in `argnums` are differentiated. Others are treated as constants.

---

:::{admonition} 🔗 Connection to Module 1: Information and Derivatives
:class: note

**Module 1 insight**: Entropy $S = -\sum p_i \log p_i$ measures information content.

**Autodiff insight**: Gradients measure sensitivity—how information flows through computation.

The adjoint $\bar{v}_i = \frac{\partial L}{\partial v_i}$ answers: "How much does output $L$ change if $v_i$ changes?"

This is **information propagation** through your computational graph. Same conceptual structure as MaxEnt: constraints (purity) → unique optimal solution (backpropagation algorithm).
:::

---

(jit-compilation)=
## 2.2: JIT Compilation — From Python to Machine Code

**Priority: 🔴 Essential**

### The Problem: Python is Slow for Tight Loops

**Your Project 2 integrator** calls `compute_forces()` thousands of times per orbit:

```python
# Leapfrog integration (pure Python + NumPy)
for step in range(10000):
    # Compute forces
    forces = compute_forces(positions, masses)  # Called 10,000 times!
    
    # Update velocities and positions
    velocities += 0.5 * dt * forces / masses
    positions += dt * velocities
    forces = compute_forces(positions, masses)
    velocities += 0.5 * dt * forces / masses
```

**Even with NumPy**, this is slower than compiled code because:
1. **Python interpreter overhead** on every loop iteration
2. **Function call overhead** (10,000 calls to `compute_forces`)
3. **No cross-operation optimization** (NumPy optimizes each operation independently)

**JIT compilation solves this**: Compile to machine code once, execute fast many times.

:::{margin}
**Just-In-Time (JIT) Compilation**
Compiling code at runtime based on actual data shapes/types, then executing the compiled version.
:::

---

### How JIT Compilation Works

**JAX's compilation pipeline**:

```
Python function → Traced → XLA IR → Optimized → Machine code
```

**Step 1: Tracing** — JAX runs your function once with abstract values (shapes + dtypes, no actual numbers) to build a computation graph.

**Step 2: XLA compilation** — XLA (Accelerated Linear Algebra) optimizes the graph:
- Fuse operations (eliminate intermediate arrays)
- Vectorize loops (SIMD instructions)
- Memory optimization (minimize allocations)
- Target-specific optimization (CPU, GPU, TPU)

**Step 3: Execution** — Run optimized machine code.

**Key insight**: Compilation happens once (first call). Subsequent calls are fast.

---

### Using `jax.jit`: Basic Patterns

#### Pattern 1: Simple Function

```python
import jax

def slow_function(x):
    """Without JIT: interpreted Python."""
    return jnp.sum(x**2 + jnp.sin(x))

# Add JIT decorator
@jax.jit
def fast_function(x):
    """With JIT: compiled to machine code."""
    return jnp.sum(x**2 + jnp.sin(x))

# First call: slow (compilation overhead)
x = jnp.arange(1000000)
result = fast_function(x)  # ~100ms (compilation + execution)

# Subsequent calls: fast (just execution)
result = fast_function(x)  # ~1ms (only execution)
```

**Typical speedup**: 5-50× for NumPy-style code (depends on operation complexity).

#### Pattern 2: Multi-Argument Functions

```python
@jax.jit
def gravitational_potential(positions, masses):
    """Total gravitational potential energy of N-body system."""
    N = len(positions)
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)
            U += -masses[i] * masses[j] / r
    return U

# JIT compiles based on shapes
positions = jnp.array([[0.,0.,0.], [1.,0.,0.]])  # Shape: (2, 3)
masses = jnp.array([1.0, 1.0])                   # Shape: (2,)
U = gravitational_potential(positions, masses)   # Compiles for these shapes
```

**Important**: JIT compiles separately for each unique combination of input shapes/dtypes.

---

### What JIT Can and Cannot Do

**✅ JIT-compatible patterns**:
```python
# Pure numerical operations
@jax.jit
def compute_energy(positions, velocities, masses):
    kinetic = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
    potential = gravitational_potential(positions, masses)
    return kinetic + potential
```

**❌ JIT-incompatible patterns**:
```python
# 1. Shape depends on data values (dynamic shapes)
@jax.jit
def bad_function(x):
    n = int(x[0])  # Value not known at compile time!
    return jnp.arange(n)  # Shape depends on n → error!

# 2. Side effects (I/O, mutation)
@jax.jit
def bad_function(x):
    print(f"x = {x}")  # Prints during tracing, not execution!
    return x**2

# 3. Python control flow on traced values
@jax.jit
def bad_function(x):
    if x > 0:  # x is abstract during tracing → error!
        return x**2
    else:
        return -x**2
```

**The solution**: Use JAX's structured control flow (`lax.cond`, `lax.scan`, `lax.while_loop`).

---

### Structured Control Flow

#### `lax.cond`: Conditional Branching

```python
from jax import lax

# ❌ BAD: Python if on traced value
@jax.jit
def clip(x, threshold):
    if x > threshold:  # ConcretizationError!
        return threshold
    else:
        return x

# ✅ GOOD: Use lax.cond
@jax.jit
def clip(x, threshold):
    return lax.cond(
        x > threshold,
        lambda x: threshold,  # True branch
        lambda x: x,          # False branch
        x
    )
```

#### `lax.scan`: Loop with Carry State

**Use case**: Iterative updates where each step depends on previous (ODE integration, MCMC chains).

```python
from jax import lax

def leapfrog_step(state, _):
    """One leapfrog timestep."""
    pos, vel, masses = state
    
    # Kick
    forces = compute_forces(pos, masses)
    vel = vel + 0.5 * dt * forces / masses
    
    # Drift
    pos = pos + dt * vel
    
    # Kick
    forces = compute_forces(pos, masses)
    vel = vel + 0.5 * dt * forces / masses
    
    return (pos, vel, masses), pos  # (carry, output)

# Integrate for 1000 steps (compiled efficiently!)
initial_state = (positions_0, velocities_0, masses)
final_state, trajectory = lax.scan(
    leapfrog_step,
    initial_state,
    jnp.arange(1000)  # Iteration counter (not used in body)
)
# trajectory.shape = (1000, N, 3) - positions at each timestep
```

**Key advantage**: `lax.scan` is JIT-compilable. Python `for` loops are not (when JIT-ed).

---

:::{admonition} 🔗 Connection to Project 5
:class: note

**In Project 5**, you'll refactor your N-body integrator to use:
- `jax.jit` on your force calculation (10-50× speedup)
- `lax.scan` for orbit integration (enables JIT of entire trajectory)
- `jax.vmap` for pairwise interactions (see next section)

**Don't implement it yet**—Project 5 is where you do the work. This part teaches you the tools.
:::

---

(vectorization)=
## 2.3: Vectorization with `vmap` — Eliminating Loops

**Priority: 🔴 Essential**

### The Problem: Loops Over Independent Operations

**Observational astrophysics is inherently batched**:
- Project 4: 43 Type Ia supernovae → 43 distance moduli to compute
- 10,000 Cepheids → 10,000 period-luminosity evaluations
- N-body: $N$ particles → $N$ force calculations (or $N^2$ pairwise interactions)

**The naive approach** (Python loops):
```python
# Compute distance modulus for all supernovae
distances = []
for z in redshifts:  # 43 supernovae
    d = distance_modulus(z, H0, Omega_m, Omega_Lambda)
    distances.append(d)
distances = jnp.array(distances)
```

**Problems**:
1. Slow (Python loop overhead)
2. Can't JIT (dynamic list building)
3. Doesn't parallelize on GPU

---

### What is `vmap`?

**`vmap` (vectorizing map)** automatically vectorizes a function over a batch dimension:

```python
# Single-input function
def distance_modulus(z, H0, Omega_m, Omega_Lambda):
    """Distance modulus for one supernova."""
    # ... integration over redshift ...
    return mu

# Vectorize over redshifts (batch dimension)
distance_modulus_batch = jax.vmap(
    distance_modulus,
    in_axes=(0, None, None, None)  # Batch over z, cosmology params are constant
)

# One call computes all 43 distances
redshifts = jnp.array([...])  # Shape: (43,)
distances = distance_modulus_batch(redshifts, H0, Omega_m, Omega_Lambda)
# distances.shape = (43,) - all computed in parallel
```

**Key insight**: You write code for **one example**, `vmap` handles the batch automatically.

---

### Using `vmap`: Basic Patterns

#### Pattern 1: Batch Over First Argument

```python
def f(x):
    """Function of single input."""
    return x**2 + jnp.sin(x)

# Vectorize
f_batched = jax.vmap(f)

# Apply to batch
x_batch = jnp.array([1.0, 2.0, 3.0, 4.0])
results = f_batched(x_batch)  # [f(1), f(2), f(3), f(4)]
```

#### Pattern 2: Specify Batch Axes with `in_axes`

```python
def f(x, y):
    """Function of two inputs."""
    return x * y + jnp.sum(x**2)

# Batch over first argument only
f_batched = jax.vmap(f, in_axes=(0, None))

x_batch = jnp.array([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2)
y = jnp.array([10, 20])                        # Shape: (2,) - same for all
results = f_batched(x_batch, y)                # Shape: (3,)
```

**`in_axes` values**:
- `0`: Batch over first dimension of this argument
- `1`: Batch over second dimension
- `None`: Don't batch (same value for all examples)

#### Pattern 3: Multi-Dimensional Batching

```python
# Supernova likelihood for one supernova
def log_likelihood_single(mu_obs, sigma_mu, z, H0, Omega_m, Omega_Lambda):
    """Log-likelihood for single supernova."""
    mu_pred = distance_modulus(z, H0, Omega_m, Omega_Lambda)
    return -0.5 * ((mu_obs - mu_pred) / sigma_mu)**2

# Vectorize over all 43 supernovae
log_likelihood_batch = jax.vmap(
    log_likelihood_single,
    in_axes=(0, 0, 0, None, None, None)  # Batch over data, not parameters
)

# Compute for all supernovae
log_likes = log_likelihood_batch(mu_obs, sigma_mu, redshifts, H0, Omega_m, Omega_Lambda)
# log_likes.shape = (43,)
total_log_like = jnp.sum(log_likes)
```

---

### Advanced: Nested `vmap` for Pairwise Interactions

**N-body forces**: Each particle $i$ feels force from all other particles $j \neq i$.

```python
def force_on_i_from_j(r_i, r_j, m_i, m_j):
    """Force on particle i from particle j."""
    r_vec = r_j - r_i
    r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # Softening
    F = (m_i * m_j / r**3) * r_vec
    return F

# Vectorize over j (sum forces from all particles on i)
force_on_i = jax.vmap(
    force_on_i_from_j,
    in_axes=(None, 0, None, 0)  # r_i and m_i fixed, r_j and m_j batched
)

# Vectorize over i (compute forces on all particles)
compute_all_forces = jax.vmap(
    force_on_i,
    in_axes=(0, None, 0, None)  # r_i and m_i batched, positions and masses arrays
)

# Usage
positions = jnp.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])  # 3 particles
masses = jnp.array([1.0, 1.0, 1.0])

forces = compute_all_forces(positions, positions, masses, masses)
# forces.shape = (3, 3, 3) - force on i from j in each direction
# Need to sum over j and exclude i==j
```

**Handling self-interaction**: Use masking to exclude $i=j$ terms.

:::{admonition} 🔗 Connection to Project 2
:class: note

This double-vmap pattern is exactly how you'd eliminate the $O(N^2)$ nested loops in your force calculation. In Project 5, you'll implement this pattern.

**Expected speedup**: For $N=100$ particles on GPU: 100-500× faster than Python loops.
:::

---

(composing-transformations)=
## 2.4: Composing Transformations

**Priority: 🔴 Essential**

### The Power of Composition

**JAX transformations compose** because they all respect functional purity:

```python
# Composition: jit ∘ vmap ∘ grad
@jax.jit
def optimized_batch_gradients(params, X_batch, Y_batch):
    def loss_single(params, x, y):
        prediction = model(params, x)
        return (prediction - y)**2
    
    # Gradient of single loss
    grad_loss = jax.grad(loss_single, argnums=0)
    
    # Vectorize over batch
    grad_batch = jax.vmap(grad_loss, in_axes=(None, 0, 0))
    
    # Return mean gradient
    return jnp.mean(grad_batch(params, X_batch, Y_batch), axis=0)
```

**Order matters**:
- `jit(vmap(grad(f)))`: Correct ✅ — compile the vectorized gradient
- `vmap(jit(grad(f)))`: Still works, but compiles once per batch element (less efficient)
- `grad(jit(vmap(f)))`: Usually works, but gradient includes JIT overhead

**General rule**: Outermost to innermost: `jit → vmap → grad`

---

### Common Composition Patterns

#### Pattern 1: HMC with Gradient (Project 4)

```python
# Log-posterior for cosmology
def log_posterior(theta, data):
    """Log P(theta|data) for cosmological parameters."""
    # ... your likelihood + prior ...
    return log_prob

# Build gradient function
@jax.jit
def grad_log_posterior(theta, data):
    return jax.grad(log_posterior, argnums=0)(theta, data)

# Use in HMC leapfrog
def hmc_step(theta, key, data, epsilon=0.01, L=10):
    """One HMC step with gradient-based leapfrog."""
    p = jax.random.normal(key, theta.shape)
    
    # Leapfrog integration
    p = p + 0.5 * epsilon * grad_log_posterior(theta, data)
    for _ in range(L):
        theta = theta + epsilon * p
        p = p + epsilon * grad_log_posterior(theta, data)
    p = p + 0.5 * epsilon * grad_log_posterior(theta, data)
    
    # Metropolis accept/reject (your code)
    # ...
    return theta_new
```

**Why this works**: `grad_log_posterior` is called inside `hmc_step`, which is called in a loop (MCMC chain). Composition `jit(grad(log_posterior))` ensures:
1. Gradient computed exactly via autodiff
2. Compiled once, executes fast for 10,000 steps
3. No recompilation if `theta` changes (same shape)

**Expected speedup**: ~100× faster than finite-difference gradients.

#### Pattern 2: Batch Loss with Gradients (Preview of Module 7)

```python
# Loss for single data point
def loss_single(params, x, y):
    prediction = model(params, x)
    return (prediction - y)**2

# Batch loss: mean over dataset
def loss_batch(params, X, Y):
    losses = jax.vmap(loss_single, in_axes=(None, 0, 0))(params, X, Y)
    return jnp.mean(losses)

# Gradient for optimization
@jax.jit
def grad_loss(params, X, Y):
    return jax.grad(loss_batch, argnums=0)(params, X, Y)

# Training loop (Module 7)
for step in range(1000):
    gradient = grad_loss(params, X_batch, Y_batch)
    params = params - learning_rate * gradient
```

**Composition**: `jit(grad(vmap(loss_single)))`

---

(precision-and-dtype)=

## 2.5: Floating-Point Precision and dtype Management

**Priority: 🔴 Essential**

### Why Precision Matters for Physics

**JAX defaults to 32-bit floats (float32)** for compatibility with ML accelerators (TPUs, GPUs) and speed. But **physics simulations often require 64-bit precision (float64)** to maintain accuracy over long integrations.

**The difference**:

- **float32**: ~7 decimal digits of precision, fastest on GPUs
- **float64**: ~15-16 decimal digits of precision, essential for energy conservation

:::{important} Energy Conservation Requires float64

In Project 5, you'll integrate N-body systems over thousands of timesteps. With **float32**, roundoff errors accumulate:

- Energy drift: ~$10^{-6}$ (noticeable after 1000 steps)
- Angular momentum drift: ~$10^{-5}$

With **float64**, symplectic integrators conserve energy to:

- Energy drift: ~$10^{-12}$ (near machine precision)
- Suitable for publication-quality simulations
:::

---

### Enabling 64-bit Precision

**Default behavior** (float32):

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # float32
```

**Enable float64 globally**:

```python
import jax
jax.config.update("jax_enable_x64", True)

x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # float64
```

:::{admonition} Where to Place This
:class: tip

Put `jax.config.update("jax_enable_x64", True)` at the **top of your script**, immediately after imports:

```python
import jax
import jax.numpy as jnp

# Enable 64-bit precision for physics
jax.config.update("jax_enable_x64", True)

# Now all arrays default to float64
```
:::

---

### Mixing NumPy and JAX: Precision Pitfalls

**Problem**: NumPy defaults to float64, JAX defaults to float32. Mixing them creates silent casts.

```python
import numpy as np
import jax.numpy as jnp

# NumPy array (float64)
x_np = np.array([1.0, 2.0, 3.0])
print(x_np.dtype)  # float64

# Convert to JAX (downcasts to float32 if x64 not enabled!)
x_jax = jnp.asarray(x_np)
print(x_jax.dtype)  # float32 (precision loss!)
```

**Solution**:

1. **Enable float64 globally** (recommended for physics)
2. **Or explicitly specify dtype**:

```python
x_jax = jnp.asarray(x_np, dtype=jnp.float64)
```

---

### Performance Cost of float64

**Tradeoffs**:

| Aspect | float32 | float64 |
|--------|---------|---------|
| **Memory** | 4 bytes/number | 8 bytes/number (2× larger) |
| **Speed (CPU)** | Baseline | ~5-10% slower |
| **Speed (GPU)** | Baseline | ~10-20% slower (depends on hardware) |
| **Precision** | ~$10^{-7}$ | ~$10^{-16}$ |

**When to use float64**:

- ✅ Long time integrations (N-body, stellar evolution)
- ✅ Energy/momentum conservation requirements
- ✅ Ill-conditioned problems (matrix inversions, eigenvalues)
- ✅ Publication-quality results

**When float32 is OK**:

- ✅ Short simulations (< 100 timesteps)
- ✅ ML training (gradients approximate anyway)
- ✅ Prototyping and debugging

:::{admonition} 🔗 Connection to Part 3: Energy Conservation
:class: note

In **Part 3, Section 3.1** (N-body systems), you'll validate energy conservation. Expected drift depends critically on precision:

- **float32**: Expect ~$10^{-6}$ relative energy error
- **float64**: Expect ~$10^{-12}$ relative energy error

If your energy conservation tests fail, **check your dtype first**!
:::

---

### Explicit dtype Control

Sometimes you need mixed precision (e.g., float64 physics, float32 for visualization).

**Specify dtype explicitly**:

```python
# Force float64 for positions (high precision)
positions = jnp.zeros((N, 3), dtype=jnp.float64)

# Force float32 for visualization arrays (save memory)
render_buffer = jnp.zeros((1024, 1024, 3), dtype=jnp.float32)
```

**Check dtype in functions**:

```python
def gravitational_force(r1, r2, m1, m2, G=6.67e-8):
    """Compute force, preserving input dtype."""
    r_vec = r2 - r1
    r = jnp.linalg.norm(r_vec)

    # Check: What dtype do we have?
    print(f"Working with dtype: {r_vec.dtype}")

    force_mag = G * m1 * m2 / r**2
    return force_mag * r_vec / r
```

---

### Common dtype Errors

#### Error 1: Comparing float32 to float64 Constants

```python
# ❌ BAD: Comparing float32 to hardcoded float (becomes float64)
x = jnp.array([1.0], dtype=jnp.float32)
if x > 0.5:  # 0.5 is float64 in Python!
    ...

# ✅ GOOD: Match dtypes
threshold = jnp.array(0.5, dtype=x.dtype)
if x > threshold:
    ...
```

#### Error 2: Accumulating in Wrong Precision

```python
# ❌ BAD: Summing float32, result in float32 (precision loss)
values_f32 = jnp.array([1e-8, 1e-8, ...], dtype=jnp.float32)
total = jnp.sum(values_f32)  # Roundoff dominates!

# ✅ GOOD: Accumulate in higher precision
values_f64 = values_f32.astype(jnp.float64)
total = jnp.sum(values_f64)  # Then cast back if needed
```

---

### Debugging dtype Issues

**Quick dtype audit**:

```python
def audit_dtypes(tree):
    """Print dtypes of all arrays in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    for i, leaf in enumerate(leaves):
        if hasattr(leaf, 'dtype'):
            print(f"Leaf {i}: shape={leaf.shape}, dtype={leaf.dtype}")

# Example
state = {'pos': positions, 'vel': velocities, 'mass': masses}
audit_dtypes(state)
```

---

:::{important} 💡 Key Takeaways

**Precision is a physics choice, not just a performance tuning**:

1. **Enable float64 for Project 5** (energy conservation requires it)
2. **Memory cost**: 2× larger arrays (usually acceptable for N ≤ 10,000)
3. **Speed cost**: ~10-20% slower (worth it for correct physics)
4. **Place config at top**: `jax.config.update("jax_enable_x64", True)`
5. **Check dtypes** if energy conservation tests fail

**Remember**: Fast but wrong is useless. Slow but correct is publishable.
:::

---

(common-errors)=

## 2.6: Common Errors and Debugging

**Priority: 🟡 Important**

### The Top 5 JAX Errors

#### 1. ConcretizationError

**Error message**:
```
ConcretizationError: Abstract tracer value encountered where concrete value expected
```

**What it means**: You're using a traced value in a context that needs a concrete value (e.g., Python `if`, array shape).

**Example**:
```python
@jax.jit
def bad_function(x):
    if x > 0:  # ❌ x is abstract during tracing!
        return x**2
    else:
        return -x**2
```

**Solution**: Use `lax.cond`:
```python
@jax.jit
def good_function(x):
    return lax.cond(x > 0, lambda x: x**2, lambda x: -x**2, x)
```

#### 2. Side Effects in JIT Functions

**Error**: Prints don't work, mutations ignored.

```python
@jax.jit
def bad_function(x):
    print(f"x = {x}")  # Prints during compilation, not execution!
    return x**2
```

**Solution**: Use `jax.debug.print` for inside JIT:
```python
@jax.jit
def good_function(x):
    jax.debug.print("x = {}", x)  # Prints on every execution
    return x**2
```

#### 3. vmap Batch Dimension Mismatch

**Error message**:
```
ValueError: vmap got inconsistent sizes for array axes
```

**What it means**: Arguments have incompatible batch dimensions.

**Example**:
```python
f_batched = jax.vmap(f, in_axes=(0, 0))
x = jnp.array([1, 2, 3])     # Shape: (3,)
y = jnp.array([1, 2, 3, 4])  # Shape: (4,) - mismatch!
result = f_batched(x, y)  # ❌ Error!
```

**Solution**: Ensure batch dimensions match or use `None` for non-batched:
```python
f_batched = jax.vmap(f, in_axes=(0, None))
y = jnp.array([1, 2, 3, 4])  # Same for all x
result = f_batched(x, y)  # ✅ Works
```

#### 4. NaN/Inf in Gradients

**Common cause**: Division by zero in physics calculations.

```python
# Gravitational potential
r = jnp.sqrt(jnp.sum(r_vec**2))  # Can be 0 if particles overlap!
U = -G * m1 * m2 / r  # ❌ Division by zero → NaN gradient
```

**Solution**: Add softening:
```python
r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # ✅ Never exactly 0
U = -G * m1 * m2 / r
```

**Enable NaN checking**:
```python
jax.config.update("jax_debug_nans", True)  # Catches NaN immediately
```

#### 5. Slow First Call (Compilation Overhead)

**Issue**: First call to JIT function is very slow.

**This is normal!** JIT compiles on first call. Subsequent calls are fast.

```python
# First call: slow (compilation + execution)
result = jitted_function(x)  # ~100ms

# Subsequent calls: fast (only execution)
result = jitted_function(x)  # ~1ms
```

**If recompiling every call**: Check if input shapes change.

---

### General Debugging Workflow

1. **Disable JIT temporarily**:
   ```python
   jax.config.update("jax_disable_jit", True)  # Get better error messages
   ```

2. **Add print statements**:
   ```python
   # Outside JIT: normal print (during trace)
   # Inside JIT: jax.debug.print (during execution)
   ```

3. **Check shapes**:
   ```python
   jax.debug.print("x.shape = {}", x.shape)
   ```

4. **Enable NaN checking**:
   ```python
   jax.config.update("jax_debug_nans", True)
   ```

5. **Create minimal reproduction**: Simplify to smallest failing example.

---

## Synthesis: What You've Learned

**JAX's three core transformations are unified by functional purity**:

| Transformation | What it does | Why purity matters | Astrophysics win |
|---------------|-------------|-------------------|------------------|
| **grad** | Computes derivatives via autodiff | Needs computational graph (requires pure functions) | HMC for cosmology (100× faster than finite diff) |
| **jit** | Compiles to machine code via XLA | Needs predictable trace (requires pure functions) | N-body forces (10-50× faster than NumPy) |
| **vmap** | Vectorizes over batch dimension | Needs safe parallelization (requires pure functions) | 43 supernova distances computed simultaneously |

**For your projects**:
- **Project 4 (MCMC)**: Use `jit(grad(log_posterior))` for HMC
- **Project 5 (N-body)**: Use `jit(vmap(grad(U)))` for forces
- **Project 6 (Neural Networks)**: Use `jit(vmap(grad(loss)))` for mini-batch training

**Connection to Module 1**: Statistical thinking unified diverse phenomena through moments and constraints. JAX unifies diverse computational patterns through functional purity and transformations.

---

## Understanding Checklist

Before proceeding to Part 3, ensure you can:

- [ ] **Explain** how reverse-mode autodiff computes gradients using the chain rule
- [ ] **Use** `jax.grad` with correct `argnums` to compute derivatives
- [ ] **Identify** when JIT compilation helps vs hurts performance
- [ ] **Debug** ConcretizationError by using `lax.cond` or `lax.scan`
- [ ] **Apply** `vmap` to vectorize functions over batch dimensions
- [ ] **Compose** transformations (jit ∘ vmap ∘ grad) in the correct order
- [ ] **Recognize** patterns that require `lax.scan` instead of Python loops

**If you answered "yes" to all** → Ready for Part 3: Applying JAX to Physics Simulations ✅

**Next**: Part 3 shows patterns for applying these tools to real astrophysics problems (N-body systems, ODE solving, optimization). Then you implement them yourself in Project 5.

---

:::{admonition} 📚 Additional Resources
:class: tip

**Official JAX Documentation**:
- [Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [JIT Compilation Guide](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
- [vmap Tutorial](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)

**Deep Dives**:
- Baydin et al. (2018), [Automatic Differentiation in ML: A Survey](https://arxiv.org/abs/1502.05767)
- [XLA: Optimizing Compiler for ML](https://www.tensorflow.org/xla)
:::
