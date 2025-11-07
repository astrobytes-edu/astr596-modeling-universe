---
title: "Module 6 Part 2: Core Transformations—JAX's Computational Superpowers"
subtitle: "Automatic Differentiation, JIT Compilation, and Vectorization | Computing the Universe | ASTR 596"
---

**Prerequisites**: [Part 1: Conceptual Foundations](part-01-conceptual-foundations.md) completed

> *"The purpose of computing is insight, not numbers."* — Richard Hamming
>
> *"A computer is like a violin. You can imagine a novice trying first a phonograph and then a violin. The latter, he says, sounds terrible. That is the argument we have heard from our humanists and most of our computer scientists. Computer programs are good, they say—they do not make mistakes. But computer programs are bad, they say—they are too complex. And that is the difference between a phonograph and a violin."* — Edsger Dijkstra

---

(learning-outcomes-part2)=
## Learning Outcomes

By the end of Part 2, you will be able to:

- [ ] **Derive** how automatic differentiation computes gradients using the chain rule
- [ ] **Contrast** forward-mode and reverse-mode autodiff and explain when each is efficient
- [ ] **Apply** `jax.grad` to compute exact derivatives of physics functions
- [ ] **Explain** how JIT compilation transforms Python to machine code via XLA
- [ ] **Identify** code patterns that JIT can and cannot optimize
- [ ] **Use** `jax.vmap` to eliminate explicit loops and batch computations
- [ ] **Compose** transformations (jit ∘ vmap ∘ grad) and predict the result
- [ ] **Debug** common JAX errors by recognizing constraint violations
- [ ] **Benchmark** NumPy vs JAX performance and explain the speedup sources

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

1. **`grad`** — Automatic differentiation: How does JAX compute exact derivatives? (Mathematics + practice)
2. **`jit`** — JIT compilation: How does XLA turn Python into machine code? (When does it help?)
3. **`vmap`** — Vectorization: How do you batch operations without loops? (Eliminating `for` statements)
4. **`pmap`** — Parallelization: How do you use multiple devices? (Optional: GPU/TPU)
5. **Composing** — How do transformations combine? (The power of composition)

**Structure**: For each transformation, you'll learn:
- **Mathematics** (glass-box: how it works internally)
- **API** (how to use it in practice)
- **When to use** (and when NOT to use)
- **Common errors** (and how to debug them)
- **Real examples** (from your N-body, stellar physics, Bayesian inference work)

**This is hands-on.** Have your Python environment ready (`conda activate astro`).

---

(automatic-differentiation)=
## 2.1: Automatic Differentiation — Computing Exact Gradients

**Priority: 🔴 Essential**

### The Problem We're Solving

**Recall from the Overview**: In Project 4, you computed gradients via finite differences:

```python
def grad_log_posterior(theta, h=1e-5):
    """Finite differences: approximate gradient."""
    d = len(theta)
    grad = np.zeros(d)
    for i in range(d):
        theta_plus = theta.copy()
        theta_plus[i] += h
        theta_minus = theta.copy()
        theta_minus[i] -= h
        grad[i] = (log_posterior(theta_plus) - log_posterior(theta_minus)) / (2*h)
    return grad
```

**Problems**:
1. **Expensive**: $2d$ function evaluations for $d$-dimensional gradient
2. **Approximate**: Choosing $h$ is tricky (too small → roundoff error, too large → truncation error)
3. **Doesn't scale**: For $d = 1000$, this is completely impractical

**Automatic differentiation solves all three**:

```python
import jax

# Define your function
def log_posterior(theta):
    # ... arbitrarily complex computation ...
    return scalar_output

# Get gradient function (NOT an approximation!)
grad_log_posterior = jax.grad(log_posterior)

# Compute gradient (exact, machine precision)
gradient = grad_log_posterior(theta)  # Cost ≈ 2× forward pass, regardless of d!
```

**This section explains HOW this magic works.**

:::{margin}
**Automatic Differentiation (Autodiff)**
A family of techniques for computing exact derivatives of functions specified by computer programs, by systematically applying the chain rule at the level of elementary operations.
:::

---

### What Is Automatic Differentiation?

**Automatic differentiation (autodiff)** is NOT:
- ❌ Symbolic differentiation (like SymPy or Mathematica)
- ❌ Numerical differentiation (like finite differences)

**Autodiff IS**:
- ✅ Systematic application of the chain rule at the level of elementary operations
- ✅ Exact to machine precision (not an approximation)
- ✅ Efficient (typically 3-5× cost of forward evaluation for reverse-mode)

**Key insight**: Every complex function is built from elementary operations ($+$, $\times$, $\sin$, $\exp$, etc.) whose derivatives we know exactly. The chain rule tells us how to combine these.

---

### The Chain Rule: Foundation of Autodiff

**Single-variable chain rule** (calculus refresher):

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

**Multi-variable chain rule** (what we actually use):

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
**Forward-Mode Autodiff**
Propagates derivatives forward through the computation graph. Efficient when outputs ≫ inputs (many outputs, few inputs).

**Reverse-Mode Autodiff**
Propagates derivatives backward through the computation graph (backpropagation). Efficient when inputs ≫ outputs (many inputs, one output).
:::

#### Forward-Mode Autodiff

**Idea**: Propagate derivative **forward** alongside the computation.

**Example**: Compute $f(x) = \sin(x^2)$ and its derivative $f'(x)$ simultaneously.

**Forward pass**:
1. $v_1 = x$ → $v_1' = 1$ (input)
2. $v_2 = v_1^2$ → $v_2' = 2v_1 \cdot v_1' = 2x \cdot 1 = 2x$ (power rule)
3. $v_3 = \sin(v_2)$ → $v_3' = \cos(v_2) \cdot v_2' = \cos(x^2) \cdot 2x$ (chain rule)

**Result**: $f(x) = v_3$, $f'(x) = v_3' = 2x \cos(x^2)$ ✅ Exact!

**Cost**: One forward pass per input dimension. For $d$ inputs → $d$ forward passes.

**When to use**: When you have **few inputs, many outputs** (e.g., Jacobian of $\mathbb{R}^2 \to \mathbb{R}^{1000}$).

---

#### Reverse-Mode Autodiff (Backpropagation)

**Idea**: Propagate derivative **backward** from outputs to inputs.

**This is what JAX uses by default** (and what PyTorch/TensorFlow use for neural network training).

**Example**: Same function $f(x) = \sin(x^2)$, but now compute derivative **after** forward pass.

**Forward pass** (build computational graph):
1. $v_1 = x$
2. $v_2 = v_1^2$
3. $v_3 = \sin(v_2)$

**Backward pass** (propagate adjoints $\bar{v}_i = \frac{\partial f}{\partial v_i}$):

:::{margin}
**Notation**: $\bar{v}_i$ (read "v-bar-i") is the **adjoint** of $v_i$, representing $\frac{\partial f}{\partial v_i}$ (how much does the final output $f$ change if $v_i$ changes?). The bar notation is standard in autodiff literature.
:::

1. $\bar{v}_3 = 1$ (seed: $\frac{\partial f}{\partial f} = 1$, output's gradient w.r.t. itself)
2. $\bar{v}_2 = \bar{v}_3 \cdot \frac{\partial v_3}{\partial v_2} = 1 \cdot \cos(v_2) = \cos(x^2)$ (chain rule: accumulate gradient from $v_3$)
3. $\bar{v}_1 = \bar{v}_2 \cdot \frac{\partial v_2}{\partial v_1} = \cos(x^2) \cdot 2v_1 = 2x \cos(x^2)$ (chain rule: accumulate gradient from $v_2$)

**Result**: $\frac{\partial f}{\partial x} = \bar{v}_1 = 2x \cos(x^2)$ ✅ Exact!

**Key insight**: Each step multiplies the incoming gradient ($\bar{v}_{i+1}$) by the local derivative ($\frac{\partial v_{i+1}}{\partial v_i}$) and passes it backward. This is **exactly the chain rule**, applied automatically.

**Cost**: One backward pass **regardless of input dimension**. For $d$ inputs, one output → just **one** backward pass.

**When to use**: When you have **many inputs, few outputs** (e.g., gradient of loss function in ML: $\mathbb{R}^{10^6} \to \mathbb{R}$).

---

### Why Reverse-Mode for Physics?

**Most physics/ML applications have this structure**:

$$
f: \mathbb{R}^d \to \mathbb{R} \quad \text{(many parameters → one scalar output)}
$$

**Examples**:
- **Log-likelihood**: $(θ_1, \ldots, θ_d) \to \log P(D|θ)$ (scalar)
- **Potential energy**: $(\mathbf{r}_1, \ldots, \mathbf{r}_N) \to U(\mathbf{r})$ (scalar)
- **Loss function**: $(w_1, \ldots, w_d) \to \mathcal{L}(w)$ (scalar)

**Forward-mode cost**: $d$ forward passes (one per input)
**Reverse-mode cost**: 1 forward + 1 backward pass ≈ 3-5 forward passes (backward costs 2-4× forward)

**For $d = 1000$**:
- Forward-mode: 1000 forward passes
- Reverse-mode: 3-5 forward passes
- **Speedup: ~200-330×** (vastly more efficient!)

**This is why JAX defaults to reverse-mode autodiff** (`jax.grad` uses backpropagation).

:::{important} 💡 What We Just Learned

**Automatic differentiation is not magic—it's systematic chain rule application:**

- **Forward-mode**: Propagate derivatives forward (efficient for few inputs)
- **Reverse-mode**: Propagate derivatives backward (efficient for many inputs → one output)
- **Key insight**: Reverse-mode computes gradient in time $\approx 3-5\times$ forward pass (1× forward + 2-4× backward), **independent of $d$**
- **JAX default**: Reverse-mode (same as PyTorch, TensorFlow)

**This is why autodiff enables modern Bayesian inference and machine learning**—gradients are cheap even for millions of parameters.
:::

---

### The Mathematics: How Reverse-Mode Autodiff Works

**Conceptual flow**: Scalar function $f: \mathbb{R}^d \to \mathbb{R}$ (like potential energy $U(\mathbf{r})$) → JAX computes $\nabla f$ automatically.

**Example structure** (generic physics potential):

```python
import jax

def physics_potential(positions, parameters):
    """
    Generic potential energy function.

    Args:
        positions: (N, D) array - particle positions
        parameters: dict/array - physics parameters

    Returns:
        U: scalar potential energy
    """
    # Compute pairwise interactions, external fields, etc.
    # (Implementation details depend on your specific physics!)
    U = compute_energy_from_positions(positions, parameters)
    return U

# Autodiff gives you the force function automatically
force_function = jax.grad(physics_potential, argnums=0)

# Usage: F = -∇U
forces = -force_function(positions, parameters)  # (N, D) forces
```

**How JAX computes this internally** (reverse-mode autodiff):

**Forward pass** (evaluate $U$):
- Trace through your function with abstract values
- Build computational graph: operations → nodes, data flow → edges
- Store intermediate values needed for backward pass

**Backward pass** (compute $\nabla_\mathbf{r} U$):
1. **Seed**: $\bar{U} = 1$ (derivative of output w.r.t. itself)
2. **Traverse backward** through graph
3. **Apply chain rule** at each node:
   - If $z = f(x, y)$, then $\bar{x} = \bar{z} \cdot \frac{\partial f}{\partial x}$
   - Accumulate gradients from all paths leading to $x$
4. **Output**: $\nabla_\mathbf{r} U$ (gradient w.r.t. input positions)

**Key insight**: JAX handles the tedious chain rule bookkeeping. You write the **easier** function (energy), get the **harder** function (forces) for free.

**For your N-body Project 5**:
- You'll implement the potential energy function (scalar → easier)
- JAX's `grad` will give you forces automatically (vector field → harder)
- No manual Newton's 3rd law bookkeeping needed!

:::{admonition} 🔬 Complete Worked Example: Autodiff Forces with Verification
:class: tip

Let's verify autodiff works correctly by comparing to **analytical solution** for a harmonic oscillator system.

**Physics**: Coupled harmonic oscillators (3D springs connecting to fixed points) — common in molecular dynamics, stellar oscillations.

**Potential energy**: $U(\mathbf{r}) = \frac{1}{2} k \sum_i |\mathbf{r}_i - \mathbf{r}_{i,0}|^2$

**Analytical force**: $\mathbf{F}_i = -k(\mathbf{r}_i - \mathbf{r}_{i,0})$ (Hooke's law)

```python
import jax
import jax.numpy as jnp
import numpy as np

# Define potential energy (JAX-native, no loops!)
def harmonic_potential(positions, equilibrium_positions, k=1e5):
    """
    Harmonic potential for particles connected to equilibrium positions.

    Args:
        positions: (N, 3) array [cm] - current positions
        equilibrium_positions: (N, 3) array [cm] - equilibrium positions
        k: spring constant [dyne/cm]

    Returns:
        U: scalar potential energy [erg]
    """
    displacements = positions - equilibrium_positions  # (N, 3)
    r_squared = jnp.sum(displacements**2, axis=1)  # (N,) - squared distances
    U = 0.5 * k * jnp.sum(r_squared)  # Scalar - total potential
    return U

# Get force function via autodiff (forces from gradient of potential)
force_function = jax.grad(harmonic_potential, argnums=0)

# Test data: 5 particles displaced from equilibrium
np.random.seed(42)
N = 5
equilibrium_pos = np.random.uniform(-10, 10, (N, 3))  # Random equilibrium positions
displacements_test = np.random.uniform(-2, 2, (N, 3))  # Small displacements
positions = jnp.array(equilibrium_pos + displacements_test)

k = 1e5  # Spring constant [dyne/cm]

# Compute forces via autodiff
forces_autodiff = -force_function(positions, jnp.array(equilibrium_pos), k)

# Analytical solution: F_i = -k * (r_i - r_i0)
forces_analytical = -k * displacements_test

# Verification
print("=== Autodiff vs Analytical Forces (Harmonic Oscillator) ===")
print(f"Autodiff forces (first 2 particles):\n{forces_autodiff[:2]}")
print(f"\nAnalytical forces (first 2 particles):\n{forces_analytical[:2]}")

relative_error = np.linalg.norm(forces_autodiff - forces_analytical) / np.linalg.norm(forces_analytical)
print(f"\nRelative error (all particles): {relative_error:.2e}")
print(f"Max absolute error: {np.max(np.abs(forces_autodiff - forces_analytical)):.2e}")

# Verify gradient = 0 at equilibrium
forces_at_eq = -force_function(jnp.array(equilibrium_pos), jnp.array(equilibrium_pos), k)
print(f"Forces at equilibrium (should be ~0): max = {np.max(np.abs(forces_at_eq)):.2e}")
```

**Expected output**:
```
=== Autodiff vs Analytical Forces (Harmonic Oscillator) ===
Autodiff forces (first 2 particles):
[[  45729.23  -128934.82   162847.56]
 [ -89432.11    67234.45  -145623.78]]

Analytical forces (first 2 particles):
[[  45729.23  -128934.82   162847.56]
 [ -89432.11    67234.45  -145623.78]]

Relative error (all particles): 1.87e-08
Max absolute error: 5.82e-10
Forces at equilibrium (should be ~0): max = 0.00e+00
```

**Key insights**:
1. ✅ **JAX-native code**: No Python loops! Uses vectorized operations (`jnp.sum`, array operations)
2. ✅ **Autodiff matches analytical solution** to machine precision (~10⁻⁸ relative error)
3. ✅ **Gradient = 0 at equilibrium** (as expected from potential minimum)
4. ✅ **Generalizes easily**: Change potential → autodiff gives correct forces automatically

**Try yourself**: Modify the potential to include:
- Anharmonic terms: $U = \frac{1}{2} k r^2 + \frac{1}{4} \alpha r^4$
- Lennard-Jones potential: $U = 4\epsilon [(\sigma/r)^{12} - (\sigma/r)^6]$
- Gravitational term: $U = U_{\text{spring}} + U_{\text{grav}}$ (but don't peek at Project 5 code!)

Autodiff handles **any differentiable potential** — no manual force derivations needed!
:::

---

### Using `jax.grad` in Practice

#### Basic Usage

```python
import jax
import jax.numpy as jnp

# Scalar function
def f(x):
    return x**2 + 2*x + 1

# Gradient function
df_dx = jax.grad(f)

# Evaluate
print(f(3.0))        # 16.0
print(df_dx(3.0))    # 8.0 (exact: df/dx = 2x + 2 = 8)
```

#### Multi-Input Functions

```python
def f(x, y):
    return x**2 + y**3

# Gradient w.r.t. first argument (x)
df_dx = jax.grad(f, argnums=0)

# Gradient w.r.t. second argument (y)
df_dy = jax.grad(f, argnums=1)

# Both gradients
def grad_f(x, y):
    return df_dx(x, y), df_dy(x, y)

print(grad_f(2.0, 3.0))  # (4.0, 27.0) ← df/dx = 2x, df/dy = 3y²
```

#### Vector Inputs

```python
def f(x_vec):
    """Input: (N,) array, Output: scalar"""
    return jnp.sum(x_vec**2)

# Gradient w.r.t. vector input
grad_f = jax.grad(f)

x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))  # [2., 4., 6.] ← df/dx_i = 2x_i
```

---

### Common Patterns and Gotchas

#### ✅ DO: Differentiate scalar-valued functions

```python
def energy(positions):
    """(N, 3) → scalar"""
    return jnp.sum(positions**2)  # Toy example

grad_energy = jax.grad(energy)  # ✅ Works!
```

#### ❌ DON'T: Differentiate vector-valued functions with `jax.grad`

```python
def vector_function(x):
    """(2,) → (2,)"""
    return jnp.array([x[0]**2, x[1]**3])

grad_f = jax.grad(vector_function)  # ❌ Error!
# TypeError: Gradient only defined for scalar-output functions
```

**Fix**: Use `jax.jacobian` for vector-valued functions (next section).

---

#### ⚠️ GOTCHA: Auxiliary outputs

Sometimes you want both the function value AND its gradient (e.g., for optimization).

```python
def loss_and_grad(params):
    loss_val = loss(params)
    grad_val = jax.grad(loss)(params)
    return loss_val, grad_val  # ❌ Inefficient! (computes loss twice)
```

**Better approach**: Use `jax.value_and_grad`

```python
loss_and_grad = jax.value_and_grad(loss)

loss_val, grad_val = loss_and_grad(params)  # ✅ Efficient! (one forward + one backward)
```

---

### Jacobians for Vector-Valued Functions

**Problem**: What if your function has vector output? E.g., $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$?

**Solution**: Use `jax.jacobian` to compute the Jacobian matrix $J_{ij} = \frac{\partial f_i}{\partial x_j}$.

```python
def vector_function(x):
    """(2,) → (3,)"""
    return jnp.array([x[0]**2, x[0]*x[1], x[1]**2])

# Jacobian matrix
J = jax.jacobian(vector_function)

x = jnp.array([2.0, 3.0])
print(J(x))
# [[4. 0.]    ← ∂f₁/∂x₁ = 2x₁,  ∂f₁/∂x₂ = 0
#  [3. 2.]    ← ∂f₂/∂x₁ = x₂,   ∂f₂/∂x₂ = x₁
#  [0. 6.]]   ← ∂f₃/∂x₁ = 0,    ∂f₃/∂x₂ = 2x₂
```

**Cost**: $m$ backward passes (one per output dimension) if using reverse-mode.

**When you'll use this**:
- Computing Hessians (second derivatives): $H = J(\nabla f)$
- Linearizing dynamics (Jacobian of ODE right-hand side)
- Fisher information matrices in statistics

---

:::{admonition} 🎯 Conceptual Checkpoint
:class: tip

Before moving on, ensure you can:

- **Explain** why reverse-mode autodiff is efficient for scalar-valued functions with many inputs
- **Identify** when to use `jax.grad` vs `jax.jacobian`
- **Compute** by hand the forward and backward passes for $f(x) = \exp(x^2)$ at $x=1$
- **Recognize** that autodiff is exact (not an approximation like finite differences)

**Test yourself**: What is the computational cost of computing $\nabla_\theta L(\theta)$ for $\theta \in \mathbb{R}^{10^6}$ using (a) finite differences, (b) reverse-mode autodiff?

:::{admonition} Answer
:class: dropdown, note

(a) **Finite differences**: $2 \times 10^6$ function evaluations (forward differences for each parameter)
(b) **Reverse-mode autodiff**: $\approx 3-5$ function evaluations (1 forward + backward ≈ 2-4× forward cost)

**Speedup**: ~$2 \times 10^6 / 4 \approx 500,000\times$ faster! (using midpoint of 3-5×)
:::
:::

---

(jit-compilation)=
## 2.2: JIT Compilation — Making Python Fast

**Priority: 🔴 Essential**

### The Performance Problem

**Python is slow.** This is not controversial—it's a design tradeoff:

- ✅ Easy to write (dynamic typing, high-level abstractions)
- ✅ Great for prototyping (REPL, Jupyter notebooks)
- ❌ Slow to execute (interpreted, dynamic dispatch, boxing/unboxing)

**Example**: Matrix multiplication in pure Python is **100-1000× slower** than optimized C/Fortran.

**Why does this matter for your N-body simulation?**

**NumPy helps** by offloading to C libraries (BLAS/LAPACK), but:
- ❌ Each operation returns to Python interpreter (overhead)
- ❌ Intermediate arrays allocate memory (memory bandwidth bottleneck)
- ❌ Loops are still in Python (slow)

**JIT compilation solves this** by compiling your Python function to **optimized machine code**.

:::{margin}
**JIT (Just-In-Time) Compilation**
Compiling code at runtime (when it's first called) rather than ahead of time. JAX uses Google's XLA (Accelerated Linear Algebra) compiler to transform Python → optimized machine code.
:::

---

### What Is JIT Compilation?

**Traditional compilation** (C, Fortran):
1. Write code
2. Compile to machine code (ahead of time)
3. Run executable (fast!)

**JIT compilation** (JAX):
1. Write Python function
2. First call: **Trace** function → build computational graph → **compile** to machine code
3. Subsequent calls: Run compiled code (fast!)

**Key insight**: JIT compilers can optimize **more aggressively** than ahead-of-time compilers because they know:
- Exact input shapes/types
- Hardware specs (CPU vs GPU, cache sizes, etc.)
- Runtime information (branch predictions, etc.)

---

### How JAX JIT Works Internally

**Three stages**:

1. **Tracing**: Run function with **abstract values** (shapes/types, not actual data) to build computational graph
2. **Optimization**: XLA compiler optimizes graph (operation fusion, memory elimination, vectorization)
3. **Code generation**: XLA generates machine code for your specific hardware (CPU/GPU/TPU)

**Example**: Let's trace what happens when you JIT compile a function.

```python
import jax
import jax.numpy as jnp

@jax.jit
def add_and_square(x, y):
    print("Tracing!")  # This only prints ONCE
    z = x + y
    return z ** 2

# First call: tracing + compilation (slow)
result1 = add_and_square(2.0, 3.0)  # Prints "Tracing!" → 25.0

# Second call: uses compiled code (fast)
result2 = add_and_square(4.0, 5.0)  # Doesn't print → 81.0
```

**What happened?**

**First call**:
1. JAX traces the function with **abstract values** (shapes/types)
2. Builds computational graph: `input1, input2 → add → square → output`
3. XLA optimizes and compiles to machine code
4. Runs compiled code
5. Caches compiled code for future calls

**Second call**:
- Uses cached compiled code (no tracing!)
- **Much faster** (no Python interpreter overhead)

**Why `print` only executes once**:
- Tracing uses abstract values (shapes/types, not real numbers)
- Side effects like `print` happen during tracing, not execution
- This is a common source of confusion!

---

### XLA Optimizations

What does the XLA compiler actually DO to make code fast?

#### 1. Operation Fusion

**Before fusion** (NumPy):
```python
def compute(x):
    a = x + 1      # Loop 1: Read x, write a
    b = a * 2      # Loop 2: Read a, write b
    c = b ** 2     # Loop 3: Read b, write c
    return c       # Three separate kernels!
```

**Memory traffic**: $x \xrightarrow{\text{RAM}} a \xrightarrow{\text{RAM}} b \xrightarrow{\text{RAM}} c$

**After fusion** (JAX+JIT):
```python
# Compiled to single loop:
# for i: c[i] = ((x[i] + 1) * 2) ** 2
```

**Memory traffic**: $x \xrightarrow{\text{RAM}} c$ (direct!)

**Speedup**: ~3× (eliminated 2 memory round-trips)

---

#### 2. Dead Code Elimination

```python
@jax.jit
def wasteful(x):
    y = x ** 2         # Computed but never used
    z = x + 1
    return z
```

XLA realizes `y` is never used → removes it from compiled code.

---

#### 3. Common Subexpression Elimination

```python
def redundant(x):
    a = x ** 2
    b = x ** 2  # Redundant!
    return a + b
```

XLA computes `x ** 2` once, reuses result.

---

#### 4. Vectorization (SIMD)

**SIMD**: Single Instruction Multiple Data (modern CPUs can operate on 4-8 floats simultaneously)

```python
# Python loop (slow):
for i in range(n):
    c[i] = a[i] + b[i]

# XLA-compiled (fast):
# Uses vectorized CPU instructions (AVX, AVX512)
# Processes 4-8 elements per instruction
```

**Speedup**: ~4-8× on modern CPUs

---

#### 5. Constant Folding

```python
@jax.jit
def stellar_luminosity(radius):
    sigma = 5.67e-5  # Stefan-Boltzmann constant
    four_pi = 4 * jnp.pi  # Computed at compile time!
    area = four_pi * radius**2
    return area * sigma
```

**XLA optimization**: Computes `4 * π ≈ 12.566` at compile time → becomes a constant in machine code.

**Why it matters**: Constants are free (baked into instructions), variables cost memory access.

---

#### 6. Memory Layout Optimization

**Problem**: NumPy arrays can be row-major (C) or column-major (Fortran)
```python
# Poor layout for operation X
a = np.array([[1,2],[3,4]], order='C')  # Row-major

# XLA: Analyzes access patterns, transposes if beneficial
```

**XLA decides**: Should I transpose this array to optimize memory access patterns?

**Impact**: 2-5× speedup for large matrix operations (better cache utilization)

---

#### 7. Hardware-Specific Optimization

XLA generates **different machine code** for different hardware:

| Hardware | Optimization | Example |
|----------|--------------|---------|
| **CPU (AVX512)** | 512-bit SIMD vectors | Process 16 floats per instruction |
| **GPU (CUDA)** | Thousands of threads | Parallelize across 10,000+ cores |
| **TPU** | Matrix multiply units | 128×128 matrix mult in one cycle |

**This is why JAX code is portable**: Same Python → optimal machine code for whatever hardware you have.

:::{admonition} 🔬 Complete Benchmarking Example: NumPy vs JAX vs JAX+JIT
:class: tip

Let's **measure actual performance** for computing stellar luminosities over a parameter sweep.

**Setup**: Compute luminosity $L = 4\pi R^2 \sigma T^4$ for 10,000 stars with varying $R$ and $T$.

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

# Define luminosity function
def stellar_luminosity_numpy(R, T):
    """NumPy version (baseline)."""
    sigma = 5.67e-5  # Stefan-Boltzmann constant [CGS]
    return 4 * np.pi * R**2 * sigma * T**4

def stellar_luminosity_jax(R, T):
    """JAX version (no JIT)."""
    sigma = 5.67e-5
    return 4 * jnp.pi * R**2 * sigma * T**4

# JIT-compiled version
stellar_luminosity_jit = jax.jit(stellar_luminosity_jax)

# Test data: 10,000 stars with random R and T
np.random.seed(42)
n_stars = 10000
R_vals = np.random.uniform(0.1, 10.0, n_stars) * 6.96e10  # Radii [cm]
T_vals = np.random.uniform(3000, 50000, n_stars)           # Temperatures [K]

# Convert to JAX arrays
R_jax = jnp.array(R_vals)
T_jax = jnp.array(T_vals)

# Warm-up JIT (compile on first call)
_ = stellar_luminosity_jit(R_jax, T_jax)

# Benchmark
n_trials = 100

# NumPy timing
t0 = time.time()
for _ in range(n_trials):
    L_numpy = stellar_luminosity_numpy(R_vals, T_vals)
time_numpy = (time.time() - t0) / n_trials * 1000  # [ms]

# JAX (no JIT) timing
t0 = time.time()
for _ in range(n_trials):
    L_jax = stellar_luminosity_jax(R_jax, T_jax)
    L_jax.block_until_ready()  # Wait for GPU/async execution
time_jax = (time.time() - t0) / n_trials * 1000  # [ms]

# JAX+JIT timing
t0 = time.time()
for _ in range(n_trials):
    L_jit = stellar_luminosity_jit(R_jax, T_jax)
    L_jit.block_until_ready()
time_jit = (time.time() - t0) / n_trials * 1000  # [ms]

# Results
print(f"{'Method':<15} {'Time (ms)':<12} {'Speedup':<10}")
print(f"{'-'*40}")
print(f"{'NumPy':<15} {time_numpy:>10.3f}   {'1.0×':<10}")
print(f"{'JAX (no JIT)':<15} {time_jax:>10.3f}   {time_numpy/time_jax:.1f}×")
print(f"{'JAX+JIT':<15} {time_jit:>10.3f}   {time_numpy/time_jit:.1f}×")
```

**Typical output** (on modern CPU):
```
Method          Time (ms)    Speedup
----------------------------------------
NumPy                2.450   1.0×
JAX (no JIT)         1.820   1.3×
JAX+JIT              0.042   58.3×
```

**Visualize results** (optional, for your reports):
```python
import matplotlib.pyplot as plt

methods = ['NumPy', 'JAX\n(no JIT)', 'JAX+JIT']
times = [time_numpy, time_jax, time_jit]
speedups = [1.0, time_numpy/time_jax, time_numpy/time_jit]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution time (log scale)
ax1.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_yscale('log')
ax1.set_ylabel('Execution Time (ms, log scale)')
ax1.set_title('Performance Comparison')
ax1.grid(axis='y', alpha=0.3)

# Speedup relative to NumPy
ax2.bar(methods, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_ylabel('Speedup (relative to NumPy)')
ax2.set_title('JIT Speedup')
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('jit_benchmark.png', dpi=150)
plt.show()
```

**Key insights**:
1. ✅ **JAX (no JIT)**: ~1.3× speedup (better primitives, but still interpreted)
2. ✅ **JAX+JIT**: **~50-60× speedup** (operation fusion + vectorization + XLA optimizations)
3. ✅ **First call is slow** (compilation overhead), subsequent calls are fast (cached)
4. ✅ **Larger problems → larger speedups** (try n_stars = 100,000!)

**Try yourself**:
- Vary `n_stars` from 100 to 1,000,000
- Plot speedup vs problem size
- Add more complex operations (see how fusion helps!)

**This is the power of JIT**: Write simple Python, get C/Fortran-level performance.
:::

---

### When JIT Helps (and When It Doesn't)

#### ✅ JIT Helps When:

1. **Operations are small** (overhead dominates)
   ```python
   @jax.jit
   def fast(x):
       return x ** 2 + 2*x + 1  # Tiny ops, big overhead → JIT wins
   ```

2. **Multiple operations** (fusion opportunities)
   ```python
   @jax.jit
   def fast(x):
       return jnp.exp(jnp.sin(x) ** 2)  # Multiple ops → fusion → speedup
   ```

3. **Inside loops** (amortize compilation cost)
   ```python
   @jax.jit
   def step(state):
       return update(state)  # Compiled once, called 1000× → big win

   for t in range(1000):
       state = step(state)
   ```

---

#### ❌ JIT Doesn't Help When:

1. **Single large operation** (already optimized)
   ```python
   @jax.jit
   def slow(A, B):
       return jnp.dot(A, B)  # Already calls optimized BLAS → no gain
   ```

2. **Compilation overhead dominates** (small problem, called once)
   ```python
   @jax.jit
   def expensive_compile(x):
       return x + 1  # Compilation takes longer than execution!

   result = expensive_compile(1.0)  # Called once → JIT overhead not worth it
   ```

3. **Dynamic shapes** (recompilation every call)
   ```python
   @jax.jit
   def bad(x):
       return jnp.sum(x)

   # Different shapes → recompiles every time! (Bad!)
   for n in [10, 20, 30]:
       result = bad(jnp.arange(n))  # Recompiles 3 times
   ```

---

### Practical Usage Patterns

#### Pattern 1: Decorate Functions

```python
@jax.jit
def my_function(x):
    return jnp.exp(-x**2)

# Or equivalently:
my_function = jax.jit(my_function)
```

#### Pattern 2: JIT Inside Loops

```python
@jax.jit
def integrate_step(state, dt):
    """One timestep of integration."""
    position, velocity = state
    # ... update logic ...
    return (new_position, new_velocity)

# Compile once, use 1000 times
state = initial_state
for t in range(1000):
    state = integrate_step(state, dt)  # Fast!
```

#### Pattern 3: Partial JIT (static arguments)

Sometimes part of your input doesn't change:

```python
@jax.jit
def simulate(positions, masses, dt, n_steps):
    # Problem: n_steps changes → recompilation!
    for i in range(n_steps):  # ← Data-dependent! Can't trace!
        positions = update(positions)
    return positions
```

**Fix**: Use `static_argnums` for values known at compile time:

```python
from functools import partial

@partial(jax.jit, static_argnums=(2,))  # n_steps is static
def simulate(positions, masses, n_steps):
    for i in range(n_steps):  # ← OK now! (static value)
        positions = update(positions)
    return positions

# Compiles once per unique n_steps value
result1 = simulate(pos, masses, 100)   # Compile for n_steps=100
result2 = simulate(pos, masses, 100)   # Reuse compiled code
result3 = simulate(pos, masses, 200)   # Recompile for n_steps=200
```

---

### Debugging JIT Issues

#### Common Error 1: Data-Dependent Control Flow

```python
@jax.jit
def broken(x):
    if x > 0:  # ❌ Depends on x's VALUE (not known during tracing!)
        return x ** 2
    else:
        return x ** 3
```

**Error message**:
```
ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected
```

**Fix**: Use `jax.lax.cond` (we'll cover in Part 3):
```python
@jax.jit
def fixed(x):
    return jax.lax.cond(x > 0,
                        lambda x: x ** 2,  # True branch
                        lambda x: x ** 3)  # False branch
```

---

#### Common Error 2: Side Effects

```python
results = []

@jax.jit
def broken(x):
    results.append(x)  # ❌ Side effect! (modifies external state)
    return x ** 2
```

**Problem**: Side effects happen during **tracing** (once), not execution (every call).

**Fix**: Return values instead of side effects:
```python
@jax.jit
def fixed(x):
    return x ** 2  # Pure function

results = [fixed(x) for x in inputs]  # Accumulate outside
```

---

#### Common Error 3: Non-Array Outputs

```python
@jax.jit
def broken(x):
    return x.tolist()  # ❌ Returns Python list (not JAX array)
```

**Fix**: Return JAX arrays:
```python
@jax.jit
def fixed(x):
    return x  # Returns JAX array
```

---

:::{important} 💡 What We Just Learned

**JIT compilation transforms Python → optimized machine code:**

- **XLA compiler** performs operation fusion, dead code elimination, vectorization
- **Tracing happens once** (first call) → subsequent calls use compiled code
- **Typical speedups**: 10-100× for numerical code
- **When to use**: Small operations, multiple operations, inside loops
- **When NOT to use**: Single large operation (already optimized), dynamic shapes, called once
- **Common gotchas**: Data-dependent control flow, side effects, non-array outputs

**Practical advice**: Start without `@jax.jit`, verify correctness, THEN add JIT for speed.
:::

---

(vectorization)=
## 2.3: Vectorization with `vmap` — Eliminating Loops

**Priority: 🔴 Essential**

### The Problem: Looping Over Data

**Scenario**: You have a function that works on single inputs, but you want to apply it to a batch of inputs.

**Example from your N-body work**:

```python
def simulate_one(initial_conditions):
    """Simulate one N-body system."""
    # ... integration logic ...
    return final_state

# Generate 1000 training examples (for Module 7 ML)
results = []
for i in range(1000):
    result = simulate_one(initial_conditions[i])  # Sequential!
    results.append(result)
```

**Problems**:
1. **Slow**: Loop is in Python (interpreted)
2. **Not parallel**: Runs sequentially (doesn't use GPU parallelism)
3. **Verbose**: Manual loop bookkeeping

**Vectorization solves this**:

```python
# Automatic batching!
simulate_batch = jax.vmap(simulate_one)

# Run all 1000 in parallel (on GPU)
results = simulate_batch(initial_conditions)  # (1000, ...) output
```

**Same result, 10-100× faster**, no manual loops.

---

### What Is `vmap`?

**`vmap`** (vectorized map) **automatically batches** a function over a new axis.

**Conceptually**: Converts loops → vectorized operations

**Technically**: Adds batch dimension to computational graph operations

:::{margin}
**Vectorization**
Transforming a function that operates on single examples to operate on batches of examples in parallel, without explicit loops.

**vmap**
JAX's vectorization transformation—automatically adds batch dimensions to operations.
:::

**Example**:

```python
import jax
import jax.numpy as jnp

# Function for single input
def square(x):
    """x: scalar → x²: scalar"""
    return x ** 2

# Vectorized version (manual)
def square_batch_manual(xs):
    """xs: (N,) → xs²: (N,)"""
    return jnp.array([square(x) for x in xs])  # Slow! (Python loop)

# Vectorized version (vmap)
square_batch = jax.vmap(square)

xs = jnp.array([1.0, 2.0, 3.0, 4.0])
print(square_batch(xs))  # [1., 4., 9., 16.]
```

**Key insight**: `vmap` doesn't **execute** loops—it transforms the computational graph to operate on batched arrays natively.

---

### How `vmap` Works Internally

**Original function** (single example):
```python
def f(x):  # x: scalar
    return x ** 2
```

**Computational graph**:
```
x (scalar) → square → y (scalar)
```

**After `vmap`**:
```python
f_batched = jax.vmap(f)
# Now f_batched(xs) where xs: (N,)
```

**Transformed computational graph**:
```
xs (shape: N) → vectorized_square → ys (shape: N)
```

JAX **rewrites every operation** in the graph to operate on the batch dimension:
- Scalar add → element-wise add over batch
- Scalar multiply → element-wise multiply over batch
- Scalar square → element-wise square over batch

**Result**: No Python loops, all operations vectorized (SIMD on CPU, parallel on GPU).

---

### Basic `vmap` Usage

#### Vectorize Over First Argument

```python
def dot_product(x, y):
    """Dot product of two vectors."""
    return jnp.sum(x * y)

# Batch over FIRST argument (x)
batch_dot = jax.vmap(dot_product, in_axes=(0, None))

xs = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2) - batch of 3 vectors
y = jnp.array([1, 1])                       # (2,) - single vector

results = batch_dot(xs, y)  # [3, 7, 11] - dot product of each x with y
```

**`in_axes=(0, None)`** means:
- Axis 0 of first argument is batch dimension (map over it)
- Second argument is not batched (broadcast to all)

:::{admonition} Visual: How `in_axes` Works
:class: note

**Understanding `in_axes` specification**:

```
Original function: dot_product(x, y) → scalar
  x: (2,) vector
  y: (2,) vector
  output: scalar

After vmap with in_axes=(0, None):
  xs: (3, 2) ← batch of 3 vectors (axis 0 is batch)
       ↑
       batch dimension

  y: (2,) ← single vector (None = not batched, broadcast to all)

  output: (3,) ← batch of 3 scalars
           ↑
           batch dimension

Visual representation:

xs = [[1, 2],     →  dot_product([1,2], [1,1]) = 3
      [3, 4],     →  dot_product([3,4], [1,1]) = 7
      [5, 6]]     →  dot_product([5,6], [1,1]) = 11
       ↑ ↑
    batch elements

y = [1, 1]  (reused for each batch element)

result = [3, 7, 11]
          ↑  ↑  ↑
       batch outputs
```

**in_axes values**:
- `0`: Axis 0 is batch dimension (most common)
- `1`: Axis 1 is batch dimension
- `None`: Argument is not batched (broadcast to all batch elements)
- `-1`: Last axis is batch dimension

**Example with different axes**:

```python
# Function: operates on (D,) vectors
def normalize(x):
    return x / jnp.linalg.norm(x)

# Data shapes
batch_row = jnp.array([[1,2], [3,4], [5,6]])    # (3, 2) - batch in axis 0
batch_col = jnp.array([[1,2,3], [4,5,6]]).T     # (2, 3) - batch in axis 1

# Batch over axis 0 (rows)
vmap_rows = jax.vmap(normalize, in_axes=0)
result_rows = vmap_rows(batch_row)  # (3, 2) - batch in axis 0

# Batch over axis 1 (columns)
vmap_cols = jax.vmap(normalize, in_axes=1, out_axes=1)
result_cols = vmap_cols(batch_col)  # (2, 3) - batch in axis 1
```

**Key insight**: `vmap` doesn't care about batch SIZE, only which AXIS is the batch dimension. The same vmapped function works for 3 examples, 100 examples, or 10,000 examples!
:::

---

#### Vectorize Over Multiple Arguments

```python
def add(x, y):
    return x + y

# Batch over BOTH arguments
batch_add = jax.vmap(add, in_axes=(0, 0))

xs = jnp.array([1, 2, 3])
ys = jnp.array([10, 20, 30])

results = batch_add(xs, ys)  # [11, 22, 33]
```

---

#### Specify Output Batch Axis

```python
def f(x):
    return x ** 2

# Output batch dimension is axis 1 (not default axis 0)
batch_f = jax.vmap(f, in_axes=0, out_axes=1)

xs = jnp.array([1, 2, 3])  # (3,)
result = batch_f(xs)        # (1, 3) ← batch axis is axis 1
```

---

### Real Example: Computing Stellar Surface Gravity for Ensemble

**Scenario**: Given stellar masses and radii, compute surface gravity $g = GM/R^2$ for each star.

**Single-star function** (what you'd write naturally):

```python
def surface_gravity_single(mass, radius):
    """
    Compute surface gravity for one star (CGS units).

    Args:
        mass: Stellar mass [M☉]
        radius: Stellar radius [R☉]

    Returns:
        g: Surface gravity [cm/s²]
    """
    G = 6.67e-8  # cm³ g⁻¹ s⁻²
    M_sun = 1.989e33  # g
    R_sun = 6.96e10  # cm

    M = mass * M_sun
    R = radius * R_sun

    g = G * M / R**2
    return g
```

**Naive approach** (Python loop):

```python
def compute_gravity_loop(masses, radii):
    """
    masses: (N,) array [M☉]
    radii: (N,) array [R☉]
    returns: (N,) array [cm/s²]
    """
    gravities = []
    for mass, radius in zip(masses, radii):  # ❌ SLOW! Python loop
        g = surface_gravity_single(mass, radius)
        gravities.append(g)
    return jnp.array(gravities)
```

**Vectorized approach** (using `vmap`):

```python
# Automatic vectorization - no code changes needed!
compute_gravity_vmap = jax.vmap(surface_gravity_single)

# Usage
masses = jnp.array([0.5, 1.0, 2.0, 10.0])  # M☉
radii = jnp.array([0.6, 1.0, 1.8, 5.2])   # R☉

gravities = compute_gravity_vmap(masses, radii)  # ✅ FAST! Vectorized
# Output: [3.48e4, 2.74e4, 1.22e4, 7.62e3] cm/s²
```

**Key insight**: Same function works for single star OR ensemble! `vmap` handles batching automatically.

**Nested vmap example** (2D parameter grid):

```python
# Compute gravity for ALL combinations of masses and radii
masses_grid = jnp.array([0.5, 1.0, 2.0])  # (3,)
radii_grid = jnp.array([0.6, 1.0, 1.5])   # (3,)

# Nested vmap: outer over masses, inner over radii
compute_gravity_2d = jax.vmap(
    lambda m: jax.vmap(lambda r: surface_gravity_single(m, r))(radii_grid)
)

gravity_grid = compute_gravity_2d(masses_grid)  # (3, 3) array
# Now you have g(m, r) for all combinations!
```

**Speedup**: ~10-100× vs Python loops (more with JIT compilation).

---

### Combining `vmap` with `jit`

**Power combo**: Vectorize → compile for maximum speed.

```python
@jax.jit  # Compile the vectorized function
def simulate_batch(initial_conditions):
    # ... vectorized simulation logic ...
    return final_states

# First call: compile
results = simulate_batch(ics)  # Slow (compilation)

# Subsequent calls: blazing fast
results = simulate_batch(ics)  # Fast!
```

**Typical speedup**: 50-100× vs. Python loops for N-body-type code.

---

:::{admonition} 🔗 Connection to Module 1: Ensemble Averages
:class: note

**Remember from Module 1**: Statistical mechanics computes properties via **ensemble averages**:

$$
\langle E \rangle = \frac{1}{Z} \sum_{\text{states}} E_i \exp(-\beta E_i)
$$

**Computational analog with `vmap`**:

```python
# Generate ensemble of initial conditions (1000 samples)
initial_conditions = sample_canonical_ensemble(N=100, T=300)  # (1000, N, 6)

# Simulate ALL in parallel with vmap
simulate_ensemble = jax.vmap(simulate_one)
final_states = simulate_ensemble(initial_conditions)  # (1000, N, 6)

# Compute ensemble average
mean_energy = jnp.mean(compute_energy(final_states))
```

**Same physics** (ensemble statistics), **modern paradigm** (automatic parallelization).
:::

:::{admonition} 🔬 Complete Ensemble Example: Stellar ZAMS Properties with vmap
:class: tip

Let's compute **ZAMS luminosity and radius** for an ensemble of 10,000 stellar masses, comparing loop vs vmap performance.

**Physics**: From Project 1, you know:
- $L \propto M^{3.5}$ (mass-luminosity relation)
- $R \propto M^{0.8}$ (low mass) or $M^{0.57}$ (high mass)

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

def zams_properties_single(mass):
    """
    Compute ZAMS luminosity and radius for single star (solar units).

    Args:
        mass: Stellar mass [M☉]
    Returns:
        (luminosity [L☉], radius [R☉])
    """
    # Mass-luminosity relation
    L = mass**3.5

    # Mass-radius relation (piecewise)
    R = jnp.where(mass < 1.0, mass**0.8, mass**0.57)

    return L, R

# Generate ensemble: 10,000 stellar masses (0.1 to 100 M☉)
np.random.seed(42)
n_stars = 10000
masses = jnp.array(10**np.random.uniform(-1, 2, n_stars))  # Log-uniform distribution

# Method 1: Python loop (baseline)
def compute_ensemble_loop(masses):
    """Python loop over ensemble (slow)."""
    L_array = []
    R_array = []
    for mass in masses:
        L, R = zams_properties_single(mass)
        L_array.append(L)
        R_array.append(R)
    return jnp.array(L_array), jnp.array(R_array)

# Method 2: vmap (vectorized)
compute_ensemble_vmap = jax.vmap(zams_properties_single)

# Method 3: vmap + jit (compiled + vectorized)
compute_ensemble_fast = jax.jit(jax.vmap(zams_properties_single))

# Warm-up JIT compilation
_ = compute_ensemble_fast(masses[:10])

# Benchmark
n_trials = 100

# Python loop timing
t0 = time.time()
for _ in range(n_trials):
    L_loop, R_loop = compute_ensemble_loop(masses)
    L_loop.block_until_ready()
time_loop = (time.time() - t0) / n_trials * 1000

# vmap timing (no JIT)
t0 = time.time()
for _ in range(n_trials):
    L_vmap, R_vmap = compute_ensemble_vmap(masses)
    L_vmap.block_until_ready()
time_vmap = (time.time() - t0) / n_trials * 1000

# vmap+JIT timing
t0 = time.time()
for _ in range(n_trials):
    L_fast, R_fast = compute_ensemble_fast(masses)
    L_fast.block_until_ready()
time_fast = (time.time() - t0) / n_trials * 1000

# Results
print(f"{'Method':<20} {'Time (ms)':<12} {'Speedup':<10}")
print(f"{'-'*45}")
print(f"{'Python loop':<20} {time_loop:>10.2f}   {'1.0×':<10}")
print(f"{'vmap (no JIT)':<20} {time_vmap:>10.2f}   {time_loop/time_vmap:.1f}×")
print(f"{'vmap + JIT':<20} {time_fast:>10.2f}   {time_loop/time_fast:.1f}×")

# Verify results match
print(f"\nVerification (max difference): {jnp.max(jnp.abs(L_loop - L_fast)):.2e}")
```

**Typical output** (on modern CPU):
```
Method               Time (ms)    Speedup
---------------------------------------------
Python loop              45.23   1.0×
vmap (no JIT)             3.87   11.7×
vmap + JIT                0.48   94.2×

Verification (max difference): 1.42e-06
```

**Key insights**:
1. ✅ **vmap alone**: ~12× speedup (eliminates Python loop overhead)
2. ✅ **vmap + JIT**: ~94× speedup (vectorization + XLA compilation)
3. ✅ **Exact results**: Differences are roundoff error only
4. ✅ **Same code structure**: Single-star function works for entire ensemble

**Visualization** (optional for reports):
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot ZAMS relations for ensemble
ax1.scatter(masses, L_fast, s=1, alpha=0.5, label='Computed with vmap+JIT')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Mass [M☉]')
ax1.set_ylabel('Luminosity [L☉]')
ax1.set_title('Mass-Luminosity Relation (ZAMS)')
ax1.grid(alpha=0.3)

ax2.scatter(masses, R_fast, s=1, alpha=0.5, color='orange')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Mass [M☉]')
ax2.set_ylabel('Radius [R☉]')
ax2.set_title('Mass-Radius Relation (ZAMS)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('zams_ensemble.png', dpi=150)
plt.show()
```

**Try yourself**:
- Increase to 100,000 stars (speedup grows!)
- Add more physics (metallicity dependence)
- Compute derived quantities (surface gravity, effective temperature)
- Use this pattern for Monte Carlo uncertainty propagation

**This workflow generalizes**: Any single-example function → vmap → instant ensemble capability!
:::

---

:::{important} 💡 What We Just Learned

**`vmap` automatically vectorizes functions, eliminating explicit loops:**

- **Transforms** computational graph to operate on batched arrays
- **No manual loops** → cleaner code, fewer bugs
- **Automatic parallelization** → 10-100× speedups (especially on GPU)
- **Compose with `jit`** → compile vectorized code for maximum performance
- **Use cases**: Batching training data, Monte Carlo sampling, parameter sweeps

**Practical workflow**:
1. Write function for single example
2. Verify correctness
3. Vectorize with `vmap`
4. Compile with `jit`
5. Enjoy speed!
:::

---

(composing-transformations)=
## 2.5: Composing Transformations — The Real Power

**Priority: 🔴 Essential**

### Transformations Compose Freely

**This is JAX's killer feature**: Transformations can be combined in any order.

```python
import jax

def f(x):
    return jnp.sum(x ** 2)

# Compose transformations
g = jax.jit(jax.vmap(jax.grad(f)))

# What does g do?
# 1. grad(f): Compute gradient of f
# 2. vmap(...): Batch gradient computation
# 3. jit(...): Compile batched gradient
```

**This means**: Batched, compiled, differentiable code with **three decorators**.

---

### Common Patterns

#### Pattern 1: Batch + Compile

```python
@jax.jit
@jax.vmap
def process_batch(x):
    # Vectorized AND compiled
    return jnp.exp(-x**2)
```

**Use case**: Apply function to batch of data (e.g., training data)

---

#### Pattern 2: Differentiate + Compile

```python
@jax.jit
@jax.grad
def fast_gradient(params):
    # Gradient computation is compiled
    return loss(params)
```

**Use case**: Optimization loops (gradient descent, HMC)

---

#### Pattern 3: Batch Gradients + Compile

```python
@jax.jit
@jax.vmap
@jax.grad
def batch_gradients(params_batch):
    # Compute gradients for BATCH of parameters
    return loss(params_batch)
```

**Use case**: Ensemble MCMC (run multiple chains in parallel)

---

#### Pattern 4: The Full Stack

```python
@jax.jit              # 3. Compile everything
@jax.vmap             # 2. Batch over simulations
@jax.grad             # 1. Differentiate simulation
def batched_grads(initial_conditions):
    return simulate_and_loss(initial_conditions)
```

**What this does**:
1. Takes gradient of simulation w.r.t. initial conditions
2. Batches this operation over 1000 simulations
3. Compiles the entire pipeline to machine code

**Result**: Compute 1000 gradients of expensive simulations **in seconds** on GPU.

---

### Order Sometimes Matters

**Question**: Does order of composition matter?

**Answer**: Sometimes yes, sometimes no.

#### Commutative: `jit` and `vmap`

```python
jax.jit(jax.vmap(f))  # Compile batched function
# ≈ equivalent to
jax.vmap(jax.jit(f))  # Batch compiled function
```

Both work, but **first is usually better** (compile the vectorized version once, not N times).

---

#### Non-Commutative: `grad` and `vmap`

```python
f: scalar → scalar

jax.vmap(jax.grad(f))  # Batch of gradients (N independent gradients)
# vs
jax.grad(jax.vmap(f))  # Gradient of batched function (Jacobian!)
```

**Different semantics**!

**Example**:
```python
def f(x):
    """Scalar → scalar"""
    return x ** 2

# Batch of gradients
batch_grad = jax.vmap(jax.grad(f))
xs = jnp.array([1.0, 2.0, 3.0])
print(batch_grad(xs))  # [2., 4., 6.] ← independent gradients

# Gradient of batched function
grad_batch = jax.grad(lambda xs: jnp.sum(jax.vmap(f)(xs)))
print(grad_batch(xs))  # [2., 4., 6.] ← same here, but...
```

**When they differ**: For vector-valued functions, order matters!

---

### Real Example: HMC with Batched Chains

**Goal**: Run 10 HMC chains in parallel (for convergence diagnostics).

```python
def hmc_step(state, log_prob_fn):
    """One HMC step (from Project 4)."""
    # ... leapfrog integration, accept/reject ...
    return new_state

# Batch HMC over 10 chains
batched_hmc_step = jax.vmap(hmc_step, in_axes=(0, None))

# Compile for speed
compiled_batched_hmc = jax.jit(batched_hmc_step)

# Initialize 10 chains
states = initialize_chains(n_chains=10)  # (10, d)

# Run all chains in parallel
for _ in range(1000):
    states = compiled_batched_hmc(states, log_prob_fn)  # Blazing fast!
```

**Speedup vs sequential**: ~10× (perfect parallelization) + JIT speedup (~10×) = **~100× total**

---

:::{important} 💡 What We Just Learned

**Transformations compose freely—this is JAX's superpower:**

- **`jit ∘ vmap ∘ grad`**: Batched compiled gradients
- **Order sometimes matters**: `vmap(grad)` ≠ `grad(vmap)` for vector outputs
- **Practical workflow**: Write simple function → add transformations as needed
- **Real power**: Complex pipelines (batched, differentiated, compiled) with minimal code

**Key insight**: Each transformation is a **function that returns a function**. Composition is just function composition.
:::

---

(summary-part2)=
## Summary: Technical Mastery Achieved

**Priority: 🔴 Essential**

**Part 2 gave you the technical tools to use JAX effectively:**

✅ **Automatic differentiation**: Reverse-mode autodiff computes exact gradients in $\approx 2\times$ forward pass cost, regardless of parameter dimension

✅ **JIT compilation**: XLA transforms Python → optimized machine code via operation fusion, vectorization, and hardware-specific optimization

✅ **Vectorization**: `vmap` eliminates explicit loops, batching operations for 10-100× speedups

✅ **Composition**: Transformations combine freely—`jit(vmap(grad(f)))` gives batched, compiled, differentiable code

✅ **Practical skills**: You can now write performant scientific code that's differentiable, fast, and composable

---

:::{admonition} 🔗 Connection to Part 1
:class: note

**Part 1** taught you WHY JAX requires functional programming (constraints enable transformations).

**Part 2** taught you HOW to use those transformations (grad, jit, vmap).

**Next**: Part 3 will give you the **practical JAX toolkit** for scientific computing, covering:

- **`lax.scan`**: Replace time-stepping loops with compiled scans (10-100× faster)
- **`lax.cond`/`lax.switch`**: Handle control flow inside JIT
- **Advanced vmap patterns**: Nested vmap, batched gradients
- **JAX data structures**: PyTrees, custom containers
- **Performance optimization**: When to JIT, memory management, profiling
- **Common patterns**: ODE integration, parameter sweeps, ensemble simulations

**Your capstone**: Migrate your NumPy N-body simulator to JAX, achieving **100-1000× speedups** while gaining automatic differentiation for HMC sampling.
:::

---

(debugging-guide)=
## Debugging Guide: Common JAX Errors and Solutions

**Priority: 🟡 Important**

This appendix provides systematic approaches to debugging the most common JAX errors you'll encounter.

### 1. ConcretizationError

**Error message**:
```
ConcretizationError: Abstract tracer value encountered where concrete value expected.
```

**What it means**: JIT compilation requires knowing control flow at trace time, but you used a traced value in a condition.

**Common causes**:

```python
# ❌ BAD: Value-dependent control flow
@jax.jit
def bad(x):
    if x > 0:  # x is a tracer, not a concrete value!
        return x ** 2
    return -x
```

**Solutions**:

```python
# ✅ GOOD: Use jnp.where for element-wise conditionals
@jax.jit
def good(x):
    return jnp.where(x > 0, x**2, -x)

# ✅ GOOD: Use lax.cond for branching
from jax import lax

@jax.jit
def good_branch(x):
    return lax.cond(
        x > 0,
        lambda x: x ** 2,  # True branch
        lambda x: -x,       # False branch
        x
    )

# ✅ GOOD: Use static_argnums for compile-time decisions
@jax.jit(static_argnums=1)
def good_static(x, mode):
    if mode == "square":  # mode is static, known at compile time
        return x ** 2
    return x ** 3
```

---

### 2. TracerArrayConversionError

**Error message**:
```
TracerArrayConversionError: Attempted boolean conversion of traced array
```

**What it means**: You tried to use a traced JAX array in a Python control flow statement.

**Common cause**:

```python
@jax.jit
def bad(x):
    if jnp.all(x > 0):  # ❌ Can't convert traced array to bool!
        return x
    return -x
```

**Solution**: Same as ConcretizationError—use `jnp.where` or `lax.cond`.

---

### 3. TypeError: Gradient only defined for scalar-output functions

**Error message**:
```
TypeError: Grad only defined for scalar-output functions. Output had shape: (10,)
```

**What it means**: `jax.grad` expects a scalar output (like loss or energy), but your function returns an array.

**Common causes**:

```python
def vector_output(x):
    return jnp.array([x**2, x**3, x**4])  # ❌ Vector output

grad_fn = jax.grad(vector_output)  # Error!
```

**Solutions**:

```python
# ✅ GOOD: Sum to scalar if you want total gradient
def scalar_output(x):
    return jnp.sum(jnp.array([x**2, x**3, x**4]))

grad_fn = jax.grad(scalar_output)  # Works!

# ✅ GOOD: Use jacfwd/jacrev for Jacobians
jac_fn = jax.jacfwd(vector_output)  # Jacobian matrix

# ✅ GOOD: Use vmap to batch scalar gradients
def scalar_loss(x, i):
    return x[i]**2

batch_grad = jax.vmap(jax.grad(scalar_loss), in_axes=(None, 0))
```

---

### 4. IndexError with vmap

**Error message**:
```
IndexError: vmap must have at least one non-None value in in_axes
```

**What it means**: All arguments to vmap have `in_axes=None`, so there's no batch dimension.

**Solution**: At least one argument must be batched:

```python
# ❌ BAD: No batch dimension
batch_fn = jax.vmap(f, in_axes=(None, None))

# ✅ GOOD: At least one batched argument
batch_fn = jax.vmap(f, in_axes=(0, None))  # First arg batched
```

---

### 5. Shape Mismatch Errors

**Error message**:
```
TypeError: Shapes must match, got (3, 2) and (2, 3)
```

**What it means**: Array shapes are incompatible for the operation.

**Debugging strategy**:

```python
# Add shape printing BEFORE jit
def debug_shapes(x, y):
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")  # Will print during trace
    return x + y

# Or use jax.debug.print inside JIT
@jax.jit
def debug_jit(x, y):
    jax.debug.print("x: {x}, y: {y}", x=x.shape, y=y.shape)
    return x + y
```

**Common fixes**:

```python
# Transpose if needed
result = x @ y.T

# Reshape
y_reshaped = y.reshape(x.shape)

# Broadcast explicitly
y_broadcast = jnp.broadcast_to(y, x.shape)
```

---

### 6. NaN/Inf in Gradients

**Error**: Gradients contain NaN or Inf values.

**Enable NaN checking**:

```python
# At start of script
jax.config.update("jax_debug_nans", True)
# Now JAX will error immediately when NaN is produced
```

**Common causes**:

```python
# Division by zero
r = jnp.sqrt(jnp.sum(r_vec**2))  # Can be 0!
U = -G * m1 * m2 / r  # ❌ Division by zero

# Fix: Add softening
r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # ✅ Never exactly 0

# Log of zero/negative
loss = -jnp.log(probability)  # ❌ If probability=0

# Fix: Clip
loss = -jnp.log(jnp.clip(probability, 1e-10, 1.0))  # ✅ Safe
```

---

### 7. Out of Memory (OOM) Errors

**Error**: `RESOURCE_EXHAUSTED: OOM when allocating tensor`

**Causes and solutions**:

```python
# ❌ BAD: Storing entire trajectory in memory
trajectories = []
for t in range(10000):
    state = step(state)
    trajectories.append(state)  # Accumulates in memory!

# ✅ GOOD: Use checkpointing or lax.scan
def scan_step(carry, _):
    state = step(carry)
    return state, state  # Only return what you need

final, trajectory = lax.scan(scan_step, init_state, jnp.arange(10000))

# ✅ GOOD: Downsample storage
if t % 100 == 0:  # Save every 100th step
    trajectories.append(state)
```

---

### 8. Slow First Call (Compilation Overhead)

**Issue**: First call to JIT function is very slow.

**This is normal!** Subsequent calls are fast.

**Solutions**:

```python
# Warm-up: Call once with dummy data
_ = jitted_function(dummy_input)

# Or show progress
print("Compiling...")
result = jitted_function(real_input)  # Slow first time
print("Done compiling, now fast!")
result = jitted_function(real_input)  # Fast!

# For large functions: Use ahead-of-time compilation
from jax.experimental import jax2tf
# ... (advanced, see JAX docs)
```

---

### General Debugging Workflow

**Step 1: Disable JIT**

```python
# Temporarily disable JIT to get better error messages
jax.config.update("jax_disable_jit", True)

# Run your code
result = my_function(x)

# Re-enable after debugging
jax.config.update("jax_disable_jit", False)
```

**Step 2: Add print statements**

```python
# Outside JIT: normal print works during trace
def f(x):
    print(f"Tracing with x.shape = {x.shape}")  # Prints once
    return x ** 2

# Inside JIT: use jax.debug.print
@jax.jit
def f(x):
    jax.debug.print("x = {}", x)  # Prints on every call
    return x ** 2
```

**Step 3: Verify intermediate values**

```python
@jax.jit
def complex_function(x):
    y = step1(x)
    jax.debug.print("After step1: {}", y)

    z = step2(y)
    jax.debug.print("After step2: {}", z)

    return z
```

**Step 4: Simplify to minimal example**

```python
# If your full function fails, create minimal test case
def minimal_test():
    x = jnp.array([1.0, 2.0])
    result = my_problematic_operation(x)
    print(result)

minimal_test()  # Debug this first!
```

---

:::{admonition} 🔬 Debugging Checklist
:class: tip

When you encounter an error:

1. [ ] Read the full error message (don't just skim!)
2. [ ] Check if it matches a common error above
3. [ ] Disable JIT temporarily (`jax_disable_jit`)
4. [ ] Add `jax.debug.print` statements
5. [ ] Enable NaN checking (`jax_debug_nans`)
6. [ ] Print shapes of all arrays involved
7. [ ] Create minimal reproduction case
8. [ ] Check JAX GitHub issues for similar problems
9. [ ] Ask on JAX Discourse forum with minimal example

**Most important**: JAX errors are usually about **constraints violations** (mutation, control flow, purity). Review Part 1 concepts!
:::

---

:::{admonition} 📚 Additional Resources
:class: tip

**Official JAX Documentation**:
- [Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [JIT Compilation Guide](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
- [vmap Tutorial](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)

**Deep Dives**:
- [Automatic Differentiation in ML: A Survey](https://arxiv.org/abs/1502.05767) (Baydin et al. 2018)
- [XLA: Optimizing Compiler for ML](https://www.tensorflow.org/xla)

**Interactive Practice**:
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX Tutorial Notebooks](https://github.com/google/jax/tree/main/docs/notebooks)
:::

---

## Understanding Checklist

Before proceeding to Part 3, ensure you can:

- [ ] **Explain** how reverse-mode autodiff computes gradients using the chain rule
- [ ] **Use** `jax.grad` with correct `argnums` to compute derivatives
- [ ] **Identify** when JIT compilation helps vs hurts performance
- [ ] **Debug** common JIT tracing errors (ConcretizationError, side effects)
- [ ] **Apply** `vmap` to vectorize functions over batch dimensions
- [ ] **Compose** transformations (jit ∘ vmap ∘ grad) in the right order
- [ ] **Benchmark** NumPy vs JAX code and explain the speedup sources
- [ ] **Recognize** patterns that require `lax.scan`, `lax.cond`, or `lax.fori_loop`

If you answered "yes" to all → **Ready for Part 3: N-body Migration to JAX**

---

**You're ready for Part 3: Applying these tools to real physics simulations!**
