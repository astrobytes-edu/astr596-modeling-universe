# Module 6: Computing the Universe

## From NumPy Scripts to Scientific Software Engineering

> *"The purpose of computing is insight, not numbers."* — Richard Hamming
>
> *"Science is what we understand well enough to explain to a computer. Art is everything else."* — Donald Knuth

---

## Module Overview

**Position in Course**: Bridge module between statistical inference (Modules 1-5) and machine learning (Module 7)

**Scope**: Comprehensive foundation - transforms students from script writers to scientific software engineers

**The Transformation**: This module transforms you from **script writers to scientific software engineers**. You'll learn to:

- Write production-quality scientific code (not just scripts that "work")
- Differentiate through any computation automatically (autodiff)
- Batch and parallelize naturally (vectorization)
- Compile for performance (10-100× speedups)
- Package your code for others to use (from notebook to `pip install`)
- Generate reproducible training data (for machine learning)

---

## The Big Picture: Why This Module Matters

**You just finished Project 4**, computing gradients by hand for HMC. It worked, but it was painful:

```python
# What you did in Project 4 (finite differences)
def grad_log_posterior(theta, h=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus, theta_minus = theta.copy(), theta.copy()
        theta_plus[i] += h; theta_minus[i] -= h
        grad[i] = (log_posterior(theta_plus) - log_posterior(theta_minus)) / (2*h)
    return grad  # 2d function calls, approximate, fragile
```

**What if you could do this instead?**

```python
# What you'll do after Module 6 (JAX)
grad_log_posterior = jax.grad(log_posterior)  # Exact, fast, automatic
```

That's just the beginning. By the end of this module + Project 5, you'll have transformed your N-body simulator from a NumPy script into a **professional JAX package** that:

- ✅ Runs 10-100× faster (JIT compilation)
- ✅ Computes forces via automatic differentiation (no hand-coded derivatives)
- ✅ Batches thousands of simulations efficiently (auto-vectorization)
- ✅ Is packaged, tested, and documented (pip-installable: `pip install nbody-jax`)
- ✅ Generates training data for machine learning (ready for Module 7)

**The Complete Arc**:
```
Project 2 → NumPy N-body (you built physics from scratch)
Project 4 → Manual gradients for HMC (you saw why gradients matter)
Module 6 → JAX foundations (you learn the tools)
Project 5 → Professional JAX package (you become a software engineer)
Module 7 → Machine learning (you use your package + data)
Final Project → Physics-informed learning (you do novel research)
```

**This isn't just learning a new library. It's learning how modern scientific software is built.**

---

## Module Learning Outcomes

By the end of Module 6, you will be able to:

- **LO-JAX-1**: Explain how automatic differentiation enables gradient-based learning and optimization (conceptual)
- **LO-JAX-2**: Write pure, functional code compatible with JAX transformations (implementation)
- **LO-JAX-3**: Apply `jit`, `grad`, `vmap`, and `pmap` to scientific computing workflows (technical skill)
- **LO-JAX-4**: Manage randomness explicitly using JAX's PRNG system (technical skill)
- **LO-JAX-5**: Migrate NumPy scientific code to JAX with performance optimization (applied)
- **LO-JAX-6**: Build training loops using Optax for gradient-based optimization (applied)
- **LO-JAX-7**: Navigate the JAX ecosystem for ML applications (Equinox, Diffrax, GPJax) (awareness)
- **LO-JAX-8**: Debug JAX code and optimize performance systematically (professional)
- **LO-JAX-9**: Structure code as a Python package with tests and documentation (professional)
- **LO-JAX-10**: Generate and manage training data for machine learning (research skill)

These map to course-level outcomes:

- **Course LO7**: Implement algorithms using modern computational paradigms (functional programming, autodiff)
- **Course LO8**: Apply automatic differentiation to physical systems
- **Course LO9**: Optimize computational workflows (JIT, vectorization, profiling)
- **Course LO10**: Build research-grade scientific software (packaging, testing, documentation)

---

## Pedagogical Structure (Consistent Across All Parts)

Each part follows the same pattern:

1. **YAML frontmatter** with title and subtitle
2. **Learning Outcomes** with checkboxes (action verbs)
3. **Priority markers** on sections (🔴 Essential, 🟡 Important, 🟢 Enrichment)
4. **MyST anchors** before all major headers (for cross-referencing)
5. **Pedagogical elements**: margin notes, "What We Just Learned" boxes, conceptual checkpoints
6. **Glass-box methodology**: Explain HOW things work internally, not just API usage
7. **Cross-module connections**: Link to Projects 1-4, Module 1, Module 5
8. **CGS units throughout**: G = 6.67e-8 cm³ g⁻¹ s⁻²

---

## Module Structure

### **Overview: The JAX Revolution in Scientific Computing** 🔴 Essential

**Scope**: Motivation and context (NOT a tutorial - pure motivation and framing)
**File**: `overview-module-6-jax-revolution.md` (~2090 lines)

**Learning Paths**:
- **Fast Track**: Essential sections only - core value proposition
- **Standard Track**: Add computational landscape, JAX origin story, three eras
- **Complete Track**: Include career context (dropdown), all checkpoints, reflection prompts

**Learning Outcomes** (6 checkboxes):
- Explain why JAX exists and how Google's TPU constraints led to composable transformations
- Identify the three computational eras and where JAX fits in scientific computing evolution
- Connect your Project 4 finite differences pain to automatic differentiation solutions
- Describe the three core JAX transformations (grad, jit, vmap) and what they enable
- Articulate the transformation from script writer to scientific software engineer
- Evaluate when Era 3 approaches (ML surrogates) are advantageous over Era 2 (direct MCMC)

**Content**:

#### The Crisis That Changed Everything
Google Brain spring 2017: TPU hardware exists, but software can't exploit it. TensorFlow too rigid, PyTorch too slow on specialized hardware. JAX emerges as the solution: flexible Python code + composable transformations + hardware acceleration.

#### What This Module Is About
Transform from script writers → scientific software engineers. Build professional JAX package that:
- Runs 10-100× faster (JIT compilation)
- Computes gradients automatically (autodiff)
- Batches 10,000 simulations (vectorization)
- Is pip-installable, tested, documented

#### The Computational Scientific Landscape
Why most academics still use "old" tools (MATLAB, Fortran, basic NumPy). The academic-industry gap. Why you're in a unique position learning modern paradigm from day one.

#### Career Context (Dropdown Admonition, 🟢 Enrichment)
- What's never taught this way (JAX rarely in curricula)
- Academic-industry gap (2025)
- The rare combination you're building (physics + ML + software engineering)
- Concrete opportunities (academic track, industry track, hybrid path)
- Immediate portfolio value (Project 5 as talking point)

#### Google's JAX — Why Does This Library Exist?
- The problem Google faced (2015-2017): TPUs exist, software can't use them
- Existing frameworks insufficient (TensorFlow static graphs, PyTorch slow on specialized hardware)
- What researchers actually wanted: write Python → get gradients, compilation, hardware acceleration
- The solution: JAX (Just After eXecution) with composable transformations
- Why composability matters: `grad`, `jit`, `vmap`, `pmap` combine freely

#### The Pain Point — Your Project 4 Experience
Remember Project 4? Manual gradients via finite differences:
- Computational cost: 2d function evaluations for d-dimensional gradient
- Inaccuracy: approximate derivatives, truncation errors
- Computational waste: For d=100 params, HMC becomes impractical
- With autodiff: ~2× forward pass cost, regardless of d. Exact. Automatic.

#### A Taste of JAX — Light Technical Preview
**Don't worry if you don't understand yet—that's what Parts 1-2 are for.**

Examples with actual code (CGS units):
- **Automatic differentiation**: Compute gravitational forces via `jax.grad` of potential energy
- **JIT compilation**: 37× speedup with one `@jax.jit` decorator
- **Vectorization**: Batch 1000 simulations with `jax.vmap`, 10× speedup from parallelism

Connection to Module 1: Ensemble averages from statistical mechanics → computational analog with vmap.

#### The Three Computational Eras
1. **Era 1 (1950s-2010s)**: Computers simulate physics (forward problems)
   - Your Projects 1-3: Monte Carlo, N-body, radiative transfer
   - Limitation: Can't efficiently answer "what parameters produced this?"

2. **Era 2 (1990s-2020s)**: Computers infer parameters (inverse problems)
   - Your Module 5 + Project 4: Bayesian inference, HMC
   - Limitation: Expensive for realistic likelihoods (N-body: seconds, radiative transfer: hours)

3. **Era 3 (2020s-present)**: Computers learn from simulations (amortized inference)
   - Your Module 6-7 + Project 5: Generate 10k simulations → train ML surrogate → instant predictions
   - Enables: Real-time inference, high-d parameter spaces, interactive exploration

Real-world examples: Cosmological parameter inference, exoplanet atmospheric retrievals.

#### Looking Ahead — What You'll Build
**Module 6 Structure**:
- Part 1 (~3h): Conceptual foundations (functional programming, pure functions, computational graphs)
- Part 2 (~5h): Core transformations (grad, jit, vmap, composing)
- Part 3 (~4h): N-body migration to JAX
- Part 4A (~2h): Optax (optimization library)
- Part 4B (~1h): JAX ecosystem (Equinox, Diffrax, Lineax, NumPyro)
- Part 5 (~3h): Professional software engineering (packaging, testing, docs)
- Part 6 (~2h): Synthesis (reflection, cross-module connections)

**Project 5**: Your JAX N-body Package
- Runs 10-100× faster (JIT compilation)
- Computes forces via autodiff (no hand-coded derivatives)
- Batches thousands of simulations (automatic vectorization)
- Is pip-installable with tests and documentation
- Generates training data for ML (10,000 simulations for Module 7)

**The Transformation**: Script writer → scientific software engineer
- Before: Manual gradients, sequential loops, NumPy only, one-off scripts
- After: Automatic differentiation, auto-vectorization, JIT compilation, professional packages

**Pedagogical Goal**: Connect lived experience → show what's possible → explain why JAX exists → preview transformation → motivate Part 1-2 learning.

---

### **Part 1: Conceptual Foundations — Why JAX Requires Functional Programming** 🔴 Essential

**Scope**: Conceptual foundation (WHY before HOW)
**Prerequisites**: Overview completed
**File**: `part-01-conceptual-foundations.md` (~1933 lines)

**Core Question**: Why does JAX require such strange constraints? (No mutation, explicit control flow, pure functions)

**The Answer**: These constraints aren't arbitrary—they're what make transformations possible.

**Learning Outcomes** (6 checkboxes):
- Contrast object-oriented and functional programming paradigms
- Identify pure vs impure functions in Python code
- Explain why mutation breaks JAX transformations
- Recognize data-dependent control flow that prevents JIT compilation
- Describe computational graphs conceptually
- Articulate why JAX's constraints enable its transformations

**Content Structure**:

#### 1.1: OOP vs Functional Programming 🔴
You already know OOP (your `Star` class from Project 1). Functional style organizes code differently:
- **OOP**: Objects with mutable state (`star.evolve()` modifies `self.age`)
- **Functional**: Immutable data transformations (`new_star = evolve(old_star, dt)`)

**Side-by-side comparison**:
| Aspect | OOP (Your `Star` Class) | Functional (JAX Style) |
|--------|-------------------------|------------------------|
| State | Mutable (`self.age` changes) | Immutable (return new state) |
| Methods | Modify object | Transform data |
| Dependencies | Hidden in `self` | Explicit in parameters |

**Why JAX requires functional style**:
- JIT compilation needs predictable control flow
- Autodiff needs clear dependency graphs
- Vectorization (vmap) needs independent computations

**Practical JAX example**: N-body integrator with pure functions, no mutation.

**Connection to learning this semester**: You learned OOP for organizing systems. Now learning functional for JAX transformations. Both valuable—you'll use OOP for project structure and functional for computational core.

#### 1.2: Pure Functions — The Foundation of JAX 🔴

**Definition**: A pure function has two properties:
1. **Deterministic**: Same inputs → always same outputs (no randomness, no hidden state)
2. **No side effects**: Doesn't modify anything outside its scope (no mutation, no I/O)

**Examples**:
```python
# ✅ Pure
def compute_force(m1, m2, r):
    G = 6.67e-8  # CGS
    return G * m1 * m2 / r**2

# ❌ Impure (global state)
counter = 0
def impure_func(x):
    global counter
    counter += 1  # Side effect!
    return x + counter

# ❌ Impure (mutation)
def impure_modify(arr):
    arr[0] = 999  # Modifies input!
    return arr
```

**Why purity matters for JAX**:

1. **JIT compilation**: JAX traces your function once with abstract values, compiles graph, reuses. If function is impure (depends on hidden state), compiled graph won't match reality.

2. **Automatic differentiation**: Autodiff tracks computational graph. If function has side effects or hidden dependencies, gradient computation breaks.

3. **Vectorization**: `vmap` assumes computations are independent. Side effects break parallelization.

**Practical test for purity**: Can you replace `f(x)` with its return value everywhere and program behaves identically? If yes → pure.

**By Project 5**: Checking for purity becomes second nature.

#### 1.3: Control Flow Constraints — Why `if` and `for` Break JAX 🔴

**The problem**: Data-dependent control flow

```python
def f(x):
    if x > 0:        # ❌ Breaks JIT!
        return x**2
    else:
        return -x**2
```

**Why this breaks**: During JIT compilation, JAX traces with abstract values (knows shape/dtype, not actual value). Can't evaluate `x > 0` without concrete value!

**JAX alternatives**:
- **`jnp.where`**: `jnp.where(x > 0, x**2, -x**2)` — computes BOTH branches, selects result
- **`lax.cond`**: `lax.cond(x > 0, lambda: x**2, lambda: -x**2)` — compiles both paths, chooses at runtime
- **`lax.scan`**: Replaces Python for-loops with functional scan operation
- **`lax.fori_loop`**: Compiled fixed-iteration loop

**When Python control flow IS okay**:
- Shape-dependent code (loop over array dimensions known at trace time)
- Hyperparameters (fixed before compilation)

**Part 2 will cover technical details and practical patterns**. For now, understand WHY: JAX's constraints enable compilation, and compilation enables speed.

#### 1.4: Computational Graphs — The Mental Model 🟡

**How does JAX "see" your code?**

When you write:
```python
def f(x):
    y = x**2
    z = jnp.sin(y)
    return z
```

JAX sees a **computational graph** (directed acyclic graph/DAG):
```
x → [square] → y → [sin] → z
```

**During JIT compilation**:
1. **Tracing**: JAX calls your function with "tracer" (abstract value, knows shape/dtype)
2. **Graph building**: Records all operations and data dependencies
3. **Compilation**: XLA compiles graph to optimized machine code
4. **Execution**: Runs compiled code (not Python interpreter!)

**Why computational graphs matter**:
- **JIT**: Compile graph once, reuse for all future calls
- **Autodiff**: Traverse graph backward computing derivatives (chain rule)
- **vmap**: Add batch dimensions to graph nodes

**Pure functions → predictable graphs**: No hidden dependencies, no mutations, graph fully describes computation.

**Conceptual checkpoint**: Before moving on, ensure you can:
- Draw simple computational graph for `y = (x + 1) * (x - 1)`
- Explain why `print()` inside JIT function only executes once
- Recognize that JAX builds graphs during tracing, executes them later

#### 1.5: What Are JAX Transformations? 🔴

**Transformations are functions that transform functions**. This sounds abstract, but it's powerful:

```python
def f(x):
    return x ** 2

fast_f = jax.jit(f)       # Compile for speed
grad_f = jax.grad(f)      # Compute derivative
batched_f = jax.vmap(f)   # Vectorize over batch
```

**The four core transformations**:

1. **`jit`** (Just-In-Time Compilation): Compile function to machine code for 10-100× speedups
2. **`grad`** (Automatic Differentiation): Compute exact derivatives automatically
3. **`vmap`** (Vectorization): Automatic batching over dimensions, eliminating loops
4. **`pmap`** (Parallelization): Multi-device parallelism across GPUs/TPUs

**The magic**: These compose freely!
```python
fast_batched_gradients = jax.jit(jax.vmap(jax.grad(f)))
```

**Part 2 covers**:
- HOW each transformation works technically
- WHEN to use which transformation
- HOW to combine them effectively
- Common patterns and pitfalls

For now, understand the **concept**: JAX gives you building blocks (transformations) that you compose to build fast, differentiable, parallel scientific code.

#### 1.6: Why JAX Requires Functional Programming (Synthesis) 🔴

**Connecting all the pieces**:

You've learned:
1. **OOP vs Functional**: Different paradigms for organizing code
2. **Pure Functions**: Same inputs → same outputs, no side effects
3. **Control Flow**: Why Python `if`/`for` break, what `lax` provides
4. **Computational Graphs**: How JAX "sees" your code
5. **Transformations**: What `jit`, `grad`, `vmap`, `pmap` do

**Now: Why do all these constraints exist?**

**The key insight**: JAX's constraints ENABLE its capabilities.

- **Functional programming** → enables JAX to build predictable computational graphs
- **Pure functions** → enable JIT compilation, autodiff, vectorization
- **Explicit control flow** → enables tracing with abstract values (shapes/types, not data)
- **Immutability** → enables safe parallelization and composable transformations

**The paradigm shift**: From "writing scripts that compute values" to "composing mathematical transformations that JAX can optimize, differentiate, and parallelize automatically."

This isn't just faster code—it's a fundamentally different way of thinking about scientific computation.

**Checklist: "Is my function JAX-ready?"**

Before Part 2 (where you'll write actual JAX code), develop intuition:

1. **Is this function pure?**
   - No global state access?
   - No side effects (print, file I/O, mutation)?
   - Deterministic (same inputs → same outputs)?

2. **Is control flow JAX-compatible?**
   - Using `jnp.where` instead of `if`?
   - Using `lax.cond` for conditionals?
   - Using `lax.scan` instead of for-loops?

3. **Am I creating new data, not mutating?**
   - Using `x = x + 1`, not `x += 1`?
   - Using `.at[].set()`, not `[i] = value`?

**If yes to all → JAX will work smoothly. If no → expect errors.**

**Looking Ahead to Part 2**:

Part 1 built your conceptual foundation. You understand:
- WHY functional programming
- WHY pure functions
- WHY explicit control flow
- HOW computational graphs work (conceptually)
- WHAT transformations do (conceptually)

**Part 2 teaches technical execution**:
- Using `jax.grad` with argnums for specific arguments
- Debugging JIT compilation errors
- Composing transformations effectively
- Debugging common errors
- Performance optimization

**You're ready.** The conceptual understanding you've built makes Part 2's technical details make sense.

---

### **Part 2: Core Transformations — Technical Mastery** 🔴 Essential

**Scope**: HOW to use JAX transformations (grad, jit, vmap)
**Prerequisites**: Part 1 completed
**File**: `part-02-core-transformations.md` (~1399 lines)

**Core Goal**: Technical mastery of JAX's three superpowers

**Learning Outcomes** (9 checkboxes):
- Explain reverse-mode autodiff mathematics
- Compute gradients using `jax.grad` with correct argnums
- Identify when JIT compilation helps vs hurts
- Debug JIT tracing errors
- Vectorize functions using `vmap` with correct in_axes/out_axes
- Compose transformations (jit ∘ vmap ∘ grad)
- Benchmark performance (NumPy vs JAX)
- Choose appropriate transformations for specific problems
- Recognize common JAX error patterns

**Content Structure**:

#### 2.1: Automatic Differentiation (`grad`) 🔴

**The Problem**: Finite differences (Project 4 pain point)
- Expensive: O(d) function calls for d-dimensional gradient
- Approximate: truncation errors, numerical instability
- Fragile: scale differences between parameters break
- Manual: different code for each function

**Forward-mode vs Reverse-mode Autodiff**:
- Mathematics: Systematic chain rule application
- Why reverse-mode for physics: many parameters → scalar output (e.g., log-likelihood, potential energy, loss function)
- Cost: ~2× forward pass, regardless of d (vs d× for finite differences)
- Why this matters: For d=1000, reverse-mode is 1000× more efficient than forward-mode

**Practical Usage**:
```python
# Basic gradient
grad_f = jax.grad(f)

# Multiple arguments
grad_U = jax.grad(U, argnums=0)  # w.r.t. first argument

# Jacobians
jac = jax.jacobian(f)  # Full Jacobian matrix
```

**Real Example**: Gravitational forces via grad of potential energy (CGS units)
```python
def gravitational_potential(positions, masses):
    """
    Gravitational potential energy (CGS units).

    Parameters:
        positions: (N, 3) [cm]
        masses: (N,) [g]

    Returns:
        U: scalar [erg]
    """
    G = 6.67e-8  # cm³ g⁻¹ s⁻²
    N = len(masses)
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # Softening
            U += -G * masses[i] * masses[j] / r
    return U

# Get force function automatically: F = -∇U
force_function = jax.grad(gravitational_potential, argnums=0)
forces = -force_function(positions, masses)
```

**Common Gotchas**:
- Non-differentiable operations (integer arrays, discrete ops)
- Forgetting to return scalar for `grad` (use `jnp.sum` if needed)
- Wrong `argnums` for multi-argument functions

**Connection to Project 4**: This is exactly what you needed for HMC gradients! Now automatic and exact.

#### 2.2: JIT Compilation (`jit`) 🔴

**How Tracing Works**:
1. JAX calls function with "tracer" (abstract value: knows shape/dtype, not actual data)
2. Records all operations → builds computational graph
3. XLA compiles graph to optimized machine code
4. Reuses compiled code for all future calls with same shapes

**XLA Optimizations**:
- **Operation fusion**: Combine multiple ops into single kernel (eliminate Python overhead)
- **Dead code elimination**: Remove unused computations
- **Vectorization (SIMD)**: Use CPU vector instructions
- **Hardware-specific optimization**: Tailored for your CPU/GPU/TPU

**Typical Speedups**: 10-100× for numerical code

**When JIT Helps**:
- Repeated function calls (amortize compilation cost)
- Numerical computations with loops
- Functions with many small operations

**When JIT Hurts**:
- One-time function calls (compilation overhead)
- I/O operations (can't compile)
- Already-optimized library calls (NumPy BLAS)

**Debugging**:
- `ConcretizationTypeError`: Trying to use abstract value as concrete → use `jnp.where` or `lax.cond`
- `print()` only runs once: It executes during tracing, not execution
- Use `jax.debug.print()` for runtime debugging

**Example**:
```python
@jax.jit
def compute_forces_jit(positions, masses):
    # Same code as before, but compiled!
    return forces

# First call: traces + compiles (~100ms)
# Subsequent calls: just execute compiled code (~1ms)
# Result: ~40× speedup typical
```

#### 2.3: Vectorization (`vmap`) 🔴

**The Problem**: Manual loops over batch dimensions are slow

```python
# Slow: Python loop
results = []
for ic in initial_conditions:
    results.append(simulate(ic))
```

**The Solution**: `vmap` automatically adds batch dimension

```python
# Fast: Automatic vectorization
batched_simulate = jax.vmap(simulate)
results = batched_simulate(initial_conditions_batch)
```

**How vmap Works**:
- Takes function designed for single input
- Automatically broadcasts over batch dimension
- Returns vectorized function that processes entire batch

**Syntax**:
```python
# Basic: batch over first axis of all arguments
vmap(f)

# Control which arguments to batch
vmap(f, in_axes=(0, None))  # Batch first arg, broadcast second

# Control output axes
vmap(f, out_axes=1)  # Batch dimension in axis 1 of output
```

**Real Example**: Vectorized N-body forces (CGS)
```python
# Single N-body system
def compute_forces_single(positions, masses):
    # ... compute forces for one system ...
    return forces

# Batch over 100 different initial conditions
compute_forces_batch = jax.vmap(compute_forces_single)

# Process 100 systems in parallel!
all_forces = compute_forces_batch(positions_batch, masses_batch)
# positions_batch: (100, N, 3)
# all_forces: (100, N, 3)
```

**Connection to Module 1**: Ensemble averages from statistical mechanics
- Statistical mechanics: Average over many realizations from canonical ensemble
- Computational analog: `vmap` to simulate 1000 initial conditions, compute ensemble statistics
- Same physics, new computational paradigm

**Practical Impact for Project 5**:
- Without vmap: 10,000 simulations × 1 second = 3 hours
- With vmap on GPU (batches of 100): 100 batches × 2 seconds = 3 minutes
- Speedup: 60× faster → makes Project 5 feasible within semester

#### 2.4: Composing Transformations 🔴

**The Power**: Transformations compose freely

**Common Patterns**:
```python
# Batched gradients (for mini-batch training)
batched_grads = jax.vmap(jax.grad(loss_fn))

# Compiled batched gradients (FAST!)
fast_batched_grads = jax.jit(jax.vmap(jax.grad(loss_fn)))

# Higher-order: gradient of gradient
hessian_diag = jax.grad(jax.grad(f))
```

**When Order Matters**:
- `vmap(grad(f))`: Batch dimension of gradients → grad for each input in batch
- `grad(vmap(f))`: Gradient of vectorized function → different mathematical object
- Both useful, but mean different things!

**Real Example**: HMC with batched chains (10 chains in parallel)
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

# Speedup: ~10× (parallelism) × ~10× (JIT) = ~100× total
```

**Summary: Technical Mastery Achieved**

Part 2 gave you the technical tools to use JAX effectively:

✅ **Automatic differentiation**: Reverse-mode autodiff computes exact gradients in ~2× forward pass cost, regardless of parameter dimension

✅ **JIT compilation**: XLA transforms Python → optimized machine code via operation fusion, vectorization, hardware-specific optimization (10-100× speedups typical)

✅ **Vectorization**: `vmap` eliminates explicit loops, batching operations for 10-100× speedups from parallelism

✅ **Composition**: Transformations combine freely—`jit(vmap(grad(f)))` gives batched, compiled, differentiable code

✅ **Practical skills**: You can now write performant scientific code that's differentiable, fast, and composable

**Connection to Part 1**:
- Part 1 taught you WHY JAX requires functional programming (constraints enable transformations)
- Part 2 taught you HOW to use those transformations (grad, jit, vmap)

**Next**: Part 3 will apply these tools to migrate your N-body code to JAX, combining everything you've learned into a working differentiable simulator.

---

### **Part 3: Differentiable Physics — Your N-body Code in JAX** 🔴 Essential

**Scope**: Applied integration - Project 5 preparation
**Effort**: High - systematic migration workshop
**Prerequisites**: Parts 1-2 completed, Project 2 N-body code available

**Epigraph**:
> *"The best way to understand JAX is to JAX-ify code you already understand."* — Anonymous
>
> *"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."* — Martin Fowler

**Core Question**: How do we systematically migrate real physics code from NumPy to JAX?

**This is Project 5 preparation**: You'll transform your Project 2 N-body simulator into a professional JAX package.

:::{admonition} 🔗 Connection to Your Work
:class: note

**Project 2 (NumPy N-body)**: You built from scratch

- Force calculations via direct summation
- Leapfrog integration (symplectic, energy-conserving)
- Position/velocity updates (mutable NumPy arrays)

**Module 6 Part 3**: Transform to JAX

- Forces via autodiff (more elegant, less error-prone)
- Same leapfrog algorithm (functional style)
- Batch simulations with vmap (new superpower!)

**Project 5**: Package and productionize

- Testing framework to validate correctness
- Performance benchmarks to prove speedups
- Training data generation for Module 7
:::

#### 3.1 The Migration Process 🔴

**The 7-Step Checklist**:
```
NumPy Code (Project 2)
  ↓ Step 1: Identify pure vs impure functions
  ↓ Step 2: Remove mutations → functional updates
  ↓ Step 3: Replace np → jnp (mostly automatic)
  ↓ Step 4: Add @jit incrementally (test each function)
  ↓ Step 5: Add vmap for batch processing
  ↓ Step 6: Use grad for force computation (optional but elegant)
  ↓ Step 7: Structure as package (see Part 5)
JAX Package (production-ready!)
```

**Common Translation Patterns**:

```python
# Mutation → Functional updates
x[i] = value          →  x = x.at[i].set(value)
x += delta            →  x = x + delta
x[mask] = value       →  x = x.at[mask].set(value)

# Random numbers → Explicit keys
np.random.normal()    →  jax.random.normal(key, shape)
np.random.seed(42)    →  key = jax.random.PRNGKey(42)
# ... sample ...      →  key, subkey = jax.random.split(key)
                         sample = jax.random.normal(subkey, shape)

# Control flow (inside @jit) → Functional equivalents
if x > 0: ...         →  jnp.where(x > 0, ..., ...)
for i in range(n): ...→  jax.lax.fori_loop(...) or vmap
while cond: ...       →  jax.lax.while_loop(...)
```

#### 3.2 N-body Migration — Step by Step 🔴

**Your Starting Point**: Project 2 N-body (NumPy, mutable, slow)

**Step 1: Pure Functions** (no classes, no mutation)

*Pedagogical approach*:

```python
def compute_forces_loops(positions, masses, G=6.67e-8):
    """Pure function: same inputs → same outputs

    Educational version - shows structure clearly
    """
    n = len(masses)
    forces = jnp.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = jnp.linalg.norm(r_vec)
                f_ij = G * masses[i] * masses[j] * r_vec / (r**3 + 1e-10)
                forces = forces.at[i].add(f_ij)  # Functional update!
    return forces
```

*Modern JAX approach* (vectorized):

```python
def compute_forces_vectorized(positions, masses, G=6.67e-8):
    """Vectorized force calculation - the JAX way!

    Faster and more elegant than loops
    """
    # Pairwise displacement vectors: r_ij = r_j - r_i
    # Shape: (n, n, 3) via broadcasting
    r_ij = positions[None, :, :] - positions[:, None, :]

    # Pairwise distances
    # Shape: (n, n)
    r_mag = jnp.linalg.norm(r_ij, axis=2)

    # Avoid self-interaction and division by zero
    # Set diagonal (self-interactions) to large value
    r_mag = jnp.where(jnp.eye(len(masses), dtype=bool), 1e10, r_mag)

    # Pairwise force magnitudes: F_ij = G * m_i * m_j / r_ij^3
    # Shape: (n, n, 1) - broadcasting for vector multiplication
    f_mag = G * masses[:, None, None] * masses[None, :, None] / (r_mag[:, :, None]**3 + 1e-10)

    # Force vectors: F_ij_vec = f_mag * r_ij
    # Shape: (n, n, 3)
    f_vectors = f_mag * r_ij

    # Sum over j to get total force on each particle i
    # Shape: (n, 3)
    forces = jnp.sum(f_vectors, axis=1)

    return forces
```

**Pedagogical note**: Show both approaches! Loops are clearer conceptually, but vectorization is the JAX way.

**Step 2: Forces via Autodiff** (the most elegant approach!)

```python
def gravitational_potential(positions, masses, G=6.67e-8):
    """Total gravitational potential energy

    U = -G * sum_{i<j} (m_i * m_j / r_ij)
    """
    n = len(masses)
    U = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r_ij = jnp.linalg.norm(positions[i] - positions[j])
            U -= G * masses[i] * masses[j] / (r_ij + 1e-10)
    return U

# Forces automatically via autodiff!
def compute_forces_autodiff(positions, masses, G=6.67e-8):
    """Compute forces as gradient of potential energy

    F_i = -∇_i U(positions)

    This is the glass-box philosophy in action:
    - More elegant than manual force calculation
    - Guaranteed to be consistent with energy
    - Fewer places for bugs to hide
    """
    grad_fn = jax.grad(gravitational_potential, argnums=0)
    return -grad_fn(positions, masses, G)
```

**Which to use?** All three are valid! Choose based on:

- **Loops**: Educational clarity, easy debugging
- **Vectorized**: Fastest for moderate N (10-1000 particles)
- **Autodiff**: Most elegant, guaranteed energy consistency

**Step 3: JIT Compilation.**

```python
@jit
def nbody_step(positions, velocities, masses, dt, G=6.67e-8):
    """Single leapfrog integration step — JIT compiled!

    This is your Project 2 integrator, now compiled for speed
    """
    # Compute forces
    forces = compute_forces_autodiff(positions, masses, G)

    # Leapfrog kick-drift-kick
    # Kick (half step)
    velocities = velocities + 0.5 * forces / masses[:, None] * dt
    # Drift (full step)
    positions = positions + velocities * dt
    # Recompute forces at new positions
    forces = compute_forces_autodiff(positions, masses, G)
    # Kick (half step)
    velocities = velocities + 0.5 * forces / masses[:, None] * dt

    return positions, velocities
```

**Benchmark expectation**: 10-100× speedup over NumPy after JIT warm-up!

**Step 4: Batch with vmap.**

```python
def single_simulation(initial_pos, initial_vel, masses, dt, n_steps, G=6.67e-8):
    """Run single N-body simulation

    Note: This is a pure function - no mutation!
    """
    pos, vel = initial_pos, initial_vel

    # Use lax.fori_loop for efficient iteration (JIT-friendly)
    def step_fn(i, state):
        p, v = state
        p, v = nbody_step(p, v, masses, dt, G)
        return (p, v)

    final_pos, final_vel = jax.lax.fori_loop(0, n_steps, step_fn, (pos, vel))
    return final_pos, final_vel

# Vectorize over batch dimension!
batched_simulate = vmap(
    single_simulation,
    in_axes=(0, 0, None, None, None, None)  # Batch over IC, not masses/dt/n_steps
)

# Run 1000 different initial conditions simultaneously
results = batched_simulate(
    batch_initial_positions,  # Shape: (1000, n_particles, 3)
    batch_initial_velocities,  # Shape: (1000, n_particles, 3)
    masses,                    # Shape: (n_particles,)
    dt,                        # Scalar
    n_steps,                   # Scalar
    G                          # Scalar
)
# results: tuple of (final_positions, final_velocities)
# Each shape: (1000, n_particles, 3)
```

**Step 5: Differentiate Through Simulation** 🤯

```python
def simulation_observable(initial_pos, masses, dt, n_steps, G=6.67e-8):
    """Run simulation and return scalar observable

    Example: final separation of first two particles
    """
    initial_vel = jnp.zeros_like(initial_pos)
    final_pos, _ = single_simulation(initial_pos, initial_vel, masses, dt, n_steps, G)

    # Observable: distance between particles 0 and 1
    separation = jnp.linalg.norm(final_pos[0] - final_pos[1])
    return separation

# Gradient of observable w.r.t. initial conditions!
grad_observable = jax.grad(simulation_observable, argnums=0)

# Compute sensitivity
initial_pos = jax.random.normal(key, (n_particles, 3))
sensitivity = grad_observable(initial_pos, masses, dt, n_steps, G)

# sensitivity tells you: how does small change in IC affect final separation?
# This enables optimization: find IC that produces desired outcome!
```

**This is mind-blowing**: You can now:

- Optimize initial conditions to produce desired final states
- Compute parameter sensitivities for any observable
- Enable gradient-based control of physical systems

**Real applications**: Trajectory optimization, inverse design, physics-informed learning

#### 3.3 Code Architecture for Packaging 🟡

**Modular Design** (detailed in Part 5):

```bash
nbody_jax/
├── __init__.py           # Package interface
├── physics.py            # Pure physics functions
│   ├── gravitational_potential()
│   ├── compute_forces_loops()      # Pedagogical
│   ├── compute_forces_vectorized() # Fast
│   ├── compute_forces_autodiff()   # Elegant
├── integrators.py        # Integration schemes
│   ├── leapfrog_step()
│   ├── euler_step()      # For comparison
│   ├── rk4_step()        # Higher order
├── simulation.py         # High-level simulation API
│   ├── simulate()        # Single simulation
│   ├── simulate_batch()  # Batched simulations
├── config.py             # Physical constants, defaults
│   ├── G_GRAVITY
│   ├── DEFAULT_DT
└── utils.py              # I/O, visualization (not JAX)
    ├── save_trajectory()
    ├── load_trajectory()
    ├── plot_energy_conservation()
```

**Separation of Concerns**:

- **physics.py**: Pure JAX, fully JIT-able, no I/O
- **integrators.py**: Pure JAX, reusable across problems
- **simulation.py**: Orchestration, still pure functions
- **utils.py**: I/O, plotting (not JIT-compiled, uses matplotlib/h5py)

**Why this matters for Project 5**: Clean architecture makes:

- Testing easier (test each module independently)
- Reuse natural (import what you need)
- Maintenance simple (change one place)
- Extension obvious (where does new code go?)

#### 3.4 Validation 🔴

**Critical**: Ensure JAX version matches NumPy version numerically

```python
def validate_migration():
    """Test that JAX and NumPy implementations agree

    This is essential before claiming "it works"!
    """
    # Same initial conditions
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, shape=(5, 3))
    velocities = jnp.zeros((5, 3))
    masses = jnp.ones(5)
    dt = 0.01
    n_steps = 100

    # Run NumPy version (your Project 2 code)
    np_pos, np_vel = integrate_numpy(
        np.array(positions),
        np.array(velocities),
        np.array(masses),
        dt,
        n_steps
    )

    # Run JAX version (new implementation)
    jax_pos, jax_vel = single_simulation(
        positions,
        velocities,
        masses,
        dt,
        n_steps
    )

    # Compare with reasonable tolerance (numerical precision)
    pos_match = jnp.allclose(jax_pos, np_pos, atol=1e-5, rtol=1e-5)
    vel_match = jnp.allclose(jax_vel, np_vel, atol=1e-5, rtol=1e-5)

    if pos_match and vel_match:
        print("✓ Validation passed! JAX and NumPy agree.")
    else:
        print("✗ Validation failed!")
        print(f"Max position error: {jnp.max(jnp.abs(jax_pos - np_pos))}")
        print(f"Max velocity error: {jnp.max(jnp.abs(jax_vel - np_vel))}")
        raise ValueError("Numerical disagreement between JAX and NumPy!")

    return True
```

**Also validate**:

- Energy conservation (within tolerance for leapfrog)
- Momentum conservation (should be exact for isolated system)
- Angular momentum conservation (should be exact)

**Deliverables**:

- Migration workshop: Transform 3 physics functions (guided practice)
- Performance comparison: Detailed benchmarks (NumPy baseline vs JAX variants)
- Validation tests: Numerical agreement + conservation laws
- Modular code: Organized for packaging (ready for Part 5)

---

### **Part 4A: Optax and Training Loops** 🔴 Essential

**Scope**: Gradient-based optimization foundations
**Effort**: Moderate - essential for ML applications
**Prerequisites**: Part 2 (`grad` mastery)

**Epigraph**:
> *"Machine learning is just optimization with extra steps."* — Anonymous
>
> *"Optimization is the key to everything."* — Stephen Boyd

**Core Question**: How do we build training loops and fit models to data?

:::{admonition} 🔗 Connection to Project 4
:class: note

**You already did gradient-based optimization in HMC!**

HMC uses gradients to propose efficient moves in parameter space. Optax does something similar but for deterministic optimization:

- **HMC**: Stochastic sampling of posterior distribution
- **Gradient descent**: Deterministic search for maximum/minimum
- **Common foundation**: Both use ∇ log p(θ) to navigate parameter space
:::

#### 4A.1 Gradient Descent from Scratch 🔴 (Glass-Box First!)

```python
def gradient_descent(loss_fn, init_params, data, lr, n_steps):
    """Vanilla gradient descent - build it yourself first!

    Args:
        loss_fn: Function computing loss(params, data)
        init_params: Initial parameter values
        data: Training data
        lr: Learning rate (step size)
        n_steps: Number of optimization steps

    Returns:
        params: Optimized parameters
        losses: Loss history
    """
    params = init_params
    losses = []

    for i in range(n_steps):
        # Compute loss and gradient
        loss_val, grad_val = jax.value_and_grad(loss_fn)(params, data)

        # Gradient descent update
        params = params - lr * grad_val  # The essence of optimization!

        losses.append(loss_val)

    return params, jnp.array(losses)
```

**Why vanilla gradient descent fails**:

1. **Sensitive to learning rate**: Too large → diverges, too small → slow
2. **Gets stuck in saddle points**: Gradient is zero but not minimum
3. **No momentum**: Can't escape local minima
4. **Doesn't adapt to parameter scales**: Different params need different step sizes

#### 4A.2 Optax — Professional Optimizers 🔴

**The modern way** (industry standard):

```python
import optax

# Create optimizer with adaptive learning rates
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training loop
for step in range(n_steps):
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, data)

    # Update rule (Optax handles momentum, adaptive rates, etc.)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

**Common optimizers** (from simple to sophisticated):

- `optax.sgd(lr)` — Stochastic gradient descent
- `optax.sgd(lr, momentum=0.9)` — SGD with momentum (much better!)
- `optax.rmsprop(lr)` — Adaptive learning rates per parameter
- `optax.adam(lr)` — **Most popular** - combines momentum + adaptive rates
- `optax.adamw(lr, weight_decay=0.01)` — Adam with weight decay (better generalization)

**Learning rate schedules** (essential for training neural networks):

```python
# Warm up, then decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,        # Start at zero
    peak_value=0.01,       # Warm up to 0.01
    warmup_steps=1000,     # Over 1000 steps
    decay_steps=10000,     # Then decay for 10k steps
    end_value=0.0001       # Down to 0.0001
)

optimizer = optax.adam(learning_rate=schedule)
```

#### 4A.3 Application: Cepheid Period-Luminosity Relation 🟡

**Real astrophysics with Optax** (connects to Module 5 - Bayesian inference!):

```python
# Leavitt's Period-Luminosity relation
# M = slope * log10(P) + intercept

def cepheid_model(params, log_periods):
    """Forward model: predict absolute magnitude from period"""
    slope, intercept = params
    return slope * log_periods + intercept

def chi_squared_loss(params, data):
    """Loss function: χ² for weighted least squares

    This is equivalent to maximizing Gaussian likelihood!
    """
    log_periods, observed_mags, errors = data

    # Predictions
    predicted_mags = cepheid_model(params, log_periods)

    # Weighted residuals
    residuals = (predicted_mags - observed_mags) / errors

    # Chi-squared (sum of squared residuals)
    return jnp.sum(residuals**2)

# Data (example - use real Cepheid data for Project!)
log_periods = jnp.array([0.5, 0.7, 0.9, 1.1, 1.3])  # log10(P in days)
magnitudes = jnp.array([-3.2, -3.8, -4.3, -4.9, -5.4])  # Absolute V magnitude
errors = jnp.array([0.1, 0.1, 0.15, 0.15, 0.2])  # Measurement errors
data = (log_periods, magnitudes, errors)

# Initial guess (from Leavitt's original work)
params = jnp.array([-2.5, -4.0])  # (slope, intercept)

# Optimize with Optax
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

losses = []
for i in range(2000):
    loss, grads = jax.value_and_grad(chi_squared_loss)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)

print(f"Optimal parameters: slope={params[0]:.2f}, intercept={params[1]:.2f}")
# Should get close to: slope ≈ -2.43, intercept ≈ -4.05 (modern calibration)
```

**Connection to Module 5 (Bayesian Inference)**:

- Minimizing $χ²$ is equivalent to **Maximum A Posteriori (MAP) estimation** with Gaussian likelihood + flat prior
- Optax finds the posterior mode (most likely parameters)
- Next step (Module 7): Uncertainty quantification via Gaussian Processes

**Deliverables**:

- Implement vanilla gradient descent from scratch
- Use Optax for Cepheid P-L relation fitting
- Compare optimizers: SGD vs Adam vs AdamW (which converges fastest?)
- Learning rate experiments: How does schedule affect convergence?

---

### **Part 4B: JAX Ecosystem Preview** 🟡 Important

**Scope**: Awareness of ML landscape (details deferred to Module 7)
**Effort**: Low - conceptual overview only
**Prerequisites**: Part 4A completed

**Epigraph**:
> *"You don't understand anything until you learn it more than one way."* — Marvin Minsky

**Core Message**: This is a **PREVIEW** — you'll learn these tools deeply in Module 7. For now, just see what's possible and which path excites you for your final project.

:::{admonition} ⚠️ Important Framing
:class: warning

**Do NOT try to master these tools now!**

This part shows you the landscape and helps you choose your final project path. Detailed learning happens in Module 7 after you've completed Project 5 and have training data ready.
:::

#### 4B.1 The Three Paths to Module 7

All three paths use the same JAX foundations you've learned:

- `jit` for compilation speed
- `grad` for training (automatic differentiation)
- `vmap` for batching data
- Optax for optimization

**Path 1: Gaussian Process Emulators**
*Fast surrogate models with uncertainty quantification*

**Concept**: Learn mapping (inputs) → (outputs) from expensive simulations

```python
# Conceptual example (API details will change - focus on the idea!)

# Your N-body simulator: expensive (seconds per run)
def expensive_nbody(initial_conditions):
    return simulate_physics(initial_conditions)  # Slow!

# Step 1: Generate training data (do once, expensive)
n_train = 100
X_train = sample_initial_conditions(n_train)
y_train = vmap(expensive_nbody)(X_train)  # 100 expensive simulations

# Step 2: Train Gaussian Process (fast - learn the pattern)
# GP learns: "what function maps inputs to outputs?"
gp_model = train_gaussian_process(X_train, y_train)

# Step 3: Instant predictions on new data (microseconds!)
X_test = new_initial_conditions(n_test=10000)
predictions = gp_model.predict(X_test)  # No N-body needed!

# Bonus: GP gives uncertainty estimates (how confident is the prediction?)
mean, std = gp_model.predict_with_uncertainty(X_test)
```

**Use cases**:

- Parameter inference (use GP as cheap likelihood)
- Sensitivity analysis (explore parameter space quickly)
- Uncertainty quantification (confidence intervals on predictions)

---

**Path 2: Neural Network Surrogates**
*Deep learning for complex non-linear mappings*

**Concept**: Train neural network to approximate expensive simulations

```python
# Conceptual example

# Define neural network (Equinox makes this feel like JAX!)
model = neural_network(input_size=10, output_size=1, hidden_layers=[64, 64, 64])

# Training data (same as GP path)
X_train = sample_initial_conditions(1000)
y_train = vmap(expensive_nbody)(X_train)

# Train with Optax (you already know this!)
optimizer = optax.adam(learning_rate=1e-3)
for epoch in range(1000):
    loss, grads = compute_loss_and_gradients(model, X_train, y_train)
    model = update_parameters(model, grads, optimizer)

# Predictions
predictions = model(X_test)  # Fast!
```

**Use cases**:

- High-dimensional inputs (100s of parameters)
- Complex non-linear relationships
- When you have lots of training data (1000s of simulations)

---

**Path 3: Neural ODEs**
*Learn the dynamics directly - the most advanced path*

**Concept**: Instead of learning (state → outcome), learn the dynamics (state → derivative)

```python
# Conceptual example

# Neural network defines dynamics: dx/dt = NN(x)
class NeuralODE:
    def __init__(self):
        self.network = neural_network(input_size=6, output_size=6)  # (pos, vel)

    def dynamics(self, state, t):
        """Returns dx/dt given current state"""
        return self.network(state)

    def solve(self, initial_state, t0, t1):
        """Integrate ODE from t0 to t1"""
        return ode_solver(self.dynamics, initial_state, t0, t1)

# Train on trajectories (not just final states!)
trajectories = run_many_nbody_simulations()  # Expensive
neural_ode = train_on_trajectories(trajectories)

# Now neural ODE has learned the physics!
prediction = neural_ode.solve(new_initial_state, t0=0, t1=10)
```

**Use cases**:

- Time-series forecasting
- Learning unknown physics from data
- Control problems (optimize trajectory)

#### 4B.2 How to Choose Your Path?

**Questions to guide your final project choice**:

1. **How much training data can you generate?**
   - Limited (10-100): GP Emulator
   - Moderate (100-1000): Either GP or NN
   - Lots (1000+): Neural Network or Neural ODE

2. **Do you care about uncertainty quantification?**
   - Yes, critical: GP Emulator (gives confidence intervals)
   - Nice to have: Neural Network with dropout
   - Not important: Any path works

3. **Is the mapping simple or complex?**
   - Simple (quasi-linear): GP Emulator
   - Complex (highly non-linear): Neural Network
   - Unknown (learn from data): Neural ODE

4. **What excites you most?**
   - Statistics/uncertainty: GP path
   - Deep learning/scaling: Neural Network path
   - Dynamical systems/control: Neural ODE path

**All paths are valuable** - choose based on your interests and project goals!

**Deliverables**:

- Ecosystem exploration: Read documentation for one tool from each path
- Toy example: Run one minimal example for your chosen path (provided)
- Final project brainstorm: Write 1-page proposal - which path and why?
- Reflection: How does this connect to your research interests?

---

### **Part 5: Professional JAX — From Scripts to Scientific Software** 🔴 Essential

**Scope**: Production-quality software engineering
**Effort**: High - professional practices take time but pay off
**Prerequisites**: Parts 1-4 completed

**Epigraph**:
> *"Programs are meant to be read by humans and only incidentally for computers to execute."* — Donald Knuth
>
> *"The competent programmer is fully aware of the limited size of his own skull."* — Edsger W. Dijkstra

**Core Question**: How do we write JAX code that other scientists (including future you) can use and trust?

**This is "Becoming a Scientific Software Engineer"**: Not just "make it work" but "make it professional."

:::{admonition} 🔗 Why Professional Practices Matter
:class: note

**Research reproducibility crisis**:

- Many published results can't be reproduced
- Often due to undocumented code, untested software, missing dependencies

**Your advantage**:

- Package your N-body code → others can `pip install` and use it
- Write tests → prove your code works correctly
- Generate versioned data → reproducible ML experiments
- Document thoroughly → future you (6 months from now) will thank present you!

**This makes you employable**: Industry and national labs need scientists who can write production-quality code.
:::

#### 5.1 Debugging JAX Code 🔴

**The Challenges** (JAX is amazing but debugging is different from NumPy):

- Cryptic error messages (stack traces through JAX internals - confusing!)
- Can't use `print()` inside `@jit` (tracing mode limitations)
- Abstract values during compilation (not actual numbers)
- Errors appear far from their source (deferred execution)

**The Solutions**:

**1. Debug printing inside JIT**:

```python
@jit
def f(x):
    jax.debug.print("x = {x}", x=x)  # Works inside jit!
    result = x**2
    jax.debug.print("result = {result}", result=result)
    return result
```

**2. Temporarily disable JIT** (most powerful debugging technique):

```python
# Turn off JIT globally
with jax.disable_jit():
    result = buggy_function(x)  # Now runs normally, can use print()/pdb
```

**3. Check for NaN/Inf automatically**:
```python
# Enable NaN detection (slow! only for debugging)
jax.config.update("jax_debug_nans", True)

result = compute_something(x)  # Will raise error if NaN appears
```

**4. Common errors and fixes**:

**ConcretizationError** (trying to use abstract values as Python bools):

```python
# Problem: Python control flow with traced values
@jit
def bad(x):
    if x > 0:  # Error! Can't convert traced value to bool
        return x**2
    else:
        return x**3

# Solution: Use jnp.where for conditionals
@jit
def good(x):
    return jnp.where(x > 0, x**2, x**3)

# Solution 2: Use lax.cond for complex branches
@jit
def better(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x**2,  # True branch
        lambda x: x**3,  # False branch
        x
    )
```

**Shape mismatches in vmap**:
```python
# Problem: vmap expects consistent batching
f = vmap(my_func, in_axes=(0, 0))  # Both args batched
f(x_batched, y_scalar)  # Error! Second arg not batched

# Solution: Specify which axes are batched
f = vmap(my_func, in_axes=(0, None))  # Only first arg batched
f(x_batched, y_scalar)  # Works!
```

**Random key reuse** (deterministic but wrong!):
```python
# Problem: Reusing keys gives identical results
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
y = jax.random.normal(key, (10,))  # y == x! Deterministic but wrong!

# Solution: Always split keys
key = jax.random.PRNGKey(0)
key, subkey1 = jax.random.split(key)
x = jax.random.normal(subkey1, (10,))
key, subkey2 = jax.random.split(key)
y = jax.random.normal(subkey2, (10,))  # y != x, as intended
```

#### 5.2 Performance Optimization 🟡

**The Low-Hanging Fruit** (in order of importance):

1. ✅ **JIT everything hot** (repeated calls in tight loops)
   - Don't JIT one-time setup functions
   - Profile first to find hot spots

2. ✅ **Use vmap instead of Python loops**
   - 10-100× speedup typical
   - Works even when loop is "vectorizable" in NumPy

3. ✅ **Let XLA optimize** (don't micromanage)
   - Write clear code, let compiler fuse operations
   - Trust the optimizer - it's smarter than hand-tuning

4. ✅ **Minimize host-device transfers**
   - Keep data on device (GPU/TPU) between operations
   - Only transfer results back when needed

**Memory Management with `lax.scan`** (essential for long simulations):

```python
# Problem: Storing all timesteps uses O(n_steps) memory
def simulate_all_steps(initial_state, n_steps):
    states = [initial_state]  # Growing list - bad for memory!
    state = initial_state
    for i in range(n_steps):
        state = physics_step(state)
        states.append(state)
    return jnp.stack(states)  # Convert list to array

# Solution 1: Use lax.scan for constant memory
def simulate_scan(initial_state, n_steps):
    def step_fn(carry, i):
        state = carry
        new_state = physics_step(state)
        return new_state, new_state  # (carry, output)

    final_state, all_states = jax.lax.scan(
        step_fn,
        initial_state,
        jnp.arange(n_steps)
    )
    return all_states

# Solution 2: Only save final state (O(1) memory)
def simulate_final_only(initial_state, n_steps):
    def step_fn(i, state):
        return physics_step(state)

    final_state = jax.lax.fori_loop(0, n_steps, step_fn, initial_state)
    return final_state
```

**Profiling with JAX** (find actual bottlenecks):

```python
# Chrome tracing (visualize in chrome://tracing)
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    result = expensive_computation(x)

# Then open chrome://tracing and load /tmp/jax-trace file
# Shows: where time is spent, memory usage, device utilization
```

**Float precision** (trade-off between speed and accuracy):

```python
# JAX defaults to float32 (faster, less memory, standard for ML)
x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # float32

# Enable float64 for scientific computing (if needed)
jax.config.update("jax_enable_x64", True)
x = jnp.array([1.0, 2.0, 3.0])
print(x.dtype)  # float64

# Physics note: N-body usually fine with float32 for ~1000 steps
# For long-term (millions of steps), use float64 to prevent accumulation
```

#### 5.3 Package Structure 🔴 (Critical for Project 5!)

**From Jupyter Notebook → Python Package**:

```
nbody_jax/                    # Your package root!
├── pyproject.toml            # Package metadata + dependencies
├── README.md                 # Usage instructions, examples
├── LICENSE                   # BSD-3-Clause or MIT (open source)
├── .gitignore                # Don't commit __pycache__, .ipynb_checkpoints
├── nbody_jax/                # Source code directory
│   ├── __init__.py           # Package interface (what users import)
│   ├── physics.py            # Potential energy, force calculations
│   ├── integrators.py        # Leapfrog, RK4, Euler
│   ├── simulation.py         # High-level simulation API
│   ├── config.py             # Physical constants, default parameters
│   └── utils.py              # I/O, visualization (not JAX/JIT)
├── tests/                    # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_physics.py       # Test force calculations
│   ├── test_integrators.py  # Test energy conservation
│   └── test_simulation.py   # Test end-to-end workflows
├── examples/                 # Tutorial notebooks
│   ├── quickstart.ipynb      # 5-minute intro
│   ├── basic_usage.ipynb     # Common use cases
│   └── advanced_usage.ipynb  # Batching, autodiff, optimization
└── docs/                     # Documentation (optional but nice)
    └── API.md                # Function reference
```

**`pyproject.toml`** (modern Python packaging):

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nbody-jax"
version = "0.1.0"
description = "Fast, differentiable N-body simulations in JAX"
authors = [{name = "Your Name", email = "you@email.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"

dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "ruff",
]
viz = [
    "matplotlib>=3.5",
    "h5py",
]

[project.urls]
Homepage = "https://github.com/yourusername/nbody-jax"
Repository = "https://github.com/yourusername/nbody-jax"
```

**Installation for development**:
```bash
cd nbody_jax/
pip install -e .           # Editable mode (changes reflected immediately)
pip install -e ".[dev]"    # Also install dev tools (pytest, etc.)
pip install -e ".[viz]"    # Also install visualization tools
```

**Clean API Design** (`nbody_jax/__init__.py`):

```python
"""nbody-jax: Fast, differentiable N-body simulations in JAX

Example:
    >>> import nbody_jax as nbody
    >>> result = nbody.simulate(initial_conditions, dt=0.01, n_steps=1000)
    >>> results = nbody.simulate_batch(batch_of_ICs, dt=0.01, n_steps=1000)
"""

from .simulation import simulate, simulate_batch
from .integrators import leapfrog_step, euler_step, rk4_step
from .physics import gravitational_potential, compute_forces_autodiff

__all__ = [
    # High-level API (most users use these)
    "simulate",
    "simulate_batch",
    # Lower-level (for advanced users)
    "leapfrog_step",
    "euler_step",
    "rk4_step",
    "gravitational_potential",
    "compute_forces_autodiff",
]

__version__ = "0.1.0"
```

**Users can now**:
```python
import nbody_jax as nbody

# Simple API
result = nbody.simulate(initial_conditions, dt=0.01, n_steps=1000)

# Batch API (vmap under the hood)
results = nbody.simulate_batch(batch_of_1000_ICs, dt=0.01, n_steps=1000)

# Advanced: use lower-level components
from nbody_jax import leapfrog_step, gravitational_potential
```

#### 5.4 Testing JAX Code 🔴

**Why test?** (scientific imperative!)
- **Catch bugs early** (before they contaminate results)
- **Enable refactoring safely** (change code without breaking things)
- **Document expected behavior** (tests are executable specifications)
- **Build trust** (yours and others' confidence in your code)

**Testing with pytest**:

```python
# tests/test_physics.py
import pytest
import jax
import jax.numpy as jnp
from nbody_jax import gravitational_potential, compute_forces_autodiff

def test_potential_two_body():
    """Two equal masses at distance r should give U = -G*m²/r"""
    positions = jnp.array([[0., 0., 0.], [1., 0., 0.]])
    masses = jnp.array([1.0, 1.0])
    G = 1.0

    U = gravitational_potential(positions, masses, G)
    expected = -G * 1.0**2 / 1.0  # = -1.0

    assert jnp.isclose(U, expected), f"Got U={U}, expected {expected}"

def test_forces_opposite():
    """Newton's 3rd law: F_12 = -F_21"""
    positions = jnp.array([[0., 0., 0.], [1., 0., 0.]])
    masses = jnp.array([1.0, 1.0])
    G = 1.0

    forces = compute_forces_autodiff(positions, masses, G)

    # Forces should be equal and opposite
    assert jnp.allclose(forces[0], -forces[1]), \
        f"F_0={forces[0]}, F_1={forces[1]} not opposite!"

def test_forces_radial():
    """Gravitational force should point along line connecting masses"""
    positions = jnp.array([[0., 0., 0.], [1., 2., 3.]])
    masses = jnp.array([1.0, 1.0])
    G = 1.0

    forces = compute_forces_autodiff(positions, masses, G)
    r_vec = positions[1] - positions[0]

    # Force on particle 0 should be parallel to r_vec
    # Check via cross product (should be zero for parallel vectors)
    cross = jnp.cross(forces[0], r_vec)
    assert jnp.allclose(cross, 0.0, atol=1e-6), \
        "Force not radial!"

def test_energy_conservation():
    """Leapfrog integrator should conserve total energy"""
    from nbody_jax import simulate

    # Initial conditions: two-body circular orbit
    positions = jnp.array([[0., 0., 0.], [1., 0., 0.]])
    # Circular orbit velocity: v = sqrt(G*M/r)
    v_circ = jnp.sqrt(1.0 * 1.0 / 1.0)
    velocities = jnp.array([[0., v_circ/2, 0.], [0., -v_circ/2, 0.]])
    masses = jnp.array([1.0, 1.0])

    # Compute initial energy
    U_init = gravitational_potential(positions, masses, G=1.0)
    K_init = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    E_init = K_init + U_init

    # Simulate for 100 steps
    final_pos, final_vel = simulate(positions, velocities, masses, dt=0.01, n_steps=100)

    # Compute final energy
    U_final = gravitational_potential(final_pos, masses, G=1.0)
    K_final = 0.5 * jnp.sum(masses[:, None] * final_vel**2)
    E_final = K_final + U_final

    # Energy should be conserved to ~0.01% for leapfrog
    rel_error = jnp.abs(E_final - E_init) / jnp.abs(E_init)
    assert rel_error < 1e-4, \
        f"Energy not conserved! ΔE/E = {rel_error:.2e}"
```

**Testing numerical accuracy** (beyond correctness - validate physics!):

```python
def test_symplectic_property():
    """Leapfrog should preserve phase space volume (det(Jacobian) = 1)"""
    from jax import jacobian
    from nbody_jax.integrators import leapfrog_step

    # Initial state
    pos = jnp.array([[0., 0., 0.], [1., 0., 0.]])
    vel = jnp.array([[0., 0.5, 0.], [0., -0.5, 0.]])
    state = (pos, vel)
    masses = jnp.array([1.0, 1.0])
    dt = 0.01

    # Flatten state for Jacobian computation
    def step_flat(state_flat):
        n = len(masses)
        pos = state_flat[:n*3].reshape((n, 3))
        vel = state_flat[n*3:].reshape((n, 3))
        new_pos, new_vel = leapfrog_step(pos, vel, masses, dt)
        return jnp.concatenate([new_pos.flatten(), new_vel.flatten()])

    state_flat = jnp.concatenate([pos.flatten(), vel.flatten()])

    # Compute Jacobian matrix
    J = jacobian(step_flat)(state_flat)

    # Symplectic integrator preserves volume: det(J) = 1
    det_J = jnp.linalg.det(J)
    assert jnp.isclose(det_J, 1.0, atol=1e-6), \
        f"Not symplectic! det(J) = {det_J} ≠ 1"

def test_gradient_accuracy():
    """Autodiff gradients should match finite differences"""
    from jax import grad
    from nbody_jax import gravitational_potential

    # Test setup
    positions = jnp.array([[0., 0., 0.], [1., 0.5, 0.2], [0.5, 1., 0.3]])
    masses = jnp.array([1.0, 0.8, 1.2])
    G = 1.0

    # Autodiff gradient
    grad_fn = grad(gravitational_potential, argnums=0)
    grad_autodiff = grad_fn(positions, masses, G)

    # Finite difference gradient (slow but trusted)
    h = 1e-5
    grad_fd = jnp.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(3):
            pos_plus = positions.at[i, j].add(h)
            pos_minus = positions.at[i, j].add(-h)
            U_plus = gravitational_potential(pos_plus, masses, G)
            U_minus = gravitational_potential(pos_minus, masses, G)
            grad_fd = grad_fd.at[i, j].set((U_plus - U_minus) / (2*h))

    # Should agree to numerical precision
    assert jnp.allclose(grad_autodiff, grad_fd, rtol=1e-4, atol=1e-6), \
        "Autodiff and finite differences disagree!"
```

**Testing stochastic functions** (handle random keys properly):

```python
def test_random_reproducibility():
    """Same key → same result (deterministic)"""
    from nbody_jax.utils import sample_initial_conditions  # Hypothetical

    key = jax.random.PRNGKey(42)

    result1 = sample_initial_conditions(key, n=10)
    result2 = sample_initial_conditions(key, n=10)

    assert jnp.allclose(result1, result2), \
        "Same key should give identical results!"

def test_random_different_keys():
    """Different keys → different results"""
    from nbody_jax.utils import sample_initial_conditions

    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    result1 = sample_initial_conditions(key1, n=10)
    result2 = sample_initial_conditions(key2, n=10)

    assert not jnp.allclose(result1, result2), \
        "Different keys should give different results!"
```

**Run tests**:
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=nbody_jax tests/

# Run specific test
pytest tests/test_physics.py::test_energy_conservation -v

# Run tests in parallel (fast!)
pytest -n auto tests/
```

#### 5.5 Data Generation Workflows 🔴 (Critical for Final Projects!)

**The Goal**: Generate reproducible training data for machine learning (Module 7)

**Pipeline**:
```
1. Sample parameter space (initial conditions, masses, etc.)
2. Batch simulate with vmap (leverage JAX speed!)
3. Extract observables (what you want ML to predict)
4. Save efficiently (HDF5 for large arrays)
5. Version and document (reproducibility!)
```

**Complete Example: Generate 10,000 N-body Trajectories**

```python
import h5py
import jax
import jax.numpy as jnp
from nbody_jax import simulate_batch
from datetime import datetime

def generate_training_data(
    n_samples=10000,
    n_particles=10,
    dt=0.01,
    n_steps=1000,
    seed=42,
    output_file='nbody_training_data.h5'
):
    """Generate large-scale N-body training dataset

    Args:
        n_samples: Number of different initial conditions
        n_particles: Particles per simulation
        dt: Integration timestep
        n_steps: Steps per simulation
        seed: Random seed for reproducibility
        output_file: Where to save data
    """

    print(f"Generating {n_samples} N-body simulations...")
    print(f"Each: {n_particles} particles, {n_steps} steps, dt={dt}")

    # 1. Sample parameter space
    key = jax.random.PRNGKey(seed)

    # Random initial positions (Gaussian distribution)
    key, subkey = jax.random.split(key)
    initial_positions = jax.random.normal(subkey, (n_samples, n_particles, 3))

    # Random initial velocities (smaller variance)
    key, subkey = jax.random.split(key)
    initial_velocities = 0.1 * jax.random.normal(subkey, (n_samples, n_particles, 3))

    # Random masses (IMF sampling - power law)
    key, subkey = jax.random.split(key)
    # Sample from power-law: dN/dm ∝ m^(-2.35) (Salpeter IMF)
    uniform = jax.random.uniform(subkey, (n_samples, n_particles))
    masses = (0.1 + 9.9 * uniform) ** (-1/2.35)  # Masses between 0.1-10 solar masses
    masses = masses / jnp.sum(masses, axis=1, keepdims=True)  # Normalize to unit total mass

    # 2. Batch simulate (THIS IS WHERE JAX SHINES!)
    print("Running batch simulation (this will take a few minutes)...")
    import time
    t0 = time.time()

    final_positions, final_velocities = simulate_batch(
        initial_positions,
        initial_velocities,
        masses,
        dt=dt,
        n_steps=n_steps
    )

    t1 = time.time()
    print(f"Simulation complete! Took {t1-t0:.1f} seconds")
    print(f"That's {n_samples/(t1-t0):.1f} simulations per second!")

    # 3. Extract observables (what ML will predict)
    print("Computing observables...")

    # Virial ratio: 2K / |W| (should be ~1 for virial equilibrium)
    def compute_virial_ratio(pos, vel, masses):
        # Kinetic energy
        K = 0.5 * jnp.sum(masses[:, None] * vel**2)
        # Potential energy
        U = gravitational_potential(pos, masses)
        return 2 * K / jnp.abs(U)

    # Half-mass radius
    def compute_half_mass_radius(pos, masses):
        center = jnp.sum(pos * masses[:, None], axis=0) / jnp.sum(masses)
        distances = jnp.linalg.norm(pos - center, axis=1)
        sorted_idx = jnp.argsort(distances)
        cumulative_mass = jnp.cumsum(masses[sorted_idx])
        half_mass_idx = jnp.searchsorted(cumulative_mass, 0.5 * jnp.sum(masses))
        return distances[sorted_idx[half_mass_idx]]

    # Velocity dispersion
    def compute_velocity_dispersion(vel, masses):
        v_mean = jnp.sum(vel * masses[:, None], axis=0) / jnp.sum(masses)
        v_rel = vel - v_mean
        return jnp.sqrt(jnp.sum(masses[:, None] * v_rel**2) / jnp.sum(masses))

    # Compute for all simulations (vmap!)
    virial_ratios = jax.vmap(compute_virial_ratio)(final_positions, final_velocities, masses)
    half_mass_radii = jax.vmap(compute_half_mass_radius)(final_positions, masses)
    velocity_dispersions = jax.vmap(compute_velocity_dispersion)(final_velocities, masses)

    # 4. Save to HDF5 (efficient for large arrays)
    print(f"Saving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Inputs (for ML training)
        inputs = f.create_group('inputs')
        inputs.create_dataset('initial_positions', data=initial_positions)
        inputs.create_dataset('initial_velocities', data=initial_velocities)
        inputs.create_dataset('masses', data=masses)

        # Outputs (what ML will predict)
        outputs = f.create_group('observables')
        outputs.create_dataset('virial_ratio', data=virial_ratios)
        outputs.create_dataset('half_mass_radius', data=half_mass_radii)
        outputs.create_dataset('velocity_dispersion', data=velocity_dispersions)

        # Also save final states (useful for Neural ODE training)
        states = f.create_group('final_states')
        states.create_dataset('positions', data=final_positions)
        states.create_dataset('velocities', data=final_velocities)

        # 5. Metadata (crucial for reproducibility!)
        f.attrs['n_samples'] = n_samples
        f.attrs['n_particles'] = n_particles
        f.attrs['dt'] = dt
        f.attrs['n_steps'] = n_steps
        f.attrs['seed'] = seed
        f.attrs['date_generated'] = datetime.now().isoformat()
        f.attrs['jax_version'] = jax.__version__
        f.attrs['nbody_jax_version'] = '0.1.0'  # Your package version
        f.attrs['description'] = 'N-body training data for ML emulator'

    print("✓ Data generation complete!")
    print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")

    return output_file

# Generate data
if __name__ == '__main__':
    output_file = generate_training_data(
        n_samples=10000,
        n_particles=10,
        dt=0.01,
        n_steps=1000,
        seed=42
    )
```

**Loading data for ML** (Module 7 preview):

```python
# Load training data
with h5py.File('nbody_training_data.h5', 'r') as f:
    # Inputs: initial conditions
    X_positions = f['inputs/initial_positions'][:]
    X_velocities = f['inputs/initial_velocities'][:]
    X_masses = f['inputs/masses'][:]

    # Combine into single feature array (flatten)
    n_samples = len(X_positions)
    n_particles = X_positions.shape[1]
    X = jnp.concatenate([
        X_positions.reshape(n_samples, -1),
        X_velocities.reshape(n_samples, -1),
        X_masses
    ], axis=1)
    # Shape: (n_samples, n_particles * 7)
    # 7 = 3 (pos) + 3 (vel) + 1 (mass) per particle

    # Target: what we want to predict
    y = f['observables/virial_ratio'][:]  # Shape: (n_samples,)

    print(f"Loaded {len(X)} training examples")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Metadata
    print(f"\nDataset info:")
    print(f"Generated: {f.attrs['date_generated']}")
    print(f"Seed: {f.attrs['seed']}")
    print(f"Description: {f.attrs['description']}")

# Now ready for Module 7: train GP/NN to learn X → y!
```

**Data versioning best practices**:

```bash
# Tag dataset with git commit
git rev-parse HEAD > data_git_commit.txt

# Document in filename
mv nbody_training_data.h5 nbody_training_v1_10k_samples_seed42.h5

# Keep metadata file
cat > nbody_training_v1_README.md <<EOF
# N-body Training Data v1

Generated: 2025-11-06
Samples: 10,000
Particles: 10
Integration: dt=0.01, n_steps=1000
Code version: nbody-jax v0.1.0 (git commit: abc123)
Purpose: GP emulator training for Project 5
EOF
```

#### 5.6 Project 5 Roadmap 🔴

**Recommended Timeline** (2 weeks total):

**Week 1: Refactor & Test**
- Days 1-2: Migrate N-body to JAX (apply Part 3 patterns)
- Days 3-4: Structure as package (implement Part 5.3 architecture)
- Day 5: Write tests, validate numerically (energy conservation, etc.)

**Week 2: Optimize & Generate**
- Days 6-7: Profile and optimize performance (JIT, vmap, benchmarks)
- Days 8-9: Generate training data (10k simulations, save as HDF5)
- Day 10: Documentation, README, final submission

**Deliverables for Project 5** (checklist):
- [ ] `nbody_jax` package (pip-installable with `pip install -e .`)
- [ ] Test suite (pytest, ≥80% coverage, all tests passing)
- [ ] Performance benchmarks (vs NumPy: >10× speedup documented)
- [ ] Training data (HDF5, 10k simulations, versioned and documented)
- [ ] Documentation (README with examples, API reference)
- [ ] Reflection essay (500 words): "What I learned about software engineering"

**What you'll have at the end**:
- ✅ Professional scientific software (portfolio piece!)
- ✅ Training data ready for Module 7 (GP/NN experiments)
- ✅ Deep understanding of JAX (not just surface knowledge)
- ✅ Foundation for final project (differentiable physics simulator)

**Deliverables for Part 5** (practice before Project 5):
- Debugging workshop: Fix 10 broken JAX functions (common error patterns)
- Performance analysis: Profile your code, identify and optimize bottlenecks
- Package setup: Convert example code to package structure (template provided)
- Testing workshop: Write tests for 5 physics functions (pytest practice)
- Data generation: Create small dataset (100 samples, validate pipeline)

---

### **Part 6: Synthesis — The Transformation** 🔴 Essential

**Scope**: Integration, reflection, and looking ahead
**Effort**: Moderate - metacognitive synthesis
**Prerequisites**: All parts completed

**Epigraph**:
> *"The whole is greater than the sum of its parts."* — Aristotle
>
> *"Everything should be made as simple as possible, but not simpler."* — Albert Einstein

**Core Question**: How has JAX transformed the way we think about scientific computing?

#### 6.1 What We've Learned: The JAX Revolution

**Before Module 6**, you wrote **scripts**:
- ❌ Compute gradients manually (finite differences - slow, approximate)
- ❌ Loop over data points (Python loops - painfully slow)
- ❌ NumPy only (CPU-bound, no compilation)
- ❌ Scripts that "work" but aren't reusable or testable

**After Module 6**, you build **software**:
- ✅ Automatic differentiation (exact derivatives, essentially free)
- ✅ Auto-vectorization (batch everything with vmap)
- ✅ JIT compilation (10-100× speedups from XLA)
- ✅ Tested, packaged, documented code (professional quality)

**This isn't just faster code. It's a paradigm shift.**

From **procedural scripting** → **functional software engineering**
From **approximate methods** → **exact mathematical transformations**
From **one-off scripts** → **reusable scientific software**

#### 6.2 Universal Patterns Across the Course

**The same mathematical structures everywhere**:

| Module | What We Compute | JAX Transformation Used |
|--------|-----------------|-------------------------|
| **Module 1 (Stats)** | Random variables from distributions | `jax.random` with explicit keys |
| **Project 2 (N-body)** | Particles in phase space | `jit` + `vmap` for fast dynamics |
| **Project 3 (Monte Carlo)** | Photons through stellar atmospheres | `vmap` for batch sampling |
| **Project 4 (MCMC)** | Parameters via MCMC/HMC | `grad` for exact posterior gradients |
| **Module 6** | Neural network weights | `grad` + Optax for training |
| **Module 7** | GP hyperparameters, NN training | All transformations combined! |

**The profound unity**: Whether you're:
- Simulating gravitational dynamics (Module 2-3)
- Running Markov Chain Monte Carlo (Module 5)
- Training neural networks (Module 7)

**You're using variations of the same tools**: differentiation, vectorization, compilation.

**This is why JAX is powerful**: One unified framework, infinite applications.

#### 6.3 From Scripts to Scientific Software Engineering

**You've learned to**:
- ✅ Write pure, functional code (composable, testable)
- ✅ Use automatic differentiation (gradients for free)
- ✅ Batch and vectorize naturally (10-100× speedups)
- ✅ Compile for performance (JIT → machine code)
- ✅ Structure code as packages (pip-installable libraries)
- ✅ Test systematically (pytest, >80% coverage)
- ✅ Generate reproducible data (versioned, documented HDF5)
- ✅ Document for others and future you (README, docstrings, examples)

**You're not just a "coder" anymore. You're a scientific software engineer.**

**This distinction matters**:
- **Coder**: "Does it run?" → Ships when it produces output
- **Engineer**: "Is it correct? Tested? Maintainable?" → Ships when it's trusted

#### 6.4 Looking Ahead: Module 7 and Final Projects

**Module 7: Machine Learning Foundations**

You'll learn:
- **Gaussian Processes**: Uncertainty quantification for surrogate models
- **Neural Networks**: Deep learning from scratch in JAX/Equinox
- **Physics-Informed Learning**: Combining data-driven + physics-based approaches

**Prerequisites (you now have them!)**:
- ✅ JAX fluency (`jit`, `grad`, `vmap`, `pmap`)
- ✅ Optax for optimization (Adam, learning rate schedules)
- ✅ Working N-body package (differentiable physics simulator!)
- ✅ Training data generated (10k simulations ready to use)

**Final Projects: Physics-Informed Learning on Star Clusters**

**Three paths, one foundation** (all use your JAX N-body package):

**Path 1: GP Emulator** (Uncertainty quantification)
- Train GP on 10k N-body simulations
- Learn mapping: (initial conditions) → (virial ratio, half-mass radius, etc.)
- Instant predictions with confidence intervals
- **Application**: Parameter inference, sensitivity analysis

**Path 2: Neural Network Surrogate** (High-dimensional regression)
- Train neural network on same data
- Learn complex, non-linear relationships
- Scale to high-dimensional parameter spaces
- **Application**: Fast emulation for optimization

**Path 3: Neural ODE** (Learning dynamics)
- Learn the dynamics directly: train NN to predict $d\mathbf{r}/dt$
- Integrate learned dynamics with Diffrax
- Physics-informed: incorporate known conservation laws
- **Application**: Forecasting, trajectory optimization, control

**The beautiful part**: All three paths use the same JAX foundations you've mastered.

#### 6.5 Self-Assessment: How Far Have You Come?

**Level 1: Conceptual Understanding** ✓ or ✗

- [ ] I can explain why autodiff enables gradient-based learning (not just "it's automatic")
- [ ] I understand pure functions and why mutation breaks JAX transformations
- [ ] I know when to use `jit` vs `grad` vs `vmap` (and can explain trade-offs)
- [ ] I can articulate the glass-box philosophy (understanding over using)

**Level 2: Technical Skills** ✓ or ✗

- [ ] I can write JAX code from scratch without copying tutorials
- [ ] I can debug ConcretizationErrors and other JAX-specific issues
- [ ] I can systematically migrate NumPy code to JAX (following the 7-step checklist)
- [ ] I can profile and optimize JAX performance (chrome://tracing, lax.scan)

**Level 3: Software Engineering** ✓ or ✗

- [ ] I can structure code as a Python package (pyproject.toml, __init__.py)
- [ ] I can write comprehensive tests for scientific code (pytest, physics validation)
- [ ] I can generate and manage training data (HDF5, versioning, documentation)
- [ ] I can document code professionally (README, docstrings, examples)

**Level 4: Connections and Synthesis** ✓ or ✗

- [ ] I see how JAX connects to Project 4 (HMC gradients - now automatic!)
- [ ] I understand how this enables Module 7 (ML requires autodiff + batching)
- [ ] I can explain the transformation from scripts to research-grade software
- [ ] I'm excited to build my final project using these tools!

**Honest reflection**: Which levels feel solid? Which need more work?

**Use Project 5 to cement weak areas.** By the end, all boxes should be ✓.

#### 6.6 The Path Forward

**You've completed Module 6.** This is what comes next:

**→ Project 5 (2 weeks)**:
- Transform understanding into software
- Build professional JAX package (nbody-jax)
- Generate training data (10k simulations)
- Create portfolio piece (show employers!)

**→ Module 7 (3-4 weeks)**:
- Gaussian Processes (uncertainty quantification)
- Neural Networks (deep learning from scratch)
- Physics-Informed Learning (data + physics)

**→ Final Project (4 weeks)**:
- Novel research combining everything:
  - Your N-body package (Project 5)
  - Your training data (Project 5)
  - Your ML models (Module 7)
  - Your inference skills (Module 5)
  - Your statistical thinking (Module 1)
- Publication-quality results
- Portfolio piece for job market

**You've built the foundation. Now you'll build the science.**

#### 6.7 Reflective Questions for Growth Memo 🆕

As you write your "From Scripts to Software" reflection essay, consider:

**1. Conceptual Shift**:
How has your understanding of "computing gradients" changed from Project 4 (manual finite differences) to Module 6 (automatic differentiation)? What does it mean that gradients are "free"? What new capabilities does this unlock?

**2. Functional Programming Struggle**:
What was the hardest part of shifting from imperative (NumPy mutations) to functional (JAX immutability) programming? Give a specific example where you struggled. How did you overcome it? Would you now choose functional style even when not using JAX?

**3. Glass-Box Understanding**:
Can you explain (in your own words, to a peer who hasn't taken this course) how automatic differentiation works? Not just "JAX does it" but the actual mechanism - computational graphs, chain rule, reverse mode. Could you implement a simple autodiff system from scratch?

**4. Software Engineering Mindset**:
How did packaging your code change how you think about research software? Would you go back to Jupyter notebooks and one-off scripts, or is this your new standard? What's the cost vs. benefit of professional practices (testing, documentation, packaging)?

**5. Looking Forward**:
Which JAX ecosystem path (GP/NN/NeuralODE) excites you most for your final project? Why? What scares you about it? What do you need to learn between now and then to be ready?

**6. Struggle and Growth**:
What was the moment of greatest frustration in this module? How did you work through it? What did you learn about yourself as a learner? How will you approach future technical challenges differently?

**7. The Bigger Picture**:
How does learning JAX change your view of what's possible in computational astrophysics? What research problems that seemed impossible a month ago now feel tractable?

**These questions should guide your reflection essay** (1000 words, Part 6 deliverable).

**Deliverables for Part 6**:
- Completed self-assessment checklist (4 levels, honest evaluation)
- Reflection essay (1000 words): "From Scripts to Software — My Transformation"
  - Use the 7 reflective questions as prompts
  - Include specific examples from your work
  - Discuss both successes and struggles
- Final project proposal (1 page):
  - Which path (GP/NN/NeuralODE) and why?
  - What scientific question will you investigate?
  - What makes this novel or interesting?
- Project 5 planning document:
  - Week-by-week timeline
  - Risk mitigation (what could go wrong?)
  - Success criteria (how will you know you're done?)

---

## Module-Wide Integration

### The JAX Pyramid

```
                 Final Projects
            (Physics-Informed Learning)
                       ↑
              Module 7 (GP/NN/NeuralODE)
                       ↑
          Project 5 (JAX Package + Data)
                       ↑
       Part 5 (Professional Practices)
                       ↑
    Part 4B (Ecosystem Preview)
                       ↑
    Part 4A (Optax + Training Loops)
                       ↑
       Part 3 (N-body Migration)
                       ↑
     Part 2 (Core Transformations)
                       ↑
    Part 1 (Why JAX? Concepts + Math)
```

**Each level builds on the previous. Skip nothing.**

### Backward Connections (to earlier modules)

- **Module 1 (Statistical Thinking)**: Ensemble statistics (`vmap`), gradient descent (`grad`), random sampling (`jax.random`)
- **Module 2 (Stellar Structure)**: Statistical mechanics → batched particle systems
- **Module 3 (N-body Dynamics)**: Hamiltonian dynamics → differentiable simulation, leapfrog → JAX integrator
- **Project 3 (Monte Carlo)**: Random sampling → JAX random keys, batching photons with `vmap`
- **Module 5 (Bayesian Inference)**: HMC gradients → `jax.grad`, MCMC → stochastic optimization

### Forward Connections (to future work)

- **Project 5**: Apply everything from Module 6 (professional JAX package)
- **Module 7**: Use `jit`, `grad`, `vmap` for GP/NN training (build on Optax foundation)
- **Final Project**: Combine N-body package + ML for novel research (differentiable physics!)

### Running Examples Thread

**The Harmonic Oscillator Family** (pedagogical - simple to understand):
1. Part 1: Simple 1D oscillator (conceptual introduction to autodiff)
2. Part 2: 1D oscillator with all transformations (jit, grad, vmap applied)
3. Part 3: Coupled oscillators (N-body analogy, migration practice)
4. Part 4: Fitting oscillator parameters (optimization with Optax)
5. Part 5: Packaged oscillator module (software engineering practice)

**The N-body Thread** (practical - your actual work):
1. Part 1: Why differentiate through simulations? (motivation)
2. Part 2: Forces via `grad`, ensembles via `vmap` (transformations in action)
3. Part 3: Complete migration from Project 2 (systematic workflow)
4. Part 4A: Optimization example - find IC for target orbit (Optax application)
5. Part 5: Package + generate data (Project 5 preparation)
6. Part 6: Connect to Module 7 (ML emulation, Neural ODEs)

### Assessment Strategy

**Formative (during module)**:
- Conceptual checkpoints after each part
- Code transformation exercises (NumPy → JAX practice)
- Debugging challenges (fix broken code, learn error patterns)
- Performance benchmarking (measure speedups, understand JIT)
- Mini-projects (one per part - hands-on practice)

**Summative (end of module)**:
- Self-assessment rubric (4 levels × 4 dimensions)
- Reflection essay: "Scripts → Software transformation" (1000 words)
- Final project proposal (1 page - which path and why?)
- Project 5 planning document (2-week timeline)

**Project 5 itself** is the ultimate assessment:
- Working JAX package (code quality, modularity)
- Test suite (correctness, >80% coverage)
- Performance benchmarks (documented speedups)
- Training data (10k simulations, properly versioned)
- Documentation (README, examples, API reference)

**Suggested Grading Weights**:
- Conceptual understanding (quizzes, reflections): 20%
- Technical skills (exercises, debugging challenges): 30%
- Software engineering (Project 5 code quality): 25%
- Professional practices (tests, docs, data): 15%
- Synthesis and connections (reflection, self-assessment): 10%

---

## Teaching Notes

### Pedagogical Challenges

**1. Functional Programming Resistance**
- **Challenge**: Students trained in imperative style find mutation natural, immutability strange
- **Strategy**: Physics analogies (conservation laws are immutable!), show performance isn't sacrificed
- **Reassurance**: "Feels weird at first, natural by Project 5's end - trust the process"

**2. Abstract Error Messages**
- **Challenge**: JAX errors propagate through internal machinery - cryptic stack traces
- **Strategy**: Build comprehensive error pattern library, dedicated debugging workshop (Part 5.1)
- **Tool**: Create "Common JAX Errors" cheatsheet (ConcretizationError, shape mismatches, key reuse)

**3. Overwhelming Ecosystem**
- **Challenge**: Too many libraries (Optax, Equinox, GPJax, Diffrax, NumPyro, Lineax...)
- **Strategy**: Part 4B is awareness only - master in Module 7 after having Project 5 foundation
- **Framing**: "See what's possible now, learn details later when you need them"

**4. The Package Structure Leap**
- **Challenge**: From Jupyter notebooks to professional packages is a big conceptual jump
- **Strategy**: Part 5.3 provides detailed template, workshop format with hands-on guidance
- **Support**: Office hours specifically for packaging issues, import problems, directory structure

**5. "Why Not Just Use NumPy?"**
- **Challenge**: Students don't see value until they feel the pain
- **Strategy**: Lead with Project 4 pain point (manual gradient computation was tedious!)
- **Proof**: Performance benchmarks showing 10-100× speedup (seeing is believing)

### Time Management

**If running short**, prioritize:
1. ✅ Part 1 (concepts + autodiff math) — foundation is critical
2. ✅ Part 2 (transformations) — core technical skills
3. ✅ Part 3 (N-body migration) — Project 5 direct preparation
4. ⚠️ Part 4B (ecosystem) — can condense to brief overview in 1 lecture
5. ✅ Part 5 (professional practices) — critical for Project 5 success

**If extra time available**, expand:
- Part 1: Deeper XLA internals (how compilation actually works)
- Part 2: More physics examples per transformation (oscillators, orbits, collisions)
- Part 4B: Hands-on mini-projects with GPJax/Equinox/Diffrax
- Part 5: Advanced testing (property-based testing with Hypothesis, fuzzing)
- New topic: GPU/TPU acceleration with `pmap` (distributed computing)

### Office Hours Topics

**Expect students to struggle with**:
- **Functional thinking** (hardest conceptual shift) - provide imperative → functional translation practice
- **ConcretizationErrors** (most common error) - show many examples, pattern matching
- **Random key management** (forgetting to split) - emphasize "key hygiene" mantra
- **Package structure** (import failures, `__init__.py` mysteries) - walk through step-by-step
- **Test writing** (what to test? how?) - provide examples, start with physics validation

**Common questions**:
- "Why is my vmap slower than a loop?" → Overhead dominates for small arrays; show breakeven point
- "When should I use jit?" → Repeated calls only, show compilation overhead vs speedup
- "How do I profile JAX code?" → Demo chrome://tracing workflow live
- "My package imports fail!" → sys.path issues, editable install troubleshooting

### Extension Activities

**For advanced students**:
- Implement custom VJP rules for special functions
- Explore `pmap` for multi-GPU parallelization
- Build custom Optax optimizer (learning rate schedule engineering)
- Integrate C++/Fortran legacy code with JAX (foreign function interface)
- Read original JAX papers (autodiff theory, XLA compiler architecture)
- Contribute to open-source JAX ecosystem (documentation, bug fixes, examples)

### Historical Context to Weave In

**The Autodiff Revolution**:
- 1960s: Robert Wengert invents automatic differentiation
- 1974: Bert Speelpenning's PhD thesis formalizes forward/reverse mode
- 1980s: Backpropagation rediscovered for neural networks (Rumelhart, Hinton, Williams)
- 2015: TensorFlow brings autodiff to the masses (Google Brain)
- 2018: JAX launched by Google Brain (Matt Johnson, Roy Frostig) - composable transformations
- 2020s: JAX becomes standard for scientific ML (physics-informed learning, differentiable simulation)

**Key Figures**:
- Roy Wengert (invented autodiff, 1964)
- David Rumelhart (backprop for neural nets, 1986)
- Yann LeCun (deep learning pioneer, championed autodiff)
- Matt Johnson, Roy Frostig (JAX creators at Google)

**The XLA Story**:
- Developed initially for TensorFlow optimization
- JAX leverages XLA for compilation (reusing Google infrastructure)
- Same technology powers TPUs (Google's AI chips)
- Now open-source, used across industry (TensorFlow, JAX, PyTorch via torch_xla)

---

## Success Metrics

**Students successfully completing Module 6 + Project 5 should be able to:**

**Technical Skills**:
1. ✅ Explain autodiff conceptually AND mathematically (chain rule, computational graphs)
2. ✅ Write pure, functional JAX code fluently (no NumPy habits)
3. ✅ Apply jit/grad/vmap to any physics problem (composing transformations)
4. ✅ Migrate NumPy code to JAX with >10× speedup (systematic process)
5. ✅ Build training loops using Optax (Adam, schedules, convergence monitoring)
6. ✅ Debug JAX code systematically (ConcretizationErrors, shape issues, NaN detection)

**Software Engineering**:
7. ✅ Structure code as Python package (pyproject.toml, clean API)
8. ✅ Write comprehensive tests (pytest, physics validation, >80% coverage)
9. ✅ Document professionally (README, docstrings, examples)
10. ✅ Generate and manage training data (HDF5, versioning, reproducibility)

**Research Preparation**:
11. ✅ Choose appropriate ML tools (GP vs NN vs NeuralODE) for research problems
12. ✅ Navigate JAX ecosystem (know what exists, where to learn more)
13. ✅ See JAX as paradigm shift, not just "another library"
14. ✅ Ready for Module 7 (ML foundations using existing tools)

**Mindset Transformation**:
15. ✅ Think like software engineer, not just script writer
16. ✅ Value testing, documentation, reproducibility (professional standards)
17. ✅ Build tools others can use (not just code that "works for me")
18. ✅ See computation as enabling discovery (not just calculation)

**Portfolio Piece**:
19. ✅ Have working, professional JAX package to show employers/collaborators
20. ✅ Understand codebase deeply enough to extend and maintain it

---

## The Path from Here

**You've completed Module 6.** You understand JAX deeply — not as a black box, but as a set of mathematical transformations that enable modern scientific computing.

**Project 5 awaits**: Transform understanding → software. Take your N-body simulator from a script to a professional package. Generate the data that will power your final project.

**Then Module 7**: Gaussian Processes and Neural Networks. You'll use your JAX fluency to build ML models from scratch, understanding every line of code.

**Finally, your Final Project**: Physics-informed learning on star cluster simulations. You'll combine everything:
- Your N-body package (Project 5)
- Your training data (Project 5)
- Your ML models (Module 7)
- Your inference skills (Module 5)
- Your statistical thinking (Module 1)

**The result**: Novel research, publication-quality, portfolio-ready.

**You're not just learning to code. You're learning to do science computationally at the highest level.**

---

**Now go build something beautiful.** 🚀

---

## Further Resources

**Essential Reading**:
- **JAX Documentation**: https://jax.readthedocs.io — Start here, official docs
- **JAX Tutorial**: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
- **JAX GitHub**: https://github.com/google/jax — Source code, issues, discussions

**Deep Dives (Autodiff Theory)**:
- **Autodiff Survey**: Baydin et al. (2018), "Automatic Differentiation in Machine Learning: a Survey"
- **JAX Paper**: Bradbury et al. (2018), "JAX: Composable transformations of Python+NumPy programs"
- **XLA Compiler**: https://www.tensorflow.org/xla — How JIT compilation works

**Ecosystem Documentation**:
- **Optax**: https://optax.readthedocs.io — Gradient-based optimization
- **Equinox**: https://docs.kidger.site/equinox/ — Neural networks as PyTrees
- **Diffrax**: https://docs.kidger.site/diffrax/ — Differential equation solving
- **GPJax**: https://gpjax.readthedocs.io — Gaussian Processes
- **NumPyro**: https://num.pyro.ai — Probabilistic programming, MCMC

**Scientific Software Engineering**:
- **The Pragmatic Programmer** (Thomas & Hunt) — Classic software wisdom
- **Research Software Engineering with Python** (Irving et al.) — Free online book
- **Python Packages** (https://py-pkgs.org) — Modern package development guide

**Historical Papers**:
- Wengert (1964): "A simple automatic derivative evaluation program"
- Speelpenning (1974): PhD thesis on reverse-mode autodiff
- Rumelhart et al. (1986): "Learning representations by back-propagating errors"

---

**Next**: Project 5 — Building Your JAX N-body Package 🎯
