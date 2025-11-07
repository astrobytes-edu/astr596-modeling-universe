---
title: "Overview: The JAX Revolution in Scientific Computing"
subtitle: "Scientific Computing with JAX | Computing the Universe | ASTR 596"
---

(crisis-that-changed)=
## The Crisis That Changed Everything

:::{epigraph}
*Mountain View, California. 2017-2018.*

Researchers at Google Brain faced a familiar research bottleneck. They wanted the **ease of NumPy** for prototyping ML experiments, but needed **automatic differentiation** and **hardware acceleration** on Google's custom TPUs. Existing frameworks forced an uncomfortable choice.

TensorFlow required pre-defining static computational graphs. Experimentation meant rebuilding graphs repeatedly. PyTorch offered flexibility but didn't efficiently target specialized hardware like TPUs. And manually distributing large computations across multiple accelerators? A complex engineering nightmare for each new algorithm.

**The team's insight**: What if, instead of one monolithic framework, they built a library of **composable function transformations**?

Write ordinary Python and NumPy code. Then apply transformations as you need them:

- `grad` for automatic differentiation
- `jit` for Just-In-Time compilation to XLA (Accelerated Linear Algebra)
- `vmap` for automatic vectorization
- `pmap` for parallelization across devices

Compose them like Lego blocks: `jit(vmap(grad(f)))` just works.

They presented this approach at **SysML 2018** and released it as **JAX** later that year. The key innovation: **composable transformations** that separated concerns — automatic differentiation (`grad()`), just-in-time compilation (`jit`), and parallelization (`pmap`) became independent operations you could mix and match freely.

What Google built for ML research would accidentally revolutionize scientific computing. The same tools that enabled large-scale neural network training would solve one of computational physics' deepest bottlenecks: the astronomical cost of gradients.
:::

---

In Project 4, you spent hours implementing Hamiltonian Monte Carlo. You hand-coded finite difference gradients, watching your computer evaluate the log-posterior thousands of times just to approximate derivatives. It worked. But deep down, you knew: *There has to be a better way.*

In Project 2, you wrote an N-body simulator. It took minutes to run 100 particles for 100 timesteps. You accepted this as "the cost of doing physics."

You stand at a crossroads.

Down one path: continue with NumPy, finite differences, CPU-bound loops. The well-trodden path of academic computational physics. It works. Down the other path: automatic differentiation ("autodiff"), just-in-time (JIT) compilation, GPU acceleration. The path that **enables previously impractical science**. Physics-informed machine learning. Instant parameter inference from surrogate models. Real-time exploration of billion-dimensional parameter spaces.

**This module is that crossroads.**

By the time you finish Project 5, your N-body code will run 30-100× faster, compute numerically exact gradients automatically (machine precision), and generate 10,000 simulation realizations overnight—training data that will power the machine learning surrogates you'll build in Module 7.

---

(learning-outcomes)=
## Learning Outcomes

By the end of this overview, you will be able to:

- [ ] **Explain** why JAX exists and how Google's TPU constraints led to composable transformations
- [ ] **Identify** the three computational eras and where JAX fits in scientific computing evolution
- [ ] **Connect** your Project 4 finite differences pain to automatic differentiation solutions
- [ ] **Describe** the three core JAX transformations (grad, jit, vmap) and what they enable
- [ ] **Articulate** the transformation from script writer to scientific software engineer

---

(learning-paths)=

## Learning Paths & Priorities

This overview supports **three learning paths**. Choose based on your background and time constraints:

::::{tab-set}

:::{tab-item} Fast Track (Essential)
**For students with ML experience or time pressure**

Focus on sections marked 🔴 Essential:

- What This Module Is About
- The Pain Point
- A Taste of JAX
- Looking Ahead

**Outcome**: Understand JAX's core value proposition and what you'll build. Proceed directly to Part 1.
:::

:::{tab-item} Standard Track (Recommended)
**For most students — balances context with efficiency**

Add sections marked 🟡 Important:

- All Essential sections above
- The Computational Landscape
- Google's JAX Origin Story
- The Three Computational Eras

**Outcome**: Full context for WHY JAX matters, HOW it came to exist, and WHERE it fits in scientific computing history.
:::

:::{tab-item} Complete Track (Full depth)
**For students interested in career context or research trajectory**

Include all sections:

- All Essential and Important sections
- Career Context (dropdown admonition)
- All conceptual checkpoints and reflection prompts

**Outcome**: Deep understanding of JAX's role in the academic-industry landscape, career implications, and full historical context.
:::

::::

:::{admonition} 📖 How to Use This Overview
:class: note

**This is NOT Part 1.** You don't need to code anything here. This is pure **motivation and context**.

**Goals**:

1. Connect to your **lived experience** (Project 4 pain, Project 2 performance limits)
2. Show you what's **possible** (automatic differentiation, 100× speedups, batching 10k simulations)
3. Explain **why** JAX exists and why **you** need to learn it now
4. Preview the **transformation** you'll undergo (script writer → software engineer)

**After this overview**: Proceed to **Part 1** for conceptual foundations (functional programming, pure functions, computational graphs).
:::

---

(what-this-is)=

## What This Module Is About

**Priority: 🔴 Essential**

This module transforms you from **script writers to scientific software engineers**. You'll learn to:

- **Compute gradients automatically and exactly** (no more finite differences!)
- **Speed up code by 10-100×** through JIT compilation
- **Batch thousands of computations** effortlessly via vectorization
- **Build professional, reusable scientific software** (not just scripts that "work")
- **Generate reproducible training data** for machine learning (Module 7 foundation)

:::{margin}
**Automatic differentiation**: Computing numerically exact derivatives (machine precision) via computational graphs and the chain rule, not finite-difference approximations.

**JIT compilation**: Just-In-Time compilation to optimized machine code via Google's XLA compiler.

**Vectorization**: Automatic batching of operations across data, enabling massive parallelism.
:::

**This isn't just learning a new library. It's learning how *modern* scientific software is built.**

By the end of Module 6 + Project 5, you'll have transformed your N-body simulator from a NumPy script into a professional JAX package that:

- Runs **10-100× faster** (seconds instead of minutes)
- Computes forces via **automatic differentiation** (goodbye, hand-coded derivatives!)
- **Batches 10,000 simulations** overnight (training data for Module 7)
- Is **pip-installable, tested, and documented** (portfolio-quality code)

:::{admonition} 🔗 Connection to Your Course Journey
:class: note

**Where you've been**:

- **Projects 1-3**: Forward simulation (Monte Carlo stellar populations, N-body dynamics, radiative transfer)
- **Module 5 & Project 4**: Inverse problems (Bayesian inference, HMC for cosmological parameters)

**Where you're going**:

- **Module 6 & Project 5**: Learn JAX, build differentiable physics simulator, generate massive training datasets
- **Module 7**: Machine learning surrogates (train GP/NN on your 10k simulations)
- **Final Project**: Physics-informed learning (combine simulation + ML for instant inference)

**Module 6 is the inflection point**—where you cross from Era 2 (expensive inference) to Era 3 (amortized inference via learned surrogates).
:::

---

(jax-origin)=

## Google's JAX — Why Does This Library Exist?

**Priority: 🟡 Important**

### The Problem Google Faced (2015-2017)

Google had built custom AI accelerators called **TPUs** (Tensor Processing Units) for machine learning. These chips could run computations **10-100× faster than GPUs**—but only if software could exploit them properly.

**The problem**:

- **TensorFlow** (Google's first ML framework): Required **static computational graphs**. You couldn't write flexible, exploratory Python code—you had to pre-define everything before running.
- **PyTorch** (Facebook's framework): **Flexible** (dynamic graphs) but didn't compile efficiently to specialized hardware.
- **CUDA** (NVIDIA's low-level GPU programming): Full control but required **hardware expertise** most researchers didn't have.

**What researchers actually wanted**: Write ordinary Python code describing physics or ML models, and automatically get:

- ✅ Gradients (for optimization/inference)
- ✅ Compilation (for speed)
- ✅ Hardware acceleration (TPU/GPU support)
- ✅ Parallelization (across multiple devices)

**Without** needing to be hardware experts or ML engineers.

---

### The Innovation: JAX's Key Insight

JAX's breakthrough was **complete separation of concerns**:

**You write**: Pure Python functions (just describe your computation—your physics)

**JAX provides**: Independent transformations you **compose**:

:::{mermaid}
graph LR
    A["Your Function
    f(x)"] --> B["grad f
    Differentiate"]
    A --> C["jit f
    Compile"]
    A --> D["vmap f
    Vectorize"]
    A --> E["pmap f
    Parallelize"]

    B --> F["Compose!
    jit(grad(f))"]
    C --> F
    D --> F
    E --> F

    F --> G["Fast, batched,
    differentiable,
    parallel code"]
:::

**The magic**: These transformations **compose freely**:

```python
# All of these are valid!
jit(grad(f))                  # Fast gradients
vmap(grad(f))                 # Batched gradients
jit(vmap(grad(f)))            # Fast batched gradients
pmap(jit(vmap(grad(f))))      # Multi-GPU fast batched gradients
```

:::{margin}
**Composability**: The property that transformations can be combined in any order, with the result being predictable and well-defined.

**Separation of concerns**: Design principle where different aspects (differentiation, compilation, parallelization) are independent and don't interfere with each other.
:::

**Why this was novel**: Previous frameworks **entangled** differentiation with execution. You couldn't easily say "I want gradients of this function, but also compile it and run it on GPU."

JAX **decouples everything** — you describe transformations independently, compose them like Lego blocks.

---

### Why It Matters for You (Not Just Google)

Google didn't build JAX for speed alone — they built it because **ambitious research questions required automatic differentiation through arbitrary custom code**:

- **Physics simulators** (like your N-body code)
- **Domain-specific algorithms** (stellar evolution, radiative transfer, hydrodynamics)
- **Novel ML architectures** (combining data-driven models + physics constraints)

**The key insight**: If you can **differentiate through ANY computation**, new science becomes possible.

:::{admonition} 🔗 Connection to Project 2: N-body Forces
:class: note

In **Project 2**, you hand-coded force calculations:

```python
# You manually derived: F = -G * m1 * m2 / r^2 * r_hat
for i in range(n):
    for j in range(i+1, n):
        r_vec = positions[j] - positions[i]
        r = np.linalg.norm(r_vec)
        force_mag = G * masses[i] * masses[j] / r**2
        force_vec = force_mag * r_vec / r  # Unit vector
        forces[i] += force_vec
        forces[j] -= force_vec  # Newton's 3rd law
```

**With JAX autodiff**, you can instead:

```python
# Define potential energy (easier to get right)
def U(positions, masses):
    return -sum(G * m_i * m_j / r_ij for all pairs)

# Get forces automatically: F = -∇U
forces = -jax.grad(U, argnums=0)(positions, masses)
```

**Why this matters**:

- Fewer bugs (potential energy is scalar, simpler than vector forces)
- Generalizes easily (add magnetic fields, relativity, exotic potentials)
- Numerically exact derivatives (machine precision, not hand-coded approximations that might have errors)

**This is the power JAX unlocks.**
:::

---

### Current Reality (2025)

Frontier research labs (DeepMind, Anthropic, Google Research, national labs) now use JAX as **production infrastructure**.

You're learning **tools that enable cutting-edge research**, not pedagogical simplifications.

**JAX wasn't built for astrophysics — but it solves our problems perfectly.**

---

:::{important} 💡 What We Just Learned

**Key insights about JAX's origin**:

1. **JAX emerged from hardware constraints** (TPUs needed better software)
2. **Composable transformations are the innovation** (not just "another array library")
3. **Separation of concerns enables flexibility** (grad, jit, vmap work independently and compose)
4. **Autodiff through arbitrary code unlocks new science** (not just faster old science)
5. **Production infrastructure, not toy** (DeepMind, Anthropic use this for frontier research)

**Why this matters**: Understanding JAX's design philosophy helps you **use it effectively**—you'll know WHY functional programming is required, WHY transformations compose, WHY it's worth the learning curve.
:::

:::{admonition} 💼 Why This Matters for Your Career
:class: dropdown, note

**The rare combination you're building**:

Most astronomy and physics masters and PhD students graduate with either **(1)** domain expertise (astrophysics, statistical mechanics) **or** (2) modern ML infrastructure (PyTorch, production systems). **You're learning both** — plus professional software engineering (testing, packaging, documentation).

**Where this combination is explicitly hiring** (2025):

- **Research labs**: DeepMind, Anthropic, OpenAI (physics-informed ML, differentiable simulators)
- **Scientific ML**: Climate modeling, materials science, drug discovery (simulation + learning)
- **Computational astrophysics**: Research-intensive universities (modern methods attract top journals)

**What makes you different**: By the Final Project, you'll have a **portfolio-quality GitHub package** — differentiable N-body simulator in JAX, professionally tested and documented, with 10k training runs for ML surrogates. Most graduates won't build this until years into their career, if ever.

**The signal**: "I built differentiable simulators that generate training data 100× faster" opens doors that "I trained models on existing data" doesn't. You're learning to **create** the data that powers modern science, not just analyze it.

**JAX specifically**: Growing rapidly in research (not production ML). Complements PyTorch (which you can learn easily once you know JAX). Positions you at the intersection of physics + ML where the most interesting problems live.
:::

---

(pain-point)=

## The Pain Point — Your Project 4 Experience

**Priority: 🔴 Essential**

### Remember Project 4?

In Project 4, you implemented **Hamiltonian Monte Carlo** to measure dark energy parameters. For each HMC step, you needed **gradients of the log-posterior**, which you computed using **finite differences** — approximating derivatives by evaluating the function at slightly different parameter values.

**This worked.** You got results. Dark energy parameters estimated. Project submitted.

**But the pain was real**:

- **Scaling nightmare**: Each gradient required **2d function evaluations** (d = number of parameters)
  - d=2 (your Project 4): 200,000 likelihood evaluations for one HMC run — tolerable
  - d=100 (realistic astronomy problems): 10 million evaluations — **completely impractical**

- **Numerical fragility**: Tuning the step size $h$ was finicky
  - Too large: inaccurate derivatives ($\mathcal{O}(h^2)$ error)
  - Too small: floating-point precision errors
  - Optimal $h$ varies by parameter—manual tuning required

- **Approximate, not exact**: Gradient errors compound through thousands of HMC steps, causing poor mixing and biased estimates

### What You'll Be Able to Do After Module 6

**Project 4** (Finite Differences):

```python
# Many lines of careful numerical code
grad = np.zeros_like(theta)
for i in range(len(theta)):
    theta_plus, theta_minus = theta.copy(), theta.copy()
    theta_plus[i] += h; theta_minus[i] -= h
    grad[i] = (log_posterior(theta_plus) - log_posterior(theta_minus)) / (2*h)
# Cost: 2d likelihood evaluations (approximate)
```

**After Module 6** (Automatic Differentiation):

```python
grad_log_posterior = jax.grad(log_posterior)
# Cost: ~3-5× forward pass, INDEPENDENT of d (exact!)
```

**One line.** Not just shorter — **exact** (to machine precision), **fast** (400× faster for d=1000), and **automatic** (works for any function JAX can trace).

:::{admonition} 🔗 Connection to HMC
:class: note

HMC requires ~50,000 gradient evaluations per run (50 leapfrog steps × 1000 proposals). **With finite differences**, this becomes computationally impossible for d > 10. **With autodiff**, high-dimensional inference becomes feasible. This is why modern tools (PyMC, Stan, NumPyro) all use autodiff.
:::

---

(taste-of-jax)=

## A Taste of JAX — Three Transformations That Change Everything

**Priority: 🔴 Essential**

This is a brief preview of JAX's core capabilities. Don't worry about understanding every detail — **Parts 1-2** will explain the how and why. For now, see what's possible.

---

### 1. Automatic Differentiation: `jax.grad`

**Problem**: In Project 2, you hand-coded gravitational forces with nested loops, manual vector math, and careful Newton's 3rd law bookkeeping.

**JAX solution**: Write the simpler function (potential energy), get forces for free:

```python

import jax
import jax.numpy as jnp

def gravitational_potential(positions, masses):
    """Compute potential energy U (scalar, simple)."""
    n = len(masses)
    eps =0.01  # Softening length
    G = 6.67e-8  # Gravitational constant in CGS
    U = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = jnp.linalg.norm(positions[j] - positions[i]) + eps**2.0
            U -= G * masses[i] * masses[j] / r
    return U

# Automatic differentiation: F = -∇U
force_function = jax.grad(gravitational_potential, argnums=0)
```

**What you get**: Exact forces via chain rule, automatically. Want to add magnetic fields or modified gravity? Just update `U(r)` and call `jax.grad`—forces come free.

**Connection to Project 4**: This is the same `jax.grad` you'll use for HMC gradients—exact, fast, automatic.

---

### 2. JIT Compilation: `jax.jit` — 10-100× Speedups

**Problem**: NumPy loops are slow (Python interpreter overhead). Your Project 2 N-body code took minutes for 100 particles.

**JAX solution**: Compile Python to optimized machine code:

```python
def compute_forces(positions, masses):
    # ... your N-body force calculation ...
    n = len(masses)
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec) + 1e-10
                forces[i] += G * masses[i] * masses[j] * r_vec / r**3
    return forces

def compute_forces_jax(positions, masses, G=6.67e-8):
    """N-body forces (JAX version—same logic, jnp instead of np)."""
    n = len(masses)
    forces = jnp.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = jnp.linalg.norm(r_vec) + 1e-10
                forces[i] += G * masses[i] * masses[j] * r_vec / r**3
    return forces

# JIT-compile the JAX version (one line!)
compute_forces_jit = jax.jit(compute_forces_jax)
```

Now **benchmark**:

```python
# Setup: 100 particles
n_particles = 100
positions_np = np.random.rand(n_particles, 3) * 1e13  # Random positions [cm]
masses_np = np.random.rand(n_particles) * 1e33        # Random masses [g]

# NumPy timing
start = time.time()
for _ in range(100):
    forces_np = compute_forces_numpy(positions_np, masses_np)
time_numpy = (time.time() - start) / 100

# JAX timing (convert arrays)
positions_jax = jnp.array(positions_np)
masses_jax = jnp.array(masses_np)

# First call: compilation overhead (ignore this one)
_ = compute_forces_jit(positions_jax, masses_jax).block_until_ready()

# Actual timing
start = time.time()
for _ in range(100):
    forces_jax = compute_forces_jit(positions_jax, masses_jax).block_until_ready()
time_jax_jit = (time.time() - start) / 100

print(f"NumPy time: {time_numpy*1000:.2f} ms")
print(f"JAX+JIT time: {time_jax_jit*1000:.2f} ms")
print(f"Speedup: {time_numpy / time_jax_jit:.1f}×")
```

:::{margin}
**.block_until_ready()**: JAX evaluates lazily—this forces computation to complete before timing. Without it, you'd measure time to **launch** computation, not time to **finish** it.
:::

**Typical output** (your mileage may vary):

```
NumPy time: 45.2 ms
JAX+JIT time: 1.2 ms
Speedup: 37.7×
```

**What happened?**

JAX compiled your Python function to **optimized machine code** (via XLA compiler). The XLA compiler:

- **Fuses operations** (one kernel instead of many Python loops)
- **Eliminates intermediate arrays** (no memory allocation overhead)
- **Optimizes for your specific CPU/GPU** (uses SIMD, caching, etc.)

**One decorator** (`@jax.jit`) gave you ~40× speedup. This is **typical** for numerical code.

:::{admonition} 🔬 Note: Hardware Dependence
:class: note

Actual speedups depend on:

- **Hardware**: CPU (10-30×), GPU (30-100×), TPU (50-200×)
- **Problem size**: Bigger problems → better speedups (more to optimize)
- **Code structure**: More operations to fuse → better optimization

**Typical range**: 10-100× for scientific computing workloads.

See **Part 6, Appendix A** for detailed benchmarks with hardware specs, problem sizes, and breakdown of where speedups come from.
:::

:::{admonition} ⚠️ Common Misconception
:class: warning

**Misconception**: "JIT just makes Python loops faster."

**Reality**: JIT **compiles your function to machine code** (like C/Fortran). The speedup comes from:

1. **Operation fusion** (eliminate Python interpreter overhead)
2. **Memory optimization** (eliminate intermediate arrays)
3. **Hardware-specific code** (use CPU vector instructions, GPU parallelism)

**Implication**: You still need to write "JAX-friendly" code (pure functions, no data-dependent control flow). Part 1 teaches you what "JAX-friendly" means.
:::

---

### 3. Vectorization: `jax.vmap` — Batch 1000 Simulations Automatically

**Problem**: You need 10,000 N-body simulations for ML training (Module 1 ensemble averages, computational style). NumPy sequential loop takes hours.

**JAX solution**: Automatic vectorization over batch dimension:

```python
def simulate(initial_conditions):
    # ... your N-body integration ...
    return final_state

# Automatic batching (no manual loops!)
batched_simulate = jax.vmap(simulate)
results = batched_simulate(initial_conditions_batch)  # 1000 sims in parallel
```

**Impact for Project 5**: 10,000 simulations goes from ~3 hours (sequential) to ~3 minutes (GPU batched) = **60× faster**. This makes ML training data generation feasible within the semester. Without JAX, you'd generate 1000 sims max—ML models would be data-starved.

---

### The Key: Composable Transformations

JAX's superpower is that these three transformations **compose freely**:

```python
# Combine them: fast, batched, automatic gradients
fast_batched_gradients = jax.jit(jax.vmap(jax.grad(loss_fn)))
```

**This enables new science**—not just faster old science. Workflows that were computationally impossible (10k gradient evaluations for ML training) become routine.

**Parts 1-2 explain the how and why.** This was just a taste to build excitement.

---

(three-eras)=

## The Three Computational Eras

**Priority: 🟡 Important**

Scientific computing evolved through three complementary approaches. **All three coexist today**—you'll use each depending on the problem:

| Era | Question | Example | Status |
|-----|----------|---------|--------|
| **Era 1**: Forward Simulation (1950s+) | "What happens if...?" | Run N-body with these ICs | Still essential foundation |
| **Era 2**: Inverse Problems (1990s+) | "What parameters fit...?" | MCMC for cosmological params | Standard for inference |
| **Era 3**: Learning from Sims (2020s+) | "Amortize inference" | Train surrogate, instant predictions | Emerging frontier |

**Module 6 is your bridge from Era 2 to Era 3.**

---

### Era 1: Forward Simulation

**What**: Physics equations → simulate forward → observe results

**Your course examples**: Projects 1-3 (stellar populations, N-body, radiative transfer)

**Limitation**: Can't efficiently answer **"what parameters produced this observation?"** (requires expensive trial-and-error or grid search)

---

### Era 2: Inverse Problems (Bayesian Inference)

**What**: Have observations → infer parameters via probability (Bayes' theorem)

**Your course example**: Project 4 (HMC for dark energy parameters from supernova data)

**Key tools**: MCMC, HMC, nested sampling—all require **many gradient evaluations** (this is where JAX's autodiff saves you)

**Limitation**: Expensive! Each posterior sample requires running the forward model. For complex simulations (hours each), inference takes weeks.

### Era 3: Amortized Inference via ML Surrogates

**What**: Train ML models on thousands of simulations → make instant predictions

**The breakthrough**: Pay upfront cost once (generate training data with JAX's speed), then inference is cheap forever

**Your course arc**:
1. **Project 5** (Module 6): Generate 10,000 N-body simulations with JAX (`jit` + `vmap` makes this feasible)
2. **Module 7**: Train surrogate model (GP/NN) to learn: parameters → observables
3. **Final Project**: Use surrogate for instant Bayesian inference (milliseconds instead of hours)

**Why JAX enables this**: Generating 10k simulations used to take weeks (prohibitive). With JAX's 10-100× speedups, it's overnight (feasible). This is the computational bottleneck JAX solves.

---

### Course Arc Through the Eras

- **Projects 1-3**: Era 1 (forward simulation—build physics intuition)
- **Project 4, Module 5**: Era 2 (inverse problems—learn Bayesian inference)
  → **Pain point**: Finite differences for gradients don't scale
- **Module 6 (JAX)**: **Bridge to Era 3** (tools for fast simulation + gradient computation)
- **Module 7**: Era 3 (train surrogates on simulation ensembles)
- **Final Project**: Combine all three (physics-informed learning)

:::{admonition} 🔑 Key Insight: These Eras Coexist
:class: note

Eras 1-3 are **not** chronological replacements—they're **complementary approaches**:

- **Era 3 requires Era 1**: Generate thousands of training simulations (JAX-accelerated forward modeling)
- **Era 3 enhances Era 2**: Surrogates become fast likelihoods for MCMC/HMC
- **Era 2 validates Era 3**: Use full simulations on parameter subsets to verify surrogate accuracy

**Your Final Project workflow**: Generate 10k N-body sims (Era 1) → Train surrogate (Era 3) → Use in HMC (Era 2) → Validate with full sims (Era 1)
:::

---

(looking-ahead)=

## Looking Ahead — What You'll Build

**Priority: 🔴 Essential**

### Module 6 Arc

**Part 1: Conceptual Foundations** (🔴) → Functional programming, pure functions, computational graphs

**Part 2: Core Transformations** (🔴) → `grad`, `jit`, `vmap` (mathematics + practice)

**Part 3: N-body Migration** (🟡) → Rewrite Project 2 in JAX with autodiff forces

**Part 4: JAX Ecosystem** (🟢) → Optax, Equinox, Diffrax, NumPyro (enrichment)

**Part 5: Professional Software Engineering** (🟡) → Packaging, testing, documentation

**Part 6: Synthesis** (🟡) → Reflection on transformation (script writer → software engineer)

---

### Project 5: Professional JAX N-body Package

By the end of Project 5, you'll have a **pip-installable JAX package** that:

**✅ Runs 10-100× faster** (JIT compilation: NumPy 60s → JAX 2s → GPU 0.5s)

**✅ Computes forces via autodiff** (no hand-coded derivatives):

```python
def U(positions, masses):  # Potential energy (simple!)
    return -sum(G * m_i * m_j / r_ij for all pairs)

forces = -jax.grad(U)(positions, masses)  # Automatic, exact
```

**✅ Batches thousands of simulations** (`vmap` → 10k sims overnight instead of weeks)

**✅ Generates ML training data** (10k N-body runs → HDF5 dataset for Module 7 surrogates)

**✅ Is professionally packaged** (tests, docs, CI/CD—portfolio-quality artifact)

---

### The Transformation

**Before Module 6**: Script writer (finite differences, slow loops, one-off NumPy code)

**After Module 6**: Scientific software engineer (autodiff, vectorization, tested packages)

**Paradigm shift**:
```
Approximate methods    →  Exact transformations
One-off scripts        →  Reusable software
Black-box tools        →  Glass-box understanding
```

**This is the bridge to Era 3**: Your JAX package enables Module 7 (train GP/NN surrogates) and Final Project (physics-informed ML).

---

## Ready to Begin?

**Self-check before Part 1**:

- [ ] Can you explain why JAX exists (Google's TPU problem)?
- [ ] Which computational era do Projects 1-4 belong to?
- [ ] How does autodiff solve Project 4's finite differences pain?
- [ ] What do `grad`, `jit`, `vmap` do (conceptually)?

If yes → **Continue to [Part 1: Conceptual Foundations](part-01-conceptual-foundations.md)**

**Part 1 teaches WHY** (functional programming, pure functions, computational graphs). **Part 2 teaches HOW** (transformation mechanics, composability, debugging).

---

*Continue to **[Part 1: Conceptual Foundations](part-01-conceptual-foundations.md)** →*
