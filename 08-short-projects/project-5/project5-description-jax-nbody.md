# Project 5: JAX N-Body Engine

**ASTR 596 Fall 2025**  
**Instructor:** Anna Rosen  
**Due:** Wednesday, November 24, 2025 by 11:59 PM  

---

## Overview

In Project 2, you built an N-body simulator in NumPy. It worked, but it was slow — simulating 500 particles probably took an hour per run. Now imagine you're a researcher generating thousands of simulations to train a machine learning model (your final project). With Project 2's code, that's impossible. With JAX, it's feasible.

Beyond speed, this project teaches you to build **research-grade tools**, not just scripts:

- Python package structure with modules and `__init__.py`
- Functional programming patterns used in modern scientific computing
- Reproducible science (explicit random number generation, version control)

**The transformation**: From writing scripts that work once on your laptop to building organized, importable packages that work reliably for anyone.

**Goals**:

- **10-100× speedup** vs. pure-Python Project 2 (nested loops)
- **Speedups vs. vectorized NumPy on CPU are workload-dependent; you must benchmark and report**. Aim for a measurable win after JIT warmup.
- JIT compilation and vectorization for significant performance gains
- Reproducible simulations with JAX's functional random number generation
- Python package structure with modular organization
- Foundation for your final ML project (fast ground truth generation)

**Academic Integrity**: Hints in this document describe algorithms and patterns; do not copy scaffolds from the internet or LLMs. Your code must be your own implementation.

---

## Learning Objectives

- Transform NumPy → JAX (`jit`, `vmap`, `grad`)
- Use `jax.lax.scan` for timestepping and `jax.lax.while_loop` for bounded rejection sampling
- Implement pure functional integrators
- Generate and evolve virialized star clusters ($\alpha_{\rm vir} = 1$)
- Design, test, and document a scientific Python package
- Profile and optimize performance

---

## Critical Configuration

::::{important} **JAX Precision Requirements**
JAX defaults to **float32** precision to optimize for machine learning workloads. However, gravitational N-body simulations require **float64** precision for proper energy conservation. Without this, you'll see energy drift orders of magnitude worse than your Project 2 NumPy code.

Add this to the **top of your main module** (before any other JAX imports):

```python
import jax
jax.config.update("jax_enable_x64", True)
```

**Critical**: This configuration **must execute before any `jax.numpy` import in any module**. If you import `jax.numpy` before setting `x64=True`, the configuration won't apply and you'll silently get float32 precision.

**Why float64 matters**: Gravitational forces scale as $1/r^2$. Small numerical errors in position accumulate quadratically in force calculations, leading to spurious energy drift. Float32 has ~7 decimal digits of precision; float64 has ~16. For N-body work, you need those extra digits.

**Additional dtype safety**: In modules that construct arrays from Python floats, cast explicitly: `jnp.asarray(x, dtype=jnp.float64)`. This prevents silent float32 literals from leaking into compiled code.

**For production and all science runs, you must use float64.** All validation tests, performance benchmarks, and final simulations should use float64. You may experiment with float32 during development to speed up debugging iterations, but be aware that energy conservation will be poor and results will not be scientifically valid.

**Test this works**: After importing JAX, check: `jax.numpy.array(1.0).dtype` should show `dtype('float64')`, not `dtype('float32')`.
::::

::::{note} **Device Selection**
Assume CPU for this project. If you have a GPU (64-bit) and want to force CPU for apples-to-apples performance comparisons with NumPy:

```python
jax.config.update('jax_platform_name', 'cpu')
```

This must run before any JAX imports. For benchmarking, it's best to keep everything on CPU to isolate the compilation and vectorization benefits of JAX from hardware differences.

**For quickstart.py**: You may want to add `jax.default_device(jax.devices("cpu")[0])` at the top to ensure everything runs on CPU even if a GPU is present.
::::

::::{tip} **Performance Optimization Hint**
For your main integration function, consider using `@jax.jit(..., donate_argnums=(0,))` to enable zero-copy state passing. This allows JAX to reuse memory buffers rather than copying, which can significantly reduce memory pressure on laptop CPUs.
::::

---

## Part 1: Core Implementation

**Getting started tip**: A good way to familiarize yourself with JAX is to start by repurposing your Project 2 IMF and Plummer sampling modules. Convert them to use `jax.random` and `jax.numpy`, JIT-compile them, and validate they produce the same distributions. This gives you a low-stakes way to learn JAX's functional programming patterns before tackling the more complex integrator.

### Step 1: Initial Conditions in JAX

**Critical**: Clusters must start in virial equilibrium to avoid artificial dynamics.

#### 1.1 Kroupa IMF Sampler

Two-segment broken power law:

$$
\xi(m) \propto \begin{cases}
m^{-1.3} & 0.08 \le m/M_\odot < 0.5 \\
m^{-2.3} & 0.5 \le m/M_\odot \le 100
\end{cases}
$$

**Inverse-CDF** for segment $[a, b]$ with slope $\alpha \neq 1$:

$$
m(u) = \left[ a^{1-\alpha} + u(b^{1-\alpha} - a^{1-\alpha}) \right]^{1/(1-\alpha)}, \quad u \sim \mathcal{U}(0,1)
$$

**Requirements**:

- Use `jax.random.PRNGKey` and `jax.random.split` (no global RNG)
- JIT-compile with `@jit`
- Sample exactly $N$ stars
- Default limits: $m_{\min} = 0.08\,M_\odot$, $m_{\max} = 100\,M_\odot$ (configurable)
- **Note**: $m_{\max} = 100\,M_\odot$ is conventional for star clusters (very massive stars are rare and short-lived)
- **Validate**: Recover slopes $\{-1.3, -2.3\}$ from log-binned histogram

**JAX control flow reminder**: Inside `@jit`, avoid Python `for`/`while` loops on traced arrays. Use `jax.lax.scan` for time-stepping loops and `jax.lax.while_loop` for bounded rejection sampling or fixed-point iterations. These compile to efficient XLA control flow.

#### 1.2 Plummer Sphere Positions

The **Plummer sphere** is a spherically symmetric stellar system with density profile:

$$
\rho(r) = \frac{3M_{\rm total}}{4\pi a^3} \left(1 + \frac{r^2}{a^2}\right)^{-5/2}
$$

where $a$ is the **Plummer scale length** (characteristic radius).

**Radius inverse-CDF**:

$$
r(u) = a \left( u^{-2/3} - 1 \right)^{-1/2}, \quad u \sim \mathcal{U}(0,1)
$$

**Angles** (uniform on sphere):

$$
\cos\theta \sim \mathcal{U}(-1, 1), \quad \phi \sim \mathcal{U}(0, 2\pi)
$$

**Requirements**:

- Re-center: $\sum_i m_i \vec{r}_i = \vec{0}$
- **Validate**: Density profile matches $\rho(r) \propto (1 + r^2/a^2)^{-5/2}$

#### 1.3 Virial Equilibrium Velocities

**Goal**: Initialize with virial parameter $\alpha_{\rm vir} = 1$ (equilibrium)

The **virial parameter** (or virial ratio) is defined as:

$$
\alpha_{\rm vir} = \frac{2K}{|W|}
$$

where $K$ is total kinetic energy and $W$ is gravitational potential energy (negative).

**Physical meaning**:

- $\alpha_{\rm vir} = 1$: Virial equilibrium ($2K + W = 0$), bound and stable
- $\alpha_{\rm vir} > 1$: Super-virial (unbound, will expand)
- $\alpha_{\rm vir} < 1$: Sub-virial (will contract)

Your clusters must start at $\alpha_{\rm vir} = 1 \pm 0.05$ to avoid artificial dynamics.

**Plummer Equilibrium Velocities (Exact Distribution Function) — REQUIRED:**

At radius $r$ with escape speed $v_{\rm esc}(r) = \sqrt{2\psi(r)}$, draw a dimensionless speed $q = v/v_{\rm esc} \in [0,1]$ from:

$$
f(q) \propto q^2 (1 - q^2)^{7/2}
$$

This comes from Plummer's exact distribution function (Dejonghe 1987). The peak is at $q^\star = \sqrt{2}/3 \approx 0.471$.

**Acceptance-Rejection Implementation**:

1. For star at radius $r$, compute $\psi(r) = \frac{GM_{\rm total}}{\sqrt{r^2 + a^2}}$ and $v_{\rm esc} = \sqrt{2\psi}$
2. Use `jax.lax.while_loop` with bounded `max_trials` (e.g., 256) to implement rejection sampling
3. Draw $q \sim \mathcal{U}(0,1)$ and accept/reject against $f(q) \propto q^2 (1-q^2)^{7/2}$
4. Use a constant envelope $M = f(q^\star)$ with $q^\star = \sqrt{2}/3$
5. **On failure after `max_trials`**: Fallback deterministically to $q = q^\star$ (the mode) and increment a counter
6. Record the **fraction of particles that used the fallback** — it should be $\ll 1\%$. If not, tighten your envelope
7. Direction: Sample uniform on sphere
8. Compute $\vec{v} = q v_{\rm esc} \hat{n}$
9. Vectorize with `vmap` over all particles

**Important JAX note**: Rejection sampling with `while_loop` requires careful state management. Your loop must carry the accepted value, current attempt count, and a boolean indicating whether to continue.

**Critical RNG requirement**: Carry a PRNG key in your state and split inside `while_loop`/`scan`. No global RNG or hidden state—JAX's functional random number generation requires explicit key threading through all operations.

**Verification**:

- Clusters should achieve $\alpha_{\rm vir} = 1 \pm 0.05$ after this initialization
- Re-center velocities: $\sum_i m_i \vec{v}_i = \vec{0}$

**Optional: Maxwellian Approximation** (for comparison)

If you want to compare methods, you can also implement:

$$
v_{\rm disp}(r) = \frac{1}{\sqrt{6}} v_{\rm esc}(r)
$$

Draw components from $\mathcal{N}(0, v_{\rm disp}^2)$. This is an approximation to the exact distribution but is much simpler and still produces roughly virial equilibrium.

---

### Step 2: Gravitational Force Calculation

**Physics**: The gravitational acceleration on particle $i$ is:

$$
\vec{a}_i = -G \sum_{j \neq i} m_j \frac{\vec{r}_i - \vec{r}_j}{(r_{ij}^2 + \epsilon^2)^{3/2}}
$$

where $\epsilon$ is a **softening length** that prevents singularities in close encounters.

**Why softening?** Real stellar systems have finite-size stars. In simulations, point masses can get arbitrarily close, causing:

1. Numerical instability ($r \to 0 \Rightarrow a \to \infty$)
2. Unrealistically large energy changes from two-body encounters

Softening models the smoothing effect of finite stellar radii. For star clusters, typical values: $\epsilon \sim 0.1 - 0.5$ pc, or adaptively: $\epsilon \sim 0.3 r_h / \sqrt{N}$ where $r_h$ is the half-mass radius.

**Deliverable for report**: State your $\epsilon$ policy (constant vs. $\epsilon \propto r_h/\sqrt{N}$), the chosen value/factor $\kappa$, and provide a 2-line justification tied to energy drift and timestep collapse rates observed in your simulations.

#### Two Implementation Strategies

**Strategy A: Direct Vectorization (Pairwise Distance Matrix)**

Implement a function `accelerations_matrix(x, m, eps2, G)` that:

1. Builds all pairwise separations $\Delta \vec{r}_{ij} = \vec{r}_i - \vec{r}_j$
2. Computes softened distances $r_{ij}^2 + \epsilon^2$ for all pairs
3. Zeroes the self-interaction terms (diagonal)
4. Returns accelerations $\vec{a}_i = -G \sum_{j \neq i} m_j \frac{\Delta \vec{r}_{ij}}{(r_{ij}^2 + \epsilon^2)^{3/2}}$
   - **Critical**: Weight each term by the mass $m_j$ of the source particle

**Requirements**:

- Fully vectorized (no Python loops)
- JIT-compiled with `@jax.jit`
- Float64 precision
- Memory: $O(N^2)$ temporaries

**Hint**: Your validation tests will check force symmetry and agreement with Strategy B to tight tolerances — this will force you to get the broadcasting and mass weighting exactly right.

- **Pros**: Straightforward vectorization, easier to reason about
- **Cons**: Memory: $O(N^2)$ temporaries; not scalable to very large $N$
- **When to use**: Small-to-medium $N$ (< 1000), learning JAX fundamentals

**Strategy B: `vmap` Over Particles**

Implement a function `accelerations_vmap(x, m, eps2, G)` by mapping a single-particle force function over all particles.

**Design considerations**:

- Write a helper that computes force on one particle due to all others
- Exclude self-interaction by index, not by float equality — your tests will catch float comparison errors
- Use `jax.vmap` to vectorize over all particles
- Must match Strategy A within `rtol=1e-12, atol=1e-14` in float64

**Deliverable for report**: Compare accelerations from both strategies on the same positions/masses and assert `max_rel_err < 1e-12`. Submit a short table showing measured error and wall time for N=32, 64, 128.

**Requirements**:

- JIT-compiled
- Float64 precision
- Agreement with Strategy A verified by tests

**Pros**: Lower peak memory ($O(N)$ per particle), composable with other transformations like `grad`

**Cons**: Slightly more complex logic, requires understanding `vmap` semantics

**When to use**: Want to learn `vmap`, planning to scale to larger $N$, or composing with `grad` for differentiable physics

---

**Implementation Strategy**: Implement **both strategies** (matrix-based and vmap-based)

- Compare memory usage and performance
- Ensure they produce identical results (see Validation)
- Wrap in `@jit` for production use

**Softening guidance**:

- Use $\epsilon = 0.01$ pc for validation tests
- For realistic clusters, estimate $r_h$ (half-mass radius) and use $\epsilon \sim 0.3 r_h / \sqrt{N}$
- Document your choice

---

### Step 3: Time Integration

**Leapfrog (DKD) Integration** — REQUIRED

The leapfrog integrator is second-order symplectic, meaning it exactly conserves a "shadow Hamiltonian" close to the true Hamiltonian. This gives excellent long-term energy conservation.

**DKD Algorithm** (Drift-Kick-Drift):

1. **Drift**: $\vec{x}_{1/2} = \vec{x}_n + \vec{v}_n \cdot \frac{\Delta t}{2}$
2. **Kick**: $\vec{v}_{n+1} = \vec{v}_n + \vec{a}(\vec{x}_{1/2}) \cdot \Delta t$
3. **Drift**: $\vec{x}_{n+1} = \vec{x}_{1/2} + \vec{v}_{n+1} \cdot \frac{\Delta t}{2}$

**Why leapfrog?**

- **Symplectic**: Preserves phase-space volume (Liouville's theorem)
- **Time-reversible**: Run backward to recover initial conditions (within numerical precision)
- **Excellent energy conservation**: Typical $|\Delta E / E| \sim 10^{-6}$ or better for sufficiently small $\Delta t$

**Implementation Requirements**:

Implement two functions following the DKD equations above:

1. `leapfrog_step(state, dt, params)` → `new_state`
   - Pure function (no mutation)
   - `state = (x, v)`, `params` contains masses, softening, etc.
   - Returns updated `(x_new, v_new)`

2. `integrate(state0, dt, n_steps, params)` → `(state_final, trajectory)`
   - Use `jax.lax.scan` for the timestepping loop (not Python `for`)
   - Returns final state and trajectory of states at each timestep

**Requirements**:

- JIT-compile both functions
- Use `jax.lax.scan` (**not** Python loops) for timestepping
- Float64 precision
- **Deliverable**: Show that $|\Delta E/E| \propto \Delta t^2$ by plotting error vs. timestep

---

### Step 4: Adaptive Timestepping

**Fixed timesteps are inefficient**: During expansion, you can take large steps. During close encounters, you need small steps.

**Criterion**: Choose $\Delta t$ such that:

$$
\Delta t = C \min_i \left( \frac{\epsilon}{|\vec{a}_i|} \right)^{1/2}
$$

where $C \sim 0.01 - 0.1$ is the **Courant factor**.

**Physical meaning**: Limit timestep so no particle moves more than $\sim \epsilon$ under its current acceleration.

**Implementation Requirements**:

- Compute $\Delta t$ at each integration step based on maximum acceleration
- Clamp: $\Delta t \in [\Delta t_{\min}, \Delta t_{\max}]$ to avoid runaway
- Integrate adaptive $\Delta t$ into your `lax.scan` loop (carry updated `dt` in state)

**Validation**: Plot $\Delta t(t)$ over a simulation. It should:

- Decrease during collapse/close encounters (high $|\vec{a}|$)
- Increase during expansion (low $|\vec{a}|$)
- Never reach $\Delta t_{\min}$ or $\Delta t_{\max}$ for reasonable parameters

**Discussion point for report**: Compare energy conservation and runtime for fixed vs. adaptive timesteps.

---

### Step 5: Diagnostic Utilities

Implement helper functions for energy analysis and validation:

**Required Functions**:

1. `kinetic_energy(v, m)` → `K`
   - Total kinetic energy of the system

2. `potential_energy(x, m, eps2, G)` → `W`
   - Total gravitational potential energy (negative)
   - Formula: $W = -G \sum_{i<j} \frac{m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}$
   - **Critical**: Avoid double-counting pairs (each pair $i,j$ counted once)
   - Must handle softening correctly and include gravitational constant $G$ explicitly

3. `total_energy(x, v, m, eps2, G)` → `(K, W, E)`
   - Returns kinetic, potential, and total energy
   - $E = K + W$

4. `virial_ratio(K, W)` → `α_vir`
   - Computes virial parameter $\alpha_{\rm vir} = 2K / |W|$

**Requirements**:

- All functions JIT-compiled
- Float64 precision
- Numerically stable

**Test Case** (validates your implementation):

- Two-body system with specific setup should give: $K = 0.25$, $W = -0.5$ (in your units), $\alpha_{\rm vir} = 1.0$
- Use this to verify you're not double-counting potential energy

**Usage**: These standardize your energy diagnostics and make plotting/validation consistent. Use them in your validation tests and include virial ratio plots in your technical report.

---

## Part 2: Production Pipeline

**Goal**: Generate a diverse ensemble of 50-100 simulations to serve as training data for your final ML project.

### Requirements

1. **Parameter sweep**:
   - Vary $N \in [200, 500]$
   - Vary $a$ (Plummer scale length) over $\sim 10\times$ range
   - Vary total mass $M_{\rm total}$ (via IMF sampling)
   - All clusters start at $\alpha_{\rm vir} = 1 \pm 0.05$

2. **Automated pipeline**:
   - A simple Python script (e.g., `run_ensemble.py`) that generates all simulations
   - Explicit random seed management (log seeds for reproducibility)
   - Can be sequential—no need for complex parallelization

3. **Data storage**:
   - Organized directory structure (e.g., `sim_0000/`, `sim_0001/`, ...)
   - Each simulation saves:
     - Initial conditions: $(x_0, v_0, m)$
     - Trajectories: $(x(t), v(t))$ at $\sim 50-100$ snapshots
     - Metadata: $N$, $a$, $M_{\rm total}$, $\alpha_{\rm vir,0}$, seed
     - Diagnostics: $E(t)$, $\alpha_{\rm vir}(t)$, $\Delta t(t)$
   - Format: NumPy `.npz` files (simple and efficient) or HDF5 if you prefer. Document your choice in the report.
   - Make sure you include a .gitignore to avoid committing large data files to your repo

4. **Statistical validation**:
   - Create a **single summary plot** showing your ensemble is physically consistent
   - Example: Energy conservation histogram across all runs, or $\alpha_{\rm vir}$ distribution
   - Quantify: "95% of simulations achieve $|\Delta E/E| < 10^{-5}$"

### Suggested Workflow

Your pipeline should:

1. **Define parameter grid**: Vary $N \in [200, 500]$, Plummer scale $a$ over $\sim 10\times$ range, total mass via IMF sampling
2. **Manage random seeds**: Use a master seed, split for each simulation, log all seeds for reproducibility
3. **Generate initial conditions**: For each parameter combination, create virialized cluster
4. **Integrate**: Run simulation, save trajectories and diagnostics
5. **Organize output**: Save to structured directories with metadata
6. **Statistical validation**: Aggregate results and verify ensemble consistency

**Implementation tips**:

- Use `itertools.product` or nested loops to generate parameter combinations
- Save incrementally (don't wait until all runs complete)
- Include progress logging so you know how many simulations are done
- Keep it simple—a straightforward loop over parameters is perfectly fine

---

## Part 3: Performance Analysis

### Benchmarking Requirements

1. **Force calculation comparison**:
   - Matrix-based vs. vmap: Time both implementations for $N \in \{50, 100, 200, 300, 400, 500\}$
   - Plot: runtime vs. $N$ on log-log scale
   - Quantify: Which is faster? When does the crossover occur?

2. **JAX vs. NumPy speedup**:
   - Compare your JAX implementation to your Project 2 vectorized NumPy code
   - Fair comparison: Same force calculation strategy, same timestep, measure after JIT warmup
   - Quantify: "JAX is Xfactor faster than NumPy for N=500"
   - **Note**: Speedups depend strongly on your Project 2 implementation quality. The order-of-magnitude of the runtime depends on your machine—measure and report on your hardware.

3. **JIT warmup**:
   - Measure first call (includes compilation) vs. subsequent calls
   - Demonstrate the cost of compilation is amortized over many runs

4. **Scaling analysis**:
   - Plot: total integration time vs. $N$ for fixed $t_{\rm final}$ and adaptive $\Delta t$
   - Quantify: How does your integrator scale? $O(N^2)$? Better?

### Profiling

Use `jax.profiler.trace` to identify bottlenecks. This generates a Chrome trace file that visualizes where time is spent (compilation, execution, data transfer).

**Steps**:

1. Wrap your integration call with `jax.profiler.trace(output_path)`
2. Load the trace in Chrome at `chrome://tracing`
3. Analyze: Is force calculation JIT-compiled? Does it dominate runtime? Any unexpected Python overhead?

**Report**: Include one profiling screenshot showing your force calculation is JIT-compiled and dominates runtime (not Python overhead).

---

## Validation Requirements

**Required tests** (implement at least these 6):

**1. Two-body orbit**:

- Set up circular or elliptical orbit with known analytic solution
- Integrate for 1000 steps
- Verify: trajectories match theory, energy conserved to $\lesssim 10^{-6}$ over $10^3$ steps at sufficiently small $\Delta t$. Show the integrator exhibits $\propto \Delta t^2$ error scaling.

**2. Force method agreement**:

- Generate a random cluster (N=100)
- Compute forces with both matrix and vmap strategies
- Verify: `jnp.allclose(a1, a2, rtol=1e-12, atol=1e-14)` in float64

**3. Force symmetry**:

- Set up an equilateral triangle of equal masses
- Verify zero net force at the centroid, or verify pairwise force antisymmetry: $\vec{F}_{ij} = -\vec{F}_{ji}$

**4. Units sanity check**:

- With $G=1$, place two unit masses at positions $\pm 1$ (separation $d=2$)
- **For this check, set softening to $\epsilon = 0$ (i.e., `eps2 = 0.0`)—with any $\epsilon > 0$ the analytic value changes**
- By symmetry, each particle experiences $|\vec{a}| = G m / d^2 = 1/4$
- Verify your force calculation returns exactly this value (tests unit handling and self-exclusion)

**5. Sampler validation**:

- **IMF**: Sample $10^4$ masses, verify power-law slopes in log-binned histogram
- **Plummer**: Sample $10^3$ positions, compare density profile to analytical form
- **Velocities**: Both methods (exact and approximate) should produce $\alpha_{\rm vir} \approx 1 \pm 0.05$

**6. Gradient safety (mini-check)**:

- Define a scalar objective $J = \|x_T\|^2$ for a 2-body run of 10 steps
- Verify `jax.grad(lambda x0: loss(run(x0)))` returns finite values in float64
- **Purpose**: Ensures your integrator is differentiable (critical for final project physics-informed learning)

**N-body cluster validation** (included in your 50-100 simulation ensemble):

- Generate realistic clusters (N=200-500, Kroupa IMF, Plummer sphere, $\alpha_{\rm vir} = 1$)
- Integrate for several dynamical times
- Check that $\alpha_{\rm vir}$ remains near 1 (some evolution is physical)
- Verify momentum conservation

---

## Deliverables

### 1. Code Repository

**Structure**:

**Recommended directory layout** (your first Python package!):

```bash
project5-jax-nbody/
├── jax_nbody/              # Your package directory
│   ├── __init__.py         # Makes this a package (can be empty)
│   ├── samplers.py         # IMF, Plummer sampling
│   ├── forces.py           # Force calculations
│   ├── integrators.py      # Leapfrog, adaptive dt
│   └── diagnostics.py      # Energy, virial ratio
├── tests/
│   ├── test_forces.py      # At least 6 tests total
│   ├── test_samplers.py
│   └── ...
├── quickstart.py           # Demo script for grading
├── run_ensemble.py         # Generate 50-100 simulations
├── data/                   # Simulation outputs
│   ├── sim_0000/
│   └── ...
└── README.md
```

**Why this structure?**

- **`__init__.py`**: Makes `jax_nbody/` a Python package so you can `import jax_nbody.forces`
- **Modular files**: Separate concerns (sampling vs. forces vs. integration)
- **`tests/` directory**: Keep tests organized
- **Scripts at root**: `quickstart.py` and `run_ensemble.py` import from your package

**Example imports in your scripts**:

```python
# In quickstart.py
from jax_nbody.samplers import sample_kroupa_imf, sample_plummer_positions
from jax_nbody.forces import compute_forces_matrix
from jax_nbody.integrators import leapfrog_integrate

# Or if you organize differently:
import jax_nbody.samplers as samplers
```

**What you DON'T need**:

- ❌ `pyproject.toml` or `setup.py` (no pip installation)
- ❌ Complex packaging machinery
- ❌ Version numbers, dependencies lists

**What you DO need**:

- ✅ `__init__.py` in your package directory (can be empty)
- ✅ Organized modules with clear responsibilities
- ✅ Tests that import and verify your functions
- ✅ Scripts that use your package

**Code quality** (optional): Consider using `ruff check jax_nbody/` for linting, but not required

**Git**: 20+ meaningful commits showing development progression

**RNG threading**: All samplers and integrators must accept and return a JAX `PRNGKey`, with no hidden global RNG state. Tests will fail if randomness isn't threaded explicitly through function calls

**Quickstart for grading**: `quickstart.py` at project root that runs N=64 for 100 steps on CPU and prints: wall-time (after JIT warmup), `max|ΔE/E|`, and $\alpha_{\rm vir}$ summary. I will run `python quickstart.py` to verify basic functionality

### 2. Technical Report (4-5 pages)

Write a technical report demonstrating your understanding of JAX and the design tradeoffs you made. Your report must include:

**Required Components**:

1. **Algorithm Design** (1-1.5 pages):
   - Initial conditions implementation (IMF, Plummer, virial velocities)
   - Force calculation strategy choice and justification
   - Integration scheme (leapfrog DKD)
   - Adaptive timestepping approach
   - **Softening policy**: State your $\epsilon$ choice (constant vs. $\epsilon \propto r_h/\sqrt{N}$), the value/factor used, and 2-line justification tied to energy drift and timestep collapse rates

2. **Performance Analysis** (1-1.5 pages):
   - JAX vs. NumPy speedup (quantified with timings)
   - **Force strategy comparison table**: Matrix vs. vmap for N=32, 64, 128 showing measured error and wall time
   - Scaling analysis: How does runtime scale with N?
   - Profiling insights (what dominates runtime?)

3. **Validation Results** (1 page):
   - Evidence that all 6 required tests pass (include key figures/values)
   - Two-body orbit energy conservation plot
   - Virial equilibrium verification ($\alpha_{\rm vir} = 1 \pm 0.05$)
   - Multi-scale validation (N=2 → N=100+ → N=200-500 clusters)

4. **Production Pipeline** (0.5-1 page):
   - Automated simulation ensemble description (50-100 runs)
   - Parameter space coverage (N, scale length $a$, mass range)
   - Statistical validation plot demonstrating ensemble consistency
   - Data organization and reproducibility approach

**Format**: Use figures effectively. Quantify everything. Show you understand the tradeoffs, not just "it works."

### 3. AI as a Learning Accelerator

Through Projects 1-4, you've demonstrated solid Python expertise. Now it's time to leverage that foundation with modern learning tools. JAX has a sharp learning curve—its functional programming paradigm and compilation constraints are fundamentally different from NumPy. But you have an advantage: **you already understand the underlying physics and computational concepts**. 

**Use AI to understand, not to generate code.** You need to write your own implementation to truly learn JAX. Instead, use AI tools (e.g., ChatGPT, Claude, Gemini) to:

- **Explain difficult programming concepts**: "What does it mean for JAX arrays to be immutable?" or "Why can't I use Python `if` statements in JIT-compiled functions?"
- **Decode error messages**: JAX errors can be verbose and cryptic. Paste the error and ask "What is this tracer error telling me?"
- **Trace issues**: "My code runs but energy conservation is terrible—what are common causes in JAX?"
- **Compare paradigms**: "I know how to do X in NumPy with mutable arrays, how does JAX's functional approach differ conceptually?"

**Learning from mistakes is how you master new frameworks.** It's common—expected, even—to encounter JAX-specific pitfalls. These errors (tracer errors, array immutability violations, JIT compilation failures) are part of the learning process. When you hit one, use AI to understand *why* it happened and *what concept* you're missing, then fix it yourself.

**Critical caveat**: Maintain a **"docs first" mindset**. Check the [JAX documentation](https://jax.readthedocs.io/) and [Equinox API](https://docs.kidger.site/equinox/) before asking AI. Always fact-check AI responses—they're often confident but wrong about version-specific details or edge cases. Use AI to understand concepts and documentation, not as a replacement for understanding.

In your technical report, briefly describe how you used AI tools (if at all): What helped you understand JAX's paradigm? What errors did you learn from? This isn't graded for "correct" AI use—we want honest data to improve the course.

---

## Common Pitfalls & Debugging Guide

| **Issue** | **Symptom** | **Cause** | **Solution** | **Lesson** |
|-----------|------------|-----------|--------------|------------|
| "Array is not writable" | `ValueError` when trying to modify array | Tried to mutate JAX array | Use `.at[].set()` or build new array | JAX arrays are **values**, not containers |
| "Tracer error during JIT" | `TypeError` with abstract tracer | Python `if` statement on traced value | Use `jax.lax.cond()` for conditionals, `jax.lax.while_loop()` for loops, or restructure without data-dependent control flow | JIT traces **data flow**, not control flow |
| Recompiling on every call | Slow even after warmup | Shape-changing arrays or data-dependent control flow | Ensure jitted functions have static shapes; avoid constructing arrays whose shape depends on runtime data inside jitted loops. **Actionable**: Preallocate and carry fixed-size states in scan—do not reallocate inside scan. | **Static shapes required for compilation** |
| First run very slow, then fast | Long pause on first call | JIT compilation overhead | Normal behavior—first call compiles, subsequent calls use compiled code | **Warmup** before timing |
| Energy drift worse than NumPy | $\|dE/E\| > 10^{-6}$ | Using float32 (JAX default) | Set `jax.config.update("jax_enable_x64", True)` at top of module | Precision matters for physics |
| Non-reproducible results | Different answers each run | Using Python `random` instead of `jax.random` | Always use `jax.random.PRNGKey` and explicit splits | JAX manages randomness explicitly |
| $\alpha_{\rm vir} \neq 1$ initially | Cluster immediately expands/collapses | Velocity initialization wrong | Check acceptance-rejection or velocity dispersion formula carefully | **Initial conditions are physics** |
| Tests pass, simulation explodes | Insufficient validation coverage | Only tested N=2, not realistic clusters | Validate with N=100+ cluster for several dynamical times | Multi-scale validation essential |
| Code hangs during integration | Timestep → 0 | Very close encounters, $\epsilon$ too small | Use appropriate softening: $\epsilon \sim 0.3 r_h/\sqrt{N}$ | Softening prevents numerical issues |
| Massive memory usage | Out of memory error | Creating $O(N^2)$ arrays without JIT | Ensure force calculation is JIT-compiled; check for memory leaks in loops | JIT optimizes memory |

### Debugging Strategy

When something goes wrong (and it will):

1. **Start small**: Does your code work for N=2? N=3? If not, fix that first.
2. **Check conservation laws**: Energy, momentum, angular momentum are your physics smoke detectors
3. **Visualize**: Plot trajectories, $\alpha_{\rm vir}(t)$, $\Delta t(t)$ — your eyes catch patterns tests miss
4. **Compare**: Does your JAX code give the *same answer* as Project 2 NumPy code for identical initial conditions?
5. **Bisect**: Comment out half your code. Which half has the bug?
6. **Read error messages carefully**: JAX errors are verbose but informative

**Most common mistake**: Assuming initial conditions are correct. Always validate IMF slopes, Plummer density, and $\alpha_{\rm vir}$ *before* running long integrations.

---

## Essential JAX Functions & Tools

### Core JAX

| Function | Purpose |
|----------|---------|
| `jax.jit` | JIT compilation for speed |
| `jax.vmap` | Vectorize over batch dimension |
| `jax.grad` | Automatic differentiation |
| `jax.lax.scan` | Efficient loops for timestepping |
| `jax.lax.while_loop` | Bounded loops (e.g., rejection sampling) |
| `jax.random.PRNGKey` | Create random seed |
| `jax.random.split` | Split RNG state |
| `jax.random.uniform` | Uniform distribution |
| `jax.random.normal` | Normal distribution |
| `jax.numpy` (as `jnp`) | NumPy replacement |

*Note*: `jax.lax.scan` and `jax.lax.while_loop` are critical for JIT-compiled loops—do not use Python `for` or `while` inside jitted functions. Additionally, when compiled `jax.lax.scan` is preferred for fixed-iteration loops (like timestepping) and is faster than `jax.lax.while_loop` and `jax.lax.ifor_loop`, while `jax.lax.while_loop` is for data-dependent loops (like rejection sampling).

### Performance Tools

| Tool | Purpose |
|------|---------|
| `jax.profiler.trace` | Profile execution |
| `%timeit` (IPython)  or `import time` | Benchmark with warmup |
| `time.time()` | Manual timing |

### Optional Tools

| File/Tool | Purpose |
|-----------|---------|
| `pytest` | Testing framework (optional, but tests required) |
| `ruff` | Linting and formatting (recommended) |
| `pyproject.toml` | Package metadata (only if you want installable package) |

---

## Grading Approach

**Evaluation Method**: Your grade is based on the technical report, quickstart demo, and repository inspection (code reading, not installation).

| Component | Weight | Evaluation Method |
|-----------|--------|-------------------|
| **Technical Report** | 40% | Quality of analysis, figures, design justification, performance results, validation evidence |
| **Quickstart Demo** | 20% | Does `quickstart.py` run and produce correct physics (energy conservation, $\alpha_{\rm vir} \approx 1$)? |
| **Code Quality** | 25% | Repository inspection: JAX patterns correct, tests present, clean structure, documentation |
| **Validation Evidence** | 15% | Report shows all 6 required tests passed with figures/tables proving correctness |

**Note**: Grading will be based on your quickstart demo output, repository code reading, and comprehensive technical report. I will not be installing your package or running your full test suite—your report must provide convincing evidence that your implementation works correctly.

---

## Looking Ahead: Final Project

The automated simulation pipeline you built is the **foundation for your final ML project**:

- **Training data generation**: Your 50-100 simulations are a prototype. For the final project, you'll scale to 1000+ simulations to train Gaussian Processes and Neural Networks
- **Fast ground truth**: JAX's 10-100× speedup makes this computationally feasible
- **Differentiable physics**: JAX's `grad` enables physics-informed learning (gradients through entire simulations)
- **Research-grade infrastructure**: Professional package + automated pipeline = reproducible science

**What's next in the final project**:

- Gaussian Processes: Learn cluster evolution from your simulation ensemble
- Neural Networks: Emulate N-body dynamics at 1000× speedup
- Uncertainty quantification: How confident should we be in ML predictions?

The computational infrastructure you build now directly enables machine learning for astrophysics.

---
