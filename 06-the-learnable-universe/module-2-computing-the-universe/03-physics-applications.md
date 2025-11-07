---
title: "Part 3: JAX for Physics Simulations"
subtitle: "Application Patterns for N-body Systems, ODEs, and Optimization | Computing the Universe | ASTR 596"
---

**Prerequisites**: [Part 2: Core Transformations](part-02-core-transformations.md) completed

> *"The scientist does not study nature because it is useful; he studies it because he delights in it, and he delights in it because it is beautiful."* — Henri Poincaré
>
> *"Make it work, make it right, make it fast."* — Kent Beck

---

(learning-outcomes-part3)=
## Learning Outcomes

By the end of Part 3, you will be able to:

- [ ] **Design** JAX-compatible function signatures for physics simulations
- [ ] **Identify** where to apply grad/jit/vmap in N-body, ODE, and optimization problems
- [ ] **Structure** code to separate pure computation from I/O and visualization
- [ ] **Choose** appropriate integration schemes (lax.scan vs explicit loops) for time evolution
- [ ] **Validate** JAX implementations against NumPy baselines
- [ ] **Benchmark** performance improvements and understand their sources
- [ ] **Plan** your Project 5 refactoring strategy

---

(roadmap-part3)=
## Roadmap: From Tools to Applications

**Priority: 🔴 Essential**

**Part 2 taught you JAX transformations**. Part 3 shows **how to apply them to physics**.

You can now:

- ✅ Compute numerically exact gradients with `grad` (machine precision)
- ✅ Compile functions with `jit`
- ✅ Vectorize operations with `vmap`
- ✅ Compose transformations correctly

**Part 3 teaches application patterns**:

1. **N-body systems**: Forces, integration, energy conservation
2. **ODE solving**: Stellar structure, chemical networks, differential equations
3. **Optimization**: Parameter fitting, likelihood maximization
4. **Best practices**: Testing, validation, performance benchmarking

**This is NOT a cookbook.** We show **patterns and design choices**, not complete implementations. Project 5 is where you write the code.

**Goal**: By the end, you'll know:
- Which parts of your Project 2 code to refactor first
- Where to apply each transformation
- How to validate correctness
- What speedups to expect

---

(nbody-jax)=
## 3.1: N-body Systems with JAX

**Priority: 🔴 Essential**

### Architectural Overview

**Your Project 2 N-body code** has this structure:

```python
# NumPy version (from Project 2)
def simulate_nbody(positions, velocities, masses, dt, n_steps):
    """Integrate N-body system for n_steps."""
    trajectory = []
    
    for step in range(n_steps):
        # Compute forces (nested loops)
        forces = compute_forces(positions, masses)
        
        # Leapfrog integration
        velocities += 0.5 * dt * forces / masses
        positions += dt * velocities
        forces = compute_forces(positions, masses)
        velocities += 0.5 * dt * forces / masses
        
        # Store trajectory
        trajectory.append(positions.copy())
    
    return np.array(trajectory)

def compute_forces(positions, masses):
    """Compute gravitational forces via nested loops."""
    N = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.sqrt(np.sum(r_vec**2) + 1e-10)
                forces[i] += G * masses[i] * masses[j] * r_vec / r**3
    
    return forces
```

**JAX refactoring strategy** (what you'll do in Project 5):

1. **Replace force loops with vmap** → 100× faster force calculation
2. **Add jit to integration step** → 10× faster timestep
3. **Use lax.scan for trajectory** → Enables JIT of entire simulation
4. **Optional: Add grad for exact forces** → Compare to finite differences

**We'll discuss each transformation, not implement it for you.**

---

### Step 1: Vectorizing Forces with `vmap`

**Current bottleneck**: Nested loops in `compute_forces`.

**Key insight**: Force calculation is naturally parallel—each pairwise interaction is independent.

**Design choice**: Compute all pairwise forces at once, then sum per particle.

**Pseudocode structure** (you implement in Project 5):

```python
def pairwise_force(r_i, r_j, m_i, m_j):
    """Force on particle i from particle j.
    
    This is the atomic operation you'll write.
    Pure function: no side effects, deterministic.
    """
    r_vec = r_j - r_i
    r = jnp.sqrt(jnp.sum(r_vec**2) + 1e-10)  # Softening
    force = G * m_i * m_j * r_vec / r**3
    return force

# Then apply vmap twice (you figure out in_axes):
# - Once to vectorize over j (all particles acting on i)
# - Once to vectorize over i (all particles feeling forces)
compute_forces_vectorized = jax.vmap(...)  # You design this
```

**Questions to answer in Project 5**:
- What are the correct `in_axes` for double vmap?
- How do you exclude self-interaction (i==j)?
- Should you compute full $N \times N$ force matrix or just upper triangle?
- How do you sum forces correctly?

**Expected result**: Forces computed in one vectorized operation, no Python loops.

**Benchmark target**: 50-200× faster than nested loops for N=100 on GPU.

---

### Step 2: Compiling Integration with `jit`

**After vectorizing forces**, compile the timestep:

```python
@jax.jit
def leapfrog_step(state, dt):
    """One leapfrog timestep.
    
    state = (positions, velocities, masses)
    Returns: new_state
    
    Pure function: no I/O, no prints, deterministic.
    """
    positions, velocities, masses = state
    
    # Kick-drift-kick (you implement)
    forces = compute_forces_vectorized(positions, masses)
    velocities = velocities + 0.5 * dt * forces / masses[:, None]
    positions = positions + dt * velocities
    forces = compute_forces_vectorized(positions, masses)
    velocities = velocities + 0.5 * dt * forces / masses[:, None]
    
    return (positions, velocities, masses)
```

**Design principles**:
1. **Pure function**: No side effects (no `trajectory.append()` inside)
2. **Fixed shapes**: `positions.shape = (N, 3)` never changes
3. **Deterministic**: Same inputs → same outputs

**Why JIT helps**: Fuses force calculation + integration into single compiled kernel.

**Benchmark target**: 10-20× faster than NumPy per timestep.

---

### Step 3: Trajectory Integration with `lax.scan`

**Problem**: You need to store trajectory over time. Can't use `trajectory.append()` inside JIT.

**Solution**: Use `lax.scan` to iterate while JIT-compiled.

**Pattern** (you implement):

```python
from jax import lax

def scan_body(carry, t):
    """Body function for lax.scan.
    
    carry: state that persists (positions, velocities, masses)
    t: iteration counter (can use for diagnostics)
    
    Returns: (new_carry, output_to_stack)
    """
    state = carry
    new_state = leapfrog_step(state, dt)
    
    # What to output? Options:
    # - Full state (memory intensive)
    # - Just positions (for visualization)
    # - Energy + angular momentum (for validation)
    # - Downsampled (every 10th step)
    
    output = new_state[0]  # Just positions (you decide)
    return new_state, output

# Integrate
initial_state = (positions_0, velocities_0, masses)
final_state, trajectory = lax.scan(
    scan_body,
    initial_state,
    jnp.arange(n_steps)
)
# trajectory.shape = (n_steps, N, 3)
```

**Key decisions you'll make**:
- Store full trajectory or downsample?
- Track energy/momentum for validation?
- How to handle diagnostics without breaking JIT?

**Alternative**: If trajectory doesn't fit in memory, use checkpointing or only store final state.

---

### Step 4: Energy Conservation Validation

**Hamiltonian dynamics conserves energy**. Your JAX implementation should too.

:::{important} Precision Matters for Energy Conservation

Energy conservation tests are **extremely sensitive to floating-point precision**. Before running any N-body simulation, you MUST enable 64-bit precision:

```python
import jax
jax.config.update("jax_enable_x64", True)  # Essential!
```

**Why**: Roundoff errors accumulate over thousands of timesteps. With default float32, even a perfect symplectic integrator will show visible energy drift.
:::

---

#### Energy Test Pattern

```python
def total_energy(positions, velocities, masses, G=6.67e-8, eps2=1e20):
    """Kinetic + potential energy.

    Args:
        positions: (N, 3) array
        velocities: (N, 3) array
        masses: (N,) array
        G: Gravitational constant (CGS: 6.67e-8)
        eps2: Softening length squared (e.g., 1e20 cm^2 for typical clusters)
    """
    # Kinetic energy: T = (1/2) * sum(m * v^2)
    kinetic = 0.5 * jnp.sum(masses[:, None] * velocities**2)

    # Potential energy: U = -sum_{i<j} G*m_i*m_j / r_ij (with softening)
    potential = gravitational_potential(positions, masses, G=G, eps2=eps2)

    return kinetic + potential

# Compute energy at each timestep
energies = jax.vmap(total_energy, in_axes=(0, 0, None, None, None))(
    trajectory_pos,
    trajectory_vel,
    masses,
    G,
    eps2
)

# Check conservation
initial_energy = energies[0]
energy_drift = jnp.abs(energies - initial_energy) / jnp.abs(initial_energy)

print(f"Initial energy: {initial_energy:.6e} erg")
print(f"Final energy: {energies[-1]:.6e} erg")
print(f"Max relative drift: {jnp.max(energy_drift):.2e}")
print(f"Mean relative drift: {jnp.mean(energy_drift):.2e}")
```

---

#### Expected Energy Drift: Precision-Dependent Targets

**Energy conservation accuracy depends critically on dtype**:

| Configuration | Expected Drift | Notes |
|---------------|----------------|-------|
| **float64 + symplectic + small dt** | $< 10^{-12}$ | Publication quality |
| **float64 + symplectic + moderate dt** | $10^{-10}$ to $10^{-8}$ | Good for most physics |
| **float32 + symplectic** | $10^{-6}$ to $10^{-5}$ | Not suitable for long integrations |
| **float32 + Euler** | $> 10^{-3}$ | Unstable, energy grows |

:::{admonition} Rule of Thumb
:class: tip

**If your energy drift is worse than expected**:

1. **Check dtype first**: `print(positions.dtype)` → should be `float64`
2. **Check timestep**: Is $\Delta t < 0.01 \times T_{\text{dynamical}}$?
3. **Check softening**: Is $\epsilon$ consistent between potential and forces?
4. **Check integrator**: Are you using leapfrog or another symplectic method?

**Common mistake**: Forgetting to enable float64 → immediate $10^{-6}$ floor on accuracy!
:::

---

#### Softening Effects on Energy Definition

**Softening changes the Hamiltonian**. Your energy definition must match your force law.

**Example**: Plummer softening

```python
def gravitational_potential_softened(positions, masses, G=6.67e-8, eps2=1e20):
    """Potential with Plummer softening: U = -G*m1*m2 / sqrt(r^2 + eps^2)."""
    N = len(masses)
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r_vec = positions[j] - positions[i]
            r2 = jnp.sum(r_vec**2)
            r_soft = jnp.sqrt(r2 + eps2)  # Softened distance
            U -= G * masses[i] * masses[j] / r_soft
    return U
```

**Critical**: If forces use softening $\epsilon^2$, then potential MUST use same $\epsilon^2$. Mismatch → energy not conserved by construction!

:::{margin}
**Softening bias**: Even with perfect float64 and symplectic integration, softening introduces a **systematic shift** in total energy (typically < 1% for well-chosen $\epsilon$). This is **not drift**—it's a change in the Hamiltonian itself.
:::

---

#### Debugging Energy Conservation Failures

**If energy drift exceeds targets**:

1. **Verify precision**:

```python
print(f"positions dtype: {positions.dtype}")  # Must be float64
print(f"velocities dtype: {velocities.dtype}")  # Must be float64
assert positions.dtype == jnp.float64, "Enable jax_enable_x64!"
```

2. **Check softening consistency**:

```python
# Force calculation
def force_i_from_j(r_i, r_j, m_i, m_j, G, eps2):
    r_vec = r_j - r_i
    r2 = jnp.sum(r_vec**2) + eps2  # <-- Same eps2 as potential!
    r = jnp.sqrt(r2)
    return G * m_i * m_j * r_vec / (r * r2)
```

3. **Visualize drift over time**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(energies / energies[0])
plt.axhline(1.0, color='k', linestyle='--', alpha=0.3)
plt.ylabel('E(t) / E(0)')
plt.xlabel('Timestep')
plt.title('Energy Conservation')

plt.subplot(1, 2, 2)
plt.semilogy(energy_drift)
plt.ylabel('|ΔE| / E₀')
plt.xlabel('Timestep')
plt.title('Relative Energy Drift (log scale)')
plt.tight_layout()
plt.show()
```

4. **Compare integrators**:

```python
# Good: Leapfrog (symplectic, 2nd order)
# Acceptable: Velocity Verlet (symplectic, 2nd order)
# Bad: Euler (not symplectic, 1st order—energy grows!)
```

---

**Why this matters**: If energy drifts significantly beyond precision limits:

- ❌ Integration scheme may be wrong (not symplectic)
- ❌ Forces have bugs (check vectorization, softening)
- ❌ Timestep too large (reduce $\Delta t$)
- ❌ **Most common**: Forgot to enable float64!

JAX's numerically exact gradients + symplectic integration + float64 precision should conserve energy to $\sim 10^{-12}$ relative accuracy over thousands of timesteps.

---

### Optional: Computing Forces via Autodiff

**Alternative to analytical forces**: Compute gradient of potential.

```python
def gravitational_potential(positions, masses):
    """Total potential energy of system."""
    # Compute all pairwise potentials, sum
    # (You implement with vmap)
    return U_total

# Forces are negative gradient
compute_forces_autodiff = jax.grad(gravitational_potential, argnums=0)

# Compare to your analytical forces
forces_analytical = compute_forces_vectorized(positions, masses)
forces_autodiff = -compute_forces_autodiff(positions, masses)

# Should agree to machine precision
difference = jnp.max(jnp.abs(forces_analytical - forces_autodiff))
print(f"Force difference: {difference:.2e}")  # Should be ~1e-15
```

**When to use autodiff forces**:
- Complex potential (not just gravity)
- Want to be sure forces are correct
- Potential is easier to write than forces

**When to use analytical forces**:
- Simple potential (gravity, Coulomb)
- Want maximum performance (skip potential calculation)

---

:::{admonition} 🔗 Connection to Module 3: Phase Space and Hamiltonian Dynamics
:class: note

**Module 3 taught**: Hamiltonian dynamics preserves phase space volume (Liouville's theorem) and energy (if $H$ is time-independent).

**JAX enables exact Hamiltonian evolution**:
- Symplectic integrators (leapfrog) preserve structure numerically
- Autodiff gives exact $\nabla H$ (no finite-difference errors)
- Energy conservation validates both integrator and forces

Your Project 5 N-body system is a **computational realization of Hamiltonian mechanics** from Module 3. Energy drift <$10^{-10}$ over 10,000 orbits is evidence you've implemented it correctly.
:::

---

(ode-solving-jax)=
## 3.2: ODE Solving with JAX

**Priority: 🟡 Important**

### When You Need ODE Solvers

**Astrophysics is full of coupled ODEs**:
- **Stellar structure**: $\frac{dM}{dr}, \frac{dP}{dr}, \frac{dT}{dr}, \frac{dL}{dr}$ (Module 2)
- **Chemical networks**: $\frac{dn_i}{dt} = \sum_j R_{ij}(n_j, T)$ (reaction rates)
- **Orbit determination**: $\frac{d\mathbf{r}}{dt} = \mathbf{v}, \frac{d\mathbf{v}}{dt} = \mathbf{a}(\mathbf{r})$
- **Radiative transfer**: $\frac{dI}{ds} = -\kappa I + j$ (intensity along ray)

**JAX ODE solving**: Use **Diffrax** (JAX-native ODE library) or implement custom integrators.

---

### Pattern: Stellar Structure Equations

**Module 2 equations** (Eddington's 4 coupled ODEs):

$$
\frac{dM}{dr} = 4\pi r^2 \rho, \quad
\frac{dP}{dr} = -\frac{GM\rho}{r^2}, \quad
\frac{dT}{dr} = -\frac{3\kappa L}{16\pi a c r^2 T^3}, \quad
\frac{dL}{dr} = 4\pi r^2 \rho \epsilon
$$

**JAX structure** (using Diffrax):

```python
import diffrax

def stellar_structure_rhs(r, y, args):
    """Right-hand side of stellar structure ODEs.
    
    y = [M, P, T, L] at radius r
    args = (rho_c, opacity_func, epsilon_func)
    
    Returns: dy/dr = [dM/dr, dP/dr, dT/dr, dL/dr]
    
    Pure function: deterministic, no side effects.
    """
    M, P, T, L = y
    rho_c, opacity_func, epsilon_func = args
    
    # Compute density from P, T (EOS)
    rho = equation_of_state(P, T)
    
    # Compute opacity and energy generation
    kappa = opacity_func(rho, T)
    epsilon = epsilon_func(rho, T)
    
    # Derivatives
    dM_dr = 4 * jnp.pi * r**2 * rho
    dP_dr = -G * M * rho / r**2
    dT_dr = -3 * kappa * L / (16 * jnp.pi * a * c * r**2 * T**3)
    dL_dr = 4 * jnp.pi * r**2 * rho * epsilon
    
    return jnp.array([dM_dr, dP_dr, dT_dr, dL_dr])

# Solve from center to surface
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(stellar_structure_rhs),
    diffrax.Dopri5(),  # Adaptive RK45
    t0=r_center,
    t1=r_surface,
    dt0=dr_initial,
    y0=jnp.array([M_center, P_center, T_center, L_center]),
    args=(rho_c, opacity_func, epsilon_func),
    saveat=diffrax.SaveAt(ts=r_grid)
)
# solution.ys = [M(r), P(r), T(r), L(r)] on grid
```

**Why Diffrax**:
- Adaptive timestepping (handles stiff equations)
- JIT-compilable (fast repeated solves)
- Autodiff through solutions (sensitivity analysis)
- Many solvers (RK4, Dopri5, implicit methods)

---

### Gradients Through ODE Solutions

**Powerful application**: Optimize initial conditions or parameters by taking gradients through the solver.

**Example**: Find central density $\rho_c$ that produces correct stellar radius.

```python
def stellar_radius(rho_c, opacity_func, epsilon_func):
    """Solve structure equations, return surface radius.
    
    This is differentiable!
    """
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(stellar_structure_rhs),
        diffrax.Dopri5(),
        t0=0.0,
        t1=R_max,  # Integrate until surface
        dt0=1e-3,
        y0=initial_conditions(rho_c),
        args=(rho_c, opacity_func, epsilon_func),
    )
    # Extract radius where P=0 (surface)
    return find_surface_radius(solution)

# Gradient: how does radius change with central density?
dR_drho_c = jax.grad(stellar_radius)(rho_c, opacity_func, epsilon_func)

# Use in optimization: match observed radius
def loss(rho_c):
    R_model = stellar_radius(rho_c, opacity_func, epsilon_func)
    R_obs = 6.96e10  # Solar radius in cm
    return (R_model - R_obs)**2

# Gradient descent to find correct rho_c
rho_c_optimal = optimize_with_grad(loss, rho_c_init)
```

**Why this is powerful**: Autodiff gives exact sensitivity without finite differences. Enables efficient root-finding and parameter fitting.

---

:::{admonition} 🔗 Connection to Module 2: Stellar Structure
:class: note

**Module 2**: You derived Eddington's equations from hydrostatic equilibrium, energy transport, and nuclear physics.

**JAX + Diffrax**: You can now solve these equations numerically, differentiate through them, and optimize stellar models.

**Example application**: 
- Given observed $M_\star$ and $R_\star$, find $\rho_c$ and composition that match
- Compute $\frac{\partial R_\star}{\partial Z}$ (radius sensitivity to metallicity)
- Infer stellar age from surface temperature evolution

This is **computational stellar astrophysics** in practice.
:::

---

(optimization-jax)=
## 3.3: Optimization Patterns

**Priority: 🔴 Essential**

### Gradient-Based Optimization

**Common astrophysics problem**: Find parameters $\theta$ that minimize $\chi^2$ or maximize likelihood.

**Examples**:
- Fit period-luminosity relation to Cepheids (Project 1)
- Infer cosmological parameters from supernovae (Project 4)
- Optimize neural network weights (Project 6)

**JAX advantage**: Numerically exact gradients via autodiff (machine precision), fast optimization.

---

### Pattern 1: Least-Squares Fitting

**Problem**: Fit model $y = f(x; \theta)$ to data $(x_i, y_i, \sigma_i)$.

**Loss function**: $\chi^2 = \sum_i \frac{(y_i - f(x_i; \theta))^2}{\sigma_i^2}$

**JAX implementation**:

```python
def model(x, params):
    """Model prediction for single data point."""
    # Your model (linear, power-law, etc.)
    return prediction

def chi_squared(params, x_data, y_data, sigma_data):
    """Chi-squared loss over all data."""
    # Vectorize model over data
    predictions = jax.vmap(model, in_axes=(0, None))(x_data, params)
    residuals = y_data - predictions
    return jnp.sum((residuals / sigma_data)**2)

# Gradient for optimization
grad_chi_squared = jax.jit(jax.grad(chi_squared, argnums=0))

# Gradient descent (simple version)
params = params_init
for step in range(1000):
    gradient = grad_chi_squared(params, x_data, y_data, sigma_data)
    params = params - learning_rate * gradient

# Or use Optax (JAX optimization library)
import optax

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

for step in range(1000):
    loss, gradient = jax.value_and_grad(chi_squared)(params, x_data, y_data, sigma_data)
    updates, opt_state = optimizer.update(gradient, opt_state)
    params = optax.apply_updates(params, updates)
```

**Key points**:
- Use `jax.value_and_grad` to get loss and gradient in one call (efficient)
- Optax provides advanced optimizers (Adam, SGD with momentum, LBFGS)
- JIT the gradient function for speed

---

### Pattern 2: Maximum Likelihood Estimation

**Project 4 application**: Find $(\Omega_m, \Omega_\Lambda, H_0)$ that maximize $P(D|\theta)$.

**Negative log-likelihood** (minimize):

```python
def neg_log_likelihood(theta, data):
    """Negative log-likelihood for cosmological parameters.
    
    theta = [Omega_m, Omega_Lambda, H0]
    data = {'z': redshifts, 'mu_obs': observed_moduli, 'sigma_mu': uncertainties}
    """
    Omega_m, Omega_Lambda, H0 = theta
    
    # Prior (optional: constrain to physical region)
    if Omega_m < 0 or Omega_Lambda < 0 or H0 < 50 or H0 > 100:
        return jnp.inf  # Outside allowed region
    
    # Predicted distance moduli
    mu_pred = distance_modulus_batch(data['z'], H0, Omega_m, Omega_Lambda)
    
    # Gaussian likelihood
    residuals = data['mu_obs'] - mu_pred
    chi2 = jnp.sum((residuals / data['sigma_mu'])**2)
    
    return 0.5 * chi2  # Negative log-likelihood (up to constant)

# Find maximum likelihood parameters
from scipy.optimize import minimize

result = minimize(
    neg_log_likelihood,
    x0=[0.3, 0.7, 70.0],  # Initial guess
    args=(data,),
    jac=jax.grad(neg_log_likelihood),  # Numerically exact gradient (machine precision)!
    method='L-BFGS-B',
    bounds=[(0, 1), (0, 2), (50, 100)]
)

theta_MLE = result.x
print(f"Best-fit: Omega_m={theta_MLE[0]:.3f}, Omega_Lambda={theta_MLE[1]:.3f}, H0={theta_MLE[2]:.1f}")
```

**Why JAX helps**: `jax.grad` provides numerically exact gradient (machine precision), BFGS converges in 10-50 iterations (vs thousands for gradient-free methods).

---

### Pattern 3: Uncertainty Quantification

**After finding best-fit parameters**, estimate uncertainties.

**Method 1: Hessian approximation** (fast, approximate):

```python
# Hessian of negative log-likelihood at MLE
hessian = jax.hessian(neg_log_likelihood)(theta_MLE, data)

# Covariance matrix (inverse Hessian)
covariance = jnp.linalg.inv(hessian)

# Parameter uncertainties
uncertainties = jnp.sqrt(jnp.diag(covariance))
print(f"H0 = {theta_MLE[2]:.1f} ± {uncertainties[2]:.1f} km/s/Mpc")
```

**Method 2: MCMC** (Project 4, more accurate):

```python
# Use JAX-accelerated HMC (from Part 2)
grad_log_posterior = jax.jit(jax.grad(log_posterior, argnums=0))

def hmc_step(theta, key, data):
    # Your HMC implementation using grad_log_posterior
    # (Much faster than finite differences!)
    return theta_new

# Run chain
chain = run_mcmc(hmc_step, theta_init, data, n_steps=10000)
# Analyze posteriors: mean, std, correlations
```

**JAX advantage**: HMC with numerically exact gradients (machine precision) converges 10-100× faster than Metropolis-Hastings.

---

:::{admonition} 🔗 Connection to Project 4: Cosmological MCMC
:class: note

**In Project 4**, you'll implement gradient-based MCMC for supernova cosmology.

**JAX will accelerate**:
1. **Gradient computation**: `jax.grad(log_posterior)` replaces finite differences (100× faster)
2. **Likelihood evaluation**: `jit` compiles distance modulus calculation (10× faster)
3. **Vectorization**: `vmap` over 43 supernovae simultaneously

**Combined speedup**: 1000× faster than pure Python + finite differences.

**This transforms MCMC from "run overnight" to "run in 5 minutes"**—enabling rapid iteration and exploration.
:::

---

(best-practices)=
## 3.4: Best Practices and Testing

**Priority: 🟡 Important**

### Code Structure for JAX

**Separate pure computation from I/O**:

```python
# ✅ GOOD: Pure numerical core
@jax.jit
def compute_observables(state, params):
    """Pure function: state → observables."""
    energy = total_energy(state, params)
    angular_momentum = compute_angular_momentum(state, params)
    return {'energy': energy, 'L': angular_momentum}

# I/O wrapper (not JIT-compiled)
def simulate_with_logging(initial_state, params, n_steps):
    """Run simulation with diagnostics."""
    trajectory = []
    
    for step in range(n_steps):
        state = integration_step(state, params)  # JIT-compiled
        
        if step % 100 == 0:
            # Diagnostics (Python, not JIT)
            obs = compute_observables(state, params)
            print(f"Step {step}: E={obs['energy']:.6e}, L={obs['L']:.6e}")
            trajectory.append(state)
    
    return trajectory
```

**Key principle**: **JIT the hot inner loop, keep I/O in Python wrapper.**

---

### Validation Strategy

**1. Test against NumPy baseline**:

```python
# Generate same random data
key = jax.random.PRNGKey(42)
positions = jax.random.normal(key, (N, 3))
masses = jax.random.uniform(key, (N,))

# Compute forces both ways
forces_numpy = compute_forces_numpy(np.array(positions), np.array(masses))
forces_jax = compute_forces_jax(positions, masses)

# Compare
difference = np.max(np.abs(forces_numpy - np.array(forces_jax)))
print(f"Max difference: {difference:.2e}")
assert difference < 1e-10, "Forces don't match!"
```

**2. Check conservation laws**:

```python
# Energy conservation (Hamiltonian systems)
energies = [total_energy(state) for state in trajectory]
energy_drift = (energies[-1] - energies[0]) / energies[0]
assert abs(energy_drift) < 1e-8, f"Energy drift: {energy_drift:.2e}"

# Angular momentum conservation (isolated system)
L_initial = compute_angular_momentum(trajectory[0])
L_final = compute_angular_momentum(trajectory[-1])
L_drift = jnp.linalg.norm(L_final - L_initial) / jnp.linalg.norm(L_initial)
assert L_drift < 1e-8, f"Angular momentum drift: {L_drift:.2e}"
```

**3. Gradient checks** (autodiff vs finite differences):

```python
def check_gradients(f, x, h=1e-5):
    """Compare autodiff gradient to finite differences."""
    # Autodiff
    grad_autodiff = jax.grad(f)(x)
    
    # Finite differences
    grad_fd = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(h)
        x_minus = x.at[i].add(-h)
        grad_fd[i] = (f(x_plus) - f(x_minus)) / (2*h)
    
    # Compare
    difference = np.max(np.abs(np.array(grad_autodiff) - grad_fd))
    print(f"Gradient difference: {difference:.2e}")
    assert difference < 1e-6, "Gradients don't match!"

# Test on gravitational potential
check_gradients(lambda pos: gravitational_potential(pos, masses), positions)
```

---

### Performance Benchmarking

**Measure what you're optimizing**:

```python
import time

def benchmark_function(f, *args, n_runs=100, warmup=10):
    """Benchmark JIT-compiled function."""
    # Warmup (compilation)
    for _ in range(warmup):
        _ = f(*args)
    
    # Time execution
    start = time.time()
    for _ in range(n_runs):
        _ = f(*args)
    end = time.time()
    
    time_per_call = (end - start) / n_runs
    return time_per_call

# Compare NumPy vs JAX
positions = np.random.randn(100, 3)
masses = np.random.rand(100)

time_numpy = benchmark_function(compute_forces_numpy, positions, masses)
time_jax = benchmark_function(compute_forces_jax, 
                             jnp.array(positions), 
                             jnp.array(masses))

speedup = time_numpy / time_jax
print(f"NumPy: {time_numpy*1e3:.2f} ms")
print(f"JAX: {time_jax*1e3:.2f} ms")
print(f"Speedup: {speedup:.1f}×")
```

**Expected speedups** (from our testing):
- Force calculation (N=100): 50-200× on GPU, 10-50× on CPU
- Full integration step: 10-30× on GPU, 5-15× on CPU
- Full trajectory (1000 steps): 100-500× on GPU, 20-100× on CPU

**Your mileage will vary** based on hardware, problem size, and code quality.

---

### Common Pitfalls

**1. Premature optimization**:
```python
# ❌ DON'T: Optimize before you have working code
# Write NumPy version first, validate correctness, THEN refactor to JAX

# ✅ DO: Make it work, make it right, make it fast
```

**2. Over-JIT-ing**:
```python
# ❌ DON'T: JIT everything including I/O
@jax.jit  # BAD: Contains print and file I/O
def simulate_with_output(state):
    print(f"State: {state}")
    np.savetxt("trajectory.txt", state)
    return next_state(state)

# ✅ DO: JIT only pure numerical kernels
```

**3. Ignoring compilation time**:
```python
# ❌ DON'T: Call JIT function once with different shapes
for N in [10, 20, 50, 100]:  # Recompiles 4 times!
    result = jitted_function(jnp.zeros((N, 3)))

# ✅ DO: Fix shapes or accept compilation overhead
```

---

(project5-preparation)=
## 3.5: Preparing for Project 5

**Priority: 🔴 Essential**

### What You'll Refactor

**Project 5 task**: Convert your Project 2 N-body code to JAX, package as installable library.

**Refactoring checklist**:

1. **Forces**:
   - [ ] Replace nested loops with double vmap
   - [ ] Add softening parameter
   - [ ] Validate against NumPy version
   - [ ] Benchmark speedup

2. **Integration**:
   - [ ] Make leapfrog step JIT-compatible (pure function)
   - [ ] Use lax.scan for trajectory
   - [ ] Add energy/momentum tracking
   - [ ] Test energy conservation

3. **Optional enhancements**:
   - [ ] Compute forces via grad(potential)
   - [ ] Compare autodiff vs analytical forces
   - [ ] Add adaptive timestepping
   - [ ] Multi-GPU support with pmap

4. **Package structure**:
   - [ ] Create `setup.py` or `pyproject.toml`
   - [ ] Write tests (pytest)
   - [ ] Add documentation (docstrings)
   - [ ] Make pip-installable

---

### Refactoring Strategy

**Phase 1: Validate** (Week 1)
- Convert force calculation to JAX
- Test against NumPy (should match to 1e-10)
- Benchmark on small system (N=10)

**Phase 2: Optimize** (Week 1)
- Add JIT to integration step
- Use lax.scan for trajectory
- Benchmark on larger system (N=100)

**Phase 3: Package** (Week 2)
- Structure as installable package
- Write tests and documentation
- Add example scripts
- Test installation on fresh environment

**Phase 4: Extensions** (Week 2, optional)
- Autodiff forces
- Advanced diagnostics
- Performance profiling
- GPU optimization

---

### Expected Outcomes

**After Project 5, you will**:
- Have production-quality N-body code (10-500× faster than Project 2)
- Understand JAX transformation patterns deeply
- Be able to refactor any numerical code to JAX
- Have experience packaging scientific software
- Be prepared for Module 7 (neural networks need same patterns)

**This is the culmination of your "glass-box" journey**: From writing Newton's equations from scratch (Project 2) → understanding computational patterns (Module 6) → building professional scientific software (Project 5).

---

## Synthesis: From Transformations to Physics

**Module 6 Parts 1-3 complete arc**:

**Part 1**: Why functional programming? (Purity enables transformations)
**Part 2**: How do transformations work? (grad, jit, vmap mechanics)
**Part 3**: How do we apply them? (N-body, ODEs, optimization patterns)

**Unified by one principle**: **Pure functions + structured control flow → powerful automated transformations**

### The JAX Mindset

**Traditional scientific computing**:
1. Write loops
2. Optimize by hand (vectorize, parallelize, compile)
3. Debug performance issues
4. Repeat for each new problem

**JAX scientific computing**:
1. Write pure functions (single example)
2. Apply transformations (vmap, jit, grad)
3. Validate correctness
4. Benchmark (usually 10-1000× faster automatically)

**The shift**: From "optimizing code" to "designing pure functions that can be optimized."

---

### Connections Across Course

**Module 1 (Statistics)** → **Module 6 (JAX)**:
- CLT enabled inference from samples → vmap enables inference from batches
- MaxEnt found optimal distributions → transformations find optimal code

**Module 2 (Stellar Structure)** → **Module 6 (JAX)**:
- Differential equations describe stars → Diffrax solves them
- Boundary conditions constrain solutions → grad optimizes parameters

**Module 3 (Phase Space)** → **Module 6 (JAX)**:
- Hamiltonian dynamics conserve energy → Symplectic JAX integration does too
- Liouville theorem preserves volume → Pure functions preserve determinism

**Module 4 (Radiative Transfer)** → **Module 6 (JAX)**:
- Monte Carlo sampling → jit-compiled random walks (next step: GPU acceleration)
- Path tracing → vmap over photons

**Module 5 (MCMC)** → **Module 6 (JAX)**:

- Gradient-based sampling (HMC) → grad provides numerically exact gradients (machine precision)
- Likelihood evaluation → jit + vmap accelerate computation

**Module 7 (Machine Learning)** ← **Module 6 (JAX)**:
- Neural networks are just differentiable functions
- Training = gradient descent with jit(vmap(grad(loss)))
- Same patterns, different applications

---

## Understanding Checklist

Before Project 5, ensure you can:

- [ ] **Design** JAX-compatible function signatures (pure, deterministic, fixed shapes)
- [ ] **Identify** where to apply grad/jit/vmap in your N-body code
- [ ] **Explain** why lax.scan is needed for JIT-compiled loops
- [ ] **Validate** JAX implementation against NumPy baseline
- [ ] **Benchmark** performance and understand speedup sources
- [ ] **Structure** code separating pure computation from I/O
- [ ] **Plan** your Project 5 refactoring (forces → integration → packaging)

**If you answered "yes" to all** → Ready for Project 5 ✅

**If uncertain on 3+ items** → Review relevant Part 2/3 sections, ask in office hours

---

## Next Steps

**Immediate**: Start thinking about your Project 5 refactoring strategy
- What parts of your N-body code are candidates for vmap?
- Where will JIT help most?
- What validation tests do you need?

**Week 1 of Project 5**: Implement force calculation with vmap, validate
**Week 2 of Project 5**: Add JIT + lax.scan, package, document

**Then**: Module 7 uses these same JAX patterns for neural networks
- Forward pass = function composition
- Backward pass = grad (autodiff)
- Training = jit(vmap(grad(loss)))
- Prediction = vmap(forward_pass)

**You're now equipped to build high-performance scientific software.** The transformation from "script writer" to "scientific software engineer" is complete.

---

:::{admonition} 📚 Additional Resources
:class: tip

**JAX Ecosystem**:
- [Diffrax](https://docs.kidger.site/diffrax/): ODE/SDE solving in JAX
- [Optax](https://optax.readthedocs.io/): Gradient-based optimization
- [JAX-Cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo): Differentiable cosmology

**Scientific Computing with JAX**:
- [JAX MD](https://github.com/jax-md/jax-md): Molecular dynamics in JAX
- [Blackjax](https://blackjax-devs.github.io/blackjax/): MCMC in JAX
- [Neural ODEs in JAX](https://implicit-layers-tutorial.org/): Differentiable physics

**Packaging Python Projects**:
- [Python Packaging Guide](https://packaging.python.org/)
- [Testing with pytest](https://docs.pytest.org/)
:::
