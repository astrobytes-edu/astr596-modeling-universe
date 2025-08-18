# JAX Advanced Patterns: Control Flow and Optimization

## Learning Objectives
By the end of this chapter, you will:
- Master JAX control flow primitives (cond, scan, while_loop)
- Implement custom derivatives and VJP rules
- Optimize memory and performance for large-scale problems
- Use sharding for multi-GPU computations
- Debug and profile JAX programs effectively
- Build production-ready astronomical simulations

## Control Flow in JAX

### Conditional Execution with lax.cond

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax.lax import cond, scan, while_loop, fori_loop, switch
from jax import random, custom_vjp, custom_jvp
import time
import matplotlib.pyplot as plt

def jax_control_flow():
    """Master JAX's functional control flow primitives."""
    
    print("JAX CONTROL FLOW PRIMITIVES")
    print("=" * 50)
    
    # 1. Conditional execution with cond
    print("\n1. CONDITIONAL EXECUTION:")
    
    @jit
    def stellar_evolution_step(age, mass):
        """
        Evolve star based on evolutionary phase.
        Different physics for different phases.
        """
        
        def main_sequence(args):
            age, mass = args
            # Main sequence evolution
            luminosity = mass ** 3.5
            radius = mass ** 0.8
            return luminosity, radius
        
        def giant_branch(args):
            age, mass = args
            # Red giant evolution
            luminosity = mass ** 2.5 * (age / 10.0)
            radius = mass ** 0.5 * (age / 10.0) ** 0.3
            return luminosity, radius
        
        def white_dwarf(args):
            age, mass = args
            # White dwarf cooling
            luminosity = 0.001 * jnp.exp(-(age - 12.0) / 2.0)
            radius = 0.01
            return luminosity, radius
        
        # Multi-way conditional
        ms_turnoff = 10.0 / mass ** 2.5  # Simplified
        
        # Nested conditions
        return cond(
            age < ms_turnoff,
            main_sequence,
            lambda args: cond(
                age < ms_turnoff * 1.2,
                giant_branch,
                white_dwarf,
                args
            ),
            (age, mass)
        )
    
    # Test different evolutionary phases
    ages = jnp.array([0.1, 5.0, 11.0, 15.0])
    mass = 1.0
    
    for age in ages:
        L, R = stellar_evolution_step(age, mass)
        print(f"  Age {age:.1f} Gyr: L={L:.3f} L☉, R={R:.3f} R☉")
    
    # 2. Switch for multiple branches
    print("\n2. SWITCH STATEMENT:")
    
    @jit
    def process_observation(obs_type, data):
        """Process different observation types."""
        
        def process_photometry(data):
            # Magnitude calculation
            return -2.5 * jnp.log10(data) + 25.0
        
        def process_spectroscopy(data):
            # Continuum normalization
            continuum = jnp.median(data)
            return data / continuum
        
        def process_imaging(data):
            # Background subtraction
            background = jnp.percentile(data, 10)
            return data - background
        
        branches = [process_photometry, process_spectroscopy, process_imaging]
        
        return switch(obs_type, branches, data)
    
    # Different observation types
    data = jnp.array([100.0, 150.0, 200.0])
    
    for obs_type in range(3):
        result = process_observation(obs_type, data)
        print(f"  Type {obs_type}: {result[0]:.3f}")
    
    # 3. Gradient through conditionals
    print("\n3. GRADIENTS THROUGH CONDITIONALS:")
    
    @jit
    def piecewise_potential(r):
        """Piecewise gravitational potential."""
        
        def inner_region(r):
            # Constant density sphere
            return -1.5 + 0.5 * r**2
        
        def outer_region(r):
            # Point mass
            return -1.0 / r
        
        return cond(r < 1.0, inner_region, outer_region, r)
    
    # Gradient (force) is continuous at boundary!
    grad_potential = grad(piecewise_potential)
    
    radii = jnp.array([0.5, 0.99, 1.0, 1.01, 2.0])
    for r in radii:
        force = -grad_potential(r)
        print(f"  r={r:.2f}: F={force:.4f}")

jax_control_flow()
```

### Loops with scan and fori_loop

```python
def jax_loops():
    """Efficient loops in JAX using scan and fori_loop."""
    
    print("\nLOOPS IN JAX")
    print("=" * 50)
    
    # 1. scan for carrying state
    print("\n1. SCAN FOR SEQUENTIAL COMPUTATIONS:")
    
    @jit
    def runge_kutta_4(dynamics, initial_state, t_span, dt):
        """
        RK4 integration using scan.
        
        Parameters
        ----------
        dynamics : callable
            dy/dt = dynamics(y, t)
        initial_state : array
            Initial conditions
        t_span : tuple
            (t_start, t_end)
        dt : float
            Time step
        """
        
        def rk4_step(carry, t):
            y = carry
            
            k1 = dynamics(y, t)
            k2 = dynamics(y + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = dynamics(y + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = dynamics(y + dt * k3, t + dt)
            
            y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            return y_new, y_new  # carry and output
        
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        times = jnp.linspace(t_start, t_end, n_steps)
        
        _, trajectory = scan(rk4_step, initial_state, times)
        
        return times, trajectory
    
    # Kepler problem
    def kepler_dynamics(state, t):
        """Kepler two-body dynamics."""
        r, v = state[:2], state[2:]
        r_norm = jnp.linalg.norm(r)
        a = -r / r_norm**3
        return jnp.concatenate([v, a])
    
    initial = jnp.array([1.0, 0.0, 0.0, 1.0])
    times, trajectory = runge_kutta_4(kepler_dynamics, initial, (0, 10), 0.01)
    
    print(f"  Integrated {len(times)} steps")
    print(f"  Final position: {trajectory[-1, :2]}")
    
    # Check energy conservation
    def energy(state):
        r, v = state[:2], state[2:]
        return 0.5 * jnp.sum(v**2) - 1.0 / jnp.linalg.norm(r)
    
    initial_energy = energy(trajectory[0])
    final_energy = energy(trajectory[-1])
    print(f"  Energy drift: {abs(final_energy - initial_energy):.2e}")
    
    # 2. fori_loop for fixed iterations
    print("\n2. FORI_LOOP FOR FIXED ITERATIONS:")
    
    @jit
    def jacobi_iteration(A, b, x0, n_iterations):
        """Solve Ax = b using Jacobi iteration."""
        
        D = jnp.diag(jnp.diag(A))  # Diagonal part
        R = A - D  # Off-diagonal part
        
        def iteration(i, x):
            return jnp.linalg.solve(D, b - R @ x)
        
        return fori_loop(0, n_iterations, iteration, x0)
    
    # Test system
    A = jnp.array([[4.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 3.0]])
    b = jnp.array([15.0, 10.0, 10.0])
    x0 = jnp.zeros(3)
    
    solution = jacobi_iteration(A, b, x0, 50)
    print(f"  Solution: {solution}")
    print(f"  Residual: {jnp.linalg.norm(A @ solution - b):.2e}")
    
    # 3. while_loop for adaptive algorithms
    print("\n3. WHILE_LOOP FOR ADAPTIVE ALGORITHMS:")
    
    @jit
    def adaptive_integration(f, x0, x1, tol=1e-6, max_depth=10):
        """Adaptive Simpson's integration."""
        
        def simpson(a, b):
            """Simpson's rule on [a, b]."""
            mid = (a + b) / 2
            return (b - a) / 6 * (f(a) + 4*f(mid) + f(b))
        
        def should_refine(carry):
            a, b, depth, _ = carry
            return (depth < max_depth)
        
        def refine_step(carry):
            a, b, depth, integral = carry
            mid = (a + b) / 2
            
            whole = simpson(a, b)
            left = simpson(a, mid)
            right = simpson(mid, b)
            
            error = abs(whole - (left + right))
            
            # Simplified: just refine if error is large
            # In practice, would accumulate integral properly
            return (a, b, depth + 1, left + right)
        
        initial_carry = (x0, x1, 0, simpson(x0, x1))
        final_carry = while_loop(should_refine, refine_step, initial_carry)
        
        return final_carry[3]
    
    # Test function
    def test_func(x):
        return jnp.sin(x) ** 2
    
    result = adaptive_integration(test_func, 0.0, jnp.pi)
    print(f"  ∫sin²(x)dx from 0 to π = {result:.6f}")
    print(f"  Expected: {jnp.pi/2:.6f}")
    
    # 4. Nested loops
    print("\n4. NESTED LOOPS WITH SCAN:")
    
    @jit
    def double_pendulum_poincare(initial_conditions, n_periods, points_per_period):
        """
        Compute Poincaré section of double pendulum.
        Nested loop: outer for periods, inner for integration.
        """
        
        def dynamics(state, t):
            # Simplified double pendulum dynamics
            theta1, theta2, p1, p2 = state
            
            # Just for demonstration (not accurate physics)
            dtheta1 = p1
            dtheta2 = p2
            dp1 = -jnp.sin(theta1) - 0.1 * jnp.sin(theta1 - theta2)
            dp2 = -jnp.sin(theta2) + 0.1 * jnp.sin(theta1 - theta2)
            
            return jnp.array([dtheta1, dtheta2, dp1, dp2])
        
        def integrate_period(carry, _):
            state = carry
            
            # Inner loop: integrate for one period
            def step(s, _):
                new_s = s + 0.01 * dynamics(s, 0)
                return new_s, new_s
            
            final_state, trajectory = scan(
                step, state, None, length=points_per_period
            )
            
            # Return final state and Poincaré point
            return final_state, final_state[:2]  # Only positions
        
        # Outer loop: collect Poincaré points
        _, poincare_points = scan(
            integrate_period, initial_conditions, None, length=n_periods
        )
        
        return poincare_points
    
    initial = jnp.array([0.1, 0.1, 0.0, 0.0])
    poincare = double_pendulum_poincare(initial, 100, 100)
    
    print(f"  Generated {len(poincare)} Poincaré points")
    print(f"  Phase space bounds: θ₁∈[{poincare[:, 0].min():.2f}, {poincare[:, 0].max():.2f}]")

jax_loops()
```

## Custom Derivatives

### Defining Custom VJP Rules

```python
def custom_derivatives():
    """Define custom derivatives for specialized functions."""
    
    print("\nCUSTOM DERIVATIVES IN JAX")
    print("=" * 50)
    
    # 1. Custom VJP for numerical stability
    print("\n1. STABLE SOFTPLUS WITH CUSTOM VJP:")
    
    @custom_vjp
    def stable_softplus(x):
        """Softplus with numerical stability."""
        return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)
    
    def stable_softplus_fwd(x):
        """Forward pass: compute value and save residuals."""
        y = stable_softplus(x)
        return y, (x,)  # Save x for backward pass
    
    def stable_softplus_bwd(res, g):
        """Backward pass: compute VJP."""
        x, = res
        # Derivative of softplus is sigmoid
        sigmoid_x = 1 / (1 + jnp.exp(-x))
        return (g * sigmoid_x,)
    
    stable_softplus.defvjp(stable_softplus_fwd, stable_softplus_bwd)
    
    # Test gradient
    x_test = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    grad_fn = grad(lambda x: jnp.sum(stable_softplus(x)))
    grads = grad_fn(x_test)
    
    print(f"  x: {x_test}")
    print(f"  softplus(x): {stable_softplus(x_test)}")
    print(f"  gradients: {grads}")
    
    # 2. Custom derivative for interpolation
    print("\n2. DIFFERENTIABLE INTERPOLATION:")
    
    @custom_vjp
    def interp1d(x, xp, fp):
        """Linear interpolation with custom gradient."""
        return jnp.interp(x, xp, fp)
    
    def interp1d_fwd(x, xp, fp):
        y = jnp.interp(x, xp, fp)
        return y, (x, xp, fp)
    
    def interp1d_bwd(res, g):
        x, xp, fp = res
        
        # Find surrounding points
        idx = jnp.searchsorted(xp, x)
        idx = jnp.clip(idx, 1, len(xp) - 1)
        
        # Linear interpolation gradient
        x0, x1 = xp[idx - 1], xp[idx]
        f0, f1 = fp[idx - 1], fp[idx]
        
        # Gradient w.r.t. x
        dfdx = (f1 - f0) / (x1 - x0)
        
        # Gradients w.r.t. xp and fp (simplified)
        # In practice, these would be more complex
        dxp = jnp.zeros_like(xp)
        dfp = jnp.zeros_like(fp)
        
        # Weight for linear interpolation
        alpha = (x - x0) / (x1 - x0)
        dfp = dfp.at[idx - 1].add(g * (1 - alpha))
        dfp = dfp.at[idx].add(g * alpha)
        
        return g * dfdx, dxp, dfp
    
    interp1d.defvjp(interp1d_fwd, interp1d_bwd)
    
    # Test interpolation gradient
    xp = jnp.array([0.0, 1.0, 2.0, 3.0])
    fp = jnp.array([0.0, 1.0, 0.5, 2.0])
    
    def loss(x):
        return interp1d(x, xp, fp) ** 2
    
    x_test = 1.5
    value = interp1d(x_test, xp, fp)
    gradient = grad(loss)(x_test)
    
    print(f"  Interpolated value at x={x_test}: {value:.3f}")
    print(f"  Gradient: {gradient:.3f}")
    
    # 3. Custom JVP for forward-mode AD
    print("\n3. CUSTOM JVP FOR SPECIAL FUNCTIONS:")
    
    @custom_jvp
    def safe_log(x):
        """Logarithm with safe gradient at x=0."""
        return jnp.log(jnp.maximum(x, 1e-10))
    
    @safe_log.defjvp
    def safe_log_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        
        # Primal computation
        y = safe_log(x)
        
        # Tangent computation (forward-mode derivative)
        # Use safe derivative
        y_dot = x_dot / jnp.maximum(x, 1e-10)
        
        return y, y_dot
    
    # Test near zero
    x_vals = jnp.array([1e-15, 1e-10, 0.1, 1.0])
    grad_safe_log = grad(lambda x: jnp.sum(safe_log(x)))
    grads = grad_safe_log(x_vals)
    
    print(f"  x: {x_vals}")
    print(f"  safe_log(x): {safe_log(x_vals)}")
    print(f"  gradients: {grads}")
    
    # 4. Physics-informed custom derivatives
    print("\n4. PHYSICS-INFORMED DERIVATIVES:")
    
    @custom_vjp
    def gravitational_lensing_deflection(source_pos, lens_mass, lens_pos):
        """
        Gravitational lensing deflection angle.
        Custom derivative ensures correct physics.
        """
        # Vector from lens to source
        r = source_pos - lens_pos
        r_norm = jnp.linalg.norm(r)
        
        # Einstein radius effect (simplified)
        deflection = lens_mass * r / (r_norm**2 + 0.01)  # Softening
        
        return deflection
    
    def lensing_fwd(source_pos, lens_mass, lens_pos):
        deflection = gravitational_lensing_deflection(source_pos, lens_mass, lens_pos)
        return deflection, (source_pos, lens_mass, lens_pos)
    
    def lensing_bwd(res, g):
        source_pos, lens_mass, lens_pos = res
        r = source_pos - lens_pos
        r_norm = jnp.linalg.norm(r)
        
        # Gradients from physics
        # ∂α/∂source_pos
        d_source = g * lens_mass * (
            jnp.eye(2) / (r_norm**2 + 0.01) - 
            2 * jnp.outer(r, r) / (r_norm**2 + 0.01)**2
        )
        
        # ∂α/∂lens_mass
        d_mass = jnp.dot(g, r / (r_norm**2 + 0.01))
        
        # ∂α/∂lens_pos
        d_lens = -d_source  # Newton's third law!
        
        return d_source, d_mass, d_lens
    
    gravitational_lensing_deflection.defvjp(lensing_fwd, lensing_bwd)
    
    # Test lensing gradients
    source = jnp.array([1.0, 0.5])
    mass = 1.0
    lens = jnp.array([0.0, 0.0])
    
    def lensing_potential(s):
        deflection = gravitational_lensing_deflection(s, mass, lens)
        return jnp.sum(deflection**2)
    
    grad_lensing = grad(lensing_potential)
    gradient = grad_lensing(source)
    
    print(f"  Source position: {source}")
    print(f"  Deflection gradient: {gradient}")

custom_derivatives()
```

## Memory and Performance Optimization

### Checkpointing and Memory Management

```python
def memory_optimization():
    """Optimize memory usage in JAX computations."""
    
    print("\nMEMORY OPTIMIZATION IN JAX")
    print("=" * 50)
    
    # 1. Gradient checkpointing
    print("\n1. GRADIENT CHECKPOINTING:")
    
    from jax.experimental import checkpoint
    
    def deep_network(x, n_layers=50):
        """Deep network that would use lots of memory."""
        
        @checkpoint  # Don't store