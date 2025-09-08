# SUBMODULE 4: ADVANCED TECHNIQUES
*Week 5 - "When Basic Methods Fail"*

## Learning Objectives
By the end of this submodule, you will:
- Handle problems with multiple timescales using adaptive methods
- Apply Richardson extrapolation for error estimation
- Recognize and solve stiff equations with implicit methods
- Understand how errors propagate in long simulations

---

## Part 8: Multi-Scale Problems - When Physics Has Multiple Scales

### The Challenge

Real astrophysical systems rarely have single timescales:

```python
def timescale_hierarchy():
    """Show the scale separation problem"""
    # Example: Triple star system
    # Close binary: Period ~ days
    t_binary = 2 * day
    
    # Wide companion: Period ~ years  
    t_triple = 50 * year
    
    # Tidal evolution: ~ Myr
    t_tidal = 1e6 * year
    
    print(f"Timescale ratios:")
    print(f"Triple/Binary: {t_triple/t_binary:.1e}")
    print(f"Tidal/Binary: {t_tidal/t_binary:.1e}")
    
    # Naive approach: dt < 0.01 * t_binary
    # Steps needed: t_tidal / dt = 10^11 steps!
    # At 1μs per step = 3 years of CPU time!
```

### Solution 1: Adaptive Timesteps
```python
def adaptive_timestep(x, v, dt_max, tolerance=1e-6):
    """Adjust timestep based on local dynamics"""
    # Estimate local timescale
    a = compute_acceleration(x)
    v_mag = np.linalg.norm(v)
    a_mag = np.linalg.norm(a)
    
    # Various timescale estimates
    t_dyn = v_mag / a_mag if a_mag > 0 else dt_max
    t_orbit = 2*np.pi * np.linalg.norm(x) / v_mag if v_mag > 0 else dt_max
    
    # Choose timestep as fraction of shortest
    dt = 0.01 * min(t_dyn, t_orbit, dt_max)
    
    return dt
```

### Solution 2: Individual Timesteps
```python
def individual_timesteps(particles, global_time):
    """Each particle on its own schedule"""
    for p in particles:
        # Particle's personal timestep
        dt_i = compute_timestep(p.x, p.v)
        
        # Only update if due
        while p.time < global_time:
            p.x, p.v = leapfrog_step(p.x, p.v, dt_i, accel)
            p.time += dt_i
    
    # Synchronize occasionally for analysis
    return particles
```

### Real Example: Star Cluster with Binary
```python
def hierarchical_system():
    """
    Binary in cluster: extreme scale separation
    Use different integrators for different components!
    """
    # Tight binary: needs tiny timesteps
    def integrate_binary(state_binary, dt_small):
        # Use high-order symplectic
        return yoshida4_step(state_binary, dt_small)
    
    # Cluster: can use large timesteps
    def integrate_cluster(state_cluster, dt_large):
        # Binary is point mass to cluster
        com = binary_center_of_mass()
        return leapfrog_step(state_cluster, dt_large, 
                            lambda x: cluster_accel(x, com))
    
    # Couple: every dt_large, do many dt_small
    ratio = int(dt_large / dt_small)
    for _ in range(ratio):
        state_binary = integrate_binary(state_binary, dt_small)
    state_cluster = integrate_cluster(state_cluster, dt_large)
```

### 📊 Conceptual Checkpoint
- [ ] Can you identify the different timescales in a planet-moon system?
- [ ] Why can't we just use the smallest timestep everywhere?
- [ ] How does individual timestepping maintain synchronization?

---

## Part 9: Richardson Extrapolation - Extra Accuracy for Free

### The Core Idea

If you compute something two ways with different errors, you can combine them to cancel the leading error term!

### Mathematical Foundation
Given a method of order p:
- One step of size h: $y(h) = y_{exact} + Ch^{p+1} + O(h^{p+2})$
- Two steps of size h/2: $y(h/2) = y_{exact} + C(h/2)^{p+1} + O(h^{p+2})$

Combining cleverly:
$$y_{better} = \frac{2^{p+1} y(h/2) - y(h)}{2^{p+1} - 1}$$

This cancels the $h^{p+1}$ error term!

### Implementation
```python
def richardson_extrapolation(f, x, h, order=2):
    """
    Get higher accuracy by combining two calculations
    Example: 2nd order method → 3rd order result
    """
    # Coarse calculation
    y_h = f(x, h)
    
    # Fine calculation (two steps of h/2)
    y_mid = f(x, h/2)
    y_h2 = f(y_mid, h/2)
    
    # Richardson combination
    factor = 2**(order+1) - 1
    y_better = (2**(order+1) * y_h2 - y_h) / factor
    
    # Error estimate for free!
    error_est = abs(y_h2 - y_h) / factor
    
    return y_better, error_est
```

### Application: Adaptive RK45
```python
def rk45_adaptive_step(x, v, dt, accel, tol=1e-6):
    """
    Embedded Runge-Kutta with error control
    Uses Richardson to get error estimate
    """
    # RK4 step
    x4, v4 = rk4_step(x, v, dt, accel)
    
    # RK5 step (one extra evaluation)
    # ... (implementation details)
    x5, v5 = rk5_step(x, v, dt, accel)
    
    # Error estimate (difference is ~O(dt^5))
    error = np.linalg.norm([x5-x4, v5-v4])
    
    if error < tol:
        # Accept step, increase dt
        dt_new = dt * min(2, (tol/error)**0.2)
        return x5, v5, dt_new  # Use higher-order result
    else:
        # Reject step, decrease dt
        dt_new = dt * max(0.5, (tol/error)**0.2)
        return rk45_adaptive_step(x, v, dt_new, accel, tol)
```

### 🤝 Peer Instruction Question
"Richardson extrapolation gave a suspiciously large error estimate. Most likely cause?"

A) Calculation bug in the code  
B) Function not smooth enough for the assumed order  
C) Timestep in the nonlinear regime  
D) Round-off error dominating  

*Answer: C - Richardson assumes we're in the asymptotic regime where error ∝ h^p*

---

## Part 10: Stiff Equations - When Explicit Methods Need Impossible Steps

### What Makes a Problem "Stiff"?

A system is stiff when it has:
1. Multiple timescales differing by orders of magnitude
2. Fast timescale is **stable** (wants to decay)
3. You care about the slow evolution

### Classic Example: Stellar Nuclear Burning
```python
def nuclear_burning_rate(T):
    """
    Nuclear reaction rates are EXTREMELY temperature sensitive
    CNO cycle: ε ∝ T^15 to T^20
    pp chain: ε ∝ T^4
    """
    T_solar = 1.5e7  # Solar core temperature
    
    if T > 1.8e7:  # CNO dominates
        return eps_0 * (T/T_solar)**18  # !!!
    else:  # pp chain
        return eps_0 * (T/T_solar)**4
    
    # Small T change → HUGE rate change → Stiff!
```

### The Stiffness Problem Visualized
```python
def stiff_ode_example():
    """Classic stiff test problem"""
    # y' = λ(y - cos(t)) - sin(t)
    # Solution rapidly approaches cos(t), then follows slowly
    
    lambda_stiff = -1000  # Fast decay rate
    
    def explicit_euler(y0, dt, steps):
        """Explicit method - needs tiny dt"""
        y = y0
        t = 0
        for _ in range(steps):
            dydt = lambda_stiff*(y - np.cos(t)) - np.sin(t)
            y += dt * dydt  # Explicit update
            t += dt
            
            # Explodes if dt > 2/|λ| = 0.002
            if abs(y) > 1e10:
                return None  # Unstable!
        return y
    
    # Must use dt < 0.002 even though solution varies slowly!
    dt_explicit = 0.001  # Forced to be tiny
    dt_implicit = 0.1    # Can be 100x larger!
```

### Why Explicit Methods Fail
```python
def stability_requirement():
    """Show timestep constraints for stiff problems"""
    
    # Stellar evolution example
    t_nuclear = 1e-10 * year  # Fast nuclear timescale
    t_thermal = 1e7 * year    # Slow thermal timescale
    
    # Explicit method needs: dt < t_nuclear
    # But we want to simulate: t_total = t_thermal
    # Steps needed: 10^17 !!!
    
    print(f"Explicit timestep: {t_nuclear/year:.2e} years")
    print(f"Evolution time: {t_thermal/year:.2e} years")  
    print(f"Steps needed: {t_thermal/t_nuclear:.2e}")
    print(f"At 1μs/step: {t_thermal/t_nuclear*1e-6/86400/365:.1f} years of CPU!")
```

---

## Part 11: Implicit Methods - Solving for the Future

### The Implicit Euler Method

Instead of: $y_{n+1} = y_n + dt \cdot f(y_n)$ (explicit)

We use: $y_{n+1} = y_n + dt \cdot f(y_{n+1})$ (implicit)

The future appears on both sides - must solve an equation!

### Implementation with Newton-Raphson
```python
def implicit_euler(y, dt, f, df_dy, tol=1e-10):
    """
    Implicit Euler - unconditionally stable!
    Requires derivative for Newton iteration
    """
    # Solve: y_new = y + dt*f(y_new)
    # Rewrite: g(y_new) = y_new - y - dt*f(y_new) = 0
    
    y_new = y  # Initial guess
    
    for iteration in range(10):
        # Function value
        g = y_new - y - dt*f(y_new)
        
        # Jacobian
        dg_dy = 1 - dt*df_dy(y_new)
        
        # Newton update
        delta = -g/dg_dy
        y_new += delta
        
        if abs(delta) < tol:
            return y_new
    
    return y_new
```

### When to Use Implicit Methods

| Problem Type | Explicit OK? | Use Implicit? | Example |
|--------------|-------------|---------------|---------|
| Smooth orbits | ✓ | No | Planets |
| Wave propagation | ✓ | No | Sound waves |
| Chemical reactions | No | ✓ | Nuclear networks |
| Heat diffusion | No | ✓ | Stellar interiors |
| Stiff ODEs | No | ✓ | Climate models |

### Implicit Midpoint - 2nd Order and Symplectic!
```python
def implicit_midpoint(x, v, dt, force, tol=1e-10):
    """
    Implicit midpoint preserves energy exactly!
    But requires iteration to solve
    """
    # Solve: x_new = x + dt*v_mid
    #        v_new = v + dt*f(x_mid)
    # where x_mid = (x + x_new)/2, v_mid = (v + v_new)/2
    
    # Fixed-point iteration
    x_new, v_new = x + dt*v, v  # Initial guess
    
    for _ in range(10):
        x_mid = 0.5*(x + x_new)
        v_mid = 0.5*(v + v_new)
        a_mid = force(x_mid)
        
        x_next = x + dt*v_mid
        v_next = v + dt*a_mid
        
        if np.linalg.norm([x_next-x_new, v_next-v_new]) < tol:
            return x_next, v_next
            
        x_new, v_new = x_next, v_next
    
    return x_new, v_new
```

### ⚠️ Common Misconception Alert
> **"Implicit methods are always better"**
> 
> **FALSE! Trade-offs:**
> - Pro: Larger timesteps for stiff problems
> - Con: Must solve nonlinear equation each step
> - Con: Can over-damp physical oscillations
> - Con: More complex to implement

---

## Part 12: Error Propagation in Multi-Step Methods

### How Errors Accumulate

Different methods have fundamentally different error growth:

```python
def error_propagation_analysis():
    """Demonstrate error growth patterns"""
    
    steps = np.arange(0, 10000, 100)
    
    # Local error per step
    local_error = 1e-10
    
    # Different propagation modes
    linear_growth = steps * local_error  # Euler
    sqrt_growth = np.sqrt(steps) * local_error  # Statistical
    bounded = 2 * local_error * np.ones_like(steps)  # Symplectic
    exponential = local_error * 1.00001**steps  # Unstable
    
    print("After 10,000 steps:")
    print(f"Linear (Euler): {linear_growth[-1]:.2e}")
    print(f"Statistical (RK4): {sqrt_growth[-1]:.2e}")
    print(f"Bounded (Symplectic): {bounded[-1]:.2e}")
    print(f"Exponential (Unstable): {exponential[-1]:.2e}")
```

### Error Growth in Different Schemes

| Method | Local Error | Global Error | After 10^6 Steps |
|--------|-------------|--------------|------------------|
| Euler | O(dt²) | O(dt) | 10^-4 for dt=10^-10 |
| RK4 | O(dt⁵) | O(dt⁴) | 10^-16 for dt=10^-4 |
| Symplectic | O(dt³) | Bounded! | ~10^-13 forever |
| Unstable | Any | Exponential | Infinity |

### Long-Term Integration Strategy
```python
def long_term_integration_strategy(t_final, tolerance):
    """
    Choose method based on simulation length
    """
    if t_final < 100 * t_orbit:
        # Short-term: accuracy matters most
        method = "RK45 adaptive"
        
    elif t_final < 10000 * t_orbit:
        # Medium-term: stability + accuracy
        method = "4th-order symplectic"
        
    else:
        # Long-term: structure preservation critical
        method = "2nd-order symplectic with small dt"
    
    return method
```

:::{admonition} 📦 Extra: Boundary Conditions in Numerical Methods
:class: note, dropdown

**When Boundaries Matter (Not in Your Current Projects!)**

While N-body dynamics and MCRT don't have boundary conditions, many astrophysical problems do. Here's what you might encounter in future research:

**Types of Boundaries in Astrophysics:**

1. **Physical Boundaries**
```python
# Stellar photosphere: where optical depth τ = 2/3
def stellar_surface_boundary(r, T, rho):
    """Radiation escapes freely beyond this point"""
    if optical_depth(r) < 2/3:
        # Switch from diffusion to free-streaming
        return outgoing_radiation_only()
```

2. **Reflecting/Periodic Boundaries**
```python
# Cosmological simulations: universe has no edge!
def periodic_boundary(x, box_size):
    """Particle leaving one side enters other"""
    return x % box_size  # Wrap around

# Magnetic confinement
def reflecting_boundary(x, v, wall_position):
    """Particle bounces off wall"""
    if x > wall_position:
        x = 2*wall_position - x
        v = -v  # Reverse velocity
    return x, v
```

3. **Absorbing Boundaries**
```python
# Black hole event horizon
def event_horizon_boundary(r, r_schwarzschild):
    """Nothing returns from here"""
    if r < r_schwarzschild:
        remove_particle()  # Gone forever
```

4. **Open Boundaries (Hardest!)**
```python
# Stellar wind leaving system
def open_boundary(x, v, boundary):
    """Waves/particles can leave but not reflect"""
    # Requires special methods:
    # - Perfectly Matched Layers (PML)
    # - Absorbing boundary conditions
    # - Non-reflecting extrapolation
```

**Numerical Challenges with Boundaries:**

1. **Order Reduction**
   - Interior: 4th order accuracy
   - Near boundary: Often drops to 2nd order
   - Need special one-sided stencils

2. **Stability Issues**
```python
# Boundary can trigger instabilities
def unstable_boundary():
    # Simple reflection can cause energy growth
    # Need careful implementation
    pass
```

3. **Conservation Violations**
   - Momentum/energy can be lost at boundaries
   - Need careful accounting

**Ghost Zones/Cells Method:**
```python
def ghost_zone_method(grid, n_ghost=2):
    """
    Extend domain with fictitious points
    Makes boundaries look like interior
    """
    extended_grid = np.zeros(len(grid) + 2*n_ghost)
    extended_grid[n_ghost:-n_ghost] = grid
    
    # Fill ghost zones based on BC type
    if boundary_type == "periodic":
        extended_grid[:n_ghost] = grid[-n_ghost:]
        extended_grid[-n_ghost:] = grid[:n_ghost]
    elif boundary_type == "reflecting":
        extended_grid[:n_ghost] = grid[n_ghost-1::-1]
        extended_grid[-n_ghost:] = grid[-1:-n_ghost-1:-1]
    
    return extended_grid
```

**When You'll Need This:**
- Hydrodynamics simulations (shocks at boundaries)
- Stellar structure (center and surface conditions)
- Accretion disks (inner/outer boundaries)
- Magnetospheres (complex field boundaries)
- Wave propagation (non-reflecting conditions)

**The Good News:**
Your N-body simulations are boundary-free! Gravity is long-range, space is infinite (or periodic), and particles move freely. Enjoy this simplicity while it lasts! 

**Advanced Reading:**
- "Numerical Recipes" Ch. 19 (Boundary Value Problems)
- Poisson equation solvers (need boundaries)
- Perfectly Matched Layers for wave equations
:::

### Monitoring Integration Health
```python
def integration_diagnostics(trajectory):
    """
    Check if integration is reliable
    """
    # Energy conservation
    E = [energy(x, v) for x, v in trajectory]
    E_drift = (E[-1] - E[0])/E[0]
    
    # Angular momentum
    L = [angular_momentum(x, v) for x, v in trajectory]
    L_drift = (L[-1] - L[0])/L[0]
    
    # Phase space volume (symplectic check)
    volume_0 = phase_space_volume(trajectory[0])
    volume_f = phase_space_volume(trajectory[-1])
    volume_drift = (volume_f - volume_0)/volume_0
    
    print(f"Energy drift: {E_drift:.2e}")
    print(f"Angular momentum drift: {L_drift:.2e}")
    print(f"Phase volume drift: {volume_drift:.2e}")
    
    if abs(E_drift) > 0.01:
        print("WARNING: Significant energy drift!")
    if abs(L_drift) > 0.01:
        print("WARNING: Angular momentum not conserved!")
```

### 🤔 Metacognitive Reflection
*Consider these questions about error propagation:*

1. **Why do symplectic methods have bounded error?**
   - They conserve a modified Hamiltonian exactly
   - Errors oscillate rather than accumulate

2. **When does statistical error averaging help?**
   - When errors are uncorrelated
   - Monte Carlo and stochastic processes

3. **How do you know when to stop trusting results?**
   - Monitor conserved quantities
   - Compare different timesteps
   - Check against known limits

---

## Synthesis: Choosing the Right Advanced Method

```python
def method_selection_flowchart(problem):
    """
    Decision tree for advanced techniques
    """
    # Analyze problem characteristics
    timescales = analyze_timescales(problem)
    stiffness = max(timescales)/min(timescales)
    
    if stiffness > 1e6:
        if fast_scale_stable(problem):
            return "Implicit methods"
        else:
            return "Adaptive + individual timesteps"
    
    elif stiffness > 1e3:
        if need_high_accuracy(problem):
            return "RK45 adaptive with Richardson"
        else:
            return "Adaptive timesteps"
    
    else:
        if long_term_integration(problem):
            return "High-order symplectic"
        else:
            return "Standard RK4"
```

---

## Connections to Course Projects

### Immediate Applications
- **Project 2**: Use adaptive timesteps for close encounters
- **Project 3**: Richardson for Monte Carlo error estimates

### Future Connections
- **Project 4**: Implicit methods for stiff posterior distributions
- **Project 5**: Multi-scale for GP hyperparameter optimization
- **Final**: Adaptive learning rates (same as adaptive timesteps!)

---

## Practice Problems

1. **Scale Separation**: Compute timescale ratios for Earth-Moon-Sun system
2. **Richardson Test**: Verify order improvement on test problem
3. **Stiffness Detection**: Identify stiff equations in stellar evolution
4. **Error Growth**: Track error propagation for 10^6 step integration

---

## Final Assessment Questions

### Conceptual Understanding
1. "Explain why implicit methods can use larger timesteps for stiff problems"
2. "Draw the error growth patterns for different integration schemes"
3. "When would you choose adaptive timesteps over uniform?"

### Problem Solving
1. "Design an integration strategy for a globular cluster with binaries"
2. "Implement Richardson extrapolation for your favorite method"
3. "Detect and handle stiffness in a chemical reaction network"

### Critical Thinking
1. "What's the computational trade-off between implicit methods and small explicit steps?"
2. "How would you verify that Richardson extrapolation is working correctly?"
3. "Design a hybrid explicit-implicit scheme for a multi-physics problem"

---

## Module Summary

You've mastered advanced techniques for when basic methods fail:

**Multi-scale Problems**: Adaptive and individual timesteps
**Richardson Extrapolation**: Higher accuracy and error estimates
**Stiff Equations**: Implicit methods for impossible timesteps
**Error Propagation**: Understanding long-term reliability

Key insight: Real problems require combining multiple techniques. There's no universal best method - only the right method for each problem.

## Looking Forward

These numerical foundations prepare you for:
- **Project 2**: Implementing stable N-body integration
- **Project 3**: Monte Carlo methods (different error propagation!)
- **Project 4-5**: Advanced sampling and optimization
- **Final Project**: Neural networks (gradient descent = Euler for optimization!)

Remember: Every algorithm is an approximation. The art is choosing approximations that preserve what matters for your problem!