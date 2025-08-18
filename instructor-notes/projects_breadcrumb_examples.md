# Chapter 3: Bite-Sized Breadcrumb Examples
## Seeds Throughout the Chapter - Rewards for Careful Readers

---

## BOOLEAN LOGIC EXAMPLES (Section 3.2)

### Generic Pattern (Main Teaching Example)
```python
def check_conditions(value, thresholds):
    """Multi-condition checking pattern."""
    condition_A = value > thresholds['min']
    condition_B = value < thresholds['max']
    condition_C = value != thresholds['forbidden']
    
    all_good = condition_A and condition_B and condition_C
    any_problem = not condition_A or not condition_B
    
    return all_good, any_problem
```

### Breadcrumb 1: Energy Check
```python
def needs_energy_warning(system):
    """Check if energy conservation is violated.
    Essential for Projects 2 & 3: N-body validation."""
    initial_E = system['E_initial']
    current_E = system['E_current']
    
    drift = abs((current_E - initial_E) / initial_E)
    return drift > 1e-6  # Why this tolerance?
```

### Breadcrumb 2: Timestep Safety
```python
def timestep_ok(dt, velocity, cell_size):
    """CFL-like condition for numerical stability.
    Project 2: Prevents your orbits from exploding."""
    # Courant condition (students: research this!)
    max_dt = 0.5 * cell_size / velocity
    return dt < max_dt
```

### Breadcrumb 3: Proximity Check
```python
def too_close(r_squared, softening_squared):
    """Detect when special handling needed.
    Project 2: Prevents force singularities in N-body."""
    return r_squared < softening_squared * 4
    # Factor of 4? Students must understand why
```

### Breadcrumb 4: Convergence Detection
```python
def possibly_converged(old, new, iteration):
    """Multiple convergence criteria.
    Projects 3 & 5: Regression and MCMC convergence."""
    relative_change = abs(new - old) / abs(old + 1e-50)
    absolute_change = abs(new - old)
    
    return (relative_change < 1e-9 or 
            absolute_change < 1e-12 or
            iteration > 1000)
```

### Breadcrumb 5: Boundary Detection
```python
def particle_leaving(position, velocity, box_size):
    """Detect escaping particles.
    Project 4: Photons escaping dust cloud."""
    at_boundary = abs(position) > 0.95 * box_size
    moving_outward = position * velocity > 0
    
    return at_boundary and moving_outward
```

---

## CONDITIONAL STATEMENTS (Section 3.3)

### Generic Pattern
```python
def classify_value(x):
    if x < 0:
        raise ValueError("Must be positive")
    elif x < 1:
        return 'tiny'
    elif x < 100:
        return 'normal'
    else:
        return 'huge'
```

### Breadcrumb 1: Integration Method Selection
```python
def choose_integrator(accuracy_needed, system_size):
    """Select numerical integration scheme.
    Project 2: Euler vs Verlet vs RK4 decision."""
    if system_size < 10:
        return 'direct'  # What method exactly?
    elif accuracy_needed > 1e-12:
        return 'high_order'  # Which one?
    else:
        return 'standard'  # Students choose
```

### Breadcrumb 2: Stellar Evolution Phase
```python
def stellar_phase(mass, age, metallicity):
    """Determine stellar evolution stage.
    Project 1: Star class lifecycle methods."""
    if age < 1e6:
        return 'pre_main_sequence'
    elif age < compute_ms_lifetime(mass):
        return 'main_sequence'
    elif mass > 8:
        return 'supergiant'
    else:
        return 'red_giant'
```

### Breadcrumb 3: Force Calculation Mode
```python
def force_method(n_neighbors):
    """Choose force calculation strategy.
    Project 2: Optimize N-body performance."""
    if n_neighbors == 0:
        return None
    elif n_neighbors == 1:
        return 'two_body'
    elif n_neighbors < 10:
        return 'direct_sum'
    else:
        return 'tree_node'  # What's a tree node?
```

### Breadcrumb 4: Photon Fate
```python
def photon_interaction(optical_depth, random_num):
    """Determine photon's fate in medium.
    Project 4: MC radiative transfer decisions."""
    if random_num < math.exp(-optical_depth):
        return 'escaped'
    elif random_num < 0.7:  # Albedo
        return 'scattered'
    else:
        return 'absorbed'
```

---

## FOR LOOPS (Section 3.4)

### Generic Pattern
```python
def accumulate_property(items):
    total = 0
    for item in items:
        if valid(item):
            total += extract_value(item)
    return total
```

### Breadcrumb 1: Force Accumulation
```python
# One step in N-body force calculation
# Project 2: Core of gravitational dynamics
for neighbor in nearby_particles:
    if neighbor is not particle:
        # Vector from particle to neighbor
        dx = neighbor.x - particle.x
        dy = neighbor.y - particle.y
        
        # Accumulate force components
        force_x += calculate_fx(dx, dy)  # Students implement
        force_y += calculate_fy(dx, dy)
```

### Breadcrumb 2: Luminosity Function
```python
def stellar_luminosity_function(stellar_pop):
    """Build luminosity distribution.
    Project 1: StellarPopulation analysis."""
    luminosity_bins = np.zeros(n_bins)
    
    for star in stellar_pop:
        L = star.luminosity
        bin_idx = int(np.log10(L / L_min) / bin_width)
        
        if 0 <= bin_idx < n_bins:
            luminosity_bins[bin_idx] += 1
    
    return luminosity_bins
```

### Breadcrumb 3: Monte Carlo Sampling
```python
# Monte Carlo integration step
# Projects 2 & 4: Statistical sampling
total = 0
for i in range(n_samples):
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    
    if inside_region(x, y):
        total += function_value(x, y)

integral = (x_max - x_min) * (y_max - y_min) * total / n_samples
```

### Breadcrumb 4: Time Evolution
```python
# Basic Euler step (why not use this?)
# Project 2: The integration method that fails
for particle in particles:
    # Update position
    particle.x += particle.vx * dt
    particle.y += particle.vy * dt
    
    # Update velocity  
    particle.vx += particle.ax * dt
    particle.vy += particle.ay * dt
```

---

## WHILE LOOPS (Section 3.4)

### Generic Pattern
```python
while not_converged and iterations < max_iter:
    old_value = current_value
    current_value = update(current_value)
    not_converged = check_convergence(old_value, current_value)
    iterations += 1
```

### Breadcrumb 1: Adaptive Timestep
```python
def adaptive_integration(state, t_end):
    """Adaptive timestep control.
    Project 2: Keep your solar system stable."""
    dt = initial_dt
    while state.time < t_end:
        # Try a step
        error = estimate_error(state, dt)
        
        if error < tolerance:
            # Accept step
            state = advance(state, dt)
            dt = min(dt * 1.2, dt_max)  # Grow carefully
        else:
            # Reject and shrink
            dt = dt * 0.5
```

### Breadcrumb 2: Gradient Descent Step
```python
def optimize_parameters(initial_params):
    """Gradient descent optimization.
    Projects 3 & Final: Core of ML training."""
    params = initial_params
    
    while not converged:
        gradient = compute_gradient(params)  # Students derive
        params = params - learning_rate * gradient
        
        if np.linalg.norm(gradient) < 1e-6:
            converged = True
```

### Breadcrumb 3: MCMC Sampling
```python
def metropolis_step(current_state, proposal_width):
    """One Metropolis-Hastings step.
    Project 5: Heart of Bayesian inference."""
    # Propose new state
    proposed = current_state + np.random.normal(0, proposal_width)
    
    # Accept/reject (students: why this ratio?)
    ratio = likelihood(proposed) / likelihood(current_state)
    
    if ratio > np.random.uniform():
        return proposed  # Accept
    else:
        return current_state  # Reject
```

---

## NESTED LOOPS (Section 3.4)

### Generic Pattern
```python
for i in range(n):
    for j in range(i + 1, n):
        if should_process(i, j):
            process_pair(i, j)
```

### Breadcrumb 1: Gravitational Pairs
```python
def calculate_all_forces(particles):
    """All pairwise gravitational forces.
    Project 2: O(N²) complexity issue."""
    forces = np.zeros((n, 2))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Force from j on i
            f_ij = compute_force(particles[i], particles[j])
            forces[i] += f_ij
            forces[j] -= f_ij  # Newton's 3rd law
    
    return forces
```

### Breadcrumb 2: Correlation Matrix
```python
def compute_correlations(features):
    """Feature correlation matrix.
    Project 3: Understanding your data."""
    n_features = features.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):  # Include diagonal
            corr = np.corrcoef(features[:, i], features[:, j])[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Symmetric
    
    return corr_matrix
```

### Breadcrumb 3: Photon Scattering Grid
```python
def trace_photons_through_grid(grid, n_photons):
    """Photon transport through dusty medium.
    Project 4: 2D radiative transfer."""
    for photon in range(n_photons):
        x, y = initial_position()
        
        while in_grid(x, y):
            # Find current cell
            i = int(x / cell_size)
            j = int(y / cell_size)
            
            # Interact with medium
            if random.random() < opacity[i, j]:
                scatter_photon()  # Change direction
```

---

## LIST COMPREHENSIONS (Section 3.5)

### Generic Pattern
```python
filtered = [transform(x) for x in data if condition(x)]
```

### Breadcrumb 1: Main Sequence Selection
```python
# Select main sequence stars
# Project 1: Filter stellar population
main_sequence = [star for star in population 
                 if star.phase == 'MS' and star.mass < 2.0]
```

### Breadcrumb 2: Active Particles
```python
# Which particles need updating this step?
# Project 2: Adaptive timestepping
active = [p for p in particles 
          if p.last_update + p.timestep <= current_time]
```

### Breadcrumb 3: Training Data Selection
```python
# Select training samples
# Project 3: Data preprocessing
clean_data = [(x, y) for x, y in zip(features, targets)
              if not np.isnan(x).any() and abs(y) < 3 * sigma]
```

### Breadcrumb 4: Escaped Photons
```python
# Collect escaped photons
# Project 4: MCRT output
escaped = [p for p in photons 
           if p.status == 'escaped' and p.wavelength > 400]
```

---

## DEBUGGING PATTERNS (Section 3.7)

### Generic Pattern
```python
if debug_mode:
    print(f"Step {i}: value = {value:.6e}")
    assert value > 0, "Value should be positive"
```

### Breadcrumb 1: Conservation Check
```python
def check_conservation(system, initial_values):
    """Validate physics conservation.
    Projects 2 & 4: Catch integration errors."""
    current_E = calculate_energy(system)
    current_P = calculate_momentum(system)
    
    E_drift = abs(current_E - initial_values['E']) / abs(initial_values['E'])
    P_drift = np.linalg.norm(current_P - initial_values['P']) / np.linalg.norm(initial_values['P'])
    
    if E_drift > 1e-6:
        print(f"WARNING: Energy drift = {E_drift:.2e}")
    
    return E_drift < 1e-4  # Acceptable?
```

### Breadcrumb 2: NaN Detection
```python
def validate_state(state, step):
    """Catch numerical instabilities early.
    All projects: Prevent cascade failures."""
    checks = {
        'positions_finite': np.all(np.isfinite(state.positions)),
        'velocities_bounded': np.all(np.abs(state.velocities) < c),
        'masses_positive': np.all(state.masses > 0),
        'no_infinities': not np.any(np.isinf(state.values))
    }
    
    for check, passed in checks.items():
        if not passed:
            print(f"Step {step}: Failed {check}")
            return False
    
    return True
```

### Breadcrumb 3: Convergence Monitor
```python
def track_convergence(loss_history, window=10):
    """Monitor training convergence.
    Projects 3, 5, Final: ML optimization."""
    if len(loss_history) < window:
        return False
    
    recent = loss_history[-window:]
    improvement = (recent[0] - recent[-1]) / recent[0]
    
    if improvement < 1e-4:
        print("Training plateau detected")
        return True  # Might be converged
    
    return False
```

---

## WHY THIS APPROACH WORKS FOR YOUR COURSE

### Project-Specific Payoffs:

1. **Project 1 (Stellar Physics)**: Phase classification, luminosity functions
2. **Project 2 (N-body)**: Force calculations, energy checks, timestep control
3. **Project 3 (Regression)**: Gradient descent, data cleaning, convergence
4. **Project 4 (MCRT)**: Photon fate, boundary detection, scattering
5. **Project 5 (MCMC)**: Metropolis steps, convergence monitoring
6. **Project 6 (GP)**: Correlation matrices, kernel evaluations
7. **Final (Neural Nets)**: Forward propagation structure, gradient updates

### The "Aha!" Timeline:

- **Week 3**: "Oh, that's why my orbits exploded - the timestep check!"
- **Week 5**: "The convergence pattern from Chapter 3 works for gradient descent!"
- **Week 7**: "Photon boundary detection is just like particle escape!"
- **Week 10**: "MCMC is just a while loop with smart accept/reject!"
- **Week 15**: "Neural network training is nested loops all the way down!"

Students who pay attention will have a toolkit of patterns. Those who skim will struggle unnecessarily - their loss indeed!