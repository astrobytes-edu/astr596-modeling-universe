# ⚠️ JAX Fundamentals: NumPy on Steroids

## Learning Objectives
By the end of this chapter, you will:
- Understand JAX's core philosophy and why it matters for scientific computing
- Master functional programming patterns required by JAX
- Use automatic differentiation for physics problems
- Compile functions with JIT for massive speedups
- Vectorize computations with vmap
- Generate reproducible random numbers with JAX's PRNG

## Why JAX? The Revolution in Scientific Computing

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax import random
import numpy as np
import time
import matplotlib.pyplot as plt

def why_jax_matters():
    """Demonstrate JAX's game-changing features for astronomy."""
    
    print("JAX: Three Transformations That Change Everything")
    print("=" * 50)
    
    # 1. NumPy API but faster
    print("\n1. FAMILIAR BUT FASTER:")
    
    # NumPy computation
    np_array = np.random.randn(1000, 1000)
    start = time.perf_counter()
    np_result = np.sin(np_array) ** 2 + np.cos(np_array) ** 2
    numpy_time = time.perf_counter() - start
    
    # JAX computation
    jax_array = jnp.array(np_array)
    start = time.perf_counter()
    jax_result = jnp.sin(jax_array) ** 2 + jnp.cos(jax_array) ** 2
    jax_time = time.perf_counter() - start
    
    print(f"  NumPy time: {numpy_time*1000:.2f} ms")
    print(f"  JAX time: {jax_time*1000:.2f} ms")
    print(f"  Results match: {np.allclose(np_result, jax_result)}")
    
    # 2. Automatic differentiation
    print("\n2. AUTOMATIC DIFFERENTIATION:")
    
    def gravitational_potential(r, mass=1.0):
        """Gravitational potential energy."""
        return -mass / jnp.linalg.norm(r)
    
    # Get gradient automatically!
    grad_potential = grad(gravitational_potential)
    
    position = jnp.array([1.0, 0.0, 0.0])
    force = -grad_potential(position)
    print(f"  Position: {position}")
    print(f"  Force: {force}")
    print(f"  |F| = {jnp.linalg.norm(force):.4f} (expected: 1.0)")
    
    # 3. Compilation with JIT
    print("\n3. JUST-IN-TIME COMPILATION:")
    
    def orbital_step(state, dt):
        """Single step of orbital integration."""
        r, v = state[:3], state[3:]
        a = -r / jnp.linalg.norm(r)**3
        v_new = v + a * dt
        r_new = r + v_new * dt
        return jnp.concatenate([r_new, v_new])
    
    orbital_step_jit = jit(orbital_step)
    
    state = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # First call includes compilation
    start = time.perf_counter()
    _ = orbital_step_jit(state, 0.01)
    first_time = time.perf_counter() - start
    
    # Subsequent calls are fast
    start = time.perf_counter()
    for _ in range(1000):
        state = orbital_step_jit(state, 0.01)
    compiled_time = time.perf_counter() - start
    
    # Compare with non-compiled
    state = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    start = time.perf_counter()
    for _ in range(1000):
        state = orbital_step(state, 0.01)
    python_time = time.perf_counter() - start
    
    print(f"  First call (with compilation): {first_time*1000:.2f} ms")
    print(f"  1000 steps compiled: {compiled_time*1000:.2f} ms")
    print(f"  1000 steps python: {python_time*1000:.2f} ms")
    print(f"  Speedup: {python_time/compiled_time:.1f}x")
    
    # 4. Vectorization with vmap
    print("\n4. AUTOMATIC VECTORIZATION:")
    
    def distance(r1, r2):
        """Distance between two points."""
        return jnp.linalg.norm(r1 - r2)
    
    # Vectorize over first argument
    distances_from_origin = vmap(distance, in_axes=(0, None))
    
    positions = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    origin = jnp.array([0, 0, 0])
    
    dists = distances_from_origin(positions, origin)
    print(f"  Distances from origin: {dists}")
    
    return True

why_jax_matters()
```

## The Functional Programming Paradigm

### Pure Functions: The Heart of JAX

```python
def functional_programming_in_jax():
    """JAX requires pure functional programming - here's why and how."""
    
    print("PURE FUNCTIONS IN JAX")
    print("=" * 50)
    
    # ❌ BAD: Impure function with side effects
    global_counter = 0
    
    def impure_function(x):
        global global_counter
        global_counter += 1  # Side effect!
        return x * global_counter
    
    # This won't work properly with JAX transformations
    # result = jit(impure_function)(5)  # Would give unexpected results
    
    # ✅ GOOD: Pure function
    def pure_function(x, counter):
        """Pure function - output depends only on inputs."""
        return x * counter, counter + 1
    
    # This works perfectly with JAX
    pure_jit = jit(pure_function)
    result, new_counter = pure_jit(5.0, 1.0)
    print(f"Pure function result: {result}, new counter: {new_counter}")
    
    # Example: Stellar evolution step
    print("\n STELLAR EVOLUTION EXAMPLE:")
    
    # ❌ BAD: Using mutation
    class Star:
        def __init__(self, mass, luminosity):
            self.mass = mass
            self.luminosity = luminosity
        
        def evolve(self, dt):
            self.luminosity *= 1.01  # Mutating state!
            self.mass *= 0.999
    
    # ✅ GOOD: Functional approach
    def evolve_star(state, dt):
        """
        Evolve star state functionally.
        
        Parameters
        ----------
        state : dict
            Star properties {mass, luminosity, age}
        dt : float
            Time step
        
        Returns
        -------
        dict
            New star state
        """
        mass_loss_rate = 1e-7 * state['luminosity']
        
        new_state = {
            'mass': state['mass'] - mass_loss_rate * dt,
            'luminosity': state['luminosity'] * (1 + 0.01 * dt),
            'age': state['age'] + dt
        }
        
        return new_state
    
    # JAX-friendly star evolution
    @jit
    def evolve_star_jax(mass, luminosity, age, dt):
        """Evolve star with JAX."""
        mass_loss_rate = 1e-7 * luminosity
        
        new_mass = mass - mass_loss_rate * dt
        new_luminosity = luminosity * (1 + 0.01 * dt)
        new_age = age + dt
        
        return new_mass, new_luminosity, new_age
    
    # Run evolution
    mass, lum, age = 1.0, 1.0, 0.0
    for _ in range(100):
        mass, lum, age = evolve_star_jax(mass, lum, age, 0.01)
    
    print(f"After evolution: M={mass:.4f}, L={lum:.4f}, Age={age:.2f}")
    
    # Carrying state through computations
    print("\nCARRYING STATE FUNCTIONALLY:")
    
    from functools import partial
    
    @jit
    def integrate_orbit(carry, dt):
        """Single integration step."""
        position, velocity = carry
        acceleration = -position / jnp.linalg.norm(position)**3
        
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt
        
        return (new_position, new_velocity)
    
    # Use scan for sequential computations
    from jax.lax import scan
    
    def simulate_orbit(initial_state, dt, n_steps):
        """Simulate orbit for n_steps."""
        
        def step(carry, _):
            new_carry = integrate_orbit(carry, dt)
            return new_carry, new_carry  # Return carry and output
        
        final_state, trajectory = scan(step, initial_state, None, length=n_steps)
        return trajectory
    
    initial = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    trajectory = simulate_orbit(initial, 0.01, 1000)
    
    positions = trajectory[0]
    print(f"Simulated {len(positions)} orbital positions")
    print(f"Final position: {positions[-1]}")

functional_programming_in_jax()
```

## Automatic Differentiation: The Killer Feature

### Gradients for Physics

```python
def automatic_differentiation_astronomy():
    """Automatic differentiation for astronomical applications."""
    
    print("AUTOMATIC DIFFERENTIATION IN ASTRONOMY")
    print("=" * 50)
    
    # 1. Simple derivatives
    print("\n1. BASIC DERIVATIVES:")
    
    def planck_law(wavelength, temperature):
        """Planck's law for blackbody radiation."""
        h = 6.626e-34
        c = 3e8
        k = 1.38e-23
        
        wavelength = wavelength * 1e-9  # nm to m
        
        numerator = 2 * h * c**2 / wavelength**5
        denominator = jnp.exp(h * c / (wavelength * k * temperature)) - 1
        
        return numerator / denominator
    
    # Derivative with respect to temperature
    dplanck_dT = grad(planck_law, argnums=1)
    
    wavelength = 500.0  # nm
    temperature = 5778.0  # K
    
    intensity = planck_law(wavelength, temperature)
    gradient = dplanck_dT(wavelength, temperature)
    
    print(f"  B(λ={wavelength}nm, T={temperature}K) = {intensity:.3e}")
    print(f"  ∂B/∂T = {gradient:.3e}")
    
    # 2. Gradient of gravitational N-body potential
    print("\n2. N-BODY FORCES FROM POTENTIAL:")
    
    def nbody_potential(positions, masses):
        """
        Total gravitational potential energy.
        
        Parameters
        ----------
        positions : array shape (n, 3)
            Positions of n bodies
        masses : array shape (n,)
            Masses of bodies
        """
        n = len(masses)
        potential = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                r_ij = jnp.linalg.norm(positions[i] - positions[j])
                potential -= masses[i] * masses[j] / r_ij
        
        return potential
    
    # Get forces from potential gradient
    def get_forces(positions, masses):
        """Calculate forces as negative gradient of potential."""
        return -grad(nbody_potential)(positions, masses)
    
    # Three-body system
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [-0.5, 0.866, 0.0],
        [-0.5, -0.866, 0.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0])
    
    forces = get_forces(positions, masses)
    print(f"  Forces on 3 bodies:")
    for i, f in enumerate(forces):
        print(f"    Body {i}: {f}")
    print(f"  Total force: {jnp.sum(forces, axis=0)} (should be ~0)")
    
    # 3. Hessian for optimization
    print("\n3. HESSIAN FOR FINDING MINIMA:")
    
    def chi_squared(params, x_data, y_data):
        """Chi-squared for linear fit."""
        a, b = params
        y_model = a * x_data + b
        return jnp.sum((y_data - y_model)**2)
    
    # Get gradient and Hessian
    from jax import hessian
    
    grad_chi2 = grad(chi_squared)
    hess_chi2 = hessian(chi_squared)
    
    # Sample data
    x_data = jnp.linspace(0, 10, 20)
    y_true = 2.5 * x_data + 1.0
    y_data = y_true + 0.5 * random.normal(random.PRNGKey(0), shape=x_data.shape)
    
    params = jnp.array([2.0, 0.5])  # Initial guess
    
    gradient = grad_chi2(params, x_data, y_data)
    hessian_matrix = hess_chi2(params, x_data, y_data)
    
    print(f"  Gradient at {params}: {gradient}")
    print(f"  Hessian:\n{hessian_matrix}")
    
    # Use for Newton's method
    params_new = params - jnp.linalg.inv(hessian_matrix) @ gradient
    print(f"  Newton step: {params} -> {params_new}")
    
    # 4. Jacobian for coordinate transformations
    print("\n4. JACOBIAN FOR COORDINATE TRANSFORMS:")
    
    from jax import jacfwd, jacrev
    
    def spherical_to_cartesian(spherical):
        """Convert spherical to Cartesian coordinates."""
        r, theta, phi = spherical
        x = r * jnp.sin(theta) * jnp.cos(phi)
        y = r * jnp.sin(theta) * jnp.sin(phi)
        z = r * jnp.cos(theta)
        return jnp.array([x, y, z])
    
    # Jacobian matrix
    jacobian = jacfwd(spherical_to_cartesian)
    
    spherical = jnp.array([1.0, jnp.pi/4, jnp.pi/3])
    J = jacobian(spherical)
    
    print(f"  Spherical: r={spherical[0]}, θ={spherical[1]:.3f}, φ={spherical[2]:.3f}")
    print(f"  Jacobian matrix:")
    print(f"{J}")
    print(f"  Determinant: {jnp.linalg.det(J):.3f}")

automatic_differentiation_astronomy()
```

## JIT Compilation: Making Python Fast

### Understanding JIT

```python
def jit_compilation_deep_dive():
    """Deep dive into JIT compilation for scientific computing."""
    
    print("JIT COMPILATION IN DETAIL")
    print("=" * 50)
    
    # 1. Basic JIT compilation
    print("\n1. BASIC JIT:")
    
    def slow_function(x):
        """Computationally intensive function."""
        result = x
        for _ in range(100):
            result = jnp.sin(result) + jnp.cos(result) * jnp.exp(-result**2)
        return result
    
    fast_function = jit(slow_function)
    
    x = jnp.linspace(-2, 2, 1000)
    
    # Time comparison
    start = time.perf_counter()
    _ = slow_function(x)
    slow_time = time.perf_counter() - start
    
    # First call (includes compilation)
    start = time.perf_counter()
    _ = fast_function(x)
    first_time = time.perf_counter() - start
    
    # Second call (already compiled)
    start = time.perf_counter()
    _ = fast_function(x)
    fast_time = time.perf_counter() - start
    
    print(f"  Slow: {slow_time*1000:.2f} ms")
    print(f"  First JIT call: {first_time*1000:.2f} ms")
    print(f"  Subsequent JIT: {fast_time*1000:.2f} ms")
    print(f"  Speedup: {slow_time/fast_time:.1f}x")
    
    # 2. JIT with static arguments
    print("\n2. STATIC ARGUMENTS:")
    
    @partial(jit, static_argnums=(1, 2))
    def simulate_galaxy(positions, n_steps, dt):
        """
        Simulate galaxy dynamics.
        n_steps and dt are static (known at compile time).
        """
        def step(pos):
            # Simplified dynamics
            center_of_mass = jnp.mean(pos, axis=0)
            forces = -(pos - center_of_mass) / 100
            return pos + forces * dt
        
        for _ in range(n_steps):
            positions = step(positions)
        
        return positions
    
    # Different compiled versions for different static args
    pos = random.normal(random.PRNGKey(0), (100, 3))
    
    # These create different compiled functions
    result1 = simulate_galaxy(pos, 10, 0.1)   # Compilation 1
    result2 = simulate_galaxy(pos, 10, 0.1)   # Uses compilation 1
    result3 = simulate_galaxy(pos, 20, 0.1)   # New compilation!
    
    print(f"  Static arguments create specialized compiled versions")
    
    # 3. JIT pitfalls to avoid
    print("\n3. JIT PITFALLS:")
    
    # ❌ BAD: Python control flow depending on values
    def bad_function(x):
        if x > 0:  # This depends on the VALUE of x
            return jnp.sin(x)
        else:
            return jnp.cos(x)
    
    # This won't work with JIT for dynamic x
    # jitted_bad = jit(bad_function)
    # jitted_bad(jnp.array(0.5))  # Would fail!
    
    # ✅ GOOD: Use JAX control flow
    def good_function(x):
        return jax.lax.cond(
            x > 0,
            lambda x: jnp.sin(x),
            lambda x: jnp.cos(x),
            x
        )
    
    jitted_good = jit(good_function)
    result = jitted_good(0.5)
    print(f"  Conditional result: {result:.3f}")
    
    # 4. Debugging JIT compilation
    print("\n4. DEBUGGING JIT:")
    
    # Use jax.debug.print inside JIT
    @jit
    def debug_function(x):
        x = x * 2
        jax.debug.print("x after doubling: {x}", x=x)
        x = jnp.sin(x)
        jax.debug.print("x after sin: {x}", x=x)
        return x
    
    result = debug_function(1.0)
    
    # Check what XLA sees
    from jax import make_jaxpr
    
    def simple_function(x, y):
        return jnp.dot(x, y) + 1
    
    x = jnp.ones((3, 3))
    y = jnp.ones((3, 3))
    
    print("\n  JAX expression tree:")
    print(make_jaxpr(simple_function)(x, y))

jit_compilation_deep_dive()
```

## Vectorization with vmap

### Parallel Operations Made Simple

```python
def vmap_for_astronomy():
    """Vectorization patterns for astronomical computations."""
    
    print("VMAP: AUTOMATIC VECTORIZATION")
    print("=" * 50)
    
    # 1. Basic vectorization
    print("\n1. VECTORIZING DISTANCE CALCULATIONS:")
    
    def angular_distance(ra1, dec1, ra2, dec2):
        """Angular distance between two points (haversine)."""
        dra = ra2 - ra1
        ddec = dec2 - dec1
        
        a = jnp.sin(ddec/2)**2 + jnp.cos(dec1) * jnp.cos(dec2) * jnp.sin(dra/2)**2
        c = 2 * jnp.arcsin(jnp.sqrt(a))
        
        return c
    
    # Vectorize over first source (compare one to many)
    vmap_one_to_many = vmap(angular_distance, in_axes=(None, None, 0, 0))
    
    # Vectorize over both (pairwise)
    vmap_pairwise = vmap(angular_distance, in_axes=(0, 0, 0, 0))
    
    # Test data
    ra_catalog = random.uniform(random.PRNGKey(0), (1000,), minval=0, maxval=2*jnp.pi)
    dec_catalog = random.uniform(random.PRNGKey(1), (1000,), minval=-jnp.pi/2, maxval=jnp.pi/2)
    
    # Distance from single source to catalog
    ra_source, dec_source = jnp.pi, 0.0
    distances = vmap_one_to_many(ra_source, dec_source, ra_catalog, dec_catalog)
    print(f"  Distances from source: shape {distances.shape}")
    print(f"  Nearest neighbor: {jnp.min(distances):.4f} rad")
    
    # 2. Nested vmap for all-pairs
    print("\n2. ALL-PAIRS DISTANCES:")
    
    # Nested vmap for N×N distance matrix
    vmap_all_pairs = vmap(
        vmap(angular_distance, in_axes=(None, None, 0, 0)),
        in_axes=(0, 0, None, None)
    )
    
    # Small sample for all-pairs
    ra_sample = ra_catalog[:10]
    dec_sample = dec_catalog[:10]
    
    distance_matrix = vmap_all_pairs(ra_sample, dec_sample, ra_sample, dec_sample)
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    print(f"  Symmetric: {jnp.allclose(distance_matrix, distance_matrix.T)}")
    
    # 3. Vectorizing complex functions
    print("\n3. VECTORIZING ORBIT INTEGRATION:")
    
    @jit
    def integrate_single_orbit(initial_conditions, n_steps):
        """Integrate single orbit."""
        r0, v0 = initial_conditions[:3], initial_conditions[3:]
        
        def step(carry, _):
            r, v = carry
            a = -r / jnp.linalg.norm(r)**3
            v_new = v + a * 0.01
            r_new = r + v_new * 0.01
            return (r_new, v_new), r_new
        
        _, trajectory = scan(step, (r0, v0), None, length=n_steps)
        return trajectory
    
    # Vectorize over different initial conditions
    vmap_orbits = vmap(integrate_single_orbit, in_axes=(0, None))
    
    # Multiple initial conditions (different eccentricities)
    initial_conditions = jnp.array([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # Circular
        [1.0, 0.0, 0.0, 0.0, 1.2, 0.0],   # Elliptical
        [1.0, 0.0, 0.0, 0.0, 0.8, 0.0],   # Elliptical
    ])
    
    trajectories = vmap_orbits(initial_conditions, 100)
    print(f"  Integrated {len(trajectories)} orbits in parallel")
    print(f"  Trajectories shape: {trajectories.shape}")
    
    # 4. Combining vmap with grad
    print("\n4. VECTORIZED GRADIENTS:")
    
    def potential(position, mass_distribution):
        """Gravitational potential at position."""
        positions, masses = mass_distribution
        distances = vmap(lambda p: jnp.linalg.norm(position - p))(positions)
        return -jnp.sum(masses / distances)
    
    # Gradient of potential
    grad_potential = grad(potential)
    
    # Vectorize gradient calculation over multiple positions
    vmap_gradient = vmap(grad_potential, in_axes=(0, None))
    
    # Mass distribution (galaxy model)
    n_masses = 100
    mass_positions = random.normal(random.PRNGKey(2), (n_masses, 3)) * 10
    masses = random.uniform(random.PRNGKey(3), (n_masses,), minval=0.1, maxval=1.0)
    mass_distribution = (mass_positions, masses)
    
    # Calculate gradient at multiple points
    test_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, 5.0, 5.0]
    ])
    
    gradients = vmap_gradient(test_positions, mass_distribution)
    print(f"  Gradients at {len(test_positions)} positions:")
    for i, (pos, grad) in enumerate(zip(test_positions, gradients)):
        print(f"    Position {pos}: Force = {-grad}")

vmap_for_astronomy()
```

## Random Numbers in JAX

### Reproducible Randomness

```python
def jax_random_numbers():
    """JAX's approach to random numbers for reproducible science."""
    
    print("RANDOM NUMBERS IN JAX")
    print("=" * 50)
    
    # 1. JAX PRNG basics
    print("\n1. PRNG BASICS:")
    
    # Create a random key
    key = random.PRNGKey(42)
    print(f"  Initial key: {key}")
    
    # Split key for independent streams
    key, subkey = random.split(key)
    print(f"  Split keys: {key}, {subkey}")
    
    # Generate random numbers
    uniform_samples = random.uniform(subkey, shape=(5,))
    print(f"  Uniform samples: {uniform_samples}")
    
    # 2. Why explicit keys matter
    print("\n2. REPRODUCIBILITY:")
    
    def monte_carlo_integration(f, n_samples, key):
        """Monte Carlo integration of function f over [0,1]³."""
        # Split key for different random numbers
        key1, key2, key3 = random.split(key, 3)
        
        x = random.uniform(key1, (n_samples,))
        y = random.uniform(key2, (n_samples,))
        z = random.uniform(key3, (n_samples,))
        
        samples = vmap(f)(x, y, z)
        return jnp.mean(samples)
    
    def test_function(x, y, z):
        return x**2 + y**2 + z**2
    
    # Same key → same result (reproducible!)
    key1 = random.PRNGKey(123)
    result1 = monte_carlo_integration(test_function, 10000, key1)
    
    key2 = random.PRNGKey(123)  # Same seed
    result2 = monte_carlo_integration(test_function, 10000, key2)
    
    print(f"  Result 1: {result1:.6f}")
    print(f"  Result 2: {result2:.6f}")
    print(f"  Identical: {result1 == result2}")
    
    # 3. Random numbers in parallel computations
    print("\n3. PARALLEL RANDOM STREAMS:")
    
    @jit
    def parallel_monte_carlo(keys, n_samples_per_thread):
        """Run MC in parallel with independent random streams."""
        
        def single_mc(key):
            samples = random.normal(key, (n_samples_per_thread,))
            return jnp.mean(samples**2)  # Estimate <x²>
        
        # vmap over different keys
        results = vmap(single_mc)(keys)
        return results
    
    # Create independent keys for parallel execution
    main_key = random.PRNGKey(0)
    n_threads = 4
    keys = random.split(main_key, n_threads)
    
    estimates = parallel_monte_carlo(keys, 10000)
    print(f"  Parallel estimates of <x²>: {estimates}")
    print(f"  Mean: {jnp.mean(estimates):.4f} (expected: 1.0)")
    
    # 4. Sampling from distributions
    print("\n4. ASTRONOMICAL DISTRIBUTIONS:")
    
    key = random.PRNGKey(42)
    
    # Initial Mass Function (power law)
    def sample_imf(key, n_stars, alpha=-2.35):
        """Sample from Salpeter IMF."""
        key1, key2 = random.split(key)
        
        # Inverse transform sampling
        u = random.uniform(key1, (n_stars,))
        m_min, m_max = 0.1, 100.0
        
        if alpha == -1:
            masses = m_min * (m_max/m_min)**u
        else:
            masses = ((m_max**(alpha+1) - m_min**(alpha+1)) * u + 
                     m_min**(alpha+1))**(1/(alpha+1))
        
        return masses
    
    masses = sample_imf(key, 1000)
    print(f"  Sampled {len(masses)} stellar masses")
    print(f"  Mass range: {jnp.min(masses):.2f} - {jnp.max(masses):.2f} M☉")
    print(f"  Mean mass: {jnp.mean(masses):.2f} M☉")

jax_random_numbers()
```

## Putting It All Together: N-Body Simulation

### Complete Example with All JAX Features

```python
def nbody_simulation_complete():
    """Complete N-body simulation showcasing all JAX features."""
    
    print("COMPLETE N-BODY SIMULATION WITH JAX")
    print("=" * 50)
    
    # Define the physics
    @jit
    def compute_forces(positions, masses):
        """Compute gravitational forces between all bodies."""
        n = len(masses)
        forces = jnp.zeros_like(positions)
        
        for i in range(n):
            # Vectorize force calculation from body i to all others
            def force_from_j(j):
                r_ij = positions[j] - positions[i]
                dist = jnp.linalg.norm(r_ij)
                # Softening to avoid singularities
                dist_soft = jnp.maximum(dist, 0.01)
                return jax.lax.cond(
                    i == j,
                    lambda _: jnp.zeros(3),
                    lambda _: masses[j] * r_ij / dist_soft**3,
                    None
                )
            
            total_force = jnp.sum(vmap(force_from_j)(jnp.arange(n)), axis=0)
            forces = forces.at[i].set(masses[i] * total_force)
        
        return forces
    
    # Integration step
    @jit
    def leapfrog_step(state, dt):
        """Single leapfrog integration step."""
        positions, velocities, masses = state
        
        # Compute forces
        forces = compute_forces(positions, masses)
        accelerations = forces / masses[:, None]
        
        # Leapfrog update
        velocities_half = velocities + 0.5 * dt * accelerations
        positions_new = positions + dt * velocities_half
        
        forces_new = compute_forces(positions_new, masses)
        accelerations_new = forces_new / masses[:, None]
        
        velocities_new = velocities_half + 0.5 * dt * accelerations_new
        
        return (positions_new, velocities_new, masses)
    
    # Energy calculations for verification
    @jit
    def total_energy(state):
        """Calculate total energy of the system."""
        positions, velocities, masses = state
        
        # Kinetic energy
        kinetic = 0.5 * jnp.sum(masses[:, None] * velocities**2)
        
        # Potential energy
        potential = 0.0
        n = len(masses)
        for i in range(n):
            for j in range(i+1, n):
                r_ij = jnp.linalg.norm(positions[i] - positions[j])
                r_soft = jnp.maximum(r_ij, 0.01)
                potential -= masses[i] * masses[j] / r_soft
        
        return kinetic + potential
    
    # Simulate function
    @jit
    def simulate(initial_state, dt, n_steps):
        """Run full simulation."""
        
        def step(carry, _):
            state = carry
            new_state = leapfrog_step(state, dt)
            energy = total_energy(new_state)
            return new_state, (new_state[0], energy)  # Return positions and energy
        
        final_state, (trajectory, energies) = scan(
            step, initial_state, None, length=n_steps
        )
        
        return final_state, trajectory, energies
    
    # Set up initial conditions (Pythagorean 3-body)
    positions = jnp.array([
        [1.0, 3.0, 0.0],
        [-2.0, -1.0, 0.0],
        [1.0, -1.0, 0.0]
    ])
    
    velocities = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    masses = jnp.array([3.0, 4.0, 5.0])
    
    initial_state = (positions, velocities, masses)
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.perf_counter()
    
    final_state, trajectory, energies = simulate(
        initial_state, dt=0.001, n_steps=10000
    )
    
    elapsed = time.perf_counter() - start_time
    print(f"  Simulated 10,000 steps in {elapsed:.3f} seconds")
    
    # Check energy conservation
    initial_energy = total_energy(initial_state)
    final_energy = total_energy(final_state)
    
    print(f"\n  Initial energy: {initial_energy:.6f}")
    print(f"  Final energy: {final_energy:.6f}")
    print(f"  Relative error: {abs(final_energy - initial_energy) / abs(initial_energy):.2e}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectories
    for i in range(3):
        ax1.plot(trajectory[:, i, 0], trajectory[:, i, 1], 
                alpha=0.7, linewidth=1, label=f'Mass {masses[i]:.1f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Three-Body Trajectories')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Energy conservation
    ax2.plot((energies - initial_energy) / abs(initial_energy))
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Relative energy error')
    ax2.set_title('Energy Conservation')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return trajectory

trajectory = nbody_simulation_complete()
```

## Try It Yourself

### Exercise 1: Differentiable Cosmology

```python
def differentiable_cosmology():
    """
    Build a differentiable cosmological distance calculator.
    
    Tasks:
    1. Implement luminosity distance as function of z, H0, Omega_m, Omega_Lambda
    2. Use grad to get derivatives with respect to all parameters
    3. JIT compile for speed
    4. Fit to supernova data using gradient descent
    """
    # Your code here
    pass
```

### Exercise 2: Vectorized Light Curve Analysis

```python
def analyze_light_curves_jax(times, fluxes, periods_to_test):
    """
    Analyze multiple light curves using JAX.
    
    Requirements:
    1. Use vmap to process multiple light curves in parallel
    2. JIT compile the period-finding algorithm
    3. Implement Lomb-Scargle using JAX operations
    4. Return best periods and their uncertainties
    """
    # Your code here
    pass
```

### Exercise 3: Differentiable Ray Tracing

```python
def ray_tracing_jax(rays, lens_params):
    """
    Differentiable ray tracing through gravitational lens.
    
    Tasks:
    1. Trace rays through gravitational potential
    2. Use grad to optimize lens parameters
    3. Implement critical curves and caustics
    4. JIT compile for real-time visualization
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **JAX = NumPy + autodiff + JIT + vmap** - Composable transformations  
✅ **Functional programming required** - No mutations, pure functions only  
✅ **Automatic differentiation** - Get gradients of any function for free  
✅ **JIT compilation** - Near C++ speeds from Python code  
✅ **vmap for parallelization** - Vectorize any function automatically  
✅ **Explicit random keys** - Reproducible randomness in parallel  
✅ **Composable transformations** - Combine JIT + grad + vmap freely  
✅ **GPU/TPU ready** - Same code runs on accelerators  

## Next Chapter Preview
JAX Advanced Patterns: Control flow, custom derivatives, and performance optimization for large-scale astronomical simulations.