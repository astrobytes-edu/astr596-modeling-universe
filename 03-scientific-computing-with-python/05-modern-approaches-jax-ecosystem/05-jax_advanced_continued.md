# ⚠️ JAX Advanced Patterns: Control Flow and Optimization (Continued)

## Memory and Performance Optimization (Continued)

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
        
        @checkpoint  # Don't store intermediate activations
        def layer(x):
            # Expensive computation
            x = jnp.tanh(x @ jnp.ones((100, 100)))
            x = jnp.sin(x) + jnp.cos(x)
            return x
        
        for _ in range(n_layers):
            x = layer(x)
        return jnp.sum(x)
    
    # Without checkpointing: stores all intermediate values
    def deep_network_no_checkpoint(x, n_layers=50):
        for _ in range(n_layers):
            x = jnp.tanh(x @ jnp.ones((100, 100)))
            x = jnp.sin(x) + jnp.cos(x)
        return jnp.sum(x)
    
    x = random.normal(random.PRNGKey(0), (100,))
    
    # Compare memory usage (conceptual - actual measurement requires profiling)
    print("  With checkpointing: recomputes forward pass during backprop")
    print("  Without checkpointing: stores all intermediate activations")
    
    # Gradient computation
    grad_checkpoint = grad(deep_network)
    grad_no_checkpoint = grad(deep_network_no_checkpoint)
    
    g1 = grad_checkpoint(x)
    g2 = grad_no_checkpoint(x)
    
    print(f"  Gradients match: {jnp.allclose(g1, g2)}")
    
    # 2. Donation for in-place updates
    print("\n2. BUFFER DONATION:")
    
    @jit
    def evolve_without_donation(state, dt):
        """Standard evolution - creates new arrays."""
        positions, velocities = state
        new_velocities = velocities - positions * dt
        new_positions = positions + new_velocities * dt
        return (new_positions, new_velocities)
    
    @jit
    def evolve_with_donation(state, dt):
        """Evolution with buffer donation - reuses memory."""
        positions, velocities = state
        # JAX can reuse input buffers when safe
        velocities = velocities - positions * dt
        positions = positions + velocities * dt
        return (positions, velocities)
    
    # Using donate_argnums for explicit donation
    @partial(jit, donate_argnums=(0,))
    def evolve_explicit_donation(state, dt):
        """Explicitly donate input buffer."""
        positions, velocities = state
        velocities = velocities - positions * dt
        positions = positions + velocities * dt
        return (positions, velocities)
    
    state = (random.normal(random.PRNGKey(0), (1000, 3)),
             random.normal(random.PRNGKey(1), (1000, 3)))
    
    # All produce same result, but memory usage differs
    result1 = evolve_without_donation(state, 0.01)
    result2 = evolve_with_donation(state, 0.01)
    # Note: after donation, original 'state' should not be used!
    result3 = evolve_explicit_donation(state, 0.01)
    
    print(f"  Results match: {jnp.allclose(result1[0], result2[0])}")
    
    # 3. Chunking large computations
    print("\n3. CHUNKING WITH SCAN:")
    
    @jit
    def chunked_matrix_multiply(A, B, chunk_size=100):
        """
        Compute A @ B in chunks to control memory.
        Useful when A or B is very large.
        """
        n, k = A.shape
        k2, m = B.shape
        assert k == k2
        
        # Process in chunks along the k dimension
        def process_chunk(carry, chunk_idx):
            result = carry
            start = chunk_idx * chunk_size
            end = jnp.minimum(start + chunk_size, k)
            
            A_chunk = A[:, start:end]
            B_chunk = B[start:end, :]
            
            result = result + A_chunk @ B_chunk
            return result, None
        
        n_chunks = (k + chunk_size - 1) // chunk_size
        initial = jnp.zeros((n, m))
        
        result, _ = scan(process_chunk, initial, jnp.arange(n_chunks))
        return result
    
    # Test chunked multiplication
    A = random.normal(random.PRNGKey(2), (100, 1000))
    B = random.normal(random.PRNGKey(3), (1000, 50))
    
    result_direct = A @ B
    result_chunked = chunked_matrix_multiply(A, B)
    
    print(f"  Chunked multiplication error: {jnp.linalg.norm(result_direct - result_chunked):.2e}")
    
    # 4. Selective computation with stop_gradient
    print("\n4. STOP_GRADIENT FOR PARTIAL DERIVATIVES:")
    
    def loss_with_regularization(params, data, lambda_reg=0.1):
        """Loss function with optional gradient stopping."""
        weights, biases = params
        x, y = data
        
        # Forward pass
        predictions = x @ weights + biases
        
        # Main loss
        main_loss = jnp.mean((predictions - y) ** 2)
        
        # Regularization (can stop gradient if needed)
        reg_loss = lambda_reg * (jnp.sum(weights**2) + jnp.sum(biases**2))
        
        # Option to not backprop through regularization
        # reg_loss = jax.lax.stop_gradient(reg_loss)
        
        return main_loss + reg_loss
    
    # Example with and without gradient stopping
    weights = random.normal(random.PRNGKey(4), (10, 5))
    biases = random.normal(random.PRNGKey(5), (5,))
    params = (weights, biases)
    
    x = random.normal(random.PRNGKey(6), (100, 10))
    y = random.normal(random.PRNGKey(7), (100, 5))
    data = (x, y)
    
    grad_fn = grad(loss_with_regularization)
    grads = grad_fn(params, data)
    
    print(f"  Weight gradient norm: {jnp.linalg.norm(grads[0]):.3f}")
    print(f"  Bias gradient norm: {jnp.linalg.norm(grads[1]):.3f}")

memory_optimization()
```

## Multi-GPU and Distributed Computing

### Data and Model Parallelism with pmap

```python
def distributed_computing():
    """Distributed and parallel computing patterns in JAX."""
    
    print("\nDISTRIBUTED COMPUTING WITH JAX")
    print("=" * 50)
    
    # Note: These examples are conceptual - actual multi-GPU requires hardware
    
    # 1. Data parallelism with pmap
    print("\n1. DATA PARALLELISM WITH PMAP:")
    
    # Get device count (will be 1 on single GPU/CPU)
    n_devices = jax.device_count()
    print(f"  Available devices: {n_devices}")
    
    @jit
    def single_device_nbody_step(positions, velocities, masses, dt):
        """N-body step on single device."""
        n = len(masses)
        forces = jnp.zeros_like(positions)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ij = positions[j] - positions[i]
                    r_norm = jnp.linalg.norm(r_ij)
                    forces = forces.at[i].add(
                        masses[j] * r_ij / (r_norm**3 + 0.01)
                    )
        
        accelerations = forces
        velocities_new = velocities + accelerations * dt
        positions_new = positions + velocities_new * dt
        
        return positions_new, velocities_new
    
    # Parallel version (conceptual - needs multiple devices)
    def parallel_nbody_step(positions, velocities, masses, dt):
        """
        N-body with particles distributed across devices.
        Each device computes forces for its particles.
        """
        # This would use pmap in practice
        # positions shape: [n_devices, particles_per_device, 3]
        
        # All-gather pattern to share positions
        all_positions = jax.lax.all_gather(positions, axis_name='devices')
        
        # Each device computes forces for its particles
        # ... (computation here)
        
        # Return updated local positions and velocities
        pass
    
    # 2. Sharding specifications
    print("\n2. SHARDING FOR LARGE ARRAYS:")
    
    from jax.sharding import PartitionSpec as P
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding
    
    # Create a mesh (conceptual - requires actual devices)
    # devices = mesh_utils.create_device_mesh((2, 2))
    # mesh = Mesh(devices, axis_names=('x', 'y'))
    
    def create_sharding_example():
        """Example of array sharding across devices."""
        
        # Define how to shard a large array
        # P('x', 'y') means shard first dim along 'x', second along 'y'
        # P('x', None) means shard first dim, replicate second
        # P(None, 'y') means replicate first, shard second
        
        spec_2d = P('x', 'y')  # Shard both dimensions
        spec_rows = P('x', None)  # Shard rows only
        spec_cols = P(None, 'y')  # Shard columns only
        spec_replicated = P(None, None)  # Replicate everywhere
        
        return spec_2d, spec_rows, spec_cols, spec_replicated
    
    specs = create_sharding_example()
    print("  Sharding patterns created (requires multi-device setup)")
    
    # 3. Collective operations
    print("\n3. COLLECTIVE OPERATIONS:")
    
    def collective_operations_example():
        """
        Examples of collective communication patterns.
        These are used inside pmap'd functions.
        """
        
        # Inside a pmap'd function:
        # local_sum = jnp.sum(local_data)
        
        # All-reduce: sum across all devices
        # global_sum = jax.lax.psum(local_sum, axis_name='devices')
        
        # All-gather: collect from all devices
        # all_data = jax.lax.all_gather(local_data, axis_name='devices')
        
        # Scatter: distribute data
        # scattered = jax.lax.pscatter(data, axis_name='devices')
        
        print("  Collective ops: psum, all_gather, pscatter")
        print("  Used for communication in parallel computations")
    
    collective_operations_example()
    
    # 4. Pipeline parallelism pattern
    print("\n4. PIPELINE PARALLELISM PATTERN:")
    
    @jit
    def pipeline_stage_1(x):
        """First stage of pipeline."""
        return jnp.tanh(x @ jnp.ones((100, 100)))
    
    @jit
    def pipeline_stage_2(x):
        """Second stage of pipeline."""
        return jnp.sin(x) + jnp.cos(x)
    
    @jit
    def pipeline_stage_3(x):
        """Third stage of pipeline."""
        return x @ jnp.ones((100, 50))
    
    def pipelined_computation(batches):
        """
        Process batches through pipeline.
        In practice, stages would run on different devices.
        """
        results = []
        
        # Simplified pipeline (no overlap)
        for batch in batches:
            x = pipeline_stage_1(batch)
            x = pipeline_stage_2(x)
            x = pipeline_stage_3(x)
            results.append(x)
        
        return jnp.stack(results)
    
    # Test pipeline
    batches = [random.normal(random.PRNGKey(i), (100,)) for i in range(4)]
    results = pipelined_computation(batches)
    print(f"  Pipeline processed {len(batches)} batches")
    print(f"  Output shape: {results.shape}")

distributed_computing()
```

## Profiling and Debugging

### Performance Analysis Tools

```python
def profiling_and_debugging():
    """Tools and techniques for profiling JAX code."""
    
    print("\nPROFILING AND DEBUGGING JAX")
    print("=" * 50)
    
    # 1. Basic timing
    print("\n1. BASIC PERFORMANCE TIMING:")
    
    def time_function(f, *args, n_runs=100):
        """Time a JIT-compiled function."""
        # Compile
        f_jit = jit(f)
        _ = f_jit(*args)  # Trigger compilation
        
        # Time
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = f_jit(*args)
        elapsed = time.perf_counter() - start
        
        return elapsed / n_runs
    
    def test_function(x):
        for _ in range(10):
            x = jnp.sin(x) @ jnp.cos(x.T)
        return x
    
    x = random.normal(random.PRNGKey(0), (100, 100))
    avg_time = time_function(test_function, x)
    print(f"  Average time: {avg_time*1000:.3f} ms")
    
    # 2. Block until result is ready
    print("\n2. BLOCKING FOR ACCURATE TIMING:")
    
    @jit
    def async_computation(x):
        """JAX computations are asynchronous."""
        return jnp.sum(x @ x.T)
    
    x = random.normal(random.PRNGKey(1), (1000, 1000))
    
    # Wrong way (doesn't wait for completion)
    start = time.perf_counter()
    result = async_computation(x)  # Returns immediately!
    wrong_time = time.perf_counter() - start
    
    # Right way (blocks until ready)
    start = time.perf_counter()
    result = async_computation(x)
    result.block_until_ready()  # Wait for computation
    correct_time = time.perf_counter() - start
    
    print(f"  Without blocking: {wrong_time*1000:.3f} ms (incorrect!)")
    print(f"  With blocking: {correct_time*1000:.3f} ms (correct)")
    
    # 3. Compilation inspection
    print("\n3. INSPECTING COMPILATION:")
    
    from jax import make_jaxpr
    from jax._src.lib import xla_client
    
    def inspect_function(x, y):
        """Function to inspect."""
        z = x @ y
        return jnp.sum(z * z)
    
    x = jnp.ones((3, 3))
    y = jnp.ones((3, 3))
    
    # JAX expression
    jaxpr = make_jaxpr(inspect_function)(x, y)
    print("\n  JAX expression (simplified IR):")
    print("  " + str(jaxpr).split('\n')[0][:60] + "...")
    
    # Lowered to XLA
    lowered = jit(inspect_function).lower(x, y)
    print("\n  XLA HLO module available for inspection")
    
    # 4. Debug prints and assertions
    print("\n4. DEBUGGING TOOLS:")
    
    @jit
    def debug_example(x):
        """Using debug utilities inside JIT."""
        
        # Debug print (works inside JIT)
        jax.debug.print("Input shape: {}", x.shape)
        
        # Intermediate values
        y = jnp.sin(x)
        jax.debug.print("After sin - mean: {:.3f}, std: {:.3f}", 
                       jnp.mean(y), jnp.std(y))
        
        # Assertions (converted to runtime checks)
        # jax.debug.assert_(jnp.all(jnp.isfinite(y)), "NaN detected!")
        
        z = y @ y.T
        jax.debug.print("Final shape: {}", z.shape)
        
        return z
    
    x = random.normal(random.PRNGKey(2), (5, 5))
    result = debug_example(x)
    
    # 5. Finding bottlenecks
    print("\n5. IDENTIFYING BOTTLENECKS:")
    
    def find_bottlenecks():
        """Strategies for finding performance issues."""
        
        # Common bottlenecks:
        bottlenecks = {
            "Python loops": "Use scan/fori_loop instead",
            "Small operations": "Batch into larger operations",
            "Recompilation": "Check for changing shapes/types",
            "Host-device transfer": "Minimize data movement",
            "Unintended float64": "Use float32 by default",
            "Missing JIT": "JIT-compile hot functions",
            "Bad memory access": "Optimize data layout"
        }
        
        for issue, solution in bottlenecks.items():
            print(f"    {issue}: {solution}")
    
    find_bottlenecks()

profiling_and_debugging()
```

## Real-World Example: Galaxy Simulation

### Production-Ready N-Body Code

```python
def galaxy_simulation_production():
    """
    Production-ready galaxy simulation using all JAX features.
    Demonstrates best practices for scientific computing.
    """
    
    print("\nPRODUCTION GALAXY SIMULATION")
    print("=" * 50)
    
    # Configuration
    @jit
    def create_galaxy(n_stars, key, galaxy_type='spiral'):
        """Initialize a galaxy with realistic structure."""
        
        key1, key2, key3, key4 = random.split(key, 4)
        
        if galaxy_type == 'spiral':
            # Spiral galaxy with disk and bulge
            
            # Disk component (exponential profile)
            radii = random.exponential(key1, (int(0.8 * n_stars),)) * 10.0
            angles = random.uniform(key2, (int(0.8 * n_stars),), 
                                   minval=0, maxval=2*jnp.pi)
            
            # Add spiral structure
            spiral_phase = 2.0 * jnp.log(radii + 1)
            angles = angles + spiral_phase
            
            disk_x = radii * jnp.cos(angles)
            disk_y = radii * jnp.sin(angles)
            disk_z = random.normal(key3, (int(0.8 * n_stars),)) * 0.5
            
            # Bulge component (Hernquist profile)
            n_bulge = n_stars - int(0.8 * n_stars)
            r_bulge = random.uniform(key4, (n_bulge,)) ** (1/3) * 3.0
            theta = jnp.arccos(1 - 2 * random.uniform(key1, (n_bulge,)))
            phi = random.uniform(key2, (n_bulge,), minval=0, maxval=2*jnp.pi)
            
            bulge_x = r_bulge * jnp.sin(theta) * jnp.cos(phi)
            bulge_y = r_bulge * jnp.sin(theta) * jnp.sin(phi)
            bulge_z = r_bulge * jnp.cos(theta)
            
            # Combine
            positions = jnp.concatenate([
                jnp.stack([disk_x, disk_y, disk_z], axis=1),
                jnp.stack([bulge_x, bulge_y, bulge_z], axis=1)
            ])
            
            # Circular velocities (simplified)
            all_radii = jnp.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
            v_circ = jnp.sqrt(all_radii / (all_radii + 1.0))  # Rotation curve
            
            velocities = jnp.zeros_like(positions)
            velocities = velocities.at[:, 0].set(-positions[:, 1] / all_radii * v_circ)
            velocities = velocities.at[:, 1].set(positions[:, 0] / all_radii * v_circ)
            
            # Masses (IMF-like distribution)
            masses = random.pareto(key3, 2.35, (n_stars,)) * 0.1 + 0.1
            masses = jnp.clip(masses, 0.1, 10.0)
            
        return positions, velocities, masses
    
    # Optimized force calculation
    @jit
    def compute_forces_fast(positions, masses, softening=0.1):
        """
        Fast force calculation using vectorization.
        O(N²) but highly optimized.
        """
        n = len(masses)
        
        # Compute all pairwise vectors at once
        r_ij = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
        
        # Distances
        r2 = jnp.sum(r_ij**2, axis=2) + softening**2  # (n, n)
        r3 = r2 ** 1.5
        
        # Mask diagonal (self-interaction)
        mask = 1.0 - jnp.eye(n)
        
        # Forces
        F_ij = r_ij / r3[:, :, None] * mask[:, :, None]  # (n, n, 3)
        forces = jnp.sum(masses[None, :, None] * F_ij, axis=1)  # (n, 3)
        
        return forces * masses[:, None]
    
    # Adaptive time-stepping
    @jit
    def adaptive_timestep(positions, velocities, masses, base_dt=0.01):
        """Compute adaptive timestep based on local dynamics."""
        
        forces = compute_forces_fast(positions, masses)
        accelerations = forces / masses[:, None]
        
        # Criteria for timestep
        v_mag = jnp.linalg.norm(velocities, axis=1)
        a_mag = jnp.linalg.norm(accelerations, axis=1)
        
        # Courant condition
        dt_courant = jnp.min(0.1 / (v_mag + 1e-10))
        
        # Acceleration condition
        dt_accel = jnp.min(jnp.sqrt(0.1 / (a_mag + 1e-10)))
        
        # Take minimum
        dt = jnp.minimum(dt_courant, dt_accel)
        dt = jnp.minimum(dt, base_dt)
        
        return dt
    
    # Main evolution with all features
    @partial(jit, static_argnums=(3,))
    def evolve_galaxy(positions, velocities, masses, n_steps, checkpoint_every=100):
        """
        Full galaxy evolution with checkpointing.
        """
        
        def step(carry, i):
            pos, vel = carry
            
            # Adaptive timestep
            dt = adaptive_timestep(pos, vel, masses)
            
            # Leapfrog integration
            forces = compute_forces_fast(pos, masses)
            acc = forces / masses[:, None]
            
            vel_half = vel + 0.5 * dt * acc
            pos_new = pos + dt * vel_half
            
            forces_new = compute_forces_fast(pos_new, masses)
            acc_new = forces_new / masses[:, None]
            
            vel_new = vel_half + 0.5 * dt * acc_new
            
            # Energy for monitoring
            ke = 0.5 * jnp.sum(masses[:, None] * vel_new**2)
            
            # Checkpoint decision (would save to disk in practice)
            should_checkpoint = (i % checkpoint_every) == 0
            
            return (pos_new, vel_new), (pos_new, ke, dt)
        
        final_state, (trajectory, energies, timesteps) = scan(
            step, (positions, velocities), jnp.arange(n_steps)
        )
        
        return final_state, trajectory[::checkpoint_every], energies[::checkpoint_every]
    
    # Initialize and run
    key = random.PRNGKey(42)
    n_stars = 1000
    
    print(f"\nInitializing spiral galaxy with {n_stars} stars...")
    positions, velocities, masses = create_galaxy(n_stars, key)
    
    print("Running simulation...")
    start_time = time.perf_counter()
    
    final_state, checkpoints, energies = evolve_galaxy(
        positions, velocities, masses, n_steps=1000, checkpoint_every=100
    )
    
    elapsed = time.perf_counter() - start_time
    print(f"  Completed in {elapsed:.2f} seconds")
    print(f"  Steps per second: {1000/elapsed:.0f}")
    
    # Analysis
    initial_energy = energies[0]
    final_energy = energies[-1]
    print(f"\nEnergy conservation:")
    print(f"  Initial: {initial_energy:.2f}")
    print(f"  Final: {final_energy:.2f}")
    print(f"  Drift: {abs(final_energy - initial_energy) / abs(initial_energy) * 100:.2f}%")
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # Initial configuration
    ax1 = fig.add_subplot(131)
    ax1.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c=masses, cmap='YlOrRd')
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)
    ax1.set_xlabel('X [kpc]')
    ax1.set_ylabel('Y [kpc]')
    ax1.set_title('Initial Galaxy')
    ax1.set_aspect('equal')
    
    # Final configuration
    ax2 = fig.add_subplot(132)
    final_pos = final_state[0]
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], s=1, alpha=0.5, c=masses, cmap='YlOrRd')
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    ax2.set_xlabel('X [kpc]')
    ax2.set_ylabel('Y [kpc]')
    ax2.set_title('Final Galaxy')
    ax2.set_aspect('equal')
    
    # Energy evolution
    ax3 = fig.add_subplot(133)
    ax3.plot(energies)
    ax3.set_xlabel('Checkpoint')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Energy Conservation')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return checkpoints

# Run the production simulation
checkpoints = galaxy_simulation_production()
```

## Best Practices Summary

### JAX Do's and Don'ts

```python
def best_practices():
    """Summary of JAX best practices for astronomical computing."""
    
    print("\nJAX BEST PRACTICES FOR ASTRONOMY")
    print("=" * 50)
    
    practices = {
        "✅ DO": [
            "Use JIT on hot loops and expensive functions",
            "Write pure functional code without side effects",
            "Use scan for sequential computations",
            "Vectorize with vmap instead of loops",
            "Profile and measure performance",
            "Use float32 by default for speed",
            "Leverage automatic differentiation",
            "Think in terms of array operations",
            "Use static_argnums for compile-time constants",
            "Test numerical stability and accuracy"
        ],
        
        "❌ DON'T": [
            "Use Python loops inside JIT functions",
            "Mutate arrays (use .at[].set() instead)",
            "Use varying shapes (causes recompilation)",
            "Forget to block_until_ready() when timing",
            "Mix NumPy and JAX arrays carelessly",
            "Use float64 unless necessary",
            "Ignore memory consumption",
            "Use global variables in JIT functions",
            "Forget to split random keys",
            "Skip validation of gradients"
        ],
        
        "🚀 ADVANCED": [
            "Custom VJP rules for numerical stability",
            "Checkpointing for memory efficiency",
            "Sharding for multi-GPU parallelism",
            "Pipeline parallelism for deep models",
            "Mixed precision for performance",
            "Custom linear algebra primitives",
            "Optimize data layout for cache",
            "Use donation to reuse buffers",
            "Profile XLA compilation",
            "Implement custom CUDA kernels"
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

best_practices()
```

## Exercises

### Exercise 1: Adaptive Mesh Refinement
```python
def adaptive_mesh_refinement():
    """
    Implement AMR for solving Poisson equation in JAX.
    
    Requirements:
    - Use while_loop for adaptive refinement
    - Custom VJP for interpolation operators
    - JIT compile the multigrid solver
    - Verify convergence with manufactured solution
    """
    # Your code here
    pass
```

### Exercise 2: Differentiable Radiative Transfer
```python
def differentiable_rt():
    """
    Build differentiable radiative transfer solver.
    
    Tasks:
    - Implement ray marching with scan
    - Use custom derivatives for optical depth
    - Optimize source function parameters via gradient descent
    - Compare with Monte Carlo solution
    """
    # Your code here
    pass
```

### Exercise 3: Parallel MCMC Sampler
```python
def parallel_mcmc():
    """
    Implement parallel tempered MCMC in JAX.
    
    Requirements:
    - Use pmap for parallel chains
    - Implement replica exchange with collective ops
    - JIT compile the likelihood and proposals
    - Test on cosmological parameter estimation
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **Control flow** - Use lax.cond, scan, while_loop for conditionals and loops  
✅ **Custom derivatives** - Define VJP/JVP rules for numerical stability  
✅ **Memory optimization** - Checkpointing, donation, and chunking strategies  
✅ **Parallelism** - pmap for multi-device, sharding for large arrays  
✅ **Profiling** - Always measure, use block_until_ready, inspect compilation  
✅ **Production code** - Combine all features for real scientific applications  

## Next Steps

With these advanced patterns, you're ready to:
1. Port existing astronomical codes to JAX
2. Build differentiable physical models
3. Scale to multi-GPU clusters
4. Contribute to JAX ecosystem libraries
5. Develop novel computational methods

Remember: JAX rewards thinking in terms of transformations and functional programming. The initial learning curve pays off with unprecedented performance and capabilities for scientific computing!