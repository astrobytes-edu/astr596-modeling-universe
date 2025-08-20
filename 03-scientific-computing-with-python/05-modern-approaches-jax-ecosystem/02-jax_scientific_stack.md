# ⚠️ JAX Scientific Computing Stack: Equinox, Diffrax, and Optimistix

## Learning Objectives
By the end of this chapter, you will:
- Build neural networks and scientific models with Equinox
- Solve ODEs/SDEs with Diffrax's advanced solvers
- Implement root-finding and optimization with Optimistix
- Use jaxtyping for runtime type checking
- Combine these libraries for complex astronomical simulations

## Equinox: Neural Networks as PyTrees

### Introduction to Equinox

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree, jaxtyped
from typing import Optional
import matplotlib.pyplot as plt
import time

def equinox_fundamentals():
    """Learn Equinox's approach to neural networks and models."""
    
    print("EQUINOX: NEURAL NETWORKS AS PYTREES")
    print("=" * 50)
    
    # 1. Basic neural network
    print("\n1. SIMPLE NEURAL NETWORK:")
    
    class StellarClassifier(eqx.Module):
        """Classify stellar types from spectra."""
        layers: list
        dropout: eqx.nn.Dropout
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, key):
            keys = random.split(key, 4)
            
            self.layers = [
                eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
                eqx.nn.Linear(hidden_dim, output_dim, key=keys[2])
            ]
            self.dropout = eqx.nn.Dropout(p=0.1)
        
        def __call__(self, x: Float[Array, "wavelengths"], *, key: Optional[random.PRNGKey] = None) -> Float[Array, "classes"]:
            # First hidden layer
            x = self.layers[0](x)
            x = jax.nn.relu(x)
            
            # Dropout during training
            if key is not None:
                x = self.dropout(x, key=key)
            
            # Second hidden layer
            x = self.layers[1](x)
            x = jax.nn.relu(x)
            
            # Output layer
            x = self.layers[2](x)
            return jax.nn.log_softmax(x)
    
    # Initialize model
    key = random.PRNGKey(0)
    model = StellarClassifier(input_dim=1000, hidden_dim=128, output_dim=7, key=key)
    
    # Test forward pass
    spectrum = random.normal(key, (1000,))
    logits = model(spectrum)
    print(f"  Input shape: {spectrum.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Predicted class: {jnp.argmax(logits)}")
    
    # 2. Model manipulation as PyTrees
    print("\n2. PYTREE OPERATIONS:")
    
    # Get model parameters
    params, static = eqx.partition(model, eqx.is_array)
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {num_params:,}")
    
    # Modify parameters
    def add_noise(params, key, noise_scale=0.01):
        """Add noise to parameters."""
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = random.split(key, len(leaves))
        
        noisy_leaves = [
            leaf + noise_scale * random.normal(k, leaf.shape)
            for leaf, k in zip(leaves, keys)
        ]
        
        return jax.tree_util.tree_unflatten(treedef, noisy_leaves)
    
    noisy_params = add_noise(params, key)
    noisy_model = eqx.combine(noisy_params, static)
    
    # 3. Advanced model with custom layers
    print("\n3. CUSTOM LAYERS AND MODULES:")
    
    class SpectralConvolution(eqx.Module):
        """1D convolution for spectral features."""
        weight: Float[Array, "out_channels in_channels kernel_size"]
        bias: Optional[Float[Array, "out_channels"]]
        
        def __init__(self, in_channels, out_channels, kernel_size, *, key, use_bias=True):
            wkey, bkey = random.split(key)
            
            # He initialization
            lim = jnp.sqrt(2.0 / (in_channels * kernel_size))
            self.weight = random.uniform(
                wkey, 
                (out_channels, in_channels, kernel_size),
                minval=-lim, maxval=lim
            )
            
            if use_bias:
                self.bias = jnp.zeros((out_channels,))
            else:
                self.bias = None
        
        def __call__(self, x: Float[Array, "batch channels length"]) -> Float[Array, "batch out_channels new_length"]:
            # Apply convolution
            out = jax.lax.conv_general_dilated(
                x, self.weight, 
                window_strides=(1,),
                padding='SAME'
            )
            
            if self.bias is not None:
                out = out + self.bias[None, :, None]
            
            return out
    
    # Use custom layer
    conv_layer = SpectralConvolution(1, 16, kernel_size=5, key=key)
    test_input = random.normal(key, (32, 1, 100))  # batch, channels, length
    output = conv_layer(test_input)
    print(f"  Conv input: {test_input.shape}")
    print(f"  Conv output: {output.shape}")
    
    # 4. Filtering and freezing
    print("\n4. FILTERING AND FREEZING PARAMETERS:")
    
    # Filter specific layers
    def get_linear_params(model):
        """Extract only Linear layer parameters."""
        return eqx.filter(model, lambda x: isinstance(x, eqx.nn.Linear))
    
    linear_params = get_linear_params(model)
    
    # Freeze layers
    @eqx.filter_jit
    def forward_with_frozen_first_layer(model, x):
        """Forward pass with first layer frozen."""
        # Partition into first layer and rest
        first_layer = model.layers[0]
        other_layers = model.layers[1:]
        
        # Stop gradient on first layer
        x = jax.lax.stop_gradient(first_layer(x))
        x = jax.nn.relu(x)
        
        # Continue with gradients
        for layer in other_layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        
        x = other_layers[-1](x)
        return x
    
    print("  Filtered and frozen layer operations configured")

equinox_fundamentals()
```

### Training with Equinox

```python
def equinox_training():
    """Training loops and optimization with Equinox."""
    
    print("\nTRAINING WITH EQUINOX")
    print("=" * 50)
    
    import optax
    
    # 1. Define model for photometric redshift
    print("\n1. PHOTOMETRIC REDSHIFT MODEL:")
    
    class PhotoZNet(eqx.Module):
        """Estimate redshift from photometry."""
        encoder: eqx.nn.Sequential
        decoder: eqx.nn.Sequential
        
        def __init__(self, n_bands: int, key):
            key1, key2 = random.split(key)
            
            self.encoder = eqx.nn.Sequential([
                eqx.nn.Linear(n_bands, 64, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(64, 32, key=key2),
                eqx.nn.Lambda(jax.nn.relu),
            ])
            
            self.decoder = eqx.nn.Sequential([
                eqx.nn.Linear(32, 16, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(16, 1, key=key2),  # Single redshift value
            ])
        
        def __call__(self, x: Float[Array, "n_bands"]) -> Float[Array, "1"]:
            features = self.encoder(x)
            return self.decoder(features).squeeze()
    
    # Initialize
    key = random.PRNGKey(42)
    model = PhotoZNet(n_bands=5, key=key)
    
    # 2. Loss functions
    print("\n2. LOSS FUNCTIONS:")
    
    @eqx.filter_jit
    def loss_fn(model, x, y_true):
        """MSE loss with outlier robustness."""
        y_pred = vmap(model)(x)
        
        # Huber loss for robustness
        delta = 0.1
        residual = jnp.abs(y_pred - y_true)
        
        loss = jnp.where(
            residual < delta,
            0.5 * residual ** 2,
            delta * residual - 0.5 * delta ** 2
        )
        
        return jnp.mean(loss)
    
    # Generate synthetic data
    key, subkey = random.split(key)
    n_samples = 1000
    X_train = random.normal(subkey, (n_samples, 5))
    z_true = random.uniform(subkey, (n_samples,), minval=0, maxval=3)
    
    # 3. Training loop
    print("\n3. TRAINING LOOP:")
    
    # Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch):
        """Single training step."""
        # Compute loss and gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    # Training
    batch_size = 32
    n_epochs = 10
    
    for epoch in range(n_epochs):
        # Shuffle data
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        X_shuffled = X_train[perm]
        z_shuffled = z_true[perm]
        
        # Mini-batches
        epoch_loss = 0.0
        n_batches = n_samples // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            x_batch = X_shuffled[start:end]
            y_batch = z_shuffled[start:end]
            
            model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
            epoch_loss += loss
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/n_batches:.4f}")
    
    # 4. Model evaluation
    print("\n4. MODEL EVALUATION:")
    
    @eqx.filter_jit
    def evaluate(model, x, y_true):
        """Evaluate model performance."""
        y_pred = vmap(model)(x)
        
        # Metrics
        mse = jnp.mean((y_pred - y_true) ** 2)
        mae = jnp.mean(jnp.abs(y_pred - y_true))
        
        # Outlier fraction (|Δz| > 0.15)
        outliers = jnp.sum(jnp.abs(y_pred - y_true) > 0.15) / len(y_true)
        
        return {'mse': mse, 'mae': mae, 'outlier_frac': outliers}
    
    # Test set
    X_test = random.normal(random.PRNGKey(123), (200, 5))
    z_test = random.uniform(random.PRNGKey(124), (200,), minval=0, maxval=3)
    
    metrics = evaluate(model, X_test, z_test)
    print(f"  Test MSE: {metrics['mse']:.4f}")
    print(f"  Test MAE: {metrics['mae']:.4f}")
    print(f"  Outlier fraction: {metrics['outlier_frac']:.2%}")

equinox_training()
```

## Diffrax: Advanced ODE/SDE Solvers

### Solving ODEs with Diffrax

```python
import diffrax

def diffrax_ode_solvers():
    """Advanced ODE solving for astronomical systems."""
    
    print("\nDIFFRAX: ADVANCED ODE SOLVERS")
    print("=" * 50)
    
    # 1. Basic ODE: Binary star system
    print("\n1. BINARY STAR SYSTEM:")
    
    def binary_system(t, y, args):
        """Binary star dynamics with tidal effects."""
        # y = [r1, v1, r2, v2] (6D each)
        r1, v1, r2, v2 = y[0:3], y[3:6], y[6:9], y[9:12]
        m1, m2 = args['m1'], args['m2']
        
        # Gravitational forces
        r = r2 - r1
        r_norm = jnp.linalg.norm(r)
        f_grav = r / (r_norm**3 + 1e-10)
        
        # Tidal forces (simplified)
        a1 = m2 * f_grav
        a2 = -m1 * f_grav
        
        return jnp.concatenate([v1, a1, v2, a2])
    
    # Initial conditions
    y0 = jnp.array([
        1.0, 0.0, 0.0,    # r1
        0.0, 0.3, 0.0,    # v1
        -1.0, 0.0, 0.0,   # r2
        0.0, -0.3, 0.0    # v2
    ])
    
    # Solve with different methods
    args = {'m1': 1.0, 'm2': 0.8}
    t0, t1 = 0.0, 100.0
    dt0 = 0.01
    
    # Dopri5 (adaptive RK4/5)
    solver = diffrax.Dopri5()
    term = diffrax.ODETerm(binary_system)
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    
    sol = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0,
        args=args, saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10)
    )
    
    print(f"  Solved with {sol.stats['num_steps']} steps")
    print(f"  Final positions: r1={sol.ys[-1, 0:3]}, r2={sol.ys[-1, 6:9]}")
    
    # 2. Stiff ODEs: Chemical evolution
    print("\n2. STIFF SYSTEM - CHEMICAL NETWORK:")
    
    def chemical_network(t, y, args):
        """Simplified CNO cycle in stellar interior."""
        # y = [C12, N14, O16]
        C12, N14, O16 = y
        T = args['temperature']  # in 10^7 K
        
        # Reaction rates (simplified)
        k1 = 1e-2 * jnp.exp(-15.0 / T)  # C12 + p -> N14
        k2 = 1e-3 * jnp.exp(-20.0 / T)  # N14 + p -> O16
        k3 = 1e-4 * jnp.exp(-25.0 / T)  # O16 + p -> C12
        
        dC12_dt = -k1 * C12 + k3 * O16
        dN14_dt = k1 * C12 - k2 * N14
        dO16_dt = k2 * N14 - k3 * O16
        
        return jnp.array([dC12_dt, dN14_dt, dO16_dt])
    
    # Use implicit solver for stiff system
    y0_chem = jnp.array([1.0, 0.0, 0.0])  # Start with pure C12
    
    solver_stiff = diffrax.Kvaerno5()  # Implicit solver
    term_stiff = diffrax.ODETerm(chemical_network)
    
    sol_stiff = diffrax.diffeqsolve(
        term_stiff, solver_stiff, 0.0, 1e10, 1e6, y0_chem,
        args={'temperature': 2.0},  # 20 million K
        max_steps=10000
    )
    
    print(f"  Final abundances: C12={sol_stiff.ys[-1, 0]:.3f}, " +
          f"N14={sol_stiff.ys[-1, 1]:.3f}, O16={sol_stiff.ys[-1, 2]:.3f}")
    
    # 3. Event detection
    print("\n3. EVENT DETECTION - PERICENTER PASSAGE:")
    
    def detect_pericenter(state, **kwargs):
        """Detect when binary reaches pericenter."""
        r1, v1, r2, v2 = state[0:3], state[3:6], state[6:9], state[9:12]
        separation = jnp.linalg.norm(r2 - r1)
        return separation - 0.5  # Trigger when separation < 0.5
    
    event = diffrax.DiscreteTerminatingEvent(detect_pericenter)
    
    sol_event = diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0,
        args=args,
        discrete_terminating_event=event
    )
    
    if sol_event.event_mask:
        print(f"  Pericenter reached at t={sol_event.ts[-1]:.2f}")
    
    # 4. Sensitivity analysis
    print("\n4. SENSITIVITY ANALYSIS:")
    
    def binary_with_sensitivity(t, y, args):
        """Binary system with parameter sensitivity."""
        return binary_system(t, y, args)
    
    # Gradient with respect to initial conditions
    @jit
    def trajectory_loss(y0, args):
        """Loss based on final state."""
        sol = diffrax.diffeqsolve(
            term, solver, t0, 10.0, dt0, y0,
            args=args
        )
        return jnp.sum(sol.ys[-1] ** 2)
    
    grad_fn = grad(trajectory_loss)
    sensitivity = grad_fn(y0, args)
    print(f"  Sensitivity of final state to initial conditions:")
    print(f"  Max gradient: {jnp.max(jnp.abs(sensitivity)):.3e}")

diffrax_ode_solvers()
```

### Stochastic Differential Equations

```python
def diffrax_sde_solvers():
    """Solve SDEs for stochastic astronomical processes."""
    
    print("\nSTOCHASTIC DIFFERENTIAL EQUATIONS")
    print("=" * 50)
    
    # 1. Brownian motion in globular cluster
    print("\n1. STELLAR BROWNIAN MOTION:")
    
    def drift(t, y, args):
        """Drift term: gravitational force."""
        # Simplified central potential
        r = jnp.linalg.norm(y[:3])
        force = -y[:3] / (r**3 + 0.1)
        return jnp.concatenate([y[3:6], force])
    
    def diffusion(t, y, args):
        """Diffusion term: random kicks from encounters."""
        sigma = args['sigma']
        # Only velocity gets random kicks
        return jnp.concatenate([
            jnp.zeros(3),
            jnp.eye(3) * sigma
        ])
    
    # Setup SDE
    key = random.PRNGKey(0)
    
    brownian_motion = diffrax.VirtualBrownianTree(
        t0=0.0, t1=100.0, tol=1e-3, shape=(3,), key=key
    )
    
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, brownian_motion)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    
    solver_sde = diffrax.Euler()  # Euler-Maruyama for SDEs
    
    y0_sde = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # position, velocity
    
    # Solve SDE
    sol_sde = diffrax.diffeqsolve(
        terms, solver_sde, 0.0, 100.0, 0.01, y0_sde,
        args={'sigma': 0.01},
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, 100, 1000))
    )
    
    positions = sol_sde.ys[:, :3]
    print(f"  Final position: {positions[-1]}")
    print(f"  RMS displacement: {jnp.sqrt(jnp.mean(jnp.sum(positions**2, axis=1))):.3f}")
    
    # 2. Stochastic accretion
    print("\n2. STOCHASTIC ACCRETION DISK:")
    
    def accretion_drift(t, y, args):
        """Mean accretion rate."""
        M, mdot = y
        alpha = args['alpha']  # Viscosity parameter
        
        # Simplified accretion model
        dM_dt = mdot
        dmdot_dt = -alpha * mdot  # Decay of accretion
        
        return jnp.array([dM_dt, dmdot_dt])
    
    def accretion_noise(t, y, args):
        """Turbulent fluctuations."""
        M, mdot = y
        beta = args['beta']
        
        # Noise proportional to accretion rate
        return jnp.array([[0.0], [beta * jnp.sqrt(jnp.abs(mdot))]])
    
    # Setup
    key, subkey = random.split(key)
    brownian_1d = diffrax.VirtualBrownianTree(
        t0=0.0, t1=10.0, tol=1e-3, shape=(1,), key=subkey
    )
    
    drift_acc = diffrax.ODETerm(accretion_drift)
    diffusion_acc = diffrax.ControlTerm(accretion_noise, brownian_1d)
    terms_acc = diffrax.MultiTerm(drift_acc, diffusion_acc)
    
    y0_acc = jnp.array([1.0, 0.1])  # Initial mass and accretion rate
    
    sol_acc = diffrax.diffeqsolve(
        terms_acc, solver_sde, 0.0, 10.0, 0.001, y0_acc,
        args={'alpha': 0.1, 'beta': 0.05}
    )
    
    print(f"  Final mass: {sol_acc.ys[-1, 0]:.3f}")
    print(f"  Final accretion rate: {sol_acc.ys[-1, 1]:.4f}")
    
    # 3. Jump diffusion for flares
    print("\n3. JUMP DIFFUSION - STELLAR FLARES:")
    
    class FlareProcess(eqx.Module):
        """Jump process for stellar flares."""
        rate: float = 0.1  # Flare rate
        
        def __call__(self, t, y, args, key):
            """Generate jump if flare occurs."""
            subkey1, subkey2 = random.split(key)
            
            # Poisson process for flare occurrence
            occurs = random.uniform(subkey1) < self.rate * 0.01  # dt = 0.01
            
            # Flare amplitude (log-normal distribution)
            amplitude = jnp.where(
                occurs,
                jnp.exp(random.normal(subkey2) * 0.5),
                0.0
            )
            
            return jnp.array([amplitude])
    
    # Combined drift-diffusion-jump process
    def luminosity_drift(t, y, args):
        """Decay of luminosity."""
        return -0.1 * y  # Exponential decay
    
    # Solve with jumps (simplified - Diffrax doesn't have built-in jump diffusion)
    # This would require custom implementation
    
    print("  Jump diffusion setup complete (requires custom implementation)")

diffrax_sde_solvers()
```

## Optimistix: Root Finding and Optimization

### Nonlinear Solvers

```python
import optimistix as optx

def optimistix_solvers():
    """Root finding and optimization for astronomical problems."""
    
    print("\nOPTIMISTIX: ROOT FINDING & OPTIMIZATION")
    print("=" * 50)
    
    # 1. Root finding: Kepler's equation
    print("\n1. KEPLER'S EQUATION:")
    
    @jit
    def kepler_equation(E, M_and_e):
        """Kepler's equation: M = E - e*sin(E)"""
        M, e = M_and_e
        return E - e * jnp.sin(E) - M
    
    # Solve for different mean anomalies
    M_values = jnp.linspace(0, 2*jnp.pi, 10)
    e = 0.5  # Eccentricity
    
    E_solutions = []
    for M in M_values:
        solver = optx.Newton(rtol=1e-8, atol=1e-10)
        sol = optx.root_find(
            lambda E: kepler_equation(E, (M, e)),
            solver,
            M,  # Initial guess
            max_steps=100
        )
        E_solutions.append(sol.value)
    
    E_solutions = jnp.array(E_solutions)
    print(f"  Solved {len(E_solutions)} values")
    print(f"  Max residual: {max(abs(kepler_equation(E, (M, e))) for E, M in zip(E_solutions, M_values)):.2e}")
    
    # 2. Least squares: Orbit fitting
    print("\n2. ORBIT FITTING WITH LEAST SQUARES:")
    
    def orbit_residuals(params, data):
        """Residuals for orbit fitting."""
        a, e, i, omega, Omega, T0 = params
        times, ra, dec = data
        
        # Compute predicted positions (simplified)
        n = 2 * jnp.pi / (a ** 1.5)  # Mean motion
        M = n * (times - T0)
        
        # Would solve Kepler's equation here for each M
        # For simplicity, use small eccentricity approximation
        E = M + e * jnp.sin(M)
        
        # True anomaly (simplified)
        f = E + 2 * e * jnp.sin(E)
        
        # Predicted RA/Dec (very simplified!)
        ra_pred = a * jnp.cos(f + omega)
        dec_pred = a * jnp.sin(f + omega) * jnp.sin(i)
        
        return jnp.concatenate([ra - ra_pred, dec - dec_pred])
    
    # Synthetic observations
    key = random.PRNGKey(42)
    n_obs = 20
    times = jnp.linspace(0, 10, n_obs)
    true_params = jnp.array([1.0, 0.3, 0.5, 0.2, 0.1, 0.0])
    
    # Generate noisy observations
    noise_key = random.split(key, 2)
    ra_obs = true_params[0] * jnp.cos(2*jnp.pi/true_params[0]**1.5 * times) + \
              0.01 * random.normal(noise_key[0], (n_obs,))
    dec_obs = true_params[0] * jnp.sin(2*jnp.pi/true_params[0]**1.5 * times) * 0.5 + \
               0.01 * random.normal(noise_key[1], (n_obs,))
    
    data = (times, ra_obs, dec_obs)
    
    # Least squares solver
    solver_ls = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8)
    initial_guess = jnp.array([1.1, 0.2, 0.4, 0.3, 0.2, 0.1])
    
    sol_ls = optx.least_squares(
        lambda p: orbit_residuals(p, data),
        solver_ls,
        initial_guess,
        max_steps=100
    )
    
    print(f"  Converged in {sol_ls.stats['num_steps']} steps")
    print(f"  True params: {true_params}")
    print(f"  Fitted params: {sol_ls.value}")
    
    # 3. Minimization: Maximum likelihood
    print("\n3. MAXIMUM LIKELIHOOD OPTIMIZATION:")
    
    def negative_log_likelihood(params, data):
        """Negative log likelihood for power law fit."""
        alpha, x_min = params
        x_data = data
        
        # Power law likelihood
        if alpha <= 1.0 or x_min <= 0:
            return jnp.inf
        
        norm = (alpha - 1) / x_min * (x_min / jnp.maximum(x_data, x_min)) ** alpha
        log_like = jnp.sum(jnp.log(norm))
        
        return -log_like
    
    # Generate power law distributed data (e.g., mass function)
    key, subkey = random.split(key)
    alpha_true = 2.35  # Salpeter IMF
    x_min_true = 0.1
    n_stars = 1000
    
    # Inverse transform sampling
    u = random.uniform(subkey, (n_stars,))
    masses = x_min_true * (1 - u) ** (-1/(alpha_true - 1))
    
    # Optimize
    solver_opt = optx.BFGS(rtol=1e-6, atol=1e-8)
    initial = jnp.array([2.0, 0.15])
    
    sol_opt = optx.minimise(
        lambda p: negative_log_likelihood(p, masses),
        solver_opt,
        initial,
        max_steps=100
    )
    
    print(f"  True parameters: α={alpha_true}, x_min={x_min_true}")
    print(f"  MLE estimates: α={sol_opt.value[0]:.3f}, x_min={sol_opt.value[1]:.3f}")
    
    # 4. Fixed point iteration
    print("\n4. FIXED POINT - STELLAR STRUCTURE:")
    
    def stellar_structure_iteration(y, args):
        """
        Fixed point iteration for stellar structure.
        Simplified Lane-Emden equation.
        """
        rho, T = y
        gamma = args['gamma']
        
        # Hydrostatic equilibrium + energy transport
        rho_new = T ** (1/(gamma - 1))
        T_new = rho_new ** (gamma - 1)
        
        # Add some nonlinearity
        rho_new = 0.9 * rho_new + 0.1 * rho
        T_new = 0.9 * T_new + 0.1 * T
        
        return jnp.array([rho_new, T_new])
    
    solver_fp = optx.FixedPointIteration(rtol=1e-6, atol=1e-8)
    initial_structure = jnp.array([1.0, 1.0])
    
    sol_fp = optx.fixed_point(
        lambda y: stellar_structure_iteration(y, {'gamma': 5/3}),
        solver_fp,
        initial_structure,
        max_steps=100
    )
    
    print(f"  Converged to: ρ={sol_fp.value[0]:.3f}, T={sol_fp.value[1]:.3f}")

optimistix_solvers()
```

## Lineax: Linear Algebra

### Linear System Solvers

```python
import lineax as lx

def lineax_solvers():
    """Advanced linear algebra for astronomical applications."""
    
    print("\nLINEAX: LINEAR ALGEBRA SOLVERS")
    print("=" * 50)
    
    # 1. Basic linear solve
    print("\n1. BASIC LINEAR SYSTEM:")
    
    # Poisson equation for gravitational potential
    def create_poisson_system(n):
        """Create discrete Poisson equation."""
        # -∇²φ = 4πGρ
        # Finite difference discretization
        h = 1.0 / n
        
        # Tridiagonal matrix (1D simplification)
        A = (
            -2 * jnp.eye(n) +
            jnp.eye(n, k=1) +
            jnp.eye(n, k=-1)
        ) / h**2
        
        # Random density distribution
        key = random.PRNGKey(0)
        rho = random.uniform(key, (n,))
        b = 4 * jnp.pi * rho
        
        return A, b
    
    A, b = create_poisson_system(100)
    
    # Direct solve
    solver = lx.LU()
    sol = lx.linear_solve(A, b, solver)
    
    print(f"  Solved {A.shape[0]}x{A.shape[0]} system")
    print(f"  Residual norm: {jnp.linalg.norm(A @ sol.value - b):.2e}")
    
    # 2. Iterative solvers for large systems
    print("\n2. ITERATIVE SOLVERS:")
    
    # Large sparse system (PSF deconvolution)
    def psf_convolution_operator(x, psf_kernel):
        """Apply PSF convolution."""
        # Simplified: just blur with kernel
        return jax.scipy.signal.convolve(x, psf_kernel, mode='same')
    
    # Create blurred image problem
    n_pixels = 1000
    key = random.PRNGKey(1)
    
    # PSF kernel (Gaussian)
    x_kernel = jnp.arange(-5, 6)
    psf_kernel = jnp.exp(-x_kernel**2 / 2) / jnp.sqrt(2 * jnp.pi)
    
    # True image (point sources)
    true_image = jnp.zeros(n_pixels)
    source_positions = random.choice(key, n_pixels, (10,), replace=False)
    true_image = true_image.at[source_positions].set(1.0)
    
    # Blurred observation
    blurred = psf_convolution_operator(true_image, psf_kernel)
    noise = 0.01 * random.normal(key, (n_pixels,))
    observed = blurred + noise
    
    # Define linear operator
    psf_op = lx.FunctionLinearOperator(
        lambda x: psf_convolution_operator(x, psf_kernel),
        observed.shape
    )
    
    # Conjugate gradient solver
    solver_cg = lx.CG(rtol=1e-5, atol=1e-7)
    sol_cg = lx.linear_solve(psf_op, observed, solver_cg)
    
    print(f"  Deconvolved image with CG")
    print(f"  Iterations: {sol_cg.stats['num_steps']}")
    
    # 3. Matrix decompositions
    print("\n3. MATRIX DECOMPOSITIONS:")
    
    # Covariance matrix for galaxy clustering
    def create_covariance_matrix(n_galaxies, correlation_length=0.1):
        """Create spatial covariance matrix."""
        # Positions
        key = random.PRNGKey(2)
        positions = random.uniform(key, (n_galaxies, 2))
        
        # Distance matrix
        dist = jnp.sqrt(
            ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        )
        
        # Exponential covariance
        C = jnp.exp(-dist / correlation_length)
        
        return C, positions
    
    C, positions = create_covariance_matrix(50)
    
    # Eigendecomposition for PCA
    eigenvalues, eigenvectors = jnp.linalg.eigh(C)
    
    print(f"  Covariance matrix: {C.shape}")
    print(f"  Top 5 eigenvalues: {eigenvalues[-5:]}")
    print(f"  Explained variance (top 10): {jnp.sum(eigenvalues[-10:]) / jnp.sum(eigenvalues):.1%}")
    
    # 4. Regularized solutions
    print("\n4. REGULARIZED INVERSE PROBLEMS:")
    
    def tikhonov_solve(A, b, alpha=0.01):
        """Tikhonov regularization."""
        # Solve (A^T A + alpha I) x = A^T b
        n = A.shape[1]
        A_reg = A.T @ A + alpha * jnp.eye(n)
        b_reg = A.T @ b
        
        solver = lx.LU()
        sol = lx.linear_solve(A_reg, b_reg, solver)
        
        return sol.value
    
    # Ill-conditioned problem
    A_ill = random.normal(random.PRNGKey(3), (100, 50))
    # Make it ill-conditioned
    U, S, Vt = jnp.linalg.svd(A_ill, full_matrices=False)
    S = S.at[:10].set(S[:10] * 1e-6)  # Small singular values
    A_ill = U @ jnp.diag(S) @ Vt
    
    x_true = random.normal(random.PRNGKey(4), (50,))
    b_ill = A_ill @ x_true + 0.01 * random.normal(random.PRNGKey(5), (100,))
    
    # Compare regularized vs non-regularized
    x_reg = tikhonov_solve(A_ill, b_ill, alpha=0.1)
    
    print(f"  Condition number: {jnp.linalg.cond(A_ill):.2e}")
    print(f"  Regularized solution error: {jnp.linalg.norm(x_reg - x_true) / jnp.linalg.norm(x_true):.3f}")

lineax_solvers()
```

## jaxtyping: Runtime Type Checking

### Type-Safe Scientific Computing

```python
from jaxtyping import Float, Int, Bool, Complex, PyTree, Shaped, jaxtyped
from typeguard import typechecked as typechecker

def jaxtyping_examples():
    """Type checking for safer scientific code."""
    
    print("\nJAXTYPING: TYPE-SAFE SCIENTIFIC COMPUTING")
    print("=" * 50)
    
    # 1. Basic type annotations
    print("\n1. BASIC TYPE ANNOTATIONS:")
    
    @jaxtyped(typechecker=typechecker)
    def process_spectrum(
        wavelengths: Float[Array, "n_wavelengths"],
        flux: Float[Array, "n_wavelengths"],
        errors: Optional[Float[Array, "n_wavelengths"]] = None
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Process spectrum with type checking."""
        
        # Compute signal-to-noise
        if errors is not None:
            snr = flux / errors
            median_snr = jnp.median(snr)
        else:
            median_snr = jnp.nan
        
        # Compute equivalent width
        ew = jnp.trapz(1 - flux, wavelengths)
        
        return median_snr, ew
    
    # Test with correct types
    wl = jnp.linspace(4000, 7000, 1000)
    flux = 1.0 - 0.1 * jnp.exp(-(wl - 5500)**2 / 100**2)
    errors = 0.01 * jnp.ones_like(flux)
    
    snr, ew = process_spectrum(wl, flux, errors)
    print(f"  SNR: {snr:.1f}, EW: {ew:.1f} Å")
    
    # 2. Complex shape relationships
    print("\n2. COMPLEX SHAPE ANNOTATIONS:")
    
    @jaxtyped(typechecker=typechecker)
    def n_body_step(
        positions: Float[Array, "n_bodies 3"],
        velocities: Float[Array, "n_bodies 3"],
        masses: Float[Array, "n_bodies"],
        dt: float
    ) -> tuple[Float[Array, "n_bodies 3"], Float[Array, "n_bodies 3"]]:
        """N-body integration step with shape checking."""
        
        n = positions.shape[0]
        forces = jnp.zeros_like(positions)
        
        # Compute forces
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ij = positions[j] - positions[i]
                    r3 = jnp.linalg.norm(r_ij) ** 3
                    forces = forces.at[i].add(masses[j] * r_ij / r3)
        
        # Update
        accelerations = forces
        new_velocities = velocities + accelerations * dt
        new_positions = positions + new_velocities * dt
        
        return new_positions, new_velocities
    
    # Test
    n = 3
    pos = random.normal(random.PRNGKey(0), (n, 3))
    vel = random.normal(random.PRNGKey(1), (n, 3)) * 0.1
    m = jnp.ones(n)
    
    new_pos, new_vel = n_body_step(pos, vel, m, 0.01)
    print(f"  Updated {n} bodies")
    
    # 3. PyTree annotations
    print("\n3. PYTREE TYPE ANNOTATIONS:")
    
    @jaxtyped(typechecker=typechecker)
    class GalaxyModel(eqx.Module):
        """Type-annotated galaxy model."""
        
        disk_params: dict[str, Float[Array, "..."]]
        bulge_params: dict[str, Float[Array, "..."]]
        n_stars: Int[Array, ""]
        
        def __init__(self, n_stars: int):
            self.disk_params = {
                'scale_radius': jnp.array(5.0),
                'scale_height': jnp.array(0.5),
                'mass': jnp.array(1e10)
            }
            self.bulge_params = {
                'effective_radius': jnp.array(1.0),
                'sersic_index': jnp.array(4.0),
                'mass': jnp.array(1e9)
            }
            self.n_stars = jnp.array(n_stars)
        
        @jaxtyped(typechecker=typechecker)
        def density_profile(
            self, 
            r: Float[Array, "n_points"],
            component: str = 'disk'
        ) -> Float[Array, "n_points"]:
            """Compute density profile."""
            
            if component == 'disk':
                r_d = self.disk_params['scale_radius']
                return jnp.exp(-r / r_d) / r_d
            else:
                r_e = self.bulge_params['effective_radius']
                n = self.bulge_params['sersic_index']
                return jnp.exp(-7.67 * ((r / r_e) ** (1/n) - 1))
    
    galaxy = GalaxyModel(n_stars=10000)
    radii = jnp.linspace(0.1, 20, 100)
    density = galaxy.density_profile(radii, 'disk')
    print(f"  Disk density computed at {len(radii)} points")
    
    # 4. Dimension variables
    print("\n4. DIMENSION VARIABLES:")
    
    from jaxtyping import Float32, Float64
    
    @jaxtyped(typechecker=typechecker)
    def mixed_precision_computation(
        high_precision: Float64[Array, "n m"],
        low_precision: Float32[Array, "n m"]
    ) -> Float64[Array, "n n"]:
        """Mixed precision matrix computation."""
        
        # Upcast low precision
        low_as_high = low_precision.astype(jnp.float64)
        
        # Compute in high precision
        result = high_precision @ low_as_high.T
        
        return result
    
    # Test mixed precision
    hp = jnp.array(random.normal(random.PRNGKey(6), (10, 5)), dtype=jnp.float64)
    lp = jnp.array(random.normal(random.PRNGKey(7), (10, 5)), dtype=jnp.float32)
    
    result = mixed_precision_computation(hp, lp)
    print(f"  Mixed precision result: {result.dtype}, shape {result.shape}")

jaxtyping_examples()
```

## Integration Example: Complete Scientific Workflow

### Combining All Libraries

```python
def complete_scientific_workflow():
    """
    Complete workflow combining Equinox, Diffrax, Optimistix, and Lineax.
    Example: Fitting a dynamical model to observations.
    """
    
    print("\nCOMPLETE SCIENTIFIC WORKFLOW")
    print("=" * 50)
    
    # 1. Define the dynamical model with Equinox
    class DynamicalModel(eqx.Module):
        """Neural ODE for galaxy dynamics."""
        
        potential_net: eqx.nn.MLP
        
        def __init__(self, key):
            self.potential_net = eqx.nn.MLP(
                in_size=3,  # x, y, z
                out_size=1,  # potential
                width_size=64,
                depth=3,
                activation=jax.nn.tanh,
                key=key
            )
        
        def potential(self, position: Float[Array, "3"]) -> Float[Array, ""]:
            """Gravitational potential."""
            return self.potential_net(position).squeeze()
        
        def dynamics(self, t, state, args):
            """Hamiltonian dynamics."""
            q, p = state[:3], state[3:]
            
            # Gradient of potential
            dV_dq = grad(self.potential)(q)
            
            dq_dt = p  # dq/dt = p
            dp_dt = -dV_dq  # dp/dt = -∇V
            
            return jnp.concatenate([dq_dt, dp_dt])
    
    # 2. Generate synthetic observations
    key = random.PRNGKey(42)
    key, model_key, data_key = random.split(key, 3)
    
    true_model = DynamicalModel(model_key)
    
    # Integrate orbits with Diffrax
    def integrate_orbit(model, initial_state, t_obs):
        """Integrate orbit and return positions at observation times."""
        
        term = diffrax.ODETerm(model.dynamics)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=t_obs)
        
        sol = diffrax.diffeqsolve(
            term, solver, t_obs[0], t_obs[-1], 0.01, initial_state,
            saveat=saveat
        )
        
        return sol.ys[:, :3]  # Return positions only
    
    # Generate observations
    n_orbits = 5
    n_obs_per_orbit = 20
    
    observations = []
    initial_states = []
    
    for i in range(n_orbits):
        key, subkey = random.split(key)
        
        # Random initial condition
        q0 = random.normal(subkey, (3,)) * 2
        p0 = random.normal(subkey, (3,)) * 0.5
        initial = jnp.concatenate([q0, p0])
        initial_states.append(initial)
        
        # Observation times
        t_obs = jnp.linspace(0, 10, n_obs_per_orbit)
        
        # Integrate and add noise
        true_positions = integrate_orbit(true_model, initial, t_obs)
        noise = 0.05 * random.normal(subkey, true_positions.shape)
        observed = true_positions + noise
        
        observations.append(observed)
    
    print(f"  Generated {n_orbits} orbits with {n_obs_per_orbit} observations each")
    
    # 3. Define loss function
    @eqx.filter_jit
    def loss_function(model, initial_states, observations, t_obs):
        """Loss for orbit fitting."""
        
        total_loss = 0.0
        
        for initial, observed in zip(initial_states, observations):
            predicted = integrate_orbit(model, initial, t_obs)
            residuals = predicted - observed
            total_loss += jnp.sum(residuals ** 2)
        
        return total_loss / len(observations)
    
    # 4. Optimize with Optimistix
    print("\n  Fitting model to observations...")
    
    # Initialize model to fit
    key, fit_key = random.split(key)
    fitted_model = DynamicalModel(fit_key)
    
    t_obs = jnp.linspace(0, 10, n_obs_per_orbit)
    
    # Use gradient descent
    import optax
    
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(fitted_model, eqx.is_array))
    
    # Training loop
    n_epochs = 50
    
    for epoch in range(n_epochs):
        loss, grads = eqx.filter_value_and_grad(loss_function)(
            fitted_model, initial_states, observations, t_obs
        )
        
        updates, opt_state = optimizer.update(grads, opt_state)
        fitted_model = eqx.apply_updates(fitted_model, updates)
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss = {loss:.4f}")
    
    # 5. Analyze results with Lineax
    print("\n  Analyzing fitted model...")
    
    # Compute Hessian at minimum for uncertainty estimation
    def potential_at_points(model, points):
        """Evaluate potential at multiple points."""
        return vmap(model.potential)(points)
    
    # Sample points
    test_points = random.normal(random.PRNGKey(99), (100, 3))
    
    # Compare potentials
    true_pot = potential_at_points(true_model, test_points)
    fitted_pot = potential_at_points(fitted_model, test_points)
    
    error = jnp.sqrt(jnp.mean((true_pot - fitted_pot) ** 2))
    print(f"  RMS potential error: {error:.4f}")
    
    # Compute correlation
    correlation = jnp.corrcoef(true_pot, fitted_pot)[0, 1]
    print(f"  Potential correlation: {correlation:.3f}")
    
    # Visualize one orbit
    initial_test = initial_states[0]
    t_plot = jnp.linspace(0, 20, 200)
    
    true_orbit = integrate_orbit(true_model, initial_test, t_plot)
    fitted_orbit = integrate_orbit(fitted_model, initial_test, t_plot)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True model
    axes[0].plot(true_orbit[:, 0], true_orbit[:, 1], 'b-', label='True')
    axes[0].plot(fitted_orbit[:, 0], fitted_orbit[:, 1], 'r--', label='Fitted')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Orbit Comparison')
    axes[0].legend()
    axes[0].set_aspect('equal')
    
    # Potential contours
    x = y = jnp.linspace(-3, 3, 50)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    
    points_grid = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    true_pot_grid = potential_at_points(true_model, points_grid).reshape(X.shape)
    fitted_pot_grid = potential_at_points(fitted_model, points_grid).reshape(X.shape)
    
    axes[1].contour(X, Y, true_pot_grid, levels=10, colors='blue', alpha=0.5, label='True')
    axes[1].contour(X, Y, fitted_pot_grid, levels=10, colors='red', alpha=0.5, linestyles='--', label='Fitted')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Potential Contours (z=0)')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fitted_model

# Run complete workflow
fitted_model = complete_scientific_workflow()
```

## Key Takeaways

✅ **Equinox** - Neural networks and models as PyTrees, perfect for scientific ML  
✅ **Diffrax** - State-of-the-art ODE/SDE solvers with automatic differentiation  
✅ **Optimistix** - Root finding, optimization, and fixed-point solvers  
✅ **Lineax** - Linear algebra solvers with multiple backends  
✅ **jaxtyping** - Runtime type checking for safer scientific code  
✅ **Integration** - These libraries work seamlessly together for complex workflows  

## Next Chapter Preview
Deep Learning Stack: Flax for large-scale neural networks, Optax for optimization, and Orbax for checkpointing.