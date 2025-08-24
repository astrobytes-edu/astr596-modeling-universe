# Physics-Informed Neural Networks for Astronomers: Embedding Physical Laws into Deep Learning

## Prerequisites Check

Before diving into PINNs, ensure you're comfortable with these concepts from our previous tutorials:

### Required Knowledge
- **Neural Networks**: Forward/backward propagation, loss functions, optimization
- **Automatic Differentiation**: How gradients are computed in JAX
- **PDEs in Physics**: Basic differential equations (Poisson, wave, diffusion)
- **Linear Algebra**: Matrix operations, eigenvalues, norms

### Notation Convention
Throughout this document:
- $u(\mathbf{x}, t)$ denotes the solution to our PDE
- $u_{NN}(\mathbf{x}, t; \theta)$ denotes the neural network approximation with parameters $\theta$
- $\nabla = [\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z}]^T$ is the gradient operator
- $\nabla^2 = \Delta = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$ is the Laplacian
- $\mathcal{L}$ denotes loss functions
- $\Omega$ denotes the spatial domain
- $\partial\Omega$ denotes the boundary of the domain

---

## Introduction: Why Physics-Informed Neural Networks?

Imagine you're modeling the gravitational potential in a galaxy. You know it must satisfy Poisson's equation:
$$\nabla^2 \Phi = 4\pi G \rho$$

Traditional numerical methods would:
- Discretize space into a grid
- Solve a large linear system
- Suffer from curse of dimensionality
- Struggle with complex boundaries

But what if we could use a neural network that **automatically satisfies the physics**? This is the promise of Physics-Informed Neural Networks (PINNs).

### The Revolutionary Idea

Instead of using neural networks as black boxes that learn from data alone, PINNs embed physical laws directly into the learning process. The network learns solutions that:
1. **Satisfy governing PDEs** throughout the domain
2. **Match boundary conditions** at domain edges
3. **Fit any available data** from observations
4. **Respect conservation laws** and symmetries

For astronomy, this means:
- Solving for gravitational potentials without grids
- Learning velocity fields that conserve mass
- Discovering missing physics from incomplete observations
- Handling irregular geometries (spiral arms, bars, streams)

### What Makes PINNs Special?

Traditional numerical methods discretize equations. PINNs discretize the solution:

**Finite Difference**: $\Phi_{i,j,k}$ on a grid → $\nabla^2$ via stencils

**PINN**: $\Phi_{NN}(\mathbf{x}; \theta)$ continuous → $\nabla^2$ via automatic differentiation

This continuous representation has profound advantages:
- Derivatives at any order for free
- No grid means no numerical dispersion
- Natural handling of high dimensions
- Mesh-free solution in complex domains

Let's build this framework from the ground up!

---

## Part 1: The Mathematical Foundation of PINNs

### From PDEs to Optimization Problems

Consider a general PDE:
$$\mathcal{N}[u(\mathbf{x}, t)] = f(\mathbf{x}, t) \quad \text{in } \Omega \times [0, T]$$

where $\mathcal{N}$ is a differential operator (like $\nabla^2$ or $\frac{\partial}{\partial t} - \alpha\nabla^2$).

With boundary conditions:
$$\mathcal{B}[u(\mathbf{x}, t)] = g(\mathbf{x}, t) \quad \text{on } \partial\Omega \times [0, T]$$

And initial conditions:
$$u(\mathbf{x}, 0) = u_0(\mathbf{x}) \quad \text{in } \Omega$$

The PINN approach transforms this into an optimization problem:

**Find neural network parameters $\theta$ that minimize**:
$$\mathcal{L}(\theta) = \mathcal{L}_{PDE}(\theta) + \mathcal{L}_{BC}(\theta) + \mathcal{L}_{IC}(\theta) + \mathcal{L}_{data}(\theta)$$

### The Loss Components

**PDE Loss** (physics in the interior):
$$\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} |\mathcal{N}[u_{NN}(\mathbf{x}_i, t_i; \theta)] - f(\mathbf{x}_i, t_i)|^2$$

**Boundary Condition Loss**:
$$\mathcal{L}_{BC} = \frac{1}{N_{BC}} \sum_{j=1}^{N_{BC}} |\mathcal{B}[u_{NN}(\mathbf{x}_j, t_j; \theta)] - g(\mathbf{x}_j, t_j)|^2$$

**Initial Condition Loss**:
$$\mathcal{L}_{IC} = \frac{1}{N_{IC}} \sum_{k=1}^{N_{IC}} |u_{NN}(\mathbf{x}_k, 0; \theta) - u_0(\mathbf{x}_k)|^2$$

**Data Loss** (if we have observations):
$$\mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{l=1}^{N_{data}} |u_{NN}(\mathbf{x}_l, t_l; \theta) - u_{obs}(\mathbf{x}_l, t_l)|^2$$

### 🌟 Astronomical Example: Stellar Density in a Galaxy

Consider the Poisson equation for gravitational potential:
$$\nabla^2 \Phi = 4\pi G \rho(\mathbf{r})$$

For a galaxy with density:
$$\rho(r, z) = \rho_0 \exp(-r/r_d) \operatorname{sech}^2(z/z_d)$$

The PINN learns $\Phi_{NN}(\mathbf{r}; \theta)$ by minimizing:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N |\nabla^2 \Phi_{NN}(\mathbf{r}_i) - 4\pi G \rho(\mathbf{r}_i)|^2 + \frac{\lambda}{M} \sum_{j=1}^M |\Phi_{NN}(\mathbf{r}_j^{far})|^2$$

The second term ensures $\Phi \to 0$ as $r \to \infty$.

📝 **Checkpoint 1**: Why do we sample random points rather than using a fixed grid for the PDE loss?  
*Answer: Random sampling avoids the curse of dimensionality in high dimensions and naturally adapts to the solution's complexity*

---

## Part 2: Computing Derivatives with Automatic Differentiation

### The Power of Autodiff for PDEs

The key to PINNs is computing derivatives of the neural network efficiently. JAX makes this trivial!

For a function $u_{NN}(\mathbf{x}; \theta)$, we need:
- First derivatives: $\frac{\partial u}{\partial x_i}$
- Second derivatives: $\frac{\partial^2 u}{\partial x_i \partial x_j}$
- Higher-order derivatives for complex PDEs

### JAX Implementation of Differential Operators

```python
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit

def gradient(f, x):
    """Compute gradient ∇f at point x"""
    return grad(f)(x)

def laplacian(f, x):
    """Compute Laplacian ∇²f at point x"""
    # Method 1: Sum of second derivatives
    def hessian_diagonal(f, x):
        n = len(x)
        laplacian_val = 0
        for i in range(n):
            # Second derivative with respect to x_i
            grad_i = grad(f, argnums=0)
            second_deriv = grad(lambda x: grad_i(x)[i])(x)[i]
            laplacian_val += second_deriv
        return laplacian_val
    
    return hessian_diagonal(f, x)

# More efficient vectorized version
def laplacian_vectorized(f):
    """Return function that computes Laplacian"""
    def compute_laplacian(x):
        # Compute full Hessian
        hessian = jax.hessian(f)(x)
        # Laplacian is trace of Hessian
        return jnp.trace(hessian)
    return compute_laplacian
```

### 🌟 Example: Heat Equation in a Star

The temperature distribution in a star follows:
$$\frac{\partial T}{\partial t} = \kappa \nabla^2 T + S(\mathbf{r})$$

where $S(\mathbf{r})$ is the energy generation rate.

```python
def heat_equation_residual(network_params, x, t):
    """Compute PDE residual for heat equation"""
    
    # Define function for specific (x,t)
    def T(x, t):
        inputs = jnp.concatenate([x, jnp.array([t])])
        return neural_network(network_params, inputs)[0]
    
    # Compute derivatives using JAX
    dT_dt = grad(T, argnums=1)(x, t)
    laplacian_T = laplacian(lambda x: T(x, t), x)
    
    # Energy source term
    S = energy_generation(x)
    
    # PDE residual (should be zero)
    residual = dT_dt - kappa * laplacian_T - S
    
    return residual
```

### Computing Mixed Derivatives

For complex PDEs, we need mixed derivatives:

```python
def mixed_derivative(f, x, indices):
    """Compute mixed partial derivative ∂²f/∂x_i∂x_j"""
    i, j = indices
    
    # First derivative with respect to x_i
    df_dxi = grad(f, argnums=0)
    
    # Second derivative with respect to x_j
    d2f_dxidxj = grad(lambda x: df_dxi(x)[i])(x)[j]
    
    return d2f_dxidxj
```

📝 **Checkpoint 2**: For a 3D problem, how many second derivatives does the Hessian contain? How many are unique if the function is smooth?  
*Answer: The Hessian is 3×3 = 9 elements. Due to symmetry (∂²f/∂x∂y = ∂²f/∂y∂x), only 6 are unique*

---

## Part 3: Building Your First PINN

### Complete Example: 1D Poisson Equation

Let's solve a simple but complete problem:
$$\frac{d^2u}{dx^2} = -\sin(\pi x) \quad \text{for } x \in [0, 1]$$
$$u(0) = u(1) = 0$$

The analytical solution is $u(x) = \frac{1}{\pi^2}\sin(\pi x)$.

### Step 1: Define the Neural Network

```python
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax

def init_network(key, layers):
    """Initialize neural network parameters"""
    params = []
    keys = random.split(key, len(layers) - 1)
    
    for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
        W = random.normal(keys[i], (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append({'W': W, 'b': b})
    
    return params

@jit
def forward(params, x):
    """Forward pass through network"""
    # Ensure input is 2D
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Hidden layers with tanh activation
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer['W'] + layer['b'])
    
    # Output layer (no activation)
    x = x @ params[-1]['W'] + params[-1]['b']
    
    return x[0, 0] if x.shape[0] == 1 else x[:, 0]
```

### Step 2: Define the Physics Loss

```python
@jit
def pde_loss(params, x_pde):
    """Compute PDE residual loss"""
    
    def u(x):
        return forward(params, x)
    
    def pde_residual(x):
        # Compute d²u/dx²
        d2u_dx2 = grad(grad(u))(x)
        
        # Source term
        f = -jnp.sin(jnp.pi * x)
        
        # Residual: d²u/dx² - f = 0
        return d2u_dx2 - f
    
    # Vectorize over all collocation points
    residuals = vmap(pde_residual)(x_pde)
    
    return jnp.mean(residuals**2)

@jit
def boundary_loss(params, x_bc, u_bc):
    """Compute boundary condition loss"""
    u_pred = vmap(lambda x: forward(params, x))(x_bc)
    return jnp.mean((u_pred - u_bc)**2)

@jit
def total_loss(params, x_pde, x_bc, u_bc):
    """Combined loss function"""
    loss_pde = pde_loss(params, x_pde)
    loss_bc = boundary_loss(params, x_bc, u_bc)
    
    # Weight the losses
    return loss_pde + 100 * loss_bc
```

### Step 3: Training Loop

```python
def train_pinn(key, n_epochs=5000):
    """Train the PINN"""
    
    # Initialize network
    layers = [1, 20, 20, 20, 1]  # 1 input, 3 hidden layers, 1 output
    params = init_network(key, layers)
    
    # Training data
    key, subkey = random.split(key)
    x_pde = random.uniform(subkey, (100,), minval=0.0, maxval=1.0)
    x_bc = jnp.array([0.0, 1.0])
    u_bc = jnp.array([0.0, 0.0])
    
    # Optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # Training loop
    loss_history = []
    
    for epoch in range(n_epochs):
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(total_loss)(
            params, x_pde, x_bc, u_bc
        )
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        loss_history.append(loss_val)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
            
            # Resample collocation points
            key, subkey = random.split(key)
            x_pde = random.uniform(subkey, (100,), minval=0.0, maxval=1.0)
    
    return params, loss_history
```

### Step 4: Validation

```python
def validate_solution(params):
    """Compare PINN solution with analytical solution"""
    
    x_test = jnp.linspace(0, 1, 100)
    
    # PINN solution
    u_pinn = vmap(lambda x: forward(params, x))(x_test)
    
    # Analytical solution
    u_exact = jnp.sin(jnp.pi * x_test) / (jnp.pi**2)
    
    # Compute error
    l2_error = jnp.sqrt(jnp.mean((u_pinn - u_exact)**2))
    rel_error = l2_error / jnp.sqrt(jnp.mean(u_exact**2))
    
    print(f"L2 Error: {l2_error:.6f}")
    print(f"Relative Error: {rel_error:.2%}")
    
    return x_test, u_pinn, u_exact
```

📝 **Checkpoint 3**: Why do we weight the boundary loss more heavily (×100) than the PDE loss?  
*Answer: Boundary conditions are enforced at only 2 points while PDE is enforced at 100 points. The weight balances their influence*

---

## Part 4: Advanced PINN Techniques

### Hard Constraints vs Soft Constraints

**Soft Constraints** (what we've used so far):
- Boundary conditions enforced via loss function
- Simple to implement
- May not exactly satisfy BCs

**Hard Constraints** (exact satisfaction):
Build BCs into the network architecture:

```python
def hard_constraint_network(params, x):
    """Network that exactly satisfies u(0) = u(1) = 0"""
    # Distance from boundaries
    distance_function = x * (1 - x)
    
    # Neural network output
    nn_output = forward(params, x)
    
    # Solution that vanishes at boundaries
    u = distance_function * nn_output
    
    return u
```

This guarantees $u(0) = u(1) = 0$ regardless of network parameters!

### Adaptive Sampling Strategies

Sample more points where the solution varies rapidly:

```python
def adaptive_sampling(params, x_current, n_new_points):
    """Add points where PDE residual is large"""
    
    # Compute residuals at current points
    residuals = vmap(lambda x: abs(pde_residual(params, x)))(x_current)
    
    # Probability proportional to residual
    probabilities = residuals / jnp.sum(residuals)
    
    # Sample new points near high-residual regions
    key = random.PRNGKey(0)
    indices = random.choice(key, len(x_current), (n_new_points,), p=probabilities)
    
    # Add noise to create new points
    noise = random.normal(key, (n_new_points,)) * 0.01
    x_new = x_current[indices] + noise
    x_new = jnp.clip(x_new, 0, 1)  # Keep in domain
    
    return jnp.concatenate([x_current, x_new])
```

### Loss Weighting Strategies

Different terms in the loss may have different scales:

```python
class AdaptiveWeights:
    """Adaptive weighting using gradient statistics"""
    
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.lambda_pde = 1.0
        self.lambda_bc = 1.0
        
    def update(self, grad_pde, grad_bc):
        """Update weights based on gradient magnitudes"""
        # Exponential moving average of gradient norms
        avg_grad_pde = self.alpha * jnp.linalg.norm(grad_pde)
        avg_grad_bc = self.alpha * jnp.linalg.norm(grad_bc)
        
        # Balance gradients
        self.lambda_pde = avg_grad_bc / (avg_grad_pde + 1e-8)
        self.lambda_bc = avg_grad_pde / (avg_grad_bc + 1e-8)
        
        # Normalize
        total = self.lambda_pde + self.lambda_bc
        self.lambda_pde /= total
        self.lambda_bc /= total
```

---

## Part 5: PINNs for Time-Dependent Problems

### Example: Wave Equation for Stellar Oscillations

Radial pulsations in a star follow:
$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

with initial conditions:
$$u(\mathbf{x}, 0) = u_0(\mathbf{x}), \quad \frac{\partial u}{\partial t}(\mathbf{x}, 0) = v_0(\mathbf{x})$$

### Network Architecture for Space-Time

```python
def spacetime_network(params, x, t):
    """Neural network for space-time problems"""
    # Concatenate spatial and temporal coordinates
    inputs = jnp.concatenate([x, jnp.array([t])])
    
    # Forward pass
    return forward(params, inputs)

def wave_equation_loss(params, x_pde, t_pde):
    """Loss for wave equation"""
    
    def u(x, t):
        return spacetime_network(params, x, t)
    
    def wave_residual(x, t):
        # Second time derivative
        d2u_dt2 = grad(grad(u, argnums=1), argnums=1)(x, t)
        
        # Laplacian in space
        laplacian_u = laplacian(lambda x: u(x, t), x)
        
        # Wave equation residual
        return d2u_dt2 - c**2 * laplacian_u
    
    # Vectorize over all space-time points
    residuals = vmap(wave_residual)(x_pde, t_pde)
    
    return jnp.mean(residuals**2)
```

### Causal Training for Time-Dependent PDEs

Train the network respecting causality:

```python
def causal_training(params, time_windows):
    """Train PINN causally through time"""
    
    for t_start, t_end in time_windows:
        # Generate collocation points in current window
        x_pde, t_pde = generate_points(t_start, t_end)
        
        # Use solution from previous window as "data"
        if t_start > 0:
            x_interface = generate_interface_points()
            u_interface = evaluate_network(params, x_interface, t_start)
            
            # Add interface loss
            loss = pde_loss + interface_loss
        else:
            loss = pde_loss + initial_condition_loss
        
        # Optimize in current window
        params = optimize(params, loss)
    
    return params
```

📝 **Checkpoint 4**: Why might causal training work better than training on the entire time domain at once?  
*Answer: It respects the physics of information propagation and prevents the network from "using" future information to predict the past*

---

## Part 6: PINNs for Inverse Problems

### Learning Unknown Parameters

Suppose we observe a stellar oscillation but don't know the sound speed $c$:

```python
def inverse_problem_network(params_nn, params_physics):
    """Learn both solution and unknown physics parameters"""
    
    # params_physics contains unknown c
    c = params_physics['sound_speed']
    
    def loss(x_pde, t_pde, x_data, t_data, u_data):
        # PDE loss with unknown parameter
        def wave_residual(x, t):
            u = lambda x, t: forward(params_nn, jnp.concatenate([x, [t]]))
            d2u_dt2 = grad(grad(u, 1), 1)(x, t)
            laplacian_u = laplacian(lambda x: u(x, t), x)
            return d2u_dt2 - c**2 * laplacian_u
        
        loss_pde = jnp.mean(vmap(wave_residual)(x_pde, t_pde)**2)
        
        # Data loss
        u_pred = vmap(lambda x, t: forward(params_nn, jnp.concatenate([x, [t]])))(x_data, t_data)
        loss_data = jnp.mean((u_pred - u_data)**2)
        
        return loss_pde + loss_data
    
    return loss
```

### Discovery of Hidden Physics

Learn missing terms in the equation:

```python
def physics_discovery_network(params_solution, params_hidden):
    """Discover hidden physics terms"""
    
    # Known physics: ∂u/∂t = ∂²u/∂x²
    # Hidden physics: + f(u, x, t)
    
    def hidden_term(u, x, t):
        """Neural network for unknown physics"""
        inputs = jnp.array([u, x, t])
        return forward(params_hidden, inputs)
    
    def full_equation_residual(x, t):
        u = lambda x, t: forward(params_solution, jnp.array([x, t]))
        
        # Known terms
        du_dt = grad(u, argnums=1)(x, t)
        d2u_dx2 = grad(grad(u, argnums=0), argnums=0)(x, t)
        
        # Unknown term
        u_val = u(x, t)
        f_hidden = hidden_term(u_val, x, t)
        
        # Complete equation
        return du_dt - d2u_dx2 - f_hidden
    
    return full_equation_residual
```

---

## Part 7: Applications to Stellar Dynamics

### Application 1: Gravitational Potential in Irregular Galaxies

For galaxies with complex geometry (bars, spirals, mergers):

```python
def galactic_potential_pinn(stellar_positions, stellar_masses):
    """Learn potential from discrete stellar distribution"""
    
    def potential_network(params, r):
        """Neural network for gravitational potential"""
        # Use logarithmic scaling for large dynamic range
        r_scaled = jnp.log(1 + jnp.linalg.norm(r))
        inputs = jnp.concatenate([r, jnp.array([r_scaled])])
        return forward(params, inputs)
    
    def physics_loss(params, r_collocation):
        """Poisson equation loss"""
        
        def laplacian_potential(r):
            phi = lambda r: potential_network(params, r)
            return laplacian(phi, r)
        
        # Compute density at collocation points
        density = compute_density(r_collocation, stellar_positions, stellar_masses)
        
        # Poisson equation
        laplacians = vmap(laplacian_potential)(r_collocation)
        
        return jnp.mean((laplacians - 4*jnp.pi*G*density)**2)
    
    def force_accuracy_loss(params, test_positions):
        """Validate forces are accurate"""
        
        def force_from_potential(r):
            phi = lambda r: potential_network(params, r)
            return -grad(phi)(r)
        
        # Compare with direct N-body
        forces_pinn = vmap(force_from_potential)(test_positions)
        forces_nbody = compute_nbody_forces(test_positions)
        
        return jnp.mean(jnp.linalg.norm(forces_pinn - forces_nbody, axis=1)**2)
    
    return physics_loss, force_accuracy_loss
```

### Application 2: Jeans Equation with Neural Velocity Dispersion

For spherical stellar systems:

```python
def jeans_equation_pinn(density_profile, potential_profile):
    """Solve for velocity dispersion in stellar system"""
    
    def dispersion_network(params, r):
        """σ_r(r) must be positive"""
        output = forward(params, r)
        return jnp.exp(output)  # Ensure positivity
    
    def anisotropy_network(params, r):
        """β(r) must be in [-∞, 1]"""
        output = forward(params, r)
        return jnp.tanh(output)  # Bound to [-1, 1]
    
    def jeans_residual(params_sigma, params_beta, r):
        """Jeans equation residual"""
        
        # Get functions
        sigma_r = lambda r: dispersion_network(params_sigma, r)
        beta = lambda r: anisotropy_network(params_beta, r)
        rho = density_profile
        dphi_dr = grad(potential_profile)
        
        # Compute derivatives
        def lhs(r):
            # d(ρσ_r²)/dr
            rho_sigma2 = rho(r) * sigma_r(r)**2
            d_rho_sigma2 = grad(lambda r: rho(r) * sigma_r(r)**2)(r)
            
            # 2β(r)ρσ_r²/r
            beta_term = 2 * beta(r) * rho_sigma2 / r
            
            return d_rho_sigma2 + beta_term
        
        # Right hand side: -ρ dΦ/dr
        rhs = -rho(r) * dphi_dr(r)
        
        return lhs(r) - rhs
    
    return jeans_residual
```

### Application 3: Collisionless Boltzmann Equation

For the full 6D phase space evolution:

```python
def boltzmann_pinn(initial_distribution):
    """Solve collisionless Boltzmann equation"""
    
    def distribution_network(params, x, v, t):
        """f(x, v, t) must be non-negative"""
        inputs = jnp.concatenate([x, v, jnp.array([t])])
        output = forward(params, inputs)
        return jnp.nn.softplus(output)  # Ensure positivity
    
    def boltzmann_residual(params, x, v, t):
        """CBE: ∂f/∂t + v·∇_x f - ∇_x Φ·∇_v f = 0"""
        
        f = lambda x, v, t: distribution_network(params, x, v, t)
        
        # Time derivative
        df_dt = grad(f, argnums=2)(x, v, t)
        
        # Spatial gradient
        grad_x_f = grad(f, argnums=0)(x, v, t)
        
        # Velocity gradient
        grad_v_f = grad(f, argnums=1)(x, v, t)
        
        # Force from potential
        force = -grad(gravitational_potential)(x)
        
        # CBE residual
        residual = df_dt + jnp.dot(v, grad_x_f) - jnp.dot(force, grad_v_f)
        
        return residual
    
    def conservation_loss(params, sample_points):
        """Ensure conservation of mass and energy"""
        
        # Sample distribution
        f_samples = vmap(lambda p: distribution_network(params, *p))(sample_points)
        
        # Mass conservation
        mass = jnp.sum(f_samples) * phase_space_volume / len(sample_points)
        mass_error = (mass - total_mass)**2
        
        # Energy conservation
        energies = vmap(compute_energy)(sample_points)
        energy = jnp.sum(f_samples * energies) * phase_space_volume / len(sample_points)
        energy_error = (energy - total_energy)**2
        
        return mass_error + energy_error
    
    return boltzmann_residual, conservation_loss
```

📝 **Checkpoint 5**: In the Boltzmann equation PINN, why do we use softplus activation for the distribution function?  
*Answer: The distribution function f(x,v,t) represents particle density in phase space and must be non-negative. Softplus ensures f ≥ 0 everywhere*

---

## Part 8: Practical Implementation Guide

### Architecture Design Principles

**Network Depth and Width**:
- Smooth solutions: Shallow and wide (2-3 layers, 50-100 neurons)
- Sharp features: Deeper networks (4-6 layers, 20-50 neurons)
- High dimensions: More neurons per layer

**Activation Functions**:
- **Tanh**: Smooth problems, bounded outputs
- **Sin**: Periodic solutions, wave equations
- **ReLU**: Avoid for PINNs (derivatives vanish)
- **Swish/GELU**: Good compromise

```python
def multi_scale_network(params, x, frequencies=[1, 2, 4, 8]):
    """Network with multiple frequency scales"""
    
    # Fourier features for different scales
    features = []
    for freq in frequencies:
        features.append(jnp.sin(freq * jnp.pi * x))
        features.append(jnp.cos(freq * jnp.pi * x))
    
    inputs = jnp.concatenate([x] + features)
    
    return forward(params, inputs)
```

### Training Strategies

**Progressive Training**:
```python
def progressive_training(layers_schedule):
    """Gradually increase network capacity"""
    
    params = None
    
    for n_layers in layers_schedule:
        # Initialize new network
        new_params = init_network(key, n_layers)
        
        # Transfer weights from previous network
        if params is not None:
            new_params = transfer_weights(params, new_params)
        
        # Train with current architecture
        params = train(new_params, n_epochs=1000)
    
    return params
```

**Curriculum Learning**:
```python
def curriculum_learning(difficulty_schedule):
    """Start with easy problems, increase complexity"""
    
    for difficulty in difficulty_schedule:
        if difficulty == 'easy':
            # Large domain, smooth solution
            domain = [0, 10]
            n_collocation = 50
        elif difficulty == 'medium':
            # Add boundary layers
            domain = [0, 10]
            n_collocation = 200
        else:  # hard
            # Sharp features, complex geometry
            domain = [0, 100]
            n_collocation = 1000
        
        params = train(params, domain, n_collocation)
    
    return params
```

### Common Pitfalls and Solutions

**Problem 1**: Vanishing gradients in deep networks
```python
# Solution: Residual connections
def residual_network(params, x):
    h = x
    for layer in params:
        h_new = activation(h @ layer['W'] + layer['b'])
        h = h + 0.1 * h_new  # Residual connection
    return h
```

**Problem 2**: Imbalanced loss terms
```python
# Solution: Gradient normalization
def normalized_loss(params, losses):
    grads = [grad(loss)(params) for loss in losses]
    grad_norms = [jnp.linalg.norm(g) for g in grads]
    weights = [1.0 / (norm + 1e-8) for norm in grad_norms]
    
    total_loss = sum(w * loss for w, loss in zip(weights, losses))
    return total_loss
```

**Problem 3**: Poor boundary condition satisfaction
```python
# Solution: Increase BC sampling and use hard constraints
def improved_bc_handling(params, x):
    # More BC points
    x_bc = jnp.linspace(0, 1, 100)
    
    # Hard constraint
    distance = x * (1 - x)
    u = distance * neural_network(params, x)
    
    return u
```

### Validation and Error Analysis

```python
def validate_pinn_solution(params, analytical_solution=None):
    """Comprehensive validation of PINN solution"""
    
    # 1. Check PDE residual
    x_test = generate_test_points(1000)
    residuals = vmap(pde_residual)(params, x_test)
    print(f"Mean PDE residual: {jnp.mean(jnp.abs(residuals)):.6e}")
    
    # 2. Check boundary conditions
    bc_error = boundary_loss(params)
    print(f"BC error: {bc_error:.6e}")
    
    # 3. Conservation laws
    if check_conservation:
        mass = compute_mass(params)
        energy = compute_energy(params)
        print(f"Mass conservation error: {abs(mass - initial_mass):.6e}")
        print(f"Energy conservation error: {abs(energy - initial_energy):.6e}")
    
    # 4. Compare with analytical solution
    if analytical_solution:
        u_pinn = vmap(lambda x: forward(params, x))(x_test)
        u_exact = vmap(analytical_solution)(x_test)
        
        l2_error = jnp.sqrt(jnp.mean((u_pinn - u_exact)**2))
        l_inf_error = jnp.max(jnp.abs(u_pinn - u_exact))
        
        print(f"L2 error: {l2_error:.6e}")
        print(f"L∞ error: {l_inf_error:.6e}")
    
    return validation_metrics
```

---

## Quick Reference Summary

### Key Concepts

| Concept | Description | Formula |
|---------|------------|---------|
| PDE Loss | Enforce physics in domain | $\|\mathcal{N}[u_{NN}] - f\|^2$ |
| BC Loss | Enforce boundary conditions | $\|\mathcal{B}[u_{NN}] - g\|^2$ |
| Data Loss | Fit observations | $\|u_{NN} - u_{obs}\|^2$ |
| Hard Constraints | Exact BC satisfaction | $u = G(x) \cdot NN(x)$ |
| Soft Constraints | Approximate via loss | Add to $\mathcal{L}$ |

### Differential Operators in JAX

| Operator | JAX Implementation |
|----------|-------------------|
| Gradient $\nabla u$ | `grad(u)(x)` |
| Laplacian $\nabla^2 u$ | `trace(hessian(u)(x))` |
| Divergence $\nabla \cdot \mathbf{F}$ | `sum([grad(F[i], i)(x) for i in range(d)])` |
| Curl $\nabla \times \mathbf{F}$ | Use `jacfwd` for Jacobian |

### Best Practices

| Aspect | Recommendation |
|--------|---------------|
| Architecture | 3-5 layers, 20-50 neurons, tanh/sin activation |
| Sampling | Random uniform, adaptive refinement |
| Loss Weights | Start equal, adapt based on gradients |
| Training | Adam optimizer, lr=1e-3, reduce on plateau |
| Validation | Check residuals, conservation, boundary conditions |

### Common PDEs in Astronomy

| PDE | Application | PINN Advantage |
|-----|-------------|----------------|
| Poisson | Gravitational potential | Irregular boundaries |
| Boltzmann | Phase space evolution | High dimensions |
| Jeans | Stellar dynamics | Inverse problems |
| Wave | Stellar oscillations | Multi-scale |
| Diffusion | Energy transport | Complex geometry |

---

## Conclusion: The Future of Scientific Computing

You now have the tools to:
- Encode physical laws directly into neural networks
- Solve PDEs without meshes or grids
- Handle high-dimensional problems
- Discover hidden physics from data
- Solve inverse problems naturally

PINNs represent a paradigm shift in computational physics. Instead of discretizing equations and solving linear systems, we:
1. Parameterize solutions with neural networks
2. Enforce physics through automatic differentiation
3. Optimize to find solutions

For your N-body projects, PINNs enable:
- **Continuous potentials** from discrete particles
- **Solving Boltzmann equation** without particles
- **Learning corrections** to approximate methods
- **Discovering dynamics** from incomplete observations

The combination of physics knowledge and neural network flexibility opens new frontiers. As you implement PINNs in JAX, remember:
- Physics constraints prevent overfitting
- Automatic differentiation makes implementation elegant
- The continuous representation enables new insights

The future of computational astrophysics lies at the intersection of physics and machine learning. PINNs are your gateway to this frontier!

---

## Final Challenges

📝 **Challenge 1**: Design a PINN to solve the gravitational potential for a barred galaxy. What coordinate system would you use? How would you handle the bar's rotation?

📝 **Challenge 2**: You observe a stellar stream with gaps. Design a PINN that learns both the background potential and the perturbations causing gaps. What physics would you encode?

📝 **Challenge 3**: Implement a PINN for the restricted three-body problem. How would you handle the rotating reference frame and ensure the Jacobi integral is conserved?

*These challenges integrate everything: PDEs, conservation laws, coordinate transforms, and astronomical applications. Ready to revolutionize computational astrophysics!*