# JAX N-Body Neural Network Projects: Complete Mathematical and Theoretical Guide

## Part I: Theoretical Foundations

### 1.1 The N-Body Problem: Mathematical Formulation

#### Classical Formulation
The gravitational N-body problem describes the motion of N point masses under mutual gravitational attraction:

$$\frac{d^2\mathbf{r}_i}{dt^2} = -G \sum_{j=1, j \neq i}^{N} m_j \frac{\mathbf{r}_i - \mathbf{r}_j}{|\mathbf{r}_i - \mathbf{r}_j|^3}$$

where:
- $\mathbf{r}_i$ = position vector of particle $i$
- $m_i$ = mass of particle $i$  
- $G$ = gravitational constant
- $N$ = total number of particles

#### Phase Space Representation
The complete state of the system at time $t$ is given by:
$$\mathbf{\Gamma}(t) = \{\mathbf{r}_1(t), ..., \mathbf{r}_N(t), \mathbf{v}_1(t), ..., \mathbf{v}_N(t)\}$$

This 6N-dimensional phase space evolves according to Hamilton's equations:
$$\frac{d\mathbf{r}_i}{dt} = \frac{\partial H}{\partial \mathbf{p}_i}, \quad \frac{d\mathbf{p}_i}{dt} = -\frac{\partial H}{\partial \mathbf{r}_i}$$

with **Hamiltonian**:
$$H = \sum_{i=1}^{N} \frac{|\mathbf{p}_i|^2}{2m_i} - G\sum_{i=1}^{N-1}\sum_{j=i+1}^{N} \frac{m_i m_j}{|\mathbf{r}_i - \mathbf{r}_j|}$$

```{margin}
**Hamiltonian**

```

### 1.2 Computational Complexity Analysis

#### Direct Summation Complexity
- Force calculation: $O(N^2)$ operations per timestep
- Memory requirement: $O(N)$ for positions and velocities
- Total simulation cost: $O(N^2 \cdot T/\Delta t)$ where $T$ is total time, $\Delta t$ is timestep

#### Approximation Methods Overview
1. **Tree codes (Barnes-Hut)**: $O(N \log N)$ complexity
2. **Fast Multipole Method**: $O(N)$ complexity
3. **Particle-Mesh methods**: $O(N \log N)$ via FFT
4. **Neural Network approximations**: $O(N \cdot k)$ where $k$ is neighbor count (this project)

### 1.3 Statistical Mechanics of Self-Gravitating Systems

#### Virial Theorem
For a system in equilibrium:
$$2K + U = 0$$

where $K$ = total kinetic energy, $U$ = total potential energy.

Derivation from time-averaging the virial:
$$\frac{d}{dt}\sum_i \mathbf{r}_i \cdot \mathbf{p}_i = 2K + U$$

At equilibrium, the time average of the left side vanishes.

#### Relaxation Processes

**Two-body relaxation time** (Spitzer-Chandrasekhar):
$$t_{relax} = \frac{N}{8\ln(N)} \cdot \frac{v^3}{G^2 m \rho}$$

where:
- $v$ = typical velocity (velocity dispersion)
- $\rho$ = typical density
- $m$ = typical mass

**Derivation outline:**
1. Start with Fokker-Planck equation for velocity diffusion
2. Calculate diffusion coefficient from two-body encounters
3. Integrate over impact parameters with Coulomb logarithm $\ln(N)$
4. Determine time for velocity distribution to change significantly

---

## Part II: Project Implementations with Mathematical Framework

## Project 1: Neural Network Density Estimator

### Mathematical Foundation

#### 1.1 Density Field Definition
The density field from discrete particles:
$$\rho(\mathbf{r}) = \sum_{i=1}^{N} m_i \delta^3(\mathbf{r} - \mathbf{r}_i)$$

We seek a smooth approximation:
$$\tilde{\rho}(\mathbf{r}) = \sum_{i=1}^{N} m_i W(\mathbf{r} - \mathbf{r}_i, h)$$

where $W$ is a smoothing kernel with characteristic width $h$.

#### 1.2 Kernel Density Estimation Theory

**Standard kernels:**
- Gaussian: $W(r,h) = \frac{1}{(2\pi h^2)^{3/2}} \exp\left(-\frac{r^2}{2h^2}\right)$
- Cubic spline: $W(r,h) = \frac{8}{\pi h^3} \begin{cases} 1-6(r/h)^2+6(r/h)^3 & 0 \leq r/h \leq 0.5 \\ 2(1-r/h)^3 & 0.5 < r/h \leq 1 \\ 0 & r/h > 1 \end{cases}$

#### 1.3 Neural Network Formulation

**Universal Approximation Theorem**: A feedforward network with one hidden layer can approximate any continuous function on compact sets to arbitrary accuracy.

**Network architecture:**
$$\tilde{\rho}_{NN}(\mathbf{r}; \theta) = \exp\left(f_\theta(\mathbf{r})\right)$$

where $f_\theta: \mathbb{R}^3 \rightarrow \mathbb{R}$ is the neural network with parameters $\theta$.

**Why logarithmic output?**
- Ensures positive density: $\rho > 0$ always
- Handles large dynamic range: cluster cores can be $10^6$ times denser than halos
- Stabilizes gradient flow: $\nabla_\theta \log \rho$ has better conditioning

### Implementation Details

#### Training Data Generation
```python
def generate_density_training_data(positions, masses, n_samples=10000):
    """
    Mathematical implementation:
    1. Sample points uniformly in cluster volume
    2. Compute true density via kernel estimation
    3. Create (position, log_density) pairs
    """
    # Define sampling volume (3σ from center of mass)
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    radius = 3.0 * np.std(np.linalg.norm(positions - com, axis=1))
    
    training_data = []
    for _ in range(n_samples):
        # Sample point uniformly in sphere
        u, v, w = np.random.uniform(0, 1, 3)
        theta = 2 * np.pi * u
        phi = np.arccos(2*v - 1)
        r = radius * w**(1/3)
        
        sample_point = com + r * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Compute density using adaptive kernel width
        h = compute_adaptive_kernel_width(sample_point, positions)
        density = compute_kernel_density(sample_point, positions, masses, h)
        
        training_data.append((sample_point, np.log(density + 1e-10)))
    
    return np.array(training_data)
```

#### Loss Function and Optimization

**Mean Squared Error in log-space:**
$$\mathcal{L}(\theta) = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} \left[\log \tilde{\rho}_{NN}(\mathbf{r}_i; \theta) - \log \rho_{true}(\mathbf{r}_i)\right]^2$$

**Gradient computation via automatic differentiation:**
$$\nabla_\theta \mathcal{L} = \frac{2}{N_{train}} \sum_{i=1}^{N_{train}} \left[\log \tilde{\rho}_{NN} - \log \rho_{true}\right] \cdot \nabla_\theta \log \tilde{\rho}_{NN}$$

### Theoretical Analysis

#### Error Bounds
For a neural network with $H$ hidden units, the approximation error scales as:
$$\|\rho_{true} - \tilde{\rho}_{NN}\|_{L^2} \leq \frac{C}{H^{s/d}}$$

where:
- $s$ = smoothness of target function (Sobolev regularity)
- $d$ = input dimension (3 for spatial density)
- $C$ = constant depending on function class

#### Convergence Properties
Under standard assumptions (bounded weights, Lipschitz activation), SGD converges to a local minimum at rate:
$$\mathbb{E}[\mathcal{L}(\theta_T)] - \mathcal{L}(\theta^*) \leq \frac{C}{\sqrt{T}}$$

---

## Project 2: Neural ODE Cluster Evolution Predictor

### Mathematical Foundation

#### 2.1 Moment Evolution Equations

The evolution of global cluster properties follows from the Boltzmann equation:
$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_{\mathbf{r}} f + \nabla_{\mathbf{r}}\Phi \cdot \nabla_{\mathbf{v}} f = \left(\frac{\partial f}{\partial t}\right)_{coll}$$

Taking moments yields evolution equations for macroscopic quantities:

**Mass conservation:**
$$\frac{dM}{dt} = -\int_{\partial V} \rho \mathbf{v} \cdot d\mathbf{S}$$

**Energy evolution:**
$$\frac{dE}{dt} = -\int_{\partial V} \left(\frac{1}{2}\rho v^2 + \rho\Phi\right) \mathbf{v} \cdot d\mathbf{S}$$

**Virial evolution:**
$$\frac{d^2I}{dt^2} = 4K + 2U$$

where $I = \sum_i m_i r_i^2$ is the moment of inertia.

#### 2.2 Neural ODE Formulation

**State vector definition:**
$$\mathbf{s}(t) = [E_{kin}(t), E_{pot}(t), r_{half}(t), r_{core}(t), \sigma_v(t), Q(t)]^T$$

where:
- $E_{kin}$ = total kinetic energy
- $E_{pot}$ = total potential energy  
- $r_{half}$ = half-mass radius
- $r_{core}$ = core radius (King model fit)
- $\sigma_v$ = velocity dispersion
- $Q$ = virial ratio $Q = -E_{kin}/E_{pot}$

**Neural ODE system:**
$$\frac{d\mathbf{s}}{dt} = f_\theta(\mathbf{s}, t)$$

where $f_\theta$ is a neural network with parameters $\theta$.

### Implementation Framework

#### Feature Engineering and Normalization

```python
def extract_cluster_features(positions, velocities, masses):
    """
    Extract physically meaningful global features.
    All features are normalized to be O(1) for numerical stability.
    """
    N = len(masses)
    M_total = np.sum(masses)
    
    # Center of mass frame
    com = np.sum(positions * masses[:, None], axis=0) / M_total
    com_vel = np.sum(velocities * masses[:, None], axis=0) / M_total
    
    pos_rel = positions - com
    vel_rel = velocities - com_vel
    
    # Kinetic energy (in units of GM²/r₀)
    E_kin = 0.5 * np.sum(masses * np.sum(vel_rel**2, axis=1))
    
    # Potential energy (gravitational)
    E_pot = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(pos_rel[i] - pos_rel[j])
            E_pot -= masses[i] * masses[j] / r_ij
    
    # Characteristic radii
    distances = np.linalg.norm(pos_rel, axis=1)
    sorted_indices = np.argsort(distances)
    cumsum_mass = np.cumsum(masses[sorted_indices])
    
    # Half-mass radius
    idx_half = np.searchsorted(cumsum_mass, 0.5 * M_total)
    r_half = distances[sorted_indices[idx_half]]
    
    # Core radius (10% mass)
    idx_core = np.searchsorted(cumsum_mass, 0.1 * M_total)
    r_core = distances[sorted_indices[idx_core]]
    
    # Velocity dispersion
    sigma_v = np.sqrt(np.mean(np.sum(vel_rel**2, axis=1)))
    
    # Normalize features
    r_scale = r_half  # Use half-mass radius as length scale
    E_scale = M_total**2 / r_scale  # Energy scale
    v_scale = np.sqrt(M_total / r_scale)  # Velocity scale
    
    return np.array([
        E_kin / E_scale,
        E_pot / E_scale,
        r_half / r_scale,  # This will be 1.0 by definition
        r_core / r_scale,
        sigma_v / v_scale,
        -E_kin / E_pot  # Virial ratio (dimensionless)
    ])
```

#### Neural ODE Architecture

```python
class NeuralODE(nn.Module):
    """
    Neural ODE for learning cluster evolution dynamics.
    
    Mathematical structure:
    ds/dt = f_θ(s, t) where f_θ is parameterized by a neural network
    """
    
    def setup(self):
        # Network layers with physics-informed initialization
        self.layers = [
            nn.Dense(64, kernel_init=nn.initializers.normal(0.01)),
            nn.Dense(64, kernel_init=nn.initializers.normal(0.01)),
            nn.Dense(6, kernel_init=nn.initializers.zeros)  # Start near zero dynamics
        ]
    
    def __call__(self, state, t):
        """
        Compute state derivatives.
        
        Physics constraints:
        - Energy should be approximately conserved: d(E_kin + E_pot)/dt ≈ 0
        - Virial ratio should evolve toward 0.5 (virial equilibrium)
        """
        # Concatenate state and time
        x = jnp.concatenate([state, jnp.array([t/100.0])])  # Normalize time
        
        # Forward pass through network
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.tanh(x)  # Smooth activation for continuous dynamics
        
        # Output layer (no activation)
        dx_dt = self.layers[-1](x)
        
        # Physics-informed corrections (optional)
        # Enforce approximate energy conservation
        dE_kin_dt = dx_dt[0]
        dE_pot_dt = dx_dt[1]
        
        # Soft constraint: reduce total energy change
        E_total_change = dE_kin_dt + dE_pot_dt
        correction = 0.1 * E_total_change  # Damping factor
        
        dx_dt = dx_dt.at[0].add(-correction/2)
        dx_dt = dx_dt.at[1].add(-correction/2)
        
        return dx_dt
```

### Training Strategy

#### Loss Function Design

**Multi-scale temporal loss:**
$$\mathcal{L}(\theta) = \sum_{k=1}^{K} w_k \int_0^{T} \left\|\mathbf{s}_{NN}(t; \theta) - \mathbf{s}_{true}(t)\right\|^2 dt$$

where weights $w_k$ emphasize different timescales.

**Numerical implementation:**
```python
def compute_neural_ode_loss(params, initial_state, true_trajectory, times):
    """
    Loss function for Neural ODE training.
    
    Mathematical formulation:
    L = ∫||s_pred(t) - s_true(t)||² dt ≈ Σᵢ ||s_pred(tᵢ) - s_true(tᵢ)||² Δt
    """
    # Solve ODE with neural network dynamics
    ode_fn = lambda s, t: neural_ode.apply(params, s, t)
    predicted_trajectory = odeint(ode_fn, initial_state, times)
    
    # Weighted MSE loss
    time_weights = np.exp(-times / times[-1])  # Emphasize early evolution
    
    losses = []
    for i, t in enumerate(times):
        state_error = predicted_trajectory[i] - true_trajectory[i]
        
        # Different weights for different quantities
        feature_weights = np.array([
            1.0,  # E_kin
            1.0,  # E_pot
            2.0,  # r_half (more important)
            1.0,  # r_core
            1.0,  # sigma_v
            0.5   # virial ratio
        ])
        
        weighted_error = state_error * feature_weights
        losses.append(time_weights[i] * np.sum(weighted_error**2))
    
    return np.mean(losses)
```

### Theoretical Guarantees

#### Existence and Uniqueness
Under Lipschitz conditions on $f_\theta$:
$$\|f_\theta(\mathbf{s}_1, t) - f_\theta(\mathbf{s}_2, t)\| \leq L\|\mathbf{s}_1 - \mathbf{s}_2\|$$

The ODE has a unique solution by the Picard-Lindelöf theorem.

#### Approximation Error
For Neural ODEs with ReLU networks of width $m$ and depth $L$:
$$\sup_{t \in [0,T]} \|\mathbf{s}_{true}(t) - \mathbf{s}_{NN}(t)\| \leq C \cdot \frac{T \cdot e^{LT}}{m^{1/d}}$$

---

## Project 3: Force Approximation Network

### Mathematical Foundation

#### 3.1 Force Decomposition

The total force on particle $i$:
$$\mathbf{F}_i = -Gm_i \sum_{j \neq i} m_j \frac{\mathbf{r}_i - \mathbf{r}_j}{|\mathbf{r}_i - \mathbf{r}_j|^3}$$

Decompose into near and far field:
$$\mathbf{F}_i = \mathbf{F}_i^{near} + \mathbf{F}_i^{far}$$

where:
- $\mathbf{F}_i^{near}$ = force from $k$ nearest neighbors
- $\mathbf{F}_i^{far}$ = force from remaining particles

#### 3.2 Multipole Expansion Theory

For distant particles, use multipole expansion:
$$\Phi(\mathbf{r}) = -G\sum_{l=0}^{\infty} \sum_{m=-l}^{l} \frac{q_{lm}}{r^{l+1}} Y_{lm}(\theta, \phi)$$

where multipole moments are:
$$q_{lm} = \sum_j m_j r_j^l Y_{lm}^*(\theta_j, \phi_j)$$

**Truncation error:**
$$|\Phi_{exact} - \Phi_{truncated}| \leq \frac{GM}{R} \left(\frac{a}{R}\right)^{L+1}$$

where $a$ = cluster size, $R$ = distance, $L$ = truncation order.

#### 3.3 Feature Design for Local Approximation

**Physical features for particle $i$:**

1. **Radial distribution function:**
$$g(r) = \frac{1}{4\pi r^2 \rho_0} \sum_{j \in neighbors} \delta(r - r_{ij})$$

2. **Mass-weighted moments:**
$$M_k = \sum_{j \in neighbors} m_j r_{ij}^k, \quad k = 0, 1, 2, ...$$

3. **Angular distribution:**
$$A_{lm} = \sum_{j \in neighbors} m_j Y_{lm}(\theta_{ij}, \phi_{ij})$$

4. **Local density gradient:**
$$\nabla \rho|_i = \sum_{j \in neighbors} m_j \nabla W(r_{ij}, h)$$

### Implementation Architecture

#### Feature Extraction Pipeline

```python
def extract_force_features(i, positions, masses, k=50):
    """
    Extract physically meaningful features for force approximation.
    
    Features capture:
    1. Local mass distribution (monopole)
    2. Dipole and quadrupole moments
    3. Radial distribution
    4. Anisotropy measures
    """
    # Find k nearest neighbors
    distances = np.linalg.norm(positions - positions[i], axis=1)
    neighbor_indices = np.argpartition(distances, k+1)[:k+1]
    neighbor_indices = neighbor_indices[neighbor_indices != i][:k]
    
    # Relative positions and distances
    rel_positions = positions[neighbor_indices] - positions[i]
    rel_distances = np.linalg.norm(rel_positions, axis=1)
    neighbor_masses = masses[neighbor_indices]
    
    features = []
    
    # 1. Radial distribution histogram (10 bins)
    r_max = np.max(rel_distances)
    r_bins = np.linspace(0, r_max, 11)
    mass_histogram, _ = np.histogram(rel_distances, bins=r_bins, weights=neighbor_masses)
    features.extend(mass_histogram / np.sum(neighbor_masses))  # Normalize
    
    # 2. Multipole moments (up to quadrupole)
    # Monopole (total mass)
    features.append(np.sum(neighbor_masses))
    
    # Dipole (center of mass offset)
    com_offset = np.sum(rel_positions * neighbor_masses[:, None], axis=0) / np.sum(neighbor_masses)
    features.extend(com_offset / r_max)  # Normalize by max distance
    
    # Quadrupole tensor (flattened)
    Q_tensor = np.zeros((3, 3))
    for j, (r_vec, m) in enumerate(zip(rel_positions, neighbor_masses)):
        for α in range(3):
            for β in range(3):
                Q_tensor[α, β] += m * (3*r_vec[α]*r_vec[β] - (α==β)*np.dot(r_vec, r_vec))
    
    # Use eigenvalues of quadrupole (rotationally invariant)
    Q_eigenvalues = np.linalg.eigvals(Q_tensor)
    features.extend(np.sort(Q_eigenvalues) / (np.sum(neighbor_masses) * r_max**2))
    
    # 3. Anisotropy measure (velocity dispersion tensor analogue)
    inertia_tensor = np.zeros((3, 3))
    for r_vec, m in zip(rel_positions, neighbor_masses):
        inertia_tensor += m * (np.eye(3) * np.dot(r_vec, r_vec) - np.outer(r_vec, r_vec))
    
    I_eigenvalues = np.linalg.eigvals(inertia_tensor)
    I_eigenvalues = np.sort(I_eigenvalues)
    
    # Triaxiality parameters
    if I_eigenvalues[2] > 0:
        features.append((I_eigenvalues[1] - I_eigenvalues[0]) / I_eigenvalues[2])  # Oblateness
        features.append((I_eigenvalues[2] - I_eigenvalues[1]) / I_eigenvalues[2])  # Prolateness
    else:
        features.extend([0, 0])
    
    # 4. Local density and its gradient
    h = r_max / 3  # Smoothing length
    local_density = np.sum(neighbor_masses * np.exp(-rel_distances**2 / (2*h**2))) / (2*np.pi*h**2)**(3/2)
    features.append(np.log(local_density + 1e-10))
    
    # Density gradient
    density_gradient = np.zeros(3)
    for r_vec, m, r in zip(rel_positions, neighbor_masses, rel_distances):
        if r > 0:
            density_gradient += m * (r_vec/r) * np.exp(-r**2/(2*h**2)) / (2*np.pi*h**2)**(3/2)
    
    features.extend(density_gradient / (local_density + 1e-10))
    
    return np.array(features)
```

#### Neural Network Architecture

```python
class ForceApproximator(nn.Module):
    """
    Approximate gravitational force using local features.
    
    Architecture design:
    - Input: High-dimensional feature vector (~25 features)
    - Hidden layers: Progressively reduce dimension
    - Output: 3D force vector
    """
    
    def setup(self):
        # Network designed to preserve rotational covariance
        self.encoder = [
            nn.Dense(128, kernel_init=nn.initializers.glorot_uniform()),
            nn.Dense(64, kernel_init=nn.initializers.glorot_uniform()),
            nn.Dense(32, kernel_init=nn.initializers.glorot_uniform())
        ]
        
        # Separate networks for magnitude and direction
        self.magnitude_net = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())
        self.direction_net = nn.Dense(3, kernel_init=nn.initializers.glorot_uniform())
    
    def __call__(self, features):
        # Encode features
        x = features
        for layer in self.encoder:
            x = layer(x)
            x = nn.gelu(x)  # Smooth activation for force continuity
        
        # Predict force magnitude (positive)
        log_magnitude = self.magnitude_net(x)
        magnitude = nn.softplus(log_magnitude)  # Ensure positive
        
        # Predict force direction (unit vector)
        direction = self.direction_net(x)
        direction = direction / (jnp.linalg.norm(direction) + 1e-8)
        
        # Combine magnitude and direction
        force = magnitude * direction
        
        return force
```

### Error Analysis

#### Approximation Error Bounds

For k-nearest neighbor approximation:
$$\|\mathbf{F}_i^{exact} - \mathbf{F}_i^{approx}\| \leq \frac{GM_{far}}{r_{k+1}^2}$$

where $M_{far}$ = total mass beyond k-th neighbor, $r_{k+1}$ = distance to (k+1)-th neighbor.

#### Convergence with k
The error decreases as:
$$\epsilon(k) \sim \frac{1}{k^{2/3}} \quad \text{for uniform density}$$
$$\epsilon(k) \sim \frac{1}{k^{1/2}} \quad \text{for centrally concentrated systems}$$

### Validation Metrics

```python
def validate_force_approximation(nn_forces, exact_forces):
    """
    Comprehensive validation of force approximation quality.
    
    Metrics:
    1. Relative force error
    2. Angular error
    3. Energy conservation
    4. Angular momentum conservation
    """
    # 1. Relative force error
    force_magnitudes_exact = np.linalg.norm(exact_forces, axis=1)
    force_magnitudes_nn = np.linalg.norm(nn_forces, axis=1)
    
    relative_magnitude_error = np.abs(force_magnitudes_nn - force_magnitudes_exact) / (force_magnitudes_exact + 1e-10)
    
    # 2. Angular error (in radians)
    angular_errors = []
    for f_exact, f_nn in zip(exact_forces, nn_forces):
        if np.linalg.norm(f_exact) > 1e-10 and np.linalg.norm(f_nn) > 1e-10:
            cos_angle = np.dot(f_exact, f_nn) / (np.linalg.norm(f_exact) * np.linalg.norm(f_nn))
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical safety
            angular_errors.append(np.arccos(cos_angle))
    
    # 3. Check conservation laws
    # Total force (should be zero by Newton's third law)
    total_force_exact = np.sum(exact_forces, axis=0)
    total_force_nn = np.sum(nn_forces, axis=0)
    
    momentum_violation = np.linalg.norm(total_force_nn) / np.linalg.norm(total_force_exact + 1e-10)
    
    return {
        'mean_relative_error': np.mean(relative_magnitude_error),
        'max_relative_error': np.max(relative_magnitude_error),
        'mean_angular_error_deg': np.degrees(np.mean(angular_errors)),
        'momentum_violation': momentum_violation,
        'rms_error': np.sqrt(np.mean(np.sum((nn_forces - exact_forces)**2, axis=1)))
    }
```

---

## Project 4: Physics-Informed Neural Network (PINN) Potential Solver

### Mathematical Foundation

#### 4.1 Poisson Equation for Gravitational Potential

The gravitational potential satisfies:
$$\nabla^2 \Phi(\mathbf{r}) = 4\pi G \rho(\mathbf{r})$$

For point masses:
$$\rho(\mathbf{r}) = \sum_{i=1}^N m_i \delta^3(\mathbf{r} - \mathbf{r}_i)$$

With boundary condition at infinity:
$$\Phi(\mathbf{r}) \rightarrow -\frac{GM_{total}}{|\mathbf{r}|} \quad \text{as } |\mathbf{r}| \rightarrow \infty$$

#### 4.2 Variational Formulation

The potential minimizes the energy functional:
$$\mathcal{E}[\Phi] = \frac{1}{8\pi G} \int |\nabla\Phi|^2 d^3\mathbf{r} - \int \rho\Phi d^3\mathbf{r}$$

First variation yields Poisson equation:
$$\frac{\delta \mathcal{E}}{\delta \Phi} = -\frac{1}{4\pi G}\nabla^2\Phi - \rho = 0$$

#### 4.3 PINN Loss Function Construction

**Total loss:**
$$\mathcal{L}_{total} = \lambda_{PDE}\mathcal{L}_{PDE} + \lambda_{BC}\mathcal{L}_{BC} + \lambda_{data}\mathcal{L}_{data}$$

where:
- $\mathcal{L}_{PDE}$ = PDE residual loss
- $\mathcal{L}_{BC}$ = boundary condition loss
- $\mathcal{L}_{data}$ = data fitting loss (if available)

### Implementation Framework

#### Automatic Differentiation for Laplacian

```python
def compute_laplacian(potential_fn, r):
    """
    Compute Laplacian of potential using automatic differentiation.
    
    Mathematical operation:
    ∇²Φ = ∂²Φ/∂x² + ∂²Φ/∂y² + ∂²Φ/∂z²
    
    JAX implementation uses forward-mode AD for efficiency.
    """
    # Define function for single coordinate
    def potential_scalar(r_flat):
        return potential_fn(r_flat.reshape(1, 3))[0]
    
    # Compute Hessian matrix
    hessian_fn = jax.hessian(potential_scalar)
    hessian = hessian_fn(r.flatten())
    
    # Laplacian is trace of Hessian
    laplacian = jnp.trace(hessian.reshape(3, 3))
    
    return laplacian

# More efficient batched version
def compute_laplacian_batch(potential_fn, points):
    """
    Compute Laplacian at multiple points efficiently.
    Uses vmap for parallelization.
    """
    def laplacian_single(r):
        # Use jacfwd for forward-mode AD (more efficient for scalar output)
        grad_fn = jax.grad(lambda x: potential_fn(x.reshape(1, 3))[0])
        
        # Compute gradient
        def grad_component(x, idx):
            r_perturbed = r.at[idx].set(x)
            return grad_fn(r_perturbed)[idx]
        
        # Second derivatives
        d2_dx2 = jax.grad(lambda x: grad_component(x, 0))(r[0])
        d2_dy2 = jax.grad(lambda y: grad_component(y, 1))(r[1])
        d2_dz2 = jax.grad(lambda z: grad_component(z, 2))(r[2])
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    return jax.vmap(laplacian_single)(points)
```

#### PINN Architecture and Training

```python
class PINNPotential(nn.Module):
    """
    Physics-Informed Neural Network for gravitational potential.
    
    Architecture choices:
    - Fourier features for better high-frequency learning
    - Skip connections for gradient flow
    - Tanh activations for smooth potentials
    """
    
    def setup(self):
        self.fourier_features = self.param(
            'fourier_matrix',
            nn.initializers.normal(1.0),
            (32, 3)  # 32 Fourier features from 3D input
        )
        
        self.layers = [
            nn.Dense(128, kernel_init=nn.initializers.glorot_normal()),
            nn.Dense(128, kernel_init=nn.initializers.glorot_normal()),
            nn.Dense(64, kernel_init=nn.initializers.glorot_normal()),
            nn.Dense(32, kernel_init=nn.initializers.glorot_normal()),
            nn.Dense(1, kernel_init=nn.initializers.zeros)  # Single output
        ]
        
    def __call__(self, r):
        """
        Forward pass with Fourier feature encoding.
        
        Fourier features help with spectral bias problem:
        Neural networks naturally learn low frequencies first.
        """
        # Fourier feature encoding
        fourier_features = jnp.concatenate([
            jnp.sin(r @ self.fourier_features.T),
            jnp.cos(r @ self.fourier_features.T)
        ], axis=-1)
        
        # Concatenate with original coordinates (skip connection)
        x = jnp.concatenate([r, fourier_features], axis=-1)
        
        # Forward pass with skip connections
        x_input = x
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.tanh(x)
            
            # Skip connection every 2 layers
            if i % 2 == 1 and i < len(self.layers) - 2:
                x = jnp.concatenate([x, x_input], axis=-1)
        
        # Output layer (no activation)
        potential = self.layers[-1](x)
        
        # Add physically motivated prior: -GM/r behavior
        r_norm = jnp.linalg.norm(r, axis=-1, keepdims=True) + 1e-6
        potential = potential - 1.0 / r_norm  # Assuming GM=1 in code units
        
        return potential
```

#### Loss Function Implementation

```python
def pinn_loss_function(params, model, particle_positions, particle_masses, 
                       collocation_points, boundary_points):
    """
    Complete PINN loss for Poisson equation.
    
    Components:
    1. PDE loss: ∇²Φ = 4πGρ at collocation points
    2. Boundary loss: Φ → -GM/r at large r
    3. Source loss: Correct potential at particle locations (optional)
    """
    G = 1.0  # Gravitational constant in code units
    
    # 1. PDE Loss at collocation points
    pde_loss = 0.0
    potential_fn = lambda r: model.apply(params, r)
    
    # Compute Laplacian at collocation points
    laplacians = compute_laplacian_batch(potential_fn, collocation_points)
    
    # Compute density at collocation points (smoothed)
    densities = jnp.zeros(len(collocation_points))
    smoothing_length = 0.1  # Adjust based on particle separation
    
    for i, point in enumerate(collocation_points):
        for j, (pos, mass) in enumerate(zip(particle_positions, particle_masses)):
            r = jnp.linalg.norm(point - pos)
            # Plummer softening for density
            densities = densities.at[i].add(
                mass / (4*jnp.pi * (r**2 + smoothing_length**2)**(3/2))
            )
    
    # PDE residual
    pde_residual = laplacians - 4*jnp.pi*G*densities
    pde_loss = jnp.mean(pde_residual**2)
    
    # 2. Boundary condition loss
    bc_loss = 0.0
    M_total = jnp.sum(particle_masses)
    
    for point in boundary_points:
        r = jnp.linalg.norm(point)
        potential_exact = -G * M_total / r
        potential_nn = potential_fn(point.reshape(1, 3))[0, 0]
        bc_loss += (potential_nn - potential_exact)**2
    
    bc_loss = bc_loss / len(boundary_points)
    
    # 3. Optional: Source term loss at particle locations
    source_loss = 0.0
    for i, (pos_i, mass_i) in enumerate(zip(particle_positions, particle_masses)):
        # Potential at particle i due to all others
        potential_exact = 0.0
        for j, (pos_j, mass_j) in enumerate(zip(particle_positions, particle_masses)):
            if i != j:
                r_ij = jnp.linalg.norm(pos_i - pos_j)
                potential_exact -= G * mass_j / (r_ij + smoothing_length)
        
        potential_nn = potential_fn(pos_i.reshape(1, 3))[0, 0]
        source_loss += mass_i * (potential_nn - potential_exact)**2
    
    source_loss = source_loss / M_total
    
    # Combine losses with weights
    total_loss = pde_loss + 0.1 * bc_loss + 0.01 * source_loss
    
    # Return individual components for monitoring
    return total_loss, {'pde': pde_loss, 'bc': bc_loss, 'source': source_loss}
```

### Theoretical Analysis

#### Convergence Theory for PINNs

**Universal approximation in Sobolev spaces:**
For target function $\Phi \in H^s(\Omega)$ (Sobolev space), a neural network with ReLU activations, width $m$ and depth $L$ achieves:
$$\|\Phi - \Phi_{NN}\|_{H^s} \leq C \cdot m^{-s/d} \cdot L^{-s}$$

**PDE residual convergence:**
$$\|\nabla^2\Phi_{NN} - 4\pi G\rho\|_{L^2} \leq \epsilon_{approx} + \epsilon_{opt} + \epsilon_{quad}$$

where:
- $\epsilon_{approx}$ = network approximation error
- $\epsilon_{opt}$ = optimization error
- $\epsilon_{quad}$ = quadrature error from collocation points

#### Sampling Strategy for Collocation Points

**Importance sampling based on density:**
```python
def sample_collocation_points(particle_positions, particle_masses, n_points):
    """
    Sample collocation points with importance sampling.
    Higher density near particles where potential varies rapidly.
    """
    points = []
    
    # 60% near particles (high gradient region)
    n_near = int(0.6 * n_points)
    for _ in range(n_near):
        # Pick random particle
        idx = np.random.randint(len(particle_positions))
        center = particle_positions[idx]
        
        # Sample within 3 smoothing lengths
        offset = np.random.randn(3) * 0.3
        points.append(center + offset)
    
    # 30% in intermediate region
    n_mid = int(0.3 * n_points)
    com = np.average(particle_positions, weights=particle_masses, axis=0)
    r_cluster = np.std(np.linalg.norm(particle_positions - com, axis=1))
    
    for _ in range(n_mid):
        # Uniform in sphere of radius 2*r_cluster
        u = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(2*np.random.uniform(0, 1) - 1)
        r = 2 * r_cluster * u**(1/3)
        
        point = com + r * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        points.append(point)
    
    # 10% far field (boundary condition)
    n_far = n_points - n_near - n_mid
    for _ in range(n_far):
        # Large radius
        r = np.random.uniform(5, 10) * r_cluster
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(2*np.random.uniform(0, 1) - 1)
        
        point = com + r * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        points.append(point)
    
    return np.array(points)
```

---

## Project 5: Binary Encounter Predictor

### Mathematical Foundation

#### 5.1 Two-Body Problem and Binding Criteria

For two masses $m_1$, $m_2$ with relative position $\mathbf{r} = \mathbf{r}_2 - \mathbf{r}_1$ and relative velocity $\mathbf{v} = \mathbf{v}_2 - \mathbf{v}_1$:

**Binding energy:**
$$E = \frac{1}{2}\mu v^2 - \frac{Gm_1m_2}{r}$$

where $\mu = \frac{m_1m_2}{m_1+m_2}$ is the reduced mass.

**Binding criterion:**
$$E < 0 \Rightarrow \text{bound system}$$

**Orbital parameters:**
- Semi-major axis: $a = -\frac{Gm_1m_2}{2E}$ (for $E < 0$)
- Eccentricity: $e = \sqrt{1 + \frac{2EL^2}{\mu G^2(m_1+m_2)^2}}$
- Angular momentum: $L = \mu|\mathbf{r} \times \mathbf{v}|$

#### 5.2 Three-Body Encounter Dynamics

**Heggie's Law:**
- Hard binaries (high binding energy) tend to get harder
- Soft binaries (low binding energy) tend to get softer

**Critical velocity** (hard-soft boundary):
$$v_{crit} = \sqrt{\frac{G(m_1+m_2)}{a}}$$

**Cross section for binary formation:**
$$\sigma = \pi b_{max}^2$$

where maximum impact parameter:
$$b_{max} = \frac{G(m_1+m_2)}{v_{\infty}^2}$$

#### 5.3 Environmental Effects

**Tidal disruption radius:**
$$r_{tidal} = \left(\frac{m_1+m_2}{M_{enc}}\right)^{1/3} R$$

where $M_{enc}$ is enclosed mass at radius $R$.

### Feature Engineering

```python
def extract_encounter_features(m1, m2, r_rel, v_rel, local_environment):
    """
    Extract physically meaningful features for binary formation prediction.
    
    Based on:
    1. Two-body orbital parameters
    2. Three-body encounter theory
    3. Environmental influences
    """
    G = 1.0  # Code units
    
    # Basic parameters
    M_total = m1 + m2
    mu = m1 * m2 / M_total  # Reduced mass
    q = min(m1, m2) / max(m1, m2)  # Mass ratio
    
    # Relative motion
    r = np.linalg.norm(r_rel)
    v = np.linalg.norm(v_rel)
    
    # Angular momentum
    L_vec = mu * np.cross(r_rel, v_rel)
    L = np.linalg.norm(L_vec)
    
    # Energy
    E = 0.5 * mu * v**2 - G * m1 * m2 / r
    
    # Orbital parameters (current encounter)
    # Periapsis distance (closest approach)
    a_encounter = -G * m1 * m2 / (2 * E) if E < 0 else np.inf
    e_encounter = np.sqrt(1 + 2*E*L**2/(mu * G**2 * M_total**2)) if E < 0 else np.sqrt(1 + 2*E*L**2/(mu * G**2 * M_total**2))
    r_peri = a_encounter * (1 - e_encounter) if E < 0 else L**2 / (G * M_total * mu * (1 + e_encounter))
    
    # Escape velocity at current separation
    v_escape = np.sqrt(2 * G * M_total / r)
    
    # Dimensionless parameters
    beta = v / v_escape  # Velocity ratio
    
    # Impact parameter (classical)
    b = L / (mu * v) if v > 0 else 0
    b_max = G * M_total / v**2 if v > 0 else np.inf
    
    # Environmental features
    rho_local = local_environment['density']
    sigma_v_local = local_environment['velocity_dispersion']
    N_neighbors = local_environment['n_neighbors_within_10r']
    
    # Perturbation timescale vs orbital timescale
    if E < 0 and a_encounter > 0:
        t_orbital = 2 * np.pi * np.sqrt(a_encounter**3 / (G * M_total))
    else:
        t_orbital = r / v  # Crossing time for unbound
    
    t_perturbation = 1 / (np.sqrt(G * rho_local)) if rho_local > 0 else np.inf
    
    # Hardness parameter (Heggie's law)
    v_hard = sigma_v_local
    hardness = v_escape / (v_hard + 1e-10)
    
    # Feature vector
    features = np.array([
        # Masses
        np.log(M_total),           # Total mass (log scale)
        q,                          # Mass ratio
        
        # Energetics
        E / (G * M_total**2 / r),   # Dimensionless energy
        beta,                       # v/v_escape
        
        # Angular momentum
        L / (mu * np.sqrt(G * M_total * r)),  # Dimensionless L
        
        # Orbital parameters
        r_peri / r,                 # Periapsis ratio
        e_encounter,                # Eccentricity
        
        # Impact parameter
        b / b_max if b_max < np.inf else 0,  # Normalized impact parameter
        
        # Environmental
        np.log(rho_local + 1e-10),          # Local density
        sigma_v_local / v_escape,            # Velocity dispersion ratio
        np.log(N_neighbors + 1),            # Number of perturbers
        t_orbital / t_perturbation,          # Timescale ratio
        hardness,                            # Binary hardness
        
        # Directional information (may affect perturbations)
        np.abs(L_vec[2]) / (L + 1e-10),     # Angular momentum alignment with z
    ])
    
    return features
```

### Classification Model

```python
class BinaryFormationClassifier(nn.Module):
    """
    Neural network classifier for binary formation prediction.
    
    Architecture motivated by:
    - Need for non-linear decision boundaries in phase space
    - Importance of threshold effects (e.g., binding energy)
    - Interaction between orbital and environmental parameters
    """
    
    def setup(self):
        # Hidden layers with dropout for regularization
        self.layers = [
            nn.Dense(64, kernel_init=nn.initializers.he_normal()),
            nn.Dropout(0.2),
            nn.Dense(32, kernel_init=nn.initializers.he_normal()),
            nn.Dropout(0.1),
            nn.Dense(16, kernel_init=nn.initializers.he_normal()),
            nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())
        ]
        
    def __call__(self, features, training=False):
        x = features
        
        # Hidden layers with ReLU activation
        x = self.layers[0](x)
        x = nn.relu(x)
        x = self.layers[1](x, deterministic=not training)
        
        x = self.layers[2](x)
        x = nn.relu(x)
        x = self.layers[3](x, deterministic=not training)
        
        x = self.layers[4](x)
        x = nn.relu(x)
        
        # Output layer (logit)
        logit = self.layers[5](x)
        
        # Probability via sigmoid
        probability = nn.sigmoid(logit)
        
        return probability
```

### Training Strategy

#### Class Imbalance Handling

```python
def compute_weighted_loss(predictions, labels, class_weights):
    """
    Weighted binary cross-entropy for imbalanced data.
    
    Binary formation is typically rare, so we need to handle class imbalance.
    """
    # Compute class weights from training data
    n_positive = np.sum(labels)
    n_negative = len(labels) - n_positive
    
    if class_weights is None:
        # Inverse frequency weighting
        weight_positive = len(labels) / (2 * n_positive + 1e-10)
        weight_negative = len(labels) / (2 * n_negative + 1e-10)
    else:
        weight_positive, weight_negative = class_weights
    
    # Weighted BCE loss
    epsilon = 1e-7  # Numerical stability
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    loss = -weight_positive * labels * np.log(predictions) \
           -weight_negative * (1 - labels) * np.log(1 - predictions)
    
    return np.mean(loss)
```

### Theoretical Insights

#### Phase Space Structure

The binary formation probability exhibits distinct regions:
1. **Guaranteed binding** ($E \ll 0$): Nearly 100% formation
2. **Marginal binding** ($E \approx 0$): Environmental effects dominate
3. **Fly-by regime** ($E \gg 0$): Near 0% formation

#### Critical Surfaces

The decision boundary learned by the classifier approximates:
$$P(binary) = \sigma\left(\alpha_0 + \alpha_1 \frac{E}{|E_{crit}|} + \alpha_2 \log\rho + ...\right)$$

where $\sigma$ is the sigmoid function and $\alpha_i$ are learned coefficients.

---

## Project 6: Relaxation Time Predictor

### Mathematical Foundation

#### 6.1 Two-Body Relaxation Theory

**Relaxation time** (Spitzer 1987):
$$t_{relax} = \frac{0.138 N}{\ln\Lambda} \left(\frac{\bar{m}}{M}\right)^{1/2} t_{cross}$$

where:
- $N$ = number of particles
- $\ln\Lambda$ = Coulomb logarithm $\approx \ln(0.4N)$
- $\bar{m}$ = mean stellar mass
- $M$ = total cluster mass
- $t_{cross} = R/v$ = crossing time

**Refined formula including mass spectrum:**
$$t_{relax} = \frac{0.138}{\ln\Lambda} \frac{\langle m \rangle^2}{\langle m^2 \rangle} \frac{N}{\bar{\rho}} \left(\frac{3\sigma^2}{G}\right)^{3/2}$$

#### 6.2 Fokker-Planck Equation

Evolution of distribution function:
$$\frac{\partial f}{\partial t} = -\sum_j \frac{\partial}{\partial v_i}\left[f\langle\Delta v_i\rangle_j\right] + \frac{1}{2}\sum_j \frac{\partial^2}{\partial v_i \partial v_k}\left[f\langle\Delta v_i \Delta v_k\rangle_j\right]$$

Diffusion coefficients:
$$\langle\Delta v_\parallel\rangle = -\frac{4\pi G^2 m_j \ln\Lambda}{v^2}(m_i + m_j)f_j$$

$$\langle(\Delta v_\perp)^2\rangle = \frac{4\pi G^2 m_j^2 \ln\Lambda}{v}[\text{erf}(\chi) - \chi\text{erf}'(\chi)]f_j$$

#### 6.3 Dependence on Initial Conditions

Key parameters affecting relaxation:
1. **Particle number**: $t_{relax} \propto N/\ln N$
2. **Mass function**: Through $\langle m^2\rangle/\langle m\rangle^2$
3. **Concentration**: Affects local density $\bar{\rho}$
4. **Velocity distribution**: Through $\sigma$
5. **Primordial binaries**: Effective N reduction

### Feature Extraction

```python
def compute_relaxation_features(positions, velocities, masses):
    """
    Extract features predictive of relaxation time.
    
    Based on theoretical understanding and empirical correlations.
    """
    N = len(masses)
    
    # Basic statistics
    M_total = np.sum(masses)
    m_mean = np.mean(masses)
    m_std = np.std(masses)
    
    # Mass function moments (critical for relaxation)
    m2_mean = np.mean(masses**2)
    mass_ratio = m2_mean / m_mean**2  # Spitzer mass factor
    
    # Structural parameters
    com = np.average(positions, weights=masses, axis=0)
    radii = np.linalg.norm(positions - com, axis=1)
    
    # Lagrangian radii (mass-weighted percentiles)
    sorted_indices = np.argsort(radii)
    cumulative_mass = np.cumsum(masses[sorted_indices]) / M_total
    
    r_10 = radii[sorted_indices[np.searchsorted(cumulative_mass, 0.1)]]
    r_50 = radii[sorted_indices[np.searchsorted(cumulative_mass, 0.5)]]
    r_90 = radii[sorted_indices[np.searchsorted(cumulative_mass, 0.9)]]
    
    # Concentration parameter (King-like)
    concentration = np.log(r_90 / r_10)
    
    # Central density (within r_10)
    central_mask = radii < r_10
    rho_central = np.sum(masses[central_mask]) / (4/3 * np.pi * r_10**3)
    
    # Velocity dispersion (global and radial dependence)
    v_com = np.average(velocities, weights=masses, axis=0)
    v_rel = velocities - v_com
    sigma_global = np.sqrt(np.average(np.sum(v_rel**2, axis=1), weights=masses))
    
    # Anisotropy parameter
    v_radial = np.sum(v_rel * (positions - com), axis=1) / (radii + 1e-10)
    v_tangential = np.sqrt(np.sum(v_rel**2, axis=1) - v_radial**2)
    
    sigma_r = np.sqrt(np.average(v_radial**2, weights=masses))
    sigma_t = np.sqrt(np.average(v_tangential**2, weights=masses))
    beta_anisotropy = 1 - (sigma_t**2) / (2 * sigma_r**2) if sigma_r > 0 else 0
    
    # Virial ratio (equilibrium measure)
    KE = 0.5 * np.sum(masses * np.sum(v_rel**2, axis=1))
    PE = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            PE -= masses[i] * masses[j] / (r_ij + 1e-10)
    
    virial_ratio = -KE / PE
    
    # Crossing time (dynamical time)
    t_cross = r_50 / sigma_global
    
    # Coulomb logarithm
    ln_Lambda = np.log(0.4 * N)
    
    # Theoretical relaxation time estimate (for comparison)
    t_relax_theory = (0.138 * N / ln_Lambda) * np.sqrt(m_mean / M_total) * t_cross * mass_ratio
    
    # Feature vector (all dimensionless or in consistent units)
    features = np.array([
        np.log(N),                    # Log particle number
        np.log(M_total),              # Log total mass
        m_mean / M_total * N,         # Normalized mean mass
        mass_ratio,                   # Mass spectrum factor
        concentration,                # Concentration parameter
        np.log(rho_central),         # Log central density
        sigma_global,                 # Velocity dispersion
        beta_anisotropy,             # Velocity anisotropy
        virial_ratio,                # Virial equilibrium measure
        r_50 / r_10,                 # Core-halo structure
        r_90 / r_50,                 # Halo extent
        np.log(t_cross),             # Log crossing time
        ln_Lambda,                   # Coulomb logarithm
        np.log(t_relax_theory),      # Theoretical estimate (baseline)
    ])
    
    return features
```

### Regression Model

```python
class RelaxationTimePredictor(nn.Module):
    """
    Predict relaxation time from initial conditions.
    
    Architecture:
    - Input normalization layer
    - Multiple hidden layers with residual connections
    - Exponential output (ensures positive times)
    """
    
    def setup(self):
        # Batch normalization for input features
        self.input_norm = nn.BatchNorm(use_running_average=False)
        
        # Main network
        self.hidden1 = nn.Dense(64, kernel_init=nn.initializers.he_normal())
        self.hidden2 = nn.Dense(32, kernel_init=nn.initializers.he_normal())
        self.hidden3 = nn.Dense(16, kernel_init=nn.initializers.he_normal())
        
        # Output layer
        self.output_layer = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())
        
        # Residual connection from theoretical estimate
        self.theory_weight = self.param('theory_weight', nn.initializers.ones, (1,))
        
    def __call__(self, features, training=False):
        # Extract theoretical estimate (last feature)
        theory_estimate = features[:, -1:2]
        
        # Normalize features
        x = self.input_norm(features, use_running_average=not training)
        
        # First hidden layer
        h1 = self.hidden1(x)
        h1 = nn.gelu(h1)
        
        # Second hidden layer with skip connection
        h2 = self.hidden2(h1)
        h2 = nn.gelu(h2)
        
        # Third hidden layer
        h3 = self.hidden3(h2)
        h3 = nn.gelu(h3)
        
        # Output (log-scale prediction)
        log_correction = self.output_layer(h3)
        
        # Combine with theoretical estimate
        # Network learns correction factor to theory
        log_t_relax = theory_estimate + log_correction
        
        # Exponential to ensure positive time
        t_relax = jnp.exp(log_t_relax)
        
        return t_relax
```

### Training and Validation

#### Data Generation Strategy

```python
def generate_diverse_initial_conditions(n_samples):
    """
    Generate diverse cluster ICs for training.
    
    Sampling strategy covers:
    - Wide range of N
    - Different IMFs
    - Various concentrations
    - Different virial states
    """
    initial_conditions = []
    
    for i in range(n_samples):
        # Sample parameters from distributions
        N = int(10 ** np.random.uniform(2, 4))  # 100 to 10,000 particles
        
        # IMF slope (Kroupa-like)
        alpha = np.random.uniform(-2.5, -1.0)
        
        # King concentration
        W0 = np.random.uniform(3, 9)
        
        # Virial ratio (sub-virial to super-virial)
        Q_target = np.random.uniform(0.3, 0.7)
        
        # Generate cluster
        positions, velocities, masses = generate_king_model(N, W0, alpha, Q_target)
        
        initial_conditions.append({
            'positions': positions,
            'velocities': velocities,
            'masses': masses,
            'parameters': {'N': N, 'alpha': alpha, 'W0': W0, 'Q': Q_target}
        })
    
    return initial_conditions
```

#### Loss Function with Physics Constraints

```python
def physics_informed_loss(predictions, targets, features):
    """
    Loss function with physical constraints.
    
    Incorporates:
    1. Standard MSE in log-space
    2. Monotonicity constraints (larger N -> longer t_relax)
    3. Scaling law penalties
    """
    # Primary loss: MSE in log-space
    log_predictions = jnp.log(predictions + 1e-10)
    log_targets = jnp.log(targets + 1e-10)
    mse_loss = jnp.mean((log_predictions - log_targets)**2)
    
    # Physics constraint 1: Monotonicity with N
    # Sort by N (first feature after log transform)
    N_values = jnp.exp(features[:, 0])
    sorted_indices = jnp.argsort(N_values)
    sorted_predictions = predictions[sorted_indices]
    
    # Penalize violations of monotonicity
    monotonicity_violations = jnp.maximum(0, sorted_predictions[:-1] - sorted_predictions[1:])
    monotonicity_loss = jnp.mean(monotonicity_violations**2)
    
    # Physics constraint 2: Approximate scaling
    # t_relax should scale roughly as N/ln(N)
    expected_scaling = N_values / jnp.log(N_values)
    expected_scaling = expected_scaling / jnp.mean(expected_scaling)
    actual_scaling = predictions.flatten() / jnp.mean(predictions)
    
    scaling_loss = jnp.mean((jnp.log(actual_scaling) - jnp.log(expected_scaling))**2)
    
    # Combined loss
    total_loss = mse_loss + 0.1 * monotonicity_loss + 0.05 * scaling_loss
    
    return total_loss, {
        'mse': mse_loss,
        'monotonicity': monotonicity_loss,
        'scaling': scaling_loss
    }
```

### Theoretical Validation

#### Comparison with Analytical Models

```python
def validate_against_theory(predictor, test_data):
    """
    Compare NN predictions with theoretical models.
    
    Tests:
    1. Spitzer-Chandrasekhar formula
    2. Gieles & Zocchi (2015) fitting formula
    3. Direct N-body measurements
    """
    results = {
        'nn_predictions': [],
        'spitzer_predictions': [],
        'gieles_predictions': [],
        'nbody_measurements': []
    }
    
    for data in test_data:
        features = compute_relaxation_features(
            data['positions'], 
            data['velocities'], 
            data['masses']
        )
        
        # Neural network prediction
        t_nn = predictor(features.reshape(1, -1))[0, 0]
        results['nn_predictions'].append(t_nn)
        
        # Spitzer formula
        N = len(data['masses'])
        ln_Lambda = np.log(0.4 * N)
        t_spitzer = compute_spitzer_relaxation_time(data)
        results['spitzer_predictions'].append(t_spitzer)
        
        # Gieles & Zocchi formula (includes mass spectrum effects)
        t_gieles = compute_gieles_relaxation_time(data)
        results['gieles_predictions'].append(t_gieles)
        
        # Direct N-body (if available)
        if 'measured_t_relax' in data:
            results['nbody_measurements'].append(data['measured_t_relax'])
    
    # Compute statistics
    correlations = {}
    if results['nbody_measurements']:
        true_values = np.array(results['nbody_measurements'])
        
        for method in ['nn_predictions', 'spitzer_predictions', 'gieles_predictions']:
            predictions = np.array(results[method][:len(true_values)])
            
            # Pearson correlation in log-space
            corr = np.corrcoef(np.log(predictions), np.log(true_values))[0, 1]
            
            # Relative errors
            rel_errors = np.abs(predictions - true_values) / true_values
            
            correlations[method] = {
                'correlation': corr,
                'mean_relative_error': np.mean(rel_errors),
                'median_relative_error': np.median(rel_errors),
                'max_relative_error': np.max(rel_errors)
            }
    
    return correlations
```

---

## Part III: Implementation Best Practices

### JAX-Specific Optimizations

#### 1. Vectorization with vmap

```python
# Inefficient: loop over particles
def compute_densities_loop(positions, masses, sample_points):
    densities = []
    for point in sample_points:
        density = 0
        for pos, mass in zip(positions, masses):
            r = np.linalg.norm(point - pos)
            density += mass / (4*np.pi*r**3)
        densities.append(density)
    return np.array(densities)

# Efficient: vectorized with vmap
def compute_density_single(point, positions, masses):
    distances = jnp.linalg.norm(positions - point, axis=1)
    return jnp.sum(masses / (4*jnp.pi*distances**3))

compute_densities_vmap = jax.vmap(compute_density_single, in_axes=(0, None, None))
```

#### 2. JIT Compilation

```python
@jax.jit
def train_step(params, batch, optimizer_state):
    """JIT-compiled training step for maximum performance."""
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss
```

#### 3. Memory-Efficient Gradients

```python
# Use gradient checkpointing for large networks
from jax.experimental import checkpoint

@checkpoint  # Recompute activations during backprop
def large_network_block(x):
    for layer in large_layers:
        x = layer(x)
        x = activation(x)
    return x
```

### Validation and Testing Framework

#### Cross-Validation Strategy

```python
def k_fold_cross_validation(data, model_fn, k=5):
    """
    K-fold CV for robust performance estimation.
    """
    n = len(data)
    fold_size = n // k
    
    metrics = []
    
    for fold in range(k):
        # Split data
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k-1 else n
        
        val_data = data[val_start:val_end]
        train_data = np.concatenate([data[:val_start], data[val_end:]])
        
        # Train model
        model = model_fn()
        model.fit(train_data)
        
        # Validate
        val_metrics = model.evaluate(val_data)
        metrics.append(val_metrics)
    
    # Aggregate results
    return {
        'mean': np.mean(metrics, axis=0),
        'std': np.std(metrics, axis=0),
        'all_folds': metrics
    }
```

---

## Part IV: Summary and Future Directions

### Key Learning Outcomes

Through these projects, students master:

1. **Theoretical Understanding**
   - N-body dynamics and statistical mechanics
   - Relaxation processes and dynamical evolution
   - Binary formation and three-body encounters

2. **Machine Learning Techniques**
   - Supervised learning (regression and classification)
   - Physics-informed neural networks
   - Neural ODEs for dynamical systems
   - Feature engineering from physical principles

3. **Computational Skills**
   - JAX framework and automatic differentiation
   - Efficient vectorization and parallelization
   - Numerical validation and error analysis

### Extensions and Advanced Topics

1. **Hierarchical Systems**
   - Multiple stellar systems
   - Planetary dynamics in clusters
   - Tidal disruption events

2. **Advanced ML Methods**
   - Graph neural networks for N-body systems
   - Transformer architectures for time series
   - Generative models for initial conditions

3. **Hybrid Approaches**
   - ML-enhanced tree codes
   - Neural network force softening
   - Adaptive timestepping with ML

### Connections to Current Research

These projects connect to active research areas:
- Fast N-body solvers for cosmological simulations
- Machine learning in astronomy (photometric redshifts, transient classification)
- Surrogate modeling for expensive simulations
- Physics-informed ML for differential equations

The combination of classical mechanics, statistical physics, and modern machine learning provides a powerful framework for understanding complex dynamical systems, preparing students for research at the intersection of physics and artificial intelligence.