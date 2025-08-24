# Project 5: Gaussian Processes for Astronomical Data (3 weeks)

## Project Philosophy & Motivation

After mastering parametric regression through MCMC, students are ready for **non-parametric** modeling. Gaussian Processes (GPs) represent a profound shift in thinking: instead of fitting parameters of a fixed model, we're learning the *function itself* from data, with full uncertainty quantification everywhere.

**The Motivating Problem**: "Your N-body simulations take hours to run. Your MCRT code takes days. But you need to explore thousands of parameter combinations. How can you predict simulation outcomes without running them all?"

## Learning Objectives

1. **Conceptual Understanding**
   - GPs as distributions over functions
   - The role of covariance functions (kernels)
   - Connection between Bayesian regression and GPs
   - Non-parametric vs parametric models

2. **Technical Skills**
   - Implement GP regression from scratch
   - Design and combine kernels for different problems
   - Optimize hyperparameters via marginal likelihood
   - Handle large datasets with computational tricks

3. **Astrophysical Applications**
   - Emulate expensive simulations
   - Interpolate irregular time series
   - Quantify uncertainties in predictions
   - Optimal experimental design

## Week 1: From Linear Regression to Gaussian Processes

### Day 1-2: Bridge from MCMC Project

**Start with Bayesian Linear Regression Review**
```python
# They just did this in MCMC project!
def bayesian_linear_regression(X, y, sigma_noise):
    """
    Linear model: y = Xw + ε
    Prior: w ~ N(0, σ_w²I)
    """
    # Posterior mean and covariance
    S_inv = (1/sigma_w**2) * np.eye(D) + (1/sigma_noise**2) * X.T @ X
    S = np.linalg.inv(S_inv)
    mean_w = (1/sigma_noise**2) * S @ X.T @ y
    
    # Predictive distribution
    mean_pred = X_test @ mean_w
    var_pred = sigma_noise**2 + np.diag(X_test @ S @ X_test.T)
    
    return mean_pred, var_pred
```

**The Kernel Trick - Key Insight**
```python
# Instead of features X, work directly with similarities K(x,x')
def polynomial_kernel(x1, x2, degree=2):
    """Simple polynomial kernel"""
    return (1 + x1.T @ x2)**degree

# Rewrite regression in kernel form
K = np.array([[polynomial_kernel(x[i], x[j]) for j in range(n)] 
              for i in range(n)])

# This is mathematically equivalent but more flexible!
```

### Day 3-4: Your First GP Implementation

**Core GP Regression**
```python
class GaussianProcess:
    def __init__(self, kernel_func, noise_variance=1e-6):
        self.kernel = kernel_func
        self.noise = noise_variance
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky decomposition for efficiency
        
    def fit(self, X, y):
        """Condition GP on observed data"""
        self.X_train = X
        self.y_train = y
        
        # Compute covariance matrix
        K = self.compute_kernel_matrix(X, X)
        K_y = K + self.noise * np.eye(len(X))
        
        # Cholesky decomposition for numerical stability
        self.L = np.linalg.cholesky(K_y)
        self.alpha = np.linalg.solve(self.L.T, 
                                     np.linalg.solve(self.L, y))
        
    def predict(self, X_test, return_std=True):
        """Predict mean and variance at test points"""
        # Prior covariance
        K_star = self.compute_kernel_matrix(X_test, self.X_train)
        K_star_star = self.compute_kernel_matrix(X_test, X_test)
        
        # Posterior mean
        mean = K_star @ self.alpha
        
        if return_std:
            # Posterior variance
            v = np.linalg.solve(self.L, K_star.T)
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 0))  # Numerical safety
            return mean, std
        
        return mean
```

### Day 5: Understanding Kernels

**Implement Common Kernels**
```python
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Radial Basis Function - smooth functions"""
    dist_sq = np.sum((x1 - x2)**2)
    return variance * np.exp(-dist_sq / (2 * length_scale**2))

def matern_kernel(x1, x2, length_scale=1.0, nu=1.5):
    """Matérn - controllable smoothness"""
    dist = np.sqrt(np.sum((x1 - x2)**2))
    if nu == 0.5:  # Exponential
        return np.exp(-dist / length_scale)
    elif nu == 1.5:
        z = np.sqrt(3) * dist / length_scale
        return (1 + z) * np.exp(-z)
    elif nu == 2.5:
        z = np.sqrt(5) * dist / length_scale
        return (1 + z + z**2/3) * np.exp(-z)

def periodic_kernel(x1, x2, period=1.0, length_scale=1.0):
    """For periodic signals like variable stars"""
    sine_dist = np.sin(np.pi * np.abs(x1 - x2) / period)
    return np.exp(-2 * sine_dist**2 / length_scale**2)
```

**Kernel Combination Rules**
```python
# Kernels can be combined!
def combined_kernel(x1, x2):
    # Long-term trend + periodic + noise
    trend = rbf_kernel(x1, x2, length_scale=10.0)
    periodic = periodic_kernel(x1, x2, period=1.0)
    local = rbf_kernel(x1, x2, length_scale=0.1)
    
    return trend + periodic * local  # Multiplicative seasonal
```

### Week 1 Deliverable: GP Regression on Toy Data
- Implement GP class from scratch
- Fit 1D functions with different kernels
- Visualize mean predictions with uncertainty bands
- Compare kernel choices on same dataset
- **Extension**: Implement kernel addition and multiplication

---

## Week 2: Astrophysical Applications

### Day 1-2: Time Series - Variable Stars and Transients

**Application: RR Lyrae Light Curves**
```python
def fit_rr_lyrae_lightcurve(times, mags, errors):
    """
    RR Lyrae have quasi-periodic variations
    Perfect for periodic + RBF kernel
    """
    # Custom kernel for RR Lyrae
    def rr_lyrae_kernel(t1, t2, params):
        period, decay_scale, local_scale = params
        
        # Quasi-periodic kernel
        periodic = periodic_kernel(t1, t2, period=period)
        decay = rbf_kernel(t1, t2, length_scale=decay_scale)
        local = matern_kernel(t1, t2, length_scale=local_scale, nu=1.5)
        
        return periodic * decay + local
    
    # Fit GP with optimized hyperparameters
    gp = GaussianProcess(rr_lyrae_kernel)
    gp.optimize_hyperparameters(times, mags, errors)
    
    # Predict on dense grid for smooth light curve
    t_dense = np.linspace(times.min(), times.max(), 1000)
    mag_pred, mag_std = gp.predict(t_dense)
    
    # Identify period and amplitude
    period = gp.kernel_params['period']
    amplitude = np.ptp(mag_pred)
    
    return mag_pred, mag_std, period, amplitude
```

**Application: Supernova Light Curve Interpolation**
```python
def interpolate_sn_lightcurve(mjd, flux, flux_err, bands):
    """
    Multi-band SN light curves
    Different bands correlated but offset
    """
    # Multi-output GP for all bands simultaneously
    # Shared kernel for common evolution
    # Band-specific kernels for offsets
    
    class MultiOutputGP:
        def __init__(self, n_outputs):
            self.base_kernel = rbf_kernel  # Shared evolution
            self.output_kernels = [rbf_kernel for _ in range(n_outputs)]
            
        def covariance(self, X1, X2):
            """ICM: Intrinsic Coregionalization Model"""
            # Build block covariance matrix
            # Captures correlations between bands
```

### Day 3-4: Emulating Expensive Simulations

**The Big Problem**: Connect to Previous Projects!

```python
def emulate_nbody_simulation():
    """
    N-body simulations are expensive!
    Train GP on subset, predict rest
    """
    
    # Run limited number of simulations
    parameters = []  # [M_cluster, r_core, W0, ...]
    results = []     # [t_relax, frac_escaped, r_half, ...]
    
    for i in range(n_training):
        params = sample_parameter_space()
        result = run_expensive_nbody(params)  # From Project 2!
        parameters.append(params)
        results.append(result)
    
    # Train GP emulator
    gp_emulator = GaussianProcess(kernel=matern_kernel)
    gp_emulator.fit(parameters, results)
    
    # Now predict for any parameter without running simulation!
    test_params = [1e4, 1.0, 3.0]  # M, r_core, W0
    predicted_evolution, uncertainty = gp_emulator.predict(test_params)
    
    return gp_emulator
```

**Active Learning: Where to Run Next Simulation?**
```python
def active_learning_acquisition(gp, parameter_bounds):
    """
    Decide where to run next expensive simulation
    Maximum uncertainty sampling
    """
    # Create grid of candidate parameters
    candidates = create_parameter_grid(parameter_bounds)
    
    # Predict uncertainty at each point
    _, uncertainties = gp.predict(candidates, return_std=True)
    
    # Choose point with maximum uncertainty
    next_simulation = candidates[np.argmax(uncertainties)]
    
    # Or use Upper Confidence Bound (UCB)
    mean, std = gp.predict(candidates, return_std=True)
    ucb = mean + 2.0 * std  # Exploration vs exploitation
    next_simulation = candidates[np.argmax(ucb)]
    
    return next_simulation
```

### Day 5: Hyperparameter Optimization

**Marginal Likelihood Maximization**
```python
def optimize_hyperparameters(self, X, y):
    """
    Find optimal kernel parameters by maximizing marginal likelihood
    p(y|X,θ) = ∫ p(y|X,w)p(w|θ)dw
    """
    
    def neg_log_marginal_likelihood(params):
        # Update kernel with new parameters
        self.kernel_params = params
        
        # Compute covariance
        K = self.compute_kernel_matrix(X, X)
        K_y = K + self.noise * np.eye(len(X))
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K_y)
        except:
            return 1e10  # Not positive definite
        
        # Log marginal likelihood
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        log_likelihood = -0.5 * y.T @ alpha
        log_likelihood -= np.sum(np.log(np.diag(L)))
        log_likelihood -= 0.5 * len(X) * np.log(2*np.pi)
        
        return -log_likelihood  # Minimize negative
    
    # Optimize with scipy
    from scipy.optimize import minimize
    result = minimize(neg_log_marginal_likelihood, 
                     initial_params,
                     bounds=parameter_bounds,
                     method='L-BFGS-B')
    
    self.kernel_params = result.x
```

### Week 2 Deliverable: Astronomical GP Application
- Fit real variable star data (provided RR Lyrae or Cepheid observations)
- Build emulator for one aspect of N-body or MCRT simulations
- Implement hyperparameter optimization
- Compare GP interpolation vs simple splines
- **Extension**: Multi-output GP for multi-band data

---

## Week 3: Advanced Topics & Computational Efficiency

### Day 1-2: Scaling to Large Datasets

**Problem**: GP is O(n³) - fails for n > 10,000!

**Solution 1: Sparse GPs with Inducing Points**
```python
class SparseGP:
    """
    Use m << n inducing points for approximation
    Reduces complexity from O(n³) to O(nm²)
    """
    def __init__(self, n_inducing=100):
        self.n_inducing = n_inducing
        self.Z = None  # Inducing point locations
        
    def fit(self, X, y):
        # Select inducing points (k-means, random, etc.)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_inducing)
        self.Z = kmeans.fit(X).cluster_centers_
        
        # Compute required matrices
        K_nm = self.kernel(X, self.Z)      # n x m
        K_mm = self.kernel(self.Z, self.Z)  # m x m
        
        # Efficient computation using Woodbury identity
        L_mm = np.linalg.cholesky(K_mm + 1e-6 * np.eye(m))
        A = np.linalg.solve(L_mm, K_nm.T)  # m x n
        
        # Effective covariance
        Q_nn = A.T @ A  # Low-rank approximation of K_nn
        Lambda = np.diag(np.diag(K_nn - Q_nn)) + self.noise * I
        
        # Predictions use only inducing points
        # O(m²n) instead of O(n³)
```

**Solution 2: Local GPs with Spatial Partitioning**
```python
def local_gp_prediction(X_train, y_train, x_test, n_neighbors=50):
    """
    Use only nearby points for each prediction
    Perfect for spatial data from N-body
    """
    # Find nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X_train)
    
    distances, indices = nbrs.kneighbors([x_test])
    
    # Local GP on subset
    local_X = X_train[indices[0]]
    local_y = y_train[indices[0]]
    
    local_gp = GaussianProcess(kernel=rbf_kernel)
    local_gp.fit(local_X, local_y)
    
    return local_gp.predict([x_test])
```

### Day 3-4: Connection to Neural Networks

**GP as Infinite Neural Network**
```python
def neural_network_kernel(x1, x2, depth=3):
    """
    Neal (1994): Infinite-width NN → GP
    This kernel corresponds to a deep NN
    """
    h1 = x1
    h2 = x2
    
    for layer in range(depth):
        # Recursive kernel computation
        K11 = np.dot(h1, h1) + 1
        K22 = np.dot(h2, h2) + 1
        K12 = np.dot(h1, h2) + 1
        
        # Through ReLU activation
        theta = np.arccos(K12 / np.sqrt(K11 * K22))
        
        # Next layer kernel
        h1 = np.sqrt(K11/(2*np.pi)) * (np.sin(theta) + 
                                       (np.pi - theta) * np.cos(theta))
        h2 = np.sqrt(K22/(2*np.pi)) * (np.sin(theta) + 
                                       (np.pi - theta) * np.cos(theta))
        
    return np.dot(h1, h2)

# This prepares students for the NN project!
```

**Connecting to Final Project**
```python
def gp_vs_nn_comparison():
    """
    Show when to use each approach
    """
    # Small data (n < 1000): GP wins
    # - Full uncertainty quantification
    # - No training required
    # - Optimal in Bayesian sense
    
    # Large data (n > 10000): NN wins  
    # - Scales to millions of points
    # - Learns features automatically
    # - Can be more flexible
    
    # Hybrid approach for final project:
    # Use NN to learn features, GP for uncertainty
```

### Day 5: Final Integration

**Bringing It All Together: Photometric Redshift Estimation**
```python
def photo_z_with_gp():
    """
    Combine everything learned:
    - Observations from MCRT understanding
    - Statistical methods from MCMC
    - Function learning with GP
    """
    
    # Training data: galaxies with known spec-z
    colors = compute_colors(magnitudes)  # From MCRT knowledge
    spec_z = known_redshifts
    
    # Design kernel for this problem
    def photo_z_kernel(c1, c2):
        # Colors evolve smoothly with redshift
        base = rbf_kernel(c1, c2, length_scale=0.5)
        
        # But with discontinuities at Balmer break
        break_kernel = sigmoid_kernel(c1, c2, location=3700)
        
        return base * break_kernel
    
    # Train GP
    gp = GaussianProcess(kernel=photo_z_kernel)
    gp.fit(colors, spec_z)
    
    # Predict with uncertainties
    photo_z, photo_z_err = gp.predict(test_colors)
    
    # Compare to template fitting (from MCMC project)
    return photo_z, photo_z_err
```

### Week 3 Deliverable: Scaling & Integration
- Implement sparse GP or local GP for large dataset
- Apply to real astronomical survey data
- Compare GP vs simple interpolation on speed/accuracy
- Final presentation connecting all projects
- **Extension**: Implement deep kernel learning

---

## Assessment Rubric

### Code Implementation (40%)
- [ ] Correct GP regression from scratch (15%)
- [ ] Multiple kernel implementations (10%)
- [ ] Hyperparameter optimization works (10%)
- [ ] Efficient scaling solution implemented (5%)

### Physical Understanding (30%)
- [ ] Appropriate kernel choice with justification (10%)
- [ ] Correct interpretation of uncertainties (10%)
- [ ] Clear connection to previous projects (10%)

### Analysis & Visualization (20%)
- [ ] Clear uncertainty visualization (10%)
- [ ] Comparison plots (GP vs alternatives) (5%)
- [ ] Performance metrics reported (5%)

### Extension & Creativity (10%)
- [ ] Goes beyond minimum requirements
- [ ] Novel application or investigation
- [ ] Exceptional insight or discovery

---

## Provided Resources

### Datasets
```python
# Week 1: Toy datasets
- 1D regression problems with varying noise
- Mauna Loa CO2 (classic GP dataset)

# Week 2: Astronomical data  
- RR Lyrae from OGLE or Gaia
- Supernova light curves from Open Supernova Catalog
- Subset of N-body simulation results

# Week 3: Large datasets
- SDSS photometric sample (100k+ galaxies)
- Gaia variable star catalog
- Pre-computed simulation grid
```

### Starter Code
```python
# kernel_library.py - Basic kernel implementations
# gp_utils.py - Plotting functions, data loaders
# test_gp.py - Unit tests for their implementation
```

### Key Papers
1. Rasmussen & Williams (2006) - "Gaussian Processes for Machine Learning" (free online)
2. Gibson et al. (2012) - "GPs in Exoplanet Transit Fitting"
3. Foreman-Mackey et al. (2017) - "Fast GPs for Stellar Variability"

---

## Connection to Course Narrative

### Looking Back
- **From N-body**: "Your simulations are expensive. Can we predict without running them?"
- **From MCRT**: "How do we interpolate sparse spectral observations?"
- **From MCMC**: "We learned parameters. Now learn functions!"

### Looking Forward
- **To Neural Networks**: "GPs are beautiful but don't scale. What if we could learn features automatically?"
- **The Bridge**: GPs are NNs with infinite width. Students ready for deep learning!

### The Meta-Learning
Students realize they've been building toward this all along:
1. **Deterministic** (N-body): Fixed equations, numerical solutions
2. **Stochastic** (MCRT): Random sampling for complex integrals  
3. **Parametric** (MCMC): Learn fixed number of parameters
4. **Non-parametric** (GP): Learn entire functions
5. **Feature Learning** (NN): Learn representations automatically

---

## Common Pitfalls & Solutions

### Numerical Issues
```python
# Problem: Covariance matrix not positive definite
# Solution: Add jitter
K = K + 1e-6 * np.eye(n)

# Problem: Overflow in exponentials
# Solution: Work in log space
log_likelihood = -0.5 * log_det_K - 0.5 * y.T @ K_inv @ y

# Problem: Slow matrix inversions
# Solution: Use Cholesky decomposition
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
```

### Conceptual Confusions
- "Why not just use splines?" → Show uncertainty quantification
- "This seems like overkill" → Demo on sparse, noisy data
- "How is this different from regression?" → It IS regression, just infinite-dimensional!

---

## Expected Outcomes

By the end of this project, students will:
1. **Understand** GPs as distributions over functions
2. **Implement** efficient GP regression with various kernels
3. **Apply** GPs to real astronomical problems
4. **Connect** GPs to both previous work and upcoming neural networks
5. **Appreciate** when to use GPs vs other methods
6. **Master** uncertainty quantification in predictions

Most importantly, they'll see that the same mathematical framework (covariances, optimization, linear algebra) appears throughout computational astrophysics, preparing them for the neural network capstone project.