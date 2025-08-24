# ASTR 596 Project 2: N-Body Dynamics + Statistical Sampling + Stellar Systems
**Duration**: 3 weeks
**Weight**: 15% of course grade
**Theme**: "Realistic Stellar Clusters with Gravitational Dynamics"

---

## Project Overview

This project builds sophisticated N-body gravitational dynamics simulations with realistic stellar populations. You will implement multiple ODE integration schemes, master statistical sampling from astrophysical distributions, and create evolving stellar clusters that serve as input for radiation calculations in Project 3. The emphasis is on vectorization, performance optimization, and adaptive numerical methods.

## Learning Objectives

By completing this project, you will:
- **Master ODE integration**: Implement and compare multiple numerical integration schemes
- **Understand gravitational dynamics**: N-body physics, energy conservation, and cluster evolution
- **Learn statistical sampling**: Sample from Initial Mass Function and spatial distributions
- **Develop vectorization skills**: Efficient NumPy operations for computational performance
- **Implement adaptive methods**: Energy-controlled timestep adjustment
- **Generate realistic astrophysical data**: Stellar clusters for radiation modeling

## Prerequisites from Project 1
- Numerical integration techniques (trapezoid, Simpson's, Gaussian quadrature)
- Root-finding methods (Newton-Raphson for energy balance)
- Object-oriented programming (Star class design)
- Blackbody physics and stellar luminosity calculations

---

# Week 1: ODE Solvers and Energy Conservation

## Conceptual Introduction (25 min)
- Newton's laws and gravitational force in astrophysical contexts
- Converting 2nd order ODEs to 1st order systems
- Integration methods: explicit vs implicit, stability vs accuracy
- Symplectic integrators for Hamiltonian systems
- Energy and angular momentum conservation in gravitational systems

## Lab Session Objectives
Build comprehensive ODE solver library and validate on two-body dynamics.

### Task 1: ODE Solver Framework (45 min)
**Goal**: Create abstract base class and implement multiple integration methods

**Framework Design**:
```python
from abc import ABC, abstractmethod
import numpy as np

class ODESolver(ABC):
    """
    Abstract base class for ODE integration methods.
    
    Solves system: dy/dt = f(t, y) where y can be vector-valued
    """
    
    def __init__(self, derivatives_func, initial_conditions, initial_time=0.0):
        """
        Parameters:
        -----------
        derivatives_func : callable
            Function f(t, y) returning dy/dt
        initial_conditions : array_like
            Initial values y(t0)
        initial_time : float
            Initial time t0
        """
        self.f = derivatives_func
        self.y = np.array(initial_conditions, dtype=float)
        self.t = initial_time
        self.history = {'t': [initial_time], 'y': [self.y.copy()]}
    
    @abstractmethod
    def step(self, dt):
        """Take single integration step of size dt."""
        pass
    
    def evolve(self, t_final, dt):
        """Evolve system from current time to t_final."""
        while self.t < t_final:
            step_size = min(dt, t_final - self.t)
            self.step(step_size)
            self.history['t'].append(self.t)
            self.history['y'].append(self.y.copy())
        return np.array(self.history['t']), np.array(self.history['y'])

class EulerSolver(ODESolver):
    """First-order Euler method: y_{n+1} = y_n + dt * f(t_n, y_n)"""
    
    def step(self, dt):
        """Implement Euler step."""
        dydt = self.f(self.t, self.y)
        self.y += dt * dydt
        self.t += dt

class RungeKutta4Solver(ODESolver):
    """Fourth-order Runge-Kutta method."""
    
    def step(self, dt):
        """Implement RK4 step with four evaluations."""
        k1 = self.f(self.t, self.y)
        k2 = self.f(self.t + dt/2, self.y + dt*k1/2)
        k3 = self.f(self.t + dt/2, self.y + dt*k2/2)
        k4 = self.f(self.t + dt, self.y + dt*k3)
        
        self.y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        self.t += dt

class LeapfrogSolver(ODESolver):
    """
    Leapfrog integrator for Hamiltonian systems.
    Particularly good for gravitational dynamics.
    """
    
    def __init__(self, force_func, positions, velocities, masses, initial_time=0.0):
        """
        Specialized for N-body problems.
        
        Parameters:
        -----------
        force_func : callable
            Function returning accelerations given (positions, masses)
        positions : array
            Initial positions [N, 3]
        velocities : array  
            Initial velocities [N, 3]
        masses : array
            Particle masses [N]
        """
        self.force_func = force_func
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.masses = np.array(masses)
        self.t = initial_time
        self.history = {
            't': [initial_time],
            'positions': [self.positions.copy()],
            'velocities': [self.velocities.copy()]
        }
    
    def step(self, dt):
        """Leapfrog integration step."""
        # Kick: v_{1/2} = v_0 + (dt/2) * a_0
        accelerations = self.force_func(self.positions, self.masses)
        self.velocities += 0.5 * dt * accelerations
        
        # Drift: x_1 = x_0 + dt * v_{1/2}
        self.positions += dt * self.velocities
        
        # Kick: v_1 = v_{1/2} + (dt/2) * a_1
        accelerations = self.force_func(self.positions, self.masses)
        self.velocities += 0.5 * dt * accelerations
        
        self.t += dt
        self.history['t'].append(self.t)
        self.history['positions'].append(self.positions.copy())
        self.history['velocities'].append(self.velocities.copy())
```

### Task 2: Two-Body Gravitational Dynamics (60 min)
**Goal**: Validate integrators on Kepler problem with known analytical solution

**Implementation Requirements**:
```python
def gravitational_derivatives(t, state):
    """
    Derivatives for two-body problem.
    
    state = [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
    """
    # Extract positions and velocities
    pos1 = state[0:3]
    vel1 = state[3:6]
    pos2 = state[6:9]
    vel2 = state[9:12]
    
    # Calculate separation and force
    r_vec = pos2 - pos1
    r_mag = np.linalg.norm(r_vec)
    
    # Gravitational acceleration
    G = 6.674e-11  # m³/kg/s²
    m1, m2 = 1.0, 1.0  # masses
    
    acc_magnitude = G * (m1 + m2) / r_mag**3
    acc1 = acc_magnitude * r_vec
    acc2 = -acc_magnitude * r_vec
    
    # Return derivatives: [vel1, acc1, vel2, acc2]
    return np.concatenate([vel1, acc1, vel2, acc2])

def kepler_orbit_validation():
    """
    Test integrators on Earth-Sun system.
    Compare with analytical solution for energy and angular momentum.
    """
    # Earth-Sun system (simplified units)
    AU = 1.496e11  # m
    year = 365.25 * 24 * 3600  # s
    
    # Initial conditions: Earth at aphelion
    initial_state = [
        1.017*AU, 0, 0,      # Earth position
        0, 29.29e3, 0,       # Earth velocity
        0, 0, 0,             # Sun position (at origin)
        0, 0, 0              # Sun velocity
    ]
    
    # Test each integrator
    methods = {
        'Euler': EulerSolver,
        'RK4': RungeKutta4Solver
    }
    
    results = {}
    for name, SolverClass in methods.items():
        solver = SolverClass(gravitational_derivatives, initial_state)
        t_vals, y_vals = solver.evolve(t_final=year, dt=year/1000)
        results[name] = {'t': t_vals, 'y': y_vals}
    
    return results

def calculate_orbital_energy(positions, velocities, masses):
    """Calculate total energy: kinetic + potential."""
    # Kinetic energy: (1/2) * m * v²
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    # Potential energy: -G * m1 * m2 / r
    G = 6.674e-11
    pe = 0
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            pe -= G * masses[i] * masses[j] / r_ij
    
    return ke + pe

def orbital_validation_analysis(results):
    """
    Analyze energy conservation and orbital accuracy.
    Plot energy drift and orbital trajectories.
    """
    # Calculate energy conservation for each method
    # Plot trajectories and energy vs time
    # Compare with analytical orbital period
```

### Task 3: Error Analysis and Method Comparison (30 min)
**Goal**: Understand trade-offs between accuracy, stability, and computational cost

**Analysis Requirements**:
1. **Convergence Study**: Plot error vs timestep for each method
2. **Energy Conservation**: Track relative energy drift over multiple orbits
3. **Computational Cost**: Time each method for various timestep sizes
4. **Long-term Stability**: Run for 10+ orbital periods

**Week 1 Deliverable**: ODE solver library with comprehensive validation on Kepler orbits

---

# Week 2: Statistical Sampling and Multi-Body Systems

## Conceptual Introduction (25 min)
- Stellar Initial Mass Function: Salpeter, Kroupa, Chabrier prescriptions
- Spatial distributions in star clusters: Plummer sphere, King profiles
- Statistical sampling techniques: inverse transform, rejection sampling
- Virial equilibrium and cluster dynamics

## Lab Session Objectives
Implement realistic stellar cluster initialization and scale to many-body systems.

### Task 1: Initial Mass Function Implementation (50 min)
**Goal**: Sample realistic stellar mass distributions

**IMF Theory and Implementation**:
```python
class StellarIMF:
    """
    Stellar Initial Mass Function implementation.
    
    Supports multiple functional forms used in astrophysics.
    """
    
    def __init__(self, imf_type='kroupa', mass_range=(0.08, 120)):
        """
        Parameters:
        -----------
        imf_type : str
            'salpeter', 'kroupa', or 'chabrier'
        mass_range : tuple
            (minimum_mass, maximum_mass) in solar masses
        """
        self.imf_type = imf_type
        self.m_min, self.m_max = mass_range
        self.normalization = self._calculate_normalization()
    
    def pdf(self, mass):
        """
        Probability density function dN/dM.
        
        Salpeter (1955): dN/dM ∝ M^(-2.35)
        Kroupa (2001): dN/dM ∝ M^(-1.3) for M < 0.5 M☉
                               M^(-2.3) for M > 0.5 M☉
        """
        mass = np.asarray(mass)
        
        if self.imf_type == 'salpeter':
            return mass**(-2.35)
        
        elif self.imf_type == 'kroupa':
            # Broken power law
            result = np.zeros_like(mass)
            low_mass = mass < 0.5
            high_mass = mass >= 0.5
            
            result[low_mass] = mass[low_mass]**(-1.3)
            # Ensure continuity at M = 0.5
            normalization = 0.5**(-1.3 + 2.3)
            result[high_mass] = normalization * mass[high_mass]**(-2.3)
            
            return result
        
        elif self.imf_type == 'chabrier':
            # Log-normal for low masses + power law for high masses
            # Implementation left as advanced exercise
            pass
    
    def cdf(self, mass):
        """Cumulative distribution function."""
        # Analytical when possible, numerical integration otherwise
        if self.imf_type == 'salpeter':
            # CDF ∝ M^(-1.35)
            return (mass**(-1.35) - self.m_min**(-1.35)) / \
                   (self.m_max**(-1.35) - self.m_min**(-1.35))
    
    def sample_rejection(self, n_stars):
        """Sample using rejection method."""
        masses = []
        max_pdf = self.pdf(self.m_min)  # Maximum of PDF
        
        while len(masses) < n_stars:
            # Propose random mass in range
            m_proposal = self.m_min + (self.m_max - self.m_min) * np.random.random()
            
            # Accept with probability proportional to PDF
            if np.random.random() < self.pdf(m_proposal) / max_pdf:
                masses.append(m_proposal)
        
        return np.array(masses)
    
    def sample_inverse_transform(self, n_stars):
        """Sample using inverse CDF (when available)."""
        if self.imf_type == 'salpeter':
            u = np.random.random(n_stars)
            # Invert CDF analytically
            return (self.m_min**(-1.35) + u * (self.m_max**(-1.35) - self.m_min**(-1.35)))**(-1/1.35)
        else:
            # Fall back to rejection sampling
            return self.sample_rejection(n_stars)
    
    def validate_distribution(self, masses, n_bins=50):
        """Compare sampled masses with theoretical IMF."""
        # Create histogram and compare with PDF
        # Plot and calculate goodness-of-fit statistics
        pass

def mass_to_stellar_properties(masses):
    """
    Convert stellar masses to observable properties.
    Uses Project 1 stellar physics relationships.
    """
    # Mass-luminosity relation
    luminosities = np.where(masses > 1.0, 
                           masses**3.5,  # High mass: L ∝ M^3.5
                           masses**4.0)  # Low mass: L ∝ M^4.0
    
    # Mass-temperature relation (main sequence)
    temperatures = 5778 * (masses)**0.5  # Rough approximation
    
    # Mass-radius relation
    radii = np.where(masses > 1.0,
                    masses**0.8,   # High mass
                    masses**0.9)   # Low mass
    
    return {
        'luminosities': luminosities,
        'temperatures': temperatures,
        'radii': radii
    }
```

### Task 2: Plummer Sphere Spatial Distribution (45 min)
**Goal**: Sample realistic 3D stellar cluster geometry

**Plummer Model Implementation**:
```python
class PlummerSphere:
    """
    Plummer sphere model for stellar cluster spatial distribution.
    
    Density profile: ρ(r) = (3M/4πa³) * (1 + r²/a²)^(-5/2)
    where a is the scale radius.
    """
    
    def __init__(self, total_mass=1000, scale_radius=1.0):
        """
        Parameters:
        -----------
        total_mass : float
            Total cluster mass [M☉]
        scale_radius : float
            Plummer scale radius [pc]
        """
        self.M = total_mass
        self.a = scale_radius
    
    def density(self, r):
        """Density at radius r."""
        return (3*self.M/(4*np.pi*self.a**3)) * (1 + (r/self.a)**2)**(-5/2)
    
    def mass_enclosed(self, r):
        """Mass within radius r."""
        return self.M * (r/self.a)**3 / (1 + (r/self.a)**2)**(3/2)
    
    def sample_radial_positions(self, n_stars):
        """
        Sample radial distances using inverse CDF method.
        
        CDF: M(r)/M_total = (r/a)³ / (1 + (r/a)²)^(3/2)
        Inverse: r = a / sqrt(u^(-2/3) - 1)
        """
        u = np.random.random(n_stars)
        # Prevent u=0 which gives infinite radius
        u = np.clip(u, 1e-10, 1-1e-10)
        
        radii = self.a / np.sqrt(u**(-2/3) - 1)
        return radii
    
    def sample_positions(self, n_stars):
        """Sample 3D positions from Plummer distribution."""
        radii = self.sample_radial_positions(n_stars)
        
        # Sample isotropic directions
        cos_theta = 2*np.random.random(n_stars) - 1  # cos(θ) uniform in [-1,1]
        phi = 2*np.pi*np.random.random(n_stars)      # φ uniform in [0,2π]
        
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Convert to Cartesian coordinates
        x = radii * sin_theta * np.cos(phi)
        y = radii * sin_theta * np.sin(phi)
        z = radii * cos_theta
        
        return np.column_stack([x, y, z])
    
    def calculate_virial_velocities(self, positions, masses):
        """
        Calculate velocities for virial equilibrium.
        
        Uses virial theorem: 2T + U = 0 for bound system
        where T = kinetic energy, U = potential energy
        """
        n_stars = len(masses)
        velocities = np.zeros_like(positions)
        
        # Calculate potential energy
        U = 0
        for i in range(n_stars):
            for j in range(i+1, n_stars):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                U -= G * masses[i] * masses[j] / r_ij
        
        # Virial theorem: total kinetic energy = -U/2
        T_total = -U / 2
        
        # Distribute kinetic energy among particles
        # Simple approach: assume isotropic velocity dispersion
        for i in range(n_stars):
            # Individual kinetic energy proportional to mass
            T_i = T_total * masses[i] / np.sum(masses)
            v_mag = np.sqrt(2 * T_i / masses[i])
            
            # Random direction
            cos_theta = 2*np.random.random() - 1
            phi = 2*np.pi*np.random.random()
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            velocities[i] = v_mag * np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta
            ])
        
        return velocities
```

### Task 3: Vectorized N-Body Force Calculation (40 min)
**Goal**: Implement efficient O(N²) force computation

**Vectorized Implementation**:
```python
def gravitational_forces_vectorized(positions, masses, softening=0.01):
    """
    Calculate gravitational forces between all particle pairs.
    
    Parameters:
    -----------
    positions : array [N, 3]
        Particle positions
    masses : array [N]
        Particle masses
    softening : float
        Softening parameter to avoid singularities
        
    Returns:
    --------
    forces : array [N, 3]
        Gravitational forces on each particle
    """
    N = len(masses)
    G = 4.3e-3  # pc³/M☉/Myr² (convenient units)
    
    # Calculate all pairwise separations using broadcasting
    # positions[i,j] - positions[k,j] for all i,k pairs
    r_vectors = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # [N, N, 3]
    
    # Distance magnitudes with softening
    r_magnitudes = np.sqrt(np.sum(r_vectors**2, axis=2) + softening**2)  # [N, N]
    
    # Avoid self-interaction
    np.fill_diagonal(r_magnitudes, np.inf)
    
    # Force magnitudes: F = G*m1*m2/r²
    mass_products = masses[:, np.newaxis] * masses[np.newaxis, :]  # [N, N]
    force_magnitudes = G * mass_products / r_magnitudes**2  # [N, N]
    
    # Force directions: unit vectors
    r_unit = r_vectors / r_magnitudes[:, :, np.newaxis]  # [N, N, 3]
    
    # Total forces: sum over all other particles
    forces = np.sum(force_magnitudes[:, :, np.newaxis] * r_unit, axis=1)  # [N, 3]
    
    return forces

def performance_comparison():
    """Compare vectorized vs nested loop implementations."""
    import time
    
    # Test different cluster sizes
    N_values = [10, 50, 100, 200, 500]
    
    for N in N_values:
        # Generate test data
        positions = np.random.randn(N, 3)
        masses = np.random.uniform(0.1, 10, N)
        
        # Time vectorized version
        start = time.time()
        forces_vec = gravitational_forces_vectorized(positions, masses)
        time_vec = time.time() - start
        
        # Time nested loop version (for comparison)
        start = time.time()
        forces_loop = gravitational_forces_nested_loops(positions, masses)
        time_loop = time.time() - start
        
        print(f"N={N}: Vectorized={time_vec:.4f}s, Loops={time_loop:.4f}s, "
              f"Speedup={time_loop/time_vec:.1f}x")
```

**Week 2 Deliverable**: Realistic stellar cluster initialization with IMF masses and Plummer positions, plus efficient force calculations

---

# Week 3: Adaptive Timestepping and Cluster Evolution

## Conceptual Introduction (25 min)
- Energy conservation as accuracy criterion
- Adaptive timestep algorithms
- Multi-mass cluster dynamics: mass segregation, two-body relaxation
- Stellar escape and cluster dissolution
- Computational complexity and optimization strategies

## Lab Session Objectives
Implement energy-controlled adaptive integration and study realistic cluster evolution.

### Task 1: Adaptive Timestep Control (50 min)
**Goal**: Implement robust adaptive timestep algorithm based on energy conservation

**Adaptive Integration Framework**:
```python
class AdaptiveNBodySimulator:
    """
    N-body simulator with adaptive timestep control.
    
    Uses energy conservation to monitor accuracy and adjust timestep.
    """
    
    def __init__(self, positions, velocities, masses, initial_dt=0.01, 
                 energy_tolerance=1e-6):
        """
        Parameters:
        -----------
        positions : array [N, 3]
            Initial positions [pc]
        velocities : array [N, 3] 
            Initial velocities [km/s]
        masses : array [N]
            Particle masses [M☉]
        initial_dt : float
            Initial timestep [Myr]
        energy_tolerance : float
            Relative energy error tolerance
        """
        self.positions = np.array(positions)
        self.velocities = np.array(velocities) 
        self.masses = np.array(masses)
        self.dt = initial_dt
        self.tolerance = energy_tolerance
        
        # Calculate initial energy
        self.initial_energy = self.total_energy()
        
        # Statistics tracking
        self.n_accepted = 0
        self.n_rejected = 0
        self.energy_errors = []
        self.timesteps = []
        
        # History storage
        self.time = 0.0
        self.history = {
            'time': [0.0],
            'positions': [self.positions.copy()],
            'velocities': [self.velocities.copy()],
            'energy': [self.initial_energy],
            'timestep': [self.dt]
        }
    
    def total_energy(self):
        """Calculate total energy: kinetic + potential."""
        # Kinetic energy
        ke = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        
        # Potential energy
        pe = 0
        for i in range(len(self.masses)):
            for j in range(i+1, len(self.masses)):
                r_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                pe -= G * self.masses[i] * self.masses[j] / r_ij
        
        return ke + pe
    
    def energy_error(self):
        """Calculate relative energy error from initial value."""
        current_energy = self.total_energy()
        return abs((current_energy - self.initial_energy) / self.initial_energy)
    
    def leapfrog_step(self, dt):
        """Take single leapfrog integration step."""
        # Store initial state for potential rollback
        old_positions = self.positions.copy()
        old_velocities = self.velocities.copy()
        
        # Leapfrog integration
        forces = gravitational_forces_vectorized(self.positions, self.masses)
        accelerations = forces / self.masses[:, np.newaxis]
        
        # Kick-drift-kick
        self.velocities += 0.5 * dt * accelerations
        self.positions += dt * self.velocities
        
        forces = gravitational_forces_vectorized(self.positions, self.masses)
        accelerations = forces / self.masses[:, np.newaxis]
        self.velocities += 0.5 * dt * accelerations
        
        return old_positions, old_velocities
    
    def adaptive_step(self):
        """
        Take adaptive timestep with error control.
        
        Algorithm:
        1. Attempt step with current timestep
        2. Check energy conservation
        3. If error too large: reduce timestep and retry
        4. If error acceptable: possibly increase timestep for next step
        """
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Store state before step
            old_positions, old_velocities = self.leapfrog_step(self.dt)
            
            # Check energy conservation
            error = self.energy_error()
            
            if error <= self.tolerance:
                # Step accepted
                self.time += self.dt
                self.n_accepted += 1
                
                # Store results
                self.history['time'].append(self.time)
                self.history['positions'].append(self.positions.copy())
                self.history['velocities'].append(self.velocities.copy())
                self.history['energy'].append(self.total_energy())
                self.history['timestep'].append(self.dt)
                
                self.energy_errors.append(error)
                self.timesteps.append(self.dt)
                
                # Possibly increase timestep for next step
                if error < self.tolerance / 10:
                    self.dt = min(self.dt * 1.1, 0.1)  # Don't let it grow too large
                
                return True
            
            else:
                # Step rejected - restore state and reduce timestep
                self.positions = old_positions
                self.velocities = old_velocities
                self.dt *= 0.5
                self.n_rejected += 1
                
                if attempt == max_attempts - 1:
                    print(f"Warning: Max attempts reached at t={self.time:.3f}")
                    return False
        
        return False
    
    def evolve(self, t_final, max_steps=10000):
        """Evolve system to final time using adaptive timesteps."""
        step_count = 0
        
        while self.time < t_final and step_count < max_steps:
            success = self.adaptive_step()
            if not success:
                print("Simulation failed - energy errors too large")
                break
            
            step_count += 1
            
            # Progress reporting
            if step_count % 100 == 0:
                acceptance_rate = self.n_accepted / (self.n_accepted + self.n_rejected)
                print(f"t={self.time:.2f}, dt={self.dt:.4f}, "
                      f"E_error={self.energy_errors[-1]:.2e}, "
                      f"acceptance={acceptance_rate:.2f}")
        
        return self.get_results()
    
    def get_results(self):
        """Return simulation results as arrays."""
        return {
            'time': np.array(self.history['time']),
            'positions': np.array(self.history['positions']),
            'velocities': np.array(self.history['velocities']),
            'energy': np.array(self.history['energy']),
            'timesteps': np.array(self.history['timestep'])
        }
```

### Task 2: Cluster Physics and Evolution (55 min)
**Goal**: Study realistic stellar cluster evolution phenomena

**Mass Segregation Analysis**:
```python
def analyze_mass_segregation(positions, masses, times):
    """
    Track mass segregation: massive stars sink to cluster center.
    
    Quantify using mass-weighted radial distribution.
    """
    segregation_ratios = []
    
    for i, pos in enumerate(positions):
        # Calculate distance from cluster center
        center = np.average(pos, weights=masses, axis=0)
        distances = np.linalg.norm(pos - center, axis=1)
        
        # Sort by mass
        mass_order = np.argsort(masses)[::-1]  # Heaviest first
        
        # Compare radial distribution of most vs least massive stars
        n_heavy = len(masses) // 10  # Top 10%
        n_light = len(masses) // 10  # Bottom 10%
        
        r_heavy = np.mean(distances[mass_order[:n_heavy]])
        r_light = np.mean(distances[mass_order[-n_light:]])
        
        segregation_ratios.append(r_light / r_heavy)
    
    return segregation_ratios

def calculate_virial_ratio(positions, velocities, masses):
    """
    Calculate virial ratio: 2T/|U|
    
    For bound system in equilibrium, should equal 1.
    """
    # Kinetic energy
    T = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    
    # Potential energy
    U = 0
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            U -= G * masses[i] * masses[j] / r_ij
    
    return 2 * T / abs(U)

def identify_escaping_stars(positions, velocities, masses, escape_criterion=2.0):
    """
    Identify stars with velocities exceeding escape velocity.
    
    v_escape = sqrt(2 * |U| / m) at each star's location
    """
    escaping_stars = []
    
    for i in range(len(masses)):
        # Calculate potential at star i due to all other stars
        phi_i = 0
        for j in range(len(masses)):
            if i != j:
                r_ij = np.linalg.norm(positions[i] - positions[j])
                phi_i -= G * masses[j] / r_ij
        
        # Escape velocity at this location
        v_escape = np.sqrt(-2 * phi_i)
        v_star = np.linalg.norm(velocities[i])
        
        if v_star > escape_criterion * v_escape:
            escaping_stars.append(i)
    
    return escaping_stars
```

**Cluster Snapshot Generation for Project 3**:
```python
def generate_cluster_snapshots(cluster_mass=1000, n_stars=200, 
                              evolution_times=[0, 5, 20, 50]):
    """
    Generate stellar cluster at multiple evolutionary phases.
    These snapshots will be used in Project 3 for radiation calculations.
    
    Parameters:
    -----------
    cluster_mass : float
        Total cluster mass [M☉]
    n_stars : int
        Number of stars in cluster
    evolution_times : list
        Times to save snapshots [Myr]
        
    Returns:
    --------
    snapshots : list of dict
        Each dict contains stellar properties at one time
    """
    # Initialize cluster
    imf = StellarIMF(imf_type='kroupa')
    masses = imf.sample_inverse_transform(n_stars)
    masses = masses * (cluster_mass / np.sum(masses))  # Normalize total mass
    
    plummer = PlummerSphere(total_mass=cluster_mass, scale_radius=1.0)
    positions = plummer.sample_positions(n_stars)
    velocities = plummer.calculate_virial_velocities(positions, masses)
    
    # Calculate stellar properties for radiation (from Project 1)
    stellar_props = mass_to_stellar_properties(masses)
    
    # Set up adaptive simulator
    simulator = AdaptiveNBodySimulator(
        positions, velocities, masses,
        initial_dt=0.01, energy_tolerance=1e-6
    )
    
    snapshots = []
    
    for t_target in evolution_times:
        if t_target == 0:
            # Initial conditions
            snapshot = create_snapshot(
                time=0, 
                positions=simulator.positions,
                velocities=simulator.velocities,
                masses=masses,
                stellar_props=stellar_props
            )
        else:
            # Evolve to target time
            results = simulator.evolve(t_target)
            
            # Extract final state
            final_positions = results['positions'][-1]
            final_velocities = results['velocities'][-1]
            
            snapshot = create_snapshot(
                time=t_target,
                positions=final_positions,
                velocities=final_velocities,
                masses=masses,
                stellar_props=stellar_props
            )
        
        snapshots.append(snapshot)
        print(f"Snapshot created at t = {t_target} Myr")
    
    return snapshots

def create_snapshot(time, positions, velocities, masses, stellar_props):
    """Create comprehensive cluster snapshot."""
    # Calculate cluster center and properties
    center = np.average(positions, weights=masses, axis=0)
    centered_positions = positions - center
    
    # Structural parameters
    distances = np.linalg.norm(centered_positions, axis=1)
    half_mass_radius = np.median(distances)
    
    # Core radius (radius containing 10% of mass)
    mass_order = np.argsort(distances)
    core_mass_index = int(0.1 * len(masses))
    core_radius = distances[mass_order[core_mass_index]]
    
    snapshot = {
        'time': time,
        'n_stars': len(masses),
        'total_mass': np.sum(masses),
        
        # Stellar properties
        'positions': centered_positions,  # Centered on cluster
        'velocities': velocities,
        'masses': masses,
        'luminosities': stellar_props['luminosities'],
        'temperatures': stellar_props['temperatures'],
        'radii': stellar_props['radii'],
        
        # Cluster structure
        'center': center,
        'half_mass_radius': half_mass_radius,
        'core_radius': core_radius,
        'virial_ratio': calculate_virial_ratio(positions, velocities, masses),
        
        # Evolution diagnostics
        'mass_segregation_ratio': analyze_mass_segregation([positions], masses, [time])[0],
        'escaping_stars': identify_escaping_stars(positions, velocities, masses)
    }
    
    return snapshot

def save_snapshots_for_project3(snapshots, filename='cluster_evolution.pkl'):
    """Save snapshots in format suitable for Project 3."""
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(snapshots, f)
    
    print(f"Saved {len(snapshots)} cluster snapshots to {filename}")
    print("These will be used as radiation sources in Project 3")
```

### Task 3: Performance Analysis and Optimization (30 min)
**Goal**: Analyze computational efficiency and identify optimization opportunities

**Performance Studies**:
```python
def scaling_analysis():
    """Study how computational cost scales with cluster size."""
    import time
    
    N_values = [50, 100, 200, 400]
    times_force = []
    times_integration = []
    
    for N in N_values:
        # Generate test cluster
        masses = np.random.uniform(0.1, 10, N)
        positions = np.random.randn(N, 3)
        velocities = np.random.randn(N, 3)
        
        # Time force calculation
        start = time.time()
        for _ in range(10):  # Multiple iterations for averaging
            forces = gravitational_forces_vectorized(positions, masses)
        times_force.append((time.time() - start) / 10)
        
        # Time full integration step
        simulator = AdaptiveNBodySimulator(positions, velocities, masses)
        start = time.time()
        for _ in range(10):
            simulator.adaptive_step()
        times_integration.append((time.time() - start) / 10)
    
    # Analyze scaling: should be O(N²) for force calculation
    print("Scaling Analysis:")
    for i, N in enumerate(N_values):
        print(f"N={N}: Force={times_force[i]:.4f}s, Integration={times_integration[i]:.4f}s")

def memory_optimization_analysis():
    """Analyze memory usage and suggest optimizations."""
    # Profile memory usage during simulation
    # Identify opportunities for optimization
    pass
```

**Week 3 Deliverable**: Complete adaptive N-body simulator with realistic cluster evolution and snapshots for Project 3

---

# Assessment and Grading

## Grading Breakdown
- **Week 1**: ODE solvers and validation (30%)
- **Week 2**: Statistical sampling and vectorization (35%)
- **Week 3**: Adaptive methods and cluster evolution (35%)

## Evaluation Criteria

### Technical Implementation (60%)
- **Algorithm Correctness**: Do integrators conserve energy appropriately?
- **Sampling Accuracy**: Do distributions match theoretical expectations?
- **Vectorization Efficiency**: Significant speedup over naive implementations
- **Adaptive Control**: Proper timestep adjustment based on energy errors

### Scientific Understanding (25%)
- **Physics Validation**: Energy conservation, virial equilibrium, orbital mechanics
- **Statistical Analysis**: IMF and spatial distribution validation
- **Cluster Evolution**: Understanding of mass segregation and stellar escape

### Code Quality and Performance (15%)
- **Documentation**: Clear docstrings and code organization
- **Testing**: Validation against analytical solutions
- **Optimization**: Efficient use of NumPy vectorization
- **Reproducibility**: Proper random seed handling

## Connection to Project 3

The stellar cluster snapshots generated in this project become the radiation sources for Project 3:
- **Stellar positions**: Spatial distribution for radiation field calculations
- **Stellar masses and luminosities**: Heating source strengths
- **Cluster evolution**: How radiation field changes with time
- **Realistic populations**: IMF-sampled masses give proper luminosity functions

## Deliverables

### Final Submission
1. **N-Body Simulation Library**:
   - `ode_solvers.py`: Integration method implementations
   - `stellar_sampling.py`: IMF and Plummer sphere classes
   - `nbody_simulator.py`: Complete adaptive N-body framework
   - `cluster_analysis.py`: Evolution analysis tools

2. **Validation Notebooks**:
   - `orbital_mechanics_validation.ipynb`: Two-body problem tests
   - `sampling_validation.ipynb`: IMF and spatial distribution verification
   - `cluster_evolution_analysis.ipynb`: Mass segregation and dynamics

3. **Project 3 Interface**:
   - `cluster_snapshots.pkl`: Saved stellar cluster evolution data
   - `snapshot_format.md`: Documentation of data structure

This project establishes the realistic stellar systems needed for sophisticated radiation calculations while teaching essential computational physics skills: numerical integration, statistical sampling, vectorization, and adaptive methods.