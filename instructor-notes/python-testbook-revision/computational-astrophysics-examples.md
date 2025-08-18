# Computational Astrophysics Example Replacements for Chapters 1-6

## Chapter 1: Computational Environments & Scientific Workflows

### Current Observational Focus → Computational Physics Replacements

**REPLACE:** Magnitude calculation examples  
**WITH:** Setting up environments for numerical simulations
- Example: Installing scipy, numba, and other computational libraries
- Show how different solver libraries have different numerical precision
- Demonstrate why identical N-body code gives different results on different machines due to floating-point ordering

**REPLACE:** Telescope data pipeline references  
**WITH:** Computational physics workflow examples
- Example: Running a simple gravity calculation in IPython vs script vs notebook
- Show how notebook state corruption affects iterative solver convergence
- Demonstrate reproducibility issues in Monte Carlo simulations with random seeds

**REPLACE:** FITS file reading examples  
**WITH:** Scientific data format examples for simulations
- Example: Saving particle positions from N-body simulations in HDF5
- Loading initial conditions for stellar evolution models
- Checkpointing long-running simulations for restart capability

## Chapter 2: Python as Your Astronomical Calculator

### Current Observational Focus → Numerical Physics Replacements

**REPLACE:** Magnitude and flux calculations  
**WITH:** Fundamental physics calculations with precision issues
- Example: Computing gravitational potential energy between particles
  ```python
  # Show catastrophic cancellation when particles are close
  def potential_energy(m1, m2, r):
      G = 6.674e-11
      return -G * m1 * m2 / r  # Fails when r approaches 0
  ```
- Example: Energy conservation in orbital mechanics showing drift
- Example: Computing escape velocity from different stellar objects

**REPLACE:** Color index and photometric calculations  
**WITH:** Physical scales and unit conversions
- Example: Converting between CGS and SI units in astrophysics
- Example: Working with astronomical units, parsecs, and light-years
- Example: Calculating Schwarzschild radius (shows overflow with large masses)

**REPLACE:** Observational error propagation  
**WITH:** Numerical error accumulation in simulations
- Example: Tracking energy conservation error in planet orbits
  ```python
  # Show how integration error accumulates over orbits
  def track_energy_error(positions, velocities, masses):
      KE = 0.5 * sum(m * v**2 for m, v in zip(masses, velocities))
      PE = gravitational_potential(positions, masses)
      return (KE + PE) - initial_energy
  ```

## Chapter 3: Control Flow & Logic

### Current Observational Focus → Simulation Algorithm Replacements

**REPLACE:** Finding peaks in light curves  
**WITH:** Adaptive timestep algorithms
- Example: Implementing adaptive timestep for orbit integration
  ```python
  while t < t_end:
      dt = calculate_timestep(acceleration, velocity)
      if dt < dt_min:
          print("Warning: timestep collapsed")
          dt = dt_min
      position += velocity * dt
      velocity += acceleration * dt
  ```

**REPLACE:** Filtering observational data  
**WITH:** Convergence checking in iterative solvers
- Example: Iterating to hydrostatic equilibrium in stellar models
- Example: Newton-Raphson for finding Lagrange points
- Example: Bisection method for solving Kepler's equation

**REPLACE:** Observation quality checks  
**WITH:** Physical constraint validation
- Example: Checking conservation laws (energy, momentum, angular momentum)
- Example: Enforcing causality in relativistic simulations
- Example: Validating thermodynamic consistency (pressure, temperature, density)

**REPLACE:** Time series analysis loops  
**WITH:** Force calculation loops in N-body systems
- Example: Direct N-body force summation with softening
  ```python
  for i in range(n_particles):
      force[i] = 0
      for j in range(n_particles):
          if i != j:
              r = position[j] - position[i]
              r_mag = sqrt(dot(r, r) + epsilon**2)  # Softening
              force[i] += G * mass[j] * r / r_mag**3
  ```

## Chapter 4: Data Structures

### Current Observational Focus → Simulation Data Structure Replacements

**REPLACE:** Catalog and observation storage  
**WITH:** Particle and mesh data structures
- Example: Storing N-body particles efficiently
  ```python
  # Dictionary of arrays vs array of dictionaries
  # Good for vectorization:
  particles = {
      'position': np.array([[x1,y1,z1], [x2,y2,z2], ...]),
      'velocity': np.array([[vx1,vy1,vz1], ...]),
      'mass': np.array([m1, m2, ...])
  }
  # vs Bad for performance:
  particles = [
      {'pos': [x1,y1,z1], 'vel': [...], 'mass': m1},
      {'pos': [x2,y2,z2], 'vel': [...], 'mass': m2}
  ]
  ```

**REPLACE:** Observation filtering with sets  
**WITH:** Neighbor finding with spatial data structures
- Example: Using dictionaries for spatial hashing in collision detection
- Example: Octree representation for hierarchical force calculation
- Example: Grid-based fluid simulation with ghost cells

**REPLACE:** Time series storage  
**WITH:** Trajectory and state history storage
- Example: Efficient storage of orbital trajectories
- Example: Checkpointing simulation state for restart
- Example: Phase space trajectory storage for chaos analysis

**REPLACE:** Measurement uncertainty tracking  
**WITH:** Conservation quantity tracking
- Example: Data structure for monitoring energy, momentum over time
  ```python
  class ConservationMonitor:
      def __init__(self):
          self.time = []
          self.energy = []
          self.momentum = []
          self.angular_momentum = []
      
      def check_conservation(self, state, tolerance=1e-10):
          drift = abs(self.energy[-1] - self.energy[0]) / abs(self.energy[0])
          return drift < tolerance
  ```

## Chapter 5: Functions & Modules

### Current Observational Focus → Physics Solver Module Replacements

**REPLACE:** Data reduction pipeline functions  
**WITH:** Physics calculation modules
- Example: Gravity calculation module with different force laws
  ```python
  def newtonian_gravity(m1, m2, r):
      """F = G*m1*m2/r^2"""
      return G * m1 * m2 / (r**2 + softening**2)
  
  def modified_gravity(m1, m2, r, a0=1.2e-10):
      """MOND gravity for galaxy rotation curves"""
      a_newton = G * m1 * m2 / r**2
      return a_newton * mu(a_newton / a0)
  ```

**REPLACE:** Image processing functions  
**WITH:** Integration scheme modules
- Example: Module with Euler, Verlet, RK4 integrators
- Example: Symplectic integrators for long-term stability
- Example: Implicit methods for stiff equations

**REPLACE:** Statistical analysis functions  
**WITH:** Equation of state modules
- Example: Ideal gas, degenerate matter, radiation pressure
- Example: Polytropic models for stellar structure
- Example: Nuclear reaction rate calculations

**REPLACE:** Calibration functions  
**WITH:** Boundary condition modules
- Example: Reflecting, periodic, absorbing boundaries for simulations
- Example: Inflow/outflow conditions for hydrodynamics
- Example: Initial condition generators (Plummer sphere, King model)

## Chapter 6: Object-Oriented Programming

### Current Observational Focus → Physical System Class Replacements

**REPLACE:** VariableStar and observation classes  
**WITH:** Particle and Field classes
- Example: Particle class for N-body simulations
  ```python
  class Particle:
      def __init__(self, mass, position, velocity):
          self.mass = mass
          self.position = np.array(position)
          self.velocity = np.array(velocity)
          self.force = np.zeros(3)
      
      def kinetic_energy(self):
          return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
      
      def update_position(self, dt):
          self.position += self.velocity * dt
      
      def update_velocity(self, dt):
          self.velocity += self.force / self.mass * dt
  ```

**REPLACE:** LightCurve and photometry pipeline  
**WITH:** Integrator and Solver hierarchies
- Example: Base Integrator class with Euler, Verlet, RK4 subclasses
  ```python
  class Integrator:
      def step(self, state, dt):
          raise NotImplementedError
  
  class VerletIntegrator(Integrator):
      def step(self, state, dt):
          # Position Verlet algorithm
          state.position += state.velocity * dt + 0.5 * state.acceleration * dt**2
          new_acceleration = self.calculate_acceleration(state)
          state.velocity += 0.5 * (state.acceleration + new_acceleration) * dt
          state.acceleration = new_acceleration
  ```

**REPLACE:** Observatory and Telescope classes  
**WITH:** Star and StellarPopulation classes
- Example: Star class with mass-luminosity relations, evolution
- Example: StellarCluster class managing many stars
- Example: Galaxy class with dark matter halo and stellar disk

**REPLACE:** Filter and Detector classes  
**WITH:** Grid and Mesh classes for PDEs
- Example: Grid2D class for fluid simulations
- Example: AdaptiveMesh class for AMR
- Example: ParticleMesh class for hybrid methods

### Additional Computational Examples Throughout

**For "Why This Matters" boxes:**
- Chapter 1: How LIGO's gravitational wave detection required identical numerical processing across sites
- Chapter 2: How the N-body problem's chaotic nature makes precision crucial
- Chapter 3: How adaptive timestepping saved the Cassini mission fuel
- Chapter 4: How tree codes make galaxy simulations possible (N log N vs N²)
- Chapter 5: How modular design enabled MESA stellar evolution code
- Chapter 6: How object-oriented design powers modern SPH codes

**For "Common Bug Alert" boxes:**
- Chapter 1: Random seed reproducibility in Monte Carlo
- Chapter 2: Unit confusion (CGS vs SI) in gravitational calculations
- Chapter 3: Timestep instability in explicit integration
- Chapter 4: Memory layout affecting cache performance in force calculations
- Chapter 5: Global state corruption in parallel simulations
- Chapter 6: Shallow vs deep copy of particle states

**For "Debug This!" challenges:**
- Chapter 1: Why does my simulation give different results each run?
- Chapter 2: Why does my planet spiral into the sun?
- Chapter 3: Why does my adaptive timestep get stuck at dt_min?
- Chapter 4: Why is my octree N-body slower than direct summation?
- Chapter 5: Why don't my particles conserve momentum?
- Chapter 6: Why does my stellar evolution class lose mass?

**For "Check Your Understanding" boxes:**
- Chapter 1: What happens to energy conservation with different float precisions?
- Chapter 2: Calculate the Roche limit for Earth-Moon system
- Chapter 3: When would implicit integration beat explicit?
- Chapter 4: Design data structure for SPH particles with neighbors
- Chapter 5: Create module for different cooling functions
- Chapter 6: Inheritance hierarchy for different gravity solvers

## Key Principles for All Replacements

1. **Every example solves a physics problem**, not processes data
2. **Numerical methods are central**, not incidental
3. **Performance matters** because simulations are computationally expensive
4. **Conservation laws** provide natural debugging checks
5. **Examples build toward course projects** (N-body, Monte Carlo, stellar evolution)
6. **Mathematical concepts are explicit** (derivatives for forces, integrals for energy)
7. **Scales range from stellar to cosmological**, showing method generality

## Retained Elements

The variable star exercises can remain as they provide a coherent narrative thread, but should be positioned as ONE application among many, not the dominant theme. They work well for teaching:
- Progressive complexity in OOP design
- Building toward complete scientific software
- Connecting multiple concepts in one domain

But balance them with equal emphasis on:
- N-body dynamics exercises
- Stellar structure calculations  
- Hydrodynamics problems
- Monte Carlo sampling techniques
- Numerical integration challenges