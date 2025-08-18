# ASTR 596 Complete Lecture Notes: Week-by-Week Teaching Guide
## Computational Astrophysics with Modern Statistical Methods

---

# Course Introduction & Pedagogical Framework

## Core Teaching Philosophy

This course transforms astronomy graduate students into computational scientists through a carefully designed progression from foundational physics to cutting-edge machine learning. Each week builds systematically on previous knowledge while introducing new mathematical concepts through astrophysical applications. The emphasis throughout is on understanding algorithms from first principles before using modern frameworks.

### Learning Progression Strategy

Students encounter the same computational patterns at increasing sophistication levels. For example, optimization appears first as stellar parameter fitting, then as gradient descent for machine learning, then as MCMC sampling for Bayesian inference, and finally as neural network training. This spiral approach ensures deep understanding while building toward research-level competency.

---

# Week 1: Stellar Structure & Computational Foundations
**Learning Objectives**: Master stellar physics fundamentals, establish professional coding practices, understand metallicity effects on stellar evolution

## Stellar Structure Physics

### The Fundamental Stellar Relations

Understanding stars requires grasping how mass determines all other stellar properties. The most fundamental relationship in stellar astrophysics connects stellar mass to luminosity through nuclear physics and hydrostatic equilibrium.

#### Main Sequence Mass-Luminosity Relation

For main sequence stars, nuclear fusion rate depends critically on central temperature and density, both determined by stellar mass. The basic scaling relationship emerges from combining hydrostatic equilibrium with nuclear physics:

```
L ∝ M^α  where α ≈ 3.5 for solar-type stars
```

**Physical Intuition**: More massive stars have higher central pressures, leading to higher temperatures and dramatically increased fusion rates. The steep dependence (α = 3.5) means a star twice as massive as the Sun is approximately 11 times more luminous.

#### Metallicity-Dependent Relations (Tout et al. 1996)

Real stars have varying metallicity (heavy element abundance), which affects stellar structure through opacity changes. The Tout relations provide industry-standard metallicity-dependent scaling:

**Mass-Luminosity with Metallicity**:
```
log(L/L☉) = a₁ + a₂X + a₃X² + a₄X³ + a₅X⁴ + a₆X⁵
```

where X = log(M/M☉) and coefficients a₁ through a₆ depend on metallicity Z.

**Mass-Radius Relations**:
```
log(R/R☉) = b₁ + b₂Y + b₃Y² + b₄Y³ + b₅Y⁴
```

where Y = log(M/M☉) and coefficients depend on metallicity.

**Teaching Strategy**: Start with the conceptual understanding that metallicity affects stellar opacity, which changes internal structure. Higher metallicity increases opacity, affecting convection and energy transport, ultimately modifying the mass-luminosity relation.

### Stellar Evolution Fundamentals

#### Nuclear Timescales

The nuclear burning timescale determines how long a star spends on the main sequence:

```
τ_nuclear = ε × M × c² / L ≈ 10¹⁰ years × (M/M☉) / (L/L☉)
```

where ε ≈ 0.007 is the nuclear efficiency for hydrogen burning.

**Key Insight**: Combined with L ∝ M^3.5, this gives τ_nuclear ∝ M^(-2.5), explaining why massive stars evolve much faster than low-mass stars.

#### Stellar Populations and Metallicity

**Population I** (disk stars): Z ≈ 0.01 - 0.03 (metal-rich)
**Population II** (halo stars): Z ≈ 0.0001 - 0.001 (metal-poor)

These populations formed at different times in galactic evolution, with Population II stars forming from primordial gas and Population I stars forming from gas enriched by previous stellar generations.

## Mathematical Foundations: Linear Algebra for Stellar Systems

### Vectors in Stellar Physics

Stellar properties naturally form vector quantities. A single star can be represented as:

```
s⃗ = [M, R, T_eff, L, Z, age]ᵀ
```

For stellar populations, we work with matrices where each row represents one star:

```
S = [s⃗₁, s⃗₂, ..., s⃗ₙ]ᵀ
```

### Vector Operations and Physical Meaning

**Dot Products**: Calculate total stellar mass or luminosity
```
M_total = 1⃗ · M⃗ = Σᵢ Mᵢ
```

**Element-wise Operations**: Apply stellar relations across populations
```
L⃗ = K × M⃗^α  (vectorized mass-luminosity relation)
```

### Matrix Operations for Stellar Calculations

**Broadcasting**: Apply metallicity-dependent corrections
```
L_corrected = L_base × Z_correction_matrix
```

This mathematical framework enables efficient calculations on thousands of stars simultaneously, essential for stellar population synthesis.

## Object-Oriented Programming for Stellar Physics

### The Star Class Design

Object-oriented programming provides a natural way to model stellar physics because stars have properties (mass, temperature, luminosity) and behaviors (evolution, cooling, nuclear burning).

```python
class Star:
    def __init__(self, mass, metallicity, age=0):
        """Initialize star with fundamental properties"""
        self.mass = mass  # Solar masses
        self.metallicity = metallicity  # Z/Z_solar
        self.age = age  # Myr
        
        # Derive other properties from fundamental relations
        self.luminosity = self._calculate_luminosity()
        self.radius = self._calculate_radius()
        self.temperature = self._calculate_temperature()
    
    def _calculate_luminosity(self):
        """Tout et al. metallicity-dependent L-M relation"""
        # Implementation of complex polynomial fits
        pass
    
    def evolve(self, time_step):
        """Evolve star forward in time"""
        self.age += time_step
        # Update properties based on stellar evolution
```

### Composition and Inheritance

**Composition**: StellarPopulation contains many Star objects
**Inheritance**: MainSequenceStar, RedGiantStar inherit from base Star class

This design mirrors the hierarchical structure of stellar systems and makes code both readable and extensible.

## Common Misconceptions and Debugging Strategies

### Misconception: "All stars are like the Sun"

Many students initially assume solar values for all calculations. Emphasize the enormous range in stellar properties:
- **Mass range**: 0.08 M☉ (brown dwarf limit) to 100+ M☉ (most massive stars)
- **Luminosity range**: 10⁻⁴ L☉ (red dwarfs) to 10⁶ L☉ (hypergiants)
- **Lifetime range**: 10⁶ years (massive stars) to 10¹² years (low-mass stars)

### Debugging Strategy: Physical Sanity Checks

Always validate stellar calculations against known limits:
- **Eddington limit**: L < L_Edd = 4πGMc/κ (maximum luminosity before radiation pressure disrupts star)
- **Main sequence lifetime**: Must be less than age of universe (13.8 Gyr)
- **Solar neighborhood**: Most stars should be less massive than the Sun (mass function is bottom-heavy)

---

# Week 2: Advanced Stellar Physics & Population Synthesis
**Learning Objectives**: Implement complete stellar population models, understand color-magnitude diagrams, master NumPy vectorization

## Advanced Stellar Astrophysics

### Color-Magnitude Diagrams and Stellar Classification

The Hertzsprung-Russell diagram reveals stellar evolution through the relationship between luminosity and surface temperature. For observational astronomy, we use color-magnitude diagrams where color serves as a temperature proxy.

#### Color Systems and Stellar Temperatures

**B-V Color Index**:
```
(B-V) = -2.5 log(F_B/F_V)
```

This color correlates strongly with effective temperature:
```
T_eff ≈ 8540 K / (0.92(B-V) + 1.7)  (for main sequence stars)
```

**Physical Understanding**: Blue stars (hot) have negative B-V colors, red stars (cool) have positive B-V colors. The relationship emerges from blackbody radiation physics and the wavelength sensitivity of B and V filters.

#### Blackbody Radiation and Wien's Law

Stellar colors fundamentally arise from blackbody radiation. Wien's displacement law gives the peak wavelength:

```
λ_max = 2.898 × 10⁻³ / T  (meters·Kelvin)
```

For the Sun (T = 5778 K): λ_max = 502 nm (green), explaining why the Sun appears white (peak in visible spectrum).

### Stellar Evolution and Isochrones

#### Main Sequence Evolution

Stars spend most of their lives burning hydrogen in their cores. The main sequence lifetime depends critically on mass:

```
τ_MS = τ_☉ × (M/M☉)^(-2.5) × (L☉/L)
```

where τ_☉ = 10 Gyr is the solar main sequence lifetime.

#### Isochrone Construction

An isochrone shows all stars of the same age but different masses. Constructing isochrones requires:

1. **Mass grid**: Sample masses from 0.1 to 100 M☉
2. **Evolution tracks**: Calculate L(t) and T_eff(t) for each mass
3. **Age slice**: Extract L and T_eff at fixed age for all masses

**Mathematical Representation**:
```
Isochrone(t) = {(L(M,t), T_eff(M,t)) | M ∈ [M_min, M_max]}
```

## Advanced NumPy Operations for Stellar Populations

### Broadcasting and Vectorization

Stellar population synthesis requires calculations on thousands of stars simultaneously. NumPy broadcasting enables elegant vectorized operations.

#### Initial Mass Function Sampling

The Salpeter Initial Mass Function describes the distribution of stellar masses at birth:

```
dN/dM ∝ M^(-2.35)  for M > 0.5 M☉
```

**Vectorized Implementation**:
```python
def sample_salpeter_imf(n_stars, m_min=0.1, m_max=100):
    """Sample stellar masses from Salpeter IMF"""
    # Use inverse transform sampling
    alpha = -2.35
    beta = alpha + 1  # = -1.35
    
    # Generate uniform random numbers
    u = np.random.random(n_stars)
    
    # Inverse transform for power law
    masses = ((m_max**beta - m_min**beta) * u + m_min**beta)**(1/beta)
    
    return masses
```

#### Vectorized Stellar Property Calculations

Apply stellar relations to entire populations simultaneously:

```python
class StellarPopulation:
    def __init__(self, n_stars, metallicity, age):
        # Sample masses from IMF
        self.masses = sample_salpeter_imf(n_stars)
        self.metallicity = metallicity
        self.age = age
        
        # Vectorized calculations
        self.luminosities = self._calculate_luminosities_vectorized()
        self.temperatures = self._calculate_temperatures_vectorized()
        self.colors = self._calculate_colors_vectorized()
    
    def _calculate_luminosities_vectorized(self):
        """Tout et al. relation applied to all stars"""
        # Vectorized polynomial evaluation
        log_m = np.log10(self.masses)
        coeffs = self._get_tout_coefficients(self.metallicity)
        
        log_l = np.polyval(coeffs, log_m)
        return 10**log_l
```

### Advanced Array Operations

#### Boolean Indexing for Stellar Selection

Select specific stellar types using logical conditions:

```python
# Select main sequence stars
main_sequence = (population.luminosities > 0.1) & (population.masses < 10)
ms_masses = population.masses[main_sequence]
ms_luminosities = population.luminosities[main_sequence]

# Select red giants
red_giants = (population.luminosities > 100) & (population.temperatures < 4000)
```

#### Efficient Statistical Analysis

Calculate population statistics using NumPy's optimized functions:

```python
# Total stellar mass and luminosity
total_mass = np.sum(population.masses)
total_luminosity = np.sum(population.luminosities)

# Mass-weighted average metallicity
avg_metallicity = np.average(population.metallicity, weights=population.masses)

# Luminosity function
lum_bins = np.logspace(-2, 6, 50)
lum_hist, _ = np.histogram(population.luminosities, bins=lum_bins)
```

## Mathematical Foundations: Statistics and Probability

### Probability Distributions in Stellar Physics

#### The Initial Mass Function as a Probability Distribution

The IMF represents the probability distribution of stellar masses at birth. Understanding this as a formal probability distribution enables rigorous sampling and statistical analysis.

**Probability Density Function**:
```
p(M) = A × M^(-α)  where A is normalization constant
```

**Normalization Condition**:
```
∫[M_min to M_max] p(M) dM = 1
```

This gives the normalization constant:
```
A = (1-α) / (M_max^(1-α) - M_min^(1-α))  for α ≠ 1
```

#### Statistical Properties of Stellar Populations

**Mean Stellar Mass**:
```
⟨M⟩ = ∫ M × p(M) dM
```

**Variance**:
```
Var(M) = ⟨M²⟩ - ⟨M⟩²
```

For the Salpeter IMF (α = 2.35), most mass is in low-mass stars, but most light comes from high-mass stars. This fundamental asymmetry drives stellar population evolution.

### Error Propagation and Uncertainty

#### Observational Uncertainties

Real stellar observations have uncertainties that propagate through calculations. For a function f(x,y):

```
σ_f² = (∂f/∂x)² σ_x² + (∂f/∂y)² σ_y² + 2(∂f/∂x)(∂f/∂y)σ_xy
```

**Example**: Distance modulus uncertainty
```
μ = 5 log₁₀(d) - 5  (distance modulus)
σ_μ = (5/ln(10)) × (σ_d/d)  (fractional distance error)
```

#### Monte Carlo Error Estimation

For complex calculations, Monte Carlo methods provide robust uncertainty estimates:

```python
def estimate_stellar_uncertainty(masses, mass_errors, n_trials=1000):
    """Estimate uncertainty in stellar population properties"""
    results = []
    
    for _ in range(n_trials):
        # Sample masses from error distribution
        sampled_masses = np.random.normal(masses, mass_errors)
        
        # Calculate derived quantities
        luminosities = calculate_luminosities(sampled_masses)
        total_luminosity = np.sum(luminosities)
        
        results.append(total_luminosity)
    
    return np.mean(results), np.std(results)
```

## Active Learning and Conceptual Understanding

### Check Your Understanding: Stellar Evolution Timescales

**Question**: A 20 M☉ star is approximately 40,000 times more luminous than the Sun. How does its main sequence lifetime compare to the Sun's 10 Gyr lifetime?

**Solution Process**:
1. **Nuclear fuel**: Both stars have roughly proportional nuclear fuel (∝ M)
2. **Consumption rate**: Proportional to luminosity (L)
3. **Lifetime ratio**: τ_star/τ_☉ = (M_star/M_☉) × (L_☉/L_star)
4. **Calculation**: τ_star = 10 Gyr × (20/1) × (1/40,000) = 5 × 10⁻³ Gyr = 5 Myr

**Physical Insight**: Massive stars live fast and die young because their enormous luminosities consume nuclear fuel extremely rapidly.

### Computational Thinking: Vectorization Benefits

Understanding why vectorization matters develops computational intuition essential for later projects.

**Timing Comparison**:
```python
# Loop-based calculation (slow)
start_time = time.time()
luminosities_loop = []
for mass in masses:
    lum = stellar_luminosity_function(mass)
    luminosities_loop.append(lum)
loop_time = time.time() - start_time

# Vectorized calculation (fast)
start_time = time.time()
luminosities_vec = stellar_luminosity_function(masses)  # NumPy arrays
vec_time = time.time() - start_time

speedup = loop_time / vec_time
print(f"Vectorization speedup: {speedup:.1f}x")
```

**Typical Results**: 10-100x speedup for large arrays, demonstrating why vectorized thinking is essential for computational science.

---

# Week 3: Gravitational Dynamics & Numerical Integration
**Learning Objectives**: Understand N-body gravitational physics, master numerical integration methods, implement conservation law checking

## Gravitational Physics Fundamentals

### Newton's Law of Gravitation in N-Body Systems

The gravitational force between any two point masses follows Newton's inverse square law:

```
F⃗_ij = -G(m_i m_j)/|r⃗_ij|³ × r⃗_ij
```

where r⃗_ij = r⃗_i - r⃗_j is the separation vector.

**Physical Insight**: The force acts along the line connecting the masses, decreases as the square of distance, and is always attractive. The negative sign indicates the force points from mass i toward mass j.

#### Total Force on Each Particle

In an N-body system, each particle experiences gravitational forces from all other particles:

```
F⃗_i = Σ(j≠i) F⃗_ij = -Gm_i Σ(j≠i) (m_j/|r⃗_ij|³) × r⃗_ij
```

This leads to Newton's second law for each particle:

```
m_i (d²r⃗_i/dt²) = F⃗_i
```

Simplifying:
```
d²r⃗_i/dt² = -G Σ(j≠i) (m_j/|r⃗_ij|³) × r⃗_ij
```

### Two-Body Problem: Analytical Solutions

Before tackling N-body problems numerically, understanding the analytical two-body solution provides crucial validation benchmarks.

#### Reduced Mass and Relative Motion

For two masses m₁ and m₂, the relative motion follows:

```
μ (d²r⃗/dt²) = -G(m₁ + m₂)/r³ × r⃗
```

where μ = m₁m₂/(m₁ + m₂) is the reduced mass and r⃗ is the separation vector.

#### Orbital Elements and Kepler's Laws

**First Law**: Orbits are ellipses with the center of mass at one focus.

**Second Law**: Equal areas swept in equal times.
```
dA/dt = (1/2)|r⃗ × v⃗| = constant = h/2
```

where h is the specific angular momentum.

**Third Law**: Period squared proportional to semi-major axis cubed.
```
P² = (4π²/G(m₁ + m₂)) × a³
```

**Teaching Strategy**: Use these analytical solutions to validate numerical integrators. Any integration scheme should preserve orbital periods and eccentricities for two-body systems.

### Softening Parameters for Close Encounters

In stellar cluster simulations, stars can approach very close distances, leading to numerical difficulties. Softening parameters prevent singularities:

```
F⃗_ij = -G(m_i m_j)/(r_ij² + ε²)^(3/2) × r̂_ij
```

where ε is the softening length.

**Physical Interpretation**: Softening represents finite stellar size effects or tidal disruption. For stellar clusters, ε ≈ 0.01-0.1 pc provides reasonable approximation without significant dynamical effects.

## Numerical Integration Methods

### Euler Method: First-Order Integration

The simplest integration scheme uses first-order Taylor expansion:

```
r⃗(t + dt) = r⃗(t) + v⃗(t) × dt
v⃗(t + dt) = v⃗(t) + a⃗(t) × dt
```

where a⃗(t) = F⃗(t)/m is the acceleration.

**Algorithm Implementation**:
```python
def euler_step(positions, velocities, masses, dt):
    """Single Euler integration step"""
    # Calculate accelerations from current positions
    accelerations = calculate_accelerations(positions, masses)
    
    # Update velocities and positions
    new_velocities = velocities + accelerations * dt
    new_positions = positions + velocities * dt
    
    return new_positions, new_velocities
```

**Limitations**: Euler method has poor energy conservation and stability properties. Energy typically grows exponentially with time, making it unsuitable for long-term orbit integration.

### Runge-Kutta Methods: Higher-Order Accuracy

Fourth-order Runge-Kutta (RK4) provides much better accuracy through multiple intermediate evaluations:

```
k₁ = f(t, y)
k₂ = f(t + dt/2, y + k₁dt/2)
k₃ = f(t + dt/2, y + k₂dt/2)
k₄ = f(t + dt, y + k₃dt)

y(t + dt) = y(t) + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

For the gravitational N-body problem, y = [r⃗₁, r⃗₂, ..., r⃗ₙ, v⃗₁, v⃗₂, ..., v⃗ₙ]ᵀ.

**RK4 Implementation**:
```python
def rk4_step(state, masses, dt):
    """Fourth-order Runge-Kutta step for N-body system"""
    def derivatives(s):
        n = len(masses)
        positions = s[:3*n].reshape((n, 3))
        velocities = s[3*n:].reshape((n, 3))
        
        accelerations = calculate_accelerations(positions, masses)
        
        # Return derivatives: [velocities, accelerations]
        return np.concatenate([velocities.flatten(), 
                              accelerations.flatten()])
    
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

### Symplectic Integration: Leapfrog Method

Hamiltonian systems like N-body gravity benefit from symplectic integrators that preserve phase space volume and provide better long-term stability.

**Leapfrog Algorithm**:
```
v⃗(t + dt/2) = v⃗(t) + a⃗(t) × dt/2
r⃗(t + dt) = r⃗(t) + v⃗(t + dt/2) × dt
a⃗(t + dt) = F⃗(r⃗(t + dt))/m
v⃗(t + dt) = v⃗(t + dt/2) + a⃗(t + dt) × dt/2
```

**Physical Insight**: Leapfrog alternates position and velocity updates, naturally conserving energy for Hamiltonian systems. The "kick-drift-kick" sequence reflects the splitting of kinetic and potential energy operators.

```python
def leapfrog_step(positions, velocities, masses, dt):
    """Symplectic leapfrog integration step"""
    # Half-step velocity update (kick)
    accelerations = calculate_accelerations(positions, masses)
    velocities_half = velocities + 0.5 * dt * accelerations
    
    # Full-step position update (drift)
    positions_new = positions + dt * velocities_half
    
    # Half-step velocity update (kick)
    accelerations_new = calculate_accelerations(positions_new, masses)
    velocities_new = velocities_half + 0.5 * dt * accelerations_new
    
    return positions_new, velocities_new
```

## Conservation Laws and Numerical Validation

### Energy Conservation

Total energy consists of kinetic and potential components:

```
E = T + U = (1/2)Σᵢ mᵢv²ᵢ - G Σᵢ<ⱼ (mᵢmⱼ/rᵢⱼ)
```

**Numerical Implementation**:
```python
def calculate_total_energy(positions, velocities, masses):
    """Calculate total energy of N-body system"""
    # Kinetic energy
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    
    # Potential energy
    potential = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i+1, n):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            potential -= G * masses[i] * masses[j] / r_ij
    
    return kinetic + potential
```

**Energy Conservation Check**: For isolated systems, total energy should remain constant. Fractional energy change provides integration quality metric:

```
ΔE/E₀ = |E(t) - E(0)|/|E(0)|
```

Good integrators maintain ΔE/E₀ < 10⁻⁶ over many orbital periods.

### Angular Momentum Conservation

For isolated systems, total angular momentum is conserved:

```
L⃗ = Σᵢ mᵢ(r⃗ᵢ × v⃗ᵢ)
```

**Implementation**:
```python
def calculate_angular_momentum(positions, velocities, masses):
    """Calculate total angular momentum"""
    angular_momentum = np.zeros(3)
    for i in range(len(masses)):
        angular_momentum += masses[i] * np.cross(positions[i], velocities[i])
    return angular_momentum
```

### Center of Mass Motion

The center of mass should move at constant velocity (or remain stationary):

```
R⃗_cm = (Σᵢ mᵢr⃗ᵢ)/(Σᵢ mᵢ)
V⃗_cm = (Σᵢ mᵢv⃗ᵢ)/(Σᵢ mᵢ)
```

**Correction Strategy**: Remove center of mass drift by subtracting average motion:
```python
def remove_cm_motion(positions, velocities, masses):
    """Remove center of mass motion"""
    total_mass = np.sum(masses)
    cm_position = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    cm_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
    
    positions_corrected = positions - cm_position
    velocities_corrected = velocities - cm_velocity
    
    return positions_corrected, velocities_corrected
```

## Mathematical Foundations: Differential Equations

### Ordinary Differential Equations in Physics

Gravitational dynamics exemplify how physical laws naturally lead to differential equations. Newton's second law gives:

```
d²r⃗/dt² = F⃗(r⃗)/m
```

This second-order ODE can be converted to a first-order system by introducing velocity:

```
dr⃗/dt = v⃗
dv⃗/dt = F⃗(r⃗)/m
```

**State Vector Formulation**: Define state vector y⃗ = [r⃗, v⃗]ᵀ, then:

```
dy⃗/dt = f⃗(t, y⃗) = [v⃗, F⃗(r⃗)/m]ᵀ
```

This transforms the N-body problem into a standard ODE form amenable to numerical integration methods.

### Hamiltonian Mechanics and Phase Space

Gravitational systems are Hamiltonian, meaning they conserve phase space volume (Liouville's theorem). The Hamiltonian is:

```
H = T + U = Σᵢ (p⃗ᵢ²/2mᵢ) + U(r⃗₁, ..., r⃗ₙ)
```

where p⃗ᵢ = mᵢv⃗ᵢ is momentum.

**Hamilton's Equations**:
```
dr⃗ᵢ/dt = ∂H/∂p⃗ᵢ = p⃗ᵢ/mᵢ
dp⃗ᵢ/dt = -∂H/∂r⃗ᵢ = -∂U/∂r⃗ᵢ
```

**Symplectic Structure**: Phase space has natural symplectic structure that good integrators should preserve. This explains why leapfrog integration works better than Runge-Kutta for long-term orbital integration.

## Stellar Cluster Physics

### Virial Theorem and Equilibrium

For gravitationally bound systems in equilibrium, the virial theorem relates kinetic and potential energy:

```
2⟨T⟩ + ⟨U⟩ = 0
```

**Physical Interpretation**: Gravitational potential energy provides exactly twice the kinetic energy needed for equilibrium. This fundamental relationship enables estimates of cluster properties from observational data.

**Virial Velocity**:
```
v_vir = √(GM_cluster/R_cluster)
```

### Relaxation and Evolution Timescales

#### Two-Body Relaxation Time

Stellar encounters gradually change individual stellar orbits through gravitational "collisions":

```
t_relax ≈ (N/8ln(N)) × t_cross
```

where t_cross = R/v is the crossing time and N is the number of stars.

**Physical Meaning**: Time for stellar velocities to "forget" initial conditions and approach Maxwellian distribution through gravitational encounters.

#### Mass Segregation

More massive stars sink toward cluster center due to energy equipartition. The equipartition timescale is:

```
t_eq ≈ (m_i/⟨m⟩) × t_relax
```

where m_i is individual stellar mass and ⟨m⟩ is average mass.

**Observational Consequence**: Massive stars concentrate in cluster cores while low-mass stars populate the periphery.

## Common Misconceptions and Debugging

### Misconception: "Energy should be exactly conserved"

Students often expect perfect energy conservation from numerical integrators. Reality requires understanding numerical precision and accumulated errors.

**Teaching Point**: Explain that all numerical methods introduce errors. The goal is controlling error growth, not eliminating it entirely. Good integrators keep energy changes small compared to total energy.

### Debugging Strategy: Dimensional Analysis

Always check units in calculations:
- **Forces**: [M L T⁻²]
- **Gravitational constant**: G = 6.67 × 10⁻¹¹ m³ kg⁻¹ s⁻²
- **Accelerations**: [L T⁻²]

**Common Error**: Mixing unit systems (cgs vs SI vs astronomical units). Establish consistent unit system from the beginning.

### Validation Against Analytical Solutions

**Circular Orbits**: Period should equal P = 2π√(a³/GM)
**Elliptical Orbits**: Semi-major axis and eccentricity should remain constant
**Many-Body**: Total energy, angular momentum, and center of mass should be conserved

Use these checks systematically to verify integration accuracy before proceeding to complex cluster simulations.

---

# Week 4: Monte Carlo Methods & Statistical Sampling
**Learning Objectives**: Master random sampling techniques, understand Central Limit Theorem applications, implement importance sampling

## Monte Carlo Methods Fundamentals

### Random Number Generation and Quality

Monte Carlo methods depend critically on high-quality random number generators. Understanding their properties and limitations is essential for reliable scientific computing.

#### Pseudorandom Number Generators

Most computational random numbers are pseudorandom, generated by deterministic algorithms that produce sequences with statistical properties resembling true randomness.

**Linear Congruential Generator**:
```
X_{n+1} = (aX_n + c) mod m
```

**Modern Standards**: Mersenne Twister, PCG (Permuted Congruential Generator)
- **Period**: 2^19937 - 1 for Mersenne Twister
- **Statistical Quality**: Pass extensive randomness tests
- **Reproducibility**: Deterministic sequences enable debugging and verification

**Testing Randomness**:
```python
def test_uniformity(samples, n_bins=20):
    """Test if samples follow uniform distribution"""
    counts, _ = np.histogram(samples, bins=n_bins)
    expected = len(samples) / n_bins
    chi_squared = np.sum((counts - expected)**2 / expected)
    
    # Chi-squared test for uniformity
    degrees_freedom = n_bins - 1
    p_value = 1 - scipy.stats.chi2.cdf(chi_squared, degrees_freedom)
    
    return chi_squared, p_value
```

### Inverse Transform Sampling

The fundamental method for generating samples from arbitrary probability distributions uses the inverse of the cumulative distribution function (CDF).

#### Mathematical Foundation

For continuous distribution with CDF F(x), if U ~ Uniform(0,1), then X = F⁻¹(U) follows the desired distribution.

**Proof**: P(X ≤ x) = P(F⁻¹(U) ≤ x) = P(U ≤ F(x)) = F(x)

#### Power Law Distributions (IMF)

The Salpeter Initial Mass Function follows a power law:
```
p(M) ∝ M^{-α}  where α = 2.35
```

**Normalized PDF**:
```
p(M) = (1-α)/(M_max^{1-α} - M_min^{1-α}) × M^{-α}
```

**CDF**:
```
F(M) = (M^{1-α} - M_min^{1-α})/(M_max^{1-α} - M_min^{1-α})
```

**Inverse CDF**:
```
F^{-1}(u) = [u(M_max^{1-α} - M_min^{1-α}) + M_min^{1-α}]^{1/(1-α)}
```

**Implementation**:
```python
def sample_power_law(n_samples, alpha, m_min, m_max):
    """Sample from power law distribution using inverse transform"""
    u = np.random.random(n_samples)
    beta = 1 - alpha  # = -1.35 for Salpeter
    
    if np.abs(beta) < 1e-10:  # Handle α = 1 case
        return m_min * (m_max/m_min)**u
    else:
        return ((m_max**beta - m_min**beta) * u + m_min**beta)**(1/beta)
```

### Rejection Sampling

When inverse transform sampling is difficult or impossible, rejection sampling provides a general alternative method.

#### Algorithm

To sample from probability density p(x):
1. Find simple "envelope" function g(x) such that p(x) ≤ M × g(x) for some constant M
2. Sample x from g(x) and u from Uniform(0,1)
3. Accept x if u ≤ p(x)/(M × g(x)), otherwise reject and repeat

**Efficiency**: Acceptance rate = 1/M, so tight envelopes improve efficiency.

#### Example: Stellar Velocity Distributions

Sample from Maxwell-Boltzmann distribution using Gaussian envelope:

```python
def sample_maxwell_boltzmann(n_samples, sigma):
    """Sample 3D velocities from Maxwell-Boltzmann distribution"""
    # Maxwell-Boltzmann in 3D: p(v) ∝ v² exp(-v²/2σ²)
    # Use rejection sampling with Rayleigh envelope
    
    samples = []
    while len(samples) < n_samples:
        # Sample from Rayleigh distribution (good envelope)
        v_candidate = np.random.rayleigh(sigma * np.sqrt(2))
        
        # Acceptance probability
        prob_mb = v_candidate**2 * np.exp(-v_candidate**2 / (2*sigma**2))
        prob_envelope = v_candidate * np.exp(-v_candidate**2 / (2*sigma**2))
        acceptance_prob = prob_mb / (v_candidate * prob_envelope)
        
        if np.random.random() < acceptance_prob:
            samples.append(v_candidate)
    
    return np.array(samples)
```

## Statistical Physics Applications

### Central Limit Theorem and Error Estimation

The Central Limit Theorem provides the foundation for Monte Carlo error analysis and confidence intervals.

#### Theorem Statement

For independent samples X₁, X₂, ..., Xₙ from distribution with mean μ and variance σ²:

```
√n (X̄ - μ)/σ → N(0,1)  as n → ∞
```

where X̄ = (1/n)Σᵢ Xᵢ is the sample mean.

**Practical Consequence**: Sample mean is approximately normal with:
```
X̄ ~ N(μ, σ²/n)
```

Standard error of the mean:
```
SE = σ/√n
```

#### Monte Carlo Integration

Estimate integrals using random sampling:
```
I = ∫ f(x) dx ≈ (1/N) Σᵢ f(xᵢ)  where xᵢ ~ Uniform
```

**Error Estimate**:
```
σᵢ² = Var[f(X)] = ⟨f²⟩ - ⟨f⟩²
Standard Error = σᵢ/√N
```

**Example: Stellar Cluster Virial Ratio**:
```python
def estimate_virial_ratio_mc(n_trials=10000):
    """Estimate virial ratio using Monte Carlo sampling"""
    ratios = []
    
    for _ in range(n_trials):
        # Generate random cluster configuration
        positions = generate_random_cluster()
        velocities = sample_velocity_distribution()
        masses = sample_imf(len(positions))
        
        # Calculate kinetic and potential energy
        T = calculate_kinetic_energy(velocities, masses)
        U = calculate_potential_energy(positions, masses)
        
        virial_ratio = -2*T/U  # Should be ~1 for equilibrium
        ratios.append(virial_ratio)
    
    mean_ratio = np.mean(ratios)
    std_error = np.std(ratios) / np.sqrt(n_trials)
    
    return mean_ratio, std_error
```

### Importance Sampling

Standard Monte Carlo can be inefficient when the integrand has most contribution from small regions. Importance sampling concentrates samples where they matter most.

#### Mathematical Framework

To estimate integral ∫ f(x) p(x) dx where p(x) is a probability density:

```
I = ∫ f(x) p(x) dx = ∫ [f(x) p(x)/q(x)] q(x) dx ≈ (1/N) Σᵢ f(xᵢ) w(xᵢ)
```

where w(xᵢ) = p(xᵢ)/q(xᵢ) is the importance weight and xᵢ ~ q(x).

**Optimal Choice**: q(x) ∝ |f(x)| p(x) minimizes variance.

#### Stellar Encounter Rates

Estimate rates of close stellar encounters using importance sampling:

```python
def estimate_encounter_rate_importance(cluster, min_distance):
    """Estimate close encounter rate using importance sampling"""
    n_stars = len(cluster.positions)
    encounters = 0
    total_weight = 0
    
    for i in range(n_stars):
        for j in range(i+1, n_stars):
            # Sample encounter parameters with bias toward close approaches
            impact_param = sample_biased_impact_parameter(min_distance)
            relative_velocity = calculate_relative_velocity(i, j)
            
            # Calculate actual encounter probability
            prob_encounter = encounter_probability(impact_param, relative_velocity)
            
            # Importance weight
            weight = prob_encounter / biased_probability(impact_param)
            
            total_weight += weight
    
    return total_weight / n_stars
```

## Mathematical Foundations: Probability Theory

### Probability Distributions in Astrophysics

#### Continuous Distributions

**Uniform Distribution**: U(a,b)
```
p(x) = 1/(b-a)  for a ≤ x ≤ b
⟨X⟩ = (a+b)/2
Var(X) = (b-a)²/12
```

**Exponential Distribution**: Exp(λ)
```
p(x) = λe^{-λx}  for x ≥ 0
⟨X⟩ = 1/λ
Var(X) = 1/λ²
```

Used for stellar lifetimes, radioactive decay, Poisson waiting times.

**Normal Distribution**: N(μ,σ²)
```
p(x) = (1/√(2πσ²)) exp(-(x-μ)²/2σ²)
⟨X⟩ = μ
Var(X) = σ²
```

Central to measurement errors, velocity distributions, many physical processes.

#### Discrete Distributions

**Poisson Distribution**: Po(λ)
```
P(X = k) = (λᵏ e^{-λ})/k!
⟨X⟩ = λ
Var(X) = λ
```

Models rare events: supernova explosions, stellar formation events, photon arrivals.

**Binomial Distribution**: Bin(n,p)
```
P(X = k) = C(n,k) p^k (1-p)^{n-k}
⟨X⟩ = np
Var(X) = np(1-p)
```

Models binary outcomes: stellar binarity, planet detection, survey completeness.

### Joint Distributions and Correlations

#### Bivariate Normal Distribution

For correlated stellar properties (e.g., mass and metallicity):

```
p(x,y) = 1/(2πσₓσᵧ√(1-ρ²)) × exp[-1/(2(1-ρ²)) × ((x-μₓ)²/σₓ² - 2ρ(x-μₓ)(y-μᵧ)/(σₓσᵧ) + (y-μᵧ)²/σᵧ²)]
```

where ρ is the correlation coefficient.

**Sampling Algorithm**:
```python
def sample_correlated_properties(n_samples, mu_x, mu_y, sigma_x, sigma_y, rho):
    """Sample correlated stellar properties"""
    # Generate independent normal samples
    z1 = np.random.normal(0, 1, n_samples)
    z2 = np.random.normal(0, 1, n_samples)
    
    # Transform to desired correlation structure
    x = mu_x + sigma_x * z1
    y = mu_y + sigma_y * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    
    return x, y
```

#### Copulas for Complex Dependencies

When stellar properties have non-linear dependencies, copulas separate marginal distributions from dependence structure:

```
F(x,y) = C(F_X(x), F_Y(y))
```

where C is the copula function and F_X, F_Y are marginal CDFs.

## Stellar Cluster Applications

### King Profile Sampling

The King profile describes stellar density in globular clusters:

```
ρ(r) = ρ₀ [1/(1 + (r/r_c)²) - 1/(1 + (r_t/r_c)²)]²
```

where r_c is core radius and r_t is tidal radius.

**CDF Integration**: No analytical form exists, requiring numerical integration and inverse interpolation.

```python
def sample_king_profile(n_stars, r_core, r_tidal):
    """Sample stellar positions from King profile"""
    # Pre-compute CDF on radial grid
    r_grid = np.logspace(-2, np.log10(r_tidal), 1000)
    density = king_density(r_grid, r_core, r_tidal)
    
    # Integrate to get CDF (assuming spherical symmetry)
    mass_enclosed = 4 * np.pi * np.cumsum(density * r_grid**2 * np.diff(r_grid, prepend=0))
    cdf = mass_enclosed / mass_enclosed[-1]
    
    # Inverse interpolation for sampling
    u = np.random.random(n_stars)
    radii = np.interp(u, cdf, r_grid)
    
    # Generate random directions
    theta = np.random.uniform(0, np.pi, n_stars)
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    
    # Convert to Cartesian coordinates
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    
    return np.column_stack([x, y, z])
```

### Velocity Distribution Sampling

Stellar clusters approach energy equipartition through relaxation, leading to Maxwell-Boltzmann velocity distributions:

```
f(v) = 4π (m/2πkT)^{3/2} v² exp(-mv²/2kT)
```

**Mass-Dependent Temperatures**: Equipartition gives:
```
(1/2)m₁⟨v₁²⟩ = (1/2)m₂⟨v₂²⟩
σ_v(m) = σ_v,ref × √(m_ref/m)
```

More massive stars have lower velocity dispersions.

```python
def sample_equipartition_velocities(masses, sigma_ref, mass_ref):
    """Sample velocities assuming energy equipartition"""
    velocities = np.zeros((len(masses), 3))
    
    for i, mass in enumerate(masses):
        # Mass-dependent velocity dispersion
        sigma_v = sigma_ref * np.sqrt(mass_ref / mass)
        
        # Sample 3D Gaussian velocity
        velocities[i] = np.random.normal(0, sigma_v, 3)
    
    return velocities
```

## Convergence and Error Analysis

### Assessing Monte Carlo Convergence

#### Running Average Method

Monitor convergence by tracking running averages:

```python
def monitor_convergence(samples):
    """Monitor Monte Carlo convergence"""
    n_samples = len(samples)
    running_means = np.cumsum(samples) / np.arange(1, n_samples + 1)
    running_vars = np.zeros(n_samples)
    
    for i in range(1, n_samples):
        running_vars[i] = np.var(samples[:i+1])
    
    # Standard error evolution
    standard_errors = np.sqrt(running_vars / np.arange(1, n_samples + 1))
    
    return running_means, standard_errors
```

#### Batch Means Method

Divide samples into batches to assess independence:

```python
def batch_means_analysis(samples, n_batches=10):
    """Analyze convergence using batch means"""
    batch_size = len(samples) // n_batches
    batch_means = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_mean = np.mean(samples[start:end])
        batch_means.append(batch_mean)
    
    # If converged, batch means should be consistent
    overall_mean = np.mean(batch_means)
    batch_variance = np.var(batch_means)
    
    return overall_mean, batch_variance
```

### Variance Reduction Techniques

#### Control Variates

Use known analytical results to reduce variance:

```python
def control_variate_integration(f, g, analytical_integral_g, n_samples):
    """Reduce variance using control variate"""
    x = np.random.random(n_samples)
    f_samples = f(x)
    g_samples = g(x)
    
    # Optimal control coefficient
    covariance = np.cov(f_samples, g_samples)[0,1]
    g_variance = np.var(g_samples)
    c_opt = covariance / g_variance
    
    # Control variate estimator
    controlled_samples = f_samples - c_opt * (g_samples - analytical_integral_g)
    
    return np.mean(controlled_samples), np.std(controlled_samples) / np.sqrt(n_samples)
```

#### Stratified Sampling

Divide domain into strata and sample proportionally:

```python
def stratified_sampling(f, n_strata, n_total):
    """Stratified sampling for improved efficiency"""
    samples_per_stratum = n_total // n_strata
    stratum_results = []
    
    for i in range(n_strata):
        # Sample within stratum [i/n_strata, (i+1)/n_strata]
        u = np.random.random(samples_per_stratum)
        x_stratum = (i + u) / n_strata
        f_stratum = f(x_stratum)
        
        stratum_mean = np.mean(f_stratum)
        stratum_results.append(stratum_mean)
    
    # Combine stratum estimates
    overall_estimate = np.mean(stratum_results)
    return overall_estimate
```

This systematic approach to Monte Carlo methods provides the foundation for understanding both the MCRT algorithms in Project 3 and the Bayesian inference techniques in Project 5. The emphasis on validation, error analysis, and convergence assessment develops the critical thinking skills essential for computational research.

---

# Week 5: Monte Carlo Radiative Transfer Physics
**Learning Objectives**: Understand radiative transfer equation, implement photon packet methods, connect stellar heating to dust emission

## Radiative Transfer Fundamentals

### The Radiative Transfer Equation

Radiative transfer describes how light interacts with matter as it propagates through a medium. The equation governing this process is:

```
(1/c)(∂I/∂t) + n̂·∇I = -κ_ext ρ I + κ_abs ρ B(T) + (κ_scat ρ/4π) ∫ I(n̂') Φ(n̂',n̂) dΩ'
```

**Physical Interpretation**:
- **Left side**: Rate of change of specific intensity I along direction n̂
- **First term**: Extinction (absorption + scattering removes photons)
- **Second term**: Thermal emission (matter emits photons)
- **Third term**: Scattering (photons scattered from other directions)

#### Simplified Steady-State Form

For most astrophysical applications, we can neglect the time derivative:

```
n̂·∇I = -κ_ext ρ I + κ_abs ρ B(T) + (κ_scat ρ/4π) ∫ I(n̂') Φ(n̂',n̂) dΩ'
```

**Key Parameters**:
- **κ_ext = κ_abs + κ_scat**: Extinction coefficient
- **B(T)**: Planck function for thermal emission
- **Φ(n̂',n̂)**: Phase function describing scattering angular distribution
- **ρ**: Matter density

### Optical Depth and Beer's Law

Optical depth measures the cumulative opacity along a photon path:

```
τ = ∫ κ_ext ρ ds
```

where ds is the differential path length.

**Beer's Law**: For pure absorption (no scattering or emission):
```
I(τ) = I₀ e^(-τ)
```

**Physical Meaning**: 
- τ << 1: Optically thin, most photons escape without interaction
- τ >> 1: Optically thick, few photons penetrate deeply into medium
- τ = 1: Approximately where transition occurs

#### Wavelength Dependence

Dust opacity typically follows power law behavior:
```
κ(λ) = κ₀ (λ/λ₀)^(-β)
```

where β ≈ 1-2 for interstellar dust.

**Consequences**:
- Blue light more strongly absorbed than red light
- Stars appear redder when viewed through dust
- Dust heating peaks at shorter wavelengths where absorption is strongest

### Scattering Physics

#### Isotropic Scattering

Simplest case: photons scatter equally in all directions.

**Phase Function**: Φ(n̂',n̂) = 1 (normalized so ∫ Φ dΩ = 4π)

**Scattering Direction Sampling**:
```python
def sample_isotropic_direction():
    """Sample random direction on unit sphere"""
    # Method 1: Rejection sampling
    while True:
        x, y, z = np.random.uniform(-1, 1, 3)
        r_squared = x**2 + y**2 + z**2
        if r_squared <= 1:
            return np.array([x, y, z]) / np.sqrt(r_squared)
    
    # Method 2: Direct sampling (more efficient)
    cos_theta = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, 2*np.pi)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    return np.array([sin_theta * np.cos(phi), 
                     sin_theta * np.sin(phi), 
                     cos_theta])
```

#### Anisotropic Scattering (Henyey-Greenstein)

Real dust grains preferentially forward-scatter:

```
Φ(cos θ) = (1 - g²)/[4π (1 + g² - 2g cos θ)^(3/2)]
```

where g is the asymmetry parameter:
- g = 0: Isotropic scattering
- g > 0: Forward scattering preferred
- g < 0: Backward scattering preferred

**Sampling Algorithm**:
```python
def sample_henyey_greenstein(g):
    """Sample scattering angle from Henyey-Greenstein phase function"""
    if abs(g) < 1e-3:  # Nearly isotropic
        cos_theta = np.random.uniform(-1, 1)
    else:
        u = np.random.random()
        cos_theta = (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*u))**2) / (2*g)
    
    phi = np.random.uniform(0, 2*np.pi)
    return cos_theta, phi
```

## Monte Carlo Radiative Transfer Algorithm

### Photon Packet Representation

Monte Carlo methods represent radiation field using discrete photon packets, each carrying statistical weight:

```python
class PhotonPacket:
    def __init__(self, position, direction, energy, wavelength):
        self.position = np.array(position)     # Current position [x, y, z]
        self.direction = np.array(direction)   # Propagation direction (unit vector)
        self.energy = energy                   # Statistical weight (photons represented)
        self.wavelength = wavelength           # Photon wavelength
        self.alive = True                      # Whether packet is still propagating
```

### Core MCRT Algorithm

#### Step 1: Distance to Next Interaction

Sample random optical depth and convert to physical distance:

```python
def sample_interaction_distance(kappa_ext, rho):
    """Sample distance to next interaction event"""
    # Sample optical depth from exponential distribution
    tau = -np.log(np.random.random())
    
    # Convert to physical distance
    distance = tau / (kappa_ext * rho)
    
    return distance
```

#### Step 2: Interaction Type Determination

Determine whether interaction is absorption or scattering based on single scattering albedo:

```
a = κ_scat / (κ_abs + κ_scat)
```

```python
def determine_interaction_type(albedo):
    """Determine if interaction is absorption or scattering"""
    return np.random.random() < albedo  # True = scattering, False = absorption
```

#### Step 3: Photon Propagation

```python
def propagate_photon(photon, distance):
    """Move photon to interaction site"""
    photon.position += distance * photon.direction
    return photon
```

#### Step 4: Scattering Event

```python
def scatter_photon(photon, g=0):
    """Handle scattering interaction"""
    # Sample new direction from phase function
    cos_theta, phi = sample_henyey_greenstein(g)
    
    # Transform to global coordinate system
    new_direction = transform_direction(photon.direction, cos_theta, phi)
    photon.direction = new_direction
    
    return photon

def transform_direction(old_direction, cos_theta, phi):
    """Transform scattered direction to global coordinates"""
    # Choose perpendicular vectors for coordinate system
    if abs(old_direction[2]) < 0.9:
        perp1 = np.cross(old_direction, [0, 0, 1])
    else:
        perp1 = np.cross(old_direction, [1, 0, 0])
    
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(old_direction, perp1)
    
    # Construct new direction
    sin_theta = np.sqrt(1 - cos_theta**2)
    new_direction = (cos_theta * old_direction + 
                     sin_theta * np.cos(phi) * perp1 + 
                     sin_theta * np.sin(phi) * perp2)
    
    return new_direction / np.linalg.norm(new_direction)
```

### Complete MCRT Implementation

```python
def monte_carlo_radiative_transfer(stellar_sources, dust_grid, n_photons):
    """Complete MCRT simulation"""
    escaped_photons = []
    absorbed_energy = np.zeros_like(dust_grid.temperature)
    
    for source in stellar_sources:
        # Generate photon packets from stellar source
        photons = generate_stellar_photons(source, n_photons)
        
        for photon in photons:
            while photon.alive:
                # Get local dust properties
                kappa_abs, kappa_scat, rho = dust_grid.get_properties(photon.position)
                kappa_ext = kappa_abs + kappa_scat
                albedo = kappa_scat / kappa_ext if kappa_ext > 0 else 0
                
                # Sample distance to interaction
                distance = sample_interaction_distance(kappa_ext, rho)
                
                # Check for boundary escape
                new_position = photon.position + distance * photon.direction
                if dust_grid.outside_boundary(new_position):
                    escaped_photons.append(photon)
                    photon.alive = False
                    continue
                
                # Propagate to interaction site
                photon = propagate_photon(photon, distance)
                
                # Determine interaction type
                if determine_interaction_type(albedo):
                    # Scattering - change direction
                    photon = scatter_photon(photon, g=0.7)  # Forward scattering
                else:
                    # Absorption - deposit energy and terminate
                    grid_index = dust_grid.get_grid_index(photon.position)
                    absorbed_energy[grid_index] += photon.energy
                    photon.alive = False
    
    return escaped_photons, absorbed_energy
```

## Dust Physics and Stellar Heating

### Dust Temperature Calculation

Dust grains reach thermal equilibrium between absorbed stellar radiation and emitted thermal radiation:

**Energy Balance**:
```
L_absorbed = L_emitted
```

For a grain at distance r from a star of luminosity L:
```
πa² × (L/4πr²) × Q_abs = 4πa² σ T⁴ × Q_em
```

where:
- a: Grain radius
- Q_abs, Q_em: Absorption and emission efficiencies
- σ: Stefan-Boltzmann constant

**Dust Temperature**:
```
T_dust = [L_star × Q_abs / (16π σ r² Q_em)]^(1/4)
```

For typical interstellar grains: Q_abs ≈ Q_em, giving:
```
T_dust ≈ T_star × √(R_star / 2r)
```

### Multi-Star Heating

In stellar clusters, dust receives radiation from multiple sources:

```python
def calculate_dust_temperature(dust_position, stellar_sources):
    """Calculate dust temperature from multiple stellar sources"""
    total_heating = 0
    
    for source in stellar_sources:
        distance = np.linalg.norm(dust_position - source.position)
        
        # Heating rate from this star
        heating_rate = (source.luminosity * Q_abs(source.wavelength) / 
                       (4 * np.pi * distance**2))
        
        total_heating += heating_rate
    
    # Thermal equilibrium temperature
    T_dust = (total_heating / (4 * sigma_sb * Q_em))**(1/4)
    
    return T_dust
```

### Dust-to-Gas Ratio and Metallicity

The amount of dust scales with stellar metallicity:

```
(Dust/Gas) = (Z/Z_☉) × (Dust/Gas)_☉
```

where (Dust/Gas)_☉ ≈ 0.01 in the solar neighborhood.

**Opacity Scaling**:
```python
def dust_opacity(metallicity, wavelength, reference_opacity=1.0, reference_z=0.02):
    """Calculate dust opacity based on metallicity"""
    # Scale with metallicity
    metallicity_factor = metallicity / reference_z
    
    # Wavelength dependence
    wavelength_factor = (wavelength / 0.55e-6)**(-1.0)  # λ^(-1) law
    
    return reference_opacity * metallicity_factor * wavelength_factor
```

## Integration with N-Body Stellar Clusters

### Stellar Source Generation

Use stellar properties from Project 2 N-body simulations:

```python
def setup_cluster_sources(nbody_cluster):
    """Convert N-body cluster to MCRT stellar sources"""
    sources = []
    
    for i, star in enumerate(nbody_cluster.stars):
        # Use stellar properties from Project 1
        luminosity = star.luminosity  # From stellar physics
        temperature = star.temperature
        position = nbody_cluster.positions[i]  # From N-body evolution
        
        source = StellarSource(
            position=position,
            luminosity=luminosity,
            temperature=temperature,
            spectrum=blackbody_spectrum(temperature)
        )
        sources.append(source)
    
    return sources
```

### Dust Distribution Models

#### Uniform Dust Distribution

Simplest model: constant dust density within cluster radius:

```python
class UniformDustGrid:
    def __init__(self, cluster_radius, dust_to_gas_ratio, gas_density):
        self.radius = cluster_radius
        self.dust_density = dust_to_gas_ratio * gas_density
        
    def get_properties(self, position):
        """Get dust properties at given position"""
        r = np.linalg.norm(position)
        
        if r < self.radius:
    ### Elastic Net: Combining Ridge and Lasso

Elastic Net combines both L1 and L2 penalties, getting benefits of both methods:

```
J_elastic(β) = ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²₂
```

Often parameterized as:
```
J_elastic(β) = ||y - Xβ||² + α[ρ||β||₁ + (1-ρ)||β||²₂]
```

where α controls overall regularization strength and ρ ∈ [0,1] balances L1 vs L2.

**Advantages**:
- ρ = 0: Pure Ridge regression
- ρ = 1: Pure Lasso regression  
- 0 < ρ < 1: Combines benefits of both methods
- Handles groups of correlated features better than Lasso alone

```python
class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tolerance=1e-6):
        self.alpha = alpha          # Overall regularization strength
        self.l1_ratio = l1_ratio    # Balance between L1 and L2 (ρ parameter)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.beta = None
        self.converged = False
    
    def fit(self, X, y):
        """Fit Elastic Net using coordinate descent"""
        # Standardize features
        X_scaled, self.X_mean, self.X_std = self._standardize_features(X)
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        n, p = X_scaled.shape
        self.beta = np.zeros(p)
        
        # Separate L1 and L2 penalties
        lambda_1 = self.alpha * self.l1_ratio
        lambda_2 = self.alpha * (1 - self.l1_ratio)
        
        for iteration in range(self.max_iter):
            beta_old = self.beta.copy()
            
            for j in range(p):
                # Calculate partial residual
                partial_residual = (y_centered - X_scaled @ self.beta + 
                                  X_scaled[:, j] * self.beta[j])
                
                # Coordinate descent update
                rho_j = X_scaled[:, j] @ partial_residual
                z_j = np.sum(X_scaled[:, j]**2) + lambda_2  # Ridge penalty in denominator
                
                # Elastic net update with soft thresholding
                self.beta[j] = self._soft_threshold(rho_j / z_j, lambda_1 / z_j)
            
            # Check convergence
            max_change = np.max(np.abs(self.beta - beta_old))
            if max_change < self.tolerance:
                self.converged = True
                break
        
        return self
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator"""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def _standardize_features(self, X):
        """Standardize features to have mean 0 and std 1"""
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_scaled = (X - X_mean) / X_std
        return X_scaled, X_mean, X_std
    
    def predict(self, X):
        """Make predictions using fitted Elastic Net model"""
        X_scaled = (X - self.X_mean) / self.X_std
        return X_scaled @ self.beta + self.y_mean
```

## Advanced Feature Engineering

### Polynomial Features

Create non-linear relationships using polynomial expansions:

```python
def create_polynomial_features(X, degree=2, include_bias=True, interaction_only=False):
    """Create polynomial features up to given degree"""
    from itertools import combinations_with_replacement, combinations
    
    n_samples, n_features = X.shape
    feature_names = [f'x{i}' for i in range(n_features)]
    
    # Start with original features
    poly_features = [X]
    poly_names = feature_names.copy()
    
    # Add bias term if requested
    if include_bias:
        bias = np.ones((n_samples, 1))
        poly_features.insert(0, bias)
        poly_names.insert(0, '1')
    
    # Generate polynomial terms
    for deg in range(2, degree + 1):
        if interaction_only:
            # Only interaction terms (no x₁², x₂², etc.)
            feature_combinations = combinations(range(n_features), deg)
        else:
            # All polynomial terms including powers
            feature_combinations = combinations_with_replacement(range(n_features), deg)
        
        for combo in feature_combinations:
            # Create polynomial term
            poly_term = np.ones(n_samples)
            term_name = ""
            
            for feature_idx in combo:
                poly_term *= X[:, feature_idx]
                if term_name:
                    term_name += f"*x{feature_idx}"
                else:
                    term_name = f"x{feature_idx}"
            
            poly_features.append(poly_term.reshape(-1, 1))
            poly_names.append(term_name)
    
    # Combine all features
    X_poly = np.hstack(poly_features)
    
    return X_poly, poly_names

# Example usage for MCRT cluster data
def engineer_cluster_features(mcrt_data, polynomial_degree=2):
    """Engineer features for cluster parameter estimation"""
    # Extract basic observational features
    basic_features = create_cluster_features(mcrt_data)
    
    # Add polynomial terms for non-linear relationships
    poly_features, poly_names = create_polynomial_features(
        basic_features, degree=polynomial_degree, interaction_only=True
    )
    
    # Add domain-specific engineered features
    engineered_features = []
    engineered_names = []
    
    for obs in mcrt_data:
        additional_features = []
        
        # Color-color combinations (astronomical diagnostics)
        if 'flux_uv' in obs and 'flux_optical' in obs and 'flux_ir' in obs:
            uv_optical = np.log10(obs['flux_uv'] / obs['flux_optical'])
            optical_ir = np.log10(obs['flux_optical'] / obs['flux_ir'])
            
            additional_features.extend([uv_optical, optical_ir])
            if not engineered_names:  # Only add names once
                engineered_names.extend(['UV-Optical', 'Optical-IR'])
        
        # Spectral slope (indicates dust temperature)
        if 'flux_optical' in obs and 'flux_ir' in obs:
            spectral_slope = np.log10(obs['flux_ir'] / obs['flux_optical']) / np.log10(2.0)
            additional_features.append(spectral_slope)
            if 'spectral_slope' not in engineered_names:
                engineered_names.append('spectral_slope')
        
        # Total luminosity (cluster mass indicator)
        total_flux = sum(obs[key] for key in obs if 'flux' in key)
        if total_flux > 0:
            additional_features.append(np.log10(total_flux))
            if 'log_total_flux' not in engineered_names:
                engineered_names.append('log_total_flux')
        
        engineered_features.append(additional_features)
    
    engineered_array = np.array(engineered_features)
    
    # Combine all features
    all_features = np.hstack([poly_features, engineered_array])
    all_names = poly_names + engineered_names
    
    return all_features, all_names
```

### Feature Scaling and Normalization

Different features often have vastly different scales, requiring normalization:

```python
class FeatureScaler:
    def __init__(self, method='standardize'):
        self.method = method
        self.fitted = False
        self.feature_stats = {}
    
    def fit(self, X, feature_names=None):
        """Learn scaling parameters from training data"""
        self.feature_names = feature_names
        
        if self.method == 'standardize':
            # Z-score normalization: (x - μ) / σ
            self.feature_stats['mean'] = np.mean(X, axis=0)
            self.feature_stats['std'] = np.std(X, axis=0)
            self.feature_stats['std'][self.feature_stats['std'] == 0] = 1  # Avoid division by zero
            
        elif self.method == 'minmax':
            # Min-max scaling: (x - min) / (max - min)
            self.feature_stats['min'] = np.min(X, axis=0)
            self.feature_stats['max'] = np.max(X, axis=0)
            range_vals = self.feature_stats['max'] - self.feature_stats['min']
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            self.feature_stats['range'] = range_vals
            
        elif self.method == 'robust':
            # Robust scaling using median and IQR
            self.feature_stats['median'] = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1
            self.feature_stats['iqr'] = iqr
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Apply learned scaling to data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        if self.method == 'standardize':
            return (X - self.feature_stats['mean']) / self.feature_stats['std']
        elif self.method == 'minmax':
            return (X - self.feature_stats['min']) / self.feature_stats['range']
        elif self.method == 'robust':
            return (X - self.feature_stats['median']) / self.feature_stats['iqr']
    
    def fit_transform(self, X, feature_names=None):
        """Fit scaler and transform data in one step"""
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Reverse the scaling transformation"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        if self.method == 'standardize':
            return X_scaled * self.feature_stats['std'] + self.feature_stats['mean']
        elif self.method == 'minmax':
            return X_scaled * self.feature_stats['range'] + self.feature_stats['min']
        elif self.method == 'robust':
            return X_scaled * self.feature_stats['iqr'] + self.feature_stats['median']
```

## Model Selection and Cross-Validation

### Hyperparameter Tuning with Grid Search

```python
class GridSearchCV:
    def __init__(self, model_class, param_grid, cv_folds=5, scoring='mse'):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_params = None
        self.best_score = None
        self.cv_results = []
    
    def fit(self, X, y):
        """Perform grid search with cross-validation"""
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()
        
        best_score = float('inf') if self.scoring == 'mse' else float('-inf')
        
        for params in param_combinations:
            # Perform cross-validation for this parameter set
            cv_scores = self._cross_validate(X, y, params)
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Store results
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            }
            self.cv_results.append(result)
            
            # Update best parameters
            if ((self.scoring == 'mse' and mean_score < best_score) or
                (self.scoring == 'r2' and mean_score > best_score)):
                best_score = mean_score
                self.best_params = params
                self.best_score = mean_score
            
            print(f"Params: {params}, CV Score: {mean_score:.4f} ± {std_score:.4f}")
        
        # Fit final model with best parameters
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X, y)
        
        return self
    
    def _generate_param_combinations(self):
        """Generate all combinations of parameters"""
        from itertools import product
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _cross_validate(self, X, y, params):
        """Perform k-fold cross-validation for given parameters"""
        n_samples = len(X)
        fold_size = n_samples // self.cv_folds
        scores = []
        
        for fold in range(self.cv_folds):
            # Create train/validation splits
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.cv_folds - 1 else n_samples
            
            # Validation set
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            
            # Training set
            X_train = np.concatenate([X[:start_idx], X[end_idx:]])
            y_train = np.concatenate([y[:start_idx], y[end_idx:]])
            
            # Fit model with current parameters
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            
            if self.scoring == 'mse':
                score = np.mean((y_val - y_pred)**2)
            elif self.scoring == 'mae':
                score = np.mean(np.abs(y_val - y_pred))
            elif self.scoring == 'r2':
                ss_res = np.sum((y_val - y_pred)**2)
                ss_tot = np.sum((y_val - np.mean(y_val))**2)
                score = 1 - (ss_res / ss_tot)
            
            scores.append(score)
        
        return scores
    
    def predict(self, X):
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("Grid search must be fitted before making predictions")
        return self.best_model.predict(X)

# Example usage for MCRT cluster parameter estimation
def optimize_cluster_regression(X, y, target_name):
    """Optimize regression model for cluster parameter estimation"""
    # Define parameter grids for different models
    ridge_params = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    lasso_params = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 2000]
    }
    
    elastic_params = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # Test all models
    models_to_test = [
        ('Ridge', RidgeRegression, ridge_params),
        ('Lasso', LassoRegression, lasso_params),
        ('ElasticNet', ElasticNetRegression, elastic_params)
    ]
    
    results = {}
    
    for model_name, model_class, param_grid in models_to_test:
        print(f"\nOptimizing {model_name} for {target_name}...")
        
        grid_search = GridSearchCV(model_class, param_grid, cv_folds=5)
        grid_search.fit(X, y)
        
        results[model_name] = {
            'best_params': grid_search.best_params,
            'best_score': grid_search.best_score,
            'model': grid_search.best_model
        }
        
        print(f"Best {model_name} params: {grid_search.best_params}")
        print(f"Best CV score: {grid_search.best_score:.4f}")
    
    # Find overall best model
    best_model_name = min(results.keys(), 
                         key=lambda x: results[x]['best_score'])
    
    print(f"\nBest overall model for {target_name}: {best_model_name}")
    
    return results, best_model_name
```

### Learning Curves and Model Complexity

Understanding how model performance changes with training set size and complexity:

```python
def plot_learning_curves(X, y, model_class, model_params=None, 
                        train_sizes=None, cv_folds=3):
    """Plot learning curves to diagnose bias vs variance"""
    if model_params is None:
        model_params = {}
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    n_samples = len(X)
    train_scores = []
    val_scores = []
    
    for train_size in train_sizes:
        n_train = int(train_size * n_samples)
        
        # Perform cross-validation for this training size
        fold_train_scores = []
        fold_val_scores = []
        
        for fold in range(cv_folds):
            # Random subset of training data
            train_indices = np.random.choice(n_samples, n_train, replace=False)
            remaining_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            
            # Validation set from remaining data
            val_size = min(len(remaining_indices), n_train)
            val_indices = np.random.choice(remaining_indices, val_size, replace=False)
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            # Fit model and evaluate
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Training score
            y_train_pred = model.predict(X_train)
            train_mse = np.mean((y_train - y_train_pred)**2)
            fold_train_scores.append(train_mse)
            
            # Validation score
            y_val_pred = model.predict(X_val)
            val_mse = np.mean((y_val - y_val_pred)**2)
            fold_val_scores.append(val_mse)
        
        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    actual_train_sizes = train_sizes * n_samples
    plt.plot(actual_train_sizes, train_scores, 'o-', label='Training Score', color='blue')
    plt.plot(actual_train_sizes, val_scores, 'o-', label='Validation Score', color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Diagnose bias vs variance
    final_gap = val_scores[-1] - train_scores[-1]
    if final_gap > 0.1 * train_scores[-1]:
        diagnosis = "High variance (overfitting)"
        recommendation = "Try regularization or more training data"
    elif train_scores[-1] > 0.1:  # Arbitrary threshold
        diagnosis = "High bias (underfitting)"
        recommendation = "Try more complex model or better features"
    else:
        diagnosis = "Good bias-variance balance"
        recommendation = "Model appears well-tuned"
    
    plt.text(0.02, 0.98, f"Diagnosis: {diagnosis}\nRecommendation: {recommendation}", 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return train_scores, val_scores, actual_train_sizes

def plot_validation_curves(X, y, model_class, param_name, param_range):
    """Plot validation curves to find optimal hyperparameter"""
    train_scores = []
    val_scores = []
    
    for param_value in param_range:
        # Create model with current parameter value
        params = {param_name: param_value}
        model = model_class(**params)
        
        # Cross-validation
        train_fold_scores = []
        val_fold_scores = []
        
        n_folds = 5
        n_samples = len(X)
        fold_size = n_samples // n_folds
        
        for fold in range(n_folds):
            # Create splits
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
            
            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            X_train = np.concatenate([X[:start_idx], X[end_idx:]])
            y_train = np.concatenate([y[:start_idx], y[end_idx:]])
            
            # Fit and evaluate
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            train_fold_scores.append(np.mean((y_train - y_train_pred)**2))
            val_fold_scores.append(np.mean((y_val - y_val_pred)**2))
        
        train_scores.append(np.mean(train_fold_scores))
        val_scores.append(np.mean(val_fold_scores))
    
    # Plot validation curves
    plt.figure(figsize=(10, 6))
    
    plt.semilogx(param_range, train_scores, 'o-', label='Training Score', color='blue')
    plt.semilogx(param_range, val_scores, 'o-', label='Validation Score', color='red')
    
    plt.xlabel(f'{param_name}')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Validation Curves for {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark optimal parameter
    optimal_idx = np.argmin(val_scores)
    optimal_param = param_range[optimal_idx]
    plt.axvline(x=optimal_param, color='green', linestyle='--', 
                label=f'Optimal {param_name} = {optimal_param:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return optimal_param, train_scores, val_scores
```

## Information Criteria for Model Selection

### Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)

These criteria balance model fit with complexity:

**AIC**: AIC = 2k - 2ln(L)
**BIC**: BIC = k ln(n) - 2ln(L)

where k is number of parameters, n is sample size, and L is likelihood.

For linear regression with Gaussian errors:
**AIC**: AIC = n ln(RSS/n) + 2k
**BIC**: BIC = n ln(RSS/n) + k ln(n)

```python
def calculate_information_criteria(X, y, beta):
    """Calculate AIC and BIC for linear regression model"""
    n, p = X.shape
    
    # Calculate residual sum of squares
    y_pred = predict(X, beta)
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    
    # Number of parameters (including intercept)
    k = len(beta)
    
    # AIC and BIC calculations
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    
    # Adjusted R²
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (rss / ss_tot)
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - k - 1)
    
    return {
        'AIC': aic,
        'BIC': bic,
        'RSS': rss,
        'R²': r_squared,
        'Adjusted R²': adjusted_r_squared,
        'n_parameters': k
    }

def compare_models_by_criteria(X, y, models_dict):
    """Compare multiple models using information criteria"""
    results = {}
    
    for model_name, model in models_dict.items():
        # Get model parameters
        if hasattr(model, 'beta'):
            beta = model.beta
        else:
            # For sklearn-like models, might need different approach
            beta = getattr(model, 'coef_', None)
        
        if beta is not None:
            criteria = calculate_information_criteria(X, y, beta)
            results[model_name] = criteria
            
            print(f"\n{model_name}:")
            for criterion, value in criteria.items():
                print(f"  {criterion}: {value:.4f}")
    
    # Find best models by each criterion
    best_aic = min(results.keys(), key=lambda x: results[x]['AIC'])
    best_bic = min(results.keys(), key=lambda x: results[x]['BIC'])
    best_r2 = max(results.keys(), key=lambda x: results[x]['Adjusted R²'])
    
    print(f"\nBest models:")
    print(f"  AIC: {best_aic}")
    print(f"  BIC: {best_bic}")
    print(f"  Adjusted R²: {best_r2}")
    
    return results
```

## Practical Application: Complete MCRT Analysis Pipeline

```python
def complete_mcrt_analysis_pipeline(mcrt_observations, cluster_properties, 
                                  test_size=0.2, random_state=42):
    """Complete analysis pipeline for MCRT cluster data"""
    
    print("=== MCRT Cluster Parameter Estimation Pipeline ===\n")
    
    # 1. Feature Engineering
    print("Step 1: Feature Engineering...")
    X_raw = create_cluster_features(mcrt_observations)
    X_engineered, feature_names = engineer_cluster_features(mcrt_observations)
    
    # 2. Target Variable Creation
    print("Step 2: Creating Target Variables...")
    y = create_target_variables(cluster_properties)
    target_names = ['log_age_myr', 'log_metallicity', 'log_dust_ratio']
    
    # 3. Train-Test Split
    print("Step 3: Train-Test Split...")
    n_samples = len(X_engineered)
    n_test = int(test_size * n_samples)
    
    np.random.seed(random_state)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train, X_test = X_engineered[train_indices], X_engineered[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 4. Feature Scaling
    print("Step 4: Feature Scaling...")
    scaler = FeatureScaler(method='standardize')
    X_train_scaled = scaler.fit_transform(X_train, feature_names)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Selection and Training
    print("Step 5: Model Selection and Hyperparameter Tuning...")
    
    results = {}
    
    for i, target_name in enumerate(target_names):
        print(f"\n--- Analyzing {target_name} ---")
        
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        
        # Optimize models
        model_results, best_model_name = optimize_cluster_regression(
            X_train_scaled, y_train_target, target_name
        )
        
        # Train final model
        best_model = model_results[best_model_name]['model']
        
        # 6. Model Evaluation
        print(f"\nStep 6: Model Evaluation for {target_name}...")
        
        # Training performance
        y_train_pred = best_model.predict(X_train_scaled)
        train_mse = np.mean((y_train_target - y_train_pred)**2)
        train_r2 = 1 - np.sum((y_train_target - y_train_pred)**2) / np.sum((y_train_target - np.mean(y_train_target))**2)
        
        # Test performance
        y_test_pred = best_model.predict(X_test_scaled)
        test_mse = np.mean((y_test_target - y_test_pred)**2)
        test_r2 = 1 - np.sum((y_test_target - y_test_pred)**2) / np.sum((y_test_target - np.mean(y_test_target))**2)
        
        print(f"Training - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # 7. Residual Analysis
        print(f"Step 7: Residual Analysis for {target_name}...")
        analyze_residuals(X_test_scaled, y_test_target, best_model.beta, feature_names)
        
        # Store results
        results[target_name] = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_test_true': y_test_target,
            'y_test_pred': y_test_pred
        }
    
    # 8. Feature Importance Analysis
    print("\nStep 8: Feature Importance Analysis...")
    analyze_feature_importance(results, feature_names, target_names)
    
    # 9. Final Model Comparison
    print("\nStep 9: Final Model Summary...")
    create_final_summary(results)
    
    return results, scaler, feature_names, target_names

def analyze_feature_importance(results, feature_names, target_names):
    """Analyze feature importance across all targets"""
    
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target_name in enumerate(target_names):
        model = results[target_name]['best_model']
        
        if hasattr(model, 'beta'):
            # Get coefficients (excluding intercept)
            coefficients = model.beta[1:] if len(model.beta) > len(feature_names) else model.beta
            
            # Plot feature importance
            importance = np.abs(coefficients)
            sorted_indices = np.argsort(importance)[-10:]  # Top 10 features
            
            axes[i].barh(range(len(sorted_indices)), importance[sorted_indices])
            axes[i].set_yticks(range(len(sorted_indices)))
            axes[i].set_yticklabels([feature_names[idx] for idx in sorted_indices])
            axes[i].set_xlabel('|Coefficient|')
            axes[i].set_title(f'Feature Importance: {target_name}')
    
    plt.tight_layout()
    plt.show()

def create_final_summary(results):
    """Create final summary of all models"""
    
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    summary_data = []
    
    for target_name, result in results.items():
        summary_data.append([
            target_name,
            result['best_model_name'],
            f"{result['test_mse']:.4f}",
            f"{result['test_r2']:.4f}"
        ])
    
    headers = ['Target', 'Best Model', 'Test MSE', 'Test R²']
    
    # Simple table formatting
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *summary_data)]
    
    # Print headers
    header_row = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for row in summary_data:
        data_row = " | ".join(f"{item:<{width}}" for item, width in zip(row, col_widths))
        print(data_row)
    
    print("\n" + "="*60)
    
    # Overall assessment
    avg_r2 = np.mean([result['test_r2'] for result in results.values()])
    print(f"Average Test R² across all targets: {avg_r2:.4f}")
    
    if avg_r2 > 0.8:
        assessment = "Excellent model performance"
    elif avg_r2 > 0.6:
        assessment = "Good model performance"
    elif avg_r2 > 0.4:
        assessment = "Moderate model performance - consider feature engineering"
    else:
        assessment = "Poor model performance - revisit features and models"
    
    print(f"Overall Assessment: {assessment}")
```

This comprehensive treatment of advanced regression techniques, feature engineering, and model selection provides students with the sophisticated tools needed to extract maximum information from their MCRT synthetic observations. The emphasis on systematic model comparison and validation develops the critical evaluation skills essential for scientific computing, while the complete analysis pipeline demonstrates how to integrate multiple techniques into a coherent research workflow.

---

# Week 9: Bayesian Inference Foundations
**Learning Objectives**: Understand Bayesian probability framework, derive parameter posteriors, implement prior specification

## Philosophical Foundation: Frequentist vs Bayesian Paradigms

### The Fundamental Difference

The distinction between frequentist and Bayesian approaches represents one of the most important conceptual divides in statistical inference, with profound implications for how we interpret uncertainty and make scientific conclusions.

**Frequentist Interpretation**:
- Probability represents long-run frequency of events
- Parameters are fixed but unknown constants
- Confidence intervals capture procedure reliability, not parameter uncertainty
- Data are random, parameters are fixed

**Bayesian Interpretation**:
- Probability represents degree of belief or certainty
- Parameters are random variables with probability distributions
- Credible intervals directly quantify parameter uncertainty
- Parameters are random, data are fixed (observed)

#### Concrete Example: Stellar Age Estimation

Consider estimating the age of a stellar cluster from MCRT observations:

**Frequentist Approach**:
```
"If we repeated this measurement procedure many times on clusters 
of the same true age, 95% of the confidence intervals would contain 
the true age."
```

**Bayesian Approach**:
```
"Given our observations and prior knowledge, there is a 95% probability 
that the cluster age lies between 10 and 50 Myr."
```

The Bayesian statement is more intuitive and directly addresses the scientific question we actually care about.

### When Each Approach Excels

**Frequentist Methods Excel When**:
- Large sample sizes available
- Minimal prior information
- Objective, procedure-based analysis required
- Regulatory or standardized environments

**Bayesian Methods Excel When**:
- Incorporating prior knowledge is important
- Small sample sizes
- Complex, hierarchical problems
- Decision-making under uncertainty
- Sequential updating of beliefs

## Bayes' Theorem and the Bayesian Framework

### Bayes' Theorem Derivation

Starting from the definition of conditional probability:

```
P(A|B) = P(A ∩ B) / P(B)
P(B|A) = P(A ∩ B) / P(A)
```

Since P(A ∩ B) = P(B ∩ A), we have:
```
P(A|B) × P(B) = P(B|A) × P(A)
```

Rearranging gives **Bayes' Theorem**:
```
P(A|B) = P(B|A) × P(A) / P(B)
```

### Application to Parameter Estimation

For parameter estimation with data D and parameters θ:

```
P(θ|D) = P(D|θ) × P(θ) / P(D)
```

**Components**:
- **P(θ|D)**: Posterior distribution (what we want)
- **P(D|θ)**: Likelihood function (how well θ explains data)
- **P(θ)**: Prior distribution (what we knew before seeing data)
- **P(D)**: Marginal likelihood or evidence (normalization constant)

Since P(D) doesn't depend on θ, we often write:
```
P(θ|D) ∝ P(D|θ) × P(θ)
```

**Interpretation**: The posterior combines data evidence (likelihood) with prior beliefs, weighted by their relative strengths.

### The Likelihood Function

The likelihood L(θ|D) = P(D|θ) quantifies how well different parameter values explain the observed data.

#### Gaussian Likelihood for MCRT Data

For MCRT flux measurements with Gaussian errors:

```
D = {(F₁, σ₁), (F₂, σ₂), ..., (Fₙ, σₙ)}
```

where Fᵢ are observed fluxes and σᵢ are measurement uncertainties.

**Likelihood function**:
```
L(θ|D) = ∏ᵢ (1/√(2πσᵢ²)) exp(-(Fᵢ - F_model(θ))²/2σᵢ²)
```

**Log-likelihood** (easier to work with):
```
ln L(θ|D) = -½ Σᵢ [(Fᵢ - F_model(θ))²/σᵢ² + ln(2πσᵢ²)]
```

```python
def gaussian_log_likelihood(observed_fluxes, model_fluxes, uncertainties):
    """Calculate log-likelihood for Gaussian-distributed measurements"""
    residuals = observed_fluxes - model_fluxes
    
    # Chi-squared term
    chi_squared = np.sum((residuals / uncertainties)**2)
    
    # Normalization term
    normalization = np.sum(np.log(2 * np.pi * uncertainties**2))
    
    log_likelihood = -0.5 * (chi_squared + normalization)
    
    return log_likelihood

def cluster_log_likelihood(parameters, mcrt_data):
    """Log-likelihood for stellar cluster parameters"""
    age, metallicity, dust_ratio = parameters
    
    total_log_likelihood = 0
    
    for observation in mcrt_data:
        # Calculate model prediction for this observation
        model_fluxes = calculate_model_fluxes(age, metallicity, dust_ratio, 
                                            observation['wavelengths'])
        
        observed_fluxes = observation['fluxes']
        uncertainties = observation['errors']
        
        # Add to total log-likelihood
        obs_log_likelihood = gaussian_log_likelihood(observed_fluxes, 
                                                   model_fluxes, 
                                                   uncertainties)
        total_log_likelihood += obs_log_likelihood
    
    return total_log_likelihood
```

## Prior Distribution Selection

### Types of Priors

#### Informative Priors

Based on previous knowledge or physical constraints:

```python
def create_cluster_informative_priors():
    """Create physically motivated priors for cluster parameters"""
    
    priors = {}
    
    # Cluster age: Log-normal prior
    # Most clusters are young, but some can be very old
    # Prior: log(age/Myr) ~ N(1.0, 1.0) [peaked around 10 Myr]
    def log_age_prior(log_age_myr):
        if log_age_myr < -1 or log_age_myr > 4:  # 0.1 Myr to 10 Gyr
            return -np.inf
        mean_log_age = 1.0  # 10 Myr
        std_log_age = 1.0
        return -0.5 * ((log_age_myr - mean_log_age) / std_log_age)**2
    
    # Metallicity: Log-normal based on galactic enrichment
    # Prior: log(Z/Z_sun) ~ N(-0.5, 0.5) [slightly subsolar]
    def log_metallicity_prior(log_z):
        if log_z < -3 or log_z > 0.5:  # Very metal poor to super solar
            return -np.inf
        mean_log_z = -0.5  # Half solar
        std_log_z = 0.5
        return -0.5 * ((log_z - mean_log_z) / std_log_z)**2
    
    # Dust-to-gas ratio: Correlated with metallicity
    # Prior: log(dust/gas) ~ N(log(Z/Z_sun) + log(0.01), 0.3)
    def log_dust_ratio_prior(log_dust_ratio, log_z):
        expected_log_dust = log_z + np.log10(0.01)  # Dust scales with metals
        std_log_dust = 0.3
        return -0.5 * ((log_dust_ratio - expected_log_dust) / std_log_dust)**2
    
    priors['log_age'] = log_age_prior
    priors['log_metallicity'] = log_metallicity_prior
    priors['log_dust_ratio'] = log_dust_ratio_prior
    
    return priors

def evaluate_joint_prior(parameters):
    """Evaluate joint prior for all cluster parameters"""
    log_age, log_metallicity, log_dust_ratio = parameters
    
    priors = create_cluster_informative_priors()
    
    # Independent priors
    log_prior = (priors['log_age'](log_age) + 
                priors['log_metallicity'](log_metallicity))
    
    # Correlated dust prior
    log_prior += priors['log_dust_ratio'](log_dust_ratio, log_metallicity)
    
    return log_prior
```

#### Non-informative (Flat) Priors

When prior knowledge is minimal:

```python
def create_flat_priors():
    """Create flat (uninformative) priors with physical bounds"""
    
    def flat_log_age_prior(log_age_myr):
        # Flat in log space from 0.1 Myr to 13.8 Gyr (age of universe)
        if -1 <= log_age_myr <= 4.14:  # log10(13800)
            return 0.0  # Flat (constant)
        else:
            return -np.inf  # Impossible values
    
    def flat_log_metallicity_prior(log_z):
        # Flat from 1/1000 solar to 3x solar
        if -3 <= log_z <= 0.5:
            return 0.0
        else:
            return -np.inf
    
    def flat_log_dust_ratio_prior(log_dust_ratio):
        # Flat in log space for dust-to-gas ratio
        if -4 <= log_dust_ratio <= -1:  # 0.0001 to 0.1
            return 0.0
        else:
            return -np.inf
    
    return {
        'log_age': flat_log_age_prior,
        'log_metallicity': flat_log_metallicity_prior,
        'log_dust_ratio': flat_log_dust_ratio_prior
    }
```

#### Jeffreys Priors

Scale-invariant priors for parameters that span many orders of magnitude:

```python
def jeffreys_prior(parameter, min_val, max_val):
    """Jeffreys prior: uniform in log space"""
    if min_val <= parameter <= max_val:
        return -np.log(parameter * np.log(max_val / min_val))
    else:
        return -np.inf

def create_jeffreys_priors():
    """Create Jeffreys priors for cluster parameters"""
    
    def jeffreys_age_prior(age_myr):
        return jeffreys_prior(age_myr, 0.1, 13800)  # 0.1 Myr to age of universe
    
    def jeffreys_metallicity_prior(z_ratio):
        return jeffreys_prior(z_ratio, 0.001, 3.0)  # 0.1% to 300% solar
    
    def jeffreys_dust_ratio_prior(dust_gas_ratio):
        return jeffreys_prior(dust_gas_ratio, 0.0001, 0.1)
    
    return {
        'age': jeffreys_age_prior,
        'metallicity': jeffreys_metallicity_prior,
        'dust_ratio': jeffreys_dust_ratio_prior
    }
```

### Prior Sensitivity Analysis

Understanding how prior choices affect conclusions:

```python
def prior_sensitivity_analysis(mcrt_data, prior_types=['informative', 'flat', 'jeffreys']):
    """Analyze sensitivity of results to prior choice"""
    
    results = {}
    
    for prior_type in prior_types:
        print(f"\nAnalyzing with {prior_type} priors...")
        
        # Create appropriate prior functions
        if prior_type == 'informative':
            priors = create_cluster_informative_priors()
        elif prior_type == 'flat':
            priors = create_flat_priors()
        elif prior_type == 'jeffreys':
            priors = create_jeffreys_priors()
        
        # Run MCMC sampling (will implement in next section)
        posterior_samples = run_mcmc_sampling(mcrt_data, priors, n_samples=5000)
        
        # Calculate posterior statistics
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_std = np.std(posterior_samples, axis=0)
        posterior_percentiles = np.percentile(posterior_samples, [16, 50, 84], axis=0)
        
        results[prior_type] = {
            'samples': posterior_samples,
            'mean': posterior_mean,
            'std': posterior_std,
            'percentiles': posterior_percentiles
        }
        
        print(f"Posterior means: {posterior_mean}")
        print(f"Posterior stds: {posterior_std}")
    
    # Compare results across prior types
    print("\n" + "="*50)
    print("PRIOR SENSITIVITY ANALYSIS")
    print("="*50)
    
    param_names = ['log_age', 'log_metallicity', 'log_dust_ratio']
    
    for i, param in enumerate(param_names):
        print(f"\n{param}:")
        for prior_type in prior_types:
            mean = results[prior_type]['mean'][i]
            std = results[prior_type]['std'][i]
            print(f"  {prior_type}: {mean:.3f} ± {std:.3f}")
        
        # Calculate maximum difference
        means = [results[pt]['mean'][i] for pt in prior_types]
        max_diff = max(means) - min(means)
        print(f"  Max difference: {max_diff:.3f}")
        
        if max_diff < 0.1:
            sensitivity = "Low sensitivity - data dominate"
        elif max_diff < 0.5:
            sensitivity = "Moderate sensitivity"
        else:
            sensitivity = "High sensitivity - prior choice matters"
        
        print(f"  Assessment: {sensitivity}")
    
    return results
```

## Posterior Distribution Properties

### Conjugate Priors

When prior and posterior have the same functional form, enabling analytical solutions:

#### Gaussian-Gaussian Conjugacy

For Gaussian likelihood with Gaussian prior:

**Prior**: θ ~ N(μ₀, σ₀²)
**Likelihood**: x|θ ~ N(θ, σ²)
**Posterior**: θ|x ~ N(μ₁, σ₁²)

where:
```
μ₁ = (σ²μ₀ + σ₀²x) / (σ² + σ₀²)
σ₁² = (σ²σ₀²) / (σ² + σ₀²)
```

```python
def gaussian_conjugate_update(prior_mean, prior_var, data, data_var):
    """Analytical Bayesian update for Gaussian-Gaussian conjugacy"""
    
    # Precision (inverse variance) formulation often clearer
    prior_precision = 1 / prior_var
    data_precision = 1 / data_var
    
    # Posterior precision is sum of precisions
    posterior_precision = prior_precision + data_precision
    posterior_var = 1 / posterior_precision
    
    # Posterior mean is precision-weighted average
    posterior_mean = ((prior_precision * prior_mean + data_precision * data) / 
                     posterior_precision)
    
    return posterior_mean, posterior_var

def demonstrate_conjugate_updating():
    """Demonstrate sequential Bayesian updating"""
    
    # Start with prior
    prior_mean = 1.0  # log(age) = 10 Myr
    prior_var = 1.0
    
    # Sequential observations
    observations = [1.2, 0.8, 1.1, 0.9, 1.0]
    observation_var = 0.1
    
    current_mean = prior_mean
    current_var = prior_var
    
    print("Sequential Bayesian updating:")
    print(f"Prior: μ = {prior_mean:.3f}, σ² = {prior_var:.3f}")
    
    for i, obs in enumerate(observations):
        current_mean, current_var = gaussian_conjugate_update(
            current_mean, current_var, obs, observation_var
        )
        
        print(f"After obs {i+1} ({obs:.1f}): μ = {current_mean:.3f}, σ² = {current_var:.3f}")
    
    # Compare with batch update
    batch_mean = np.mean(observations)
    batch_var = observation_var / len(observations)
    
    batch_posterior_mean, batch_posterior_var = gaussian_conjugate_update(
        prior_mean, prior_var, batch_mean, batch_var
    )
    
    print(f"\nBatch update result: μ = {batch_posterior_mean:.3f}, σ² = {batch_posterior_var:.3f}")
    print(f"Sequential result:   μ = {current_mean:.3f}, σ² = {current_var:.3f}")
    print(f"Difference: {abs(batch_posterior_mean - current_mean):.6f}")
```

### Marginal and Conditional Distributions

For multi-parameter problems, we often want marginal distributions for individual parameters:

```python
def calculate_marginal_distributions(posterior_samples, parameter_names):
    """Calculate marginal distributions from joint posterior samples"""
    
    marginals = {}
    
    for i, param_name in enumerate(parameter_names):
        samples = posterior_samples[:, i]
        
        # Calculate marginal statistics
        marginal_mean = np.mean(samples)
        marginal_std = np.std(samples)
        marginal_median = np.median(samples)
        
        # Credible intervals
        ci_16, ci_84 = np.percentile(samples, [16, 84])
        ci_2p5, ci_97p5 = np.percentile(samples, [2.5, 97.5])
        
        marginals[param_name] = {
            'samples': samples,
            'mean': marginal_mean,
            'std': marginal_std,
            'median': marginal_median,
            '68_ci': (ci_16, ci_84),
            '95_ci': (ci_2p5, ci_97p5)
        }
        
        print(f"\n{param_name} marginal distribution:")
        print(f"  Mean: {marginal_mean:.3f}")
        print(f"  Median: {marginal_median:.3f}")
        print(f"  Std: {marginal_std:.3f}")
        print(f"  68% CI: ({ci_16:.3f}, {ci_84:.3f})")
        print(f"  95% CI: ({ci_2p5:.3f}, {ci_97p5:.3f})")
    
    return marginals

def calculate_parameter_correlations(posterior_samples, parameter_names):
    """Calculate correlations between parameters"""
    
    correlation_matrix = np.corrcoef(posterior_samples.T)
    
    print("\nParameter correlation matrix:")
    print("", " ".join(f"{name:>8}" for name in parameter_names))
    
    for i, param1 in enumerate(parameter_names):
        row_str = f"{param1:>8}"
        for j, param2 in enumerate(parameter_names):
            row_str += f"{correlation_matrix[i,j]:>8.3f}"
        print(row_str)
    
    # Identify strong correlations
    strong_correlations = []
    for i in range(len(parameter_names)):
        for j in range(i+1, len(parameter_names)):
            corr = correlation_matrix[i, j]
            if abs(corr) > 0.5:
                strong_correlations.append((parameter_names[i], parameter_names[j], corr))
    
    if strong_correlations:
        print("\nStrong correlations (|r| > 0.5):")
        for param1, param2, corr in strong_correlations:
            print(f"  {param1} - {param2}: r = {corr:.3f}")
    
    return correlation_matrix
```

## Model Comparison and Bayesian Evidence

### Marginal Likelihood (Evidence)

The marginal likelihood P(D) quantifies how well a model explains the data:

```
P(D|M) = ∫ P(D|θ,M) P(θ|M) dθ
```

This integral over the entire parameter space is often intractable, but crucial for model comparison.

#### Bayes Factors

Compare models M₁ and M₂ using the Bayes factor:

```
BF₁₂ = P(D|M₁) / P(D|M₂)
```

**Interpretation**:
- BF₁₂ > 10: Strong evidence for M₁
- 3 < BF₁₂ < 10: Moderate evidence for M₁  
- 1/3 < BF₁₂ < 3: Weak evidence either way
- BF₁₂ < 1/10: Strong evidence for M₂

#### Harmonic Mean Estimator

Approximate marginal likelihood from MCMC samples:

```python
def harmonic_mean_estimator(log_likelihood_samples):
    """Estimate log marginal likelihood using harmonic mean"""
    # WARNING: This estimator can be unstable!
    
    # Convert to likelihood values
    max_log_likelihood = np.max(log_likelihood_samples)
    likelihood_samples = np.exp(log_likelihood_samples - max_log_likelihood)
    
    # Harmonic mean of likelihoods
    harmonic_mean = len(likelihood_samples) / np.sum(1 / likelihood_samples)
    
    # Convert back to log space
    log_marginal_likelihood = np.log(harmonic_mean) + max_log_likelihood
    
    return log_marginal_likelihood

def calculate_bayes_factor(log_ml_1, log_ml_2):
    """Calculate Bayes factor from log marginal likelihoods"""
    log_bayes_factor = log_ml_1 - log_ml_2
    bayes_factor = np.exp(log_bayes_factor)
    
    # Interpretation
    if bayes_factor > 10:
        interpretation = "Strong evidence for Model 1"
    elif bayes_factor > 3:
        interpretation = "Moderate evidence for Model 1"
    elif bayes_factor > 1/3:
        interpretation = "Weak evidence either way"
    elif bayes_factor > 1/10:
        interpretation = "Moderate evidence for Model 2"
    else:
        interpretation = "Strong evidence for Model 2"
    
    return bayes_factor, interpretation
```

## Posterior Predictive Checking

### Validating Model Assumptions

Generate synthetic data from the posterior to check model adequacy:

```python
def posterior_predictive_check(posterior_samples, mcrt_data, n_replications=100):
    """Perform posterior predictive checks"""
    
    observed_statistics = calculate_data_statistics(mcrt_data)
    predicted_statistics = []
    
    # Generate replicated datasets
    for rep in range(n_replications):
        # Sample parameters from posterior
        sample_idx = np.random.randint(len(posterior_samples))
        sampled_params = posterior_samples[sample_idx]
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_mcrt_data(sampled_params, len(mcrt_data))
        
        # Calculate test statistics
        rep_statistics = calculate_data_statistics(synthetic_data)
        predicted_statistics.append(rep_statistics)
    
    predicted_statistics = np.array(predicted_statistics)
    
    # Compare observed vs predicted
    print("Posterior Predictive Check Results:")
    print("="*40)
    
    statistic_names = ['mean_flux', 'flux_variance', 'max_flux', 'flux_skewness']
    
    for i, stat_name in enumerate(statistic_names):
        observed = observed_statistics[i]
        predicted_mean = np.mean(predicted_statistics[:, i])
        predicted_std = np.std(predicted_statistics[:, i])
        
        # P-value: fraction of predictions more extreme than observed
        p_value = np.mean(predicted_statistics[:, i] >= observed)
        
        print(f"\n{stat_name}:")
        print(f"  Observed: {observed:.3f}")
        print(f"  Predicted: {predicted_mean:.3f} ± {predicted_std:.3f}")
        print(f"  P-value: {p_value:.3f}")
        
        if p_value < 0.05 or p_value > 0.95:
            print(f"  WARNING: Possible model inadequacy!")
    
    return predicted_statistics

def calculate_data_statistics(mcrt_data):
    """Calculate summary statistics for model checking"""
    
    all_fluxes = []
    for obs in mcrt_data:
        all_fluxes.extend(obs['fluxes'])
    
    all_fluxes = np.array(all_fluxes)
    
    statistics = [
        np.mean(all_fluxes),           # Mean flux
        np.var(all_fluxes),            # Flux variance
        np.max(all_fluxes),            # Maximum flux
        scipy.stats.skew(all_fluxes)   # Skewness
    ]
    
    return statistics
```

This comprehensive foundation in Bayesian inference provides students with both the theoretical understanding and practical tools needed to implement MCMC sampling in the following week. The emphasis on prior specification, model comparison, and posterior validation develops the critical thinking skills essential for rigorous Bayesian analysis of astronomical data.

---

# Week 10: MCMC Implementation & Convergence Diagnostics
**Learning Objectives**: Implement Metropolis-Hastings algorithm, diagnose chain convergence, understand sampling efficiency

## Markov Chain Monte Carlo Fundamentals

### The MCMC Concept

MCMC methods solve the fundamental challenge of Bayesian inference: how to sample from complex, high-dimensional posterior distributions that cannot be computed analytically. The key insight is to construct a Markov chain whose stationary distribution is the target posterior.

**Markov Property**: The next state depends only on the current state, not the entire history:
```
P(θₜ₊₁ | θₜ, θₜ₋₁, ..., θ₁) = P(θₜ₊₁ | θₜ)
```

**Detailed Balance Condition**: For the chain to have the target distribution π(θ) as its stationary distribution:
```
π(θ) × P(θ → θ') = π(θ') × P(θ' → θ)
```

This ensures that the chain, once converged, samples from the correct distribution.

### Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm provides a general framework for constructing MCMC chains:

**Algorithm Steps**:
1. Start with initial parameter value θ₀
2. For iteration t:
   - Propose new state θ' from proposal distribution q(θ'|θₜ)
   - Calculate acceptance probability α
   - Accept θ' with probability α, otherwise keep θₜ
   - Set θₜ₊₁ accordingly

**Acceptance Probability**:
```
α = min(1, [π(θ') × q(θₜ|θ')] / [π(θₜ) × q(θ'|θₜ)])
```

For symmetric proposals (q(θ'|θₜ) = q(θₜ|θ')), this simplifies to:
```
α = min(1, π(θ') / π(θₜ))
```

```python
class MetropolisHastingsSampler:
    def __init__(self, log_posterior_fn, proposal_cov, parameter_names=None):
        self.log_posterior_fn = log_posterior_fn
        self.proposal_cov = np.array(proposal_cov)
        self.parameter_names = parameter_names
        self.chain = None
        self.log_posterior_values = None
        self.acceptance_rate = 0.0
        self.n_accepted = 0
        self.n_proposals = 0
    
    def sample(self, n_samples, initial_params, burn_in=1000):
        """Run Metropolis-Hastings sampler"""
        
        n_params = len(initial_params)
        
        # Initialize chain storage
        total_samples = n_samples + burn_in
        self.chain = np.zeros((total_samples, n_params))
        self.log_posterior_values = np.zeros(total_samples)
        
        # Start chain
        current_params = np.array(initial_params)
        current_log_posterior = self.log_posterior_fn(current_params)
        
        self.chain[0] = current_params
        self.log_posterior_values[0] = current_log_posterior
        
        # MCMC loop
        for i in range(1, total_samples):
            # Propose new parameters
            proposal = self._propose_step(current_params)
            
            # Calculate acceptance probability
            proposal_log_posterior = self.log_posterior_fn(proposal)
            
            if np.isfinite(proposal_log_posterior):
                log_alpha = proposal_log_posterior - current_log_posterior
                alpha = min(1.0, np.exp(log_alpha))
            else:
                alpha = 0.0  # Reject if posterior is undefined
            
            # Accept or reject
            if np.random.random() < alpha:
                current_params = proposal
                current_log_posterior = proposal_log_posterior
                self.n_accepted += 1
            
            self.n_proposals += 1
            
            # Store state
            self.chain[i] = current_params
            self.log_posterior_values[i] = current_log_posterior
            
            # Progress reporting
            if (i + 1) % 1000 == 0:
                current_acceptance = self.n_accepted / self.n_proposals
                print(f"Iteration {i+1}/{total_samples}, "
                      f"Acceptance rate: {current_acceptance:.3f}")
        
        # Calculate final acceptance rate
        self.acceptance_rate = self.n_accepted / self.n_proposals
        
        # Remove burn-in
        self.chain = self.chain[burn_in:]
        self.log_posterior_values = self.log_posterior_values[burn_in:]
        
        print(f"\nSampling completed!")
        print(f"Final acceptance rate: {self.acceptance_rate:.3f}")
        
        return self.chain
    
    def _propose_step(self, current_params):
        """Generate proposal using multivariate normal"""
        return np.random.multivariate_normal(current_params, self.proposal_cov)
    
    def get_chain_statistics(self):
        """Calculate basic chain statistics"""
        if self.chain is None:
            raise ValueError("No chain available. Run sampling first.")
        
        stats = {}
        for i, param_name in enumerate(self.parameter_names or range(len(self.chain[0]))):
            param_chain = self.chain[:, i]
            
            stats[param_name] = {
                'mean': np.mean(param_chain),
                'std': np.std(param_chain),
                'median': np.median(param_chain),
                'q16': np.percentile(param_chain, 16),
                'q84': np.percentile(param_chain, 84),
                'q2.5': np.percentile(param_chain, 2.5),
                'q97.5': np.percentile(param_chain, 97.5)
            }
        
        return stats
```

### Proposal Distribution Tuning

The efficiency of MCMC depends critically on the proposal distribution. Poor choices lead to either very low acceptance rates (proposals too large) or very slow mixing (proposals too small).

#### Adaptive Proposal Scaling

```python
class AdaptiveMetropolisHastings(MetropolisHastingsSampler):
    def __init__(self, log_posterior_fn, initial_proposal_cov, parameter_names=None,
                 target_acceptance=0.44, adaptation_window=100):
        super().__init__(log_posterior_fn, initial_proposal_cov, parameter_names)
        self.target_acceptance = target_acceptance
        self.adaptation_window = adaptation_window
        self.adaptation_history = []
    
    def sample(self, n_samples, initial_params, burn_in=1000, adapt_until=None):
        """Sample with adaptive proposal tuning"""
        
        if adapt_until is None:
            adapt_until = burn_in
        
        n_params = len(initial_params)
        total_samples = n_samples + burn_in
        
        # Initialize storage
        self.chain = np.zeros((total_samples, n_params))
        self.log_posterior_values = np.zeros(total_samples)
        
        # Start chain
        current_params = np.array(initial_params)
        current_log_posterior = self.log_posterior_fn(current_params)
        
        self.chain[0] = current_params
        self.log_posterior_values[0] = current_log_posterior
        
        # Adaptation tracking
        recent_accepts = 0
        
        for i in range(1, total_samples):
            # Propose and evaluate
            proposal = self._propose_step(current_params)
            proposal_log_posterior = self.log_posterior_fn(proposal)
            
            # Accept/reject step
            accepted = False
            if np.isfinite(proposal_log_posterior):
                log_alpha = proposal_log_posterior - current_log_posterior
                alpha = min(1.0, np.exp(log_alpha))
                
                if np.random.random() < alpha:
                    current_params = proposal
                    current_log_posterior = proposal_log_posterior
                    self.n_accepted += 1
                    recent_accepts += 1
                    accepted = True
            
            self.n_proposals += 1
            
            # Store state
            self.chain[i] = current_params
            self.log_posterior_values[i] = current_log_posterior
            
            # Adaptive tuning during burn-in
            if i < adapt_until and (i + 1) % self.adaptation_window == 0:
                recent_acceptance = recent_accepts / self.adaptation_window
                self._adapt_proposal_covariance(recent_acceptance, i)
                recent_accepts = 0
        
        # Final statistics
        self.acceptance_rate = self.n_accepted / self.n_proposals
        
        # Remove burn-in
        self.chain = self.chain[burn_in:]
        self.log_posterior_values = self.log_posterior_values[burn_in:]
        
        return self.chain
    
    def _adapt_proposal_covariance(self, current_acceptance, iteration):
        """Adapt proposal covariance based on acceptance rate"""
        
        # Simple scaling adaptation
        if current_acceptance < self.target_acceptance - 0.05:
            # Acceptance too low, reduce proposal scale
            scale_factor = 0.9
        elif current_acceptance > self.target_acceptance + 0.05:
            # Acceptance too high, increase proposal scale
            scale_factor = 1.1
        else:
            # Acceptance rate acceptable
            scale_factor = 1.0
        
        self.proposal_cov *= scale_factor**2
        
        self.adaptation_history.append({
            'iteration': iteration,
            'acceptance_rate': current_acceptance,
            'scale_factor': scale_factor,
            'proposal_scale': np.sqrt(np.trace(self.proposal_cov))
        })
        
        if iteration % 1000 == 0:
            print(f"Adaptation at iteration {iteration}: "
                  f"acceptance = {current_acceptance:.3f}, "
                  f"scale_factor = {scale_factor:.3f}")
```

### Gibbs Sampling for Conditional Distributions

When the posterior can be factored into conditional distributions that are easy to sample from:

```python
def gibbs_sampler_example(data, n_samples, burn_in=1000):
    """Example Gibbs sampler for hierarchical model"""
    
    # Simple hierarchical model:
    # y_i ~ N(μ, σ²)
    # μ ~ N(μ₀, τ²)  
    # σ² ~ IG(α, β)
    
    n_data = len(data)
    
    # Priors
    mu_0 = 0.0
    tau_squared = 10.0
    alpha = 2.0
    beta = 1.0
    
    # Initialize parameters
    mu = np.mean(data)
    sigma_squared = np.var(data)
    
    # Storage
    total_samples = n_samples + burn_in
    mu_samples = np.zeros(total_samples)
    sigma_squared_samples = np.zeros(total_samples)
    
    for i in range(total_samples):
        # Sample μ | σ², data
        precision_mu = 1/tau_squared + n_data/sigma_squared
        mean_mu = (mu_0/tau_squared + np.sum(data)/sigma_squared) / precision_mu
        var_mu = 1/precision_mu
        
        mu = np.random.normal(mean_mu, np.sqrt(var_mu))
        
        # Sample σ² | μ, data
        alpha_post = alpha + n_data/2
        beta_post = beta + 0.5 * np.sum((data - mu)**2)
        
        sigma_squared = 1 / np.random.gamma(alpha_post, 1/beta_post)
        
        # Store samples
        mu_samples[i] = mu
        sigma_squared_samples[i] = sigma_squared
    
    # Remove burn-in
    mu_samples = mu_samples[burn_in:]
    sigma_squared_samples = sigma_squared_samples[burn_in:]
    
    return mu_samples, sigma_squared_samples
```

## Convergence Diagnostics

### Visual Diagnostics

#### Trace Plots

```python
def plot_trace_diagnostics(chain, parameter_names=None, burn_in_shown=True):
    """Plot trace plots for visual convergence assessment"""
    
    n_params = chain.shape[1]
    if parameter_names is None:
        parameter_names = [f'Parameter {i}' for i in range(n_params)]
    
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, ax) in enumerate(zip(parameter_names, axes)):
        ax.plot(chain[:, i], alpha=0.8)
        ax.set_ylabel(param_name)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        param_mean = np.mean(chain[:, i])
        param_std = np.std(chain[:, i])
        ax.axhline(param_mean, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {param_mean:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle('MCMC Trace Plots', y=1.02)
    plt.show()

def plot_autocorrelation_diagnostics(chain, parameter_names=None, max_lag=100):
    """Plot autocorrelation functions"""
    
    n_params = chain.shape[1]
    if parameter_names is None:
        parameter_names = [f'Parameter {i}' for i in range(n_params)]
    
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, ax) in enumerate(zip(parameter_names, axes)):
        # Calculate autocorrelation
        autocorr = calculate_autocorrelation(chain[:, i], max_lag)
        lags = np.arange(len(autocorr))
        
        ax.plot(lags, autocorr, 'b-', alpha=0.8)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(0.1, color='red', linestyle='--', alpha=0.7, 
                  label='Threshold (0.1)')
        
        # Estimate autocorrelation time
        tau_int = estimate_autocorr_time(autocorr)
        ax.axvline(tau_int, color='green', linestyle='--', alpha=0.7,
                  label=f'τ_int ≈ {tau_int:.1f}')
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_autocorrelation(x, max_lag):
    """Calculate autocorrelation function"""
    n = len(x)
    x_centered = x - np.mean(x)
    
    autocorr = np.zeros(max_lag + 1)
    autocorr[0] = 1.0
    
    for lag in range(1, max_lag + 1):
        if lag < n:
            numerator = np.sum(x_centered[:-lag] * x_centered[lag:])
            denominator = np.sum(x_centered**2)
            autocorr[lag] = numerator / denominator
        else:
            autocorr[lag] = 0.0
    
    return autocorr

def estimate_autocorr_time(autocorr, threshold=0.1):
    """Estimate integrated autocorrelation time"""
    # Find first lag where autocorrelation drops below threshold
    below_threshold = np.where(autocorr < threshold)[0]
    
    if len(below_threshold) > 0:
        cutoff = below_threshold[0]
    else:
        cutoff = len(autocorr) - 1
    
    # Integrated autocorrelation time
    tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
    
    return tau_int
```

### Quantitative Convergence Tests

#### Gelman-Rubin Statistic (R̂)

Compare within-chain and between-chain variance for multiple chains:

```python
def gelman_rubin_diagnostic(chains):
    """Calculate Gelman-Rubin R-hat statistic"""
    
    # chains should be array of shape (n_chains, n_samples, n_params)
    n_chains, n_samples, n_params = chains.shape
    
    r_hat_values = np.zeros(n_params)
    
    for param_idx in range(n_params):
        param_chains = chains[:, :, param_idx]  # Shape: (n_chains, n_samples)
        
        # Calculate means for each chain
        chain_means = np.mean(param_chains, axis=1)  # Shape: (n_chains,)
        
        # Overall mean
        overall_mean = np.mean(chain_means)
        
        # Between-chain variance
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        chain_variances = np.var(param_chains, axis=1, ddof=1)
        W = np.mean(chain_variances)
        
        # Variance estimate
        var_plus = ((n_samples - 1) * W + B) / n_samples
        
        # R-hat statistic
        if W > 0:
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = np.inf
        
        r_hat_values[param_idx] = r_hat
    
    return r_hat_values

def run_multiple_chains(log_posterior_fn, n_chains=4, n_samples=2000, 
                       initial_params_list=None, proposal_cov=None):
    """Run multiple MCMC chains for convergence diagnosis"""
    
    if initial_params_list is None:
        # Generate overdispersed starting values
        n_params = len(proposal_cov)
        initial_params_list = []
        for i in range(n_chains):
            # Start chains at different locations
            start_params = np.random.normal(0, 2, n_params)
            initial_params_list.append(start_params)
    
    chains = []
    samplers = []
    
    for i, initial_params in enumerate(initial_params_list):
        print(f"\nRunning chain {i+1}/{n_chains}...")
        
        sampler = AdaptiveMetropolisHastings(log_posterior_fn, proposal_cov)
        chain = sampler.sample(n_samples, initial_params, burn_in=1000)
        
        chains.append(chain)
        samplers.append(sampler)
        
        print(f"Chain {i+1} acceptance rate: {sampler.acceptance_rate:.3f}")
    
    # Convert to array format for diagnostics
    chains_array = np.array(chains)
    
    # Calculate R-hat statistics
    r_hat_values = gelman_rubin_diagnostic(chains_array)
    
    print(f"\nGelman-Rubin R-hat statistics:")
    for i, r_hat in enumerate(r_hat_values):
        status = "✓" if r_hat < 1.1 else "⚠" if r_hat < 1.2 else "✗"
        print(f"  Parameter {i}: R̂ = {r_hat:.4f} {status}")
    
    # Overall convergence assessment
    max_r_hat = np.max(r_hat_values)
    if max_r_hat < 1.1:
        convergence_status = "Excellent convergence"
    elif max_r_hat < 1.2:
        convergence_status = "Acceptable convergence"
    else:
        convergence_status = "Poor convergence - run longer chains"
    
    print(f"\nOverall assessment: {convergence_status}")
    
    return chains_array, r_hat_values, samplers
```

#### Effective Sample Size

```python
def calculate_effective_sample_size(chain):
    """Calculate effective sample size accounting for autocorrelation"""
    
    n_samples, n_params = chain.shape
    eff_sample_sizes = np.zeros(n_params)
    
    for i in range(n_params):
        param_chain = chain[:, i]
        
        # Calculate autocorrelation time
        autocorr = calculate_autocorrelation(param_chain, min(n_samples//4, 200))
        tau_int = estimate_autocorr_time(autocorr)
        
        # Effective sample size
        if tau_int > 0:
            eff_n = n_samples / (2 * tau_int + 1)
        else:
            eff_n = n_samples
        
        eff_sample_sizes[i] = eff_n
    
    return eff_sample_sizes

def assess_chain_quality(chain, parameter_names=None):
    """Comprehensive chain quality assessment"""
    
    n_samples, n_params = chain.shape
    if parameter_names is None:
        parameter_names = [f'Parameter {i}' for i in range(n_params)]
    
    print("MCMC Chain Quality Assessment")
    print("="*50)
    print(f"Total samples: {n_samples}")
    
    # Effective sample sizes
    eff_sizes = calculate_effective_sample_size(chain)
    
    print(f"\nEffective Sample Sizes:")
    for i, (name, eff_size) in enumerate(zip(parameter_names, eff_sizes)):
        efficiency = eff_size / n_samples
        print(f"  {name}: {eff_size:.0f} ({efficiency:.1%} efficiency)")
        
        if efficiency < 0.1:
            print(f"    ⚠ Low efficiency - consider improving proposal")
        elif efficiency > 0.5:
            print(f"    ✓ Good efficiency")
    
    # Monte Carlo error estimates
    print(f"\nMonte Carlo Standard Errors:")
    for i, (name, eff_size) in enumerate(zip(parameter_names, eff_sizes)):
        param_std = np.std(chain[:, i])
        mc_error = param_std / np.sqrt(eff_size)
        relative_error = mc_error / param_std
        
        print(f"  {name}: {mc_error:.4f} ({relative_error:.1%} of posterior std)")
        
        if relative_error > 0.05:
            print(f"    ⚠ High MC error - need more samples")
        else:
            print(f"    ✓ Acceptable MC error")
    
    return eff_sizes
```

## Application to MCRT Cluster Data

### Complete MCMC Implementation

```python
def setup_cluster_mcmc_analysis(mcrt_observations, cluster_properties):
    """Set up complete MCMC analysis for cluster parameters"""
    
    # Extract target parameters (known truth for validation)
    true_params = []
    for cluster in cluster_properties:
        true_params.append([
            np.log10(cluster['age_myr']),
            np.log10(cluster['metallicity']),
            np.log10(cluster['dust_to_gas_ratio'])
        ])
    
    true_params = np.array(true_params)
    
    # Define log posterior function
    def log_posterior(params):
        """Log posterior for cluster parameters"""
        log_age, log_metallicity, log_dust_ratio = params
        
        # Prior
        log_prior = evaluate_joint_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        
        # Likelihood
        log_likelihood = cluster_log_likelihood(params, mcrt_observations)
        
        return log_prior + log_likelihood
    
    # Set up proposal covariance (tune based on parameter scales)
    initial_proposal_cov = np.diag([0.1, 0.1, 0.1])**2  # Start conservative
    
    parameter_names = ['log_age_myr', 'log_metallicity', 'log_dust_ratio']
    
    return log_posterior, initial_proposal_cov, parameter_names, true_params

def run_complete_mcmc_analysis(mcrt_observations, cluster_properties, 
                              n_samples=5000, n_chains=4):
    """Run complete MCMC analysis with full diagnostics"""
    
    # Setup
    log_posterior, proposal_cov, param_names, true_params = setup_cluster_mcmc_analysis(
        mcrt_observations, cluster_properties
    )
    
    print("Starting MCMC Analysis for Cluster Parameters")
    print("="*60)
    
    # Generate starting values around truth (for testing)
    initial_params_list = []
    for i in range(n_chains):
        # Add noise to true parameters for starting values
        noisy_start = true_params[0] + np.random.normal(0, 0.5, 3)
        initial_params_list.append(noisy_start)
    
    # Run multiple chains
    chains, r_hat_values, samplers = run_multiple_chains(
        log_posterior, n_chains, n_samples, initial_params_list, proposal_cov
    )
    
    # Combine chains for analysis
    combined_chain = chains.reshape(-1, chains.shape[-1])
    
    print("\n" + "="*60)
    print("POSTERIOR ANALYSIS")
    print("="*60)
    
    # Calculate posterior statistics
    posterior_stats = {}
    for i, param_name in enumerate(param_names):
        param_samples = combined_chain[:, i]
        
        posterior_stats[param_name] = {
            'mean': np.mean(param_samples),
            'std': np.std(param_samples),
            'median': np.median(param_samples),
            'q16': np.percentile(param_samples, 16),
            'q84': np.percentile(param_samples, 84),
            'true_value': true_params[0, i]  # First cluster's true values
        }
        
        # Print summary
        stats = posterior_stats[param_name]
        print(f"\n{param_name}:")
        print(f"  True value: {stats['true_value']:.3f}")
        print(f"  Posterior mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  68% CI: ({stats['q16']:.3f}, {stats['q84']:.3f})")
        
        # Check if true value is recovered
        if stats['q16'] <= stats['true_value'] <= stats['q84']:
            print(f"  ✓ True value within 68% credible interval")
        else:
            print(f"  ⚠ True value outside 68% credible interval")
    
    # Quality assessment
    print(f"\n" + "="*60)
    print("CHAIN QUALITY ASSESSMENT")
    print("="*60)
    
    assess_chain_quality(combined_chain, param_names)
    
    # Visualizations
    plot_trace_diagnostics(combined_chain, param_names)
    plot_autocorrelation_diagnostics(combined_chain, param_names)
    plot_corner_plot(combined_chain, param_names, true_params[0])
    
    return {
        'chains': chains,
        'combined_chain': combined_chain,
        'posterior_stats': posterior_stats,
        'r_hat_values': r_hat_values,
        'samplers': samplers,
        'parameter_names': param_names
    }

def plot_corner_plot(chain, parameter_names, true_values=None):
    """Create corner plot showing all parameter relationships"""
    
    n_params = chain.shape[1]
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: marginal distributions
                ax.hist(chain[:, i], bins=30, alpha=0.7, density=True)
                
                if true_values is not None:
                    ax.axvline(true_values[i], color='red', linestyle='--', 
                              label='True value')
                    ax.legend()
                
                ax.set_xlabel(parameter_names[i])
                
            elif i > j:
                # Lower triangle: 2D scatter plots
                ax.scatter(chain[:, j], chain[:, i], alpha=0.3, s=1)
                
                if true_values is not None:
                    ax.scatter(true_values[j], true_values[i], 
                             color='red', s=50, marker='*', 
                             label='True values')
                    ax.legend()
                
                ax.set_xlabel(parameter_names[j])
                ax.set_ylabel(parameter_names[i])
                
            else:
                # Upper triangle: hide
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

This comprehensive MCMC implementation provides students with both the theoretical foundation and practical tools for Bayesian parameter estimation. The emphasis on convergence diagnostics and chain quality assessment develops the critical skills needed to ensure reliable results, while the application to MCRT cluster data demonstrates how to apply these methods to realistic astrophysical problems. The systematic comparison with known truth values enables validation of the entire analysis pipeline.

This completes the detailed week-by-week lecture notes for ASTR 596, providing a comprehensive foundation in computational astrophysics, statistical methods, and modern machine learning techniques. Each week builds systematically on previous knowledge while introducing new mathematical concepts through astrophysical applications, preparing students to become sophisticated computational scientists capable of implementing algorithms from research papers and contributing to cutting-edge astronomical research..kappa_abs, self.kappa_scat, self.dust_density
        else:
            return 0, 0, 0  # No dust outside cluster
```

#### Realistic Dust Distributions

More sophisticated models couple dust density to stellar density:

```python
def create_dust_distribution(stellar_positions, stellar_masses, metallicity):
    """Create dust distribution based on stellar content"""
    # Calculate local stellar density
    stellar_density = calculate_stellar_density(stellar_positions, stellar_masses)
    
    # Dust density proportional to stellar density and metallicity
    dust_density = stellar_density * metallicity * dust_to_stellar_ratio
    
    return dust_density
```

## Mathematical Foundations: Integral Equations

### The Formal Solution

The radiative transfer equation can be written as an integral equation:

```
I(τ) = I₀ e^(-τ) + ∫₀^τ S(τ') e^(-(τ-τ')) dτ'
```

where S(τ) is the source function:
```
S = [κ_abs B(T) + (κ_scat/4π) ∫ I(n̂') Φ(n̂',n̂) dΩ'] / κ_ext
```

**Physical Interpretation**:
- First term: Attenuated initial intensity
- Second term: Contributions from emission and scattering along the path

### Monte Carlo as Numerical Integration

MCRT essentially solves this integral equation through random sampling:

```
⟨I⟩ = ∫...∫ I(x₁, x₂, ..., xₙ) p(x₁, x₂, ..., xₙ) dx₁dx₂...dxₙ
```

where the integral is over all possible photon paths and interactions.

**Sampling Strategy**: Generate photon paths according to physical probability distributions:
- **Path length**: p(s) = κ_ext e^(-κ_ext s)
- **Scattering angle**: p(cos θ) = Φ(cos θ)/2
- **Interaction type**: Bernoulli trial with probability a

## Validation and Testing

### Analytical Test Cases

#### Plane Parallel Atmosphere

For plane parallel slab with optical depth τ₀:

**Pure Absorption**:
```
I(μ, τ₀) = I₀ e^(-τ₀/μ)
```

where μ = cos θ is the direction cosine.

**Conservative Scattering** (no absorption):
```
I(μ, τ₀) = I₀  (for τ₀ → ∞)
```

#### Single Scattering

For optically thin medium (τ << 1) with single scattering:
```
I_scattered ≈ I₀ τ Φ(θ)/4π
```

### Convergence Testing

```python
def test_mcrt_convergence(n_photon_list):
    """Test MCRT convergence with increasing photon number"""
    results = []
    
    for n_photons in n_photon_list:
        escaped_flux = run_mcrt_simulation(n_photons)
        results.append(escaped_flux)
    
    # Check convergence (should scale as 1/√N)
    errors = np.std(results) / np.sqrt(n_photon_list)
    
    return results, errors
```

### Energy Conservation Validation

Critical test: Total energy must be conserved:

```python
def validate_energy_conservation(stellar_sources, escaped_photons, absorbed_energy):
    """Verify energy conservation in MCRT simulation"""
    # Input energy from stellar sources
    total_input = sum(source.luminosity for source in stellar_sources)
    
    # Output energy: escaped + absorbed
    total_escaped = sum(photon.energy for photon in escaped_photons)
    total_absorbed = np.sum(absorbed_energy)
    total_output = total_escaped + total_absorbed
    
    # Conservation check
    energy_error = abs(total_output - total_input) / total_input
    
    print(f"Energy conservation error: {energy_error:.2e}")
    return energy_error < 1e-3  # Acceptable threshold
```

This comprehensive foundation in radiative transfer physics and Monte Carlo methods prepares students for implementing the dusty stellar cluster simulations in Project 3, while building the statistical thinking essential for the machine learning applications in Projects 4-6.

---

# Week 6: Advanced MCRT Implementation & Synthetic Observations
**Learning Objectives**: Complete 3D MCRT implementation, generate synthetic observational datasets, understand observational effects

## Three-Dimensional Geometry and Coordinate Systems

### Coordinate System Management

Three-dimensional MCRT requires careful handling of coordinate transformations and geometric calculations.

#### Cartesian Coordinate System

For stellar cluster simulations, Cartesian coordinates provide the most straightforward implementation:

```python
class CartesianGrid:
    def __init__(self, x_range, y_range, z_range, n_cells):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        
        self.n_x, self.n_y, self.n_z = n_cells
        self.dx = (self.x_max - self.x_min) / self.n_x
        self.dy = (self.y_max - self.y_min) / self.n_y
        self.dz = (self.z_max - self.z_min) / self.n_z
        
        # Initialize dust properties grid
        self.dust_density = np.zeros(n_cells)
        self.dust_temperature = np.zeros(n_cells)
    
    def get_cell_index(self, position):
        """Get grid cell indices for given position"""
        x, y, z = position
        
        i = int((x - self.x_min) / self.dx)
        j = int((y - self.y_min) / self.dy)
        k = int((z - self.z_min) / self.dz)
        
        # Boundary checking
        i = max(0, min(i, self.n_x - 1))
        j = max(0, min(j, self.n_y - 1))
        k = max(0, min(k, self.n_z - 1))
        
        return i, j, k
```

#### Ray-Grid Intersection Algorithm

Efficient photon propagation requires fast ray-grid intersection:

```python
def ray_grid_intersection(position, direction, grid):
    """Find intersection of ray with grid boundaries"""
    # Calculate distances to all grid planes
    t_x = [(grid.x_min - position[0]) / direction[0],
           (grid.x_max - position[0]) / direction[0]]
    t_y = [(grid.y_min - position[1]) / direction[1],
           (grid.y_max - position[1]) / direction[1]]
    t_z = [(grid.z_min - position[2]) / direction[2],
           (grid.z_max - position[2]) / direction[2]]
    
    # Find entry and exit distances
    t_enter = max(min(t_x), min(t_y), min(t_z))
    t_exit = min(max(t_x), max(t_y), max(t_z))
    
    # Check for intersection
    if t_enter <= t_exit and t_exit > 0:
        return t_enter if t_enter > 0 else 0, t_exit
    else:
        return None, None  # No intersection
```

### Advanced Dust Distribution Models

#### Stellar Density-Coupled Dust

Realistic dust distributions follow stellar density with metallicity scaling:

```python
def calculate_dust_from_stellar_density(stellar_cluster, grid, dust_to_gas_ratio):
    """Calculate dust density based on stellar distribution"""
    for i in range(grid.n_x):
        for j in range(grid.n_y):
            for k in range(grid.n_z):
                # Cell center position
                x = grid.x_min + (i + 0.5) * grid.dx
                y = grid.y_min + (j + 0.5) * grid.dy
                z = grid.z_min + (k + 0.5) * grid.dz
                cell_center = np.array([x, y, z])
                
                # Calculate local stellar density
                stellar_density = 0
                total_metallicity = 0
                stellar_count = 0
                
                for star_idx, star_pos in enumerate(stellar_cluster.positions):
                    distance = np.linalg.norm(cell_center - star_pos)
                    
                    # Use kernel smoothing for stellar density
                    if distance < smoothing_length:
                        weight = kernel_function(distance, smoothing_length)
                        stellar_density += stellar_cluster.masses[star_idx] * weight
                        total_metallicity += stellar_cluster.metallicities[star_idx] * weight
                        stellar_count += weight
                
                # Average metallicity in cell
                if stellar_count > 0:
                    avg_metallicity = total_metallicity / stellar_count
                    grid.dust_density[i, j, k] = stellar_density * dust_to_gas_ratio * avg_metallicity
                else:
                    grid.dust_density[i, j, k] = 0

def kernel_function(distance, h):
    """Smoothing kernel for stellar density calculation"""
    q = distance / h
    if q < 1:
        return (1 - 1.5*q**2 + 0.75*q**3) / (np.pi * h**3)
    elif q < 2:
        return 0.25 * (2 - q)**3 / (np.pi * h**3)
    else:
        return 0
```

#### Temperature Self-Consistency

Dust temperature affects opacity, requiring iterative solution:

```python
def solve_dust_temperature_iteratively(grid, stellar_sources, max_iterations=10):
    """Solve for self-consistent dust temperature"""
    for iteration in range(max_iterations):
        old_temperature = grid.dust_temperature.copy()
        
        # Calculate heating rate from all stellar sources
        for i in range(grid.n_x):
            for j in range(grid.n_y):
                for k in range(grid.n_z):
                    cell_center = grid.get_cell_center(i, j, k)
                    
                    heating_rate = 0
                    for source in stellar_sources:
                        # Distance and direction to source
                        separation = source.position - cell_center
                        distance = np.linalg.norm(separation)
                        direction = separation / distance
                        
                        # Calculate optical depth to source
                        tau = calculate_optical_depth_to_source(
                            cell_center, source.position, grid)
                        
                        # Attenuated flux
                        flux = source.luminosity * np.exp(-tau) / (4 * np.pi * distance**2)
                        heating_rate += flux * dust_opacity(grid.dust_temperature[i,j,k])
                    
                    # Thermal equilibrium temperature
                    grid.dust_temperature[i, j, k] = (
                        heating_rate / (4 * sigma_sb * dust_emissivity))**(1/4)
        
        # Check convergence
        temperature_change = np.max(np.abs(grid.dust_temperature - old_temperature))
        if temperature_change < 0.1:  # 0.1 K tolerance
            break
        
        print(f"Iteration {iteration}: max temperature change = {temperature_change:.2f} K")
```

## Advanced MCRT Features

### Thermal Emission Implementation

Dust grains emit thermal radiation according to their temperature:

```python
def generate_thermal_photons(grid, n_thermal_photons):
    """Generate thermal photon packets from dust emission"""
    thermal_photons = []
    
    # Calculate total thermal luminosity
    total_thermal_power = 0
    for i in range(grid.n_x):
        for j in range(grid.n_y):
            for k in range(grid.n_z):
                if grid.dust_density[i, j, k] > 0:
                    cell_volume = grid.dx * grid.dy * grid.dz
                    T = grid.dust_temperature[i, j, k]
                    
                    # Thermal emission power in this cell
                    thermal_power = (4 * sigma_sb * T**4 * 
                                   grid.dust_density[i, j, k] * 
                                   cell_volume * dust_emissivity)
                    total_thermal_power += thermal_power
    
    # Generate photon packets
    for _ in range(n_thermal_photons):
        # Sample emission location weighted by thermal power
        cell_i, cell_j, cell_k = sample_thermal_emission_location(grid)
        
        # Random position within cell
        x = grid.x_min + (cell_i + np.random.random()) * grid.dx
        y = grid.y_min + (cell_j + np.random.random()) * grid.dy
        z = grid.z_min + (cell_k + np.random.random()) * grid.dz
        position = np.array([x, y, z])
        
        # Isotropic emission direction
        direction = sample_isotropic_direction()
        
        # Sample wavelength from Planck spectrum
        T_cell = grid.dust_temperature[cell_i, cell_j, cell_k]
        wavelength = sample_planck_wavelength(T_cell)
        
        # Photon packet energy
        energy = total_thermal_power / n_thermal_photons
        
        photon = PhotonPacket(position, direction, energy, wavelength)
        thermal_photons.append(photon)
    
    return thermal_photons

def sample_planck_wavelength(temperature):
    """Sample wavelength from Planck distribution"""
    # Use rejection sampling or lookup table method
    # For efficiency, pre-compute CDF and use inverse sampling
    
    # Simplified version: Sample from Wien tail (good approximation)
    u = np.random.random()
    x = -np.log(u)  # Sample from exponential
    
    # Convert to wavelength using Wien displacement
    lambda_max = 2.898e-3 / temperature  # meters
    wavelength = lambda_max * x / 3  # Approximate scaling
    
    return wavelength
```

### Polarization (Advanced Extension)

For advanced students, polarization tracking adds realism:

```python
class PolarizedPhoton(PhotonPacket):
    def __init__(self, position, direction, energy, wavelength, stokes_vector):
        super().__init__(position, direction, energy, wavelength)
        self.stokes = np.array(stokes_vector)  # [I, Q, U, V]
    
    def scatter_with_polarization(self, scattering_angle, dust_properties):
        """Handle scattering with polarization changes"""
        # Mueller matrix for dust scattering
        mueller_matrix = calculate_mueller_matrix(scattering_angle, dust_properties)
        
        # Update Stokes vector
        self.stokes = np.dot(mueller_matrix, self.stokes)
        
        # Update propagation direction
        new_direction = calculate_scattered_direction(self.direction, scattering_angle)
        self.direction = new_direction
```

## Synthetic Observational Data Generation

### Multi-Wavelength Photometry

Generate realistic photometric observations across multiple wavelengths:

```python
class SyntheticObserver:
    def __init__(self, position, viewing_angles, filter_set):
        self.position = np.array(position)
        self.viewing_angles = viewing_angles  # [theta, phi] angles
        self.filters = filter_set
        self.observations = {}
    
    def observe_cluster(self, escaped_photons):
        """Generate synthetic photometric observations"""
        for filter_name, filter_response in self.filters.items():
            total_flux = 0
            
            for photon in escaped_photons:
                # Check if photon reaches observer
                if self.photon_reaches_observer(photon):
                    # Apply filter response
                    flux_contribution = (photon.energy * 
                                       filter_response(photon.wavelength))
                    total_flux += flux_contribution
            
            # Convert to magnitude
            if total_flux > 0:
                magnitude = -2.5 * np.log10(total_flux) + zero_point_magnitude
                self.observations[filter_name] = magnitude
            else:
                self.observations[filter_name] = np.inf  # Not detected
        
        return self.observations
    
    def photon_reaches_observer(self, photon):
        """Check if photon direction points toward observer"""
        observer_direction = self.position - photon.position
        observer_direction = observer_direction / np.linalg.norm(observer_direction)
        
        # Small solid angle approximation
        angle = np.arccos(np.dot(photon.direction, observer_direction))
        return angle < observer_solid_angle
```

### Realistic Noise Models

Add observational uncertainties to synthetic data:

```python
def add_observational_noise(magnitudes, noise_model='gaussian'):
    """Add realistic noise to synthetic photometry"""
    noisy_magnitudes = {}
    
    for filter_name, magnitude in magnitudes.items():
        if magnitude == np.inf:  # Non-detection
            noisy_magnitudes[filter_name] = np.inf
            continue
        
        # Magnitude-dependent noise
        if noise_model == 'realistic':
            # Bright sources: photon noise dominated
            if magnitude < 20:
                sigma_mag = 0.01 + 0.001 * magnitude
            # Faint sources: background dominated
            else:
                sigma_mag = 0.1 * 10**((magnitude - 20) / 5)
        else:
            # Simple Gaussian noise
            sigma_mag = 0.05
        
        # Add noise
        noise = np.random.normal(0, sigma_mag)
        noisy_magnitudes[filter_name] = magnitude + noise
    
    return noisy_magnitudes
```

### Color-Color Diagrams

Generate synthetic color-color data for statistical analysis:

```python
def generate_color_color_data(cluster_observations):
    """Generate color-color diagram data"""
    colors = {}
    
    for obs in cluster_observations:
        # Calculate colors (difference in magnitudes)
        if 'B' in obs and 'V' in obs and obs['B'] != np.inf and obs['V'] != np.inf:
            colors['B-V'] = obs['B'] - obs['V']
        
        if 'V' in obs and 'I' in obs and obs['V'] != np.inf and obs['I'] != np.inf:
            colors['V-I'] = obs['V'] - obs['I']
        
        if 'J' in obs and 'K' in obs and obs['J'] != np.inf and obs['K'] != np.inf:
            colors['J-K'] = obs['J'] - obs['K']
    
    return colors
```

## Mathematical Foundations: Numerical Integration

### Monte Carlo Integration Theory

MCRT essentially solves high-dimensional integrals through random sampling:

```
⟨I⟩ = ∫∫∫ I(x, μ, φ) p(x, μ, φ) dx dμ dφ
```

where the integral is over position x, direction cosine μ, and azimuth φ.

**Variance Reduction**: The key to efficient MCRT is reducing the variance of the integral estimator:

```
Var[⟨I⟩] = (1/N) ∫ [I(x) - ⟨I⟩]² p(x) dx
```

**Importance Sampling**: Sample photon paths from probability distributions that concentrate effort where the integrand is large.

### Error Analysis and Convergence

#### Statistical Error Estimation

For N photon packets, the statistical error in flux estimates scales as:

```
σ_statistical = σ_samples / √N
```

where σ_samples is the sample standard deviation.

**Implementation**:
```python
def estimate_statistical_error(photon_energies):
    """Estimate statistical error in flux measurement"""
    n_photons = len(photon_energies)
    mean_energy = np.mean(photon_energies)
    sample_variance = np.var(photon_energies, ddof=1)
    
    statistical_error = np.sqrt(sample_variance / n_photons)
    relative_error = statistical_error / mean_energy if mean_energy > 0 else np.inf
    
    return statistical_error, relative_error
```

#### Systematic Error Sources

**Grid Resolution Effects**: Finite grid size introduces discretization errors:
```python
def assess_grid_resolution_error(grid_sizes, reference_result):
    """Assess convergence with grid resolution"""
    errors = []
    
    for grid_size in grid_sizes:
        result = run_mcrt_with_grid_size(grid_size)
        error = abs(result - reference_result) / reference_result
        errors.append(error)
    
    # Should scale as (grid_size)^(-2) for second-order accuracy
    return errors
```

**Photon Number Convergence**: More photons reduce statistical error:
```python
def convergence_analysis(n_photon_list):
    """Analyze convergence with photon number"""
    results = []
    errors = []
    
    for n_photons in n_photon_list:
        # Run multiple realizations
        realizations = []
        for _ in range(10):
            result = run_mcrt_simulation(n_photons)
            realizations.append(result)
        
        mean_result = np.mean(realizations)
        std_error = np.std(realizations) / np.sqrt(len(realizations))
        
        results.append(mean_result)
        errors.append(std_error)
    
    return results, errors
```

## Performance Optimization

### Computational Efficiency Strategies

#### Adaptive Grid Refinement

Focus computational effort where needed most:

```python
class AdaptiveGrid:
    def __init__(self, base_resolution, max_refinement_level):
        self.base_resolution = base_resolution
        self.max_level = max_refinement_level
        self.cells = self.initialize_base_grid()
    
    def refine_cell(self, cell_index, criterion):
        """Refine cell if criterion is met"""
        cell = self.cells[cell_index]
        
        # Refinement criteria (e.g., high optical depth gradient)
        if criterion(cell):
            # Split cell into 8 subcells
            subcells = self.split_cell(cell)
            self.cells[cell_index] = subcells
    
    def refinement_criterion(self, cell):
        """Determine if cell should be refined"""
        # High optical depth gradient
        tau_gradient = self.calculate_optical_depth_gradient(cell)
        
        # High stellar density
        stellar_density = self.calculate_stellar_density(cell)
        
        return (tau_gradient > threshold_gradient or 
                stellar_density > threshold_density)
```

#### Parallel Processing

MCRT is embarrassingly parallel - different photon packets can be processed independently:

```python
from multiprocessing import Pool
import numpy as np

def run_parallel_mcrt(n_photons, n_processes=4):
    """Run MCRT simulation using parallel processing"""
    photons_per_process = n_photons // n_processes
    
    # Divide photon packets among processes
    process_args = [(photons_per_process, seed + i) for i, seed in 
                   enumerate(range(0, n_photons, photons_per_process))]
    
    with Pool(n_processes) as pool:
        results = pool.map(run_mcrt_single_process, process_args)
    
    # Combine results
    total_escaped = []
    total_absorbed = np.zeros_like(results[0][1])
    
    for escaped, absorbed in results:
        total_escaped.extend(escaped)
        total_absorbed += absorbed
    
    return total_escaped, total_absorbed

def run_mcrt_single_process(args):
    """MCRT simulation for single process"""
    n_photons, seed = args
    np.random.seed(seed)  # Ensure reproducible randomness
    
    # Run MCRT simulation
    escaped, absorbed = monte_carlo_radiative_transfer(n_photons)
    
    return escaped, absorbed
```

### Memory Management

Large 3D grids can consume significant memory:

```python
def optimize_memory_usage(grid):
    """Optimize memory usage for large grids"""
    # Use sparse arrays for mostly empty grids
    from scipy.sparse import csr_matrix
    
    # Store only non-zero dust density cells
    nonzero_indices = np.where(grid.dust_density > 0)
    
    # Compressed storage
    grid.sparse_dust_density = csr_matrix(
        (grid.dust_density[nonzero_indices], nonzero_indices),
        shape=grid.dust_density.shape
    )
    
    # Clear original dense array
    del grid.dust_density
```

## Validation and Quality Assurance

### Benchmark Problems

#### Optically Thin Limit

For very low optical depths (τ << 1), most photons should escape without interaction:

```python
def test_optically_thin_limit():
    """Test MCRT in optically thin limit"""
    # Create cluster with very low dust density
    low_density_cluster = create_test_cluster(dust_density=1e-10)
    
    escaped, absorbed = run_mcrt_simulation(low_density_cluster, n_photons=10000)
    
    escape_fraction = len(escaped) / 10000
    
    # Should be close to 1.0 for optically thin case
    assert escape_fraction > 0.95, f"Escape fraction too low: {escape_fraction}"
    
    print(f"Optically thin test passed: escape fraction = {escape_fraction:.3f}")
```

#### Energy Conservation

Fundamental conservation law must be satisfied:

```python
def test_energy_conservation():
    """Rigorous test of energy conservation"""
    stellar_luminosity = 1.0  # Solar luminosities
    
    # Run MCRT simulation
    escaped, absorbed = run_mcrt_simulation(stellar_luminosity, n_photons=100000)
    
    # Calculate energy budget
    escaped_energy = sum(photon.energy for photon in escaped)
    absorbed_energy = np.sum(absorbed)
    total_output = escaped_energy + absorbed_energy
    
    conservation_error = abs(total_output - stellar_luminosity) / stellar_luminosity
    
    assert conservation_error < 0.01, f"Energy conservation violated: {conservation_error}"
    
    print(f"Energy conservation test passed: error = {conservation_error:.2e}")
```

#### Comparison with Analytical Solutions

Test against known analytical results where available:

```python
def test_plane_parallel_atmosphere():
    """Test against analytical plane parallel solution"""
    # Set up plane parallel test case
    optical_depth = 1.0
    albedo = 0.5
    
    # Analytical solution for normal incidence
    analytical_transmission = np.exp(-optical_depth)
    
    # MCRT simulation
    mcrt_transmission = run_plane_parallel_mcrt(optical_depth, albedo)
    
    relative_error = abs(mcrt_transmission - analytical_transmission) / analytical_transmission
    
    assert relative_error < 0.05, f"Plane parallel test failed: error = {relative_error}"
    
    print(f"Plane parallel test passed: MCRT = {mcrt_transmission:.3f}, "
          f"Analytical = {analytical_transmission:.3f}")
```

This comprehensive treatment of advanced MCRT implementation provides students with both the theoretical understanding and practical skills needed to generate high-quality synthetic observational datasets for statistical analysis in Projects 4-5. The emphasis on validation, error analysis, and performance optimization develops the rigorous computational practices essential for research-level work.

---

# Week 7: Linear Regression Mathematics & Implementation
**Learning Objectives**: Derive regression from first principles, implement gradient descent, understand statistical inference

## Mathematical Foundations of Linear Regression

### The Linear Model Framework

Linear regression assumes the relationship between input features X and output y follows:

```
y = Xβ + ε
```

where:
- **y**: Response vector (n × 1) - our MCRT observational data
- **X**: Design matrix (n × p) - features derived from synthetic observations
- **β**: Parameter vector (p × 1) - physical parameters we want to estimate
- **ε**: Error vector (n × 1) - measurement noise and model uncertainty

**Matrix Form for MCRT Application**:
```
[F₁]   [1  log(λ₁)  T₁  Z₁] [β₀]   [ε₁]
[F₂] = [1  log(λ₂)  T₂  Z₂] [β₁] + [ε₂]
[⋮ ]   [⋮     ⋮     ⋮   ⋮ ] [β₂]   [⋮ ]
[Fₙ]   [1  log(λₙ)  Tₙ  Zₙ] [β₃]   [εₙ]
```

where Fᵢ are observed fluxes and we want to estimate cluster properties β.

### Least Squares Derivation

#### Geometric Interpretation

The least squares solution minimizes the sum of squared residuals:

```
RSS(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
```

**Physical Meaning**: We're finding the parameter values that make our model predictions closest to observations in Euclidean distance sense.

#### Analytical Solution

To minimize RSS(β), take derivative with respect to β and set to zero:

```
∂RSS/∂β = -2Xᵀ(y - Xβ) = 0
```

This gives the **normal equations**:
```
XᵀXβ = Xᵀy
```

**Closed-form solution** (when XᵀX is invertible):
```
β̂ = (XᵀX)⁻¹Xᵀy
```

#### Implementation from Scratch

```python
def linear_regression_analytical(X, y):
    """Analytical solution to linear regression"""
    # Add intercept term if not present
    if X.shape[1] == len(X.shape) - 1:  # No intercept column
        X = np.column_stack([np.ones(len(X)), X])
    
    # Normal equations: XᵀX β = Xᵀy
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Check for invertibility
    try:
        beta = np.linalg.solve(XtX, Xty)  # More stable than matrix inverse
    except np.linalg.LinAlgError:
        # Use pseudoinverse for singular matrices
        beta = np.linalg.pinv(X) @ y
        print("Warning: Using pseudoinverse due to singular XᵀX matrix")
    
    return beta

def predict(X, beta):
    """Make predictions using fitted model"""
    if X.shape[1] == len(beta) - 1:
        X = np.column_stack([np.ones(len(X)), X])
    return X @ beta

def calculate_residuals(X, y, beta):
    """Calculate model residuals"""
    y_pred = predict(X, beta)
    residuals = y - y_pred
    return residuals
```

### Statistical Properties of Least Squares

#### Gauss-Markov Theorem

Under assumptions:
1. **Linearity**: y = Xβ + ε
2. **Zero mean errors**: E[ε] = 0
3. **Homoscedasticity**: Var(εᵢ) = σ² for all i
4. **Uncorrelated errors**: Cov(εᵢ, εⱼ) = 0 for i ≠ j

The least squares estimator β̂ is **BLUE** (Best Linear Unbiased Estimator).

#### Parameter Uncertainty

**Covariance matrix** of parameter estimates:
```
Cov(β̂) = σ²(XᵀX)⁻¹
```

**Standard errors**:
```
SE(β̂ⱼ) = σ̂√[(XᵀX)⁻¹]ⱼⱼ
```

where σ̂² = RSS/(n-p) is the estimated error variance.

```python
def calculate_parameter_uncertainties(X, y, beta):
    """Calculate parameter uncertainties and confidence intervals"""
    n, p = X.shape
    
    # Calculate residuals and error variance
    residuals = calculate_residuals(X, y, beta)
    sigma_squared = np.sum(residuals**2) / (n - p)
    
    # Covariance matrix
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_matrix = sigma_squared * XtX_inv
    
    # Standard errors
    standard_errors = np.sqrt(np.diag(cov_matrix))
    
    # 95% confidence intervals
    from scipy.stats import t
    t_critical = t.ppf(0.975, n - p)  # Two-tailed 95% CI
    
    confidence_intervals = []
    for i, (beta_i, se_i) in enumerate(zip(beta, standard_errors)):
        ci_lower = beta_i - t_critical * se_i
        ci_upper = beta_i + t_critical * se_i
        confidence_intervals.append((ci_lower, ci_upper))
    
    return standard_errors, confidence_intervals, cov_matrix
```

## Gradient Descent Implementation

### The Optimization Perspective

Instead of solving normal equations, we can minimize the cost function iteratively:

```
J(β) = (1/2n)||y - Xβ||² = (1/2n)∑ᵢ(yᵢ - xᵢᵀβ)²
```

**Gradient**:
```
∇J(β) = -(1/n)Xᵀ(y - Xβ)
```

### Batch Gradient Descent

Update parameters in direction of steepest descent:

```
β^(k+1) = β^(k) - α∇J(β^(k))
```

where α is the learning rate.

```python
class GradientDescentRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        self.converged = False
    
    def fit(self, X, y):
        """Fit linear regression using gradient descent"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        n, p = X_with_intercept.shape
        
        # Initialize parameters
        self.beta = np.random.normal(0, 0.01, p)
        
        for iteration in range(self.max_iterations):
            # Forward pass: calculate predictions and cost
            y_pred = X_with_intercept @ self.beta
            cost = np.mean((y - y_pred)**2) / 2
            self.cost_history.append(cost)
            
            # Backward pass: calculate gradients
            residuals = y - y_pred
            gradients = -(1/n) * X_with_intercept.T @ residuals
            
            # Update parameters
            self.beta -= self.learning_rate * gradients
            
            # Check convergence
            if iteration > 0:
                cost_change = abs(self.cost_history[-1] - self.cost_history[-2])
                if cost_change < self.tolerance:
                    self.converged = True
                    print(f"Converged after {iteration} iterations")
                    break
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        return X_with_intercept @ self.beta
    
    def plot_convergence(self):
        """Plot cost function convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.title('Gradient Descent Convergence')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
```

### Learning Rate Selection and Adaptive Methods

#### Fixed Learning Rate Issues

- **Too large**: Oscillation or divergence
- **Too small**: Slow convergence

#### Adaptive Learning Rate

```python
def adaptive_gradient_descent(X, y, initial_lr=0.1, decay_rate=0.95):
    """Gradient descent with adaptive learning rate"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n, p = X_with_intercept.shape
    
    beta = np.random.normal(0, 0.01, p)
    learning_rate = initial_lr
    cost_history = []
    
    for iteration in range(1000):
        # Calculate cost and gradients
        y_pred = X_with_intercept @ beta
        cost = np.mean((y - y_pred)**2) / 2
        gradients = -(1/n) * X_with_intercept.T @ (y - y_pred)
        
        # Try step with current learning rate
        beta_new = beta - learning_rate * gradients
        y_pred_new = X_with_intercept @ beta_new
        cost_new = np.mean((y - y_pred_new)**2) / 2
        
        # Adapt learning rate
        if cost_new > cost:  # Cost increased
            learning_rate *= decay_rate  # Reduce learning rate
            continue  # Don't update parameters
        else:
            beta = beta_new
            cost_history.append(cost_new)
        
        # Check convergence
        if len(cost_history) > 1:
            if abs(cost_history[-1] - cost_history[-2]) < 1e-6:
                break
    
    return beta, cost_history
```

#### Momentum Method

Include momentum to accelerate convergence and avoid local minima:

```
v^(k+1) = γv^(k) + α∇J(β^(k))
β^(k+1) = β^(k) - v^(k+1)
```

```python
def gradient_descent_with_momentum(X, y, learning_rate=0.01, momentum=0.9):
    """Gradient descent with momentum"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n, p = X_with_intercept.shape
    
    beta = np.random.normal(0, 0.01, p)
    velocity = np.zeros(p)
    cost_history = []
    
    for iteration in range(1000):
        # Calculate predictions and cost
        y_pred = X_with_intercept @ beta
        cost = np.mean((y - y_pred)**2) / 2
        cost_history.append(cost)
        
        # Calculate gradients
        gradients = -(1/n) * X_with_intercept.T @ (y - y_pred)
        
        # Update velocity and parameters
        velocity = momentum * velocity + learning_rate * gradients
        beta -= velocity
        
        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
            break
    
    return beta, cost_history
```

## Application to MCRT Cluster Data

### Feature Engineering for Astrophysical Data

Transform raw MCRT outputs into meaningful features for regression:

```python
def create_cluster_features(mcrt_observations):
    """Create feature matrix from MCRT cluster observations"""
    features = []
    
    for obs in mcrt_observations:
        # Extract multi-wavelength fluxes
        flux_features = []
        
        # Logarithmic flux ratios (color indices)
        if 'flux_optical' in obs and 'flux_ir' in obs:
            color_optical_ir = np.log10(obs['flux_optical'] / obs['flux_ir'])
            flux_features.append(color_optical_ir)
        
        # Total luminosity
        total_flux = sum(obs[key] for key in obs if 'flux' in key)
        flux_features.append(np.log10(total_flux))
        
        # Flux distribution (spectral shape)
        if 'flux_uv' in obs and 'flux_optical' in obs and 'flux_ir' in obs:
            total = obs['flux_uv'] + obs['flux_optical'] + obs['flux_ir']
            uv_fraction = obs['flux_uv'] / total
            ir_fraction = obs['flux_ir'] / total
            flux_features.extend([uv_fraction, ir_fraction])
        
        # Spatial information (if available)
        if 'cluster_size' in obs:
            flux_features.append(obs['cluster_size'])
        
        features.append(flux_features)
    
    return np.array(features)

def create_target_variables(cluster_properties):
    """Create target variables from known cluster properties"""
    targets = []
    
    for cluster in cluster_properties:
        # Primary targets: age, metallicity, dust content
        target_vector = [
            np.log10(cluster['age_myr']),  # Log age in Myr
            np.log10(cluster['metallicity']),  # Log metallicity
            np.log10(cluster['dust_to_gas_ratio'])  # Log dust content
        ]
        targets.append(target_vector)
    
    return np.array(targets)
```

### Multi-Output Regression

Estimate multiple cluster parameters simultaneously:

```python
class MultiOutputRegression:
    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.target_names = None
    
    def fit(self, X, Y, feature_names=None, target_names=None):
        """Fit separate regression model for each target variable"""
        self.feature_names = feature_names
        self.target_names = target_names
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n_targets = Y.shape[1]
        
        for i in range(n_targets):
            target_name = target_names[i] if target_names else f'target_{i}'
            
            # Fit individual regression model
            model = GradientDescentRegression()
            model.fit(X, Y[:, i])
            
            self.models[target_name] = model
            
            print(f"Fitted model for {target_name}")
    
    def predict(self, X):
        """Predict all target variables"""
        predictions = {}
        
        for target_name, model in self.models.items():
            predictions[target_name] = model.predict(X)
        
        # Convert to array format
        pred_array = np.column_stack([predictions[name] for name in self.target_names])
        
        return pred_array, predictions
    
    def evaluate_performance(self, X_test, Y_test):
        """Evaluate model performance on test data"""
        Y_pred, _ = self.predict(X_test)
        
        performance = {}
        
        for i, target_name in enumerate(self.target_names):
            y_true = Y_test[:, i]
            y_pred = Y_pred[:, i]
            
            # Calculate performance metrics
            mse = np.mean((y_true - y_pred)**2)
            mae = np.mean(np.abs(y_true - y_pred))
            r_squared = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
            
            performance[target_name] = {
                'MSE': mse,
                'MAE': mae,
                'R²': r_squared
            }
        
        return performance
```

## Model Validation and Diagnostics

### Cross-Validation for Model Assessment

```python
def k_fold_cross_validation(X, y, k=5, model_class=GradientDescentRegression):
    """Perform k-fold cross-validation"""
    n_samples = len(X)
    fold_size = n_samples // k
    
    cv_scores = []
    
    for fold in range(k):
        # Create train/validation splits
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k-1 else n_samples
        
        # Validation set
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        
        # Training set
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Fit model and evaluate
        model = model_class()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred)**2)
        cv_scores.append(mse)
        
        print(f"Fold {fold+1}: MSE = {mse:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"Cross-validation MSE: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    return cv_scores, mean_cv_score, std_cv_score
```

### Residual Analysis

```python
def analyze_residuals(X, y, beta, feature_names=None):
    """Comprehensive residual analysis"""
    # Calculate residuals
    residuals = calculate_residuals(X, y, beta)
    y_pred = predict(X, beta)
    
    # Standardized residuals
    residual_std = np.std(residuals)
    standardized_residuals = residuals / residual_std
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q plot for normality
    from scipy.stats import probplot
    probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    
    # 3. Scale-Location plot
    sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
    axes[0, 2].scatter(y_pred, sqrt_abs_residuals, alpha=0.6)
    axes[0, 2].set_xlabel('Fitted Values')
    axes[0, 2].set_ylabel('√|Standardized Residuals|')
    axes[0, 2].set_title('Scale-Location Plot')
    
    # 4. Residual histogram
    axes[1, 0].hist(standardized_residuals, bins=20, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Standardized Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    
    # 5. Residuals vs Leverage (if applicable)
    if X.shape[1] > 1:
        leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
        axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.6)
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Residuals vs Leverage')
    
    # 6. Feature importance (coefficient magnitudes)
    if feature_names:
        coef_importance = np.abs(beta[1:])  # Exclude intercept
        axes[1, 2].bar(range(len(coef_importance)), coef_importance)
        axes[1, 2].set_xlabel('Feature Index')
        axes[1, 2].set_ylabel('|Coefficient|')
        axes[1, 2].set_title('Feature Importance')
        axes[1, 2].set_xticks(range(len(feature_names)))
        axes[1, 2].set_xticklabels(feature_names, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    # Normality tests
    shapiro_stat, shapiro_p = shapiro(standardized_residuals)
    jb_stat, jb_p = jarque_bera(standardized_residuals)
    
    print("Diagnostic Test Results:")
    print(f"Shapiro-Wilk normality test: statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
    print(f"Jarque-Bera normality test: statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    
    # Outlier detection
    outlier_threshold = 3.0
    outliers = np.where(np.abs(standardized_residuals) > outlier_threshold)[0]
    
    if len(outliers) > 0:
        print(f"Potential outliers (|residual| > {outlier_threshold}): {len(outliers)} observations")
        print(f"Outlier indices: {outliers}")
    else:
        print("No extreme outliers detected")
    
    return residuals, standardized_residuals, outliers
```

### Model Comparison and Selection

```python
def compare_regression_models(X, y, test_models=None):
    """Compare different regression implementations"""
    if test_models is None:
        test_models = [
            ('Analytical', linear_regression_analytical),
            ('Gradient Descent', GradientDescentRegression),
            ('Momentum GD', lambda: gradient_descent_with_momentum)
        ]
    
    results = {}
    
    for model_name, model_class in test_models:
        print(f"\nTesting {model_name}...")
        
        if model_name == 'Analytical':
            beta = model_class(X, y)
            y_pred = predict(X, beta)
        else:
            model = model_class()
            if hasattr(model, 'fit'):
                model.fit(X, y)
                y_pred = model.predict(X)
                beta = model.beta
            else:
                beta, _ = model(X, y)
                y_pred = predict(X, beta)
        
        # Calculate performance metrics
        mse = np.mean((y - y_pred)**2)
        mae = np.mean(np.abs(y - y_pred))
        r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        results[model_name] = {
            'MSE': mse,
            'MAE': mae,
            'R²': r_squared,
            'Parameters': beta
        }
        
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")
    
    return results
```

This comprehensive foundation in linear regression mathematics and implementation provides students with deep understanding of both the theoretical principles and practical implementation challenges. The emphasis on gradient descent prepares them for the neural network training in the final project, while the statistical analysis techniques build toward the Bayesian methods in Project 5.

---

# Week 8: Advanced Regression & Feature Engineering
**Learning Objectives**: Master regularization techniques, implement model selection, understand bias-variance tradeoff

## Regularization Theory and Implementation

### The Bias-Variance Tradeoff

All supervised learning algorithms face the fundamental tradeoff between bias and variance:

**Bias**: Error from overly simplistic assumptions
**Variance**: Error from sensitivity to small fluctuations in training data
**Total Error**: Bias² + Variance + Irreducible Error

For linear regression:
- **High bias, low variance**: Simple models that underfit
- **Low bias, high variance**: Complex models that overfit
- **Goal**: Find optimal balance for best generalization

#### Mathematical Framework

For a model f̂(x) trained on dataset D, the expected prediction error at point x₀ is:

```
E[(Y - f̂(x₀))²] = σ² + [E[f̂(x₀)] - f(x₀)]² + Var[f̂(x₀)]
                  = Noise + Bias² + Variance
```

where f(x₀) is the true function value.

```python
def bias_variance_decomposition(X, y, n_bootstrap=100, test_point=None):
    """Empirical bias-variance decomposition"""
    if test_point is None:
        test_point = X[0]  # Use first data point as test
    
    n_samples = len(X)
    predictions = []
    
    # Bootstrap sampling to estimate bias and variance
    for _ in range(n_bootstrap):
        # Bootstrap resample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]
        
        # Fit model on bootstrap sample
        beta = linear_regression_analytical(X_boot, y_boot)
        
        # Predict on test point
        if len(test_point.shape) == 1:
            test_point_with_intercept = np.concatenate([[1], test_point])
        else:
            test_point_with_intercept = np.column_stack([np.ones(len(test_point)), test_point])
        
        pred = test_point_with_intercept @ beta
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # True value (use full dataset prediction as proxy)
    beta_full = linear_regression_analytical(X, y)
    true_pred = np.concatenate([[1], test_point]) @ beta_full
    
    # Decomposition
    bias_squared = (np.mean(predictions) - true_pred)**2
    variance = np.var(predictions)
    
    print(f"Bias²: {bias_squared:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Total: {bias_squared + variance:.4f}")
    
    return bias_squared, variance, predictions
```

### Ridge Regression (L2 Regularization)

Ridge regression adds penalty proportional to sum of squared coefficients:

```
J_ridge(β) = ||y - Xβ||² + λ||β||²₂
```

where λ is the regularization parameter.

**Modified normal equations**:
```
β̂_ridge = (X^T X + λI)^(-1) X^T y
```

**Key Properties**:
- Shrinks coefficients toward zero
- Handles multicollinearity by distributing importance among correlated features
- Always has unique solution (X^T X + λI is always invertible)

```python
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Regularization strength
        self.beta = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """Fit Ridge regression with L2 regularization"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        n, p = X_with_intercept.shape
        
        # Create regularization matrix (don't penalize intercept)
        regularization_matrix = self.alpha * np.eye(p)
        regularization_matrix[0, 0] = 0  # No penalty for intercept
        
        # Ridge solution
        XtX_plus_reg = X_with_intercept.T @ X_with_intercept + regularization_matrix
        Xty = X_with_intercept.T @ y
        
        self.beta = np.linalg.solve(XtX_plus_reg, Xty)
        self.feature_names = feature_names
        
        return self
    
    def predict(self, X):
        """Make predictions using fitted Ridge model"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        return X_with_intercept @ self.beta
    
    def get_coefficients(self):
        """Return coefficients with feature names"""
        if self.feature_names is not None:
            coef_dict = {'intercept': self.beta[0]}
            for i, name in enumerate(self.feature_names):
                coef_dict[name] = self.beta[i+1]
            return coef_dict
        else:
            return self.beta
```

### Lasso Regression (L1 Regularization)

Lasso uses L1 penalty, promoting sparse solutions:

```
J_lasso(β) = ||y - Xβ||² + λ||β||₁
```

where ||β||₁ = Σ|βⱼ| is the L1 norm.

**Key Properties**:
- Performs automatic feature selection by setting coefficients exactly to zero
- Produces sparse models that are interpretable
- No closed-form solution; requires iterative algorithms

#### Coordinate Descent Algorithm

```python
class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tolerance=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.beta = None
        self.converged = False
    
    def fit(self, X, y):
        """Fit Lasso regression using coordinate descent"""
        # Standardize features for numerical stability
        X_scaled, self.X_mean, self.X_std = self._standardize_features(X)
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        n, p = X_scaled.shape
        self.beta = np.zeros(p)
        
        for iteration in range(self.max_iter):
            beta_old = self.beta.copy()
            
            for j in range(p):
                # Calculate partial residual
                partial_residual = y_centered - X_scaled @ self.beta + X_scaled[:, j] * self.beta[j]
                
                # Coordinate descent update with soft thresholding
                rho_j = X_scaled[:, j] @ partial_residual
                z_j = np.sum(X_scaled[:, j]**2)
                
                if z_j == 0:
                    self.beta[j] = 0
                else:
                    self.beta[j] = self._soft_threshold(rho_j / z_j, self.alpha / z_j)
            
            