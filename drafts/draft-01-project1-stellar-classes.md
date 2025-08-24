# ⚠️ Project 1: Units Module & ZAMS Stellar Classes

**Due:** Monday, September 8, 2025 at 11:59 PM PT  
**Submission:** GitHub Classroom  
**Weight:** 10% of final grade  
**Topics:** Unit Systems, Object-Oriented Programming, NumPy Vectorization, ZAMS Stellar Physics

## Learning Objectives

By completing this project, you will:
1. Build a CGS units module for scientific computing
2. Implement object-oriented design with Python classes
3. Master the Stefan-Boltzmann law as a general solver
4. Apply NumPy vectorization to avoid loops
5. Explore metallicity effects on stellar properties
6. Create scientific multi-panel visualizations
7. Practice professional coding standards

## Overview

You will first build a CGS units module to handle unit conversions throughout the course, then create two classes that model stars and stellar populations. This project establishes the foundation for all subsequent computational work.

## Part 1: CGS Units Module (20%)

Create a `units_cgs.py` module that defines all constants in CGS units. This module will be used throughout the course.

### Required Constants Table

All constants must be defined in **CGS units** (centimeters, grams, seconds):

| Constant Name | Symbol | CGS Value | CGS Units |
|--------------|--------|-----------|-----------|
| `SPEED_LIGHT` | c | 2.99792458 × 10¹⁰ | cm/s |
| `GRAVITATIONAL_CONST` | G | 6.67430 × 10⁻⁸ | cm³/(g⋅s²) |
| `PLANCK_CONST` | h | 6.62607015 × 10⁻²⁷ | erg⋅s |
| `BOLTZMANN_CONST` | k_B | 1.380649 × 10⁻¹⁶ | erg/K |
| `STEFAN_BOLTZMANN_CONST` | σ | 5.670374419 × 10⁻⁵ | erg/(cm²⋅s⋅K⁴) |
| `ELECTRON_MASS` | m_e | 9.1093837015 × 10⁻²⁸ | g |
| `PROTON_MASS` | m_p | 1.67262192369 × 10⁻²⁴ | g |
| `ELECTRON_CHARGE` | e | 4.80320425 × 10⁻¹⁰ | esu (statcoulomb) |
| `AVOGADRO_NUMBER` | N_A | 6.02214076 × 10²³ | 1/mol |
| `GAS_CONST` | R | 8.314462618 × 10⁷ | erg/(mol⋅K) |
| `WIEN_CONST` | b | 0.28977719551 | cm⋅K |
| `THOMSON_CROSS_SECTION` | σ_T | 6.6524587321 × 10⁻²⁵ | cm² |
| `RADIATION_CONST` | a | 7.565723 × 10⁻¹⁵ | erg/(cm³⋅K⁴) |

### Astronomical Constants (CGS)

| Constant Name | Symbol | CGS Value | CGS Units |
|--------------|--------|-----------|-----------|
| `SOLAR_MASS` | M_☉ | 1.9884 × 10³³ | g |
| `SOLAR_RADIUS` | R_☉ | 6.957 × 10¹⁰ | cm |
| `SOLAR_LUMINOSITY` | L_☉ | 3.828 × 10³³ | erg/s |
| `SOLAR_TEMPERATURE` | T_☉ | 5772 | K |
| `AU` | AU | 1.495978707 × 10¹³ | cm |
| `PARSEC` | pc | 3.0857 × 10¹⁸ | cm |
| `YEAR` | yr | 3.15576 × 10⁷ | s |

### Unit Conversion Factors

Include these conversion factors from CGS to other systems:

```python
# Length conversions (from cm)
CM_TO_M = 1e-2
CM_TO_KM = 1e-5
CM_TO_ANGSTROM = 1e8
CM_TO_NM = 1e7

# Mass conversions (from g)
G_TO_KG = 1e-3
G_TO_MSUN = 1 / SOLAR_MASS

# Energy conversions (from erg)
ERG_TO_JOULE = 1e-7
ERG_TO_EV = 6.242e11

# Luminosity conversions (from erg/s)
ERG_PER_S_TO_WATT = 1e-7
ERG_PER_S_TO_LSUN = 1 / SOLAR_LUMINOSITY
```

### Required Functions

```python
def wavelength_to_frequency(wavelength_cm):
    """Convert wavelength (cm) to frequency (Hz)."""
    return SPEED_LIGHT / wavelength_cm

def frequency_to_wavelength(frequency_hz):
    """Convert frequency (Hz) to wavelength (cm)."""
    return SPEED_LIGHT / frequency_hz

def temperature_to_energy(temp_k):
    """Convert temperature (K) to thermal energy (erg)."""
    return BOLTZMANN_CONST * temp_k
```

## Part 2: The `Star` Class (30%)

Create a `Star` class with the following:

### Required Attributes
- `mass` (g) - stellar mass in grams
- `radius` (cm) - stellar radius in centimeters  
- `luminosity` (erg/s) - stellar luminosity in erg/s
- `temperature` (K) - effective surface temperature (calculated)
- `metallicity` (Z) - metallicity fraction
- `ms_lifetime` (s) - main sequence lifetime in seconds

### Required Methods

#### `__init__(self, mass_msun, metallicity=0.02)`
Initialize a ZAMS star with given mass (in solar masses) and metallicity. 
- Convert to CGS units internally
- Use the provided Tout et al. module to compute ZAMS radius and luminosity
- Calculate temperature from Stefan-Boltzmann law
- Valid metallicity range: 0.001 ≤ Z ≤ 0.03

#### `stefan_boltzmann(self, L=None, R=None, T=None)`
Implement the Stefan-Boltzmann law solver using CGS units:
```
L = 4π σ R² T⁴
```
Given any two parameters, solve for the third:
```python
def stefan_boltzmann(self, L=None, R=None, T=None):
    """
    Solve Stefan-Boltzmann law given any two parameters.
    
    Parameters:
    -----------
    L : float, optional - Luminosity in erg/s
    R : float, optional - Radius in cm  
    T : float, optional - Temperature in K
    
    Returns:
    --------
    The missing parameter (L, R, or T)
    
    Example:
    --------
    T_eff = star.stefan_boltzmann(L=star.luminosity, R=star.radius)
    """
    if L is not None and R is not None:
        # Solve for T
        return (L / (4 * np.pi * STEFAN_BOLTZMANN_CONST * R**2))**0.25
    elif L is not None and T is not None:
        # Solve for R
        return np.sqrt(L / (4 * np.pi * STEFAN_BOLTZMANN_CONST * T**4))
    elif R is not None and T is not None:
        # Solve for L
        return 4 * np.pi * STEFAN_BOLTZMANN_CONST * R**2 * T**4
    else:
        raise ValueError("Must provide exactly two parameters")
```

#### `planck_function(self, wavelength_cm)`
Calculate the Planck distribution B_λ(T) in CGS units:
```
B_λ = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
```
- Input: wavelength in cm (can be array)
- Output: spectral radiance in erg/(s⋅cm²⋅sr⋅cm)
- **Hint:** Watch for numerical overflow at short wavelengths!

#### `peak_wavelength(self)`
Calculate Wien's displacement law wavelength:
```
λ_max = b/T where b = 0.28977719551 cm⋅K
```
Return in cm.

#### `luminosity_solar(self)` and `radius_solar(self)`
Return luminosity in L_☉ and radius in R_☉ for convenient display.

#### `__str__(self)` and `__repr__(self)`
Display star properties in both CGS and solar units.

### Scaffolding Provided
```python
# You will be provided:
from tout_stellar_evolution import (
    zams_properties,      # Returns R_sun, L_sun for ZAMS given M_sun, Z
    main_sequence_lifetime  # Returns t_MS in Myr given M_sun, Z
)

# Import your units module
from units_cgs import *

# Example usage:
def __init__(self, mass_msun, metallicity=0.02):
    self.mass = mass_msun * SOLAR_MASS  # Convert to grams
    self.metallicity = metallicity
    
    # Get ZAMS properties from Tout module
    r_sun, l_sun = zams_properties(mass_msun, metallicity)
    self.radius = r_sun * SOLAR_RADIUS   # Convert to cm
    self.luminosity = l_sun * SOLAR_LUMINOSITY  # Convert to erg/s
    
    # Calculate temperature from Stefan-Boltzmann
    self.temperature = self.stefan_boltzmann(L=self.luminosity, R=self.radius)
    
    # Get main sequence lifetime
    self.ms_lifetime = main_sequence_lifetime(mass_msun, metallicity) * 1e6 * YEAR  # Convert to seconds
```

## Part 3: The `StellarPopulation` Class (30%)

Create a `StellarPopulation` class that manages multiple stars efficiently using NumPy.

### Required Attributes (all in CGS)
- `n_stars` - number of stars
- `masses` - NumPy array of masses (g)
- `radii` - NumPy array of radii (cm)
- `luminosities` - NumPy array of luminosities (erg/s)
- `temperatures` - NumPy array of temperatures (K)
- `metallicity` - metallicity (single value for population)
- `ms_lifetimes` - NumPy array of main sequence lifetimes (s)

### Required Methods

#### `__init__(self, n_stars, mass_range=(0.1, 100), metallicity=0.02, sampling='linear')`
Initialize ZAMS population with:
- `n_stars`: number of stars to generate
- `mass_range`: tuple of (min_mass, max_mass) in M☉
- `metallicity`: metallicity (0.001 ≤ Z ≤ 0.03)
- `sampling`: how to generate mass array ('linear', 'log', 'arange', 'random')

Mass sampling methods (all generate masses between 0.1-100 M☉):
```python
if sampling == 'linear':
    # Linearly spaced masses
    # https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    masses_msun = np.linspace(mass_range[0], mass_range[1], n_stars)
    
elif sampling == 'log':
    # Logarithmically spaced masses (better for wide mass ranges)
    # https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
    masses_msun = np.logspace(np.log10(mass_range[0]), 
                              np.log10(mass_range[1]), n_stars)
                              
elif sampling == 'arange':
    # Fixed step size, may not give exactly n_stars
    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    step = (mass_range[1] - mass_range[0]) / n_stars
    masses_msun = np.arange(mass_range[0], mass_range[1], step)[:n_stars]
    
elif sampling == 'random':
    # Uniform random distribution
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    masses_msun = np.random.uniform(low=mass_range[0], high=mass_range[1], size=n_stars)
```

Store all values internally in CGS units.

#### `calculate_luminosities(self)`
Vectorized computation of all stellar luminosities using NumPy operations.
```python
# Vectorized Stefan-Boltzmann (no loops!)
self.luminosities = 4 * np.pi * STEFAN_BOLTZMANN_CONST * self.radii**2 * self.temperatures**4
```

#### `total_luminosity(self)`
Return sum of all stellar luminosities in erg/s and L_☉.

#### `plot_hr_diagram(self, filename='hr_diagram.png')`
Create Hertzsprung-Russell diagram:
- x-axis: Temperature (K) - use log scale, reversed (hot on left)
- y-axis: Luminosity (L_☉) - use log scale
- Points colored by mass
- Proper labels with units
- Add main sequence line for reference

#### `mass_luminosity_relation(self, filename='ml_relation.png')`
Plot the mass-luminosity relation:
- x-axis: Mass (M_☉) - log scale
- y-axis: Luminosity (L_☉) - log scale
- Include theoretical L ∝ M^3.5 line for comparison

### Vectorization Requirements

All operations must use NumPy arrays, no explicit loops for calculations:
```python
# Bad (loop-based):
luminosities = []
for i in range(n_stars):
    L = 4 * np.pi * STEFAN_BOLTZMANN_CONST * radii[i]**2 * temperatures[i]**4
    luminosities.append(L)

# Good (vectorized):
luminosities = 4 * np.pi * STEFAN_BOLTZMANN_CONST * radii**2 * temperatures**4
```

## Part 4: Analysis & Visualization (20%)

### Required Demonstrations of Mass Sampling

Create populations demonstrating each sampling method:
```python
# Demonstrate all sampling methods (100 stars each)
pop_linear = StellarPopulation(100, sampling='linear')
pop_log = StellarPopulation(100, sampling='log')
pop_arange = StellarPopulation(100, sampling='arange')
pop_random = StellarPopulation(100, sampling='random')

# Plot histograms comparing mass distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# ... plot each distribution to show differences
```

### Required Multi-Panel Plot (`zams_analysis.png`)

Create a 2x2 subplot figure showing metallicity effects on ZAMS:

1. **Panel 1: HR Diagrams for Different Metallicities**
   - Plot ZAMS for Z = 0.001, 0.004, 0.008, 0.02, 0.03
   - Use `sampling='log'` with 100 stars per metallicity (best for HR diagrams)
   - Different color for each Z
   - Include legend

2. **Panel 2: Main Sequence Lifetime vs Mass**
   - Plot t_MS vs M for same Z values
   - Use log-log scale
   - Show how metal-poor stars live longer

3. **Panel 3: Mass-Luminosity Relation**
   - Show L(M) for different Z values
   - Include theoretical L ∝ M^3.5 line
   - Demonstrate metallicity effects on luminosity

4. **Panel 4: Mass-Radius Relation**  
   - Show R(M) for different Z values
   - Use log-log scale
   - Show how metal-poor stars are more compact

### Additional Requirements

5. Create a stellar population and analyze:
   ```python
   # Use log spacing for good coverage
   population = StellarPopulation(1000, sampling='log', metallicity=0.02)
   print(f"Total luminosity: {population.total_luminosity()} erg/s")
   print(f"Total luminosity: {population.total_luminosity()/SOLAR_LUMINOSITY:.2e} L_sun")
   ```

6. Compare sampling methods visually:
   ```python
   # Show why log spacing is better for wide mass ranges
   linear_masses = np.linspace(0.1, 100, 50)
   log_masses = np.logspace(np.log10(0.1), np.log10(100), 50)
   # Plot both to show coverage differences
   ```

7. Verify your Stefan-Boltzmann solver works all three ways

## Deliverables

Your GitHub repository should contain:
- `units_cgs.py` - CGS units module with all constants
- `star.py` - Implementation of Star class
- `stellar_population.py` - Implementation of StellarPopulation class  
- `main.py` - Script demonstrating all functionality
- `test_stellar.py` - At least 5 unit tests (including units verification)
- `README.md` - Brief description and instructions
- `figures/` - Directory with all generated plots
- `requirements.txt` - Package dependencies

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Units Module** | 20 | All constants correct in CGS, conversion functions work |
| **Star Class** | 30 | Correct physics implementation, proper OOP design |
| **StellarPopulation Class** | 30 | Efficient vectorization, no unnecessary loops |
| **Visualizations** | 10 | Clear, labeled plots with proper scales |
| **Code Quality** | 10 | Documentation, tests, style, Git history |

## Hints & Tips

1. **Understanding NumPy sampling functions**:
   - `np.linspace(start, stop, n)`: n evenly spaced points (good for uniform coverage)
   - `np.logspace(log10(start), log10(stop), n)`: n log-spaced points (better for wide ranges)
   - `np.arange(start, stop, step)`: fixed step size (may not give exact n)
   - `np.random.uniform(low, high, n)`: random uniform distribution

2. **Why different samplings matter**:
   - **Linear**: Equal spacing, but poor coverage of low-mass stars
   - **Log**: Better coverage across orders of magnitude (0.1 to 100 M☉)
   - **Arange**: Good for fixed resolution studies
   - **Random**: Simple uniform random distribution

3. **CGS Unit Checks**: 
   - Luminosity should be ~10³³ erg/s for Sun
   - Radius should be ~10¹⁰ cm for Sun
   - Always verify dimensions match

4. **Numerical stability**: For Planck function at short wavelengths:
   ```python
   # Use np.expm1 for numerical stability
   denominator = np.expm1(PLANCK_CONST * SPEED_LIGHT / (wavelength * BOLTZMANN_CONST * T))
   ```

5. **Testing ideas**:
   - Solar values: Star(1.0, 0.02) should have L ≈ 3.828×10³³ erg/s
   - Sampling: Verify `len(masses) == n_stars` for all methods
   - Stefan-Boltzmann: All three solver modes should be self-consistent

## Extensions (Optional, for extra learning)

- Add stellar evolution: `evolve(dt)` method
- Implement color indices (B-V)
- Add binary star systems
- Include metallicity effects
- Animate HR diagram evolution over time

## Getting Started

```python
# Minimal working example to test your setup
import numpy as np
import matplotlib.pyplot as plt
from units_cgs import *

# Test your units module
print(f"Stefan-Boltzmann (CGS): {STEFAN_BOLTZMANN_CONST:.2e} erg/(cm²⋅s⋅K⁴)")
print(f"Solar luminosity: {SOLAR_LUMINOSITY:.2e} erg/s")

# Test your first star
sun = Star(mass_msun=1.0, age_myr=4600)  # 4.6 Gyr old Sun
print(f"Solar luminosity: {sun.luminosity:.2e} erg/s")
print(f"Solar luminosity: {sun.luminosity_solar():.2f} L_☉")

# Verify units are consistent
assert np.isclose(sun.luminosity / SOLAR_LUMINOSITY, 1.0, rtol=0.1)
```

## Questions?

- Check the course Slack first
- Office hours: Thursdays 1-2 PM
- Include minimal reproducible example with any bug reports