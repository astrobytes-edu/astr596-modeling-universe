# ⚠️ ASTR 596 Project 1: Python Fundamentals + Numerical Analysis + Stellar Physics
**Duration**: 2 weeks
**Weight**: 10% of course grade
**Theme**: "From Observations to Blackbody Physics"

---

## Project Overview

This project establishes the computational and theoretical foundations for the entire course. You will develop essential Python programming skills, implement numerical analysis methods from scratch, and apply them to fundamental stellar physics problems. By the end, you'll have a complete toolkit for stellar analysis that will be used throughout subsequent projects.

## Learning Objectives

By completing this project, you will:
- **Master Python fundamentals**: Functions, classes, data structures, and scientific computing libraries
- **Implement numerical methods**: Integration techniques and root-finding algorithms
- **Understand stellar physics**: Blackbody radiation, Wien's law, and stellar classification
- **Develop professional practices**: Version control, testing, documentation, and code organization
- **Build modular software**: Create reusable components for astrophysical calculations

---

# Week 1: Development Environment and Stellar Data Analysis

## Conceptual Introduction (30 min)
- Course overview and computational astrophysics in modern research
- Software development workflow: Git, GitHub, and collaborative coding
- Python ecosystem for astronomy: NumPy, Matplotlib, Pandas, Astropy
- Introduction to stellar observations and the Hertzsprung-Russell diagram

## Lab Session Objectives
Set up professional development environment and begin astronomical data analysis.

### Task 1: Environment Setup (30 min)
**Goal**: Establish reproducible computational environment

**Instructions**:
1. **Install Miniconda/Anaconda**
   - Download from official site
   - Create course-specific environment: `conda create -n astr596 python=3.11`
   - Install essential packages: `numpy matplotlib pandas jupyter astropy`

2. **Git and GitHub Setup**
   - Create GitHub account if needed
   - Configure Git with your name and email
   - Fork the course repository template
   - Clone your fork locally: `git clone <your-repo-url>`

3. **IDE Configuration**
   - Install VS Code or preferred editor
   - Configure Python interpreter to use conda environment
   - Install useful extensions: Python, Jupyter, GitLens

**Deliverable**: Screenshot of successful environment test

### Task 2: Python Fundamentals Review (60 min)
**Goal**: Refresh/establish core Python programming skills

**Core Concepts to Implement**:
```python
# Data types and operations
def basic_arithmetic_operations():
    """Practice with numbers, strings, lists, dictionaries."""
    
# Control structures
def stellar_magnitude_classifier(magnitude):
    """Classify stars by brightness using if/elif/else."""
    
# File I/O
def load_stellar_catalog(filename):
    """Read CSV data using both built-in and pandas methods."""
    
# List comprehensions and basic algorithms
def filter_stars_by_criteria(catalog, min_magnitude, max_magnitude):
    """Filter stellar data using comprehensions and boolean indexing."""
```

**Practice Dataset**: Hipparcos catalog subset (provided)

**Validation**: Compare your results with provided reference outputs

### Task 3: First Astronomical Analysis (40 min)
**Goal**: Apply Python skills to real astronomical data

**Implementation Requirements**:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_hipparcos_data(filename):
    """Load and clean Hipparcos stellar catalog."""
    # Handle missing data and outliers
    # Convert magnitude and color data to proper types
    # Calculate distances from parallax measurements
    
def basic_stellar_statistics(catalog):
    """Calculate fundamental statistics of stellar sample."""
    # Mean, median, standard deviation of key parameters
    # Magnitude distributions
    # Color distributions
    
def create_color_magnitude_diagram(catalog):
    """Generate first HR diagram."""
    # Plot B-V color vs absolute magnitude
    # Add proper axis labels and title
    # Include error bars where appropriate
```

**Analysis Questions**:
1. What is the range of stellar magnitudes in the sample?
2. How many stars have reliable parallax measurements?
3. What patterns do you observe in the color-magnitude diagram?

**Week 1 Deliverable**: Jupyter notebook with environment setup, Python exercises, and basic stellar analysis

---

# Week 2: Numerical Integration and Blackbody Physics

## Conceptual Introduction (25 min)
- Blackbody radiation and the Planck function
- Stefan-Boltzmann law and Wien's displacement law
- Numerical integration: why and when we need it
- Trapezoidal rule, Simpson's rule, and Gaussian quadrature

## Lab Session Objectives
Implement numerical integration methods and apply them to stellar radiation calculations.

### Task 1: Numerical Integration Library (45 min)
**Goal**: Build integration toolkit from first principles

**Required Implementations**:
```python
import numpy as np

def trapezoid_rule(func, a, b, n_points):
    """
    Implement trapezoidal rule integration.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    a, b : float
        Integration limits
    n_points : int
        Number of grid points
        
    Returns:
    --------
    integral : float
        Numerical approximation of integral
    """
    # YOUR IMPLEMENTATION HERE
    # Calculate spacing: h = (b-a)/(n_points-1)
    # Create grid points
    # Apply trapezoidal rule formula
    
def simpson_rule(func, a, b, n_points):
    """
    Implement Simpson's rule (requires odd number of points).
    More accurate than trapezoidal rule for smooth functions.
    """
    # YOUR IMPLEMENTATION HERE
    # Ensure n_points is odd
    # Apply Simpson's 1/3 rule
    
def gaussian_quadrature(func, a, b, n_points):
    """
    Implement Gaussian quadrature for high accuracy.
    Use numpy.polynomial.legendre.leggauss for weights and nodes.
    """
    # YOUR IMPLEMENTATION HERE
    # Transform from [-1,1] to [a,b]
    # Apply Gaussian quadrature formula
```

**Validation Tests**:
- Test on functions with known integrals: x², sin(x), exp(x)
- Compare accuracy vs computational cost
- Plot convergence as function of n_points

### Task 2: Blackbody Radiation Functions (60 min)
**Goal**: Implement Planck function and related stellar physics

**Core Physics Implementation**:
```python
# Physical constants (use astropy.constants for precision)
h = 6.626e-34      # Planck constant [J⋅s]
c = 2.998e8        # Speed of light [m/s]
k_B = 1.381e-23    # Boltzmann constant [J/K]
sigma_SB = 5.67e-8 # Stefan-Boltzmann constant [W/m²/K⁴]

def planck_function_frequency(nu, T):
    """
    Planck function B_ν(T) in frequency units.
    
    Parameters:
    -----------
    nu : float or array
        Frequency [Hz]
    T : float
        Temperature [K]
        
    Returns:
    --------
    B_nu : float or array
        Spectral radiance [W/m²/Hz/sr]
    """
    # B_ν(T) = (2hν³/c²) × 1/(exp(hν/kT) - 1)
    # Handle numerical issues for small and large arguments
    
def planck_function_wavelength(wavelength, T):
    """
    Planck function B_λ(T) in wavelength units.
    
    Note: B_λ dλ = B_ν dν, so B_λ = B_ν × (dν/dλ) = B_ν × (c/λ²)
    """
    # Convert wavelength to frequency and apply conversion
    
def stellar_luminosity_integral(T_eff, R_star):
    """
    Calculate stellar luminosity by integrating Planck function.
    
    L = 4πR² ∫ π B_ν(T) dν = 4πR² σT⁴
    
    Verify Stefan-Boltzmann law numerically.
    """
    # Integrate Planck function over all frequencies
    # Compare with analytical Stefan-Boltzmann result
    # Calculate relative error
```

**Stellar Applications**:
```python
def frequency_integrated_intensity(T_eff, nu_min, nu_max):
    """
    Integrate Planck function over frequency range.
    Critical for radiation pressure calculations in Project 3.
    """
    
def stellar_flux_at_distance(L_star, distance):
    """
    Calculate observed flux: F = L/(4πd²)
    """
    
def main_sequence_relations():
    """
    Implement empirical mass-luminosity and mass-temperature relations.
    L ∝ M³⋅⁵ for M > 1 M☉
    T_eff ∝ M⁰⋅⁵ for main sequence stars
    """
```

### Task 3: Integration Method Comparison (30 min)
**Goal**: Understand computational trade-offs in numerical methods

**Analysis Requirements**:
1. **Accuracy Study**: For each integration method, plot error vs n_points
2. **Performance Study**: Time each method for various n_points
3. **Function Sensitivity**: How do methods perform on oscillatory vs smooth functions?

**Test Functions**:
- Smooth: Planck function at various temperatures
- Oscillatory: Wien displacement law integral
- Sharp features: Planck function at low temperatures

**Week 2 Deliverable**: Integration library with comprehensive testing and stellar luminosity calculations

---

# Week 3: Root-Finding and Object-Oriented Design

## Conceptual Introduction (25 min)
- Root-finding problems in astrophysics
- Bisection method: guaranteed convergence
- Newton-Raphson method: fast convergence with derivatives
- Secant method: fast convergence without derivatives
- Object-oriented programming: when and why to use classes

## Lab Session Objectives
Implement root-finding algorithms and design object-oriented stellar analysis framework.

### Task 1: Root-Finding Algorithm Library (45 min)
**Goal**: Build robust root-finding toolkit

**Required Implementations**:
```python
def bisection_method(func, a, b, tolerance=1e-6, max_iterations=100):
    """
    Find root using bisection method.
    
    Guaranteed to converge if func(a) and func(b) have opposite signs.
    Slow but robust.
    """
    # Check preconditions: func(a) * func(b) < 0
    # Implement bisection algorithm
    # Return root, number of iterations, convergence status
    
def newton_raphson(func, func_derivative, x0, tolerance=1e-6, max_iterations=50):
    """
    Find root using Newton-Raphson method.
    
    Fast convergence but requires derivative and good initial guess.
    """
    # Implement Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n)
    # Handle cases where derivative is zero
    # Return root, number of iterations, convergence status
    
def secant_method(func, x0, x1, tolerance=1e-6, max_iterations=50):
    """
    Find root using secant method.
    
    Fast convergence without requiring derivative.
    """
    # Implement secant method using finite difference approximation
    # Handle cases where function values are too close
    # Return root, number of iterations, convergence status
```

### Task 2: Wien's Law and Stellar Physics Applications (60 min)
**Goal**: Apply root-finding to solve transcendental equations in stellar physics

**Wien's Displacement Law Implementation**:
```python
def wien_displacement_equation(x):
    """
    Wien's law equation: 5 - x = 5*exp(-x)
    where x = hc/(λ_max * k_B * T)
    
    Root occurs at x ≈ 4.965114
    """
    return 5 - x - 5*np.exp(-x)

def wien_displacement_derivative(x):
    """Analytical derivative for Newton-Raphson method."""
    return -1 + 5*np.exp(-x)

def find_peak_wavelength(temperature):
    """
    Find wavelength of peak emission for given temperature.
    
    Uses root-finding to solve Wien's displacement law.
    """
    # Solve Wien equation for x
    # Convert to wavelength: λ_max = hc/(x * k_B * T)
    # Validate with Wien's constant: λ_max * T = 2.898e-3 m⋅K
    
def temperature_from_color_index(color_bv, color_system='Johnson'):
    """
    Invert empirical color-temperature relations using root-finding.
    
    Example relation: log(T_eff) = 3.979 - 0.654*log(B-V + 1.334)
    """
    # Define equation to solve: observed_color - predicted_color(T) = 0
    # Use appropriate root-finding method
    # Handle edge cases and invalid inputs
```

### Task 3: Object-Oriented Stellar Analysis Framework (30 min)
**Goal**: Design clean, extensible code architecture

**Class Design**:
```python
class Star:
    """
    Represents individual star with physical and observational properties.
    """
    
    def __init__(self, name=None, mass=None, temperature=None, radius=None, 
                 magnitude_v=None, color_bv=None, parallax=None):
        """Initialize star with observational or theoretical data."""
        self.name = name
        self.mass = mass  # Solar masses
        self.temperature = temperature  # Kelvin
        self.radius = radius  # Solar radii
        self.magnitude_v = magnitude_v  # Apparent V magnitude
        self.color_bv = color_bv  # B-V color index
        self.parallax = parallax  # arcseconds
        
    def distance(self):
        """Calculate distance from parallax [pc]."""
        if self.parallax is None or self.parallax <= 0:
            return None
        return 1.0 / self.parallax
    
    def absolute_magnitude(self):
        """Calculate absolute magnitude."""
        d = self.distance()
        if d is None or self.magnitude_v is None:
            return None
        return self.magnitude_v - 5*np.log10(d/10)
    
    def luminosity(self):
        """Calculate luminosity using Stefan-Boltzmann law."""
        if self.temperature is None or self.radius is None:
            return None
        return 4*np.pi*(self.radius*R_sun)**2 * sigma_SB * self.temperature**4
    
    def estimate_temperature_from_color(self):
        """Estimate temperature from B-V color using root-finding."""
        if self.color_bv is None:
            return None
        return temperature_from_color_index(self.color_bv)
    
    def stellar_type(self):
        """Classify star based on temperature or color."""
        # Implement spectral classification (O, B, A, F, G, K, M)
        
    def radiation_pressure_luminosity(self):
        """
        Calculate luminosity for radiation pressure calculations.
        This method will be used in Project 3.
        """
        return self.luminosity()

class StellarCatalog:
    """
    Collection of stars with analysis capabilities.
    """
    
    def __init__(self, stars=None):
        """Initialize with list of Star objects."""
        self.stars = stars if stars is not None else []
    
    @classmethod
    def from_file(cls, filename):
        """Load catalog from CSV file."""
        # Read data and create Star objects
        
    def filter_by_criteria(self, **criteria):
        """Filter stars based on various criteria."""
        # magnitude_range, color_range, distance_range, etc.
        
    def create_hr_diagram(self, save_path=None):
        """
        Generate publication-quality HR diagram.
        """
        # Plot absolute magnitude vs color or temperature
        # Add theoretical main sequence track
        # Include stellar classification regions
        
    def statistical_summary(self):
        """Generate comprehensive statistical analysis."""
        # Distributions of key parameters
        # Correlations between observables
```

**Week 3 Deliverable**: Complete stellar analysis package with object-oriented design, root-finding applications, and advanced HR diagram analysis

---

# Assessment and Grading

## Grading Breakdown
- **Week 1**: Environment setup and Python fundamentals (30%)
- **Week 2**: Numerical integration and blackbody physics (35%)  
- **Week 3**: Root-finding and OOP implementation (35%)

## Evaluation Criteria

### Technical Implementation (60%)
- **Correctness**: Do algorithms produce accurate results?
- **Efficiency**: Are implementations reasonably optimized?
- **Robustness**: Does code handle edge cases and errors gracefully?
- **Testing**: Are functions validated against known results?

### Code Quality (25%)
- **Documentation**: Clear docstrings and comments
- **Organization**: Logical file structure and function design
- **Style**: Follows Python conventions (PEP 8)
- **Version Control**: Meaningful commit messages and regular commits

### Scientific Understanding (15%)
- **Physics**: Correct implementation of stellar physics
- **Validation**: Appropriate comparison with analytical results
- **Interpretation**: Understanding of numerical method trade-offs

## Deliverables

### Final Submission Requirements
1. **Complete Python Package**:
   - `stellar_physics.py`: Blackbody and stellar property functions
   - `numerical_methods.py`: Integration and root-finding algorithms
   - `stellar_analysis.py`: Star and StellarCatalog classes
   - `tests/`: Comprehensive test suite
   - `README.md`: Installation and usage instructions

2. **Analysis Notebooks**:
   - `week1_python_fundamentals.ipynb`: Environment setup and basic analysis
   - `week2_numerical_integration.ipynb`: Integration methods and stellar applications
   - `week3_stellar_classification.ipynb`: Advanced HR diagram analysis

3. **Validation Report**: Document comparing your results with literature values

## Connection to Future Projects

This project establishes foundations used throughout the course:
- **Project 2**: Stellar mass-luminosity relations for realistic N-body clusters
- **Project 3**: Blackbody stellar spectra for radiation heating calculations
- **Project 4**: Numerical integration techniques translated to JAX
- **Final Project**: Object-oriented design principles for research-grade software

## Getting Help

- **Office Hours**: Use for conceptual questions and debugging assistance
- **Pair Programming**: Collaborate during lab sessions but submit individual work
- **Discussion Forum**: Share general questions and solutions to common issues
- **Online Resources**: Python documentation, NumPy tutorials, Astropy guides

This project sets the stage for sophisticated computational astrophysics while ensuring students master fundamental programming and numerical analysis skills.