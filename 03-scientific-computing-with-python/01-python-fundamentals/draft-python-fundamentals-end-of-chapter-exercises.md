# Example End-of-Chapter Practice Exercises

## ⚠️ Chapter 1: Python Environments & Scientific Workflows

### ⚠️ Practice Exercises

#### ⚠️ Exercise 1.1: IPython Mastery

:::{admonition} Part A: Explore Scientific Libraries (5 min)
:class: exercise, dropdown

Execute these commands in IPython to explore `astropy`:

```python
# In IPython:
import astropy
import astropy.units as u
import astropy.constants as const

# Explore available constants
print("Astronomical constants:", 
      [x for x in dir(const) if not x.startswith('_')][:10])

# Quick calculation: Jeans mass
T = 10 * u.K  # Molecular cloud temperature
n = 1e4 * u.cm**-3  # Number density

# Calculate Jeans mass (simplified)
M_J = 2.0 * (const.k_B * T / (const.G * const.m_p))**(3/2) * n**(-1/2)
print(f"Jeans mass: {M_J.to(u.M_sun):.1f}")
```

:::

:::{admonition} ⚠️ Part B: Time Array Operations (10 min)
:class: exercise, dropdown

Compare different methods for calculating stellar distances:

```python
import numpy as np
import timeit

# Method 1: List comprehension
def distance_list(parallaxes_mas):
    """Calculate distances using list comprehension."""
    return [1000.0/p if p > 0 else np.nan 
            for p in parallaxes_mas]

# Method 2: NumPy vectorized
def distance_numpy(parallaxes_mas):
    """Calculate distances using NumPy."""
    par = np.array(parallaxes_mas)
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = 1000.0 / par
        distances[par <= 0] = np.nan
    return distances

# Test data: parallaxes in milliarcseconds
# Include some bad data (negative, zero)
np.random.seed(42)
test_parallaxes = np.random.exponential(2, 1000)
test_parallaxes[::50] = -1  # Some bad measurements

# Time both methods
t1 = timeit.timeit(
    lambda: distance_list(test_parallaxes), 
    number=100
)
t2 = timeit.timeit(
    lambda: distance_numpy(test_parallaxes), 
    number=100
)

print(f"List comprehension: {t1*10:.3f} ms")
print(f"NumPy vectorized:   {t2*10:.3f} ms")
print(f"NumPy is {t1/t2:.1f}x faster!")
```

:::

:::{admonition} ⚠️ Part C: Create Your Own Analysis (15 min)
:class: exercise, dropdown

Design a timing experiment for period-finding algorithms:

```python
import numpy as np
import timeit

# Generate synthetic light curve
np.random.seed(42)
n_points = 1000
times = np.sort(np.random.uniform(0, 100, n_points))
true_period = 2.35  # days
true_amplitude = 0.5  # magnitudes

# Create variable star signal
signal = true_amplitude * np.sin(2 * np.pi * times / true_period)
noise = np.random.normal(0, 0.05, n_points)
magnitudes = 15.0 + signal + noise

# Method 1: Lomb-Scargle periodogram (simplified)
def lomb_scargle_simple(t, y, periods):
    """Simplified Lomb-Scargle (for demonstration)."""
    powers = []
    for period in periods:
        omega = 2 * np.pi / period
        cos_wt = np.cos(omega * t)
        sin_wt = np.sin(omega * t)
        
        # Simplified power calculation
        c = np.sum(y * cos_wt)
        s = np.sum(y * sin_wt)
        power = c**2 + s**2
        powers.append(power)
    return np.array(powers)

# Method 2: String-length method
def string_length(t, y, periods):
    """String-length period finding."""
    lengths = []
    for period in periods:
        # Fold light curve
        phases = (t % period) / period
        # Sort by phase
        idx = np.argsort(phases)
        y_sorted = y[idx]
        
        # Calculate string length
        length = np.sum(np.abs(np.diff(y_sorted)))
        lengths.append(length)
    return np.array(lengths)

# Test periods
test_periods = np.linspace(0.5, 5.0, 100)

# Time both methods
t1 = timeit.timeit(
    lambda: lomb_scargle_simple(times, magnitudes, test_periods),
    number=10
)
t2 = timeit.timeit(
    lambda: string_length(times, magnitudes, test_periods),
    number=10
)

print(f"Lomb-Scargle: {t1*100:.1f} ms")
print(f"String-length: {t2*100:.1f} ms")
print(f"Ratio: {t1/t2:.2f}x")

# Find the period
powers = lomb_scargle_simple(times, magnitudes, test_periods)
best_period = test_periods[np.argmax(powers)]
print(f"\nTrue period: {true_period:.3f} days")
print(f"Found period: {best_period:.3f} days")
```
:::

#### ⚠️ Exercise 1.2: Notebook State Detective - Cosmology Edition

:::{admonition} Part A: Trace the Cosmological Calculation (5 min)
:class: exercise, dropdown

Given this notebook execution order, trace the state:

```python
# Execution order: Cell 1, Cell 3, Cell 2, Cell 4, Cell 2, Cell 4

Cell 1: H0 = 70.0  # km/s/Mpc
        omega_m = 0.3
        omega_lambda = 0.7

Cell 2: def age_of_universe():
            # Simplified calculation
            from math import sqrt
            H0_SI = H0 * 1000 / 3.086e22  # Convert to 1/s
            age = (2/3) / H0_SI / sqrt(omega_lambda)
            return age / (365.25 * 24 * 3600 * 1e9)  # Gyr

Cell 3: H0 = 67.4  # Planck value
        omega_lambda = 0.685

Cell 4: print(f"Age: {age_of_universe():.2f} Gyr")

```

**What age gets printed each time? Which cosmology is used?**

:::

:::{admonition} ⚠️ Part B: Find the Bug (10 min)
:class: exercise, dropdown

```python
# Simulate the execution to find the bug
H0 = 70.0  # Cell 1
omega_m = 0.3
omega_lambda = 0.7

H0 = 67.4  # Cell 3
omega_lambda = 0.685

# Cell 2 - function captures H0 and omega_lambda NOW
def age_of_universe():
    from math import sqrt
    H0_SI = H0 * 1000 / 3.086e22
    age = (2/3) / H0_SI / sqrt(omega_lambda)
    return age / (365.25 * 24 * 3600 * 1e9)

print(f"First Cell 4: Age = {age_of_universe():.2f} Gyr")

# Cell 2 again - NOW it uses updated values!
def age_of_universe():
    from math import sqrt
    H0_SI = H0 * 1000 / 3.086e22
    age = (2/3) / H0_SI / sqrt(omega_lambda)
    return age / (365.25 * 24 * 3600 * 1e9)

print(f"Second Cell 4: Age = {age_of_universe():.2f} Gyr")
print("\nThe function definition captures values when defined!")
```
:::

:::{admonition} ⚠️ Part C: Explain the Scientific Impact (15 min)
:class: exercise, dropdown

Write a paragraph explaining how this notebook behavior could affect:

1. **Cosmological parameter estimation** - Wrong H₀ leads to wrong distances
2. **Reproducibility** - Collaborators get different results
3. **Scientific conclusions** - Age estimates could be off by gigayears

**Consider:** What if this was analyzing Planck CMB data or Type Ia supernovae for dark energy constraints?
:::

#### ⚠️ Exercise 1.3: Environment Diagnostic

:::{admonition} Part A: Check Your Setup (5 min)
:class: exercise, dropdown

```python
import sys
import importlib

# Check Python and key packages
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("\nAstronomy packages:")

packages = {
    'numpy': 'Numerical computing',
    'scipy': 'Scientific algorithms', 
    'matplotlib': 'Plotting',
    'astropy': 'Core astronomy',
    'astroquery': 'Archive queries',
    'photutils': 'Photometry',
    'specutils': 'Spectroscopy'
}

for pkg, description in packages.items():
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {pkg:12} {version:10} - {description}")
    except ImportError:
        print(f"  ✗ {pkg:12} MISSING     - {description}")
```
:::

:::{admonition} ⚠️ Part B: Test Data Access (10 min)
:class: exercise, dropdown

```python
from pathlib import Path
import os

# Check standard astronomy data locations
def check_astronomy_data():
    """Check for standard astronomical data directories."""
    
    # Common environment variables
    env_vars = {
        'ASTRO_DATA': 'Local observation data',
        'CALDB': 'Calibration database',
        'PYSYN_CDBS': 'Synphot reference data',
        'WEBBPSF_PATH': 'Webb PSF data',
        'CRDS_PATH': 'Calibration references'
    }
    
    print("Astronomy Data Paths:")
    print("-" * 50)
    
    for var, description in env_vars.items():
        path = os.getenv(var)
        if path:
            path_obj = Path(path)
            exists = "✓" if path_obj.exists() else "✗"
            print(f"{exists} {var:15} = {path}")
            print(f"  └─ {description}")
        else:
            print(f"- {var:15} not set")
            print(f"  └─ {description}")
    
    # Check for common data directories
    print("\nLocal data directories:")
    for dirname in ['data', 'raw', 'reduced', 'catalogs']:
        path = Path(dirname)
        if path.exists():
            n_files = len(list(path.glob('*')))
            print(f"  ✓ ./{dirname}/ ({n_files} items)")
        else:
            print(f"  - ./{dirname}/ not found")

check_astronomy_data()
```
:::

:::{admonition} ⚠️ Part C: Complete Observatory Diagnostic (15 min)
:class: exercise, dropdown

```python
import sys
import subprocess
from pathlib import Path
import platform

def observatory_diagnostic():
    """
    Complete diagnostic for astronomical computing environment.
    
    Checks everything needed for telescope data analysis.
    """
    print("=" * 60)
    print("OBSERVATORY COMPUTING ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    
    # 1. System info
    print("\n1. SYSTEM INFORMATION:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # 2. Memory check (important for large images)
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n2. MEMORY:")
        print(f"   Total: {mem.total / 1e9:.1f} GB")
        print(f"   Available: {mem.available / 1e9:.1f} GB")
    except ImportError:
        print("\n2. MEMORY: psutil not installed")
    
    # 3. Check for astronomy tools
    print("\n3. ASTRONOMY TOOLS:")
    tools = {
        'ds9': 'SAOImage DS9 viewer',
        'fv': 'FITS viewer',
        'topcat': 'Table analysis',
        'aladin': 'Sky atlas'
    }
    
    for tool, description in tools.items():
        try:
            result = subprocess.run(
                ['which', tool], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"   ✓ {tool:8} - {description}")
            else:
                print(f"   - {tool:8} - {description}")
        except:
            print(f"   ? {tool:8} - {description}")
    
    # 4. Python packages for specific telescopes
    print("\n4. TELESCOPE-SPECIFIC PACKAGES:")
    telescope_packages = {
        'drizzlepac': 'HST data',
        'jwst': 'JWST pipeline',
        'ccdproc': 'CCD reduction',
        'pyraf': 'IRAF tasks'
    }
    
    for pkg, telescope in telescope_packages.items():
        try:
            __import__(pkg)
            print(f"   ✓ {pkg:12} - {telescope}")
        except ImportError:
            print(f"   - {pkg:12} - {telescope}")
    
    print("\n" + "=" * 60)
    return True

# Run the diagnostic
observatory_diagnostic()
```
:::

#### ⚠️ Exercise 1.4: Variable Star Exercise Thread

**TODO:** Fix to a simpler problem, this or similar exercise should be in Chapter 6 (i.e., provide partial class implementation for them to complete.)

:::{admonition} Chapter 1: Variable Star Analysis Foundation
:class: exercise, dropdown

```python
# Chapter 1: Variable Star Analysis - Professional Foundation
# This will grow into a complete variable star analysis pipeline

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

class VariableStarObservation:
    """
    Container for variable star observations.
    
    This class will be expanded in each chapter to include:
    - Chapter 2: Numerical precision for period analysis
    - Chapter 3: Time series arrays and phase folding
    - Chapter 4: Period finding algorithms
    - Chapter 5: Statistical analysis and error propagation
    - Chapter 6: Full object-oriented analysis pipeline
    """
    
    def __init__(self, star_name, star_type, period, 
                 mag_mean, mag_amplitude, epoch=None):
        """
        Initialize variable star observation.
        
        Parameters
        ----------
        star_name : str
            Star designation (e.g., 'Delta Cephei')
        star_type : str
            Variable type ('Cepheid', 'RR Lyrae', etc.)
        period : float
            Period in days
        mag_mean : float
            Mean magnitude
        mag_amplitude : float
            Peak-to-peak amplitude
        epoch : float, optional
            Reference epoch (JD)
        """
        self.star_name = star_name
        self.star_type = star_type
        self.period = period
        self.mag_mean = mag_mean
        self.mag_amplitude = mag_amplitude
        self.epoch = epoch or 2451545.0  # J2000.0 default
        
        # Metadata for reproducibility
        self.metadata = {
            'created': datetime.now().isoformat(),
            'python_version': sys.version.split()[0],
            'environment': sys.executable,
            'numpy_version': np.__version__
        }
    
    def __str__(self):
        """String representation."""
        return (f"{self.star_name} ({self.star_type}): "
                f"P={self.period:.4f}d, "
                f"<m>={self.mag_mean:.2f}, "
                f"Δm={self.mag_amplitude:.2f}")
    
    def phase_fold(self, times, magnitudes):
        """
        Phase fold observations (preview of Chapter 3).
        
        Parameters
        ----------
        times : array-like
            Observation times (JD)
        magnitudes : array-like
            Observed magnitudes
            
        Returns
        -------
        phases : array
            Phases (0-1)
        folded_mags : array
            Phase-folded magnitudes
        """
        # Calculate phases
        phases = ((times - self.epoch) % self.period) / self.period
        
        # Sort by phase
        sort_idx = np.argsort(phases)
        return phases[sort_idx], np.array(magnitudes)[sort_idx]
    
    def save(self, filename=None):
        """
        Save observation with full metadata.
        
        Parameters
        ----------
        filename : str, optional
            Output filename (default: star_name_ch1.json)
        """
        if filename is None:
            # Safe filename from star name
            safe_name = self.star_name.replace(' ', '_').replace('*', 'star')
            filename = f"{safe_name}_ch1.json"
        
        data = {
            'star': {
                'name': self.star_name,
                'type': self.star_type,
                'period': self.period,
                'mag_mean': self.mag_mean,
                'mag_amplitude': self.mag_amplitude,
                'epoch': self.epoch
            },
            'metadata': self.metadata
        }
        
        path = Path(filename)
        try:
            path.write_text(json.dumps(data, indent=2))
            print(f"✓ Saved to {path.absolute()}")
            
            # Verify save
            verify = json.loads(path.read_text())
            assert verify['star']['name'] == self.star_name
            print(f"✓ Verified: {self.star_name} data intact")
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"✗ Error saving: {e}")
            return None
        
        return path
    
    @classmethod
    def load(cls, filename):
        """
        Load observation from file.
        
        Parameters
        ----------
        filename : str
            Input filename
            
        Returns
        -------
        VariableStarObservation
            Loaded observation object
        """
        path = Path(filename)
        data = json.loads(path.read_text())
        
        star_data = data['star']
        obs = cls(
            star_name=star_data['name'],
            star_type=star_data['type'],
            period=star_data['period'],
            mag_mean=star_data['mag_mean'],
            mag_amplitude=star_data['mag_amplitude'],
            epoch=star_data.get('epoch', 2451545.0)
        )
        
        # Preserve original metadata
        obs.metadata.update(data.get('metadata', {}))
        return obs


# Create example variable stars
print("Creating variable star catalog...")
print("=" * 50)

# Classical Cepheid
delta_cep = VariableStarObservation(
    star_name="Delta Cephei",
    star_type="Classical Cepheid",
    period=5.366319,
    mag_mean=3.95,
    mag_amplitude=0.88,
    epoch=2451545.0
)
print(delta_cep)
delta_cep.save()

# RR Lyrae star
rr_lyr = VariableStarObservation(
    star_name="RR Lyrae",
    star_type="RR Lyrae",
    period=0.56686776,
    mag_mean=7.92,
    mag_amplitude=1.04,
    epoch=2451545.0
)
print(rr_lyr)
rr_lyr.save()

# Eclipsing binary
algol = VariableStarObservation(
    star_name="Algol",
    star_type="Eclipsing Binary",
    period=2.8673043,
    mag_mean=2.12,
    mag_amplitude=1.27,
    epoch=2451545.0
)
print(algol)
algol.save()

print("\n✓ Chapter 1 complete: Environment configured")
print("✓ Variable star foundation established")
print("→ Next: Chapter 2 will add numerical precision analysis")
```
:::

## ⚠️ Chapter 2: Python as Your Astronomical Calculator 

## Practice Exercises

### Exercise 2.1: Magnitude and Flux Conversions

:::{admonition} Part A: Follow These Steps (5 min)
:class: exercise, dropdown

Execute this exact code to understand magnitude-flux conversion:

```{code-cell} ipython3
import math

# Step 1: Define the conversion formula
def mag_to_flux(magnitude, zero_point=0.0):
    """Convert magnitude to relative flux."""
    flux = 10**((zero_point - magnitude) / 2.5)
    return flux

# Step 2: Test with a specific magnitude
test_mag = 10.0
test_flux = mag_to_flux(test_mag)
print(f"Magnitude {test_mag} = flux {test_flux:.6f}")

# Step 3: Verify the logarithmic relationship
mag_diff = 5.0  # 5 magnitude difference
flux_ratio = mag_to_flux(0) / mag_to_flux(mag_diff)
print(f"{mag_diff} mag difference = {flux_ratio:.1f}× flux ratio")
```
:::

:::{admonition} Part B: Modify and Extend (10 min)
:class: exercise, dropdown

Now add the inverse function and test round-trip conversion:

```{code-cell} ipython3
def flux_to_mag(flux, zero_point=0.0):
    """Convert flux to magnitude with error handling."""
    if flux <= 0:
        return float('inf')  # Infinitely faint
    
    magnitude = zero_point - 2.5 * math.log10(flux)
    return magnitude

# Test round-trip conversion
original_mag = 15.5
flux = mag_to_flux(original_mag)
recovered_mag = flux_to_mag(flux)
error = abs(original_mag - recovered_mag)

print(f"Original: {original_mag}")
print(f"Recovered: {recovered_mag}")
print(f"Error: {error:.2e}")
print(f"\nWhy isn't error exactly zero?")
print("Floating-point arithmetic introduces tiny errors!")
```
:::

:::{admonition} Part C: Apply to Real Data (15 min)
:class: exercise, dropdown

Create a function that correctly averages multiple magnitude measurements:

```{code-cell} ipython3
def average_magnitudes_wrong(mag_list):
    """WRONG: Simple arithmetic mean of magnitudes."""
    return sum(mag_list) / len(mag_list)

def average_magnitudes_correct(mag_list):
    """
    CORRECT: Convert to flux, average, convert back.
    This is how professional astronomy software works!
    """
    if not mag_list:
        raise ValueError("Empty magnitude list")
    
    # Check for unreasonable values
    for mag in mag_list:
        if mag < -30 or mag > 40:
            raise ValueError(f"Magnitude {mag} outside reasonable range")
    
    # Convert to fluxes
    fluxes = [mag_to_flux(m) for m in mag_list]
    
    # Average the fluxes
    mean_flux = sum(fluxes) / len(fluxes)
    
    # Convert back to magnitude
    return flux_to_mag(mean_flux)

# Test with example data
test_mags = [10.0, 10.5, 11.0]

wrong_result = average_magnitudes_wrong(test_mags)
correct_result = average_magnitudes_correct(test_mags)

print(f"Magnitudes: {test_mags}")
print(f"Wrong (arithmetic mean): {wrong_result:.3f}")
print(f"Correct (flux-weighted): {correct_result:.3f}")
print(f"Difference: {wrong_result - correct_result:.3f} magnitudes")
print(f"\nThis difference compounds with more measurements!")
```
:::

### Exercise 2.2: Numerical Hazard Detection

:::{admonition} Part A: Identify the Problem (5 min)
:class: exercise, dropdown

Run this code and identify the numerical hazard:

```{code-cell} ipython3
# Calculating small differences in large numbers
distance1 = 1.496e13  # 1 AU in cm
distance2 = 1.496e13 + 100  # 1 meter further

difference = distance2 - distance1
print(f"Distance 1: {distance1:.10e} cm")
print(f"Distance 2: {distance2:.10e} cm")
print(f"Difference: {difference} cm")
print(f"Expected: 100 cm")
print(f"\nWhat hazard is this demonstrating?")
```
:::

:::{admonition} Part B: Implement Detection (10 min)
:class: exercise, dropdown

Create a function to detect potential catastrophic cancellation:

```{code-cell} ipython3
def detect_cancellation_risk(a, b, threshold=0.01):
    """
    Detect if subtracting a and b risks catastrophic cancellation.
    
    Returns True if |a-b| < threshold * max(|a|, |b|)
    """
    if a == 0 or b == 0:
        return False
    
    difference = abs(a - b)
    scale = max(abs(a), abs(b))
    relative_diff = difference / scale
    
    is_risky = relative_diff < threshold
    
    if is_risky:
        print(f"WARNING: Catastrophic cancellation risk!")
        print(f"Relative difference: {relative_diff:.2e}")
    
    return is_risky

# Test with our distance example
detect_cancellation_risk(distance1, distance2)

# Test with safe calculation
detect_cancellation_risk(100, 50)
```
:::

:::{admonition} Part C: Apply to Orbital Mechanics (15 min)
:class: exercise, dropdown

Implement safe calculation of orbital energy changes:

```{code-cell} ipython3
def orbital_energy_change_unsafe(r1, r2, M):
    """
    UNSAFE: Direct calculation of energy change.
    E = -GM/(2r) for circular orbit
    """
    G = 6.67e-8  # CGS
    E1 = -G * M / (2 * r1)
    E2 = -G * M / (2 * r2)
    return E2 - E1  # Catastrophic cancellation for r1 ≈ r2!

def orbital_energy_change_safe(r1, r2, M):
    """
    SAFE: Reformulated to avoid cancellation.
    ΔE = GM/2 * (1/r1 - 1/r2) = GM/2 * (r2-r1)/(r1*r2)
    """
    G = 6.67e-8
    
    if r1 == r2:
        return 0.0
    
    # Use reformulated expression
    delta_E = G * M / 2 * (r2 - r1) / (r1 * r2)
    return delta_E

# Test with nearly equal radii (1 AU ± 1 km)
r1 = 1.496e13  # 1 AU in cm
r2 = 1.496e13 + 1e5  # 1 km further
M = 1.989e33  # Solar mass

unsafe = orbital_energy_change_unsafe(r1, r2, M)
safe = orbital_energy_change_safe(r1, r2, M)

print(f"Unsafe calculation: {unsafe:.6e} erg")
print(f"Safe calculation:   {safe:.6e} erg")
print(f"Relative difference: {abs(unsafe-safe)/abs(safe):.2%}")
```

:::

### Exercise 2.3: Build a Robust Calculator (Challenge)

:::{admonition} Complete Project (20-30 min)
:class: exercise, dropdown

Build a scientific calculator with proper error handling:

```{code-cell} ipython3
import math

class ScientificCalculator:
    """A calculator with numerical safety features."""
    
    def __init__(self):
        self.history = []
        self.epsilon = sys.float_info.epsilon
    
    def safe_log(self, x, base=math.e):
        """Logarithm with validation."""
        if x <= 0:
            raise ValueError(f"Cannot take log of {x}")
        
        if base == math.e:
            result = math.log(x)
        elif base == 10:
            result = math.log10(x)
        else:
            result = math.log(x) / math.log(base)
        
        self.history.append(f"log_{base}({x}) = {result}")
        return result
    
    def safe_power(self, base, exponent):
        """Power operation with overflow protection."""
        # Check for potential overflow
        if abs(base) > 1 and exponent > 100:
            # Use log space
            log_result = exponent * math.log10(abs(base))
            if log_result > 300:  # Would overflow
                return f"10^{log_result:.1f}"
        
        result = base ** exponent
        self.history.append(f"{base}^{exponent} = {result}")
        return result
    
    def compare_floats(self, a, b, tolerance=1e-9):
        """Safe float comparison."""
        return math.isclose(a, b, rel_tol=tolerance)

# Test your calculator
calc = ScientificCalculator()

# Test logarithm
print(f"log₁₀(1000) = {calc.safe_log(1000, 10)}")

# Test power with large numbers
print(f"10^300 = {calc.safe_power(10, 300)}")

# Test float comparison
a = 0.1 + 0.2
b = 0.3
print(f"0.1 + 0.2 == 0.3? {calc.compare_floats(a, b)}")

# Show history
print("\nCalculation history:")
for item in calc.history:
    print(f"  {item}")
```

:::

## 2.10 Variable Star Exercise Thread

## 2.10 Variable Star Exercise Thread

Let's continue building on our variable star from Chapter 1, adding magnitude calculations:

```{code-cell} ipython3
# Chapter 2: Variable Star - Adding Magnitude Calculations
import json
import math

# Create sample data (in real use, load from Chapter 1)
star = {
    'name': 'Delta Cephei',
    'period': 5.366319,
    'mag_mean': 3.95,
    'mag_amp': 0.88
}

def calculate_phase(time, period):
    """Calculate phase (0-1) for given time."""
    phase = (time % period) / period
    return phase

def magnitude_at_phase(mean_mag, amplitude, phase):
    """
    Calculate magnitude at given phase.
    Using simplified sinusoidal variation.
    Real Cepheids have asymmetric light curves!
    """
    # Magnitude gets SMALLER (brighter) at maximum
    variation = amplitude * math.cos(2 * math.pi * phase)
    return mean_mag + variation

# Test with our star
test_time = 2.7  # days
phase = calculate_phase(test_time, star['period'])
current_mag = magnitude_at_phase(star['mag_mean'], 
                                 star['mag_amp'], 
                                 phase)

print(f"{star['name']} at time {test_time:.1f} days:")
print(f"  Phase: {phase:.3f}")
print(f"  Magnitude: {current_mag:.2f}")
print(f"  Brightness: {10**(-0.4 * current_mag):.3f} (relative flux)")

# Save enhanced data for Chapter 3
star['phase_function'] = 'sinusoidal'
star['last_calculated'] = {'time': test_time, 'phase': phase, 'magnitude': current_mag}

try:
    with open('variable_star_ch2.json', 'w') as f:
        json.dump(star, f, indent=2)
    print("\n✓ Data saved for Chapter 3!")
except IOError as e:
    print(f"\n✗ Could not save: {e}")
```

## ⚠️ Chapter 3: Control Flow & Logic

```

### Practice Exercises

Now apply your control flow mastery to real astronomical problems!

#### Exercise 3.1: Phase Dispersion Minimization

:::{admonition} Complete Implementation (40-50 lines)
:class: exercise

```python
def find_period_pdm(times, magnitudes, min_period=0.1, max_period=10.0):
    """
    Find the period of a variable star using Phase Dispersion Minimization.
    This is a REAL algorithm used in astronomy!
    
    Your implementation should:
    1. Use nested loops for coarse then fine search
    2. Apply the convergence pattern from Section 3.6
    3. Include guard clauses for invalid input
    4. Use list comprehensions where appropriate
    
    Pseudocode to get started:
    - Validate inputs with guard clauses
    - Coarse search with 0.1 day steps
    - Find minimum dispersion period
    - Refine with 0.01 day steps around minimum
    - Continue until convergence
    """
    # Your implementation here
    pass
```
:::

#### Exercise 3.2: Transient Detection Pipeline

:::{admonition} Multi-Part Exercise (30-40 lines total)
:class: exercise

Part A: Implement data cleaning
Part B: Detect variability using Welford's algorithm
Part C: Classify transients with elif chains

```python
def process_survey_data(times, mags, errors):
    """
    Complete pipeline for transient detection.
    Uses all control flow patterns from this chapter!
    """
    # Part A: Clean data (guard clauses, list comprehension)
    # Part B: Find variables (Welford's algorithm)
    # Part C: Classify (elif chains)
    pass
```
:::

#### Exercise 3.3: Debug the Light Curve Folder

:::{admonition} Find and Fix Three Bugs
:class: exercise

```python
def fold_light_curve(times, mags, period):
    """This function has 3 bugs - find and fix them!"""
    phases = []
    folded_mags = []
    
    for i in range(len(times)):
        phase = times[i] / period  # Bug 1: Should use modulo!
        phases.append(phase)
        folded_mags.append(mags[i])
    
    # Sort by phase
    for i in range(len(phases)):
        for j in range(len(phases)):  # Bug 2: j should start at i+1
            if phases[i] > phases[j]:  # Bug 3: Wrong comparison
                phases[i], phases[j] = phases[j], phases[i]
                folded_mags[i], folded_mags[j] = folded_mags[j], folded_mags[i]
    
    return phases, folded_mags
```
:::

### The Variable Star Thread Continues

Let's apply our control flow knowledge to extend our variable star analysis from Chapters 1 and 2:

```{code-cell} ipython3
# Chapter 3: Variable Star - Adding Periodicity Detection
import json
import math

# Create sample data (in real use, load from Chapter 2)
star = {
    'name': 'Delta Cephei',
    'period': 5.366319,
    'mag_mean': 3.95,
    'mag_amp': 0.88,
    'phase_function': 'sinusoidal'
}

def analyze_phase_coverage(times, period, min_coverage=0.6):
    """
    Check if observations adequately sample the phase space.
    Critical for period determination accuracy!
    """
    # Guard clause
    if not times or period <= 0:
        return False, "Invalid input data"
    
    # Calculate phases
    phases = [(t % period) / period for t in times]
    
    # Divide phase space into bins
    n_bins = 10
    bins_filled = set()
    
    for phase in phases:
        bin_index = int(phase * n_bins)
        bins_filled.add(bin_index)
    
    coverage = len(bins_filled) / n_bins
    
    # Conditional logic for assessment
    if coverage >= min_coverage:
        quality = "good" if coverage > 0.8 else "adequate"
        return True, f"Phase coverage {quality}: {coverage:.1%}"
    else:
        return False, f"Insufficient coverage: {coverage:.1%} < {min_coverage:.1%}"

# Test with simulated observations
test_times = [0.5, 1.2, 2.7, 3.1, 4.8, 5.9, 7.2, 8.5, 9.1, 10.3]
adequate, message = analyze_phase_coverage(test_times, star['period'])
print(f"Delta Cephei observations: {message}")

# Save enhanced data for Chapter 4
star['last_analysis'] = {
    'phase_coverage': adequate,
    'message': message,
    'n_observations': len(test_times)
}

try:
    with open('variable_star_ch3.json', 'w') as f:
        json.dump(star, f, indent=2)
    print("✓ Data saved for Chapter 4!")
except IOError as e:
    print(f"✗ Could not save: {e}")
```

## ⚠️ Chapter 4: Data Structures - Organizing Scientific Data

### Practice Exercises

#### Exercise 4.1: Particle System Organization

**Part A: Basic List Implementation (5-10 lines)**

Follow these exact steps to create a working particle system:

```{code-cell} ipython3
def create_particle_system_list():
    """Store particles as list of lists - simple but limited."""
    particles = []
    
    # Create 5 test particles (id, mass, x, y, z)
    for i in range(5):
        particle = [i, 1.0e30 * (i+1), i*1e13, 0.0, 0.0]
        particles.append(particle)
    
    print(f"Created {len(particles)} particles")
    print(f"Particle 0: ID={particles[0][0]}, mass={particles[0][1]:.1e} g")
    return particles

particles = create_particle_system_list()
```

**Part B: Convert to Dictionary (10-15 lines)**

Improve the design with dictionaries for O(1) lookup:

```{code-cell} ipython3
def create_particle_system_dict():
    """Store particles in dictionary - better for lookups."""
    particles = {}
    
    # Create particles with meaningful structure
    for i in range(5):
        particles[f'p{i:03d}'] = {
            'mass': 1.0e30 * (i+1),  # grams
            'position': [i*1e13, 0.0, 0.0],  # cm
            'velocity': [0.0, 2e6, 0.0],  # cm/s
        }
    
    # O(1) lookup by ID!
    target = 'p002'
    print(f"Particle {target}: mass = {particles[target]['mass']:.1e} g")
    print(f"  Position: {particles[target]['position'][0]:.1e} cm")
    
    return particles

particles = create_particle_system_dict()
print(f"System has {len(particles)} particles with O(1) access")
```

**Part C: Production Version with Validation (15-20 lines)**

Add error checking and performance measurement:

```{code-cell} ipython3
import time

def create_particle_system_professional(n=1000):
    """Production-ready particle system with validation."""
    start = time.perf_counter()
    
    particles = {}
    errors = []
    
    for i in range(n):
        # Validate mass (must be positive)
        mass = 1.0e30 * (1 + i*0.001)
        if mass <= 0:
            errors.append(f"Particle {i}: invalid mass {mass}")
            continue
            
        particles[f'p{i:06d}'] = {
            'mass': mass,
            'position': [i*1e11, 0.0, 0.0],
            'velocity': [0.0, 3e6, 0.0],
            'active': True
        }
    
    elapsed = time.perf_counter() - start
    
    print(f"Created {len(particles)} particles in {elapsed*1000:.1f} ms")
    if errors:
        print(f"Skipped {len(errors)} invalid particles")
    
    # Verify O(1) access
    test_time = time.perf_counter()
    _ = particles['p000500']['mass']
    access_time = time.perf_counter() - test_time
    print(f"Single particle access: {access_time*1e6:.2f} μs")
    
    return particles

system = create_particle_system_professional()
```

#### Exercise 4.2: Collision Detection System

**Part A: Basic Neighbor Finding (10 lines)**

```{code-cell} ipython3
def find_neighbors_naive(positions, radius):
    """Find particle pairs within radius - O(n²) approach."""
    neighbors = []
    n = len(positions)
    
    for i in range(n):
        for j in range(i+1, n):  # Avoid double counting
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            r = (dx**2 + dy**2) ** 0.5
            if r < radius:
                neighbors.append((i, j))
    
    return neighbors

# Test with small system
pos = [[0, 0], [1, 0], [0, 1], [2, 2], [3, 3]]
pairs = find_neighbors_naive(pos, 1.5)
print(f"Found {len(pairs)} neighbor pairs: {pairs}")
```

**Part B: Use Sets for Efficiency (15 lines)**

```{code-cell} ipython3
def find_neighbors_with_sets(particles, radius):
    """Track unique collision pairs with sets."""
    
    # Use set to avoid duplicate pairs
    collision_pairs = set()
    checked_pairs = set()
    
    positions = [(i, p['position']) for i, p in particles.items()]
    
    for i, (id1, pos1) in enumerate(positions):
        for j, (id2, pos2) in enumerate(positions[i+1:], i+1):
            pair = tuple(sorted([id1, id2]))  # Canonical ordering
            
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                
                # Check distance
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                r = (dx**2 + dy**2) ** 0.5
                
                if r < radius:
                    collision_pairs.add(pair)
    
    print(f"Checked {len(checked_pairs)} unique pairs")
    print(f"Found {len(collision_pairs)} collision candidates")
    return collision_pairs

# Test
test_particles = {
    'p1': {'position': [0, 0, 0]},
    'p2': {'position': [1e10, 0, 0]},
    'p3': {'position': [0, 1e10, 0]}
}
collisions = find_neighbors_with_sets(test_particles, 1.5e10)
```

**Part C: Spatial Hashing for O(n) (25 lines)**

:::{admonition} 💡 Hint
:class: tip
Divide space into a grid where each cell is twice the collision radius, then only check particles in adjacent cells. This transforms O(n²) to O(n) by limiting comparisons to nearby particles.
:::

```{code-cell} ipython3
def find_neighbors_spatial_hash(particles, radius, cell_size=None):
    """Use spatial hashing for O(n) neighbor finding."""
    if cell_size is None:
        cell_size = radius * 2
    
    # Build spatial hash
    cells = {}
    for pid, data in particles.items():
        x, y = data['position'][:2]
        cell_key = (int(x / cell_size), int(y / cell_size))
        
        if cell_key not in cells:
            cells[cell_key] = set()
        cells[cell_key].add(pid)
    
    # Find neighbors (only check adjacent cells)
    neighbors = set()
    for cell_key, pids in cells.items():
        cx, cy = cell_key
        
        # Check this cell and 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cx + dx, cy + dy)
                if neighbor_cell in cells:
                    for p1 in pids:
                        for p2 in cells[neighbor_cell]:
                            if p1 < p2:  # Avoid duplicates
                                # Check actual distance
                                pos1 = particles[p1]['position']
                                pos2 = particles[p2]['position']
                                r = ((pos1[0]-pos2[0])**2 + 
                                     (pos1[1]-pos2[1])**2) ** 0.5
                                if r < radius:
                                    neighbors.add((p1, p2))
    
    print(f"Spatial hash: {len(cells)} cells, {len(neighbors)} pairs")
    print("This scales as O(n) instead of O(n²)!")
    return neighbors
```

#### Exercise 4.3: Equation of State Cache

**Part A: Basic EOS Function (10 lines)**

```{code-cell} ipython3
class SimpleEOS:
    """Basic equation of state without caching."""
    
    def __init__(self):
        self.calculations = 0
    
    def pressure(self, temperature, density):
        """Calculate pressure (ideal gas for simplicity)."""
        self.calculations += 1
        k_B = 1.381e-16  # erg/K
        m_H = 1.673e-24  # grams (hydrogen mass)
        
        # P = ρkT/m for ideal gas
        P = density * k_B * temperature / m_H
        return P

eos = SimpleEOS()
P1 = eos.pressure(1e6, 1e-3)
P2 = eos.pressure(1e6, 1e-3)  # Same calculation repeated!
print(f"Pressure: {P1:.2e} dyne/cm²")
print(f"Performed {eos.calculations} calculations (wasteful!)")
```

**Part B: Add Dictionary Cache (15 lines)**

```{code-cell} ipython3
class CachedEOS:
    """EOS with dictionary caching."""
    
    def __init__(self):
        self.cache = {}
        self.calculations = 0
        self.cache_hits = 0
    
    def pressure(self, temperature, density):
        """Calculate pressure with caching."""
        key = (temperature, density)
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        # Calculate only if not cached
        self.calculations += 1
        k_B = 1.381e-16
        m_H = 1.673e-24
        P = density * k_B * temperature / m_H
        
        self.cache[key] = P
        return P

eos = CachedEOS()
# Simulate multiple calls with same parameters
for _ in range(5):
    P = eos.pressure(1e6, 1e-3)

print(f"Calculations: {eos.calculations}, Cache hits: {eos.cache_hits}")
print(f"Saved {eos.cache_hits} expensive calculations!")
```

**Part C: Advanced Cache with Memory Limit (25 lines)**

```{code-cell} ipython3
from collections import OrderedDict

class ProductionEOS:
    """Production-ready EOS with LRU cache and statistics."""
    
    def __init__(self, cache_size=1000):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.calculations = 0
        self.cache_hits = 0
        self.evictions = 0
    
    def pressure(self, T, rho):
        """Get pressure with automatic caching."""
        key = (round(T, 2), round(rho, 6))  # Round for cache efficiency
        
        if key in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key]
        
        # Calculate
        self.calculations += 1
        k_B = 1.381e-16
        m_H = 1.673e-24
        
        # More realistic EOS (includes radiation pressure)
        a = 7.566e-15  # Radiation constant
        P_gas = rho * k_B * T / m_H
        P_rad = a * T**4 / 3
        P_total = P_gas + P_rad
        
        # Add to cache
        self.cache[key] = P_total
        
        # Evict oldest if needed
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            self.evictions += 1
        
        return P_total
    
    def stats(self):
        """Report cache performance."""
        total = self.calculations + self.cache_hits
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            print(f"Cache statistics:")
            print(f"  Size: {len(self.cache)}/{self.cache_size}")
            print(f"  Calculations: {self.calculations}")
            print(f"  Cache hits: {self.cache_hits}")
            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Evictions: {self.evictions}")

# Test with stellar interior conditions
eos = ProductionEOS(cache_size=100)

# Simulate stellar model with repeated conditions
test_conditions = [
    (1e7, 100), (2e7, 150), (1e7, 100),  # Repeated
    (3e7, 200), (1e7, 100), (2e7, 150),  # More repeats
]

for T, rho in test_conditions:
    P = eos.pressure(T, rho)
    print(f"T={T:.0e} K, ρ={rho} g/cm³ → P={P:.2e} dyne/cm²")

eos.stats()
print("\nIn astronomy: Essential for stellar evolution codes!")
```
#### NOTE: No Variable Star Trend Problem was included.

## ⚠️ Chapter 5: Functions & Modules - Building Reusable Scientific Code

## ⚠️ Chapter 6: OOP Fundamentals - Organizing Scientific Code
