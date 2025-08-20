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

## ⚠️ Chapter 3: Control Flow & Logic

## ⚠️ Chapter 4: Data Structures - Organizing Scientific Data

##  ⚠️ Chapter 5: Functions & Modules - Building Reusable Scientific Code

## ⚠️ Chapter 6: OOP Fundamentals - Organizing Scientific Code