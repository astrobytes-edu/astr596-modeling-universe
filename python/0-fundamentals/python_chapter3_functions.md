# Chapter 3: Functions and Modular Programming

## Learning Objectives
By the end of this chapter, you will:
- Design functions with clear interfaces and documentation
- Implement modular code architecture for scientific computing
- Master Python's advanced function features (decorators, generators, lambdas)
- Build reusable astronomical computation modules

## 3.1 Function Design Principles

### The Anatomy of a Good Function

```python
def planck_function(wavelength, temperature):
    """
    Calculate Planck blackbody spectral radiance.
    
    Parameters
    ----------
    wavelength : float or np.ndarray
        Wavelength in meters
    temperature : float
        Temperature in Kelvin
    
    Returns
    -------
    float or np.ndarray
        Spectral radiance in W/m^2/m/sr
    
    Notes
    -----
    Uses the Planck function:
    B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    
    Examples
    --------
    >>> import numpy as np
    >>> wavelength = 500e-9  # 500 nm
    >>> B = planck_function(wavelength, 5778)  # Solar temperature
    >>> print(f"Solar radiance at 500nm: {B:.2e} W/m^2/m/sr")
    """
    import numpy as np
    from scipy import constants
    
    h = constants.h  # Planck constant
    c = constants.c  # Speed of light
    k = constants.k  # Boltzmann constant
    
    # Avoid overflow in exponential
    x = (h * c) / (wavelength * k * temperature)
    
    # Handle the case where x is very large
    with np.errstate(over='ignore'):
        exp_term = np.exp(np.clip(x, None, 700)) - 1
    
    radiance = (2 * h * c**2 / wavelength**5) / exp_term
    
    return radiance

# Test the function
wavelengths = np.linspace(100e-9, 3000e-9, 1000)
B_sun = planck_function(wavelengths, 5778)
B_cool = planck_function(wavelengths, 3000)
```

### Function Arguments: Positional, Keyword, and Defaults

```python
def observe_target(target, 
                  exposure_time=300.0,
                  filter_name='V',
                  n_exposures=1,
                  *args,
                  dither=False,
                  **kwargs):
    """
    Plan observation of astronomical target.
    
    Parameters
    ----------
    target : str
        Target name (required positional)
    exposure_time : float
        Exposure time in seconds (default: 300)
    filter_name : str
        Filter to use (default: 'V')
    n_exposures : int
        Number of exposures (default: 1)
    *args : tuple
        Additional positional arguments
    dither : bool
        Whether to dither between exposures (keyword-only)
    **kwargs : dict
        Additional keyword arguments
    """
    total_time = exposure_time * n_exposures
    
    print(f"Observing {target}")
    print(f"Filter: {filter_name}")
    print(f"Total integration: {total_time}s")
    
    if dither:
        print("Dithering enabled")
    
    if args:
        print(f"Additional args: {args}")
    
    if kwargs:
        print(f"Additional kwargs: {kwargs}")
    
    return total_time

# Various ways to call the function
observe_target('M31')  # Positional only
observe_target('M31', 600)  # Position with one default override
observe_target('M31', filter_name='R', dither=True)  # Mix of styles
observe_target('M31', 300, 'B', 3, 'extra', dither=True, priority='high')
```

## 3.2 Modular Code Organization

### Building a Module: Coordinate Transformations

```python
# File: coordinates.py
"""
Astronomical coordinate transformation utilities.

This module provides functions for converting between different
astronomical coordinate systems.
"""

import numpy as np

def ra_dec_to_cartesian(ra, dec, r=1.0):
    """Convert spherical to Cartesian coordinates."""
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    
    return x, y, z

def cartesian_to_ra_dec(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    ra_rad = np.arctan2(y, x)
    dec_rad = np.arcsin(z / r)
    
    ra = np.degrees(ra_rad) % 360
    dec = np.degrees(dec_rad)
    
    return ra, dec, r

def angular_separation(ra1, dec1, ra2, dec2):
    """
    Calculate angular separation using Vincenty formula.
    More accurate than haversine for small angles.
    """
    # Convert to radians
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    
    dra = ra2 - ra1
    
    # Vincenty formula
    num1 = np.cos(dec2) * np.sin(dra)
    num2 = np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra)
    numerator = np.sqrt(num1**2 + num2**2)
    
    denominator = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    
    sep = np.arctan2(numerator, denominator)
    
    return np.degrees(sep)

# Using the module
if __name__ == "__main__":
    # Test conversions
    ra, dec = 123.45, 67.89
    x, y, z = ra_dec_to_cartesian(ra, dec)
    ra2, dec2, r = cartesian_to_ra_dec(x, y, z)
    
    print(f"Original: RA={ra:.2f}, Dec={dec:.2f}")
    print(f"Recovered: RA={ra2:.2f}, Dec={dec2:.2f}")
    
    # Test separation
    sep = angular_separation(0, 0, 1, 0)
    print(f"Separation: {sep:.2f} degrees")
```

### Organizing Related Functions into Submodules

```python
# File structure:
# astro_tools/
#   __init__.py
#   photometry/
#     __init__.py
#     magnitudes.py
#     filters.py
#   spectroscopy/
#     __init__.py
#     line_fitting.py
#     redshift.py

# File: astro_tools/photometry/magnitudes.py
"""Magnitude system calculations."""

import numpy as np

ZERO_POINT_FLUX = 3631  # Jy (AB magnitude system)
SOLAR_ABS_MAG = {'U': 5.61, 'B': 5.48, 'V': 4.83, 'R': 4.42, 'I': 4.08}

def flux_to_magnitude(flux, zero_point=ZERO_POINT_FLUX):
    """Convert flux to AB magnitude."""
    return -2.5 * np.log10(flux / zero_point)

def magnitude_to_flux(magnitude, zero_point=ZERO_POINT_FLUX):
    """Convert AB magnitude to flux."""
    return zero_point * 10**(-0.4 * magnitude)

def color_index(mag1, mag2):
    """Calculate color index (e.g., B-V)."""
    return mag1 - mag2

def absolute_magnitude(apparent_mag, distance_pc):
    """Calculate absolute magnitude from apparent magnitude."""
    return apparent_mag - 5 * np.log10(distance_pc) + 5

def distance_modulus(distance_pc):
    """Calculate distance modulus."""
    return 5 * np.log10(distance_pc) - 5

# File: astro_tools/__init__.py
"""Main package initialization."""

from .photometry.magnitudes import (
    flux_to_magnitude,
    magnitude_to_flux,
    absolute_magnitude
)

from .spectroscopy.redshift import (
    doppler_shift,
    cosmological_redshift
)

__all__ = [
    'flux_to_magnitude',
    'magnitude_to_flux', 
    'absolute_magnitude',
    'doppler_shift',
    'cosmological_redshift'
]

__version__ = '0.1.0'
```

## 3.3 Advanced Function Features

### Decorators for Timing and Caching

```python
import time
import functools

def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

@timer
def compute_luminosity_function(magnitudes, volume):
    """Compute galaxy luminosity function (slow operation)."""
    time.sleep(0.1)  # Simulate computation
    bins = np.arange(magnitudes.min(), magnitudes.max(), 0.5)
    counts, _ = np.histogram(magnitudes, bins)
    phi = counts / volume / 0.5  # Number density per magnitude
    return bins[:-1], phi

# Memoization for expensive calculations
@functools.lru_cache(maxsize=128)
def expensive_cosmology_calculation(z, H0=70, Om0=0.3):
    """
    Calculate luminosity distance (cached).
    Repeated calls with same arguments return cached result.
    """
    # Simulate expensive integral
    time.sleep(0.5)
    # Simplified calculation
    return (c/H0) * z * (1 + z/2 * (1 - Om0))

# Test caching
%time d1 = expensive_cosmology_calculation(0.5)  # Slow (first call)
%time d2 = expensive_cosmology_calculation(0.5)  # Fast (cached)
```

### Generators for Memory-Efficient Data Processing

```python
def read_large_catalog(filename, chunk_size=1000):
    """
    Generator to read large catalog in chunks.
    Yields chunks of data without loading entire file.
    """
    with open(filename, 'r') as f:
        header = f.readline()
        
        chunk = []
        for line in f:
            chunk.append(line.strip().split(','))
            
            if len(chunk) >= chunk_size:
                yield np.array(chunk, dtype=float)
                chunk = []
        
        # Yield remaining data
        if chunk:
            yield np.array(chunk, dtype=float)

# Process large catalog without loading it all
def process_catalog_efficiently(filename):
    """Process catalog in memory-efficient chunks."""
    total_bright = 0
    total_objects = 0
    
    for chunk in read_large_catalog(filename):
        magnitudes = chunk[:, 2]  # Assume magnitude in column 2
        total_bright += np.sum(magnitudes < 15.0)
        total_objects += len(magnitudes)
    
    fraction = total_bright / total_objects
    print(f"Fraction brighter than 15.0: {fraction:.3f}")
    
    return fraction

# Generator expression for filtering
def bright_objects(catalog):
    """Generator expression to filter bright objects."""
    return (obj for obj in catalog if obj['mag'] < 12.0)

# Chain generators for pipeline processing
def pipeline_example(filename):
    """Example of chaining generators."""
    raw_data = read_large_catalog(filename)
    filtered = (chunk[chunk[:, 2] < 15] for chunk in raw_data)
    processed = (calculate_colors(chunk) for chunk in filtered)
    
    for result in processed:
        # Process each chunk
        pass
```

### Lambda Functions and Functional Programming

```python
# Lambda functions for simple operations
magnitude_to_flux = lambda m: 10**(-0.4 * m)
flux_ratio = lambda m1, m2: 10**(-0.4 * (m1 - m2))

# Using map, filter, reduce
from functools import reduce

magnitudes = [10.5, 11.2, 9.8, 12.1, 10.0, 15.3, 8.9]

# Map: apply function to all elements
fluxes = list(map(magnitude_to_flux, magnitudes))

# Filter: select elements meeting condition
bright = list(filter(lambda m: m < 11.0, magnitudes))

# Reduce: combine elements
total_flux = reduce(lambda a, b: a + b, fluxes)

# More complex functional approach
def create_magnitude_filter(limit):
    """Create a filter function with specific magnitude limit."""
    return lambda obj: obj['mag'] < limit

# Create specialized filters
bright_filter = create_magnitude_filter(12.0)
very_bright_filter = create_magnitude_filter(10.0)

catalog = [
    {'name': 'Star1', 'mag': 9.5},
    {'name': 'Star2', 'mag': 11.2},
    {'name': 'Star3', 'mag': 13.1}
]

bright_stars = list(filter(lambda obj: bright_filter(obj), catalog))
```

## 3.4 Error Handling and Validation

### Input Validation and Error Messages

```python
def calculate_schwarzschild_radius(mass, units='kg'):
    """
    Calculate Schwarzschild radius with input validation.
    
    Parameters
    ----------
    mass : float
        Mass of object
    units : str
        Units of mass ('kg', 'solar', 'earth')
    
    Raises
    ------
    ValueError
        If mass is negative or units unknown
    TypeError
        If mass is not numeric
    """
    # Type checking
    if not isinstance(mass, (int, float, np.number)):
        raise TypeError(f"Mass must be numeric, got {type(mass)}")
    
    # Value validation
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")
    
    # Convert to kg if needed
    unit_conversions = {
        'kg': 1.0,
        'solar': 1.989e30,
        'earth': 5.972e24
    }
    
    if units not in unit_conversions:
        raise ValueError(f"Unknown units '{units}'. Use: {list(unit_conversions.keys())}")
    
    mass_kg = mass * unit_conversions[units]
    
    # Calculate radius
    from scipy import constants
    rs = 2 * constants.G * mass_kg / constants.c**2
    
    return rs

# Test error handling
try:
    r = calculate_schwarzschild_radius(-10)
except ValueError as e:
    print(f"Caught error: {e}")

try:
    r = calculate_schwarzschild_radius(1, units='pounds')
except ValueError as e:
    print(f"Caught error: {e}")
```

### Context Managers for Resource Management

```python
class ObservationLog:
    """Context manager for observation logging."""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.start_time = None
    
    def __enter__(self):
        self.file = open(self.filename, 'a')
        self.start_time = time.time()
        self.file.write(f"\n--- Session started at {time.ctime()} ---\n")
        return self
    
    def log(self, message):
        timestamp = time.time() - self.start_time
        self.file.write(f"[{timestamp:8.2f}s] {message}\n")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.file.write(f"--- Session ended after {duration:.2f}s ---\n")
        self.file.close()

# Using the context manager
with ObservationLog('observations.log') as log:
    log.log("Slewing to target")
    time.sleep(0.5)
    log.log("Starting exposure")
    time.sleep(1.0)
    log.log("Exposure complete")
# File automatically closed even if error occurs
```

## Try It Yourself

### Exercise 3.1: Build a Modular Photometry Pipeline
Create a modular pipeline for aperture photometry.

```python
def create_aperture(x, y, radius):
    """Create circular aperture mask."""
    # Your code here
    pass

def aperture_photometry(image, x, y, radius, background=None):
    """
    Perform aperture photometry on an image.
    
    Parameters
    ----------
    image : 2D array
        Image data
    x, y : float
        Center coordinates
    radius : float
        Aperture radius in pixels
    background : float, optional
        Background level to subtract
    
    Returns
    -------
    dict
        Photometry results including flux, error, SNR
    """
    # Your code here
    # 1. Create aperture mask
    # 2. Sum flux in aperture
    # 3. Estimate uncertainty
    # 4. Calculate SNR
    pass

def batch_photometry(image, catalog, radius=5.0):
    """
    Run photometry on multiple sources.
    
    Parameters
    ----------
    image : 2D array
        Image data
    catalog : list of dict
        Source catalog with 'x' and 'y' keys
    radius : float
        Aperture radius
    
    Returns
    -------
    list
        Photometry results for each source
    """
    # Your code here
    # Use generator for memory efficiency
    pass

# Test the pipeline
image = np.random.randn(100, 100) + 100  # Simulated image
sources = [{'x': 50, 'y': 50}, {'x': 25, 'y': 75}]
results = batch_photometry(image, sources)
```

### Exercise 3.2: Create a Decorator for Unit Conversion
Build a decorator that automatically handles unit conversions.

```python
def units_handler(input_units=None, output_units=None):
    """
    Decorator to handle unit conversions.
    
    Example
    -------
    @units_handler(input_units={'wavelength': 'nm'}, 
                   output_units='W/m^2')
    def calculate_flux(wavelength, temperature):
        # wavelength automatically converted from nm to m
        pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Your code here
            # 1. Convert input units
            # 2. Call original function
            # 3. Convert output units
            pass
        return wrapper
    return decorator

# Test the decorator
@units_handler(input_units={'distance': 'pc'}, output_units='mag')
def distance_to_modulus(distance):
    """Calculate distance modulus (expects distance in meters)."""
    # Your implementation
    pass

# Should handle unit conversion automatically
dm = distance_to_modulus(10)  # 10 parsecs
print(f"Distance modulus: {dm}")
```

### Exercise 3.3: Generator for FITS File Processing
Create a generator to efficiently process multiple FITS files.

```python
def fits_file_reader(file_pattern):
    """
    Generator to read FITS files matching pattern.
    
    Yields
    ------
    tuple
        (filename, header, data)
    """
    import glob
    from astropy.io import fits
    
    # Your code here
    # 1. Find files matching pattern
    # 2. Yield one at a time
    # 3. Properly close files
    pass

def process_fits_batch(file_pattern, processing_func):
    """
    Process multiple FITS files without loading all into memory.
    
    Parameters
    ----------
    file_pattern : str
        Glob pattern for files
    processing_func : callable
        Function to apply to each file
    
    Returns
    -------
    list
        Results from processing each file
    """
    # Your code here
    # Use the generator to process files one at a time
    pass

# Example processing function
def measure_seeing(data):
    """Measure seeing from stellar profiles."""
    # Simplified: just return random seeing
    return np.random.uniform(0.5, 2.0)

# Process all science frames
results = process_fits_batch('data/sci_*.fits', measure_seeing)
```

## Key Takeaways

✅ **Well-designed functions** have clear interfaces, documentation, and error handling  
✅ **Modular organization** makes code maintainable and reusable  
✅ **Generators** enable memory-efficient processing of large datasets  
✅ **Decorators** add functionality without modifying function code  
✅ **Functional programming** techniques lead to cleaner, more testable code  

## Next Chapter Preview
We'll extend these modular programming concepts to object-oriented design, creating classes that encapsulate both data and behavior for complex astronomical objects and systems.