# Chapter 2: Python Fundamentals Review

## Learning Objectives
By the end of this chapter, you will:
- Review Python's core data types and control structures
- Master NumPy array operations for scientific computing
- Understand Python's memory model and variable scoping
- Apply these concepts to astronomical calculations

## 2.1 Core Python Data Types

### Numbers and Astronomical Quantities

```python
# Python's numeric types
integer_pixels = 4096  # CCD dimension
float_redshift = 0.234  # Galaxy redshift
complex_visibility = 3.2 + 1.5j  # Radio interferometry

# Scientific notation
parsec = 3.086e16  # meters
solar_mass = 1.989e30  # kg
hubble_constant = 70.0  # km/s/Mpc

# Type checking and conversion
print(f"Type of redshift: {type(float_redshift)}")
print(f"Integer division: {integer_pixels // 100} = {integer_pixels // 100}")
print(f"Float division: {integer_pixels / 100} = {integer_pixels / 100}")
```

### Strings for Data Management

```python
# String formatting for astronomical objects
object_name = "NGC 1234"
ra = 123.456  # degrees
dec = -45.678  # degrees

# Modern f-strings (preferred)
catalog_entry = f"{object_name}: RA={ra:.3f}°, Dec={dec:.3f}°"
print(catalog_entry)

# String methods useful for parsing catalogs
fits_header = "OBJECT  = 'Vega    '           / Target name"
key, value = fits_header.split("=")
object_parsed = value.split("/")[0].strip().strip("'")
print(f"Parsed object: '{object_parsed}'")
```

### Collections: Lists, Tuples, Dictionaries

```python
# Lists: mutable, ordered
magnitudes = [12.3, 11.8, 13.2, 10.9, 14.5]
magnitudes.append(11.2)  # Add observation
magnitudes.sort()  # Sort in place
print(f"Brightest: {magnitudes[0]}, Faintest: {magnitudes[-1]}")

# Tuples: immutable, ordered (good for coordinates)
galactic_center = (17.761, -29.008)  # RA, Dec in degrees
# galactic_center[0] = 18.0  # This would raise an error!

# Dictionaries: key-value pairs for metadata
observation = {
    'target': 'M31',
    'filter': 'V',
    'exposure': 300.0,  # seconds
    'airmass': 1.2,
    'seeing': 0.8  # arcseconds
}
print(f"Observed {observation['target']} for {observation['exposure']}s")

# Sets: unique elements (useful for object matching)
catalog1 = {'NGC1234', 'NGC5678', 'M31', 'M42'}
catalog2 = {'M31', 'M42', 'NGC9999', 'NGC1234'}
common_objects = catalog1 & catalog2  # Intersection
print(f"Objects in both catalogs: {common_objects}")
```

## 2.2 Control Flow

### Conditional Logic

```python
def classify_star(temperature):
    """Classify star based on effective temperature."""
    if temperature > 30000:
        spectral_type = 'O'
    elif temperature > 10000:
        spectral_type = 'B'
    elif temperature > 7500:
        spectral_type = 'A'
    elif temperature > 6000:
        spectral_type = 'F'
    elif temperature > 5200:
        spectral_type = 'G'
    elif temperature > 3700:
        spectral_type = 'K'
    else:
        spectral_type = 'M'
    
    return spectral_type

# Test the classifier
sun_temp = 5778
print(f"Sun (T={sun_temp}K) is type {classify_star(sun_temp)}")
```

### Loops and Iteration

```python
# For loops with enumerate for indexing
filters = ['U', 'B', 'V', 'R', 'I']
wavelengths = [365, 445, 551, 658, 806]  # nm

for i, (filt, wave) in enumerate(zip(filters, wavelengths)):
    print(f"Filter {i}: {filt} at {wave} nm")

# While loops for convergence
def find_magnitude_limit(sky_brightness, target_snr=5):
    """Find limiting magnitude for given sky brightness."""
    magnitude = 15.0
    snr = 100.0
    
    while snr > target_snr:
        magnitude += 0.1
        # Simplified SNR calculation
        snr = 100 * 10**(-0.4 * (magnitude - 15))
    
    return magnitude

limit = find_magnitude_limit(21.5)
print(f"5-sigma limiting magnitude: {limit:.1f}")
```

### List Comprehensions

```python
# Efficient list creation
import math

# Convert magnitudes to fluxes
mags = [10.5, 11.2, 9.8, 12.1, 10.0]
fluxes = [10**(-0.4 * m) for m in mags]

# Filter bright objects
bright_mags = [m for m in mags if m < 11.0]
print(f"Bright objects: {bright_mags}")

# Nested comprehension for 2D grid
ra_range = [120 + i*0.1 for i in range(5)]
dec_range = [30 + j*0.1 for j in range(5)]
coordinates = [(ra, dec) for ra in ra_range for dec in dec_range]
print(f"Generated {len(coordinates)} sky positions")
```

## 2.3 NumPy Fundamentals

### Array Creation and Properties

```python
import numpy as np

# Creating arrays
zeros_image = np.zeros((512, 512))  # Blank CCD frame
ones_flat = np.ones((512, 512))  # Flat field
noise = np.random.randn(512, 512)  # Gaussian noise

# Array from lists
wavelengths = np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000])
fluxes = np.array([0.5, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3])

# Array properties
print(f"Shape: {wavelengths.shape}")
print(f"Dtype: {wavelengths.dtype}")
print(f"Size: {wavelengths.size}")
print(f"Memory: {wavelengths.nbytes} bytes")

# Arange and linspace for regular grids
time = np.arange(0, 10, 0.1)  # 0 to 10 in steps of 0.1
phase = np.linspace(0, 2*np.pi, 100)  # 100 points from 0 to 2π
```

### Array Operations

```python
# Vectorized operations (no loops needed!)
mags = np.array([10.5, 11.2, 9.8, 12.1, 10.0])
fluxes = 10**(-0.4 * mags)  # Operates on entire array

# Statistical operations
print(f"Mean magnitude: {mags.mean():.2f}")
print(f"Std deviation: {mags.std():.2f}")
print(f"Brightest: {mags.min():.2f}")
print(f"Faintest: {mags.max():.2f}")

# Boolean indexing
bright_mask = mags < 11.0
bright_mags = mags[bright_mask]
print(f"Bright objects: {bright_mags}")

# Fancy indexing
indices = [0, 2, 4]
selected_mags = mags[indices]
print(f"Selected magnitudes: {selected_mags}")
```

### Broadcasting

```python
# Broadcasting allows operations on different shaped arrays
image = np.random.randn(100, 100)  # 2D image
bias = np.mean(image, axis=0)  # 1D array of column means

# Subtract bias from each column (broadcasting)
corrected = image - bias  # bias is broadcast to match image shape

# Example: Distance matrix
n_stars = 5
x = np.random.uniform(0, 10, n_stars)
y = np.random.uniform(0, 10, n_stars)

# Compute all pairwise distances using broadcasting
dx = x[:, np.newaxis] - x[np.newaxis, :]  # Shape: (n_stars, n_stars)
dy = y[:, np.newaxis] - y[np.newaxis, :]
distances = np.sqrt(dx**2 + dy**2)

print(f"Distance matrix shape: {distances.shape}")
print(f"Distance from star 0 to star 1: {distances[0, 1]:.2f}")
```

## 2.4 Memory and Performance Considerations

### Views vs Copies

```python
# Views share memory (efficient but be careful!)
data = np.arange(10)
view = data[2:8]
view[0] = 999
print(f"Original data modified: {data}")  # data[2] is now 999!

# Copies are independent
data = np.arange(10)
copy = data[2:8].copy()
copy[0] = 999
print(f"Original data unchanged: {data}")

# Slicing creates views, fancy indexing creates copies
subset_view = data[::2]  # Every other element (view)
subset_copy = data[[0, 2, 4, 6, 8]]  # Same elements (copy)
```

### Efficient Array Operations

```python
# Pre-allocate arrays when possible
n_iterations = 1000
results = np.zeros(n_iterations)  # Pre-allocate

for i in range(n_iterations):
    results[i] = np.random.randn()  # Fill in place

# Use NumPy functions instead of Python loops
# Bad: Python loop
def magnitude_to_flux_slow(mags):
    fluxes = []
    for m in mags:
        fluxes.append(10**(-0.4 * m))
    return np.array(fluxes)

# Good: Vectorized NumPy
def magnitude_to_flux_fast(mags):
    return 10**(-0.4 * mags)

# Timing comparison
mags = np.random.uniform(10, 15, 10000)
%timeit magnitude_to_flux_slow(mags)
%timeit magnitude_to_flux_fast(mags)
```

## Try It Yourself

### Exercise 2.1: Stellar Magnitude System
Implement functions to convert between apparent magnitude, absolute magnitude, and distance.

```python
def apparent_to_absolute(m, d):
    """
    Convert apparent magnitude to absolute magnitude.
    
    Parameters
    ----------
    m : float or array
        Apparent magnitude
    d : float or array
        Distance in parsecs
    
    Returns
    -------
    float or array
        Absolute magnitude
    """
    # Your code here
    # Hint: M = m - 5*log10(d) + 5
    pass

def absolute_to_flux(M):
    """
    Convert absolute magnitude to luminosity relative to the Sun.
    
    Solar absolute magnitude = 4.83
    """
    # Your code here
    pass

# Test with Sirius
sirius_m = -1.46  # Apparent magnitude
sirius_d = 2.64  # parsecs
sirius_M = apparent_to_absolute(sirius_m, sirius_d)
print(f"Sirius absolute magnitude: {sirius_M:.2f}")
print(f"Sirius luminosity: {absolute_to_flux(sirius_M):.1f} L_sun")
```

### Exercise 2.2: Color-Magnitude Diagram
Create a synthetic color-magnitude diagram for a stellar cluster.

```python
def generate_cluster_cmd(n_stars=1000, age=1e9, metallicity=0.02):
    """
    Generate synthetic color-magnitude diagram.
    
    Parameters
    ----------
    n_stars : int
        Number of stars to generate
    age : float
        Cluster age in years
    metallicity : float
        Metallicity (solar = 0.02)
    
    Returns
    -------
    tuple
        (B-V colors, V magnitudes)
    """
    # Your code here
    # 1. Sample stellar masses from IMF
    # 2. Apply mass-luminosity relation
    # 3. Add photometric scatter
    pass

# Generate and plot CMD
colors, mags = generate_cluster_cmd()
plt.scatter(colors, mags, alpha=0.5, s=1)
plt.xlabel('B-V')
plt.ylabel('V magnitude')
plt.gca().invert_yaxis()
plt.title('Synthetic Cluster CMD')
plt.show()
```

### Exercise 2.3: Astronomical Catalog Cross-Match
Match objects between two catalogs based on position.

```python
def cross_match_catalogs(ra1, dec1, ra2, dec2, max_sep=1.0):
    """
    Cross-match two catalogs by position.
    
    Parameters
    ----------
    ra1, dec1 : array
        Coordinates of catalog 1 (degrees)
    ra2, dec2 : array
        Coordinates of catalog 2 (degrees)
    max_sep : float
        Maximum separation for match (arcseconds)
    
    Returns
    -------
    array
        Indices of matches in catalog 2 for each object in catalog 1
        (-1 for no match)
    """
    # Your code here
    # Hint: Use broadcasting to compute all pairwise distances
    # Remember to handle spherical geometry properly!
    pass

# Test with synthetic catalogs
n1, n2 = 100, 150
ra1 = np.random.uniform(0, 10, n1)
dec1 = np.random.uniform(-5, 5, n1)
ra2 = np.random.uniform(0, 10, n2)
dec2 = np.random.uniform(-5, 5, n2)

matches = cross_match_catalogs(ra1, dec1, ra2, dec2)
print(f"Found {np.sum(matches >= 0)} matches out of {n1} objects")
```

## Key Takeaways

✅ **Python's data types** map naturally to astronomical quantities  
✅ **NumPy arrays** enable efficient computation without explicit loops  
✅ **Vectorization** is key to performance in scientific computing  
✅ **Broadcasting** eliminates the need for nested loops in many cases  

## Next Chapter Preview
We'll build on these fundamentals to create modular, reusable functions that form the building blocks of larger astronomical software systems.