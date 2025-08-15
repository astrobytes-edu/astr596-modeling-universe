# Chapter 4: Data Structures & Algorithms

## Learning Objectives
By the end of this chapter, you will:
- Master Python's built-in data structures (lists, tuples, sets, dictionaries)
- Understand time and space complexity (Big-O notation)
- Choose the right data structure for astronomical computations
- Implement fundamental algorithms for scientific computing
- Optimize data structures for performance

## 4.1 Lists: Your Workhorse Collection

### List Operations and Performance

```python
import time
import numpy as np

# Lists are dynamic arrays - they can grow and shrink
observations = []  # Empty list
observations.append(('2024-01-15', 'M31', 300))  # O(1) amortized
observations.append(('2024-01-16', 'M42', 600))
observations.append(('2024-01-17', 'NGC1234', 450))

print(f"Observations: {observations}")
print(f"First observation: {observations[0]}")  # O(1) access
print(f"Last observation: {observations[-1]}")   # Negative indexing!

# Slicing creates a new list (O(k) where k is slice size)
recent = observations[-2:]  # Last two observations
print(f"Recent observations: {recent}")

# Common list operations and their complexity
magnitudes = [12.3, 11.8, 13.2, 10.9, 14.5, 11.2]

# O(n) operations
print(f"Min magnitude: {min(magnitudes)}")  # Brightest object
print(f"Max magnitude: {max(magnitudes)}")  # Faintest object
print(f"Count of 11.2: {magnitudes.count(11.2)}")
print(f"Index of 10.9: {magnitudes.index(10.9)}")

# O(n log n) operation
magnitudes.sort()  # In-place sorting
print(f"Sorted magnitudes: {magnitudes}")

# Dangerous O(n) operations in loops
def inefficient_search(data, targets):
    """BAD: O(n*m) complexity."""
    results = []
    for target in targets:  # O(m)
        if target in data:  # O(n) - searches entire list!
            results.append(target)
    return results

# Better: Use a set for O(1) lookups
def efficient_search(data, targets):
    """GOOD: O(n+m) complexity."""
    data_set = set(data)  # O(n) once
    results = []
    for target in targets:  # O(m)
        if target in data_set:  # O(1) average
            results.append(target)
    return results
```

### List Comprehensions with Conditions

```python
# Processing astronomical data with comprehensions
stars = [
    {'name': 'Sirius', 'mag': -1.46, 'distance': 2.64},
    {'name': 'Canopus', 'mag': -0.74, 'distance': 95.88},
    {'name': 'Arcturus', 'mag': -0.05, 'distance': 11.26},
    {'name': 'Vega', 'mag': 0.03, 'distance': 7.68},
    {'name': 'Capella', 'mag': 0.08, 'distance': 13.16}
]

# Filter and transform in one line
nearby_bright = [
    s['name'] for s in stars 
    if s['distance'] < 10 and s['mag'] < 0.1
]
print(f"Nearby bright stars: {nearby_bright}")

# Calculate absolute magnitudes
absolute_mags = [
    s['mag'] - 5 * np.log10(s['distance']) + 5
    for s in stars
]
print(f"Absolute magnitudes: {[f'{m:.2f}' for m in absolute_mags]}")

# Nested comprehensions for 2D data
# Create a simple CCD readout pattern
ccd_readout = [[i + j for j in range(4)] for i in range(3)]
print(f"CCD readout pattern:\n{ccd_readout}")
```

## 4.2 Tuples: Immutable Sequences

### When to Use Tuples

```python
# Tuples for fixed collections (can't be modified)
# Perfect for coordinates, which shouldn't change accidentally
galactic_center = (266.417, -29.008)  # RA, Dec in degrees

# Tuples as dictionary keys (lists can't be keys!)
star_catalog = {}
star_catalog[(10.684, +41.269)] = "Andromeda Galaxy"
star_catalog[(5.242, -8.334)] = "Orion Nebula"

# Named tuples for clarity
from collections import namedtuple

CelestialObject = namedtuple('CelestialObject', ['ra', 'dec', 'magnitude', 'name'])
sirius = CelestialObject(ra=101.287, dec=-16.716, magnitude=-1.46, name='Sirius')

print(f"Sirius RA: {sirius.ra}°")  # More readable than sirius[0]
print(f"Sirius as tuple: {sirius}")

# Unpacking tuples in loops
observations = [
    ('2024-01-15', 'M31', 300, 1.2),
    ('2024-01-16', 'M42', 600, 0.8),
    ('2024-01-17', 'NGC1234', 450, 1.5)
]

for date, target, exposure, seeing in observations:
    print(f"{date}: {target} for {exposure}s (seeing: {seeing}\")")
```

## 4.3 Sets: Unique Collections

### Set Theory in Astronomy

```python
# Sets for unique elements and fast membership testing
observed_objects = {'M31', 'M42', 'M45', 'NGC1234', 'NGC5678'}
scheduled_objects = {'M31', 'M51', 'M42', 'NGC9999', 'M45'}

# Set operations
common = observed_objects & scheduled_objects  # Intersection
print(f"Objects observed AND scheduled: {common}")

missed = scheduled_objects - observed_objects  # Difference
print(f"Scheduled but not observed: {missed}")

all_objects = observed_objects | scheduled_objects  # Union
print(f"All objects mentioned: {all_objects}")

unique_to_either = observed_objects ^ scheduled_objects  # Symmetric difference
print(f"In one list but not both: {unique_to_either}")

# Fast membership testing
print(f"Was M31 observed? {'M31' in observed_objects}")  # O(1) average

# Remove duplicates from list
detections = [1234, 5678, 1234, 9012, 5678, 3456, 1234]
unique_detections = list(set(detections))
print(f"Unique source IDs: {unique_detections}")

# Finding unique spectral lines
h_lines = {'H-alpha', 'H-beta', 'H-gamma', 'H-delta'}
he_lines = {'He-I-5876', 'He-I-6678', 'He-II-4686'}
observed_lines = {'H-alpha', 'He-I-5876', 'O-III-5007', 'H-beta'}

identified = observed_lines & (h_lines | he_lines)
unidentified = observed_lines - (h_lines | he_lines)
print(f"Identified lines: {identified}")
print(f"Mystery lines: {unidentified}")
```

## 4.4 Dictionaries: Key-Value Mapping

### Efficient Data Organization

```python
# Dictionaries for fast lookups and data organization
messier_catalog = {
    'M31': {'name': 'Andromeda Galaxy', 'type': 'Galaxy', 'distance_mly': 2.5},
    'M42': {'name': 'Orion Nebula', 'type': 'Nebula', 'distance_ly': 1344},
    'M45': {'name': 'Pleiades', 'type': 'Open Cluster', 'distance_ly': 444}
}

# Access nested data
print(f"M31 is the {messier_catalog['M31']['name']}")

# Safe access with get()
m99_data = messier_catalog.get('M99', {'name': 'Unknown', 'type': 'Unknown'})
print(f"M99 data: {m99_data}")

# Building a frequency table
photon_energies = [2.1, 3.5, 2.1, 4.2, 3.5, 2.1, 5.0, 3.5, 2.1]
energy_counts = {}
for energy in photon_energies:
    energy_counts[energy] = energy_counts.get(energy, 0) + 1

print(f"Energy distribution: {energy_counts}")

# Or use Counter for this pattern
from collections import Counter
energy_counts = Counter(photon_energies)
print(f"Most common energy: {energy_counts.most_common(1)}")

# Dictionary comprehension
stellar_data = {
    'Sirius': -1.46,
    'Canopus': -0.74,
    'Arcturus': -0.05,
    'Vega': 0.03
}

# Convert magnitudes to fluxes
fluxes = {
    star: 10**(-0.4 * mag) 
    for star, mag in stellar_data.items()
}
print(f"Relative fluxes: {fluxes}")
```

### Advanced Dictionary Patterns

```python
# Using defaultdict for cleaner code
from collections import defaultdict

# Group observations by filter
observations = [
    {'object': 'M31', 'filter': 'V', 'mag': 3.44},
    {'object': 'M31', 'filter': 'B', 'mag': 4.36},
    {'object': 'M42', 'filter': 'V', 'mag': 4.00},
    {'object': 'M31', 'filter': 'R', 'mag': 2.73}
]

by_object = defaultdict(list)
for obs in observations:
    by_object[obs['object']].append(obs)

for obj, obs_list in by_object.items():
    filters = [o['filter'] for o in obs_list]
    print(f"{obj}: observed in {filters}")

# ChainMap for configuration hierarchy
from collections import ChainMap

default_config = {'exposure': 300, 'binning': 1, 'filter': 'V'}
user_config = {'exposure': 600, 'filter': 'R'}
session_config = {'binning': 2}

# ChainMap searches in order: session -> user -> default
config = ChainMap(session_config, user_config, default_config)
print(f"Exposure time: {config['exposure']}s")  # From user_config
print(f"Binning: {config['binning']}")  # From session_config
print(f"Filter: {config['filter']}")  # From user_config
```

## 4.5 Choosing the Right Data Structure

### Performance Comparison

```python
import time
import random

def time_operation(func, *args, **kwargs):
    """Time a single operation."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result

# Compare lookup performance
n = 100000
data_list = list(range(n))
data_set = set(range(n))
data_dict = {i: i**2 for i in range(n)}

# Random lookups
lookups = random.sample(range(n), 1000)

# List lookup - O(n)
list_time, _ = time_operation(lambda: [x in data_list for x in lookups])

# Set lookup - O(1)
set_time, _ = time_operation(lambda: [x in data_set for x in lookups])

# Dict lookup - O(1)
dict_time, _ = time_operation(lambda: [data_dict.get(x) for x in lookups])

print(f"1000 lookups in {n} elements:")
print(f"  List: {list_time:.4f}s")
print(f"  Set:  {set_time:.4f}s ({list_time/set_time:.0f}x faster)")
print(f"  Dict: {dict_time:.4f}s ({list_time/dict_time:.0f}x faster)")
```

### Decision Tree for Data Structures

```python
def choose_data_structure(requirements):
    """
    Help choose the right data structure based on requirements.
    """
    if requirements.get('ordered', False):
        if requirements.get('mutable', True):
            if requirements.get('key_value', False):
                return "OrderedDict or dict (Python 3.7+)"
            else:
                return "list"
        else:
            return "tuple"
    else:
        if requirements.get('unique', False):
            return "set"
        elif requirements.get('key_value', False):
            return "dict"
        else:
            return "list or deque"

# Examples for astronomy
print("Star catalog:", choose_data_structure({'key_value': True, 'ordered': True}))
print("Unique sources:", choose_data_structure({'unique': True}))
print("Time series:", choose_data_structure({'ordered': True, 'mutable': True}))
print("Coordinates:", choose_data_structure({'ordered': True, 'mutable': False}))
```

## 4.6 Fundamental Algorithms

### Searching Algorithms

```python
def linear_search(arr, target):
    """
    O(n) - Check every element.
    Good for unsorted data or small lists.
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def binary_search(arr, target):
    """
    O(log n) - Requires sorted data.
    Much faster for large datasets.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Compare performance
import bisect

wavelengths = sorted([random.uniform(3000, 8000) for _ in range(10000)])
target = 5500.0

# Add target to ensure it exists
wavelengths.append(target)
wavelengths.sort()

# Linear search
start = time.perf_counter()
idx_linear = linear_search(wavelengths, target)
linear_time = time.perf_counter() - start

# Binary search
start = time.perf_counter()
idx_binary = binary_search(wavelengths, target)
binary_time = time.perf_counter() - start

# Python's bisect module
start = time.perf_counter()
idx_bisect = bisect.bisect_left(wavelengths, target)
bisect_time = time.perf_counter() - start

print(f"Searching {len(wavelengths)} wavelengths for {target}nm:")
print(f"  Linear search: {linear_time*1e6:.2f}μs")
print(f"  Binary search: {binary_time*1e6:.2f}μs ({linear_time/binary_time:.0f}x faster)")
print(f"  Bisect module: {bisect_time*1e6:.2f}μs")
```

### Sorting Algorithms

```python
def bubble_sort(arr):
    """
    O(n²) - Simple but inefficient.
    Good for understanding, bad for production.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quicksort(arr):
    """
    O(n log n) average - Efficient divide-and-conquer.
    Python's sort() uses Timsort, which is even better.
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Sorting astronomical objects by multiple criteria
galaxies = [
    {'name': 'NGC1234', 'redshift': 0.023, 'magnitude': 14.2},
    {'name': 'NGC5678', 'redshift': 0.015, 'magnitude': 13.8},
    {'name': 'NGC9012', 'redshift': 0.023, 'magnitude': 13.5},
    {'name': 'NGC3456', 'redshift': 0.031, 'magnitude': 14.8}
]

# Sort by redshift, then by magnitude
galaxies.sort(key=lambda g: (g['redshift'], g['magnitude']))
print("Galaxies sorted by redshift, then magnitude:")
for g in galaxies:
    print(f"  {g['name']}: z={g['redshift']:.3f}, m={g['magnitude']}")
```

### Algorithm Complexity Analysis

```python
import matplotlib.pyplot as plt

def analyze_complexity():
    """Visualize algorithm complexity."""
    n_values = np.linspace(1, 100, 100)
    
    # Different complexity classes
    O_1 = np.ones_like(n_values)
    O_log_n = np.log2(n_values)
    O_n = n_values
    O_n_log_n = n_values * np.log2(n_values)
    O_n2 = n_values ** 2
    O_2n = 2 ** np.minimum(n_values, 20)  # Cap at 20 to avoid overflow
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, O_1, label='O(1) - Hash lookup')
    plt.plot(n_values, O_log_n, label='O(log n) - Binary search')
    plt.plot(n_values, O_n, label='O(n) - Linear search')
    plt.plot(n_values, O_n_log_n, label='O(n log n) - Good sorting')
    plt.plot(n_values, O_n2, label='O(n²) - Nested loops')
    plt.plot(n_values[:20], O_2n[:20], label='O(2ⁿ) - Brute force', linestyle='--')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Operations')
    plt.title('Algorithm Complexity Comparison')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 500)
    plt.grid(True, alpha=0.3)
    plt.show()

# analyze_complexity()  # Uncomment to see plot
```

## 4.7 Memory Efficiency

### Space Complexity

```python
import sys

# Memory usage of different data structures
def get_size(obj):
    """Get memory size in bytes."""
    return sys.getsizeof(obj)

# Compare memory usage
n = 1000
list_data = list(range(n))
tuple_data = tuple(range(n))
set_data = set(range(n))
dict_data = {i: i for i in range(n)}

print(f"Memory usage for {n} integers:")
print(f"  List:  {get_size(list_data):,} bytes")
print(f"  Tuple: {get_size(tuple_data):,} bytes")
print(f"  Set:   {get_size(set_data):,} bytes")
print(f"  Dict:  {get_size(dict_data):,} bytes")

# Generators for memory efficiency
def read_large_catalog_generator(n):
    """Generate data on-the-fly instead of storing it all."""
    for i in range(n):
        yield {'id': i, 'ra': random.uniform(0, 360), 'dec': random.uniform(-90, 90)}

# Generator uses almost no memory
gen = read_large_catalog_generator(1000000)
print(f"\nGenerator for 1M objects: {get_size(gen)} bytes")

# But converting to list uses lots of memory
# data_list = list(gen)  # Would use ~100MB!
```

## Try It Yourself

### Exercise 4.1: Catalog Cross-Matching
Build an efficient catalog cross-matcher using appropriate data structures.

```python
def cross_match_catalogs(catalog1, catalog2, tolerance=1.0):
    """
    Cross-match two astronomical catalogs by position.
    
    Parameters
    ----------
    catalog1, catalog2 : list of dict
        Each dict has 'id', 'ra', 'dec' keys
    tolerance : float
        Matching radius in arcseconds
    
    Returns
    -------
    list of tuple
        Matched pairs (id1, id2)
    """
    # Your code here
    # Hint: Consider using spatial indexing or KD-trees
    # For now, implement a simple nested loop and think about optimization
    pass

# Test data
cat1 = [
    {'id': 'A1', 'ra': 150.0, 'dec': 2.0},
    {'id': 'A2', 'ra': 150.1, 'dec': 2.1},
    {'id': 'A3', 'ra': 151.0, 'dec': 2.5}
]

cat2 = [
    {'id': 'B1', 'ra': 150.01, 'dec': 2.01},
    {'id': 'B2', 'ra': 151.5, 'dec': 2.4},
    {'id': 'B3', 'ra': 150.09, 'dec': 2.11}
]

matches = cross_match_catalogs(cat1, cat2)
print(f"Found {len(matches)} matches")
```

### Exercise 4.2: Time Series Analysis
Create an efficient data structure for astronomical time series.

```python
class TimeSeries:
    """
    Efficient storage and analysis of astronomical time series data.
    
    Should support:
    - Adding observations (time, magnitude, error)
    - Finding observations in time range
    - Computing statistics (mean, std, etc.)
    - Detecting outliers
    """
    
    def __init__(self):
        # Your code here
        # Choose appropriate data structures
        pass
    
    def add_observation(self, time, magnitude, error=None):
        """Add a single observation."""
        pass
    
    def get_range(self, start_time, end_time):
        """Get observations in time range."""
        pass
    
    def find_outliers(self, sigma=3):
        """Find observations more than sigma std from mean."""
        pass

# Test your implementation
ts = TimeSeries()
# Add simulated variable star observations
for t in np.linspace(0, 10, 100):
    mag = 12.0 + 0.5 * np.sin(2 * np.pi * t / 3.0) + np.random.normal(0, 0.05)
    ts.add_observation(t, mag, error=0.05)

outliers = ts.find_outliers()
print(f"Found {len(outliers)} outliers")
```

### Exercise 4.3: Spectral Line Finder
Implement an algorithm to find emission/absorption lines in spectra.

```python
def find_spectral_lines(wavelengths, fluxes, threshold=3.0):
    """
    Find significant spectral lines in spectrum.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength array (nm)
    fluxes : array
        Flux array
    threshold : float
        Detection threshold in sigma above continuum
    
    Returns
    -------
    dict
        Line information including wavelength, type (emission/absorption)
    """
    # Your code here
    # 1. Estimate continuum level
    # 2. Find peaks above/below threshold
    # 3. Return line catalog
    pass

# Generate test spectrum with lines
wavelengths = np.linspace(400, 700, 1000)
continuum = np.ones_like(wavelengths)
spectrum = continuum.copy()

# Add emission lines
for line_wave in [486.1, 656.3]:  # H-beta, H-alpha
    sigma = 0.5
    spectrum += 2.0 * np.exp(-(wavelengths - line_wave)**2 / (2 * sigma**2))

# Add absorption line
spectrum -= 0.5 * np.exp(-(wavelengths - 589.0)**2 / (2 * 0.3**2))  # Sodium D

lines = find_spectral_lines(wavelengths, spectrum)
print(f"Found {len(lines)} spectral lines")
```

## Key Takeaways

✅ **Lists** for ordered, mutable sequences - but watch out for O(n) operations  
✅ **Tuples** for immutable data like coordinates - can be dictionary keys  
✅ **Sets** for unique elements and O(1) membership testing  
✅ **Dictionaries** for key-value mappings and O(1) lookups  
✅ **Algorithm complexity** matters: O(1) < O(log n) < O(n) < O(n²)  
✅ **Binary search** beats linear search for sorted data  
✅ **Generators** save memory for large datasets  
✅ **Choose data structures** based on access patterns and operations  

## Next Chapter Preview
We'll extend these concepts to object-oriented programming, creating classes that combine data structures with behavior to model astronomical objects and systems.