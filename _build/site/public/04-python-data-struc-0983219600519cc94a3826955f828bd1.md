# Chapter 4: Data Structures

## Learning Objectives

By the end of this chapter, you will be able to:
- Choose optimal data structures based on algorithmic requirements and performance constraints
- Predict whether operations will be O(1) constant time or O(n) linear time
- Understand memory layout and cache efficiency for scientific computing
- Implement defensive copying strategies to prevent aliasing bugs
- Profile memory usage and optimize data structure choices for large datasets
- Design data structures that prepare you for vectorized computing and JAX
- Debug common bugs related to mutability, aliasing, and hashability
- Apply data structure patterns to real scientific computing problems

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Write loops and conditional statements fluently (Chapter 3)
- ✓ Understand the difference between assignment and equality (Chapter 2)
- ✓ Use IPython for testing and timing code (Chapter 1)
- ✓ Handle floating-point numbers and comparisons safely (Chapter 2)

## Chapter Overview

When you're processing a million stellar spectra or tracking particles in an N-body simulation, the difference between choosing a list versus a set can be the difference between your code running in seconds or hours. Data structures are the fundamental ways we organize information in memory, and each structure makes certain operations efficient while making others expensive.

This chapter builds your intuition for computational complexity through empirical measurement. You'll learn not just that dictionary lookup is O(1), but *why* it's fast, *when* it might not be, and *how* to verify performance characteristics yourself. We'll explore the critical distinction between mutable and immutable objects — a concept that seems academic until your simulation corrupts its initial conditions because of an aliasing bug.

These concepts directly prepare you for the numerical computing ahead. The memory layout discussions explain why NumPy arrays are 10x more efficient than lists. The immutability concepts prepare you for JAX's functional programming requirements. The performance profiling skills will help you identify bottlenecks in your Monte Carlo simulations.

## 4.1 What Is a Data Structure?

A data structure is a way of organizing data in computer memory to enable efficient access and modification. Think of it like choosing how to organize astronomical observations: you could keep them in time order (like a list), organize by object ID for quick lookup (like a dictionary), or maintain only unique objects (like a set). Each organization serves different purposes.

### Understanding Big-O Notation

Big-O notation describes how an operation's time grows with input size. This isn't abstract computer science — it's the difference between code that scales and code that doesn't:

```python
In [1]: import time
In [2]: import random

In [3]: # Create test data
In [4]: n = 1_000_000
In [5]: big_list = list(range(n))
In [6]: big_set = set(range(n))

In [7]: # Search for element not present (worst case)
In [8]: target = -1

In [9]: # O(n) list search - checks every element
In [10]: start = time.perf_counter()
In [11]: found = target in big_list
In [12]: list_time = time.perf_counter() - start

In [13]: # O(1) set search - direct hash lookup
In [14]: start = time.perf_counter()
In [15]: found = target in big_set
In [16]: set_time = time.perf_counter() - start

In [17]: print(f"List search: {list_time*1000:.2f} ms")
In [18]: print(f"Set search:  {set_time*1000:.4f} ms")
In [19]: print(f"Set is {list_time/set_time:.0f}x faster!")

List search: 12.45 ms
Set search:  0.0012 ms
Set is 10,375x faster!
```

This 10,000x difference isn't a minor optimization — it determines whether your catalog cross-matching finishes today or next week.

### 📦 **Computational Thinking Box: The Time-Space Tradeoff**

```
UNIVERSAL PATTERN: Trading Memory for Speed

Many data structures follow this pattern:
- Use more memory to organize data
- This organization enables faster access
- The tradeoff is worthwhile when access is frequent

Examples:
- Hash tables (dict/set): ~3x memory for O(1) lookup
- Search trees: 2x memory for O(log n) ordered access
- Cacheing computed values: memory for avoiding recomputation
- Spatial indices (octrees): memory for fast neighbor finding

The pattern appears in:
- Opacity tables in radiative transfer (cache vs recompute)
- Neighbor lists in N-body simulations
- Memoization in dynamic programming
- Database indices for catalog queries
```

## 4.2 Lists: Python's Workhorse Sequence

Lists are Python's most versatile data structure, perfect for ordered collections that change size. However, understanding their internal implementation is crucial for writing efficient code.

### How Lists Really Work in Memory

Python lists don't store your data directly — they store references to objects elsewhere in memory:

```python
In [20]: import sys

In [21]: # Let's examine memory usage
In [22]: numbers = [100, 200, 300]

In [23]: # The list container
In [24]: list_size = sys.getsizeof(numbers)
In [25]: print(f"List container: {list_size} bytes")

In [26]: # Each integer is a full object
In [27]: element_sizes = [sys.getsizeof(n) for n in numbers]
In [28]: print(f"Each integer: {element_sizes[0]} bytes")

In [29]: # Total memory
In [30]: total = list_size + sum(element_sizes)
In [31]: print(f"Total: {total} bytes for 3 integers")
In [32]: print(f"That's {total/12:.1f}x more than raw integers!")

List container: 80 bytes
Each integer: 28 bytes
Total: 164 bytes for 3 integers
That's 13.7x more than raw integers!
```

Here's what's actually happening in memory:

```
Visual: List Memory Layout

Python List 'numbers':          Objects in Heap Memory:
┌─────────────────┐            
│  list header    │            ┌─────────────────┐
│  size: 3        │            │ int object      │
│  capacity: 4    │            │ type: int       │
├─────────────────┤            │ refcount: 1     │
│  ref to 100 ────┼──────────> │ value: 100      │
├─────────────────┤            └─────────────────┘
│  ref to 200 ────┼──────────> ┌─────────────────┐
├─────────────────┤            │ int object      │
│  ref to 300 ────┼──────────> │ value: 200      │
├─────────────────┤            └─────────────────┘
│  (unused slot)  │            ┌─────────────────┐
└─────────────────┘            │ int object      │
                               │ value: 300      │
                               └─────────────────┘

Key Insights:
- List stores pointers, not values
- Each integer is a full Python object (28 bytes!)
- List overallocates (capacity > size) for growth
- This is why NumPy arrays are more efficient
```

### List Operations: Performance Characteristics

Different list operations have vastly different costs:

```python
In [33]: def demonstrate_list_performance():
   ...:     """Show why operation location matters."""
   ...:     import time
   ...:     
   ...:     test_list = list(range(100_000))
   ...:     
   ...:     # Fast O(1): Operations at the END
   ...:     start = time.perf_counter()
   ...:     test_list.append(999)
   ...:     test_list.pop()
   ...:     end_time = time.perf_counter() - start
   ...:     
   ...:     # Slow O(n): Operations at the BEGINNING
   ...:     start = time.perf_counter()
   ...:     test_list.insert(0, 999)
   ...:     test_list.pop(0)
   ...:     begin_time = time.perf_counter() - start
   ...:     
   ...:     print(f"Operations at end:   {end_time*1e6:.2f} µs")
   ...:     print(f"Operations at start: {begin_time*1e6:.2f} µs")
   ...:     print(f"Beginning is {begin_time/end_time:.0f}x slower!")

In [34]: demonstrate_list_performance()
Operations at end:   0.75 µs
Operations at start: 524.32 µs
Beginning is 699x slower!
```

Why such a huge difference? Operations at the beginning require shifting all elements:

```
Visual: Why insert(0, x) is O(n)

Before insert(0, 'X'):
[0][1][2][3][4][5]

Step 1: Shift everything right
[_][0][1][2][3][4][5]

Step 2: Insert new element
[X][0][1][2][3][4][5]

With a million elements, this means moving a million references!
```

### 📊 **Performance Profile: List Growth Strategy**

Python lists use dynamic arrays that grow by ~12.5% when full:

```python
In [35]: def observe_list_growth():
   ...:     """Watch Python's list growth strategy."""
   ...:     data = []
   ...:     sizes = []
   ...:     capacities = []
   ...:     
   ...:     for i in range(20):
   ...:         old_size = sys.getsizeof(data)
   ...:         data.append(i)
   ...:         new_size = sys.getsizeof(data)
   ...:         
   ...:         if new_size != old_size:
   ...:             # Calculate capacity from size
   ...:             capacity = (new_size - sys.getsizeof([])) // 8 + 1
   ...:             sizes.append(len(data))
   ...:             capacities.append(capacity)
   ...:     
   ...:     print("Length → Capacity (overallocation)")
   ...:     for s, c in zip(sizes, capacities):
   ...:         overalloc = (c - s) / s * 100 if s > 0 else 0
   ...:         print(f"{s:4d} → {c:4d} ({overalloc:5.1f}% extra)")

In [36]: observe_list_growth()
Length → Capacity (overallocation)
   1 →    4 (300.0% extra)
   5 →    8 ( 60.0% extra)
   9 →   16 ( 77.8% extra)
  17 →   24 ( 41.2% extra)
```

This overallocation strategy makes append() *amortized* O(1) — usually fast, occasionally slow when reallocation happens.

### List Patterns for Scientific Computing

```python
# Pattern 1: Preallocate for known size
n_particles = 10000
positions = [None] * n_particles  # Preallocate
for i in range(n_particles):
    positions[i] = compute_position(i)  # Fill in

# Pattern 2: Collect results conditionally
valid_measurements = []
for measurement in sensor_data:
    if measurement.quality > threshold:
        valid_measurements.append(measurement)

# Pattern 3: In-place modification
for i in range(len(data)):
    data[i] *= scaling_factor  # Modifies existing list
```

## 4.3 Tuples: The Power of Immutability

Tuples are immutable sequences. This restriction provides powerful guarantees that prevent entire categories of bugs.

### Understanding Immutability's Value

```python
In [40]: # Lists are mutable - source of bugs
In [41]: def buggy_function(data, params):
   ...:     """This function accidentally modifies params!"""
   ...:     params.append(data.mean())  # Oops, modifying input!
   ...:     return sum(params)

In [42]: parameters = [1.0, 2.0, 3.0]
In [43]: result = buggy_function(np.array([4, 5, 6]), parameters)
In [44]: parameters
Out[44]: [1.0, 2.0, 3.0, 5.0]  # Changed unexpectedly!

In [45]: # Tuples prevent this
In [46]: def safe_function(data, params):
   ...:     """Can't accidentally modify tuple params."""
   ...:     # params.append(data.mean())  # Would raise AttributeError
   ...:     return sum(params) + data.mean()

In [47]: parameters = (1.0, 2.0, 3.0)  # Tuple
In [48]: result = safe_function(np.array([4, 5, 6]), parameters)
In [49]: parameters
Out[49]: (1.0, 2.0, 3.0)  # Unchanged, guaranteed!
```

### Tuples as Dictionary Keys

Immutability enables hashability, allowing tuples as dictionary keys:

```python
In [50]: # Cache expensive calculations using position as key
In [51]: potential_cache = {}

In [52]: def gravitational_potential(pos, mass, use_cache=True):
   ...:     """Calculate potential, with caching."""
   ...:     if use_cache and pos in potential_cache:
   ...:         return potential_cache[pos]
   ...:     
   ...:     # Expensive calculation
   ...:     x, y, z = pos
   ...:     r = (x**2 + y**2 + z**2) ** 0.5
   ...:     G = 6.67e-8
   ...:     potential = -G * mass / r
   ...:     
   ...:     if use_cache:
   ...:         potential_cache[pos] = potential
   ...:     
   ...:     return potential

In [53]: # Must use tuple for position
In [54]: pos1 = (1e10, 0, 0)  # Tuple - hashable
In [55]: V1 = gravitational_potential(pos1, 1e30)  # Computed

In [56]: pos2 = (1e10, 0, 0)  # Same position
In [57]: V2 = gravitational_potential(pos2, 1e30)  # From cache!

In [58]: # Lists can't be keys
In [59]: pos_list = [1e10, 0, 0]
In [60]: # potential_cache[pos_list] = V1  # TypeError!
```

### Named Tuples: Self-Documenting Science Code

```python
In [61]: from collections import namedtuple

In [62]: # Define structure with meaningful names
In [63]: Star = namedtuple('Star', 
   ...:     ['mass', 'radius', 'temperature', 'luminosity'])

In [64]: # Create instances with clear meaning
In [65]: sun = Star(
   ...:     mass=1.989e33,        # grams
   ...:     radius=6.96e10,       # cm
   ...:     temperature=5778,     # Kelvin
   ...:     luminosity=3.828e33   # erg/s
   ...: )

In [66]: # Clear, self-documenting access
In [67]: print(f"Solar mass: {sun.mass:.2e} g")
In [68]: print(f"Solar radius: {sun.radius:.2e} cm")

In [69]: # Still works as regular tuple
In [70]: M, R, T, L = sun
In [71]: density = M / ((4/3) * 3.14159 * R**3)
```

### 🐛 **Debug This!**

This code has a subtle bug. Can you find it?

```python
def process_coordinates(coords_list):
    """Process a list of coordinate tuples."""
    
    results = []
    for coords in coords_list:
        # Try to normalize coordinates
        coords[0] = coords[0] / 1000  # Convert to km
        coords[1] = coords[1] / 1000
        coords[2] = coords[2] / 1000
        results.append(coords)
    
    return results

# Test
positions = [(1000, 2000, 3000), (4000, 5000, 6000)]
normalized = process_coordinates(positions)
```

<details>
<summary>Bug and Solution</summary>

**Bug**: Tuples are immutable! Can't modify coords[0].

**Solution 1**: Create new tuples
```python
def process_coordinates(coords_list):
    results = []
    for coords in coords_list:
        normalized = (
            coords[0] / 1000,
            coords[1] / 1000,
            coords[2] / 1000
        )
        results.append(normalized)
    return results
```

**Solution 2**: Use list comprehension
```python
def process_coordinates(coords_list):
    return [(x/1000, y/1000, z/1000) for x, y, z in coords_list]
```

</details>

## 4.4 The Mutable vs Immutable Distinction

Understanding mutability is crucial for avoiding bugs and preparing for functional programming paradigms (essential for JAX).

### Python's Reference Model

```python
In [72]: # Visualize Python's reference model
In [73]: import id

In [74]: # Immutable: creates new objects
In [75]: a = 1000
In [76]: b = a
In [77]: print(f"Initially: a={a}, b={b}, same object: {a is b}")

In [78]: b = 2000  # Creates NEW object
In [79]: print(f"After b=2000: a={a}, b={b}, same object: {a is b}")

Initially: a=1000, b=1000, same object: True
After b=2000: a=1000, b=2000, same object: False

In [80]: # Mutable: modifies existing object
In [81]: list1 = [1, 2, 3]
In [82]: list2 = list1  # Both refer to SAME list
In [83]: print(f"Initially: same object: {list1 is list2}")

In [84]: list2.append(4)  # Modifies THE list
In [85]: print(f"After append: list1={list1}, list2={list2}")

Initially: same object: True
After append: list1=[1, 2, 3, 4], list2=[1, 2, 3, 4]
```

Visual representation:

```
Immutable (after b = 2000):        Mutable (after append):
a ──→ [1000]                       list1 ──→ [1,2,3,4]
b ──→ [2000]                       list2 ──┘

Separate objects                   Same object!
```

### The Classic Mutable Default Argument Bug

```python
In [86]: # THE BUG: Mutable default created ONCE
In [87]: def accumulate_data_buggy(value, data=[]):
   ...:     """BUGGY: Default list created once at definition!"""
   ...:     data.append(value)
   ...:     return data

In [88]: result1 = accumulate_data_buggy(10)
In [89]: print(f"First call: {result1}")

In [90]: result2 = accumulate_data_buggy(20)
In [91]: print(f"Second call: {result2}")  # Contains both!

In [92]: result1 is result2
Out[92]: True  # Same list object!

First call: [10]
Second call: [10, 20]

In [93]: # THE FIX: Use None sentinel
In [94]: def accumulate_data_fixed(value, data=None):
   ...:     """Safe version using None default."""
   ...:     if data is None:
   ...:         data = []  # Fresh list each call
   ...:     data.append(value)
   ...:     return data
```

### 📈 **Algorithm Archaeology: Why Mutable Defaults Exist**

Python evaluates default arguments once when the function is defined, not each time it's called. This was a design choice for efficiency — evaluating defaults every call would be expensive.

This decision made sense in 1991 when Python was created, but it's been a source of bugs ever since. Modern languages like Rust and Swift evaluate defaults at call time. Python keeps this behavior for backward compatibility.

The mutable default bug is so common that linters specifically check for it. Always use the None sentinel pattern for mutable defaults.

### Shallow vs Deep Copies: Critical for Scientific Data

```python
In [95]: import copy

In [96]: # Original nested structure - like a 2D grid
In [97]: grid = [[1, 2], [3, 4], [5, 6]]

In [98]: # Shallow copy - new outer list, same inner lists
In [99]: shallow = grid.copy()

In [100]: # Modify through shallow copy
In [101]: shallow[0][0] = 999

In [102]: print(f"Original: {grid}")
In [103]: print(f"Shallow:  {shallow}")

Original: [[999, 2], [3, 4], [5, 6]]  # Changed!
Shallow:  [[999, 2], [3, 4], [5, 6]]
```

Memory visualization:

```
Shallow Copy:
grid ──→ [ ref1, ref2, ref3 ] ──→ [1,2] [3,4] [5,6]
                                     ↑     ↑     ↑
shallow → [ ref1, ref2, ref3 ] ─────┘     │     │
          (new outer list, same inner lists!)
```

Deep copy solves this:

```python
In [104]: # Reset
In [105]: grid = [[1, 2], [3, 4], [5, 6]]

In [106]: # Deep copy - all new objects
In [107]: deep = copy.deepcopy(grid)

In [108]: deep[0][0] = 999
In [109]: print(f"Original: {grid}")
In [110]: print(f"Deep:     {deep}")

Original: [[1, 2], [3, 4], [5, 6]]  # Unchanged!
Deep:     [[999, 2], [3, 4], [5, 6]]
```

### Defensive Copying in Scientific Code

```python
def safe_normalize(data, reference=None):
    """
    Normalize data without modifying inputs.
    Demonstrates defensive copying patterns.
    """
    # Defensive copy of mutable input
    working_data = copy.deepcopy(data)
    
    # Safe to modify working_data now
    if reference is None:
        reference = max(max(row) for row in working_data)
    
    for i in range(len(working_data)):
        for j in range(len(working_data[i])):
            working_data[i][j] /= reference
    
    return working_data

# Original data unchanged
original = [[100, 200], [300, 400]]
normalized = safe_normalize(original)
print(f"Original unchanged: {original}")
print(f"Normalized: {normalized}")
```

## 4.5 Dictionaries: O(1) Lookup via Hash Tables

Dictionaries provide near-instantaneous lookup regardless of size, using a hash table implementation.

### Understanding Hash Tables (Simplified)

```python
# Conceptual demonstration of hashing
def simple_hash_demo():
    """Show how hash tables enable O(1) lookup."""
    
    # Python's hash() converts objects to integers
    keys = ["mass", "radius", "temperature"]
    
    for key in keys:
        hash_value = hash(key)
        # In real hash table: index = hash_value % table_size
        index = abs(hash_value) % 10
        print(f"'{key}' → hash: {hash_value:12d} → bucket: {index}")
    
    print("\nThis is why lookup is O(1):")
    print("1. Hash the key (fast)")
    print("2. Go directly to bucket (fast)")
    print("3. Check if key matches (fast)")

simple_hash_demo()
```

### Dictionary Performance in Practice

```python
In [120]: # Compare list search vs dict lookup
In [121]: n = 1_000_000

In [122]: # List of tuples (slow search)
In [123]: star_list = [(f"HD{i}", random.random()) 
   ...:                for i in range(n)]

In [124]: # Dictionary (fast lookup)
In [125]: star_dict = {f"HD{i}": random.random() 
   ...:                for i in range(n)}

In [126]: # Search for specific star
In [127]: target = "HD500000"

In [128]: # List search - O(n)
In [129]: %timeit next((mag for name, mag in star_list if name == target))
24.3 ms ± 312 µs per loop

In [130]: # Dict lookup - O(1)
In [131]: %timeit star_dict[target]
41.2 ns ± 0.8 ns per loop

Dict is 590,000x faster!
```

### Dictionary Patterns for Scientific Computing

```python
# Pattern 1: Configuration management
simulation_params = {
    'n_particles': 10000,
    'timestep': 1e-4,
    'total_time': 100.0,
    'G': 6.67e-8,
    'softening': 1e-6,
    'output_freq': 100
}

# Safe access with defaults
dt = simulation_params.get('timestep', 1e-3)
n = simulation_params.get('n_particles', 1000)

# Pattern 2: Caching expensive computations
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_calculation(n):
    """Automatically caches last 1000 results."""
    result = sum(i**2 for i in range(n))
    return result

# Pattern 3: Grouping data
from collections import defaultdict

def group_by_type(observations):
    """Group observations by object type."""
    groups = defaultdict(list)
    
    for obs in observations:
        groups[obs['type']].append(obs)
    
    return dict(groups)

# Pattern 4: Counting occurrences
from collections import Counter

def analyze_spectral_types(stars):
    """Count distribution of spectral types."""
    types = [star.spectral_type for star in stars]
    return Counter(types)
```

### 📦 **Computational Thinking Box: The Caching Pattern**

```
UNIVERSAL PATTERN: Trading Memory for Computation

cache = {}

FUNCTION compute_expensive(input):
    IF input IN cache:
        RETURN cache[input]
    
    result = expensive_calculation(input)
    cache[input] = result
    RETURN result

This pattern appears everywhere:
- Opacity tables in radiative transfer
- Basis function evaluation in spectral methods
- Distance matrices in clustering
- Factorial/combinatorial calculations
- Interpolation table lookups

Python's @lru_cache decorator implements this pattern automatically.
```

## 4.6 Sets: Mathematical Operations on Unique Elements

Sets provide O(1) membership testing and elegant mathematical operations:

```python
In [140]: # Catalog cross-matching example
In [141]: observed = {'HD209458', 'HD189733', 'WASP-12', 'HAT-P-7'}
In [142]: confirmed = {'HD209458', 'WASP-12', 'Kepler-7', 'WASP-43'}

In [143]: # Set operations
In [144]: both = observed & confirmed  # Intersection
In [145]: either = observed | confirmed  # Union
In [146]: only_observed = observed - confirmed  # Difference
In [147]: different = observed ^ confirmed  # Symmetric difference

In [148]: print(f"Both catalogs: {both}")
In [149]: print(f"Only in observed: {only_observed}")

Both catalogs: {'HD209458', 'WASP-12'}
Only in observed: {'HAT-P-7', 'HD189733'}
```

### Set Performance for Membership Testing

```python
In [150]: # Create large catalogs
In [151]: catalog_list = [f"Object_{i}" for i in range(1_000_000)]
In [152]: catalog_set = set(catalog_list)

In [153]: # Test membership for non-existent object
In [154]: %timeit "Object_-1" in catalog_list
15.2 ms ± 89.3 µs per loop

In [155]: %timeit "Object_-1" in catalog_set
42.3 ns ± 0.5 ns per loop

Set is 359,000x faster!
```

### Common Set Patterns

```python
# Pattern 1: Remove duplicates while preserving order
def remove_duplicates_ordered(items):
    """Remove duplicates, preserve order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Pattern 2: Find common elements efficiently
def find_common_objects(catalog1, catalog2, catalog3):
    """Find objects in all three catalogs."""
    return set(catalog1) & set(catalog2) & set(catalog3)

# Pattern 3: Check if all elements are unique
def all_unique(items):
    """Check if all elements are unique."""
    return len(items) == len(set(items))
```

## 4.7 Optional: Hash Table Implementation Details

*This section provides deeper understanding of dictionary/set performance for those interested.*

### How Hash Tables Really Work

```python
class SimpleHashTable:
    """
    Simplified hash table to understand dict/set internals.
    Real Python dicts are much more sophisticated!
    """
    
    def __init__(self, size=8):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0
    
    def _hash(self, key):
        """Convert key to array index."""
        return hash(key) % self.size
    
    def put(self, key, value):
        """Insert key-value pair."""
        index = self._hash(key)
        
        # Linear probing for collision resolution
        while self.keys[index] is not None:
            if self.keys[index] == key:
                # Update existing
                self.values[index] = value
                return
            index = (index + 1) % self.size
        
        # Insert new
        self.keys[index] = key
        self.values[index] = value
        self.count += 1
        
        # Resize if getting full
        if self.count > self.size * 0.7:
            self._resize()
    
    def get(self, key):
        """Retrieve value for key."""
        index = self._hash(key)
        
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        
        raise KeyError(key)
    
    def _resize(self):
        """Double the table size when getting full."""
        old_keys = self.keys
        old_values = self.values
        
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        
        for key, value in zip(old_keys, old_values):
            if key is not None:
                self.put(key, value)
```

### Why Hash Tables Can Degrade to O(n)

```python
# Pathological case: hash collisions
class BadHash:
    """Object with terrible hash function."""
    def __init__(self, value):
        self.value = value
    
    def __hash__(self):
        return 42  # Always same hash!
    
    def __eq__(self, other):
        return self.value == other.value

# All objects hash to same bucket - O(n) performance!
bad_dict = {BadHash(i): i for i in range(1000)}
```

## 4.8 Memory and Performance Considerations

### Memory Profiling Your Data Structures

```python
In [160]: from memory_profiler import profile

In [161]: @profile
   ...: def compare_memory_usage():
   ...:     """Compare memory usage of different structures."""
   ...:     n = 100_000
   ...:     
   ...:     # List of integers
   ...:     int_list = list(range(n))
   ...:     
   ...:     # List of lists (2D)
   ...:     nested_list = [[i, i+1] for i in range(n)]
   ...:     
   ...:     # Dictionary
   ...:     int_dict = {i: i**2 for i in range(n)}
   ...:     
   ...:     # Set
   ...:     int_set = set(range(n))
   ...:     
   ...:     # NumPy array (preview)
   ...:     import numpy as np
   ...:     np_array = np.arange(n)
   ...:     
   ...:     return int_list, nested_list, int_dict, int_set, np_array
```

### Cache Efficiency and Memory Layout

```python
In [162]: def demonstrate_cache_effects():
   ...:     """Show why memory layout matters."""
   ...:     import time
   ...:     
   ...:     # Create 2D array (list of lists)
   ...:     size = 1000
   ...:     matrix = [[i*size + j for j in range(size)] 
   ...:               for i in range(size)]
   ...:     
   ...:     # Row-wise access (cache-friendly)
   ...:     start = time.perf_counter()
   ...:     total = 0
   ...:     for i in range(size):
   ...:         for j in range(size):
   ...:             total += matrix[i][j]
   ...:     row_time = time.perf_counter() - start
   ...:     
   ...:     # Column-wise access (cache-hostile)
   ...:     start = time.perf_counter()
   ...:     total = 0
   ...:     for j in range(size):
   ...:         for i in range(size):
   ...:             total += matrix[i][j]
   ...:     col_time = time.perf_counter() - start
   ...:     
   ...:     print(f"Row-wise:    {row_time*1000:.1f} ms")
   ...:     print(f"Column-wise: {col_time*1000:.1f} ms")
   ...:     print(f"Column-wise is {col_time/row_time:.1f}x slower")

In [163]: demonstrate_cache_effects()
Row-wise:    42.3 ms
Column-wise: 78.9 ms
Column-wise is 1.9x slower
```

Visual explanation:

```
Cache-Friendly Access (Row-wise):
Memory: [row0][row1][row2]...
Access: →→→→→→ (sequential, cache hits)

Cache-Hostile Access (Column-wise):
Memory: [row0][row1][row2]...
Access: ↓  ↓  ↓ (jumping, cache misses)
```

## 4.9 Choosing the Right Data Structure

### Decision Framework

```python
def choose_data_structure(requirements):
    """
    Decision tree for data structure selection.
    
    This is the thought process you should follow.
    """
    
    if "unique elements only" in requirements:
        if "need ordering" in requirements:
            return "sorted set or sorted(set(...))"
        else:
            return "set"
    
    if "key-value pairs" in requirements:
        if "need ordering" in requirements:
            return "OrderedDict or dict (Python 3.7+)"
        else:
            return "dict"
    
    if "immutable" in requirements:
        return "tuple"
    
    if "fast membership test" in requirements:
        return "set or dict"
    
    if "ordered sequence" in requirements:
        if "fast random access" in requirements:
            return "list or array"
        if "fast insertion/deletion at ends" in requirements:
            return "collections.deque"
    
    return "list (default choice)"
```

### Performance Comparison Table

| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Access by index | O(1) | O(1) | N/A | N/A |
| Search for value | O(n) | O(n) | O(1)* | O(1) |
| Add to end | O(1)† | N/A | O(1)† | O(1)† |
| Add to beginning | O(n) | N/A | O(1)† | O(1)† |
| Remove value | O(n) | N/A | O(1) | O(1) |
| Memory (relative) | 1x | 0.9x | 3x | 3x |

\* Dict searches by key, not value  
† Amortized - occasionally O(n) during resize

### Real-World Examples

```python
# Example 1: Particle simulation
class ParticleSystem:
    def __init__(self):
        # Lists for ordered, mutable data
        self.positions = []  # Will modify every timestep
        self.velocities = []
        
        # Tuple for immutable constants
        self.bounds = (0, 0, 100, 100)  # Can't accidentally change
        
        # Dict for parameters
        self.params = {'G': 6.67e-8, 'dt': 0.01}
        
        # Set for spatial hashing
        self.occupied_cells = set()  # Fast collision detection

# Example 2: Data processing pipeline
class DataPipeline:
    def __init__(self):
        # List for sequential processing
        self.stages = []
        
        # Dict for caching results
        self.cache = {}
        
        # Set for tracking processed IDs
        self.processed = set()
        
        # deque for rolling buffer
        from collections import deque
        self.recent = deque(maxlen=1000)
```

## Practice Exercises

### Exercise 4.1: Performance Profiler

```python
def profile_operations(n=10000):
    """
    Profile common operations on different data structures.
    
    Tasks:
    1. Create list, dict, set with n elements
    2. Time: membership test, addition, deletion
    3. Measure memory usage
    4. Plot results
    
    Return summary statistics.
    """
    # Your code here
    pass
```

### Exercise 4.2: Deep Copy Debugger

```python
def find_aliasing_bugs(data_structure):
    """
    Detect potential aliasing issues in nested structures.
    
    Tasks:
    1. Identify all mutable objects
    2. Check if any are referenced multiple times
    3. Suggest where deep copies might be needed
    4. Return diagnostic report
    """
    # Your code here
    pass
```

### Exercise 4.3: Cache Implementation

```python
class SmartCache:
    """
    Implement a cache with:
    1. Maximum size limit
    2. LRU eviction policy
    3. Hit/miss statistics
    4. Performance metrics
    
    Use this for expensive function results.
    """
    
    def __init__(self, maxsize=100):
        # Your code here
        pass
    
    def get(self, key):
        # Your code here
        pass
    
    def put(self, key, value):
        # Your code here
        pass
```

### Exercise 4.4: 🐛 **Debug This!**

```python
def process_observations(observations):
    """
    This function has multiple data structure bugs.
    Find and fix them all.
    """
    
    # Bug 1: Mutable default
    def add_metadata(obs, metadata={}):
        metadata['processed'] = True
        obs.update(metadata)
        return obs
    
    # Bug 2: Aliasing
    processed = []
    for obs in observations:
        processed.append(obs)
        processed[-1]['timestamp'] = time.time()
    
    # Bug 3: Modifying during iteration
    for obs in processed:
        if obs['quality'] < 0.5:
            processed.remove(obs)
    
    return processed
```

## Key Takeaways

Data structure choice determines algorithm efficiency. A O(n²) algorithm with the wrong data structure becomes O(n³). Always profile with realistic data sizes.

Lists are versatile but have O(n) search and O(n) insertion at the beginning. Use them for ordered data that you'll access by index.

Dictionaries and sets provide O(1) average-case lookup through hash tables. Use them when you need fast membership testing or key-value mapping.

Immutability prevents entire categories of bugs. Use tuples for data that shouldn't change. This prepares you for functional programming in JAX.

The shallow vs deep copy distinction is critical when working with nested structures. Unexpected aliasing is a common source of bugs in scientific code.

Memory layout affects cache performance. Row-wise vs column-wise access can differ by 2-10x in speed for large arrays.

Python's reference-based model has overhead. This is why NumPy arrays (which store raw data contiguously) are so much more efficient for numerical work.

## Quick Reference: Data Structure Operations

| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Create empty | `[]` | `()` | `{}` | `set()` |
| Create with items | `[1,2,3]` | `(1,2,3)` | `{'a':1}` | `{1,2,3}` |
| Add item | `.append(x)` | N/A | `d[k]=v` | `.add(x)` |
| Remove item | `.remove(x)` | N/A | `del d[k]` | `.remove(x)` |
| Check membership | `x in list` | `x in tuple` | `k in dict` | `x in set` |
| Get by index | `list[i]` | `tuple[i]` | N/A | N/A |
| Get by key | N/A | N/A | `dict[k]` | N/A |
| Length | `len(list)` | `len(tuple)` | `len(dict)` | `len(set)` |
| Iterate | `for x in list` | `for x in tuple` | `for k in dict` | `for x in set` |
| Copy (shallow) | `list.copy()` | N/A | `dict.copy()` | `set.copy()` |
| Copy (deep) | `deepcopy(list)` | N/A | `deepcopy(dict)` | N/A |

### Common Methods

| Structure | Method | Purpose | Example |
|-----------|--------|---------|---------|
| list | `.append(x)` | Add to end | `lst.append(5)` |
| list | `.extend(iter)` | Add multiple | `lst.extend([1,2,3])` |
| list | `.insert(i,x)` | Insert at position | `lst.insert(0, 'first')` |
| list | `.pop(i=-1)` | Remove and return | `last = lst.pop()` |
| list | `.sort()` | Sort in place | `lst.sort()` |
| dict | `.get(k, default)` | Safe access | `d.get('key', 0)` |
| dict | `.keys()` | Get all keys | `for k in d.keys():` |
| dict | `.values()` | Get all values | `sum(d.values())` |
| dict | `.items()` | Get (key,value) pairs | `for k,v in d.items():` |
| set | `.add(x)` | Add element | `s.add(42)` |
| set | `.union(other)` | Combine sets | `s1 | s2` |
| set | `.intersection(other)` | Common elements | `s1 & s2` |
| set | `.difference(other)` | Elements in s1 not s2 | `s1 - s2` |

## Next Chapter Preview

With data structures mastered, Chapter 5 will explore functions and modules — how to organize code for reusability, testing, and collaboration. You'll learn how Python's function model, with first-class functions and closure support, enables powerful patterns like decorators and functional programming techniques. These concepts prepare you for the modular algorithm design essential for complex simulations and the functional programming paradigm required for JAX.