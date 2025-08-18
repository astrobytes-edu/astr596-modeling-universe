# Chapter 7: NumPy - The Foundation of Scientific Computing

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand why NumPy arrays are 10-100x faster than Python lists for numerical computation
- Create and manipulate arrays using various initialization methods and slicing techniques
- Apply vectorization to eliminate explicit loops and write efficient scientific code
- Master broadcasting rules to perform operations on arrays of different shapes elegantly
- Use NumPy's mathematical functions for scientific calculations
- Understand memory layout and its impact on performance
- Debug common NumPy errors and understand when operations create copies vs views
- Choose when NumPy is appropriate versus other tools like pandas or sparse matrices
- Integrate NumPy with the scientific Python ecosystem

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Work with Python lists and understand indexing/slicing (Chapter 4)
- ✓ Write functions and understand scope (Chapter 5)
- ✓ Understand object methods and attributes (Chapter 6)
- ✓ Use list comprehensions for data transformation (Chapter 4)
- ✓ Work with nested data structures (Chapter 4)

## Chapter Overview

So far, you've been using Python lists for numerical data. But try this simple experiment: compute the sum of squares for a million numbers using a list comprehension versus a loop. Even with list comprehensions, Python is surprisingly slow for numerical work. This is where NumPy transforms Python from a general-purpose language into a scientific computing powerhouse.

NumPy (Numerical Python) is not just a library—it's the foundation upon which the entire scientific Python ecosystem is built. Every plot you make with Matplotlib, every optimization you run with SciPy, every dataframe you manipulate with Pandas, ultimately relies on NumPy arrays. Understanding NumPy deeply means understanding how scientific computing works in Python.

This chapter reveals why NumPy is fast (hint: it's not written in Python), how its mental model differs from pure Python (vectorization over loops), and how its design patterns appear throughout scientific computing. You'll learn to think in arrays, not elements—a fundamental shift that makes the difference between code that takes hours and code that takes seconds. By the end, you'll understand why that `Star` class you created in Chapter 6 might be better represented as a structured NumPy array when you have millions of stars to process.

## 7.1 Why NumPy? The Performance Revolution

Let's start with a motivating example that shows why NumPy exists:

```python
In [1]: import time
In [2]: import numpy as np

# Pure Python: sum of squares for 1 million numbers
In [3]: def python_sum_of_squares(n):
   ...:     """Pure Python implementation using list comprehension."""
   ...:     numbers = list(range(n))
   ...:     return sum(x**2 for x in numbers)

# NumPy: same calculation
In [4]: def numpy_sum_of_squares(n):
   ...:     """NumPy implementation using vectorization."""
   ...:     numbers = np.arange(n)
   ...:     return np.sum(numbers**2)

# Time both approaches
In [5]: n = 1_000_000

In [6]: start = time.perf_counter()
In [7]: python_result = python_sum_of_squares(n)
In [8]: python_time = time.perf_counter() - start

In [9]: start = time.perf_counter()
In [10]: numpy_result = numpy_sum_of_squares(n)
In [11]: numpy_time = time.perf_counter() - start

In [12]: print(f"Python: {python_time:.3f} seconds")
In [13]: print(f"NumPy:  {numpy_time:.3f} seconds")
In [14]: print(f"Speedup: {python_time/numpy_time:.1f}x")
Python: 0.142 seconds
NumPy:  0.003 seconds
Speedup: 47.3x

In [15]: python_result == numpy_result  # Same answer!
Out[15]: True
```

NumPy is nearly 50 times faster! But why? The answer reveals fundamental truths about scientific computing.

### The Secret: NumPy Arrays Are Not Python Lists

Understanding the fundamental difference between Python lists and NumPy arrays is crucial for writing efficient scientific code. Python lists are incredibly flexible—they can hold any type of object, grow and shrink dynamically, and support arbitrary nesting. This flexibility comes at a cost: each element in a list is actually a pointer to a Python object stored elsewhere in memory. When you perform operations on lists, Python must follow these pointers, check types, and handle each element individually.

NumPy arrays, by contrast, store raw numerical data in contiguous blocks of memory, just like arrays in C or Fortran. All elements must be the same type, and the array size is fixed when created. These restrictions enable dramatic performance improvements.

```{mermaid}
flowchart TD
    subgraph "Python List"
        L[List Object] --> P1[Pointer 1]
        L --> P2[Pointer 2]
        L --> P3[Pointer 3]
        L --> PN[Pointer N]
        
        P1 --> O1[Integer Object<br/>type: int<br/>value: 0<br/>refcount: 1]
        P2 --> O2[Integer Object<br/>type: int<br/>value: 1<br/>refcount: 1]
        P3 --> O3[Integer Object<br/>type: int<br/>value: 2<br/>refcount: 1]
        PN --> ON[Integer Object<br/>type: int<br/>value: N-1<br/>refcount: 1]
    end
    
    subgraph "NumPy Array"
        A[Array Header<br/>dtype: int64<br/>shape: (N,)<br/>strides: (8,)] --> M[Contiguous Memory Block<br/>0 | 1 | 2 | 3 | ... | N-1]
    end
    
    style L fill:#f9f
    style A fill:#9f9
    style M fill:#9ff
```

With Python lists, accessing an element means following a pointer, checking the object type, extracting the value, and potentially handling reference counting. With NumPy arrays, accessing an element is just reading from a memory offset—the same operation that happens in compiled languages. This difference becomes dramatic when operating on millions of elements.

### The Mental Model Shift: Vectorization

The performance gain from NumPy requires a different programming paradigm called vectorization. Instead of thinking about operations on individual elements (the Python way), you think about operations on entire arrays (the NumPy way). This isn't just a syntactic difference—it's a fundamental shift in how you approach problems.

```python
# Python style: loop over elements explicitly
def python_distance(x_coords, y_coords):
    """
    Calculate distances from origin using Python loops.
    Note: We process each coordinate pair individually.
    """
    distances = []
    for x, y in zip(x_coords, y_coords):
        dist = (x**2 + y**2)**0.5
        distances.append(dist)
    return distances

# NumPy style: operate on entire arrays at once
def numpy_distance(x_coords, y_coords):
    """
    Calculate distances from origin using vectorization.
    The entire operation happens in compiled C code.
    """
    return np.sqrt(x_coords**2 + y_coords**2)

# Test with 100,000 points
n_points = 100_000
x = np.random.randn(n_points)  # Random x coordinates
y = np.random.randn(n_points)  # Random y coordinates

# Convert to lists for Python version
x_list = x.tolist()
y_list = y.tolist()

# Time both approaches
%timeit python_distance(x_list, y_list)
# 31.2 ms ± 501 µs per loop

%timeit numpy_distance(x, y)
# 371 µs ± 5.2 µs per loop

# 84x faster with vectorization!
```

Vectorization means the loop still happens, but it's implemented in compiled C code rather than interpreted Python. The CPU can also use SIMD (Single Instruction, Multiple Data) instructions to process multiple array elements simultaneously, further improving performance.

### 📦 **Computational Thinking Box: The Two-Language Problem**

```
PATTERN: The Two-Language Problem in Scientific Computing

Many scientific computing ecosystems face a fundamental dilemma:
- High-level languages (Python, MATLAB, R) are great for experimentation
- Low-level languages (C, Fortran) are needed for performance
- Scientists want to think about science, not memory management

NumPy's Solution:
- Python interface for thinking and prototyping
- C/Fortran implementation for computation
- Seamless boundary between the two worlds

This pattern appears throughout scientific Python:
- NumPy: Python interface, C implementation
- SciPy: Python interface, Fortran/C++ implementation  
- Pandas: Python interface, Cython implementation
- Scikit-learn: Python interface, Cython/C++ implementation

The key insight: put the language boundary at the right abstraction level.
For NumPy, that's the array operation, not the element operation.
This lets scientists write Python while getting C performance.
```

## 7.2 Creating Arrays: From Lists to Grids

NumPy provides many ways to create arrays, each optimized for different use cases. Understanding these is crucial for efficient scientific computing.

### From Python Sequences

The most straightforward way to create NumPy arrays is by converting existing Python data structures:

```python
In [16]: # From a simple list
In [17]: list_data = [1, 2, 3, 4, 5]
In [18]: arr = np.array(list_data)
In [19]: print(f"Array: {arr}")
In [20]: print(f"Type: {type(arr)}")  # Note: it's an object!
In [21]: print(f"Dtype: {arr.dtype}")  # Data type of elements
Array: [1 2 3 4 5]
Type: <class 'numpy.ndarray'>
Dtype: int64

In [22]: # From nested lists (creates 2D array)
In [23]: matrix_data = [[1, 2, 3],
   ...:                 [4, 5, 6],
   ...:                 [7, 8, 9]]
In [24]: matrix = np.array(matrix_data)
In [25]: print(f"Matrix:\n{matrix}")
In [26]: print(f"Shape: {matrix.shape}")  # (rows, columns)
In [27]: print(f"Dimensions: {matrix.ndim}")
Matrix:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Shape: (3, 3)
Dimensions: 2
```

Remember from Chapter 6 that NumPy arrays are objects! They have attributes (`shape`, `dtype`, `size`) and methods (`reshape()`, `mean()`, `sum()`). This is object-oriented programming in action—the array object encapsulates both data and operations on that data.

### Initialization Functions

Creating arrays from scratch is often more efficient than converting lists, especially for large arrays:

```python
In [28]: # Arrays of zeros - useful for accumulating results
In [29]: zeros = np.zeros((3, 4))  # 3 rows, 4 columns
In [30]: print(f"Zeros:\n{zeros}")
Zeros:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

In [31]: # Arrays of ones - useful for counting or normalization
In [32]: ones = np.ones((2, 3), dtype=np.int32)  # Can specify dtype
In [33]: print(f"Ones:\n{ones}")
Ones:
[[1 1 1]
 [1 1 1]]

In [34]: # Identity matrix - useful for linear algebra
In [35]: identity = np.eye(3)
In [36]: print(f"Identity:\n{identity}")
Identity:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

In [37]: # Uninitialized array - fastest but DANGEROUS
In [38]: empty = np.empty((2, 2))  # Contains garbage values!
In [39]: print(f"Empty (undefined values):\n{empty}")
Empty (undefined values):
[[4.67e-310 0.00e+000]
 [0.00e+000 0.00e+000]]
```

### ⚠️ **Common Bug Alert: Uninitialized Arrays**

```python
# WRONG: Assuming empty arrays contain zeros
def calculate_sums_wrong(data, n_bins):
    """This function has a subtle bug."""
    sums = np.empty(n_bins)  # Contains garbage values!
    for i, value in enumerate(data):
        bin_idx = int(value) % n_bins
        sums[bin_idx] += value  # Adding to garbage!
    return sums

# CORRECT: Use zeros for accumulation
def calculate_sums_correct(data, n_bins):
    """Always initialize accumulators to zero."""
    sums = np.zeros(n_bins)  # Properly initialized
    for i, value in enumerate(data):
        bin_idx = int(value) % n_bins
        sums[bin_idx] += value  # Now safe to accumulate
    return sums

# The bug might not be obvious in testing!
test_data = np.array([1.5, 2.7, 3.2])
print(calculate_sums_wrong(test_data, 5))   # Unpredictable results!
print(calculate_sums_correct(test_data, 5))  # [0, 1.5, 2.7, 3.2, 0]
```

Always use `zeros()` for accumulation, `ones()` for counting, and only use `empty()` when you'll immediately overwrite all values. The performance gain from `empty()` is rarely worth the risk of bugs.

### Range Arrays

For sequences of numbers, NumPy provides optimized functions that are much more memory-efficient than converting Python ranges:

```python
In [40]: # Like Python's range, but returns an array
In [41]: integers = np.arange(10)  # 0 to 9
In [42]: print(f"Integers: {integers}")
Integers: [0 1 2 3 4 5 6 7 8 9]

In [43]: # With start, stop, step (half-open interval like Python)
In [44]: evens = np.arange(0, 10, 2)
In [45]: print(f"Evens: {evens}")
Evens: [0 2 4 6 8]

In [46]: # Floating-point ranges (be careful with precision!)
In [47]: floats = np.arange(0, 1, 0.1)
In [48]: print(f"Floats: {floats}")
In [49]: print(f"Length: {len(floats)}")  # Might not be what you expect!
Floats: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
Length: 10  # Note: doesn't include 1.0!

In [50]: # Linear spacing - specify number of points instead of step
In [51]: linear = np.linspace(0, 1, 11)  # 11 points from 0 to 1 inclusive
In [52]: print(f"Linear: {linear}")
Linear: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

In [53]: # Logarithmic spacing - for log-scale plots or sampling
In [54]: logarithmic = np.logspace(0, 3, 4)  # 10^0 to 10^3
In [55]: print(f"Logarithmic: {logarithmic}")
Logarithmic: [   1.   10.  100. 1000.]
```

### 🔍 **Check Your Understanding**

What's the difference between `np.arange(0, 1, 0.1)` and `np.linspace(0, 1, 11)`?

<details>
<summary>Answer</summary>

Both create arrays from 0 to 1, but they work fundamentally differently and this matters for scientific computing:

`np.arange(0, 1, 0.1)` uses a step size of 0.1, similar to a for loop with floating-point increment. Due to floating-point arithmetic limitations, this can accumulate rounding errors and might not include exactly 1.0. The exact number of points depends on floating-point precision.

`np.linspace(0, 1, 11)` creates exactly 11 evenly spaced points including both endpoints. It calculates the spacing to ensure exact endpoints and uniform distribution. This is more numerically stable and predictable.

```python
# Demonstration of the subtle but important difference
arange_arr = np.arange(0, 1, 0.1)
linspace_arr = np.linspace(0, 1, 11)

print(f"arange length: {len(arange_arr)}")      # 10 (doesn't include 1.0)
print(f"linspace length: {len(linspace_arr)}")  # 11 (includes both endpoints)
print(f"arange last: {arange_arr[-1]}")         # 0.9
print(f"linspace last: {linspace_arr[-1]}")     # 1.0

# Floating-point precision issues with arange
step = 0.1
accumulated = 0.0
for i in range(3):
    accumulated += step
print(f"0.1 + 0.1 + 0.1 = {accumulated}")  # 0.30000000000000004 (not 0.3!)
```

Use `linspace` when you need a specific number of points including endpoints (common in plotting and interpolation). Use `arange` for integer sequences or when you need a specific step size and can tolerate floating-point imprecision.

</details>

### Random Arrays

Scientific computing often needs random data for Monte Carlo simulations, statistical sampling, or algorithm testing:

```python
In [56]: # ALWAYS set seed for reproducibility in scientific work!
In [57]: np.random.seed(42)

In [58]: # Uniform distribution [0, 1)
In [59]: uniform = np.random.rand(3, 3)
In [60]: print(f"Uniform:\n{uniform}")
Uniform:
[[0.374 0.950 0.731]
 [0.598 0.156 0.155]
 [0.058 0.866 0.601]]

In [61]: # Standard normal distribution (mean=0, std=1)
In [62]: normal = np.random.randn(3, 3)
In [63]: print(f"Normal:\n{normal}")
Normal:
[[ 0.708 -0.757 -1.316]
 [ 0.386  1.749  0.297]
 [-0.814 -0.454 -1.150]]

In [64]: # Random integers for discrete problems
In [65]: integers = np.random.randint(0, 10, size=(2, 4))
In [66]: print(f"Random integers:\n{integers}")
Random integers:
[[7 6 6 8]
 [8 3 9 8]]

In [67]: # Poisson distribution for photon counting
In [68]: # Mean photon count = 5 photons per pixel
In [69]: photon_counts = np.random.poisson(lam=5, size=10)
In [70]: print(f"Photon counts: {photon_counts}")
Photon counts: [3 5 6 3 8 4 3 3 6 2]
```

Note: NumPy 1.17+ introduced a new random API with better practices for parallel computing:
```python
# Modern approach (recommended for new code)
rng = np.random.default_rng(seed=42)  # Create generator
data = rng.standard_normal((3, 3))    # Use generator methods
```

## 7.3 Array Attributes and Memory Layout

Understanding array attributes and memory layout is crucial for writing efficient code and debugging unexpected behavior.

### Essential Attributes

Every NumPy array is an object with attributes that completely describe its structure:

```python
In [71]: # Create a 3D array for demonstration
In [72]: arr = np.random.randn(2, 3, 4)  # 2 blocks, 3 rows, 4 columns each

In [73]: print(f"Shape: {arr.shape}")        # Dimensions (most important!)
In [74]: print(f"Size: {arr.size}")          # Total number of elements
In [75]: print(f"Ndim: {arr.ndim}")          # Number of dimensions
In [76]: print(f"Dtype: {arr.dtype}")        # Data type of elements
In [77]: print(f"Itemsize: {arr.itemsize}")  # Bytes per element
In [78]: print(f"Nbytes: {arr.nbytes}")      # Total memory usage
Shape: (2, 3, 4)
Size: 24
Ndim: 3
Dtype: float64
Itemsize: 8
Nbytes: 192

In [79]: # Memory layout information (advanced but important)
In [80]: print(f"Strides: {arr.strides}")  # Bytes to jump for next element
In [81]: print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
In [82]: print(f"Fortran-contiguous: {arr.flags['F_CONTIGUOUS']}")
Strides: (96, 32, 8)  # Jump 96 bytes for next block, 32 for next row, 8 for next element
C-contiguous: True
Fortran-contiguous: False
```

### Memory Layout: Row-Major vs Column-Major

NumPy can store multidimensional arrays in different memory layouts. Understanding this is crucial for performance when working with large datasets or interfacing with other languages:

```{mermaid}
flowchart LR
    subgraph "Row-Major (C-style, NumPy default)"
        RM[2D Array<br/>[[1,2,3],<br/>[4,5,6]]] --> RMM[Memory: 1|2|3|4|5|6]
        RMM --> RMD[Traverse rows first]
    end
    
    subgraph "Column-Major (Fortran-style)"
        CM[2D Array<br/>[[1,2,3],<br/>[4,5,6]]] --> CMM[Memory: 1|4|2|5|3|6]
        CMM --> CMD[Traverse columns first]
    end
    
    style RM fill:#9f9
    style CM fill:#f9f
```

```python
In [83]: # Default is C-order (row-major) - rows are contiguous
In [84]: c_array = np.array([[1, 2, 3],
   ...:                       [4, 5, 6]])
In [85]: print(f"C-order strides: {c_array.strides}")
C-order strides: (24, 8)  # 24 bytes to next row (3 elements × 8 bytes)

In [86]: # Can create Fortran-order (column-major) - columns are contiguous
In [87]: f_array = np.array([[1, 2, 3],
   ...:                       [4, 5, 6]], order='F')
In [88]: print(f"F-order strides: {f_array.strides}")
F-order strides: (8, 16)  # 8 bytes to next row (1 element × 8 bytes)

In [89]: # Performance implications: access contiguous data when possible
In [90]: large = np.random.randn(1000, 1000)

In [91]: # Summing along rows (axis=1) is fast for C-order
In [92]: # because we read memory sequentially
In [93]: %timeit large.sum(axis=1)
574 µs ± 12.3 µs per loop

In [94]: # Summing along columns (axis=0) is slower for C-order
In [95]: # because we jump around in memory
In [96]: %timeit large.sum(axis=0)
1.28 ms ± 23.4 µs per loop  # 2x slower!

# Why? CPU cache works best with sequential memory access
```

### Data Types and Memory Usage

NumPy provides precise control over data types, crucial for memory efficiency and numerical precision in scientific computing:

```python
In [97]: # Integer types with different ranges and memory usage
In [98]: int8 = np.array([1, 2, 3], dtype=np.int8)    # -128 to 127
In [99]: int16 = np.array([1, 2, 3], dtype=np.int16)  # -32,768 to 32,767
In [100]: int32 = np.array([1, 2, 3], dtype=np.int32)  # ~±2 billion
In [101]: int64 = np.array([1, 2, 3], dtype=np.int64)  # ~±9 quintillion

In [102]: print(f"int8 uses {int8.nbytes} bytes for 3 elements")
In [103]: print(f"int64 uses {int64.nbytes} bytes for 3 elements")
int8 uses 3 bytes for 3 elements
int64 uses 24 bytes for 3 elements  # 8x more memory!

In [104]: # Floating-point types - precision vs memory tradeoff
In [105]: float16 = np.array([1.0, 2.0], dtype=np.float16)  # Half precision
In [106]: float32 = np.array([1.0, 2.0], dtype=np.float32)  # Single precision
In [107]: float64 = np.array([1.0, 2.0], dtype=np.float64)  # Double precision

In [108]: # Complex numbers for signal processing or quantum mechanics
In [109]: complex_arr = np.array([1+2j, 3+4j], dtype=np.complex128)
In [110]: print(f"Complex array: {complex_arr}")
In [111]: print(f"Real parts: {complex_arr.real}")
In [112]: print(f"Imaginary parts: {complex_arr.imag}")
Complex array: [1.+2.j 3.+4.j]
Real parts: [1. 3.]
Imaginary parts: [2. 4.]
```

### 🔊 **Performance Profile: Data Type Impact**

```python
# Memory and speed tradeoffs with different dtypes
n = 10_000_000  # 10 million elements

# Create arrays with different precision
float64_arr = np.random.randn(n)  # Default double precision
float32_arr = float64_arr.astype(np.float32)  # Single precision
float16_arr = float64_arr.astype(np.float16)  # Half precision

print(f"float64: {float64_arr.nbytes / 1e6:.1f} MB")
print(f"float32: {float32_arr.nbytes / 1e6:.1f} MB")
print(f"float16: {float16_arr.nbytes / 1e6:.1f} MB")
# Output:
# float64: 80.0 MB
# float32: 40.0 MB  
# float16: 20.0 MB

# Performance comparison
%timeit float64_arr.sum()  # 7.92 ms
%timeit float32_arr.sum()  # 3.96 ms (2x faster!)
%timeit float16_arr.sum()  # 15.8 ms (slower - limited hardware support)

# But beware precision loss!
large_number = 1e10
small_number = 1.0
print(f"float64: {large_number + small_number}")  # 10000000001.0 (correct)
print(f"float32: {np.float32(large_number) + np.float32(small_number)}")  # 10000000000.0 (lost precision!)
```

Choose dtypes based on your scientific requirements: float64 for high precision calculations, float32 for large datasets where some precision loss is acceptable, integers for counting and indexing.

## 7.4 Indexing and Slicing: Views vs Copies

NumPy's indexing is powerful but has subtleties that can cause bugs if not understood properly. The key concept is understanding when NumPy creates a view (shared memory) versus a copy (independent memory).

### Basic Indexing (Creates Views)

Basic slicing with integers and colons creates views that share memory with the original array:

```python
In [113]: # 1D indexing - similar to Python lists
In [114]: arr = np.arange(10)
In [115]: print(f"Original: {arr}")
In [116]: print(f"Element at index 3: {arr[3]}")
In [117]: print(f"Slice [2:5]: {arr[2:5]}")
In [118]: print(f"Every 2nd element: {arr[::2]}")
In [119]: print(f"Reverse: {arr[::-1]}")
Original: [0 1 2 3 4 5 6 7 8 9]
Element at index 3: 3
Slice [2:5]: [2 3 4]
Every 2nd element: [0 2 4 6 8]
Reverse: [9 8 7 6 5 4 3 2 1 0]

In [120]: # CRITICAL: Slices are views, not copies!
In [121]: slice_view = arr[2:5]
In [122]: slice_view[0] = 999  # Modifying the view
In [123]: print(f"Original after modification: {arr}")
Original after modification: [  0   1 999   3   4   5   6   7   8   9]
# The original array changed!
```

### ⚠️ **Common Bug Alert: Unexpected Mutation**

```python
# DANGEROUS: Functions that modify views change the original!
def process_middle(data):
    """Process middle section of data - has a hidden side effect!"""
    middle = data[len(data)//4:3*len(data)//4]  # This is a view!
    middle *= 2  # This modifies the original array!
    return middle

original = np.arange(10)
print(f"Before: {original}")
result = process_middle(original)
print(f"After: {original}")  # Original is changed unexpectedly!
# Before: [0 1 2 3 4 5 6 7 8 9]
# After: [0 1 4 6 8 5 6 7 8 9]

# SAFE: Explicitly copy when you need independence
def process_middle_safe(data):
    """Process middle section without side effects."""
    middle = data[len(data)//4:3*len(data)//4].copy()  # Explicit copy
    middle *= 2  # Only affects the copy
    return middle
```

### Multidimensional Indexing

For 2D arrays and higher dimensions, indexing becomes more sophisticated:

```python
In [124]: # Create a 2D array (matrix)
In [125]: matrix = np.array([[1, 2, 3],
    ...:                      [4, 5, 6],
    ...:                      [7, 8, 9]])

In [126]: # Single element access
In [127]: print(f"Element at row 1, column 2: {matrix[1, 2]}")
Element at row 1, column 2: 6

In [128]: # Entire row or column extraction
In [129]: print(f"Row 1: {matrix[1, :]}")     # Can also write matrix[1]
In [130]: print(f"Column 2: {matrix[:, 2]}")
Row 1: [4 5 6]
Column 2: [3 6 9]

In [131]: # Submatrix extraction
In [132]: print(f"Top-left 2x2 submatrix:\n{matrix[:2, :2]}")
Top-left 2x2 submatrix:
[[1 2]
 [4 5]]

In [133]: # Strided access for sampling
In [134]: print(f"Every other element:\n{matrix[::2, ::2]}")
Every other element:
[[1 3]
 [7 9]]
```

### Fancy Indexing (Creates Copies)

Using arrays or lists as indices creates copies, not views. This is called "fancy indexing":

```python
In [135]: arr = np.arange(10) * 10  # [0, 10, 20, ..., 90]

In [136]: # Integer array indexing
In [137]: indices = np.array([1, 3, 5])
In [138]: selected = arr[indices]  # This is a COPY!
In [139]: print(f"Selected elements: {selected}")
Selected elements: [10 30 50]

In [140]: selected[0] = 999  # Modify the copy
In [141]: print(f"Original unchanged: {arr}")
Original unchanged: [ 0 10 20 30 40 50 60 70 80 90]

In [142]: # Boolean indexing (masking) - also creates copies
In [143]: mask = arr > 40
In [144]: print(f"Boolean mask: {mask}")
In [145]: filtered = arr[mask]  # Copy of elements where mask is True
In [146]: print(f"Filtered elements: {filtered}")
Boolean mask: [False False False False False  True  True  True  True  True]
Filtered elements: [50 60 70 80 90]

In [147]: # Combining conditions with & (and), | (or), ~ (not)
In [148]: # Note: Use &, not 'and' for element-wise operations
In [149]: complex_mask = (arr > 20) & (arr < 70)
In [150]: print(f"Complex filter result: {arr[complex_mask]}")
Complex filter result: [30 40 50 60]
```

### 📦 **Computational Thinking Box: Views vs Copies**

```
PATTERN: Memory Efficiency Through Views

Views are NumPy's mechanism for providing different perspectives 
on the same underlying data without copying it. This pattern is 
crucial for both memory efficiency and performance.

When NumPy creates views (shares memory):
- Basic slicing: arr[1:5], arr[:, 2], arr[::2]
- Reshaping: arr.reshape(new_shape)
- Transposing: arr.T
- Type casting sometimes: arr.view(new_dtype)

When NumPy creates copies (independent memory):
- Fancy indexing: arr[[1,3,5]], arr[arr > 0]
- Explicit copy: arr.copy()
- Operations that change size: arr.flatten()

Testing if something is a view:
    if arr.base is not None:
        print("arr is a view of", arr.base)
    else:
        print("arr owns its data")

This pattern appears throughout scientific computing:
- Pandas DataFrames (often views of NumPy arrays)
- Memory-mapped files (views of disk data)
- GPU computing (minimizing expensive memory transfers)

Understanding views vs copies helps you:
1. Avoid unexpected data modification
2. Minimize memory usage with large datasets
3. Write more efficient algorithms
```

## 7.5 Vectorization: Thinking in Arrays

Vectorization is the key to NumPy's performance and elegance. It means expressing operations on entire arrays rather than individual elements, pushing loops into compiled code.

### Universal Functions (ufuncs)

NumPy provides "universal functions" that operate element-wise on arrays with optimized C implementations:

```python
In [151]: # Arithmetic operations are vectorized
In [152]: a = np.array([1, 2, 3, 4])
In [153]: b = np.array([10, 20, 30, 40])

In [154]: # These operations happen in parallel in C
In [155]: print(f"Addition: {a + b}")
In [156]: print(f"Multiplication: {a * b}")
In [157]: print(f"Power: {a ** 2}")
Addition: [11 22 33 44]
Multiplication: [10 40 90 160]
Power: [ 1  4  9 16]

In [158]: # Mathematical functions are vectorized
In [159]: angles = np.array([0, np.pi/4, np.pi/2, np.pi])
In [160]: print(f"Sin: {np.sin(angles)}")
In [161]: print(f"Cos: {np.cos(angles)}")
Sin: [0.000e+00 7.071e-01 1.000e+00 1.225e-16]
Cos: [ 1.000e+00  7.071e-01  6.123e-17 -1.000e+00]

In [162]: # Comparison operations return boolean arrays
In [163]: arr = np.arange(5)
In [164]: print(f"Greater than 2: {arr > 2}")
In [165]: print(f"Equal to 3: {arr == 3}")
Greater than 2: [False False False  True  True]
Equal to 3: [False False False  True False]
```

### Vectorizing Custom Functions

You can vectorize your own functions, though true vectorization (using NumPy operations throughout) is faster than using `np.vectorize`:

```python
In [166]: # Example: photon energy from wavelength
In [167]: def photon_energy_scalar(wavelength_nm):
    ...:     """
    ...:     Calculate photon energy in eV from wavelength in nm.
    ...:     E = hc/λ where h is Planck constant, c is speed of light
    ...:     """
    ...:     h = 4.135667e-15  # Planck constant in eV·s
    ...:     c = 2.998e17      # Speed of light in nm/s
    ...:     return h * c / wavelength_nm

In [168]: # Works on single values
In [169]: print(f"Energy at 500nm: {photon_energy_scalar(500):.3f} eV")
Energy at 500nm: 2.480 eV

In [170]: # np.vectorize for convenience (but not optimal performance)
In [171]: photon_energy_vec = np.vectorize(photon_energy_scalar)
In [172]: wavelengths = np.array([400, 500, 600, 700])  # nm
In [173]: print(f"Energies: {photon_energy_vec(wavelengths)}")
Energies: [3.099 2.480 2.066 1.771]

In [174]: # Better: write truly vectorized code using NumPy operations
In [175]: def photon_energy_fast(wavelength_nm):
    ...:     """Truly vectorized version - works on arrays natively."""
    ...:     h = 4.135667e-15  # eV·s
    ...:     c = 2.998e17      # nm/s
    ...:     return h * c / wavelength_nm  # NumPy handles arrays automatically

In [176]: # Performance comparison
In [177]: large_wavelengths = np.random.uniform(300, 800, 100000)
In [178]: %timeit photon_energy_vec(large_wavelengths)   # 35.2 ms
In [179]: %timeit photon_energy_fast(large_wavelengths)  # 326 µs
# True vectorization is 100x faster!
```

### Aggregation Functions

Aggregations reduce arrays to scalars or smaller arrays, with optimized implementations for common operations:

```python
In [180]: # Generate sample data
In [181]: data = np.random.randn(1000)

In [182]: # Basic statistics - all optimized C implementations
In [183]: print(f"Mean: {data.mean():.4f}")
In [184]: print(f"Standard deviation: {data.std():.4f}")
In [185]: print(f"Min: {data.min():.4f}, Max: {data.max():.4f}")
In [186]: print(f"Median: {np.median(data):.4f}")
Mean: -0.0234
Standard deviation: 0.9897
Min: -3.2384, Max: 3.0234
Median: -0.0365

In [187]: # Percentiles for outlier detection
In [188]: print(f"5th percentile: {np.percentile(data, 5):.4f}")
In [189]: print(f"95th percentile: {np.percentile(data, 95):.4f}")
5th percentile: -1.6422
95th percentile: 1.5967

In [190]: # Aggregation along specific axes for multidimensional arrays
In [191]: matrix = np.random.randn(3, 4)
In [192]: print(f"Matrix:\n{matrix}")
In [193]: print(f"Column means (axis=0): {matrix.mean(axis=0)}")
In [194]: print(f"Row means (axis=1): {matrix.mean(axis=1)}")
Matrix:
[[-0.245  1.234 -0.567  0.891]
 [ 2.345 -1.234  0.123 -0.456]
 [ 0.789 -0.012  1.234 -2.345]]
Column means (axis=0): [ 0.963 -0.004  0.263 -0.637]
Row means (axis=1): [ 0.328  0.195 -0.084]
```

### 🔍 **Check Your Understanding**

Given a 2D array representing an image, how would you normalize it so all values are between 0 and 1?

<details>
<summary>Answer</summary>

There are several normalization approaches depending on your scientific requirements:

```python
# Create sample "image" data
image = np.random.randn(100, 100) * 50 + 128  # Centered at 128, std=50

# Method 1: Min-Max normalization (scales to exact [0, 1])
def min_max_normalize(arr):
    """
    Scale array to [0, 1] range.
    Good for: display, when you need exact bounds
    """
    return (arr - arr.min()) / (arr.max() - arr.min())

normalized1 = min_max_normalize(image)
print(f"Range: [{normalized1.min():.3f}, {normalized1.max():.3f}]")  # [0.000, 1.000]

# Method 2: Clipping to known range (e.g., 0-255 for 8-bit images)
def clip_normalize(arr, min_val=0, max_val=255):
    """
    Clip to range then normalize.
    Good for: when you know the expected data range
    """
    clipped = np.clip(arr, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)

normalized2 = clip_normalize(image, 0, 255)

# Method 3: Z-score normalization (standardization)
def z_score_normalize(arr):
    """
    Standardize to mean=0, std=1.
    Good for: machine learning, statistical analysis
    Note: doesn't guarantee [0,1] range!
    """
    return (arr - arr.mean()) / arr.std()

standardized = z_score_normalize(image)
print(f"Mean: {standardized.mean():.6f}, Std: {standardized.std():.6f}")

# Choose based on your scientific needs!
# - Min-max for display (guarantees [0,1])
# - Clipping when you know valid data range
# - Z-score for statistical processing
```

The key insight: vectorized operations make this efficient even for large images. No loops needed!

</details>

## 7.6 Broadcasting: NumPy's Superpower

Broadcasting allows NumPy to perform operations on arrays of different shapes without explicit loops or data copying. It's one of NumPy's most powerful and elegant features.

### The Broadcasting Rules

Broadcasting follows strict rules to determine how arrays of different shapes can be combined. Understanding these rules is essential for writing efficient NumPy code:

```{mermaid}
flowchart TD
    A[Arrays A and B] --> B{Compare shapes<br/>right to left}
    B --> C{Dimensions<br/>equal?}
    C -->|Yes| D[Compatible]
    C -->|No| E{One dimension<br/>is 1?}
    E -->|Yes| F[Broadcast:<br/>stretch size-1 dimension]
    E -->|No| G[Error!<br/>Cannot broadcast]
    
    F --> H[Perform operation]
    D --> H
    
    style D fill:#9f9
    style F fill:#9ff
    style G fill:#f99
```

The rules are:
1. Compare shapes element-wise starting from the rightmost dimension
2. Two dimensions are compatible if they're equal or one is 1
3. Arrays with fewer dimensions are padded with 1s on the left
4. After broadcasting, each dimension is the maximum of the input dimensions

```python
In [195]: # Broadcasting examples
In [196]: arr = np.array([[1, 2, 3],
    ...:                   [4, 5, 6],
    ...:                   [7, 8, 9]])

In [197]: # Scalar broadcasting (scalar is treated as shape ())
In [198]: print(f"Array + 10:\n{arr + 10}")
Array + 10:
[[11 12 13]
 [14 15 16]
 [17 18 19]]

In [199]: # 1D array broadcasts to each row
In [200]: row_vector = np.array([100, 200, 300])  # Shape: (3,)
In [201]: print(f"Array + row vector:\n{arr + row_vector}")
Array + row vector:
[[101 202 303]
 [104 205 306]
 [107 208 309]]

In [202]: # Column vector broadcasts to each column
In [203]: col_vector = np.array([[1000],
    ...:                          [2000],
    ...:                          [3000]])  # Shape: (3, 1)
In [204]: print(f"Array + column vector:\n{arr + col_vector}")
Array + column vector:
[[1001 1002 1003]
 [2004 2005 2006]
 [3007 3008 3009]]
```

### Practical Broadcasting Examples

Broadcasting makes many scientific calculations elegant and efficient. An important note: broadcasting doesn't actually copy data in memory - it creates sophisticated views with different strides, making it memory-efficient even for large arrays. This means you can "stretch" a small array to match a large one without memory concerns.

```python
In [205]: # Example: Normalize each column of a matrix independently
In [206]: # Common in machine learning preprocessing
In [207]: data = np.random.randn(100, 3) * [10, 50, 100] + [0, 100, 200]
In [208]: print(f"Original means: {data.mean(axis=0)}")
In [209]: print(f"Original stds: {data.std(axis=0)}")
Original means: [  0.234  99.876 200.123]
Original stds: [ 9.987 49.234 98.765]

In [210]: # Subtract mean and divide by std for each column
In [211]: # Broadcasting handles the dimension mismatch automatically
In [212]: normalized = (data - data.mean(axis=0)) / data.std(axis=0)
In [213]: print(f"Normalized means: {normalized.mean(axis=0)}")  # Should be ~0
In [214]: print(f"Normalized stds: {normalized.std(axis=0)}")    # Should be ~1
Normalized means: [-1.23e-17  2.45e-17  3.67e-17]
Normalized stds: [1. 1. 1.]

In [215]: # Example: Distance matrix between points
In [216]: # Calculate all pairwise distances efficiently
In [217]: points = np.random.randn(5, 2)  # 5 points in 2D
In [218]: 
In [219]: # Use broadcasting to compute all pairwise differences
In [220]: # Reshape for broadcasting: (5,1,2) - (1,5,2) -> (5,5,2)
In [221]: diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
In [222]: distances = np.sqrt((diff**2).sum(axis=2))
In [223]: print(f"Distance matrix shape: {distances.shape}")
In [224]: print(f"Distance from point 0 to point 1: {distances[0,1]:.3f}")
Distance matrix shape: (5, 5)
Distance from point 0 to point 1: 1.234
```

### ⚠️ **Common Bug Alert: Broadcasting Surprises**

```python
# UNEXPECTED: Broadcasting can hide dimension mismatches
a = np.array([[1, 2, 3]])     # Shape: (1, 3)
b = np.array([[10], [20]])    # Shape: (2, 1)

# This works but might not be what you intended!
try:
    result = a + b  # Broadcasts to (2, 3)
    print(f"Unexpected broadcasting result:\n{result}")
except ValueError as e:
    print(f"Error: {e}")
# Output:
# [[11 12 13]
#  [21 22 23]]

# DEFENSIVE: Check shapes when unsure
def safe_add(a, b):
    """Add arrays with shape checking."""
    if a.shape != b.shape:
        print(f"Warning: Broadcasting {a.shape} and {b.shape}")
        result_shape = np.broadcast_shapes(a.shape, b.shape)
        print(f"Result will have shape: {result_shape}")
    return a + b

# EXPLICIT: Use np.newaxis to be clear about intent
row = np.array([1, 2, 3])
col = np.array([10, 20])

# Make broadcasting explicit and intentional
result = row[np.newaxis, :] + col[:, np.newaxis]
print(f"Explicit broadcasting result shape: {result.shape}")
```

## 7.7 Mathematical Operations and Linear Algebra

NumPy provides comprehensive mathematical functions optimized for arrays, from basic arithmetic to sophisticated linear algebra operations.

### Element-wise Mathematics

All standard mathematical functions are available and vectorized:

```python
In [225]: # Trigonometric functions
In [226]: angles = np.linspace(0, 2*np.pi, 5)
In [227]: print(f"Angles (radians): {angles}")
In [228]: print(f"Sin: {np.sin(angles)}")
In [229]: print(f"Arcsin of 0.5: {np.arcsin(0.5)} radians")
Angles (radians): [0.    1.571 3.142 4.712 6.283]
Sin: [ 0.000e+00  1.000e+00  1.225e-16 -1.000e+00 -2.449e-16]
Arcsin of 0.5: 0.524 radians

In [230]: # Exponential and logarithmic functions
In [231]: x = np.array([1, 2, 3])
In [232]: print(f"Exp(x): {np.exp(x)}")         # e^x
In [233]: print(f"Log(x): {np.log(x)}")         # Natural log
In [234]: print(f"Log10(x): {np.log10(x)}")     # Base-10 log
In [235]: print(f"2^x: {np.exp2(x)}")           # 2^x for information theory
Exp(x): [ 2.718  7.389 20.086]
Log(x): [0.    0.693 1.099]
Log10(x): [0.    0.301 0.477]
2^x: [2. 4. 8.]
```

### Linear Algebra Operations

NumPy includes a comprehensive linear algebra module crucial for scientific computing:

```python
In [236]: # Matrix multiplication - different from element-wise!
In [237]: A = np.array([[1, 2],
    ...:                [3, 4]])
In [238]: B = np.array([[5, 6],
    ...:                [7, 8]])

In [239]: # Element-wise multiplication (Hadamard product)
In [240]: print(f"Element-wise A * B:\n{A * B}")
Element-wise A * B:
[[ 5 12]
 [21 32]]

In [241]: # True matrix multiplication
In [242]: print(f"Matrix multiplication A @ B:\n{A @ B}")
In [243]: # Also: np.dot(A, B) or np.matmul(A, B)
Matrix multiplication A @ B:
[[19 22]
 [43 50]]

In [244]: # Essential linear algebra operations
In [245]: matrix = np.array([[3, 1],
    ...:                      [1, 2]])

In [246]: # Determinant
In [247]: det = np.linalg.det(matrix)
In [248]: print(f"Determinant: {det:.3f}")
Determinant: 5.000

In [249]: # Eigenvalues and eigenvectors
In [250]: eigenvalues, eigenvectors = np.linalg.eig(matrix)
In [251]: print(f"Eigenvalues: {eigenvalues}")
In [252]: print(f"Eigenvectors:\n{eigenvectors}")
Eigenvalues: [3.618 1.382]
Eigenvectors:
[[ 0.851 -0.526]
 [ 0.526  0.851]]

In [253]: # Matrix inverse (use with caution!)
In [254]: inverse = np.linalg.inv(matrix)
In [255]: print(f"Inverse:\n{inverse}")
In [256]: print(f"Check A @ A^(-1):\n{matrix @ inverse}")  # Should be identity
Inverse:
[[ 0.4 -0.2]
 [-0.2  0.6]]
Check A @ A^(-1):
[[1. 0.]
 [0. 1.]]
```

### Numerical Stability Considerations

Not all mathematically correct operations are numerically stable. Understanding this is crucial for scientific computing:

```python
In [257]: # Example: Solving linear systems Ax = b
In [258]: A = np.array([[3, 1],
    ...:                [1, 2]])
In [259]: b = np.array([9, 8])

In [260]: # Method 1: Using inverse (NOT RECOMMENDED)
In [261]: x_inverse = np.linalg.inv(A) @ b
In [262]: print(f"Solution using inverse: {x_inverse}")
Solution using inverse: [2. 3.]

In [263]: # Method 2: Using solve (RECOMMENDED)
In [264]: x_solve = np.linalg.solve(A, b)
In [265]: print(f"Solution using solve: {x_solve}")
Solution using solve: [2. 3.]

In [266]: # Why solve is better: check condition number
In [267]: cond = np.linalg.cond(A)
In [268]: print(f"Condition number: {cond:.2f}")
Condition number: 2.62

# Small condition number = stable
# Large condition number (>1000) = potentially unstable

In [269]: # Example of numerical instability
In [270]: # Ill-conditioned matrix (nearly singular)
In [271]: A_bad = np.array([[1.0, 1.0],
    ...:                     [1.0, 1.0000001]])  # Almost singular!
In [272]: print(f"Condition number: {np.linalg.cond(A_bad):.2e}")
Condition number: 4.00e+07  # Huge! Very unstable

# Small input errors lead to large output errors with ill-conditioned matrices
```

### 📦 **Computational Thinking Box: Numerical Stability**

```
PATTERN: Numerical Stability in Scientific Computing

Not all mathematically equivalent formulations are numerically equal.
Floating-point arithmetic has finite precision, and errors accumulate.

Classic Example: Variance calculation
Mathematically: Var(X) = E[X²] - E[X]²
But this can suffer from catastrophic cancellation!

# Naive implementation (unstable)
def variance_naive(x):
    return np.mean(x**2) - np.mean(x)**2

# Stable implementation (what NumPy uses)
def variance_stable(x):
    mean = np.mean(x)
    return np.mean((x - mean)**2)

# Test with data that has large mean, small variance
data = np.random.randn(1000) * 0.01 + 1e6  # Mean=1e6, std=0.01
print(f"Naive: {variance_naive(data):.6f}")    # Can be negative!
print(f"Stable: {variance_stable(data):.6f}")   # Always correct
print(f"NumPy: {np.var(data):.6f}")            # Uses stable algorithm

Key principles:
1. Avoid subtracting large similar numbers
2. Use library functions (they implement stable algorithms)
3. Check condition numbers for linear algebra
4. Be aware of accumulation order for sums
5. Use higher precision when necessary

This is why we use NumPy/SciPy functions instead of 
reimplementing algorithms from textbooks!
```

## 7.8 When NumPy Isn't the Right Tool

While NumPy is powerful, it's important to know when other tools are more appropriate:

### When to Use Other Tools

```python
# 1. HETEROGENEOUS DATA: Use Pandas for mixed types
# NumPy requires homogeneous data
stellar_data_mixed = {
    'name': ['Sirius', 'Vega', 'Altair'],  # Strings
    'magnitude': [-1.46, 0.03, 0.77],       # Floats
    'observed': [True, True, False],        # Booleans
    'notes': ['Binary star', None, 'Fast rotator']  # Mixed
}
# This is awkward in NumPy, natural in Pandas
import pandas as pd
df = pd.DataFrame(stellar_data_mixed)

# 2. SPARSE DATA: Use scipy.sparse for mostly-zero matrices
# NumPy stores all zeros explicitly
from scipy import sparse
# If your matrix is 99% zeros, don't use NumPy!
sparse_matrix = sparse.random(10000, 10000, density=0.01)
print(f"Dense size: {10000*10000*8 / 1e9:.2f} GB")  # 0.80 GB
print(f"Sparse size: ~{0.01*10000*10000*8 / 1e9:.3f} GB")  # ~0.008 GB

# 3. SYMBOLIC MATH: Use SymPy for algebraic manipulation
from sympy import symbols, expand
x, y = symbols('x y')
expression = expand((x + y)**2)  # x**2 + 2*x*y + y**2
# NumPy can't do symbolic algebra!

# 4. GRAPHS/NETWORKS: Use NetworkX for graph algorithms
# NumPy adjacency matrices become unwieldy for graph operations

# 5. VARIABLE-LENGTH SEQUENCES: Use Python lists
# NumPy requires fixed dimensions
variable_sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
]  # Can't efficiently represent in NumPy

# 6. SMALL DATA: Pure Python might be faster!
# NumPy has overhead; for <100 elements, Python can be faster
small_list = [1, 2, 3, 4, 5]
# sum(small_list) might beat np.array(small_list).sum()
```

### Decision Guide

```python
def choose_data_structure(data_characteristics):
    """
    Guide for choosing the right tool for your data.
    """
    if data_characteristics['homogeneous'] and data_characteristics['numerical']:
        if data_characteristics['size'] > 100:
            if data_characteristics['dense']:
                return "NumPy array"
            else:
                return "scipy.sparse matrix"
        else:
            return "Python list might be sufficient"
    elif data_characteristics['tabular'] and data_characteristics['mixed_types']:
        return "Pandas DataFrame"
    elif data_characteristics['symbolic']:
        return "SymPy expressions"
    elif data_characteristics['graph_structure']:
        return "NetworkX graph"
    else:
        return "Python native structures"
```

## 7.9 Advanced Topics (Optional)

The following sections cover specialized NumPy features that you may encounter in existing code or need for specific use cases. Feel free to skip these on first reading and return when you need them.

### Structured Arrays: NumPy's Tabular Data

Before pandas became the standard for tabular data in Python, NumPy provided structured arrays as a way to handle heterogeneous data. While most modern code uses pandas DataFrames for mixed-type tabular data, structured arrays still have specific use cases where they excel.

**When to use structured arrays:**
- You need the absolute minimum memory footprint for millions of records
- You're interfacing with C/Fortran code that expects structured data
- You're working with memory-mapped files that need fixed record layouts
- You want to stay within pure NumPy without pandas dependencies

**When to use alternatives instead:**
- **Pandas DataFrames**: For any complex data manipulation, joining, grouping, or analysis (99% of cases)
- **Lists of dicts**: For small datasets where convenience matters more than performance
- **Custom classes**: When you need methods and complex behavior with your data

Here's how structured arrays work and when they might be useful:

```python
# Structured arrays store heterogeneous data efficiently
import numpy as np

# Define the structure of each record
star_dtype = np.dtype([
    ('name', 'U20'),        # Unicode string, max 20 chars
    ('ra', 'f8'),           # Right ascension (float64) in degrees
    ('dec', 'f8'),          # Declination (float64) in degrees  
    ('magnitude', 'f4'),    # Apparent magnitude (float32)
    ('distance', 'f4'),     # Distance in parsecs
])

# Create structured array - data stored contiguously in memory
stars = np.array([
    ('Sirius', 101.287, -16.716, -1.46, 2.64),
    ('Canopus', 95.988, -52.696, -0.74, 95.0),
    ('Arcturus', 213.915, 19.182, -0.05, 11.26),
], dtype=star_dtype)

# Access fields with bracket notation
print(stars['name'])      # ['Sirius' 'Canopus' 'Arcturus']
print(stars['magnitude'])  # [-1.46 -0.74 -0.05]

# NumPy operations work directly on fields
bright_stars = stars[stars['magnitude'] < 0]
abs_mag = stars['magnitude'] - 5*np.log10(stars['distance']) + 5

# Compare with alternatives:

# 1. Pandas DataFrame (most convenient for analysis)
import pandas as pd
df = pd.DataFrame({
    'name': ['Sirius', 'Canopus', 'Arcturus'],
    'ra': [101.287, 95.988, 213.915],
    'dec': [-16.716, -52.696, 19.182],
    'magnitude': [-1.46, -0.74, -0.05],
    'distance': [2.64, 95.0, 11.26]
})
# Rich functionality but more memory overhead
bright_df = df[df['magnitude'] < 0]

# 2. List of dictionaries (most Pythonic for small data)
stars_dicts = [
    {'name': 'Sirius', 'ra': 101.287, 'dec': -16.716, 'magnitude': -1.46},
    {'name': 'Canopus', 'ra': 95.988, 'dec': -52.696, 'magnitude': -0.74},
]
# Flexible but slow for large datasets

# Memory comparison for 1 million stars:
# Structured array: ~40 MB (compact, fixed layout)
# Pandas DataFrame: ~100+ MB (flexible, rich features)
# List of dicts: ~400+ MB (maximum flexibility, poor performance)
```

**Record Arrays**: A variant of structured arrays that allows attribute-style access using dot notation. They're essentially structured arrays with syntactic sugar:

```python
# Convert to record array for attribute access
rec_stars = np.rec.fromarrays(
    [stars['name'], stars['ra'], stars['dec']], 
    names='name,ra,dec'
)
print(rec_stars.name)  # Attribute style - convenient but ~10% slower
print(rec_stars['name'])  # Still works

# In practice, if you want attribute access, use pandas:
print(df.name)  # Pandas provides this naturally with more features
```

The bottom line: structured arrays are a specialized tool. For learning NumPy, understanding regular arrays is far more important. You'll rarely create structured arrays in new code, but you might encounter them when reading data from binary files or working with legacy scientific codebases.

### Memory-Mapped Arrays for Huge Datasets

When working with datasets larger than your RAM (common in astronomy), memory-mapped arrays let you work with data stored on disk as if it were in memory:

```python
# Create a memory-mapped array on disk
# Useful when data doesn't fit in RAM
filename = 'large_data.dat'
shape = (1000000, 1000)  # 1 million x 1000 array
dtype = np.float32

# Create and write to memory-mapped array
mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

# Only accessed parts are loaded into RAM
mmap_array[0, :] = np.arange(1000)  # Only this row in memory
mmap_array[999999, :] = np.arange(1000, 2000)  # And now this row

# Ensure data written to disk
del mmap_array  # Flush and close

# Later, read the memory-mapped file
readonly_mmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
print(readonly_mmap[0, 0])  # Only loads what's needed

# Clean up
import os
os.remove(filename)

# This is invaluable for:
# - Large astronomical images that don't fit in memory
# - Time series data from long observations
# - Simulation outputs that are generated incrementally
```

### Modern Random Number Generation

NumPy 1.17+ introduced an improved random number API that's better for reproducible science and parallel computing. You'll see both the old and new approaches in existing code:

```python
In [326]: # Old way (still widely used in existing code)
In [327]: np.random.seed(42)
In [328]: old_random = np.random.randn(5)

In [329]: # New way - better for parallel computing and cleaner design
In [330]: rng = np.random.default_rng(seed=42)  # Create generator object
In [331]: new_random = rng.standard_normal(5)   # Use generator methods

In [332]: print(f"Old API: {old_random}")
In [333]: print(f"New API: {new_random}")
Old API: [ 0.496  0.861  0.697 -0.817  0.673]
New API: [ 0.308 -1.299  1.966  0.404  0.224]

In [334]: # Why use the new API for new code?
In [335]: # 1. Better statistical properties (improved algorithms)
In [336]: # 2. Thread-safe for parallel computing
In [337]: # 3. Can create independent random streams easily
In [338]: rng1 = np.random.default_rng(seed=42)
In [339]: rng2 = np.random.default_rng(seed=43)
In [340]: # rng1 and rng2 produce independent, reproducible streams

# Both APIs will coexist for years - know both!
```

## 7.11 Common Pitfalls and Debugging

### The View vs Copy Confusion

```python
# PITFALL: Not knowing when you have a view
arr = np.arange(10)
subset = arr[2:5]  # This is a VIEW
subset[0] = 999
print(arr)  # [0 1 999 3 4 5 6 7 8 9] - Original changed!

# SOLUTION: Be explicit about views and copies
subset_copy = arr[2:5].copy()  # Explicit copy
subset_view = arr[2:5]  # Clear that it's a view
```

### Integer Division Changes

```python
# PITFALL: Integer division behavior
arr = np.array([1, 2, 3, 4, 5])

# In Python 3, / always gives float
result1 = arr / 2
print(result1.dtype)  # float64

# Use // for integer division
result2 = arr // 2
print(result2.dtype)  # int64

# Be explicit about dtype when needed
result3 = (arr / 2).astype(int)
```

### Broadcasting Errors

```python
# PITFALL: Unexpected broadcasting
a = np.ones((3, 3))
b = np.array([1, 2, 3, 4])  # Wrong size!

try:
    c = a + b
except ValueError as e:
    print(f"Error: {e}")
    # operands could not be broadcast together with shapes (3,3) (4,)

# SOLUTION: Check shapes before operations
def debug_broadcasting(a, b):
    """Helper to understand broadcasting."""
    print(f"a.shape: {a.shape}")
    print(f"b.shape: {b.shape}")
    try:
        result_shape = np.broadcast_shapes(a.shape, b.shape)
        print(f"Result shape: {result_shape}")
    except ValueError:
        print("Cannot broadcast these shapes!")
```

### 🛠️ **Debug This!**

This code has a subtle bug. Can you find it?

```python
def normalize_columns(data):
    """Normalize each column to have mean=0, std=1."""
    for col in range(data.shape[1]):
        data[:, col] -= data[:, col].mean()
        data[:, col] /= data[:, col].std()
    return data

# Test it
test_data = np.array([[1.0, 100.0],
                       [2.0, 200.0],
                       [3.0, 300.0]])
                       
normalized = normalize_columns(test_data)
print(f"Original data:\n{test_data}")
print(f"Normalized:\n{normalized}")
```

<details>
<summary>Bug and Solution</summary>

**Bug**: The function modifies the input array in-place but also returns it, which is confusing. Worse, the original data is lost! After calling the function, both `test_data` and `normalized` point to the same modified array.

```python
print(test_data is normalized)  # True - same object!
```

**Solutions**:

Option 1: Work on a copy (preserve original)
```python
def normalize_columns_safe(data):
    """Normalize columns without modifying input."""
    result = data.copy()  # Work on copy
    for col in range(result.shape[1]):
        col_data = result[:, col]
        result[:, col] = (col_data - col_data.mean()) / col_data.std()
    return result
```

Option 2: Make in-place operation explicit
```python
def normalize_columns_inplace(data):
    """Normalize columns in-place. Returns None to signal in-place."""
    for col in range(data.shape[1]):
        col_data = data[:, col]
        data[:, col] = (col_data - col_data.mean()) / col_data.std()
    # Don't return anything for in-place operations
```

Option 3: Use vectorization (best!)
```python
def normalize_columns_vectorized(data):
    """Vectorized normalization - fastest and clearest."""
    return (data - data.mean(axis=0)) / data.std(axis=0)
```

The vectorized version is not only faster but also automatically returns a new array, avoiding the confusion entirely.

</details>

## 7.12 Working with Scientific Data Formats (Optional)

While we'll cover these in more detail later, here's a brief introduction to common scientific data formats:

### HDF5 for Large Datasets

HDF5 is ideal for large, complex scientific datasets:

```python
# Basic HDF5 usage with h5py
import h5py

# Create HDF5 file with datasets
with h5py.File('scientific_data.h5', 'w') as f:
    # Create datasets
    f.create_dataset('temperature', data=np.random.randn(1000, 1000))
    f.create_dataset('pressure', data=np.random.randn(1000, 1000))
    
    # Add metadata as attributes
    f['temperature'].attrs['units'] = 'Kelvin'
    f['temperature'].attrs['date'] = '2024-01-15'

# Read HDF5 file
with h5py.File('scientific_data.h5', 'r') as f:
    temp = f['temperature'][:]  # Load into NumPy array
    print(f"Temperature shape: {temp.shape}")
    print(f"Units: {f['temperature'].attrs['units']}")

# Clean up
import os
os.remove('scientific_data.h5')
```

### FITS for Astronomical Data

FITS (Flexible Image Transport System) is the standard for astronomical data:

```python
# Basic FITS usage with astropy (when available)
try:
    from astropy.io import fits
    
    # Create FITS file
    data = np.random.randn(512, 512)  # Simulated image
    hdu = fits.PrimaryHDU(data)
    hdu.header['OBSERVER'] = 'Your Name'
    hdu.header['EXPTIME'] = 300.0  # Exposure time in seconds
    
    # Write and read
    hdu.writeto('test.fits', overwrite=True)
    
    # Read FITS file
    with fits.open('test.fits') as hdul:
        image = hdul[0].data  # NumPy array
        header = hdul[0].header
        print(f"Image shape: {image.shape}")
        print(f"Exposure time: {header['EXPTIME']} seconds")
    
    # Clean up
    os.remove('test.fits')
    
except ImportError:
    print("astropy not installed - FITS example skipped")
```

These formats integrate seamlessly with NumPy arrays, making them ideal for scientific data storage and exchange.

## Practice Exercises

### Exercise 7.1: Implement Moving Average

Create a function that computes a moving average efficiently:

```python
"""
Implement a moving average function that:
1. Takes a 1D array and window size
2. Returns array of moving averages
3. Handles edge cases appropriately
4. Is vectorized (no Python loops)

Example:
data = [1, 2, 3, 4, 5]
window = 3
result = [1.5, 2, 3, 4, 4.5]  # Edges handled with smaller windows

Hint: Consider np.convolve or cumulative sum approach
"""

def moving_average(data, window_size):
    """
    Compute moving average using vectorization.
    
    Parameters
    ----------
    data : array-like
        Input data
    window_size : int
        Size of moving window
    
    Returns
    -------
    array
        Moving averages
    """
    # Your implementation here
    pass

# Test cases
test_data = np.random.randn(1000)
ma = moving_average(test_data, 10)
assert len(ma) == len(test_data), "Output length should match input"
assert np.isfinite(ma).all(), "All values should be finite"
```

### Exercise 7.2: Image Processing with Broadcasting

Implement image transformations using broadcasting:

```python
"""
Create functions for basic image processing:
1. Brightness adjustment (add constant to all pixels)
2. Contrast adjustment (multiply all pixels)
3. Gamma correction (power transformation)
4. RGB to grayscale conversion

Work with images as arrays where:
- Grayscale: (height, width)
- RGB: (height, width, 3)

Use broadcasting to avoid loops!
"""

def adjust_brightness(image, delta):
    """
    Adjust brightness by adding delta.
    Ensure result stays in valid range [0, 1].
    """
    # Your implementation here
    pass

def adjust_gamma(image, gamma):
    """
    Apply gamma correction: out = in^gamma
    Handles negative values properly.
    """
    # Your implementation here
    pass

def rgb_to_grayscale(rgb_image):
    """
    Convert RGB to grayscale using standard weights:
    gray = 0.299*R + 0.587*G + 0.114*B
    """
    # Your implementation here
    pass

# Test with synthetic image
test_rgb = np.random.rand(100, 100, 3)
gray = rgb_to_grayscale(test_rgb)
assert gray.shape == (100, 100), "Should be 2D grayscale"
assert 0 <= gray.min() and gray.max() <= 1, "Should be in [0,1] range"
```

### Exercise 7.3: Optimize Star Catalog Operations

Compare different approaches for astronomical calculations:

```python
"""
Given a star catalog with positions and magnitudes,
implement these operations multiple ways and compare performance:

1. Find all stars within a given angular distance from a point
2. Calculate total flux from all stars (flux = 10^(-0.4 * magnitude))
3. Find the brightest N stars in a region

Implement using:
a) Pure Python loops (baseline)
b) NumPy vectorization
c) Boolean masking

Measure performance differences.
"""

# Generate synthetic catalog
n_stars = 100000
catalog = {
    'ra': np.random.uniform(0, 360, n_stars),      # Right ascension (degrees)
    'dec': np.random.uniform(-90, 90, n_stars),    # Declination (degrees)
    'mag': np.random.uniform(-1, 20, n_stars)      # Magnitude
}

def angular_distance(ra1, dec1, ra2, dec2):
    """
    Calculate angular distance between points on sphere.
    Uses haversine formula for numerical stability.
    """
    # Convert to radians
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    
    # Haversine formula
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return np.degrees(c)

def find_nearby_stars_loop(catalog, ra_center, dec_center, radius):
    """Pure Python implementation."""
    # Your implementation here
    pass

def find_nearby_stars_numpy(catalog, ra_center, dec_center, radius):
    """Vectorized NumPy implementation."""
    # Your implementation here
    pass

# Compare performance
import time
# Your timing code here
```

### Exercise 7.4: Memory-Efficient Large Array Processing

Work with arrays too large to fit in memory:

```python
"""
Process a large dataset in chunks to avoid memory issues:

1. Create a large dataset (simulate with smaller array)
2. Process in chunks of fixed size
3. Combine results appropriately

Example task: Calculate statistics for a 10GB array
on a machine with 4GB RAM.

Implement:
- Chunked mean calculation
- Chunked standard deviation (trickier!)
- Chunked percentiles
"""

def chunked_mean(data_generator, chunk_size=1000000):
    """
    Calculate mean of data that comes in chunks.
    Uses numerically stable online algorithm.
    """
    total_sum = 0.0
    total_count = 0
    
    for chunk in data_generator:
        # Your implementation here
        pass
    
    return total_sum / total_count if total_count > 0 else 0.0

def chunked_std(data_generator, chunk_size=1000000):
    """
    Calculate standard deviation in chunks.
    Uses Welford's online algorithm for numerical stability.
    """
    n = 0
    mean = 0.0
    M2 = 0.0
    
    for chunk in data_generator:
        # Your implementation here
        # Hint: Update mean and M2 incrementally
        pass
    
    return np.sqrt(M2 / n) if n > 1 else 0.0

# Test with generator that simulates large data
def data_generator(total_size, chunk_size):
    """Generate random data in chunks."""
    n_chunks = total_size // chunk_size
    for _ in range(n_chunks):
        yield np.random.randn(chunk_size)
    
    remainder = total_size % chunk_size
    if remainder:
        yield np.random.randn(remainder)

# Verify your implementation
total = 10000000  # 10 million points
gen = data_generator(total, chunk_size=100000)
mean = chunked_mean(gen)
print(f"Chunked mean: {mean:.6f} (should be ~0)")
```

## Key Takeaways

✅ **NumPy arrays are fundamentally different from Python lists** - They store homogeneous data in contiguous memory blocks, enabling 10-100x performance improvements through vectorized operations in compiled C code.

✅ **Vectorization is the key mental shift** - Think in terms of operations on entire arrays, not individual elements. This leverages CPU vector instructions and eliminates Python interpreter overhead.

✅ **Broadcasting enables elegant code** - Operations between arrays of different shapes follow simple rules, eliminating explicit loops while maintaining memory efficiency.

✅ **Views vs copies matter for correctness and performance** - Basic slicing creates views (shared memory), while fancy indexing creates copies. Understanding this prevents bugs and memory issues.

✅ **Data types affect both memory and precision** - Choose float32 for speed/memory with acceptable precision loss, float64 for accuracy, and appropriate integer types for counting and indexing.

✅ **Memory layout impacts performance** - Row-major (C) vs column-major (Fortran) ordering affects cache efficiency. Access patterns should match memory layout for optimal performance.

✅ **Numerical stability matters** - Not all mathematically equivalent operations are numerically stable. Use library functions that implement stable algorithms.

✅ **NumPy isn't always the answer** - Use pandas for heterogeneous data, scipy.sparse for sparse matrices, and native Python for small datasets or variable-length sequences.

✅ **NumPy is the foundation** - Every major scientific Python library builds on NumPy. Understanding NumPy deeply means understanding the entire ecosystem.

## Quick Reference Tables

### Array Creation Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.array()` | From Python sequence | `np.array([1, 2, 3])` |
| `np.zeros()` | Initialize with zeros | `np.zeros((3, 4))` |
| `np.ones()` | Initialize with ones | `np.ones((2, 3))` |
| `np.empty()` | Uninitialized (fast but dangerous) | `np.empty((2, 2))` |
| `np.arange()` | Range of values | `np.arange(0, 10, 2)` |
| `np.linspace()` | N evenly spaced points | `np.linspace(0, 1, 11)` |
| `np.logspace()` | Log-spaced values | `np.logspace(0, 3, 4)` |
| `np.eye()` | Identity matrix | `np.eye(3)` |
| `np.random.rand()` | Uniform [0,1) | `np.random.rand(3, 3)` |
| `np.random.randn()` | Standard normal | `np.random.randn(3, 3)` |

### Essential Array Attributes

| Attribute | Description | Example Output |
|-----------|-------------|----------------|
| `.shape` | Dimensions | `(3, 4)` |
| `.ndim` | Number of dimensions | `2` |
| `.size` | Total elements | `12` |
| `.dtype` | Data type | `dtype('float64')` |
| `.nbytes` | Total bytes | `96` |
| `.T` | Transpose | Array view |
| `.flags` | Memory layout info | Dict of flags |
| `.base` | Base array if view | Array or None |

### Common Array Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `.reshape()` | Change dimensions | `arr.reshape(2, 3)` |
| `.flatten()` | To 1D copy | `arr.flatten()` |
| `.ravel()` | To 1D view/copy | `arr.ravel()` |
| `.transpose()` | Swap axes | `arr.transpose()` |
| `.sum()` | Sum elements | `arr.sum(axis=0)` |
| `.mean()` | Average | `arr.mean()` |
| `.std()` | Standard deviation | `arr.std()` |
| `.min()/.max()` | Extrema | `arr.max()` |
| `.argmin()/.argmax()` | Index of extrema | `arr.argmax()` |
| `.sort()` | Sort in-place | `arr.sort()` |
| `.copy()` | Deep copy | `arr.copy()` |

### Broadcasting Rules Quick Reference

| Shape A | Shape B | Result | Rule Applied |
|---------|---------|--------|--------------|
| `(3,)` | `()` | `(3,)` | Scalar broadcasts |
| `(3, 4)` | `(4,)` | `(3, 4)` | 1D broadcasts to rows |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | Column broadcasts |
| `(3, 1, 4)` | `(1, 5, 4)` | `(3, 5, 4)` | Both broadcast |
| `(3, 4)` | `(2, 3, 4)` | `(2, 3, 4)` | Smaller adds dimensions |
| `(3, 4)` | `(5, 4)` | Error! | Incompatible shapes |

## Debugging Checklist

When NumPy code doesn't work as expected:

1. **Check shapes**: `print(f"Shape: {arr.shape}")`
2. **Check dtype**: `print(f"Dtype: {arr.dtype}")`
3. **Check if view or copy**: `print(f"Owns data: {arr.flags['OWNDATA']}")`
4. **Check for NaN/Inf**: `print(f"Has NaN: {np.isnan(arr).any()}")`
5. **Check memory layout**: `print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")`
6. **Check broadcasting**: `np.broadcast_shapes(a.shape, b.shape)`

## Further Resources

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) - Official comprehensive guide
- [NumPy API Reference](https://numpy.org/doc/stable/reference/index.html) - Complete function documentation
- [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) - Transition guide
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) - Advanced vectorization techniques

## Next Chapter Preview

With NumPy mastery achieved, Chapter 8 introduces Matplotlib for visualization. You'll discover how Matplotlib's object-oriented design (building on Chapter 6) works seamlessly with NumPy arrays. Every plot you create will use NumPy arrays as its foundation, and you'll learn to create publication-quality figures that bring your data to life.

The NumPy-Matplotlib synergy is fundamental: plot data is NumPy arrays, image data is NumPy arrays, and all transformations use NumPy operations. Your deep understanding of NumPy will make mastering visualization natural and intuitive!# Chapter 7: NumPy - The Foundation of Scientific Computing

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand why NumPy arrays are 10-100x faster than Python lists for numerical computation
- Create and manipulate arrays using various initialization methods and slicing techniques
- Apply vectorization to eliminate explicit loops and write efficient scientific code
- Master broadcasting rules to perform operations on arrays of different shapes elegantly
- Use NumPy's mathematical functions for scientific calculations
- Understand memory layout and its impact on performance
- Debug common NumPy errors and understand when operations create copies vs views
- Integrate NumPy with the scientific Python ecosystem

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Work with Python lists and understand indexing/slicing (Chapter 4)
- ✓ Write functions and understand scope (Chapter 5)
- ✓ Understand object methods and attributes (Chapter 6)
- ✓ Use list comprehensions for data transformation (Chapter 4)
- ✓ Work with nested data structures (Chapter 4)

## Chapter Overview

So far, you've been using Python lists for numerical data. But try this simple experiment: compute the sum of squares for a million numbers using a list comprehension versus a loop. Even with list comprehensions, Python is surprisingly slow for numerical work. This is where NumPy transforms Python from a general-purpose language into a scientific computing powerhouse.

NumPy (Numerical Python) is not just a library—it's the foundation upon which the entire scientific Python ecosystem is built. Every plot you make with Matplotlib, every optimization you run with SciPy, every dataframe you manipulate with Pandas, ultimately relies on NumPy arrays. Understanding NumPy deeply means understanding how scientific computing works in Python.

This chapter reveals why NumPy is fast (hint: it's not written in Python), how its mental model differs from pure Python (vectorization over loops), and how its design patterns appear throughout scientific computing. You'll learn to think in arrays, not elements—a fundamental shift that makes the difference between code that takes hours and code that takes seconds. By the end, you'll understand why that `Star` class you created in Chapter 6 might be better represented as a structured NumPy array when you have millions of stars to process.

## 7.1 Why NumPy? The Performance Revolution

Let's start with a motivating example that shows why NumPy exists:

```python
In [1]: import time
In [2]: import numpy as np

# Pure Python: sum of squares for 1 million numbers
In [3]: def python_sum_of_squares(n):
   ...:     """Pure Python implementation."""
   ...:     numbers = list(range(n))
   ...:     return sum(x**2 for x in numbers)

# NumPy: same calculation
In [4]: def numpy_sum_of_squares(n):
   ...:     """NumPy implementation."""
   ...:     numbers = np.arange(n)
   ...:     return np.sum(numbers**2)

# Time both approaches
In [5]: n = 1_000_000

In [6]: start = time.perf_counter()
In [7]: python_result = python_sum_of_squares(n)
In [8]: python_time = time.perf_counter() - start

In [9]: start = time.perf_counter()
In [10]: numpy_result = numpy_sum_of_squares(n)
In [11]: numpy_time = time.perf_counter() - start

In [12]: print(f"Python: {python_time:.3f} seconds")
In [13]: print(f"NumPy:  {numpy_time:.3f} seconds")
In [14]: print(f"Speedup: {python_time/numpy_time:.1f}x")
Python: 0.142 seconds
NumPy:  0.003 seconds
Speedup: 47.3x

In [15]: python_result == numpy_result  # Same answer!
Out[15]: True
```

NumPy is nearly 50 times faster! But why? The answer reveals fundamental truths about scientific computing.

### The Secret: NumPy Arrays Are Not Python Lists

```{mermaid}
flowchart TD
    subgraph "Python List"
        L[List Object] --> P1[Pointer 1]
        L --> P2[Pointer 2]
        L --> P3[Pointer 3]
        L --> PN[Pointer N]
        
        P1 --> O1[Integer Object<br/>type: int<br/>value: 0]
        P2 --> O2[Integer Object<br/>type: int<br/>value: 1]
        P3 --> O3[Integer Object<br/>type: int<br/>value: 2]
        PN --> ON[Integer Object<br/>type: int<br/>value: N-1]
    end
    
    subgraph "NumPy Array"
        A[Array Header<br/>dtype: int64<br/>shape: (N,)<br/>strides: (8,)] --> M[Contiguous Memory Block<br/>0 | 1 | 2 | 3 | ... | N-1]
    end
    
    style L fill:#f9f
    style A fill:#9f9
    style M fill:#9ff
```

**Python lists** store pointers to Python objects scattered throughout memory. Each integer is a full Python object with type information, reference counting, and other overhead. Accessing an element means following a pointer, checking the type, extracting the value—expensive operations repeated millions of times.

**NumPy arrays** store raw numbers in contiguous memory, like C arrays. The array header contains metadata (data type, shape, strides), but the data itself is just bytes in memory. Operations can be passed directly to optimized C/Fortran code that processes memory blocks efficiently, leveraging CPU vector instructions and cache locality.

### The Mental Model Shift: Vectorization

The performance gain requires a different programming style. Instead of thinking about individual elements, think about entire arrays:

```python
# Python style: loop over elements
def python_distance(x_coords, y_coords):
    """Calculate distances from origin, Python style."""
    distances = []
    for x, y in zip(x_coords, y_coords):
        dist = (x**2 + y**2)**0.5
        distances.append(dist)
    return distances

# NumPy style: operate on entire arrays
def numpy_distance(x_coords, y_coords):
    """Calculate distances from origin, NumPy style."""
    return np.sqrt(x_coords**2 + y_coords**2)

# Test with 100,000 points
n_points = 100_000
x = np.random.randn(n_points)
y = np.random.randn(n_points)

# Convert to lists for Python version
x_list = x.tolist()
y_list = y.tolist()

%timeit python_distance(x_list, y_list)
# 31.2 ms ± 501 µs per loop

%timeit numpy_distance(x, y)
# 371 µs ± 5.2 µs per loop

# 84x faster!
```

This is **vectorization**: expressing operations on entire arrays rather than individual elements. The loop still happens, but it's in compiled C code, not interpreted Python.

### 📦 **Computational Thinking Box: The Two-Language Problem**

```
PATTERN: The Two-Language Problem in Scientific Computing

Many scientific computing ecosystems face a dilemma:
- High-level languages (Python, MATLAB, R) are great for experimentation
- Low-level languages (C, Fortran) are needed for performance
- Scientists want to think about science, not memory management

NumPy's Solution:
- Python interface for thinking and prototyping
- C/Fortran implementation for computation
- Seamless boundary between the two

This pattern appears throughout scientific Python:
- NumPy: Python interface, C implementation
- SciPy: Python interface, Fortran/C++ implementation  
- Pandas: Python interface, Cython implementation
- Scikit-learn: Python interface, Cython/C++ implementation

The key insight: put the boundary at the right abstraction level.
For NumPy, that's the array operation, not the element operation.
```

## 7.2 Creating Arrays: From Lists to Grids

NumPy provides many ways to create arrays, each optimized for different use cases. Understanding these is crucial for efficient scientific computing.

### From Python Sequences

The most straightforward way is converting existing Python data:

```python
In [16]: # From a list
In [17]: list_data = [1, 2, 3, 4, 5]
In [18]: arr = np.array(list_data)
In [19]: print(f"Array: {arr}")
In [20]: print(f"Type: {type(arr)}")  # Note: it's an object!
In [21]: print(f"Dtype: {arr.dtype}")  # Data type of elements
Array: [1 2 3 4 5]
Type: <class 'numpy.ndarray'>
Dtype: int64

In [22]: # From nested lists (creates 2D array)
In [23]: matrix_data = [[1, 2, 3],
   ...:                 [4, 5, 6],
   ...:                 [7, 8, 9]]
In [24]: matrix = np.array(matrix_data)
In [25]: print(f"Matrix:\n{matrix}")
In [26]: print(f"Shape: {matrix.shape}")  # (rows, columns)
In [27]: print(f"Dimensions: {matrix.ndim}")
Matrix:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Shape: (3, 3)
Dimensions: 2
```

Remember from Chapter 6: NumPy arrays are objects! They have attributes (`shape`, `dtype`, `size`) and methods (`reshape()`, `mean()`, `sum()`). This is OOP in action.

### Initialization Functions

Creating arrays from scratch is often more efficient than converting lists:

```python
In [28]: # Arrays of zeros (useful for accumulation)
In [29]: zeros = np.zeros((3, 4))  # 3 rows, 4 columns
In [30]: print(f"Zeros:\n{zeros}")
Zeros:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

In [31]: # Arrays of ones (useful for counting)
In [32]: ones = np.ones((2, 3), dtype=np.int32)  # Can specify dtype
In [33]: print(f"Ones:\n{ones}")
Ones:
[[1 1 1]
 [1 1 1]]

In [34]: # Identity matrix (useful for linear algebra)
In [35]: identity = np.eye(3)
In [36]: print(f"Identity:\n{identity}")
Identity:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

In [37]: # Uninitialized array (fastest, but contains garbage)
In [38]: empty = np.empty((2, 2))  # DANGER: random values!
In [39]: print(f"Empty (undefined values):\n{empty}")
Empty (undefined values):
[[4.67e-310 0.00e+000]
 [0.00e+000 0.00e+000]]
```

### ⚠️ **Common Bug Alert: Uninitialized Arrays**

```python
# WRONG: Assuming empty arrays contain zeros
def calculate_sums_wrong(data, n_bins):
    sums = np.empty(n_bins)  # Contains garbage!
    for i, value in enumerate(data):
        bin_idx = int(value) % n_bins
        sums[bin_idx] += value  # Adding to garbage!
    return sums

# CORRECT: Use zeros for accumulation
def calculate_sums_correct(data, n_bins):
    sums = np.zeros(n_bins)  # Initialized to zero
    for i, value in enumerate(data):
        bin_idx = int(value) % n_bins
        sums[bin_idx] += value
    return sums

# The bug might not be obvious in testing!
test_data = np.array([1.5, 2.7, 3.2])
print(calculate_sums_wrong(test_data, 5))   # Unpredictable!
print(calculate_sums_correct(test_data, 5))  # [0, 1.5, 2.7, 3.2, 0]
```

Always use `zeros()` for accumulation, `ones()` for counting, and only use `empty()` when you'll immediately overwrite all values.

### Range Arrays

For sequences of numbers, NumPy provides optimized functions:

```python
In [40]: # Like Python's range, but returns an array
In [41]: integers = np.arange(10)  # 0 to 9
In [42]: print(f"Integers: {integers}")
Integers: [0 1 2 3 4 5 6 7 8 9]

In [43]: # With start, stop, step
In [44]: evens = np.arange(0, 10, 2)
In [45]: print(f"Evens: {evens}")
Evens: [0 2 4 6 8]

In [46]: # Floating-point ranges
In [47]: floats = np.arange(0, 1, 0.1)
In [48]: print(f"Floats: {floats}")
Floats: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

In [49]: # Linear spacing (specify number of points, not step)
In [50]: linear = np.linspace(0, 1, 11)  # 11 points from 0 to 1
In [51]: print(f"Linear: {linear}")
Linear: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]

In [52]: # Logarithmic spacing (for log-scale plots)
In [53]: logarithmic = np.logspace(0, 3, 4)  # 10^0 to 10^3
In [54]: print(f"Logarithmic: {logarithmic}")
Logarithmic: [   1.   10.  100. 1000.]
```

### 🔍 **Check Your Understanding**

What's the difference between `np.arange(0, 1, 0.1)` and `np.linspace(0, 1, 11)`?

<details>
<summary>Answer</summary>

Both create arrays from 0 to 1, but they work differently:

- `np.arange(0, 1, 0.1)` uses a step size of 0.1. Due to floating-point arithmetic, it might not include exactly 1.0 and might have slight inaccuracies.
- `np.linspace(0, 1, 11)` creates exactly 11 evenly spaced points including both endpoints. It's more precise for floating-point ranges.

```python
# Demonstration of the difference
arange_arr = np.arange(0, 1, 0.1)
linspace_arr = np.linspace(0, 1, 11)

print(f"arange length: {len(arange_arr)}")      # 10 (doesn't include 1.0)
print(f"linspace length: {len(linspace_arr)}")  # 11 (includes both endpoints)
print(f"arange last: {arange_arr[-1]}")         # 0.9
print(f"linspace last: {linspace_arr[-1]}")     # 1.0

# Floating-point issues with arange
print(f"0.1 + 0.1 + 0.1 == 0.3? {0.1 + 0.1 + 0.1 == 0.3}")  # False!
```

Use `linspace` when you need a specific number of points including endpoints. Use `arange` for integer sequences or when you need a specific step size.

</details>

### Random Arrays

Scientific computing often needs random data for Monte Carlo simulations, testing, or initialization:

```python
In [55]: # Set seed for reproducibility (important for science!)
In [56]: np.random.seed(42)

In [57]: # Uniform distribution [0, 1)
In [58]: uniform = np.random.rand(3, 3)
In [59]: print(f"Uniform:\n{uniform}")
Uniform:
[[0.374 0.950 0.731]
 [0.598 0.156 0.155]
 [0.058 0.866 0.601]]

In [60]: # Standard normal distribution (mean=0, std=1)
In [61]: normal = np.random.randn(3, 3)
In [62]: print(f"Normal:\n{normal}")
Normal:
[[ 0.708 -0.757 -1.316]
 [ 0.386  1.749  0.297]
 [-0.814 -0.454 -1.150]]

In [63]: # Random integers
In [64]: integers = np.random.randint(0, 10, size=(2, 4))
In [65]: print(f"Random integers:\n{integers}")
Random integers:
[[7 6 6 8]
 [8 3 9 8]]

In [66]: # Custom distribution (e.g., Poisson for photon counts)
In [67]: photon_counts = np.random.poisson(lam=5, size=10)
In [68]: print(f"Photon counts: {photon_counts}")
Photon counts: [3 5 6 3 8 4 3 3 6 2]
```

## 7.3 Array Attributes and Memory Layout

Understanding array attributes and memory layout is crucial for writing efficient code and debugging strange behavior.

### Essential Attributes

Every NumPy array has attributes that describe its structure:

```python
In [69]: # Create a 3D array for demonstration
In [70]: arr = np.random.randn(2, 3, 4)  # 2 blocks, 3 rows, 4 columns

In [71]: print(f"Shape: {arr.shape}")        # Dimensions
In [72]: print(f"Size: {arr.size}")          # Total elements
In [73]: print(f"Ndim: {arr.ndim}")          # Number of dimensions
In [74]: print(f"Dtype: {arr.dtype}")        # Data type
In [75]: print(f"Itemsize: {arr.itemsize}")  # Bytes per element
In [76]: print(f"Nbytes: {arr.nbytes}")      # Total bytes
Shape: (2, 3, 4)
Size: 24
Ndim: 3
Dtype: float64
Itemsize: 8
Nbytes: 192

In [77]: # Memory layout information
In [78]: print(f"Strides: {arr.strides}")  # Bytes to next element
In [79]: print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
In [80]: print(f"Fortran-contiguous: {arr.flags['F_CONTIGUOUS']}")
Strides: (96, 32, 8)
C-contiguous: True
Fortran-contiguous: False
```

### Memory Layout: Row-Major vs Column-Major

NumPy can store arrays in different memory layouts, which affects performance:

```{mermaid}
flowchart LR
    subgraph "Row-Major (C-style)"
        RM[2D Array<br/>[[1,2,3],<br/>[4,5,6]]] --> RMM[Memory: 1|2|3|4|5|6]
        RMM --> RMD[Row 0 then Row 1]
    end
    
    subgraph "Column-Major (Fortran-style)"
        CM[2D Array<br/>[[1,2,3],<br/>[4,5,6]]] --> CMM[Memory: 1|4|2|5|3|6]
        CMM --> CMD[Column 0 then Column 1 then Column 2]
    end
    
    style RM fill:#9f9
    style CM fill:#f9f
```

```python
In [81]: # Default is C-order (row-major)
In [82]: c_array = np.array([[1, 2, 3],
   ...:                       [4, 5, 6]])
In [83]: print(f"C-order strides: {c_array.strides}")  # (24, 8)
C-order strides: (24, 8)  # 3 elements * 8 bytes to next row

In [84]: # Can create Fortran-order (column-major)
In [85]: f_array = np.array([[1, 2, 3],
   ...:                       [4, 5, 6]], order='F')
In [86]: print(f"F-order strides: {f_array.strides}")  # (8, 16)
F-order strides: (8, 16)  # 1 element * 8 bytes to next row

In [87]: # Performance implications
In [88]: large = np.random.randn(1000, 1000)

In [89]: # Summing along rows (axis=1) is fast for C-order
In [90]: %timeit large.sum(axis=1)
574 µs ± 12.3 µs per loop

In [91]: # Summing along columns (axis=0) is slower for C-order
In [92]: %timeit large.sum(axis=0)
1.28 ms ± 23.4 µs per loop

# Why? Cache locality! Accessing contiguous memory is faster.
```

### Data Types and Memory Usage

NumPy provides precise control over data types, crucial for memory efficiency and numerical precision:

```python
In [93]: # Integer types
In [94]: int8 = np.array([1, 2, 3], dtype=np.int8)    # -128 to 127
In [95]: int16 = np.array([1, 2, 3], dtype=np.int16)  # -32,768 to 32,767
In [96]: int32 = np.array([1, 2, 3], dtype=np.int32)  # ~±2 billion
In [97]: int64 = np.array([1, 2, 3], dtype=np.int64)  # ~±9 quintillion

In [98]: print(f"int8 bytes: {int8.nbytes}")   # 3 bytes
In [99]: print(f"int64 bytes: {int64.nbytes}") # 24 bytes
int8 bytes: 3
int64 bytes: 24

In [100]: # Floating-point types
In [101]: float16 = np.array([1.0, 2.0], dtype=np.float16)  # Half precision
In [102]: float32 = np.array([1.0, 2.0], dtype=np.float32)  # Single precision
In [103]: float64 = np.array([1.0, 2.0], dtype=np.float64)  # Double precision

In [104]: # Complex numbers for signal processing
In [105]: complex_arr = np.array([1+2j, 3+4j], dtype=np.complex128)
In [106]: print(f"Complex array: {complex_arr}")
In [107]: print(f"Real parts: {complex_arr.real}")
In [108]: print(f"Imaginary parts: {complex_arr.imag}")
Complex array: [1.+2.j 3.+4.j]
Real parts: [1. 3.]
Imaginary parts: [2. 4.]
```

### 🔊 **Performance Profile: Data Type Impact**

```python
# Memory and speed tradeoffs with different dtypes
n = 10_000_000  # 10 million elements

# Create arrays with different dtypes
float64_arr = np.random.randn(n)  # Default
float32_arr = float64_arr.astype(np.float32)
float16_arr = float64_arr.astype(np.float16)

print(f"float64: {float64_arr.nbytes / 1e6:.1f} MB")
print(f"float32: {float32_arr.nbytes / 1e6:.1f} MB")
print(f"float16: {float16_arr.nbytes / 1e6:.1f} MB")

# Performance comparison
%timeit float64_arr.sum()  # 7.92 ms
%timeit float32_arr.sum()  # 3.96 ms (2x faster!)
%timeit float16_arr.sum()  # 15.8 ms (slower - no hardware support)

# But beware precision loss!
print(f"float64 sum: {float64_arr.sum()}")
print(f"float32 sum: {float32_arr.sum()}")  # Slightly different
print(f"float16 sum: {float16_arr.sum()}")  # Very different!
```

Choose dtypes based on your needs: float64 for precision, float32 for speed/memory with acceptable precision loss, integers when appropriate.

## 7.4 Indexing and Slicing: Views vs Copies

NumPy's indexing is powerful but has subtleties that can cause bugs if not understood properly.

### Basic Indexing (Creates Views)

Basic slicing creates **views** that share memory with the original array:

```python
In [109]: # 1D indexing - similar to lists
In [110]: arr = np.arange(10)
In [111]: print(f"Original: {arr}")
In [112]: print(f"Element 3: {arr[3]}")
In [113]: print(f"Slice 2:5: {arr[2:5]}")
In [114]: print(f"Every 2nd: {arr[::2]}")
In [115]: print(f"Reverse: {arr[::-1]}")
Original: [0 1 2 3 4 5 6 7 8 9]
Element 3: 3
Slice 2:5: [2 3 4]
Every 2nd: [0 2 4 6 8]
Reverse: [9 8 7 6 5 4 3 2 1 0]

In [116]: # CRITICAL: Slices are views, not copies!
In [117]: slice_view = arr[2:5]
In [118]: slice_view[0] = 999
In [119]: print(f"Original after modification: {arr}")
Original after modification: [  0   1 999   3   4   5   6   7   8   9]
```

### ⚠️ **Common Bug Alert: Unexpected Mutation**

```python
# DANGEROUS: Modifying a view changes the original!
def process_middle(data):
    """Process middle section of data."""
    middle = data[len(data)//4:3*len(data)//4]  # View!
    middle *= 2  # This modifies the original!
    return middle

original = np.arange(10)
print(f"Before: {original}")
result = process_middle(original)
print(f"After: {original}")  # Original is changed!
# Before: [0 1 2 3 4 5 6 7 8 9]
# After: [0 1 4 6 8 5 6 7 8 9]

# SAFE: Explicitly copy when needed
def process_middle_safe(data):
    """Process middle section without modifying original."""
    middle = data[len(data)//4:3*len(data)//4].copy()  # Copy!
    middle *= 2
    return middle
```

### Multidimensional Indexing

For 2D arrays and higher, indexing becomes more powerful:

```python
In [120]: # Create a 2D array
In [121]: matrix = np.array([[1, 2, 3],
    ...:                      [4, 5, 6],
    ...:                      [7, 8, 9]])

In [122]: # Single element
In [123]: print(f"Element [1,2]: {matrix[1, 2]}")  # Row 1, Column 2
Element [1,2]: 6

In [124]: # Entire row or column
In [125]: print(f"Row 1: {matrix[1, :]}")     # or just matrix[1]
In [126]: print(f"Column 2: {matrix[:, 2]}")
Row 1: [4 5 6]
Column 2: [3 6 9]

In [127]: # Submatrix
In [128]: print(f"Top-left 2x2:\n{matrix[:2, :2]}")
Top-left 2x2:
[[1 2]
 [4 5]]

In [129]: # Strided access
In [130]: print(f"Every other element:\n{matrix[::2, ::2]}")
Every other element:
[[1 3]
 [7 9]]
```

### Fancy Indexing (Creates Copies)

Using arrays as indices creates **copies**, not views:

```python
In [131]: arr = np.arange(10) * 10

In [132]: # Integer array indexing
In [133]: indices = np.array([1, 3, 5])
In [134]: selected = arr[indices]  # This is a COPY
In [135]: print(f"Selected: {selected}")
Selected: [10 30 50]

In [136]: selected[0] = 999
In [137]: print(f"Original unchanged: {arr}")  # Original intact
Original unchanged: [ 0 10 20 30 40 50 60 70 80 90]

In [138]: # Boolean indexing (masking)
In [139]: mask = arr > 40
In [140]: print(f"Mask: {mask}")
In [141]: filtered = arr[mask]  # Also a COPY
In [142]: print(f"Filtered: {filtered}")
Mask: [False False False False False  True  True  True  True  True]
Filtered: [50 60 70 80 90]

In [143]: # Combining conditions
In [144]: complex_mask = (arr > 20) & (arr < 70)  # Note: & not 'and'
In [145]: print(f"Complex filter: {arr[complex_mask]}")
Complex filter: [30 40 50 60]
```

### 📦 **Computational Thinking Box: Views vs Copies**

```
PATTERN: Memory Efficiency Through Views

Views are NumPy's way of providing different perspectives 
on the same data without copying it. This is crucial for:

1. Memory efficiency: No duplication of large datasets
2. Performance: No time spent copying
3. Consistency: Changes visible everywhere

When NumPy creates views:
- Basic slicing (arr[1:5], arr[:, 2])
- Reshaping (arr.reshape())
- Transposing (arr.T)

When NumPy creates copies:
- Fancy indexing (arr[[1,3,5]])
- Boolean indexing (arr[arr > 0])
- Explicit copy (arr.copy())

Testing if something is a view:
arr.base is not None  # True if arr is a view

This pattern appears in:
- Pandas DataFrames (views of underlying NumPy arrays)
- Memory-mapped files (views of disk data)
- GPU computing (avoiding expensive memory transfers)
```

## 7.5 Vectorization: Thinking in Arrays

Vectorization is the key to NumPy's performance. It means expressing operations on entire arrays rather than individual elements.

### Universal Functions (ufuncs)

NumPy provides "universal functions" that operate element-wise on arrays:

```python
In [146]: # Arithmetic operations are vectorized
In [147]: a = np.array([1, 2, 3, 4])
In [148]: b = np.array([10, 20, 30, 40])

In [149]: print(f"Addition: {a + b}")
In [150]: print(f"Multiplication: {a * b}")
In [151]: print(f"Power: {a ** 2}")
Addition: [11 22 33 44]
Multiplication: [10 40 90 160]
Power: [ 1  4  9 16]

In [152]: # Mathematical functions are vectorized
In [153]: angles = np.array([0, np.pi/4, np.pi/2, np.pi])
In [154]: print(f"Sin: {np.sin(angles)}")
In [155]: print(f"Cos: {np.cos(angles)}")
Sin: [0.000e+00 7.071e-01 1.000e+00 1.225e-16]
Cos: [ 1.000e+00  7.071e-01  6.123e-17 -1.000e+00]

In [156]: # Comparison operations are vectorized
In [157]: arr = np.arange(5)
In [158]: print(f"Greater than 2: {arr > 2}")
In [159]: print(f"Equal to 3: {arr == 3}")
Greater than 2: [False False False  True  True]
Equal to 3: [False False False  True False]
```

### Vectorizing Custom Functions

You can vectorize your own functions to work on arrays:

```python
In [160]: # Scalar function
In [161]: def photon_energy(wavelength_nm):
    ...:     """Calculate photon energy in eV from wavelength in nm."""
    ...:     h = 4.135667e-15  # Planck constant in eV·s
    ...:     c = 299792458     # Speed of light in m/s
    ...:     return h * c / (wavelength_nm * 1e-9)

In [162]: # Works on single values
In [163]: print(f"Energy at 500nm: {photon_energy(500):.2f} eV")
Energy at 500nm: 2.48 eV

In [164]: # Vectorize to work on arrays
In [165]: photon_energy_vec = np.vectorize(photon_energy)

In [166]: # Now works on arrays!
In [167]: wavelengths = np.array([400, 500, 600, 700])
In [168]: energies = photon_energy_vec(wavelengths)
In [169]: print(f"Energies: {energies}")
Energies: [3.10 2.48 2.07 1.77]

# Note: np.vectorize is convenient but not fast (it's still a Python loop)
# For performance, write truly vectorized code:
In [170]: def photon_energy_fast(wavelength_nm):
    ...:     """Truly vectorized version."""
    ...:     h = 4.135667e-15
    ...:     c = 299792458
    ...:     return h * c / (wavelength_nm * 1e-9)  # Works on arrays!

In [171]: # This is much faster for large arrays
In [172]: large_wavelengths = np.random.uniform(300, 800, 100000)
In [173]: %timeit photon_energy_vec(large_wavelengths)  # 35.2 ms
In [174]: %timeit photon_energy_fast(large_wavelengths) # 326 µs (100x faster!)
```

### Aggregation Functions

Aggregations reduce arrays to scalar values or smaller arrays:

```python
In [175]: data = np.random.randn(1000)

In [176]: # Basic statistics
In [177]: print(f"Mean: {data.mean():.4f}")
In [178]: print(f"Std: {data.std():.4f}")
In [179]: print(f"Min: {data.min():.4f}")
In [180]: print(f"Max: {data.max():.4f}")
In [181]: print(f"Median: {np.median(data):.4f}")
Mean: -0.0234
Std: 0.9897
Min: -3.2384
Max: 3.0234
Median: -0.0365

In [182]: # Percentiles (useful for outlier detection)
In [183]: print(f"5th percentile: {np.percentile(data, 5):.4f}")
In [184]: print(f"95th percentile: {np.percentile(data, 95):.4f}")
5th percentile: -1.6422
95th percentile: 1.5967

In [185]: # Along specific axes for multidimensional arrays
In [186]: matrix = np.random.randn(3, 4)
In [187]: print(f"Matrix:\n{matrix}")
In [188]: print(f"Column means: {matrix.mean(axis=0)}")  # Average each column
In [189]: print(f"Row means: {matrix.mean(axis=1)}")     # Average each row
Matrix:
[[-0.245  1.234 -0.567  0.891]
 [ 2.345 -1.234  0.123 -0.456]
 [ 0.789 -0.012  1.234 -2.345]]
Column means: [ 0.963 -0.004  0.263 -0.637]
Row means: [ 0.328  0.195 -0.084]
```

### 🔍 **Check Your Understanding**

Given a 2D array representing an image, how would you normalize it so all values are between 0 and 1?

<details>
<summary>Answer</summary>

There are several approaches depending on what you mean by normalization:

```python
# Create sample "image" data
image = np.random.randn(100, 100) * 50 + 128  # Centered at 128

# Method 1: Min-Max normalization (scales to exact [0, 1])
def min_max_normalize(arr):
    """Scale array to [0, 1] range."""
    return (arr - arr.min()) / (arr.max() - arr.min())

normalized1 = min_max_normalize(image)
print(f"Range: [{normalized1.min()}, {normalized1.max()}]")  # [0.0, 1.0]

# Method 2: Clipping to known range (e.g., 0-255 for 8-bit images)
def clip_normalize(arr, min_val=0, max_val=255):
    """Clip to range then normalize."""
    clipped = np.clip(arr, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)

normalized2 = clip_normalize(image)

# Method 3: Z-score normalization (mean=0, std=1, not [0,1])
def z_score_normalize(arr):
    """Standardize to mean=0, std=1."""
    return (arr - arr.mean()) / arr.std()

standardized = z_score_normalize(image)
print(f"Mean: {standardized.mean():.6f}, Std: {standardized.std():.6f}")

# Important: Choose based on your needs!
# - Min-max for display (guarantees [0,1])
# - Clipping for known ranges
# - Z-score for machine learning
```

The key insight: vectorized operations make this efficient even for large images. No loops needed!

</details>

## 7.6 Broadcasting: NumPy's Superpower

Broadcasting allows NumPy to perform operations on arrays of different shapes without explicit loops or copies. It's one of NumPy's most powerful features.

### The Broadcasting Rules

Broadcasting follows simple rules to determine how arrays of different shapes can be combined:

```{mermaid}
flowchart TD
    A[Arrays A and B] --> B{Compare shapes<br/>right to left}
    B --> C{Dimensions<br/>equal?}
    C -->|Yes| D[Compatible]
    C -->|No| E{One dimension<br/>is 1?}
    E -->|Yes| F[Broadcast:<br/>stretch size-1 dimension]
    E -->|No| G[Incompatible!<br/>Cannot broadcast]
    
    F --> H[Perform operation]
    D --> H
    
    style D fill:#9f9
    style F fill:#9ff
    style G fill:#f99
```

```python
In [190]: # Broadcasting in action
In [191]: arr = np.array([[1, 2, 3],
    ...:                   [4, 5, 6],
    ...:                   [7, 8, 9]])

In [192]: # Adding a scalar (broadcasts to all elements)
In [193]: print(f"Array + 10:\n{arr + 10}")
Array + 10:
[[11 12 13]
 [14 15 16]
 [17 18 19]]

In [194]: # Adding a 1D array to rows (broadcasts to each row)
In [195]: row_vector = np.array([100, 200, 300])
In [196]: print(f"Array + row vector:\n{arr + row_vector}")
Array + row vector:
[[101 202 303]
 [104 205 306]
 [107 208 309]]

In [197]: # Adding a column vector (broadcasts to each column)
In [198]: col_vector = np.array([[1000],
    ...:                          [2000],
    ...:                          [3000]])
In [199]: print(f"Array + column vector:\n{arr + col_vector}")
Array + column vector:
[[1001 1002 1003]
 [2004 2005 2006]
 [3007 3008 3009]]
```

### Practical Broadcasting Examples

Broadcasting makes many scientific calculations elegant:

```python
In [200]: # Normalize each column of a matrix
In [201]: data = np.random.randn(100, 3) * [10, 50, 100] + [0, 100, 200]
In [202]: print(f"Original means: {data.mean(axis=0)}")
In [203]: print(f"Original stds: {data.std(axis=0)}")

In [204]: # Subtract mean and divide by std for each column
In [205]: normalized = (data - data.mean(axis=0)) / data.std(axis=0)
In [206]: print(f"Normalized means: {normalized.mean(axis=0)}")  # ~0
In [207]: print(f"Normalized stds: {normalized.std(axis=0)}")    # ~1
Original means: [  0.234  99.876 200.123]
Original stds: [ 9.987 49.234 98.765]
Normalized means: [-1.23e-17  2.45e-17  3.67e-17]
Normalized stds: [1. 1. 1.]

In [208]: # Distance matrix between points
In [209]: points = np.random.randn(5, 2)  # 5 points in 2D
In [210]: # Use broadcasting to compute all pairwise differences
In [211]: diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
In [212]: # Shape: (5, 1, 2) - (1, 5, 2) = (5, 5, 2)
In [213]: distances = np.sqrt((diff**2).sum(axis=2))
In [214]: print(f"Distance matrix:\n{distances}")
Distance matrix:
[[0.    1.234 2.345 0.567 1.890]
 [1.234 0.    1.567 0.890 2.234]
 [2.345 1.567 0.    1.234 0.789]
 [0.567 0.890 1.234 0.    1.456]
 [1.890 2.234 0.789 1.456 0.   ]]
```

### ⚠️ **Common Bug Alert: Broadcasting Surprises**

```python
# UNEXPECTED: Broadcasting can hide dimension mismatches
a = np.array([[1, 2, 3]])     # Shape: (1, 3)
b = np.array([[10], [20]])    # Shape: (2, 1)

# This works but might not be what you intended!
result = a + b  # Broadcasts to (2, 3)
print(f"Result:\n{result}")
# [[11 12 13]
#  [21 22 23]]

# DEFENSIVE: Check shapes when unsure
def safe_add(a, b):
    """Add arrays with shape checking."""
    if a.shape != b.shape:
        print(f"Warning: Broadcasting {a.shape} and {b.shape}")
        print(f"Result shape will be: {np.broadcast_shapes(a.shape, b.shape)}")
    return a + b

# EXPLICIT: Use np.newaxis to be clear about intent
row = np.array([1, 2, 3])
col = np.array([10, 20])

# Clear that we want to broadcast
result = row[np.newaxis, :] + col[:, np.newaxis]
print(f"Explicit broadcasting:\n{result}")
```

## 7.7 Mathematical Operations and Linear Algebra

NumPy provides a comprehensive suite of mathematical functions optimized for arrays.

### Element-wise Mathematics

```python
In [215]: # Trigonometric functions
In [216]: angles = np.linspace(0, 2*np.pi, 5)
In [217]: print(f"Sin: {np.sin(angles)}")
In [218]: print(f"Arcsin of 0.5: {np.arcsin(0.5)}")

In [219]: # Exponential and logarithmic
In [220]: x = np.array([1, 2, 3])
In [221]: print(f"Exp: {np.exp(x)}")           # e^x
In [222]: print(f"Log: {np.log(x)}")           # Natural log
In [223]: print(f"Log10: {np.log10(x)}")       # Base-10 log
In [224]: print(f"Exp2: {np.exp2(x)}")         # 2^x

In [225]: # Special functions for scientific computing
In [226]: from scipy import special  # Extended special functions
In [227]: x = np.linspace(0, 10, 100)
In [228]: bessel = special.j0(x)  # Bessel function of first kind
In [229]: gamma = special.gamma(x + 1)  # Gamma function
```

### Linear Algebra Operations

NumPy includes a full linear algebra module:

```python
In [230]: # Matrix multiplication
In [231]: A = np.array([[1, 2],
    ...:                [3, 4]])
In [232]: B = np.array([[5, 6],
    ...:                [7, 8]])

In [233]: # Element-wise multiplication (NOT matrix multiplication)
In [234]: print(f"Element-wise A * B:\n{A * B}")

In [235]: # True matrix multiplication
In [236]: print(f"Matrix multiplication A @ B:\n{A @ B}")
In [237]: # Or: np.dot(A, B) or np.matmul(A, B)
Element-wise A * B:
[[ 5 12]
 [21 32]]
Matrix multiplication A @ B:
[[19 22]
 [43 50]]

In [238]: # Common linear algebra operations
In [239]: matrix = np.array([[3, 1],
    ...:                      [1, 2]])

In [240]: # Determinant
In [241]: det = np.linalg.det(matrix)
In [242]: print(f"Determinant: {det}")

In [243]: # Eigenvalues and eigenvectors
In [244]: eigenvalues, eigenvectors = np.linalg.eig(matrix)
In [245]: print(f"Eigenvalues: {eigenvalues}")
In [246]: print(f"Eigenvectors:\n{eigenvectors}")

In [247]: # Inverse
In [248]: inverse = np.linalg.inv(matrix)
In [249]: print(f"Inverse:\n{inverse}")
In [250]: print(f"Check: A @ A^-1 =\n{matrix @ inverse}")  # Should be identity

In [251]: # Solving linear systems: Ax = b
In [252]: A = np.array([[3, 1],
    ...:                [1, 2]])
In [253]: b = np.array([9, 8])
In [254]: x = np.linalg.solve(A, b)
In [255]: print(f"Solution: {x}")
In [256]: print(f"Check: Ax = {A @ x}")  # Should equal b
```

### 📦 **Computational Thinking Box: Numerical Stability**

```
PATTERN: Numerical Stability in Linear Algebra

Not all mathematically correct operations are numerically stable.
Small floating-point errors can explode into wrong answers.

Example: Solving Ax = b
- Mathematically: x = A^(-1) @ b
- Numerically: DON'T compute inverse! Use np.linalg.solve()

Why? Matrix inversion is:
1. Expensive: O(n³) operations
2. Unstable: Errors amplify with condition number
3. Unnecessary: Solving systems is more stable

Condition number measures sensitivity to errors:
cond = np.linalg.cond(A)
- cond ~ 1: Well-conditioned (stable)
- cond >> 1: Ill-conditioned (unstable)
- cond = ∞: Singular (no unique solution)

This pattern appears throughout scientific computing:
- Use np.linalg.lstsq() for overdetermined systems
- Use QR decomposition instead of normal equations
- Use SVD for rank-deficient problems
- Use specialized solvers for specific matrix structures

The lesson: Numerical computing ≠ symbolic math
Always consider stability, not just correctness.
```

## 7.8 Structured Arrays: Beyond Simple Numbers

NumPy can handle heterogeneous data through structured arrays, bridging the gap to databases and complex data:

```python
In [257]: # Define a structured dtype for star catalog
In [258]: star_dtype = np.dtype([
    ...:     ('name', 'U20'),        # Unicode string, max 20 chars
    ...:     ('ra', 'f8'),           # Right ascension (float64)
    ...:     ('dec', 'f8'),          # Declination (float64)
    ...:     ('magnitude', 'f4'),    # Apparent magnitude (float32)
    ...:     ('spectral_type', 'U10') # Spectral classification
    ...: ])

In [259]: # Create structured array
In [260]: stars = np.array([
    ...:     ('Sirius', 101.287, -16.716, -1.46, 'A1V'),
    ...:     ('Canopus', 95.988, -52.696, -0.74, 'A9II'),
    ...:     ('Arcturus', 213.915, 19.182, -0.05, 'K1.5III'),
    ...:     ('Vega', 279.234, 38.784, 0.03, 'A0V')
    ...: ], dtype=star_dtype)

In [261]: # Access fields like object attributes
In [262]: print(f"Names: {stars['name']}")
In [263]: print(f"Magnitudes: {stars['magnitude']}")
Names: ['Sirius' 'Canopus' 'Arcturus' 'Vega']
Magnitudes: [-1.46 -0.74 -0.05  0.03]

In [264]: # Boolean indexing works with structured arrays
In [265]: bright = stars[stars['magnitude'] < 0]
In [266]: print(f"Bright stars: {bright['name']}")
Bright stars: ['Sirius' 'Canopus' 'Arcturus']

In [267]: # Sorting by field
In [268]: sorted_stars = np.sort(stars, order='magnitude')
In [269]: print(f"Sorted by brightness: {sorted_stars['name']}")
Sorted by brightness: ['Sirius' 'Canopus' 'Arcturus' 'Vega']
```

This connects to Chapter 6's OOP concepts: structured arrays are like arrays of objects, but with better performance for large datasets. When you have millions of stars, structured arrays are more efficient than lists of Star objects.

## 7.9 Memory Management and Performance Tips

Understanding memory usage helps write efficient code for large datasets:

### Memory Views and Copies

```python
In [270]: # Check if array owns its data
In [271]: original = np.arange(1000000)
In [272]: view = original[::2]  # Every other element
In [273]: copy = original[::2].copy()

In [274]: print(f"View owns data: {view.flags['OWNDATA']}")  # False
In [275]: print(f"Copy owns data: {copy.flags['OWNDATA']}")  # True
In [276]: print(f"View base is original: {view.base is original}")  # True

In [277]: # Memory usage
In [278]: print(f"Original size: {original.nbytes / 1e6:.1f} MB")
In [279]: print(f"View size: {view.nbytes / 1e6:.1f} MB")  # Half size
In [280]: print(f"But view doesn't use extra memory!")
```

### In-place Operations

Modify arrays in-place to save memory:

```python
In [281]: # Out-of-place (creates new array)
In [282]: a = np.arange(1000000, dtype=np.float64)
In [283]: b = a * 2  # New array created

In [284]: # In-place (modifies existing array)
In [285]: a *= 2  # No new array

In [286]: # Many functions have in-place versions
In [287]: arr = np.random.randn(1000, 1000)
In [288]: # Out-of-place
In [289]: normalized = arr / arr.std()

In [290]: # In-place
In [291]: arr /= arr.std()  # Modifies arr directly

In [292]: # Some functions have 'out' parameter
In [293]: result = np.empty_like(arr)
In [294]: np.sqrt(arr**2, out=result)  # Writes to existing array
```

### 🔊 **Performance Profile: Memory Access Patterns**

```python
# Cache-friendly vs cache-unfriendly operations
large = np.random.randn(10000, 10000)

# Row-wise sum (cache-friendly for C-order arrays)
%timeit large.sum(axis=1)  # 57.4 ms

# Column-wise sum (cache-unfriendly for C-order arrays)
%timeit large.sum(axis=0)  # 128 ms

# Transpose makes column-wise cache-friendly
large_T = large.T
%timeit large_T.sum(axis=0)  # 58.1 ms (fast again!)

# But transpose is just a view (no copy)
print(f"Transpose is view: {large_T.base is large}")  # True

# For best performance:
# 1. Access memory sequentially when possible
# 2. Use contiguous arrays
# 3. Consider memory layout for your access pattern
```

## 7.10 Common Pitfalls and Debugging

### Integer Division Changed in Python 3

```python
# Python 2 vs Python 3 difference that affects NumPy
arr = np.array([1, 2, 3, 4, 5])

# Python 3: / always gives float
print(arr / 2)  # [0.5 1.  1.5 2.  2.5]

# Use // for integer division
print(arr // 2)  # [0 1 1 2 2]
```

### Modifying Arrays During Iteration

```python
# WRONG: Modifying array while iterating
arr = np.array([1, 2, 3, 4, 5])
for i, val in enumerate(arr):
    if val > 2:
        arr[i] = 0  # Dangerous!

# CORRECT: Use vectorized operations
arr[arr > 2] = 0

# Or if you must loop, work on a copy
for i, val in enumerate(arr.copy()):
    if val > 2:
        arr[i] = 0
```

### 🛠️ **Debug This!**

This code has a subtle bug. Can you find it?

```python
def normalize_columns(data):
    """Normalize each column to have mean=0, std=1."""
    for col in range(data.shape[1]):
        data[:, col] -= data[:, col].mean()
        data[:, col] /= data[:, col].std()
    return data

# Test it
test_data = np.array([[1, 100],
                       [2, 200],
                       [3, 300]], dtype=np.float64)
                       
normalized = normalize_columns(test_data)
print(f"Original data:\n{test_data}")
print(f"Normalized:\n{normalized}")
```

<details>
<summary>Bug and Solution</summary>

**Bug**: The function modifies the input array in-place but also returns it, which can be confusing. Worse, the original data is lost!

After normalization, both `test_data` and `normalized` point to the same modified array:
```python
print(test_data is normalized)  # True - same object!
```

**Solutions**:

Option 1: Work on a copy
```python
def normalize_columns(data):
    """Normalize columns without modifying input."""
    result = data.copy()  # Work on copy
    for col in range(result.shape[1]):
        result[:, col] -= result[:, col].mean()
        result[:, col] /= result[:, col].std()
    return result
```

Option 2: Make in-place operation explicit
```python
def normalize_columns_inplace(data):
    """Normalize columns in-place. Returns None."""
    for col in range(data.shape[1]):
        data[:, col] -= data[:, col].mean()
        data[:, col] /= data[:, col].std()
    # Don't return anything for in-place operations
```

Option 3: Use vectorization (best!)
```python
def normalize_columns_vectorized(data):
    """Vectorized normalization."""
    return (data - data.mean(axis=0)) / data.std(axis=0)
```

</details>

## Practice Exercises

### Exercise 7.1: Implement Moving Average

Create a function that computes a moving average efficiently:

```python
"""
Implement a moving average function that:
1. Takes a 1D array and window size
2. Returns array of moving averages
3. Handles edge cases appropriately
4. Is vectorized (no Python loops)

Example:
data = [1, 2, 3, 4, 5]
window = 3
result = [1.5, 2, 3, 4, 4.5]  # Edges handled with smaller windows

Hint: Consider np.convolve or clever use of cumsum
"""

def moving_average(data, window_size):
    """Compute moving average using vectorization."""
    # Your implementation here
    pass

# Test cases
test_data = np.random.randn(1000)
ma = moving_average(test_data, 10)
assert len(ma) == len(test_data)
assert np.isfinite(ma).all()
```

### Exercise 7.2: Image Processing with Broadcasting

Implement image transformations using broadcasting:

```python
"""
Create functions for basic image processing:
1. Brightness adjustment (add constant to all pixels)
2. Contrast adjustment (multiply all pixels)
3. Gamma correction (power transformation)
4. Color channel mixing (for RGB images)

Work with images as arrays where:
- Grayscale: (height, width)
- RGB: (height, width, 3)

Use broadcasting to avoid loops!
"""

def adjust_brightness(image, delta):
    """Adjust brightness by adding delta."""
    # Ensure result stays in valid range [0, 255] or [0, 1]
    pass

def adjust_gamma(image, gamma):
    """Apply gamma correction: out = in^gamma."""
    pass

def rgb_to_grayscale(rgb_image):
    """Convert RGB to grayscale using standard weights:
    gray = 0.299*R + 0.587*G + 0.114*B
    """
    pass

# Test with synthetic image
test_rgb = np.random.rand(100, 100, 3)
gray = rgb_to_grayscale(test_rgb)
assert gray.shape == (100, 100)
```

### Exercise 7.3: Optimize Star Catalog Operations

Compare different approaches for astronomical calculations:

```python
"""
Given a star catalog with positions and magnitudes,
implement these operations multiple ways and compare performance:

1. Find all stars within a given angular distance from a point
2. Calculate total flux from all stars (flux = 10^(-0.4 * magnitude))
3. Find the brightest N stars in a region

Implement using:
a) Pure Python loops (baseline)
b) NumPy vectorization
c) Boolean masking

Measure performance differences.
"""

# Generate synthetic catalog
n_stars = 100000
catalog = {
    'ra': np.random.uniform(0, 360, n_stars),      # Right ascension
    'dec': np.random.uniform(-90, 90, n_stars),    # Declination  
    'mag': np.random.uniform(-1, 20, n_stars)      # Magnitude
}

def find_nearby_stars_loop(catalog, ra_center, dec_center, radius):
    """Pure Python implementation."""
    # Your implementation
    pass

def find_nearby_stars_numpy(catalog, ra_center, dec_center, radius):
    """Vectorized NumPy implementation."""
    # Your implementation
    # Hint: Use broadcasting for angular distance
    pass

# Compare performance
import time
# Your timing code here
```

### Exercise 7.4: Memory-Efficient Large Array Processing

Work with arrays too large to fit in memory:

```python
"""
Process a large dataset in chunks to avoid memory issues:

1. Create a large dataset (simulate with smaller array)
2. Process in chunks of fixed size
3. Combine results appropriately

Example task: Calculate statistics for a 10GB array
on a machine with 4GB RAM.

Implement:
- Chunked mean calculation
- Chunked standard deviation (trickier!)
- Chunked percentiles
"""

def chunked_mean(data_generator, chunk_size=1000000):
    """Calculate mean of data that comes in chunks."""
    total_sum = 0
    total_count = 0
    
    for chunk in data_generator:
        # Your implementation
        pass
    
    return total_sum / total_count

def chunked_std(data_generator, chunk_size=1000000):
    """Calculate standard deviation in chunks.
    Hint: Use Welford's online algorithm or two-pass approach
    """
    # Your implementation
    pass

# Test with generator that simulates large data
def data_generator(total_size, chunk_size):
    """Generate random data in chunks."""
    n_chunks = total_size // chunk_size
    for _ in range(n_chunks):
        yield np.random.randn(chunk_size)
    
    remainder = total_size % chunk_size
    if remainder:
        yield np.random.randn(remainder)

# Verify your implementation
# Should work even if total_size > available memory
```

## Key Takeaways

✅ **NumPy arrays are not Python lists** - They store homogeneous data in contiguous memory, enabling 10-100x performance improvements through vectorized operations in compiled code.

✅ **Vectorization is the key mental shift** - Think in terms of operations on entire arrays, not individual elements. This leverages CPU vector instructions and eliminates Python loop overhead.

✅ **Broadcasting enables elegant code** - Operations between arrays of different shapes follow simple rules, eliminating the need for explicit loops while maintaining memory efficiency.

✅ **Views vs copies matter for performance** - Basic slicing creates views (shared memory), while fancy indexing creates copies. Understanding this prevents bugs and memory issues.

✅ **Data types affect both memory and precision** - Choose float32 for speed/memory with acceptable precision loss, float64 for accuracy, and appropriate integer types for counting.

✅ **Memory layout impacts performance** - Row-major (C) vs column-major (Fortran) ordering affects cache efficiency. Access patterns should match memory layout.

✅ **NumPy is the foundation** - Every major scientific Python library builds on NumPy. Understanding NumPy deeply means understanding scientific Python.

✅ **Debugging NumPy requires different tools** - Use shape, dtype, and flags attributes to understand arrays. Check for views vs copies, broadcasting behavior, and memory layout.

## Quick Reference Tables

### Array Creation Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.array()` | From Python sequence | `np.array([1, 2, 3])` |
| `np.zeros()` | Initialize with zeros | `np.zeros((3, 4))` |
| `np.ones()` | Initialize with ones | `np.ones((2, 3))` |
| `np.empty()` | Uninitialized (fast) | `np.empty((2, 2))` |
| `np.arange()` | Range of values | `np.arange(0, 10, 2)` |
| `np.linspace()` | N evenly spaced | `np.linspace(0, 1, 11)` |
| `np.logspace()` | Log-spaced values | `np.logspace(0, 3, 4)` |
| `np.eye()` | Identity matrix | `np.eye(3)` |
| `np.random.rand()` | Uniform [0,1) | `np.random.rand(3, 3)` |
| `np.random.randn()` | Standard normal | `np.random.randn(3, 3)` |

### Essential Array Attributes

| Attribute | Description | Example Output |
|-----------|-------------|----------------|
| `.shape` | Dimensions | `(3, 4)` |
| `.ndim` | Number of dimensions | `2` |
| `.size` | Total elements | `12` |
| `.dtype` | Data type | `dtype('float64')` |
| `.nbytes` | Total bytes | `96` |
| `.T` | Transpose | Array view |
| `.flat` | Flattened iterator | Iterator object |
| `.real` | Real part | Array |
| `.imag` | Imaginary part | Array |

### Common Array Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `.reshape()` | Change dimensions | `arr.reshape(2, 3)` |
| `.flatten()` | To 1D copy | `arr.flatten()` |
| `.ravel()` | To 1D view/copy | `arr.ravel()` |
| `.transpose()` | Swap axes | `arr.transpose()` |
| `.swapaxes()` | Swap two axes | `arr.swapaxes(0, 1)` |
| `.sum()` | Sum elements | `arr.sum(axis=0)` |
| `.mean()` | Average | `arr.mean()` |
| `.std()` | Standard deviation | `arr.std()` |
| `.min()/.max()` | Extrema | `arr.max()` |
| `.argmin()/.argmax()` | Index of extrema | `arr.argmax()` |
| `.sort()` | Sort in-place | `arr.sort()` |
| `.copy()` | Deep copy | `arr.copy()` |

### Mathematical Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.add()` | Addition | `np.add(a, b)` or `a + b` |
| `np.multiply()` | Element-wise multiply | `np.multiply(a, b)` or `a * b` |
| `np.dot()` | Dot product | `np.dot(a, b)` |
| `np.matmul()` | Matrix multiply | `np.matmul(a, b)` or `a @ b` |
| `np.sqrt()` | Square root | `np.sqrt(arr)` |
| `np.exp()` | Exponential | `np.exp(arr)` |
| `np.log()` | Natural log | `np.log(arr)` |
| `np.sin()/.cos()` | Trigonometric | `np.sin(arr)` |
| `np.abs()` | Absolute value | `np.abs(arr)` |
| `np.round()` | Round to nearest | `np.round(arr, 2)` |

### Broadcasting Rules Quick Reference

| Shape A | Shape B | Result | Rule Applied |
|---------|---------|--------|--------------|
| `(3,)` | `()` | `(3,)` | Scalar broadcasts |
| `(3, 4)` | `(4,)` | `(3, 4)` | 1D broadcasts to rows |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | Column broadcasts |
| `(3, 1, 4)` | `(1, 5, 4)` | `(3, 5, 4)` | Both broadcast |
| `(3, 4)` | `(2, 3, 4)` | `(2, 3, 4)` | Smaller adds dimensions |
| `(3, 4)` | `(5, 4)` | Error! | Incompatible |

### Linear Algebra Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.linalg.inv()` | Matrix inverse | `np.linalg.inv(A)` |
| `np.linalg.det()` | Determinant | `np.linalg.det(A)` |
| `np.linalg.eig()` | Eigenvalues/vectors | `vals, vecs = np.linalg.eig(A)` |
| `np.linalg.solve()` | Solve Ax = b | `np.linalg.solve(A, b)` |
| `np.linalg.lstsq()` | Least squares | `np.linalg.lstsq(A, b)` |
| `np.linalg.norm()` | Matrix/vector norm | `np.linalg.norm(A)` |
| `np.linalg.svd()` | Singular value decomp | `U, s, Vh = np.linalg.svd(A)` |
| `np.linalg.qr()` | QR decomposition | `Q, R = np.linalg.qr(A)` |

## Common Patterns Reference

### Pattern: Normalize Data
```python
# Min-max normalization to [0, 1]
normalized = (data - data.min()) / (data.max() - data.min())

# Z-score normalization (standardization)
standardized = (data - data.mean()) / data.std()

# Normalize each column independently
col_normalized = (data - data.mean(axis=0)) / data.std(axis=0)
```

### Pattern: Find Indices
```python
# Find indices where condition is true
indices = np.where(arr > threshold)

# Find first/last occurrence
first_idx = np.argmax(arr > threshold)  # First True
last_idx = len(arr) - np.argmax((arr > threshold)[::-1]) - 1

# Find N largest/smallest indices
n_largest_idx = np.argpartition(arr, -n)[-n:]
n_smallest_idx = np.argpartition(arr, n)[:n]
```

### Pattern: Sliding Window
```python
# Using stride tricks for sliding windows
from numpy.lib.stride_tricks import sliding_window_view

# Sliding window of size 3
windows = sliding_window_view(arr, window_shape=3)

# Manual approach with as_strided (advanced)
from numpy.lib.stride_tricks import as_strided
window_size = 3
shape = (len(arr) - window_size + 1, window_size)
strides = (arr.strides[0], arr.strides[0])
windows = as_strided(arr, shape=shape, strides=strides)
```

### Pattern: Batch Processing
```python
# Process large array in batches
batch_size = 1000
n_samples = len(data)

for start_idx in range(0, n_samples, batch_size):
    end_idx = min(start_idx + batch_size, n_samples)
    batch = data[start_idx:end_idx]
    # Process batch
    result[start_idx:end_idx] = process(batch)
```

## Debugging Checklist

When NumPy code doesn't work as expected, check:

1. **Shape mismatch**: Print shapes of all arrays
   ```python
   print(f"A shape: {A.shape}, B shape: {B.shape}")
   ```

2. **Data type issues**: Check and convert if needed
   ```python
   print(f"dtype: {arr.dtype}")
   arr = arr.astype(np.float64)
   ```

3. **View vs copy**: Check if modification affects original
   ```python
   print(f"Is view: {arr.base is not None}")
   ```

4. **Broadcasting**: Verify broadcast behavior
   ```python
   result_shape = np.broadcast_shapes(A.shape, B.shape)
   ```

5. **Memory layout**: Check for performance issues
   ```python
   print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
   ```

6. **NaN/Inf values**: Check for numerical issues
   ```python
   print(f"Has NaN: {np.isnan(arr).any()}")
   print(f"Has Inf: {np.isinf(arr).any()}")
   ```

## Further Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) - Comprehensive user guide
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html) - Complete API reference
- [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) - Transition guide

### Advanced Topics
- [NumPy's internals](https://numpy.org/doc/stable/reference/internals.html) - How NumPy works under the hood
- [Array interface](https://numpy.org/doc/stable/reference/arrays.interface.html) - For creating NumPy-compatible objects
- [C-API](https://numpy.org/doc/stable/reference/c-api/index.html) - For extending NumPy with C

### Performance Resources
- [Performance tips](https://numpy.org/doc/stable/user/c-info.how-to-extend.html) - Official optimization guide
- [Memory layout](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray) - Understanding strides and memory

## Next Chapter Preview

With NumPy mastery achieved, Chapter 8 introduces Matplotlib for visualization. You'll discover how Matplotlib's object-oriented design (building on Chapter 6) works seamlessly with NumPy arrays. You'll learn to create publication-quality figures, from simple line plots to complex multi-panel visualizations, understanding how every plot element is an object you can customize.

The NumPy-Matplotlib connection is fundamental: every data point you plot is a NumPy array, every image you display is a NumPy array, and every transformation you apply uses NumPy operations. Understanding NumPy deeply means you can manipulate plot data directly, create custom colormaps as arrays, and even implement your own visualization algorithms.

Get ready to make your data sing through visualization!