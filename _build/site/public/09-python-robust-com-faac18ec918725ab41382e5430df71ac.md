# Chapter 9: Robust Computing Fundamentals - Error Handling and Best Practices

## Learning Objectives

By the end of this chapter, you will be able to:
- Read and interpret Python **error messages** to diagnose problems efficiently
- Write **try/except blocks** to handle errors gracefully
- Validate inputs to prevent errors before they occur
- Use **assertions** to document and verify assumptions
- Replace print statements with proper **logging**
- Write simple **tests** to verify your functions work correctly
- Debug code systematically using proven strategies
- Understand how errors propagate through scientific calculations

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Write and call functions with parameters (Chapter 5)
- ✓ Work with NumPy arrays (Chapter 7)
- ✓ Use if/else statements and loops (Chapter 3)
- ✓ Work with lists and dictionaries (Chapter 4)
- ✓ Create simple plots with Matplotlib (Chapter 8)

## Chapter Overview

Your code will fail. This isn't pessimism—it's reality. The difference between beginners and professionals isn't that professionals write perfect code. It's that professionals write code that fails gracefully, tells them what went wrong, and helps them fix problems quickly.

Remember in Chapter 5 when we wrote this simple function?

```python
def calculate_mean(values):
    return sum(values) / len(values)
```

This optimistic code assumes values is never empty, always contains numbers, and never has missing data. In Chapter 7, we processed NumPy arrays without checking for **NaN** (Not a Number) values. In Chapter 8, we plotted data without verifying it was plottable. Real scientific data breaks all these assumptions.

This chapter transforms that naive code into **robust code**—code that handles unexpected situations gracefully rather than crashing. You'll learn techniques that prevented disasters like the Mars Climate Orbiter loss and that catch the kinds of errors that have led to retracted papers. By the end, your functions will validate inputs, your scripts will log their progress, and your errors will guide rather than frustrate you.

## 9.1 Understanding Error Messages

**Error messages** are structured reports that Python generates when something goes wrong during code execution. They tell you exactly what went wrong and where. Learning to read them transforms debugging from guesswork into detective work.

### Your First Error Message

Let's start with a simple error and learn to decode it:

```python
# Callback to Chapter 5: Remember our temperature conversion?
def celsius_to_fahrenheit(celsius):
    return celsuis * 9/5 + 32  # Typo: 'celsuis' not 'celsius'

# Try to use it
temp = 25
result = celsius_to_fahrenheit(temp)
```

This produces an error message with three critical parts:

```
Traceback (most recent call last):
  File "example.py", line 6, in <module>
    result = celsius_to_fahrenheit(temp)
  File "example.py", line 2, in celsius_to_fahrenheit
    return celsuis * 9/5 + 32
NameError: name 'celsuis' is not defined
```

**Read error messages from bottom to top:**

1. **Error Type** (bottom line): `NameError` tells you the category of problem. A **NameError** specifically means Python encountered a variable name it doesn't recognize.

2. **Error Message**: "name 'celsuis' is not defined" explains what's wrong. Python is looking for a variable called 'celsuis' but can't find it in the current **namespace** (the collection of currently defined variables).

3. **Location** (lines above): Shows exactly where the error occurred. The error happened in the file "example.py" on line 2, inside the function `celsius_to_fahrenheit`.

4. **Call Stack** (traceback): The **traceback** shows the sequence of function calls that led to the error. Think of it like breadcrumbs showing Python's path through your code. Each level shows which function called the next, helping you understand how the program reached the error.

This systematic reading approach works for any error. The fix here is obvious—we typed 'celsuis' instead of 'celsius'.

### Common Error Types

Let's understand the four **exception types** you'll encounter most often. An **exception** is Python's way of signaling that something exceptional (unusual) has happened that prevents normal execution:

```python
# TypeError: Wrong type for operation
text = "5"
result = text * 2      # Works! Gives "55" (string repetition)
result = text + 2      # TypeError! Can't add string and number

# Why this matters: Reading data from files often gives strings
# when you expect numbers, causing TypeErrors in calculations
```

A **TypeError** occurs when you try to perform an operation on a value of the wrong **type**. Python is strongly typed, meaning it doesn't automatically convert between types like strings and numbers.

```python
# ValueError: Right type, wrong value  
import math
math.sqrt(25)     # Works: 5.0
math.sqrt(-25)    # ValueError! Can't take sqrt of negative

# Why this matters: Physical calculations have constraints
# like non-negative masses or temperatures above absolute zero
```

A **ValueError** means the type is correct but the value is inappropriate for the operation. The square root function expects a non-negative number—giving it a negative number is the right type but wrong value.

```python
# IndexError: Accessing beyond list bounds
data = [10, 20, 30]
print(data[2])    # Works: 30 (remember: indexing starts at 0)
print(data[3])    # IndexError! No index 3

# Why this matters: Off-by-one errors are incredibly common
# when processing arrays of scientific data
```

An **IndexError** occurs when you try to access a list element that doesn't exist. Python uses **zero-based indexing**, meaning the first element is at index 0, which often causes **off-by-one errors**.

```python
# KeyError: Dictionary key doesn't exist
sensor = {'id': 'A1', 'temp': 25.3}
print(sensor['temp'])       # Works: 25.3
print(sensor['pressure'])   # KeyError! No 'pressure' key

# Why this matters: Data files might be missing expected fields
# or use different naming conventions than expected
```

A **KeyError** happens when you try to access a dictionary using a key that doesn't exist. This is common when processing data files where not all records have the same fields.

### Understanding Error Propagation

**Error propagation** refers to how errors spread through your program, potentially corrupting results far from the original problem. In scientific computing, understanding how errors cascade through calculations is crucial. One bad value can corrupt your entire analysis:

```python
# Demonstration: How one error ruins everything
def process_measurements(readings):
    """Show how errors propagate through calculations."""
    
    # Step 1: Calculate mean (fails if any reading is None)
    total = sum(readings)  # TypeError here if None in list
    mean = total / len(readings)
    
    # Step 2: Never reached due to error above
    normalized = [r / mean for r in readings]
    
    # Step 3: Never reached either
    return normalized

# One bad value stops everything
data = [23.5, 24.1, None, 23.8]  # None from sensor failure
result = process_measurements(data)  # Crashes at sum()
```

When Python encounters an error it can't handle, it immediately stops execution. This is called **raising an exception**. The exception travels up through the **call stack** until it either finds code that handles it or reaches the top level and crashes the program.

**Visualization of Error Propagation:**
```
Input: [23.5, 24.1, None, 23.8]
   ↓
Step 1: sum() → TypeError (can't add None)
   ✗ CRASH (Exception raised)
Step 2: normalize → Never executed
Step 3: return → Never reached

Result: No output, just an error
```

This demonstrates the **fail-fast principle**—it's better to stop immediately when something's wrong rather than continue with corrupted data that could produce misleading results.

### 🔍 **Check Your Understanding**

What error would this code produce and which line would cause it?

```python
temperatures = [20.5, 21.0, "22.5", 20.8]
total = 0
for temp in temperatures:
    total = total + temp
average = total / len(temperatures)
```

<details>
<summary>Answer</summary>

This produces a **TypeError** on line 4 (inside the loop). When the loop reaches "22.5", Python tries to execute `total + temp` which becomes `41.5 + "22.5"`. You can't add a number and a string.

The error message would be:
```
TypeError: unsupported operand type(s) for +: 'float' and 'str'
```

This error message tells us that the + operator doesn't support combining a float and a string. The term "operand" refers to the values being operated on (41.5 and "22.5"), and "unsupported" means Python doesn't know how to add these different types together.

To fix it, convert the string to a float:
```python
total = total + float(temp)
```

This is extremely common when reading data from CSV files where numbers might be stored as strings.
</details>

## 9.2 Handling Errors with Try/Except

Sometimes errors are expected. Files might not exist. Network connections might fail. Data might be corrupted. **Try/except blocks** let your program handle these situations gracefully instead of crashing.

### Basic Try/Except Structure

A **try/except block** is a control structure that attempts to execute code and provides alternative behavior if an error occurs:

```python
def safe_divide(a, b):
    """Divide two numbers, handling division by zero."""
    try:
        # The try block contains code that might fail
        result = a / b
        return result
    except ZeroDivisionError:
        # The except block runs only if this specific error occurs
        print(f"Warning: Attempted to divide {a} by zero")
        return None

# Use it safely
print(safe_divide(10, 2))   # Output: 5.0
print(safe_divide(10, 0))   # Output: Warning message, then None
```

The **try block** contains code that might raise an exception. If an exception occurs, Python immediately jumps to the **except block** that matches the exception type. If no exception occurs, the except block is skipped entirely. This is called **exception handling**—catching and responding to errors rather than letting them crash your program.

### Building Robust File Readers

File operations are where try/except blocks shine. Files might not exist, you might lack permissions, or the content might be corrupted. Let's build up a robust file reader step by step:

```python
# Step 1: Handle missing files (8 lines)
def read_file_basic(filename):
    """First lesson: Handle missing files."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
```

This handles the most common file error—the file doesn't exist. The **with statement** ensures the file is properly closed even if an error occurs, which is called **context management**.

```python
# Step 2: Add handling for permission errors (12 lines)
def read_file_safer(filename):
    """Second lesson: Handle multiple error types."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except PermissionError:
        print(f"No permission to read {filename}")
        return None
```

Now we handle two different exceptions. Python checks each except block in order, running the first one that matches the raised exception.

```python
# Step 3: Process content safely (18 lines)
def read_numbers_from_file(filename):
    """Third lesson: Handle content errors too."""
    try:
        with open(filename, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    
    # Now parse the content safely
    try:
        numbers = [float(line) for line in text.strip().split('\n')]
        return numbers
    except ValueError as e:
        print(f"Invalid number in file: {e}")
        return None
```

The `as e` syntax captures the exception object, allowing us to access its error message. This is useful for debugging because it tells us exactly which value caused the problem.

### 🎯 **Why This Matters: The Mars Climate Orbiter Disaster**

In 1999, NASA lost the $125 million Mars Climate Orbiter because one team used metric units while another used imperial units. The software didn't validate or handle unit mismatches. A simple check could have saved the mission:

```python
def combine_thrust_data(value1, unit1, value2, unit2):
    """What the Mars software should have done."""
    try:
        if unit1 != unit2:
            # Raising an exception explicitly signals an error
            raise ValueError(f"Unit mismatch: {unit1} vs {unit2}")
        return value1 + value2
    except ValueError as e:
        # Log the error and halt rather than proceed with bad data
        print(f"CRITICAL ERROR: {e}")
        print("Halting operation for safety")
        return None
```

The **raise statement** explicitly creates and throws an exception. This is how you signal that something is wrong in your own code. The disaster illustrates why error handling isn't bureaucracy—it prevents catastrophes.

### When NOT to Use Try/Except

Not all errors should be caught. Programming mistakes should fail loudly so you can fix them. This is an important distinction between **expected errors** (like missing files) and **programming errors** (like typos):

```python
# BAD: Hiding programming errors
def bad_statistics(data):
    try:
        mean = sum(data) / len(dta)  # Typo: 'dta' not 'data'
        return mean
    except:  # Never use bare except!
        return 0  # Hides the typo error!
```

A **bare except** catches all exceptions, including ones you don't expect. This is dangerous because it hides programming errors.

```python
# GOOD: Only catch specific, expected errors
def good_statistics(data):
    """Only handle the error we expect."""
    if len(data) == 0:
        raise ValueError("Cannot calculate mean of empty dataset")
    
    mean = sum(data) / len(data)  # Typo would crash (good!)
    return mean
```

The rule: catch errors you expect and can handle. Let unexpected errors crash so you can fix them. This is called **selective exception handling**.

### ⚠️ **Common Bug Alert: The Silent Except**

```python
# THE WORST ANTI-PATTERN IN PYTHON
try:
    result = complex_calculation()
except:
    result = 0  # Silently returns 0 for ANY error

# This hides critical errors like:
# - Typos in variable names (NameError)
# - Missing imports (ImportError)
# - Out of memory (MemoryError)
# - Keyboard interrupts (KeyboardInterrupt)
```

This **anti-pattern** (a common but harmful coding pattern) makes debugging nearly impossible because errors disappear silently. Always catch specific exceptions.

## 9.3 Validating Inputs

The best error is one that never happens. **Input validation** is the practice of checking that data meets expected requirements before processing it. This follows the **fail-fast principle**—detect problems as early as possible.

### 💡 **Computational Thinking: The Guard Clause Pattern**

**Guard clauses** are conditional statements at the beginning of a function that check preconditions and exit early if they're not met. This pattern creates a clear separation between validation and logic:

```python
# Without guard clauses - nested complexity
def process_data_nested(data):
    if data is not None:
        if len(data) > 0:
            if all(isinstance(x, (int, float)) for x in data):
                # Actual work buried in nested ifs
                return sum(data) / len(data)
    return None
```

This nested structure is hard to read and understand. Each level of indentation adds cognitive load.

```python
# With guard clauses - linear flow
def process_data_clean(data):
    # Guards at the top
    if data is None:
        return None
    if len(data) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in data):
        return None
    
    # Main logic clear and unindented
    return sum(data) / len(data)
```

Guard clauses create **linear code flow**—you can read from top to bottom without tracking nested conditions. This pattern reduces cognitive load by handling edge cases first, leaving the main logic clean and readable.

### Building Validation Layer by Layer

Effective validation checks multiple aspects of data. Let's build robust validation step by step, each focusing on one aspect:

```python
# Layer 1: Check for data existence (6 lines)
def validate_not_empty(data):
    """First check: Do we have data?"""
    if not data:
        raise ValueError("Cannot process empty data")
    return True
```

The **truthiness** check `if not data` works because empty containers (lists, strings, dicts) evaluate to False in Python. This is the cheapest validation—just checking if data exists.

```python
# Layer 2: Check data types (10 lines)
def validate_numeric(values):
    """Second check: Is data the right type?"""
    for i, val in enumerate(values):
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"Item {i} is {type(val).__name__}, expected number"
            )
    return True
```

The **isinstance() function** checks if a value is of a specific type or types. The `__name__` attribute gives us a human-readable type name for error messages.

```python
# Layer 3: Check physical constraints (12 lines)
def validate_temperature_kelvin(temps):
    """Third check: Does data make physical sense?"""
    for i, temp in enumerate(temps):
        if temp < 0:
            raise ValueError(
                f"Temperature {temp}K at position {i} "
                f"violates absolute zero"
            )
    return True
```

**Domain validation** checks if values make sense in your problem domain. Temperature can't be below absolute zero (0 Kelvin), masses can't be negative, probabilities must be between 0 and 1.

Now combine them into a complete validation pipeline:

```python
def process_temperature_data(measurements):
    """Complete validation pipeline."""
    # Validate in order of increasing cost
    validate_not_empty(measurements)      # Cheap check first
    validate_numeric(measurements)        # Medium cost
    validate_temperature_kelvin(measurements)  # Expensive last
    
    # Now safe to process
    return {
        'mean': sum(measurements) / len(measurements),
        'min': min(measurements),
        'max': max(measurements)
    }
```

The validation order matters for **performance optimization**. Check cheap conditions first (like emptiness) before expensive ones (like complex calculations).

### Validating NumPy Arrays

NumPy arrays from Chapter 7 need special validation for **NaN** (Not a Number) and **infinity** values. NaN represents undefined results (like 0/0), while infinity represents overflow:

```python
import numpy as np

def validate_array(arr):
    """Check array for common problems."""
    # Convert to array if needed (defensive programming)
    data = np.asarray(arr)
    
    # Check size
    if data.size == 0:
        raise ValueError("Empty array")
    
    # Check for NaN (Not a Number - undefined values)
    n_nan = np.sum(np.isnan(data))
    if n_nan > 0:
        print(f"Warning: {n_nan} NaN values found")
    
    # Check for infinity (overflow values)
    n_inf = np.sum(np.isinf(data))
    if n_inf > 0:
        raise ValueError(f"{n_inf} infinite values found")
    
    return data
```

The **np.isnan()** and **np.isinf()** functions return boolean arrays indicating which elements are NaN or infinite. These special values can corrupt calculations if not handled properly.

### Performance Cost of Validation

Validation has a **performance cost**—it takes time to check conditions. Let's measure it to understand the tradeoff:

```python
import time
import numpy as np

def process_without_validation(data):
    """No safety checks."""
    return np.mean(data)

def process_with_validation(data):
    """With safety checks."""
    if len(data) == 0:
        raise ValueError("Empty data")
    if np.any(np.isnan(data)):
        raise ValueError("Contains NaN")
    return np.mean(data)

# Measure the cost
data = np.random.randn(1000000)  # 1 million random numbers

start = time.time()
for _ in range(100):
    process_without_validation(data)
no_check_time = time.time() - start

start = time.time()
for _ in range(100):
    process_with_validation(data)
check_time = time.time() - start

print(f"Without validation: {no_check_time:.3f}s")
print(f"With validation: {check_time:.3f}s")
print(f"Overhead: {(check_time/no_check_time - 1)*100:.1f}%")

# Typical output:
# Without validation: 0.123s
# With validation: 0.145s
# Overhead: 17.9%
```

The ~18% **overhead** (additional time cost) is worth it for catching errors that could invalidate hours of computation. This is a classic **tradeoff**—spending a little time upfront to save a lot of time debugging later.

### 🔍 **Check Your Understanding**

Which validation should come first and why?
1. Checking if temperature is positive
2. Checking if list is empty
3. Checking if values are numbers

<details>
<summary>Answer</summary>

The correct order is:
1. Check if list is empty (fastest, most fundamental)
2. Check if values are numbers (can't check temperature if not numbers)
3. Check if temperature is positive (domain-specific, most expensive)

This follows the principle of "fail fast with cheapest check first." An empty list check is **O(1)** (constant time), type checking is **O(n)** (linear time), and domain validation might involve complex calculations. By ordering checks from cheapest to most expensive, we minimize the average time spent on validation.

The **dependency order** also matters—you can't check if temperatures are positive if you haven't verified they're numbers first!
</details>

## 9.4 Using Assertions

**Assertions** are debugging aids that verify assumptions about your program's state. They're like scientific hypotheses in your code—statements you believe must be true. Python checks them and alerts you if they're violated. They're your safety net during development.

### Assertions vs Validation: Know the Difference

There's a critical distinction between validating external input and asserting internal correctness:

```python
def analyze_spectrum(wavelengths, intensities):
    # VALIDATION: Check external inputs
    if len(wavelengths) == 0:
        raise ValueError("No wavelength data provided")
    if len(wavelengths) != len(intensities):
        raise ValueError("Wavelength and intensity arrays must match")
    
    # Process data
    normalized = intensities / np.max(intensities)
    
    # ASSERTION: Verify our logic is correct
    assert len(normalized) == len(intensities), "Lost data during normalization!"
    assert np.all(normalized <= 1.0), "Normalization failed!"
    
    return normalized
```

**Validation** protects against bad input from external sources (users, files, networks). **Assertions** catch bugs in your logic—they verify that your code does what you think it does. Assertions can be disabled in production with the `-O` flag, while validation always runs.

### Using Assertions to Document Assumptions

Assertions make your **assumptions** explicit—they document what you believe to be true at specific points in your code:

```python
def find_peak(data):
    """Find the maximum value and its index."""
    # Precondition: what must be true at start
    assert len(data) > 0, "Requires non-empty data"
    
    max_val = data[0]
    max_idx = 0
    
    for i, val in enumerate(data[1:], 1):
        if val > max_val:
            max_val = val
            max_idx = i
    
    # Postconditions: what we guarantee at end
    assert 0 <= max_idx < len(data), "Index out of bounds"
    assert data[max_idx] == max_val, "Index doesn't match value"
    
    return max_idx, max_val
```

**Preconditions** are assumptions about input state, while **postconditions** are guarantees about output state. Together they form a **contract**—if the preconditions are met, the postconditions will be satisfied.

### Assertions in Numerical Algorithms

Assertions are particularly valuable for checking **numerical stability**—whether calculations maintain mathematical properties despite floating-point limitations:

```python
def normalize_to_unit_range(values):
    """Scale values to [0, 1] range."""
    min_val = min(values)
    max_val = max(values)
    
    # Mathematical requirement
    assert max_val >= min_val, "Max less than min!"
    
    if max_val == min_val:
        # All values identical - special case
        return [0.5] * len(values)
    
    # Normalize using linear transformation
    range_val = max_val - min_val
    normalized = [(v - min_val) / range_val for v in values]
    
    # Verify our math preserved the mathematical properties
    assert all(0 <= v <= 1 for v in normalized), \
        f"Normalization produced values outside [0,1]"
    
    return normalized
```

The assertion checks that our normalization formula `(v - min) / (max - min)` actually produces values in [0, 1]. This catches **numerical errors** that could arise from floating-point arithmetic.

### 🛠️ **Debug This!**

This function has a subtle bug that the assertion will catch:

```python
def calculate_variance(data):
    """Calculate variance with Bessel's correction."""
    n = len(data)
    assert n > 1, "Need at least 2 values for variance"
    
    mean = sum(data) / n
    squared_diffs = [(x - mean)**2 for x in data]
    variance = sum(squared_diffs) / (n - 1)
    
    # This assertion sometimes fails. Why?
    assert variance >= 0, f"Variance {variance} is negative!"
    
    return variance

# Test case that breaks it
data = [1e20, 1, 2, 3]
result = calculate_variance(data)
```

<details>
<summary>Bug Explanation and Fix</summary>

The bug is **catastrophic cancellation**—a form of numerical instability that occurs when subtracting nearly equal floating-point numbers. When data contains values of very different magnitudes (1e20 vs 1), the mean is dominated by the large value. Subtracting this large mean from small values can produce negative squared differences due to **floating-point rounding errors**.

Here's what happens:
1. Mean ≈ 2.5e19 (dominated by 1e20)
2. (1 - 2.5e19)² should be positive
3. But floating-point arithmetic loses precision
4. Result can be slightly negative due to rounding

Fix using the numerically stable two-pass algorithm:
```python
def calculate_variance_stable(data):
    n = len(data)
    assert n > 1, "Need at least 2 values for variance"
    
    # First pass: accurate mean
    mean = sum(data) / n
    
    # Second pass: stable sum of squares
    sum_sq = 0
    for x in data:
        sum_sq += (x - mean) ** 2
    
    variance = sum_sq / (n - 1)
    
    # Allow tiny negative values from rounding
    assert variance >= -1e-10, f"Numerical error: {variance}"
    
    # Clamp to zero if slightly negative
    return max(0, variance)
```

This demonstrates why assertions are crucial for catching numerical instabilities!
</details>

## 9.5 Logging Instead of Print

Professional code uses **logging** instead of print statements. Logging is a systematic way to record program events with timestamps, severity levels, and structured output. It's the difference between scribbled notes and a proper lab notebook.

### From Print to Logging: A Transformation

Let's transform print-based debugging into professional logging:

```python
# Before: Using print (what we did in Chapter 5)
def process_data_print(data):
    print("Starting processing")
    print(f"Got {len(data)} items")
    
    results = []
    for item in data:
        if item < 0:
            print(f"Warning: negative value {item}")
        results.append(abs(item))
    
    print("Done")
    return results
```

```python
# After: Using logging
import logging

# Configure once at program start
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_data_logged(data):
    logging.info(f"Starting processing of {len(data)} items")
    
    results = []
    for i, item in enumerate(data):
        if item < 0:
            logging.warning(f"Negative value {item} at index {i}")
        results.append(abs(item))
    
    logging.info(f"Completed: processed {len(results)} items")
    return results
```

The **logging module** provides structured output with:
- **Timestamps** showing exactly when events occurred
- **Severity levels** indicating importance (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Consistent formatting** making logs easy to parse
- **Flexible output** to console, files, or network

The logged output includes all this metadata:
```
2024-11-15 10:23:45 - INFO - Starting processing of 5 items
2024-11-15 10:23:45 - WARNING - Negative value -2 at index 1
2024-11-15 10:23:45 - INFO - Completed: processed 5 items
```

### Logging Levels and When to Use Them

Different **severity levels** serve different purposes. Think of them like different types of lab notebook entries:

```python
import logging

def analyze_measurement(value, expected_range=(0, 100)):
    """Demonstrate all logging levels."""
    
    # DEBUG: Detailed information for diagnosing problems
    logging.debug(f"Raw input: {value}")
    
    if value < expected_range[0]:
        # ERROR: Something went wrong that prevents normal operation
        logging.error(f"Value {value} below minimum {expected_range[0]}")
        return None
    elif value > expected_range[1]:
        # WARNING: Something unexpected but not fatal
        logging.warning(f"Value {value} above typical maximum")
    
    result = value * 2.54  # Convert to metric
    
    # INFO: Normal program flow confirmation
    logging.info(f"Converted {value} to {result}")
    
    return result

# Set level to control what's shown
logging.getLogger().setLevel(logging.DEBUG)  # See everything
# logging.getLogger().setLevel(logging.WARNING)  # Only warnings and above
```

The **logging level** acts as a filter—only messages at or above the set level are displayed. This lets you control verbosity without changing code.

### Logging to Files for Permanent Records

For long-running computations, **file logging** creates permanent records you can analyze later:

```python
import logging

# Configure file logging
logging.basicConfig(
    filename='computation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def long_computation(data):
    """Simulate a long-running process."""
    logging.info(f"Starting computation with {len(data)} points")
    
    for i, point in enumerate(data):
        if i % 1000 == 0:
            # Progress indicators help track long runs
            logging.info(f"Processed {i}/{len(data)} points")
        
        # Actual computation here
        result = complex_calculation(point)
        
        if result is None:
            logging.error(f"Failed at point {i}")
    
    logging.info("Computation complete")
```

File logs provide an **audit trail**—a permanent record of what happened during execution. This is invaluable for debugging issues that only appear after hours of computation.

### 📊 **Performance Profile: Print vs Logging**

```python
import time
import logging
import sys

# Test: Impact of debug output
data = list(range(10000))

# Measure print
start = time.time()
for i in data:
    if i % 1000 == 0:
        print(f"Processing {i}", file=sys.stderr)
print_time = time.time() - start

# Measure logging
start = time.time()
for i in data:
    if i % 1000 == 0:
        logging.info(f"Processing {i}")
log_time = time.time() - start

# Measure logging with DEBUG level (not shown)
logging.getLogger().setLevel(logging.WARNING)
start = time.time()
for i in data:
    if i % 1000 == 0:
        logging.debug(f"Processing {i}")  # Not displayed
debug_time = time.time() - start

print(f"Print time: {print_time:.4f}s")
print(f"Logging time: {log_time:.4f}s")
print(f"Silent debug time: {debug_time:.4f}s")

# Typical output:
# Print time: 0.0234s
# Logging time: 0.0275s (17% slower but adds timestamps)
# Silent debug time: 0.0089s (debug calls still have cost)
```

Key insight: Logging is slightly slower than print but provides much more value. Even "silent" debug statements (below the current logging level) have a small **performance cost** because Python still evaluates the arguments.

## 9.6 Writing Simple Tests

**Testing** is the practice of verifying that code behaves as expected. It isn't about proving code is perfect—it's about catching obvious bugs before they waste your time. Think of tests as experimental verification of your code's hypotheses.

### Your First Test Function

A **test function** is code that verifies other code works correctly. Let's test a function from Chapter 5, now with better structure:

```python
def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def test_kelvin_to_celsius():
    """Test temperature conversion."""
    
    # Test 1: Known values (ground truth)
    assert kelvin_to_celsius(273.15) == 0, "Freezing point wrong"
    assert kelvin_to_celsius(373.15) == 100, "Boiling point wrong"
    
    # Test 2: Boundary conditions
    assert kelvin_to_celsius(0) == -273.15, "Absolute zero wrong"
    
    # Test 3: Round trip (inverse operations)
    temp_c = 25
    temp_k = temp_c + 273.15
    assert kelvin_to_celsius(temp_k) == temp_c, "Round trip failed"
    
    print("✓ All temperature tests passed!")

# Run the test
test_kelvin_to_celsius()
```

Good tests check multiple aspects:
- **Known values**: Cases where you know the exact answer
- **Boundary conditions**: Edge cases and limits
- **Round trips**: Operations that should cancel out
- **Properties**: Mathematical relationships that must hold

### Testing Properties, Not Just Values

**Property-based testing** verifies that mathematical properties hold regardless of specific values:

```python
def test_mean_properties():
    """Test properties that must hold for any mean function."""
    
    # Property 1: Mean of identical values equals that value
    same = [42.0] * 10
    assert calculate_mean(same) == 42.0
    
    # Property 2: Mean is within data range
    data = [1, 2, 3, 4, 5]
    mean = calculate_mean(data)
    assert min(data) <= mean <= max(data)
    
    # Property 3: Scaling data scales mean (linearity)
    scaled = [x * 2 for x in data]
    assert calculate_mean(scaled) == mean * 2
    
    # Property 4: Mean of two values is their midpoint
    assert calculate_mean([10, 20]) == 15
    
    print("✓ Mean properties verified!")
```

Properties are more robust than specific values because they test the underlying mathematics rather than individual cases.

### Testing Edge Cases

**Edge cases** are unusual inputs that often reveal bugs. They're the boundaries and special conditions where code is most likely to fail:

```python
def remove_outliers(data, threshold=3):
    """Remove values more than threshold stdevs from mean."""
    if len(data) == 0:
        return []
    
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    stdev = variance ** 0.5
    
    if stdev == 0:  # All values identical
        return data
    
    return [x for x in data if abs(x - mean) <= threshold * stdev]

def test_remove_outliers():
    """Test outlier removal with edge cases."""
    
    # Normal case
    data = [1, 2, 3, 100, 4, 5]
    cleaned = remove_outliers(data)
    assert 100 not in cleaned
    assert 3 in cleaned
    
    # Edge case 1: Empty list
    assert remove_outliers([]) == []
    
    # Edge case 2: Single value
    assert remove_outliers([42]) == [42]
    
    # Edge case 3: All identical (zero variance)
    same = [5, 5, 5, 5]
    assert remove_outliers(same) == same
    
    # Edge case 4: Two values far apart
    two = [0, 1000]
    result = remove_outliers(two, threshold=1)
    assert len(result) <= 2  # Might remove one or both
    
    print("✓ Edge cases handled correctly!")

test_remove_outliers()
```

Common edge cases to test:
- **Empty input**: No data at all
- **Single element**: Minimum valid input
- **Identical values**: No variation
- **Extreme values**: Very large or small numbers
- **Boundary values**: Exactly at limits

### 🎯 **Why This Matters: The Ariane 5 Disaster**

In 1996, the Ariane 5 rocket exploded 37 seconds after launch, destroying $370 million in satellites. The cause? **Reused code** from Ariane 4 wasn't tested with Ariane 5's flight parameters. A single untested edge case—a velocity value that exceeded 16-bit integer limits—caused an **integer overflow** error.

```python
def velocity_to_int16(velocity):
    """What went wrong in Ariane 5."""
    # This should have been tested!
    assert -32768 <= velocity <= 32767, \
        f"Velocity {velocity} exceeds 16-bit range"
    return int(velocity)

# Ariane 4 test (passed)
test_velocity_to_int16(25000)  # OK

# Ariane 5 test (never run!)
test_velocity_to_int16(40000)  # Would have caught the bug!
```

Testing with realistic data ranges would have prevented this disaster. The lesson: always test with the actual conditions your code will face, not just convenient test values.

## 9.7 Debugging Strategies

**Debugging** is the process of finding and fixing errors in code. It's detective work. Instead of randomly changing code hoping it works, follow a systematic approach that mirrors the scientific method.

### The Scientific Method of Debugging

Debugging follows the same process as scientific research—observation, hypothesis, experimentation, and analysis:

```python
def demonstrate_debugging_process():
    """Show systematic debugging approach."""
    
    # THE PROBLEM: Function returns wrong result
    def buggy_variance(data):
        """Calculate variance (has a bug)."""
        mean = sum(data) / len(data)
        diffs = [x - mean for x in data]
        squares = [d*d for d in diffs]
        return sum(squares) / len(data) - 1  # Bug here!
    
    # STEP 1: OBSERVE - Identify the symptom
    test_data = [2, 4, 6]
    result = buggy_variance(test_data)
    expected = 4.0  # Known correct answer
    print(f"Expected {expected}, got {result}")  # Wrong!
    
    # STEP 2: HYPOTHESIZE - Form theories
    # Theory 1: Mean calculation wrong?
    # Theory 2: Squared differences wrong?
    # Theory 3: Final division wrong?
    
    # STEP 3: EXPERIMENT - Test each theory
    mean = sum(test_data) / len(test_data)
    print(f"Mean: {mean}")  # Correct: 4.0
    
    diffs = [x - mean for x in test_data]
    print(f"Differences: {diffs}")  # Correct: [-2, 0, 2]
    
    squares = [d*d for d in diffs]
    print(f"Squares: {squares}")  # Correct: [4, 0, 4]
    
    # Found it! The bug is here:
    print(f"Sum/len: {sum(squares)/len(test_data)}")  # 2.67
    print(f"Sum/len - 1: {sum(squares)/len(test_data) - 1}")  # 1.67 (wrong!)
    print(f"Sum/(len-1): {sum(squares)/(len(test_data)-1)}")  # 4.0 (correct!)
    
    # STEP 4: FIX - Correct the bug
    def variance_fixed(data):
        mean = sum(data) / len(data)
        diffs = [x - mean for x in data]
        squares = [d*d for d in diffs]
        return sum(squares) / (len(data) - 1)  # Fixed!

demonstrate_debugging_process()
```

This **systematic approach** is much more efficient than random changes. By testing hypotheses one at a time, you isolate the problem quickly.

### Binary Search Debugging

**Binary search debugging** uses the divide-and-conquer principle to isolate problems in complex code:

```python
def complex_calculation(data):
    """A multi-step calculation to debug."""
    # Add checkpoints to bisect the problem
    
    # First half of calculation
    step1 = [x * 2 for x in data]
    print(f"After step 1: {step1[:3]}...")  # Checkpoint 1
    
    step2 = [x + 10 for x in step1]
    print(f"After step 2: {step2[:3]}...")  # Checkpoint 2
    
    # If error occurs here, problem is in first half
    # If error occurs below, problem is in second half
    
    step3 = [x / 3 for x in step2]
    print(f"After step 3: {step3[:3]}...")  # Checkpoint 3
    
    step4 = sum(step3) / len(step3)
    print(f"Final result: {step4}")  # Checkpoint 4
    
    return step4
```

By adding **checkpoints** (diagnostic output) at strategic locations, you can quickly determine which section contains the bug. This is much faster than checking every line.

### Debugging Flowchart

```
Start: Code produces wrong output
    ↓
Can you reproduce the error?
    No → Add logging, gather more info
    Yes ↓
    
Is the input what you expected?
    No → Fix input validation
    Yes ↓
    
Add checkpoint prints at midpoint
    ↓
Is the error before or after midpoint?
    Before → Check first half
    After → Check second half
    ↓
    
Repeat bisection until problem isolated
    ↓
Found the specific line with the bug
    ↓
Fix and verify with test case
    ↓
End: Add regression test to prevent reoccurrence
```

This **decision tree** approach ensures you don't miss steps and helps you debug efficiently.

### Common Debugging Patterns

Certain bugs appear repeatedly. Recognizing these patterns speeds debugging:

```python
# Pattern 1: The Off-By-One Error
def find_median_buggy(sorted_data):
    """Common bug: forgetting 0-indexing."""
    n = len(sorted_data)
    middle = n // 2
    return sorted_data[middle]  # Bug: wrong for even-length lists

def find_median_fixed(sorted_data):
    """Fixed: handle even and odd lengths."""
    n = len(sorted_data)
    if n % 2 == 1:  # Odd length: single middle value
        return sorted_data[n // 2]
    else:  # Even length: average of two middle values
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
```

**Off-by-one errors** occur when you forget that Python uses zero-based indexing or miscalculate array boundaries.

```python
# Pattern 2: The Mutation Surprise
def normalize_buggy(data):
    """Bug: modifying input data while reading it."""
    for i in range(len(data)):
        data[i] = data[i] / max(data)  # Max changes as we modify!
    return data

def normalize_fixed(data):
    """Fixed: calculate max first."""
    max_val = max(data)  # Store before modifying
    return [x / max_val for x in data]  # Create new list
```

**Mutation bugs** happen when you modify data while still using it, causing unexpected behavior.

### 🔍 **Check Your Understanding**

This code should double each value, but sometimes produces wrong results. What's the bug?

```python
def double_values(data):
    for i in range(len(data)):
        data[i] *= 2
    return data

# Test
original = [1, 2, 3]
doubled = double_values(original)
print(f"Original: {original}")
print(f"Doubled: {doubled}")
```

<details>
<summary>Answer</summary>

The bug is **in-place modification**—the function modifies the input list directly. After calling `double_values(original)`, both `original` and `doubled` point to the same modified list.

Output:
```
Original: [2, 4, 6]  # Changed!
Doubled: [2, 4, 6]
```

This violates the **principle of least surprise**—functions shouldn't modify their inputs unless that's explicitly their purpose. In Python, lists are **mutable** (can be changed), and when you pass a list to a function, you're passing a reference to the same list, not a copy.

Fix:
```python
def double_values_fixed(data):
    """Create new list without modifying input."""
    return [x * 2 for x in data]

# Or if you must modify in-place, make it clear:
def double_values_inplace(data):
    """Modifies data in-place (changes input!)."""
    for i in range(len(data)):
        data[i] *= 2
    # Don't return anything to signal in-place modification
```

This bug is common because Python's **pass-by-object-reference** behavior isn't always intuitive.
</details>

## Practice Exercises

### Exercise 9.1: Robust Data Reader

Create a function that safely reads numeric data from a file. Build it incrementally:

```python
# Part A: Handle missing files (5 lines)
def read_file_basic(filename):
    """Step 1: Handle missing files."""
    # Your code here
    pass

# Part B: Parse numbers safely (10 lines)
def read_numbers_safe(filename):
    """Step 2: Add number parsing."""
    # Build on Part A
    pass

# Part C: Skip invalid lines (15 lines)
def read_data_file(filename):
    """Step 3: Complete robust reader.
    
    Should:
    - Handle missing files gracefully
    - Skip invalid lines with warning
    - Return None if no valid data
    - Return list of floats if successful
    """
    # Your complete implementation
    pass

# Test cases:
# 1. test_missing.txt (doesn't exist)
# 2. test_empty.txt (empty file)
# 3. test_mixed.txt (numbers and text)
# 4. test_valid.txt (all valid numbers)
```

### Exercise 9.2: Validated Statistics Function

Build on Chapter 7 to create a robust statistics calculator:

```python
import numpy as np

def calculate_stats(data):
    """
    Calculate statistics with full validation.
    
    Should:
    - Validate input is numeric array
    - Handle empty arrays
    - Check for NaN and infinity
    - Warn about outliers (values > 3 std from mean)
    - Return dict with mean, std, min, max, n_valid
    
    Returns None if data cannot be processed.
    """
    # Your implementation here
    pass

# Test with:
test_cases = [
    [1, 2, 3, 4, 5],           # Normal
    [],                         # Empty
    [1, 2, np.nan, 4],         # Contains NaN
    [1, 2, 3, 100],            # Contains outlier
    [1, np.inf, 3],            # Contains infinity
]
```

### Exercise 9.3: Comprehensive Test Suite

Write thorough tests for this function:

```python
def find_peaks(data, threshold=0):
    """Find local maxima above threshold.
    
    A peak is a value greater than both neighbors
    and above the threshold.
    """
    if len(data) < 3:
        return []  # No peaks possible
    
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > threshold:
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
    
    return peaks

def test_find_peaks():
    """Write comprehensive tests.
    
    Should test:
    - Normal case with clear peaks
    - No peaks (monotonic data)
    - All peaks (zigzag data)
    - Edge cases (empty, single value, two values)
    - Threshold filtering
    - Plateau handling (consecutive equal values)
    """
    # Your tests here
    pass
```

### Exercise 9.4: Debug and Fix

This data processing pipeline has multiple bugs. Find and fix them:

```python
def process_sensor_data(readings, calibration_offset):
    """Process sensor readings with calibration.
    
    This function has 3 bugs. Find them using debugging
    techniques from the chapter.
    """
    
    # Apply calibration
    calibrated = []
    for reading in readings:
        calibrated.append(reading - calibration_offset)
    
    # Remove negative values (physically impossible)
    valid = []
    for i in range(len(calibrated)):
        if calibrated[i] >= 0:
            valid.append(calibrated[i])
    
    # Calculate statistics
    mean = sum(valid) / len(valid)
    variance = 0
    for value in valid:
        variance += (value - mean) ** 2
    variance = variance / len(valid) - 1
    
    return {
        'mean': mean,
        'variance': variance,
        'n_valid': len(valid),
        'n_rejected': len(readings) - len(valid)
    }

# Debug with these test cases:
test1 = process_sensor_data([10, 20, 30], 5)
test2 = process_sensor_data([1, 2, 3], 10)  # All become negative
test3 = process_sensor_data([], 0)  # Empty input
```

## Main Takeaways (Summary)

This chapter transformed you from writing hopeful code to creating robust, professional software. Here are the essential concepts you've mastered:

**Error Understanding**: You now read error messages systematically from bottom to top, understanding that **exceptions** are Python's way of communicating problems. Each error type (TypeError, ValueError, IndexError, KeyError) tells you something specific about what went wrong.

**Exception Handling**: You've learned to use **try/except blocks** to gracefully handle expected errors like missing files or invalid input, while letting programming errors crash loudly so you can fix them. The key principle: catch only what you expect and can handle.

**Input Validation**: You implement the **fail-fast principle** using **guard clauses** to check inputs at function boundaries. Validation happens in order of cost (cheap checks first) and catches problems before they corrupt results.

**Assertions as Documentation**: You use **assertions** to verify your code's logic and document assumptions. They're your safety net during development, catching mathematical impossibilities and numerical instabilities.

**Professional Logging**: You've replaced print statements with structured **logging** that provides timestamps, severity levels, and permanent records. This creates an audit trail for debugging long-running computations.

**Systematic Testing**: You write **test functions** that verify known values, mathematical properties, and edge cases. Tests prevent **regression**—old bugs reappearing when you modify code.

**Scientific Debugging**: You approach debugging like a scientist—observing symptoms, forming hypotheses, experimenting to test them, and analyzing results. **Binary search debugging** helps you quickly isolate problems in complex code.

The overarching theme: **defensive programming**. Every technique in this chapter helps you write code that anticipates problems, handles them gracefully, and helps you fix issues quickly when they arise. This is what separates scripts that work once from tools you can trust with your research.

## Definitions

**Anti-pattern**: A common but harmful coding pattern that should be avoided (e.g., bare except clauses).

**Assertion**: A debugging aid that verifies assumptions about program state; can be disabled in production.

**Bare except**: An except clause without specifying exception type; dangerous because it catches all errors.

**Binary search debugging**: Debugging technique that isolates problems by repeatedly dividing code in half.

**Call stack**: The sequence of function calls that led to the current point in execution.

**Catastrophic cancellation**: Numerical instability from subtracting nearly equal floating-point numbers.

**Context management**: Ensuring resources (like files) are properly acquired and released using 'with' statements.

**Defensive programming**: Writing code that anticipates and handles potential problems.

**Domain validation**: Checking if values make sense in your problem domain (e.g., positive temperatures).

**Edge case**: Unusual or boundary input that often reveals bugs.

**Error propagation**: How errors spread through calculations, potentially corrupting all downstream results.

**Exception**: Python's way of signaling that something exceptional has happened preventing normal execution.

**Exception handling**: Catching and responding to errors using try/except blocks.

**Fail-fast principle**: Detecting and reporting problems as early as possible.

**Guard clause**: Conditional statement at function start that checks preconditions and exits early if not met.

**In-place modification**: Changing data directly rather than creating a new copy.

**IndexError**: Exception raised when accessing a list index that doesn't exist.

**Input validation**: Checking that data meets requirements before processing.

**Integer overflow**: When a number exceeds the maximum value for its type.

**KeyError**: Exception raised when accessing a dictionary key that doesn't exist.

**Linear code flow**: Code structure that can be read top to bottom without nested conditions.

**Logging**: Systematic recording of program events with timestamps and severity levels.

**Logging level**: Filter controlling which log messages are displayed (DEBUG, INFO, WARNING, ERROR, CRITICAL).

**Mutation bug**: Error caused by modifying data while still using it.

**NameError**: Exception raised when referencing an undefined variable.

**Namespace**: The collection of currently defined variables and their values.

**NaN (Not a Number)**: Special floating-point value representing undefined results.

**Numerical stability**: Whether calculations maintain accuracy despite floating-point limitations.

**Off-by-one error**: Common bug from miscalculating array boundaries or forgetting zero-based indexing.

**Overhead**: Additional time or resource cost of an operation.

**Pass-by-object-reference**: Python's parameter passing mechanism where functions receive references to objects.

**Performance cost**: Time or resources required for an operation.

**Postcondition**: What a function guarantees about its output state.

**Precondition**: What must be true about input for a function to work correctly.

**Property-based testing**: Testing mathematical properties rather than specific values.

**Raise statement**: Explicitly creating and throwing an exception.

**Regression**: When previously fixed bugs reappear after code changes.

**Regression test**: Test that ensures old bugs don't reappear.

**Robust code**: Code that handles unexpected situations gracefully.

**Selective exception handling**: Only catching specific, expected exceptions.

**Test function**: Code that verifies other code works correctly.

**Testing**: Process of verifying code behaves as expected.

**Traceback**: Report showing the sequence of function calls leading to an error.

**Tradeoff**: Balancing competing concerns (e.g., safety vs performance).

**Try block**: Code section that might raise an exception.

**TypeError**: Exception raised when operation receives wrong type.

**Validation**: Checking that external input meets requirements.

**ValueError**: Exception raised when operation receives right type but wrong value.

**Zero-based indexing**: Numbering system where first element is at index 0.

## Key Takeaways

✅ **Error messages are maps to bugs** - Read from bottom (what went wrong) to top (where it happened) for quick diagnosis

✅ **Try/except handles expected failures** - Catch specific exceptions for files, network, and user input; let programming errors crash

✅ **Validation is your first defense** - Check inputs at function boundaries using the guard clause pattern to fail fast

✅ **Assertions verify your logic** - Use them to document assumptions and catch mathematical impossibilities during development

✅ **Logging provides persistent insight** - Replace print with logging for timestamps, severity levels, and permanent records

✅ **Tests prevent regression** - Simple tests of properties and edge cases catch bugs before they waste hours of debugging

✅ **Debugging is systematic science** - Follow observe→hypothesize→experiment→fix rather than random changes

✅ **Errors propagate and compound** - One unhandled error can corrupt entire pipelines; catch problems early

✅ **Real disasters come from missing validation** - Mars Climate Orbiter, Ariane 5, and other failures were preventable with proper error handling

## Quick Reference Tables

### Error Types and Meanings

| Exception | Meaning | Common Cause | Example |
|-----------|---------|--------------|---------|
| `NameError` | Variable undefined | Typo in name | `print(resuIt)` not `result` |
| `TypeError` | Wrong type | String not number | `"5" + 2` |
| `ValueError` | Invalid value | Outside range | `math.sqrt(-1)` |
| `IndexError` | Index too large | Off-by-one | `arr[len(arr)]` |
| `KeyError` | Missing dict key | Typo or absent | `dict['temp']` not there |
| `ZeroDivisionError` | Division by zero | Empty dataset | `sum([])/len([])` |
| `FileNotFoundError` | File missing | Wrong path | Wrong filename |

### Validation Strategy

| Check Type | Code Pattern | When to Use | Cost |
|------------|--------------|-------------|------|
| Empty check | `if not data:` | Always first | O(1) |
| Type check | `isinstance(x, type)` | Mixed inputs | O(1) |
| Range check | `min <= x <= max` | Physical limits | O(1) |
| NaN check | `np.isnan(x)` | Numerical data | O(n) |
| Uniqueness | `len(set(x)) == len(x)` | Duplicates bad | O(n) |

### Logging Best Practices

| Level | Use Case | Example Message |
|-------|----------|-----------------|
| `DEBUG` | Variable values | "Array shape: (100, 50)" |
| `INFO` | Normal progress | "Processing file 3 of 10" |
| `WARNING` | Concerning but OK | "Low sample size: n=5" |
| `ERROR` | Operation failed | "Cannot read config file" |
| `CRITICAL` | Must stop | "Database connection lost" |

### Testing Checklist

| Test Type | What to Test | Example |
|-----------|--------------|---------|
| Normal case | Common usage | Valid input range |
| Edge cases | Boundaries | Empty, single item |
| Error cases | Invalid input | Wrong type, NaN |
| Properties | Math invariants | Mean in [min, max] |
| Regression | Previous bugs | Specific failure case |

## Next Chapter Preview

Now that your code can handle errors gracefully, Chapter 10 will explore reading and writing scientific data formats. You'll learn to work with CSV files, JSON data, and binary formats like HDF5. The error handling skills from this chapter will be essential when dealing with external data files where formats might be inconsistent, values might be missing, and files might be corrupted.

You're building the foundation for robust scientific computing. Your code no longer just works—it works reliably, tells you when something's wrong, and helps you fix problems quickly. This transformation from hopeful code to professional code is what separates scripts that work once from tools you can trust with your research.