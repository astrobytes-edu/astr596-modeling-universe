# Chapter 2: Python as a Calculator & Basic Data Types

## Learning Objectives

By the end of this chapter, you will be able to:
- Use Python as an interactive scientific calculator with proper operator precedence
- Understand how computers represent integers, floats, and complex numbers in memory
- Explain why 0.1 + 0.2 ≠ 0.3 and handle floating-point comparisons correctly
- Recognize and avoid catastrophic cancellation and numerical overflow/underflow
- Choose appropriate numeric types for different computational scenarios
- Format output elegantly using f-strings with scientific notation and alignment
- Convert between data types safely and understand when conversions lose information
- Create defensive numerical code that catches precision problems early

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Launch IPython and use basic magic commands (Chapter 1)
- ✓ Understand the difference between scripts and interactive sessions (Chapter 1)
- ✓ Navigate your file system and activate your conda environment (Chapter 1)

## Chapter Overview

Before diving into complex simulations or data analysis, you need to understand how Python handles the fundamental building blocks of computation: numbers and text. This chapter explores Python as a scientific calculator, but more importantly, reveals the hidden complexity of numerical computation that can make or break your scientific results.

The floating-point precision issues we explore here aren't academic exercises — they're the source of real bugs that have delayed papers, corrupted simulations, and led to wrong scientific conclusions. Understanding these fundamentals now will save you weeks of debugging later when your orbital integrator accumulates errors or your Monte Carlo simulation produces impossible results.

### A Note on Units

Throughout this course, we'll use CGS (centimeter-gram-second) units, standard in astrophysics:
- Distances in centimeters (Earth-Sun distance: 1.496 × 10¹³ cm)
- Masses in grams (Solar mass: 1.989 × 10³³ g)  
- G = 6.67 × 10⁻⁸ cm³ g⁻¹ s⁻²

These huge numbers will teach you to work with scientific notation naturally and understand when numerical overflow becomes a real concern.

## 2.1 Python as Your Scientific Calculator

Open IPython (not the basic Python interpreter) to follow along:

```python
In [1]: 2 + 2
Out[1]: 4

In [2]: 10 / 3
Out[2]: 3.3333333333333335  # Note: not exactly 1/3!

In [3]: 2 ** 10  # Exponentiation
Out[3]: 1024
```

### Operator Precedence: A Source of Bugs

Python follows PEMDAS, but relying on memorized rules causes errors. Let's see with a real calculation:

```python
In [4]: # Calculate orbital velocity: v = sqrt(GM/r)
In [5]: G = 6.67e-8   # CGS units
In [6]: M = 1.989e33  # Solar mass in grams
In [7]: r = 1.496e13  # 1 AU in cm

In [8]: # WRONG - operator precedence error!
In [9]: v_wrong = G * M / r ** 0.5
In [10]: v_wrong
Out[10]: 27347197.71  # Way too fast!

In [11]: # CORRECT - parentheses clarify intent
In [12]: v_right = (G * M / r) ** 0.5
In [13]: v_right
Out[13]: 2978469.18  # ~30 km/s, Earth's orbital speed

In [14]: # Even clearer - break into steps
In [15]: gravitational_parameter = G * M
In [16]: v_clear = (gravitational_parameter / r) ** 0.5
```

The wrong version calculated (GM/√r) instead of √(GM/r) — a factor of √r error!

**⏸️ Pause and Predict**: What will `-2 ** 2` evaluate to?

<details>
<summary>Answer</summary>

`-4` (not `4`). Exponentiation happens before the negative sign, so this is `-(2²)`. For squaring negative numbers, use `(-2) ** 2`.

</details>

### Complete Arithmetic Operators

```python
In [17]: 17 / 3   # True division (always float)
Out[17]: 5.666666666666667

In [18]: 17 // 3  # Floor division
Out[18]: 5

In [19]: 17 % 3   # Modulo (remainder)
Out[19]: 2

In [20]: -17 // 3  # Warning: rounds toward -infinity!
Out[20]: -6  # Not -5!
```

## 2.2 How Python Stores Numbers: Critical for Scientific Computing

Understanding number representation prevents subtle bugs that can destroy your simulations.

### Integers: Arbitrary Precision

Python integers have unlimited precision:

```python
In [21]: googol = 10 ** 100
In [22]: googol + 1  # Still exact!
Out[22]: 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

In [23]: import sys
In [24]: sys.getsizeof(42)
Out[24]: 28  # Small int: 28 bytes

In [25]: sys.getsizeof(googol)
Out[25]: 72  # Big int: 72 bytes
```

This is why NumPy arrays are crucial later — a million Python integers would use ~28MB, while a NumPy array uses ~4MB.

### Floating-Point: The Heart of Numerical Computing

Floats use IEEE 754 representation: 64 bits split into sign (1 bit), exponent (11 bits), and mantissa (52 bits). This creates fundamental limitations:

```python
In [26]: 0.1 + 0.2
Out[26]: 0.30000000000000004  # Not 0.3!

In [27]: 0.1 + 0.2 == 0.3
Out[27]: False  # Never use == with floats!
```

Why does this happen? Binary can't represent 0.1 exactly, just like decimal can't represent 1/3 exactly:

```
Decimal: 1/3 = 0.33333... (repeating forever)
Binary:  1/10 = 0.0001100110011... (repeating forever)

What's actually stored for 0.1:
0.1000000000000000055511151231257827...

So: 0.1 + 0.2 = 0.3000000000000000444089209850062616...
But 0.3 stored = 0.2999999999999999888977697537484345...

They're different in the 17th decimal place!
```

### Machine Epsilon: The Smallest Distinguishable Difference

```python
In [28]: import sys
In [29]: sys.float_info.epsilon
Out[29]: 2.220446049250313e-16

In [30]: 1.0 + 1e-16 == 1.0
Out[30]: True  # Too small to detect!

In [31]: 1.0 + 1e-15 == 1.0
Out[31]: False  # Large enough to matter
```

This matters when checking convergence in iterative algorithms. You can't get precision better than machine epsilon.

### Safe Floating-Point Comparisons

```python
In [32]: # WRONG
In [33]: if velocity == 299792458.0:  # Speed of light
   ...:     print("At light speed!")

In [34]: # CORRECT - absolute tolerance
In [35]: if abs(velocity - 299792458.0) < 1.0:  # Within 1 m/s
   ...:     print("Effectively at light speed!")

In [36]: # BETTER - relative tolerance
In [37]: import math
In [38]: if math.isclose(velocity, 299792458.0, rel_tol=1e-9):
   ...:     print("At light speed within relative tolerance!")
```

### Catastrophic Cancellation: When Subtraction Destroys Precision

```python
In [39]: # Computing 1 - cos(x) for small x
In [40]: import math
In [41]: x = 1e-8

In [42]: # Direct computation - catastrophic cancellation!
In [43]: math.cos(x)
Out[43]: 0.9999999999999999  # Lost most precision

In [44]: 1 - math.cos(x)
Out[44]: 0.0  # Complete precision loss!

In [45]: # Better: use mathematical identity
In [46]: 2 * math.sin(x/2) ** 2
Out[46]: 4.999999999999999e-17  # Maintains precision!
```

This appears in orbital mechanics when computing small changes in energy or angular momentum.

### Overflow and Underflow in Astronomical Calculations

```python
In [47]: sys.float_info.max
Out[47]: 1.7976931348623157e+308  # Largest float

In [48]: sys.float_info.min
Out[48]: 2.2250738585072014e-308  # Smallest positive float

In [49]: # Overflow example
In [50]: L_sun = 3.828e33  # Solar luminosity (erg/s)
In [51]: n_galaxies = 1e12
In [52]: L_universe = L_sun * 1e11 * n_galaxies
Out[52]: inf  # Overflow to infinity!

In [53]: # Underflow example
In [54]: probability = 1e-200
In [55]: prob_squared = probability ** 2
Out[55]: 0.0  # Underflow to zero!

In [56]: # Solution: work in log space
In [57]: log_prob = math.log10(probability)
Out[57]: -200.0
In [58]: log_prob_squared = 2 * log_prob
Out[58]: -400.0  # Maintains precision in log space
```

### Defensive Programming with Numerical Checks

Build habits that catch numerical problems early:

```python
In [59]: def safe_divide(a, b, epsilon=1e-10):
   ...:     """Division with zero check."""
   ...:     if abs(b) < epsilon:
   ...:         raise ValueError(f"Division by near-zero: {b}")
   ...:     return a / b

In [60]: def check_finite(value, name="value"):
   ...:     """Ensure value is finite (not inf or nan)."""
   ...:     if not math.isfinite(value):
   ...:         raise ValueError(f"{name} is not finite: {value}")
   ...:     return value
```

## 2.3 Complex Numbers for Wave Physics

Python handles complex numbers natively:

```python
In [61]: z = 3 + 4j  # Engineers use j, physicists use i
In [62]: abs(z)  # Magnitude
Out[62]: 5.0

In [63]: import cmath
In [64]: cmath.phase(z)  # Phase in radians
Out[64]: 0.9272952180016122

In [65]: # Euler's formula: e^(iπ) = -1
In [66]: cmath.exp(1j * math.pi)
Out[66]: (-1+1.2246467991473532e-16j)  # Small imaginary part is roundoff
```

## 2.4 Variables and Assignment

Variables in Python are names that refer to objects:

```python
In [67]: mass = 1.989e33  # Solar mass in grams
In [68]: radius = 6.96e10  # Solar radius in cm

In [69]: # Calculate density
In [70]: volume = (4/3) * math.pi * radius**3
In [71]: density = mass / volume
In [72]: print(f"Solar density: {density:.2f} g/cm³")
Solar density: 1.41 g/cm³
```

Assignment doesn't copy values; it creates references:

```python
In [73]: a = [1, 2, 3]
In [74]: b = a  # b refers to SAME list
In [75]: b.append(4)
In [76]: a  # a changed too!
Out[76]: [1, 2, 3, 4]
```

## 2.5 Strings and Formatting

Strings are immutable sequences of characters:

```python
In [77]: star = "Betelgeuse"
In [78]: star[0]  # Indexing
Out[78]: 'B'
In [79]: star[-1]  # Negative indexing from end
Out[79]: 'e'
In [80]: star[0:5]  # Slicing
Out[80]: 'Betel'
```

### F-Strings: Modern Python Formatting

F-strings (formatted string literals) are the preferred way to format output:

```python
In [81]: # Basic f-string
In [82]: object_name = "M31"
In [83]: distance = 2.537e6  # light-years
In [84]: print(f"The {object_name} galaxy is {distance:.2e} light-years away")
The M31 galaxy is 2.54e+06 light-years away

In [85]: # Format specifications
In [86]: x = 1234.56789
In [87]: print(f"{x:.2f}")   # 2 decimal places
1234.57
In [88]: print(f"{x:.2e}")   # Scientific notation
1.23e+03
In [89]: print(f"{x:10.2f}") # Width 10, 2 decimals
   1234.57
In [90]: print(f"{x:,.0f}")  # Thousands separator
1,235

In [91]: # Debugging with = (Python 3.8+)
In [92]: velocity = 29784.7
In [93]: print(f"{velocity=}")  # Shows name and value
velocity=29784.7
```

Common f-string patterns for scientific computing:

```python
In [94]: # Aligning columns
In [95]: for i, (name, mag) in enumerate([("Sirius", -1.46), ("Canopus", -0.74)]):
   ...:     print(f"{i:2d}. {name:15s} {mag:6.2f}")
 0. Sirius          -1.46
 1. Canopus         -0.74

In [96]: # Percentage formatting
In [97]: efficiency = 0.8732
In [98]: print(f"Efficiency: {efficiency:.1%}")
Efficiency: 87.3%
```

## 2.6 Type System and Conversions

Python is dynamically typed but strongly typed:

```python
In [99]: # Type checking
In [100]: type(42)
Out[100]: int
In [101]: isinstance(3.14, float)
Out[101]: True

In [102]: # Type conversion
In [103]: int(3.14)  # Truncates toward zero
Out[103]: 3
In [104]: int(-3.14)
Out[104]: -3  # Not -4!

In [105]: float("1.23e-4")
Out[105]: 0.000123

In [106]: # Common error
In [107]: "Distance: " + 2.5  # TypeError!
In [108]: # Fix with f-string
In [109]: f"Distance: {2.5}"
Out[109]: 'Distance: 2.5'
```

## 2.7 Booleans and None

Understanding truthiness is crucial for scientific computing:

```python
In [110]: bool(0)
Out[110]: False
In [111]: bool(0.0)
Out[111]: False
In [112]: bool(1e-100)  # Tiny but not zero!
Out[112]: True

In [113]: # Common bug in scientific code
In [114]: error = 1e-15  # Tiny numerical error
In [115]: if error:  # WRONG - triggers for any non-zero!
   ...:     print("Error detected!")
Error detected!

In [116]: # CORRECT
In [117]: threshold = 1e-10
In [118]: if error > threshold:
   ...:     print("Significant error!")
# No output - error below threshold

In [119]: # None checks
In [120]: result = None
In [121]: if result is None:  # Correct way to check
   ...:     print("No result yet")
```

## 2.8 The Math Module

Essential mathematical functions for scientific computing:

```python
In [122]: import math
In [123]: math.pi
Out[123]: 3.141592653589793
In [124]: math.e
Out[124]: 2.718281828459045

In [125]: # Trigonometry (radians)
In [126]: math.sin(math.pi / 6)
Out[126]: 0.49999999999999994  # Should be 0.5 exactly

In [127]: # Logarithms
In [128]: math.log(math.e)  # Natural log
Out[128]: 1.0
In [129]: math.log10(1000)  # Base-10 log
Out[129]: 3.0
In [130]: math.log2(1024)  # Base-2 log
Out[130]: 10.0

In [131]: # Special functions
In [132]: math.gamma(5)  # Gamma function: (n-1)!
Out[132]: 24.0
In [133]: math.erf(1)  # Error function
Out[133]: 0.8427007929497149
```

## 2.9 From Interactive to Script

Convert your IPython explorations into reusable scripts:

```python
#!/usr/bin/env python
"""
stellar_calculations.py
Demonstrate numerical calculations for stellar physics.
"""

import math

def schwarzschild_radius(mass_grams):
    """Calculate Schwarzschild radius in cm.
    
    Rs = 2GM/c^2
    """
    G = 6.67e-8  # cm^3 g^-1 s^-2
    c = 2.998e10  # cm/s
    
    # Check for numerical issues
    if mass_grams <= 0:
        raise ValueError(f"Mass must be positive: {mass_grams}")
    
    rs = 2 * G * mass_grams / c**2
    
    # Check result is reasonable
    if not math.isfinite(rs):
        raise ValueError(f"Calculation overflow for mass {mass_grams}")
    
    return rs

if __name__ == "__main__":
    # Test with solar mass
    M_sun = 1.989e33  # grams
    rs_sun = schwarzschild_radius(M_sun)
    
    print(f"Solar mass: {M_sun:.3e} g")
    print(f"Schwarzschild radius: {rs_sun:.3e} cm")
    print(f"That's {rs_sun/1e5:.1f} km")
    
    # Test with Earth mass
    M_earth = 5.972e27  # grams
    rs_earth = schwarzschild_radius(M_earth)
    print(f"\nEarth would need to be compressed to {rs_earth:.2f} cm")
```

## Practice Exercises

### Exercise 2.1: Numerical Precision Investigation

```python
def explore_precision():
    """
    Investigate floating-point precision limits.
    
    Tasks:
    1. Find a case where a + b == a even though b != 0
    2. Find the distance where parallax < machine epsilon
    3. Demonstrate loss of precision in variance calculation
    """
    # Your code here
    pass
```

### Exercise 2.2: Safe Numerical Functions

```python
def safe_magnitude(values):
    """
    Calculate magnitude avoiding overflow/underflow.
    
    For values = [v1, v2, ..., vn]
    Return sqrt(v1^2 + v2^2 + ... + vn^2)
    
    Handle cases where direct calculation would overflow.
    Hint: Factor out the largest value.
    """
    # Your code here
    pass
```

### Exercise 2.3: Format Scientific Output

```python
def format_stellar_data(stars):
    """
    Create formatted table of stellar data.
    
    Input: List of (name, mass, luminosity) tuples
    Output: Formatted table with proper alignment and units
    
    Use f-strings to create professional-looking output.
    """
    # Your code here
    pass
```

## Key Takeaways

Floating-point arithmetic is approximate by design. Never use `==` to compare floats; always use a tolerance. This isn't a Python quirk — it's fundamental to how computers work.

Catastrophic cancellation occurs when subtracting nearly equal numbers. Use mathematical identities to avoid it. This will matter when computing small changes in conserved quantities.

Overflow and underflow are real concerns in astronomy with its extreme scales. Know when to work in log space or rescale your units.

Machine epsilon sets the fundamental limit of floating-point precision. You cannot distinguish numbers closer than ~2.2e-16 relative difference.

Defensive programming with explicit checks catches numerical problems early. A simple assertion can save weeks of debugging corrupted simulations.

These concepts aren't academic — they're the difference between simulations that conserve energy and ones that explode, between detecting exoplanets and missing them due to numerical noise.

## Quick Reference: New Functions and Commands

| Function/Method | Purpose | Example |
|----------------|---------|---------|
| `**` | Exponentiation | `2 ** 10` → `1024` |
| `//` | Floor division | `17 // 3` → `5` |
| `%` | Modulo (remainder) | `17 % 3` → `2` |
| `abs()` | Absolute value | `abs(-3.14)` → `3.14` |
| `round()` | Round to n decimals | `round(3.14159, 2)` → `3.14` |
| `int()` | Convert to integer | `int(3.14)` → `3` |
| `float()` | Convert to float | `float("1.23e-4")` → `0.000123` |
| `complex()` | Create complex number | `complex(3, 4)` → `(3+4j)` |
| `math.isclose()` | Safe float comparison | `math.isclose(0.1+0.2, 0.3)` |
| `math.isfinite()` | Check not inf/nan | `math.isfinite(result)` |
| `math.isnan()` | Check for NaN | `math.isnan(value)` |
| `math.isinf()` | Check for infinity | `math.isinf(value)` |
| `math.pi` | π constant | `3.141592653589793` |
| `math.e` | e constant | `2.718281828459045` |
| `math.sin()` | Sine (radians) | `math.sin(math.pi/2)` → `1.0` |
| `math.cos()` | Cosine (radians) | `math.cos(0)` → `1.0` |
| `math.log()` | Natural logarithm | `math.log(math.e)` → `1.0` |
| `math.log10()` | Base-10 logarithm | `math.log10(1000)` → `3.0` |
| `math.sqrt()` | Square root | `math.sqrt(2)` → `1.414...` |
| `math.exp()` | Exponential | `math.exp(1)` → `2.718...` |
| `cmath.phase()` | Complex phase | `cmath.phase(1+1j)` → `0.785...` |
| `sys.float_info` | Float limits | `.max`, `.min`, `.epsilon` |
| `f"{x:.2f}"` | Format 2 decimals | `f"{3.14159:.2f}"` → `"3.14"` |
| `f"{x:.2e}"` | Scientific notation | `f"{1234:.2e}"` → `"1.23e+03"` |
| `f"{x:10.2f}"` | Width and decimals | `f"{3.14:10.2f}"` → `"      3.14"` |

## Next Chapter Preview

With a solid understanding of Python's type system and numerical precision, Chapter 3 will introduce control flow — the if statements and loops that make your code dynamic. You'll learn to write pseudocode first, implement algorithms systematically, and handle the special challenges of floating-point comparisons in conditional statements. The numerical foundations from this chapter will be essential when you're checking convergence criteria or detecting numerical instabilities in your simulations.