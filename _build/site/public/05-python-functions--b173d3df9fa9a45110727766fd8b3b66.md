# Chapter 5: Functions & Modules - Building Reusable Scientific Code

## Learning Objectives

By the end of this chapter, you will be able to:
- Design functions as clear contracts with well-defined inputs and outputs
- Understand Python's scope rules and how they affect variable access
- Write functions with flexible parameter handling using *args and **kwargs
- Apply functional programming patterns like map, filter, and lambda functions
- Create and import your own modules for code organization
- Document functions properly using docstrings
- Recognize and avoid common function-related bugs
- Build modular, reusable code for scientific applications

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Write loops and conditionals fluently (Chapter 3)
- ✓ Choose appropriate data structures for different tasks (Chapter 4)
- ✓ Handle floating-point arithmetic safely (Chapter 2)
- ✓ Use IPython for testing and timing code (Chapter 1)
- ✓ Design algorithms with pseudocode (Chapter 3)

## Chapter Overview

Functions are the fundamental building blocks of organized code. Without functions, you'd be copying and pasting the same code repeatedly, making bugs harder to fix and improvements impossible to maintain. But functions are more than just a way to avoid repetition — they're how we create abstractions, manage complexity, and build reliable software.

This chapter teaches you to think about functions as contracts between different parts of your code. When you write a function that converts temperature units, you're creating a promise: given a valid temperature in one unit, the function will reliably return the equivalent in another unit. This contract mindset helps you write functions that others (including future you) can trust and use effectively.

We'll explore Python's scope rules, which determine where variables can be accessed, and learn how seemingly simple concepts like default arguments can create subtle bugs. You'll discover how Python's flexible parameter system enables powerful interfaces, and how functional programming concepts prepare you for modern scientific computing frameworks. By the end, you'll be organizing your code into modules that can be shared, tested, and maintained professionally.

## 5.1 Defining Functions: The Basics

A function encapsulates a piece of logic that transforms inputs into outputs. Think of a function as a machine: you feed it raw materials (inputs), it performs some process (the function body), and it produces a product (output). In programming terms, a function takes arguments, executes code, and returns results.

### Your First Function

Let's start with the simplest possible function and understand every part:

```python
In [1]: def celsius_to_fahrenheit(celsius):
   ...:     """Convert Celsius to Fahrenheit."""
   ...:     fahrenheit = celsius * 9/5 + 32
   ...:     return fahrenheit

In [2]: # Using the function
In [3]: temp_f = celsius_to_fahrenheit(25)
In [4]: print(f"25°C = {temp_f}°F")
25°C = 77.0°F
```

Let's break down the anatomy of this function:

1. **`def` keyword**: Tells Python we're defining a function
2. **Function name** (`celsius_to_fahrenheit`): Follows snake_case convention, describes what it does
3. **Parameters** (`celsius`): Variables that receive values when function is called
4. **Docstring**: Brief description of what the function does (always include this!)
5. **Function body**: Indented code that does the actual work
6. **`return` statement**: Sends a value back to whoever called the function

When Python executes `celsius_to_fahrenheit(25)`, it creates a temporary namespace where `celsius = 25`, runs the function body, and returns the result (77.0).

### Functions Without Return Values

Not all functions return values. Some perform actions like printing or modifying external state:

```python
In [5]: def print_statistics(numbers):
   ...:     """Print basic statistics for a list of numbers."""
   ...:     if not numbers:  # Handle empty list case
   ...:         print("No data provided")
   ...:         return  # Early exit, returns None
   ...:     
   ...:     print(f"Count: {len(numbers)}")
   ...:     print(f"Min: {min(numbers)}")
   ...:     print(f"Max: {max(numbers)}")
   ...:     print(f"Average: {sum(numbers)/len(numbers):.2f}")

In [6]: data = [23, 45, 67, 89, 12]
In [7]: result = print_statistics(data)
Count: 5
Min: 12
Max: 89
Average: 47.20

In [8]: print(result)
None  # Functions without return automatically return None
```

Every Python function returns something. If you don't explicitly return a value, Python returns `None`. This is Python's way of representing "nothing" or "no value."

### 🔍 **Check Your Understanding**

What will this code print?

```python
def double(x):
    x * 2  # No return statement!

result = double(5)
print(result)
```

<details>
<summary>Answer</summary>

It prints `None`. The function calculates `x * 2` but doesn't return it. Without an explicit `return` statement, Python functions return `None`.

To fix it:
```python
def double(x):
    return x * 2  # Now it returns the value
```

</details>

### Returning Multiple Values

Python functions can return multiple values using tuples:

```python
In [9]: def convert_temperature(value, from_unit):
   ...:     """Convert temperature to both Celsius and Fahrenheit."""
   ...:     if from_unit == 'C':
   ...:         celsius = value
   ...:         fahrenheit = value * 9/5 + 32
   ...:     elif from_unit == 'F':
   ...:         fahrenheit = value
   ...:         celsius = (value - 32) * 5/9
   ...:     else:
   ...:         raise ValueError(f"Unknown unit: {from_unit}")
   ...:     
   ...:     return celsius, fahrenheit  # Returns a tuple

In [10]: # Unpack the returned tuple
In [11]: c, f = convert_temperature(100, 'C')
In [12]: print(f"100°C = {c}°C = {f}°F")
100°C = 100°C = 212.0°F
```

### The Design Process: From Problem to Function

Before writing any function, you should design it first. This means thinking through what the function needs to do, what inputs it requires, what output it produces, and what could go wrong. This design-first approach prevents the common mistake of coding yourself into a corner.

```python
"""
PSEUDOCODE: Design a function to validate measurement data

FUNCTION validate_measurement(value, min_valid, max_valid):
    INPUT: measurement value, valid range boundaries
    OUTPUT: boolean indicating if measurement is valid
    
    IF value is not a number:
        RETURN False
    IF value < min_valid OR value > max_valid:
        RETURN False
    IF value is NaN or Infinity:
        RETURN False
    RETURN True
"""

# Implementation following the design
import math

def validate_measurement(value, min_valid, max_valid):
    """
    Check if a measurement falls within valid range.
    
    Parameters
    ----------
    value : float
        The measurement to validate
    min_valid : float
        Minimum acceptable value
    max_valid : float
        Maximum acceptable value
    
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    # Check if it's a number
    if not isinstance(value, (int, float)):
        return False
    
    # Check for special values (NaN, infinity)
    if not math.isfinite(value):
        return False
    
    # Check range
    return min_valid <= value <= max_valid

# Test the function with various inputs
measurements = [23.5, -999, float('inf'), 'bad', 45.2]
valid_range = (0, 100)

for m in measurements:
    is_valid = validate_measurement(m, *valid_range)
    print(f"{m}: {'Valid' if is_valid else 'Invalid'}")
```

Notice how the pseudocode clarifies our thinking before we write Python. This approach helps you catch design problems early — much easier than debugging complex code later.

### 📦 **Computational Thinking Box: Function Design Principles**

```
PATTERN: Function Contract Design

A well-designed function follows these principles:

1. Single Responsibility
   - Does ONE thing well
   - Name clearly indicates what it does

2. Clear Interface
   - Parameters are obvious
   - Return value is predictable
   
3. Defensive Programming
   - Validates inputs
   - Handles edge cases
   
4. No Surprises
   - No hidden side effects
   - Behavior matches name

Example progression:
BAD:  process(data, flag=True)  # What does it do?
OKAY: calculate_mean(numbers)   # Clear but limited
GOOD: calculate_mean(numbers, ignore_nan=False)  # Flexible and clear
```

## 5.2 Function Arguments In-Depth

Python provides flexible ways to handle function parameters, from simple positional arguments to sophisticated keyword-only parameters. Understanding these mechanisms allows you to create functions that are both powerful and easy to use.

### Positional vs Keyword Arguments

When you call a function, you can pass arguments by position or by name. Positional arguments must appear in the order defined by the function. Keyword arguments can appear in any order because you specify which parameter each value corresponds to.

```python
In [13]: def calculate_density(mass, volume, units='g/cm³'):
   ...:     """Calculate density from mass and volume."""
   ...:     if volume == 0:
   ...:         raise ValueError("Volume cannot be zero")
   ...:     density = mass / volume
   ...:     return f"{density:.2f} {units}"

In [14]: # Different ways to call the same function
In [15]: calculate_density(100, 50)  # Positional only
Out[15]: '2.00 g/cm³'

In [16]: calculate_density(volume=50, mass=100)  # Keyword (any order!)
Out[16]: '2.00 g/cm³'

In [17]: calculate_density(100, 50, units='kg/m³')  # Mixed
Out[17]: '2.00 kg/m³'
```

Keyword arguments make function calls more readable, especially when a function has many parameters. Compare `process(data, True, False, 10)` with `process(data, normalize=True, validate=False, threshold=10)` — the second version is self-documenting.

### Default Arguments and the Mutable Default Trap

Default arguments allow functions to be called with fewer arguments than they're defined with. This makes functions more flexible and easier to use. However, there's a critical trap that catches even experienced programmers: mutable default arguments.

Python evaluates default arguments once when the function is defined, not each time it's called. This seems like a minor implementation detail, but it creates one of Python's most notorious bugs:

```python
In [18]: # THE TRAP - Mutable default
In [19]: def add_measurement(value, data_list=[]):  # DANGER!
   ...:     """Add measurement to list - BUGGY VERSION."""
   ...:     data_list.append(value)
   ...:     return data_list

In [20]: list1 = add_measurement(10)
In [21]: print(f"First call: {list1}")
First call: [10]

In [22]: list2 = add_measurement(20)  # Surprise!
In [23]: print(f"Second call: {list2}")
Second call: [10, 20]  # Contains both values!

In [24]: list1 is list2  # They're the same object!
Out[24]: True
```

What happened? Python created the default list `[]` once when the function was defined. Every call that uses the default gets the same list object. When we modify it, we're modifying the one shared list that all calls reference.

The fix uses `None` as a sentinel value — a placeholder that signals "no value provided":

```python
In [25]: def add_measurement_fixed(value, data_list=None):
   ...:     """Add measurement to list - CORRECT VERSION."""
   ...:     if data_list is None:
   ...:         data_list = []  # Create new list each time
   ...:     data_list.append(value)
   ...:     return data_list

In [26]: list1 = add_measurement_fixed(10)
In [27]: list2 = add_measurement_fixed(20)
In [28]: print(f"First: {list1}, Second: {list2}")
First: [10], Second: [20]  # Separate lists as expected!
```

This pattern — using `None` as a default for mutable arguments — is so common it's considered standard Python idiom. You'll see it throughout scientific libraries and should always use it in your own code.

### ⚠️ **Common Bug Alert**

```python
import time

# This captures the time when function is DEFINED, not called!
def log_event(message, timestamp=time.time()):  # BUG!
    print(f"[{timestamp}] {message}")

# All calls will have the same timestamp!

# CORRECT approach:
def log_event_fixed(message, timestamp=None):
    if timestamp is None:
        timestamp = time.time()  # Evaluated when called
    print(f"[{timestamp}] {message}")
```

### Variable-Length Arguments (*args)

Sometimes you don't know how many arguments a function will receive. For example, a function that calculates the mean could work with 2 numbers or 200. Python's `*args` syntax collects any number of positional arguments into a tuple:

```python
In [29]: def calculate_mean(*values):
   ...:     """Calculate mean of any number of values."""
   ...:     if not values:
   ...:         raise ValueError("At least one value required")
   ...:     return sum(values) / len(values)

In [30]: calculate_mean(10)
Out[30]: 10.0

In [31]: calculate_mean(10, 20, 30)
Out[31]: 20.0

In [32]: calculate_mean(10, 20, 30, 40, 50)
Out[32]: 30.0

In [33]: # How it works internally
In [34]: def show_args(*args):
   ...:     print(f"args is a {type(args)}: {args}")

In [35]: show_args(1, 2, 3)
args is a <class 'tuple'>: (1, 2, 3)
```

The asterisk (`*`) tells Python to collect all remaining positional arguments into a tuple called `args`. You can name it anything (`*values`, `*numbers`), but `*args` is the conventional name. This pattern is particularly useful for mathematical functions that naturally work with varying numbers of inputs.

### Keyword Arguments (**kwargs)

Just as `*args` collects positional arguments, `**kwargs` collects keyword arguments into a dictionary. This enables incredibly flexible interfaces where users can specify only the options they care about:

```python
In [36]: def create_plot(x, y, **options):
   ...:     """Create a plot with flexible options."""
   ...:     print(f"Plotting {len(x)} points")
   ...:     print("Options provided:")
   ...:     for key, value in options.items():
   ...:         print(f"  {key}: {value}")

In [37]: create_plot([1, 2, 3], [4, 5, 6], 
   ...:              title="My Plot", 
   ...:              color='red',
   ...:              linewidth=2)
Plotting 3 points
Options provided:
  title: My Plot
  color: red
  linewidth: 2
```

The double asterisk (`**`) tells Python to collect all keyword arguments into a dictionary. This pattern appears throughout scientific libraries where functions need many optional parameters. Rather than defining dozens of parameters with defaults, libraries use `**kwargs` to accept any configuration option.

### Combining Different Argument Types

```python
def flexible_function(required, *args, default=10, **kwargs):
    """
    Demonstrates all parameter types.
    
    Parameters:
    - required: positional, required
    - *args: variable positional
    - default: keyword with default
    - **kwargs: variable keyword
    """
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Default: {default}")
    print(f"Kwargs: {kwargs}")

# Examples of calling it
flexible_function(1)
# Required: 1, Args: (), Default: 10, Kwargs: {}

flexible_function(1, 2, 3, default=20, extra='test')
# Required: 1, Args: (2, 3), Default: 20, Kwargs: {'extra': 'test'}
```

### 🔍 **Check Your Understanding**

What's wrong with this function definition?

```python
def process_data(default=5, *values, **options):
    # Process the data
    pass
```

<details>
<summary>Answer</summary>

The order is wrong! Python requires this order:
1. Regular positional parameters
2. *args
3. Keyword parameters with defaults
4. **kwargs

Correct version:
```python
def process_data(*values, default=5, **options):
    # Process the data
    pass
```

</details>

## 5.3 Scope and Namespaces

Understanding scope — where variables can be accessed — is crucial for writing bug-free code. Python's scope rules determine which variables are visible at any point in your program. Without understanding scope, you'll encounter confusing bugs where variables don't have the values you expect, or worse, where changing a variable in one place mysteriously affects code elsewhere.

### The LEGB Rule

Python resolves variable names using the LEGB rule, searching in this order:
- **L**ocal: Inside the current function
- **E**nclosing: In the enclosing function (for nested functions)  
- **G**lobal: At the top level of the module
- **B**uilt-in: In the built-in namespace (print, len, etc.)

Python stops searching as soon as it finds a match. This means a local variable can "shadow" (hide) a global variable with the same name:

```python
In [38]: # Demonstrating LEGB
In [39]: x = "global"  # Global scope

In [40]: def outer():
   ...:     x = "enclosing"  # Enclosing scope
   ...:     
   ...:     def inner():
   ...:         x = "local"  # Local scope
   ...:         print(f"Inner sees: {x}")
   ...:     
   ...:     inner()
   ...:     print(f"Outer sees: {x}")

In [41]: outer()
Inner sees: local
Outer sees: enclosing

In [42]: print(f"Global sees: {x}")
Global sees: global
```

Each function creates its own namespace — a mapping of names to objects. When you use a variable, Python searches through these namespaces in LEGB order until it finds the name.

### Variable Scope Visualization

Let's trace how Python finds variables in a more complex example:

```python
# Let's trace how Python finds variables

global_var = 100  # Global scope

def function_a():
    local_var = 200  # Local to function_a
    
    def function_b():
        nested_var = 300  # Local to function_b
        # Can access all three levels
        total = global_var + local_var + nested_var
        return total
    
    return function_b()

result = function_a()  # Returns 600
```

Here's how Python resolves each variable in the line `total = global_var + local_var + nested_var`:

```
Variable Resolution Process:

Looking for 'global_var':
  Local (function_b): Not found
  Enclosing (function_a): Not found
  Global: Found! Value = 100

Looking for 'local_var':
  Local (function_b): Not found
  Enclosing (function_a): Found! Value = 200

Looking for 'nested_var':
  Local (function_b): Found! Value = 300
```

### The Global Statement (Use Sparingly!)

```python
In [43]: counter = 0  # Global variable

In [44]: def increment_wrong():
   ...:     counter += 1  # UnboundLocalError!
   ...:     return counter

In [45]: def increment_with_global():
   ...:     global counter
   ...:     counter += 1  # Now modifies global
   ...:     return counter

In [46]: def increment_better(current_count):
   ...:     """Better approach - no global state."""
   ...:     return current_count + 1
```

### 📦 **Computational Thinking Box: Why Global Variables Are Dangerous**

```
PROBLEMS with global variables:

1. Hidden Dependencies
   - Function behavior depends on external state
   - Can't understand function in isolation

2. Testing Nightmare
   - Must set up global state before testing
   - Tests can interfere with each other

3. Debugging Difficulty
   - Value could be changed anywhere
   - Hard to track down bugs

4. No Parallelization
   - Multiple threads accessing same global = race conditions

Better approach: Pass state explicitly
BAD:  temperature = 100; adjust_temp()
GOOD: new_temp = adjust_temp(current_temp)
```

### Closures: Functions That Remember

```python
In [47]: def create_multiplier(factor):
   ...:     """Create a function that multiplies by factor."""
   ...:     def multiplier(x):
   ...:         return x * factor  # 'Closes over' factor
   ...:     return multiplier

In [48]: double = create_multiplier(2)
In [49]: triple = create_multiplier(3)

In [50]: double(10)
Out[50]: 20

In [51]: triple(10)
Out[51]: 30
```

### 🔍 **Check Your Understanding**

What will this print?

```python
x = 10

def modify():
    x = 20
    
def modify_global():
    global x
    x = 30

modify()
print(x)
modify_global()
print(x)
```

<details>
<summary>Answer</summary>

```
10  # modify() creates local x, doesn't affect global
30  # modify_global() changes the global x
```

The first function creates a local variable named `x` that shadows the global. The second explicitly modifies the global variable.

</details>

## 5.4 Functional Programming Elements

Python supports functional programming — a style that treats computation as the evaluation of mathematical functions. While Python isn't a pure functional language like Haskell, it provides powerful functional features that lead to cleaner, more maintainable code. These concepts are especially important because they prepare you for modern scientific computing frameworks like JAX that embrace functional programming.

### Lambda Functions

A lambda function is a small, anonymous function that you can define inline. Think of it as a function without a name, useful when you need a simple function just once. The syntax is `lambda arguments: expression`.

```python
In [52]: # Regular function
In [53]: def square(x):
   ...:     return x ** 2

In [54]: # Equivalent lambda
In [55]: square_lambda = lambda x: x ** 2

In [56]: square(5) == square_lambda(5)
Out[56]: True

In [57]: # Lambdas are most useful as arguments to other functions
In [58]: data = [(1, 'z'), (3, 'a'), (2, 'b')]
In [59]: sorted(data)  # Default: sorts by first element
Out[59]: [(1, 'z'), (2, 'b'), (3, 'a')]

In [60]: sorted(data, key=lambda x: x[1])  # Sort by second element
Out[60]: [(3, 'a'), (2, 'b'), (1, 'z')]
```

Lambda functions are limited to single expressions — you can't use statements like `if/else` blocks or loops. This limitation is intentional: lambdas are meant for simple operations. If you need anything complex, write a regular function.

### Map, Filter, and Reduce

These three functions embody the functional programming paradigm: instead of telling the computer how to loop through data (imperative), you describe what transformation you want (declarative).

**Map** applies a function to every element in a sequence:

```python
In [61]: # MAP: Transform each element
In [62]: temperatures_c = [0, 10, 20, 30, 40]
In [63]: temperatures_f = list(map(lambda c: c * 9/5 + 32, temperatures_c))
In [64]: print(temperatures_f)
[32.0, 50.0, 68.0, 86.0, 104.0]
```

**Filter** selects elements that satisfy a condition:

```python
In [65]: # FILTER: Select elements
In [66]: numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
In [67]: evens = list(filter(lambda x: x % 2 == 0, numbers))
In [68]: print(evens)
[2, 4, 6, 8, 10]
```

**Reduce** aggregates a sequence to a single value by repeatedly applying a binary function:

```python
In [69]: # REDUCE: Aggregate to single value
In [70]: from functools import reduce
In [71]: product = reduce(lambda x, y: x * y, [1, 2, 3, 4, 5])
In [72]: print(product)  # 1*2*3*4*5
120
```

While these functional approaches are powerful, Python programmers often prefer list comprehensions for readability:

```python
# Three equivalent approaches
numbers = [1, 2, 3, 4, 5]

# Functional with map
squares_map = list(map(lambda n: n ** 2, numbers))

# List comprehension (more Pythonic)
squares_comp = [n ** 2 for n in numbers]

# Traditional loop (most explicit)
squares_loop = []
for n in numbers:
    squares_loop.append(n ** 2)
```

Each approach has its place. Use functional style when you already have the function defined, list comprehensions for simple transformations, and loops when the logic is complex.

### Functions as First-Class Objects

In Python, functions are objects that can be passed around:

```python
In [73]: def apply_operation(data, operation):
   ...:     """Apply an operation function to data."""
   ...:     return [operation(x) for x in data]

In [74]: def double(x):
   ...:     return x * 2

In [75]: def square(x):
   ...:     return x ** 2

In [76]: numbers = [1, 2, 3, 4, 5]
In [77]: apply_operation(numbers, double)
Out[77]: [2, 4, 6, 8, 10]

In [78]: apply_operation(numbers, square)
Out[78]: [1, 4, 9, 16, 25]

In [79]: # Can even pass built-in functions
In [80]: apply_operation(numbers, abs)
Out[80]: [1, 2, 3, 4, 5]
```

### Higher-Order Functions

Functions that operate on other functions:

```python
In [81]: def make_validator(min_val, max_val):
   ...:     """Create a validation function for a range."""
   ...:     def validator(x):
   ...:         return min_val <= x <= max_val
   ...:     return validator

In [82]: # Create specific validators
In [83]: valid_percentage = make_validator(0, 100)
In [84]: valid_temperature = make_validator(-273.15, float('inf'))

In [85]: valid_percentage(50)
Out[85]: True

In [86]: valid_percentage(150)
Out[86]: False

In [87]: valid_temperature(-300)
Out[87]: False
```

### Decorators in Scientific Computing

Decorators modify function behavior without changing the function's code. You'll encounter decorators throughout scientific Python libraries. For example, NumPy uses decorators to mark deprecated functions, and Numba uses them to compile Python to machine code:

```python
# Example: How scientific libraries use decorators

# 1. Numba JIT compilation (you'll see this in performance-critical code)
from numba import jit

@jit  # Decorator compiles function to machine code!
def monte_carlo_pi(n):
    """Estimate pi using Monte Carlo - runs 100x faster with @jit."""
    count = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            count += 1
    return 4.0 * count / n

# 2. Simple deprecation warning (how libraries manage API changes)
import warnings
import functools

def deprecated(replacement=None):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated"
            if replacement:
                msg += f", use {replacement} instead"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated(replacement="new_function")
def old_function(x):
    """This function is being phased out."""
    return x * 2

# When called, prints deprecation warning but still works
result = old_function(5)  # DeprecationWarning: old_function is deprecated
```

Understanding decorators helps you read scientific library documentation and use advanced features like JIT compilation, memoization, and parallel processing that are common in high-performance scientific computing.

### 🔍 **Check Your Understanding**

Rewrite this loop using `map` and a lambda:

```python
celsius = [0, 10, 20, 30]
fahrenheit = []
for c in celsius:
    fahrenheit.append(c * 9/5 + 32)
```

<details>
<summary>Answer</summary>

```python
celsius = [0, 10, 20, 30]
fahrenheit = list(map(lambda c: c * 9/5 + 32, celsius))
```

Note: The list comprehension version is often more readable:
```python
fahrenheit = [c * 9/5 + 32 for c in celsius]
```

</details>

## 5.5 Modules and Packages

As your code grows from scripts to projects, organization becomes critical. Modules and packages are Python's way of organizing code into reusable, maintainable units. A module is simply a Python file containing functions, classes, and variables. A package is a directory containing multiple modules. This organization isn't just about tidiness — it's about creating code that can be shared, tested, and maintained by teams.

### Creating Your First Module

Let's create a module for common scientific conversions. Save this code as `conversions.py`:

```python
"""
conversions.py
A module for unit conversions.
"""

# Module-level constant
ABSOLUTE_ZERO_C = -273.15

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9/5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    if celsius < ABSOLUTE_ZERO_C:
        raise ValueError(f"Temperature below absolute zero: {celsius}°C")
    return celsius + 273.15

def meters_to_feet(meters):
    """Convert meters to feet."""
    return meters * 3.28084

# Code that runs when module is imported
print(f"Loaded conversions module")
```

This module groups related functions together. Anyone who needs temperature or distance conversions can import this module rather than rewriting these functions.

### Using Your Module

Once you've created a module, you can import and use it in several ways:

```python
In [88]: # Method 1: Import the entire module
In [89]: import conversions
In [90]: temp_f = conversions.celsius_to_fahrenheit(25)
In [91]: print(f"25°C = {temp_f}°F")
25°C = 77.0°F

In [92]: # Method 2: Import specific functions
In [93]: from conversions import celsius_to_kelvin
In [94]: temp_k = celsius_to_kelvin(25)
In [95]: print(f"25°C = {temp_k}K")
25°C = 298.15K

In [96]: # Method 3: Import with an alias (nickname)
In [97]: import conversions as conv
In [98]: distance = conv.meters_to_feet(10)
In [99]: print(f"10 meters = {distance:.1f} feet")
10 meters = 32.8 feet
```

Each import method has its use case. Import the entire module when you'll use many functions from it. Import specific functions when you only need one or two. Use aliases to shorten long module names or avoid naming conflicts.

### The `if __name__ == "__main__"` Pattern

This pattern is one of Python's most important idioms. It makes modules both importable and executable. When Python runs a file, it sets a special variable `__name__`. If the file is being run directly, `__name__` is set to `"__main__"`. If the file is being imported, `__name__` is set to the module's name.

```python
# calculations.py

def calculate_statistics(data):
    """Calculate mean and standard deviation."""
    n = len(data)
    if n == 0:
        return None, None
    
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    return mean, std_dev

# Test code that only runs when script is executed directly
if __name__ == "__main__":
    # This code runs when: python calculations.py
    # But NOT when: import calculations
    
    test_data = [1, 2, 3, 4, 5]
    mean, std = calculate_statistics(test_data)
    print(f"Test data: {test_data}")
    print(f"Mean: {mean:.2f}, Std Dev: {std:.2f}")
```

This pattern allows you to include test code, examples, or a command-line interface in your modules without that code running when someone imports your module.

### Understanding Module Search Path

```python
In [99]: import sys
In [100]: # Where Python looks for modules
In [101]: for path in sys.path[:5]:  # Show first 5
   ...:      print(path)

# Typical output:
# '' (current directory)
# /path/to/python/lib/python3.x
# /path/to/python/lib/python3.x/lib-dynload
# /path/to/site-packages
```

### Creating a Package

A package is a directory containing modules:

```
my_science_tools/
    __init__.py          # Makes it a package
    conversions.py       # Temperature, distance conversions
    constants.py         # Physical constants
    statistics.py        # Statistical functions
```

`my_science_tools/__init__.py`:
```python
"""
My Science Tools Package
A collection of useful scientific functions.
"""

# Import commonly used functions for convenience
from .conversions import celsius_to_fahrenheit, meters_to_feet
from .constants import SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

# Package metadata
__version__ = '0.1.0'
__author__ = 'Your Name'

print(f"Loading my_science_tools v{__version__}")
```

`my_science_tools/constants.py`:
```python
"""Physical constants in SI units."""

SPEED_OF_LIGHT = 299792458  # m/s
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³ kg⁻¹ s⁻²
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
```

Using the package:
```python
# Import entire package
import my_science_tools

# Use through package
c = my_science_tools.SPEED_OF_LIGHT

# Import specific module
from my_science_tools import conversions
temp = conversions.celsius_to_fahrenheit(100)

# Import specific function
from my_science_tools.statistics import calculate_mean
```

### 📦 **Computational Thinking Box: Module Design Principles**

```
PATTERN: Cohesive Module Organization

Group related functionality together:

Good Module Structure:
- conversions.py: All unit conversions
- validation.py: All data validation functions
- io_tools.py: All file reading/writing

Bad Module Structure:
- utils.py: Random mix of everything
- helpers.py: Unclear purpose
- misc.py: Dumping ground

Benefits of good organization:
1. Easy to find functions
2. Clear dependencies
3. Simpler testing
4. Better documentation
5. Easier maintenance
```

### Import Best Practices

```python
# GOOD: Clear, explicit imports
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# BAD: Wildcard imports pollute namespace
from math import *  # Now you have 70+ names!
# What if multiple modules have 'sqrt'?

# GOOD: Conditional imports for optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas not available, some features disabled")

def process_dataframe(data):
    if not HAS_PANDAS:
        raise RuntimeError("This function requires pandas")
    # Process with pandas...
```

### The Danger of Namespace Pollution

Namespace pollution occurs when you import too many names into your current namespace, making it unclear where functions come from and risking name collisions. This is particularly dangerous in scientific computing where common function names appear in multiple libraries:

```python
# DANGEROUS: Multiple libraries with same function names
from numpy import *      # Has sum, mean, std, max, min, etc.
from statistics import * # Also has mean, median, mode, stdev
from math import *       # Has sqrt, log, sin, cos, etc.
from scipy.special import * # Has gamma, beta, etc.

# Which mean() function gets called?
result = mean([1, 2, 3])  # numpy's? statistics'? Who knows!

# This caused a real bug in a published paper where scipy's gamma
# function (the mathematical function) was confused with numpy.random's
# gamma (the distribution), leading to completely wrong results.

# SAFE: Explicit namespaces prevent confusion
import numpy as np
import statistics as stats
import math
import scipy.special as special

# Now it's clear which function is being used
np_mean = np.mean([1, 2, 3])      # NumPy's version
stat_mean = stats.mean([1, 2, 3])  # Statistics module's version
gamma_fn = special.gamma(5)        # Gamma function: Γ(5) = 24
gamma_dist = np.random.gamma(2, 2) # Random sample from gamma distribution
```

The rule is simple: never use `from module import *` except in interactive sessions where you're exploring. In production code, namespace clarity prevents bugs that can corrupt entire analyses. Five extra keystrokes for `np.` can save five months of debugging when you realize your Monte Carlo used the wrong random distribution.

## 5.6 Documentation and Testing

Good documentation and basic testing make your functions trustworthy and reusable.

### Writing Good Docstrings

```python
def calculate_rms(values, ignore_negative=False):
    """
    Calculate root mean square of values.
    
    Parameters
    ----------
    values : list or array-like
        Numeric values to process
    ignore_negative : bool, optional
        If True, ignore negative values (default: False)
    
    Returns
    -------
    float
        Root mean square of the values
        
    Raises
    ------
    ValueError
        If no valid values remain after filtering
    
    Examples
    --------
    >>> calculate_rms([3, 4])
    3.5355...
    
    >>> calculate_rms([3, -4], ignore_negative=True)
    3.0
    
    Notes
    -----
    RMS = sqrt(mean(x²)) for all valid x
    """
    if ignore_negative:
        values = [v for v in values if v >= 0]
    
    if not values:
        raise ValueError("No valid values to process")
    
    sum_squares = sum(v ** 2 for v in values)
    mean_square = sum_squares / len(values)
    return mean_square ** 0.5
```

### Simple Testing with Assertions

```python
def test_calculate_rms():
    """Test the calculate_rms function."""
    
    # Test basic functionality
    result = calculate_rms([3, 4])
    expected = 3.5355339059327378
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    # Test with negative values
    result = calculate_rms([3, -4], ignore_negative=True)
    assert result == 3.0, f"Expected 3.0, got {result}"
    
    # Test error handling
    try:
        calculate_rms([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("All tests passed!")

# Run the test
test_calculate_rms()
```

### Using Assertions for Defensive Programming

Assertions help catch bugs early during development:

```python
def process_data(measurements, calibration_factor):
    """
    Process measurement data with calibration.
    
    Uses assertions to validate assumptions.
    """
    # Validate inputs with assertions
    assert len(measurements) > 0, "Need at least one measurement"
    assert calibration_factor > 0, "Calibration factor must be positive"
    assert all(isinstance(m, (int, float)) for m in measurements), \
           "All measurements must be numeric"
    
    # Process the data
    calibrated = [m * calibration_factor for m in measurements]
    
    # Validate output
    assert len(calibrated) == len(measurements), "Output length mismatch"
    
    return calibrated

# Note: Assertions can be disabled with python -O
# Use explicit checks for production code validation
```

### Why Scientists Often Skip Testing (And Why That's Dangerous)

The scientific computing community has a testing problem. Many researchers view their code as "one-off" analysis scripts that don't need formal testing. This assumption has led to serious consequences. The infamous Reinhart-Rogoff economics paper that influenced global austerity policies contained an Excel error that proper testing would have caught—a missing row in a calculation that changed their conclusions about debt and economic growth. In bioinformatics, a script error in the conversion between gene identifiers led to corrupted data in thousands of published papers, with gene names like SEPT2 (Septin 2) being auto-converted to dates by Excel.

Testing isn't about perfection; it's about catching the obvious errors that exhausted graduate students make at 2 AM. A simple test that verifies your function produces known results for known inputs can save months of debugging contaminated results. The five minutes you spend writing a test today saves five weeks of re-running analyses when you discover a sign error three papers later.

## 5.7 Performance Considerations

Understanding function performance helps you write efficient code that scales to large datasets.

### 📊 **Performance Profile: Function Call Overhead**

Let's measure the cost of function calls:

```python
In [102]: import time

In [103]: def empty_function():
   ....:     pass

In [104]: def inline_calculation():
   ....:     """Everything in one function."""
   ....:     total = 0
   ....:     for i in range(1000):
   ....:         total += i * 2
   ....:     return total

In [105]: def with_helper(x):
   ....:     """Helper function for calculation."""
   ....:     return x * 2

In [106]: def using_helper():
   ....:     """Uses helper function - more overhead."""
   ....:     total = 0
   ....:     for i in range(1000):
   ....:         total += with_helper(i)
   ....:     return total

In [107]: # Time the difference
In [108]: %timeit inline_calculation()
45.2 µs ± 312 ns per loop

In [109]: %timeit using_helper()
112.3 µs ± 1.02 µs per loop

# Function calls add ~2.5x overhead for this simple case!
```

### When Function Overhead Matters

```python
# CASE 1: Overhead negligible - complex function
def complex_calculation(data):
    """When function does substantial work, call overhead is negligible."""
    # Lots of computation here
    result = sum(x**2 for x in data)
    result = (result / len(data)) ** 0.5
    # ... more work ...
    return result

# CASE 2: Overhead significant - trivial function in tight loop
def add_one(x):
    return x + 1

# Bad: Calling trivial function millions of times
data = range(1_000_000)
result = [add_one(x) for x in data]  # Slow!

# Better: Inline the operation
result = [x + 1 for x in data]  # Much faster!
```

### Memoization for Expensive Functions

Cache results of expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(n):
    """
    Simulate expensive calculation.
    Results are cached automatically.
    """
    # Simulate expensive work
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

# First call: slow
result1 = expensive_calculation(100)  # Takes time

# Second call with same input: instant!
result2 = expensive_calculation(100)  # From cache

# Check cache statistics
print(expensive_calculation.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)
```

### 🛠️ **Debug This!**

This function has a performance bug. Can you find it?

```python
def process_large_dataset(data):
    """Process large dataset - has performance bug."""
    results = []
    
    for item in data:
        # Process item
        processed = item * 2
        
        # Check if already processed (BUG HERE!)
        if processed not in results:
            results.append(processed)
    
    return results

# Why is this slow for large datasets?
```

<details>
<summary>Answer and Fix</summary>

**Bug**: `if processed not in results` is O(n) for lists! For 10,000 items, this becomes O(n²) total.

**Fix**: Use a set for O(1) membership testing:

```python
def process_large_dataset_fixed(data):
    """Process large dataset - fixed version."""
    results = []
    seen = set()  # O(1) membership testing
    
    for item in data:
        processed = item * 2
        
        if processed not in seen:
            results.append(processed)
            seen.add(processed)
    
    return results
```

For 10,000 items:
- Original: ~1 second
- Fixed: ~0.001 seconds
- 1000x speedup!

</details>

## Practice Exercises

### Exercise 5.1: Temperature Converter Module

Create a module called `temp_convert.py` that provides comprehensive temperature conversion:

```python
"""
Create a temperature conversion module with these requirements:

1. Functions to convert between Celsius, Fahrenheit, and Kelvin
2. Each function should validate that temperature is above absolute zero
3. Include a function that converts from any unit to any other unit
4. Add helpful constants (absolute zero, water freezing/boiling points)
5. Include proper docstrings and error handling

Pseudocode first:
FUNCTION convert_temperature(value, from_unit, to_unit):
    VALIDATE temperature is physically possible
    IF from_unit == to_unit:
        RETURN value
    CONVERT to Celsius first (common base)
    CONVERT from Celsius to target unit
    RETURN converted value
"""

# Your implementation here
```

### Exercise 5.2: Function Performance Analysis

Write a program that compares three different ways to calculate factorials:

```python
"""
Compare factorial implementations:

1. Recursive approach
2. Iterative approach  
3. Memoized recursive approach

Requirements:
- Implement all three methods
- Time each method for n = 10, 20, 30, ..., 100
- Plot the results (optional)
- Explain why the performance differs

Hint: Be careful with recursion depth!
"""

def factorial_recursive(n):
    # Your implementation
    pass

def factorial_iterative(n):
    # Your implementation
    pass

# Create memoized version
# Time and compare all three
```

### Exercise 5.3: Scope Detective

Debug and fix this code that has scope-related bugs:

```python
total = 0
count = 0

def add_to_average(value):
    """Add value and update running average - BUGGY!"""
    total += value  # Bug 1
    count += 1      # Bug 2
    return total / count

def reset_statistics():
    """Reset the statistics - BUGGY!"""
    total = 0  # Bug 3
    count = 0  # Bug 4

# Fix the bugs and explain:
# 1. What's wrong with each function?
# 2. What error messages would you get?
# 3. How would you fix it properly?
# 4. Is using global state a good idea here?
```

### Exercise 5.4: Module Organization

Design a module structure for a scientific calculator package:

```python
"""
Design a package called 'sci_calc' with the following capabilities:

Modules to create:
- basic.py: add, subtract, multiply, divide with error checking
- scientific.py: power, sqrt, log, exp, trig functions
- statistics.py: mean, median, mode, std_dev
- constants.py: pi, e, golden_ratio, etc.

Requirements:
1. Create the package structure
2. Write __init__.py to expose common functions
3. Handle errors appropriately (divide by zero, domain errors)
4. Include at least 3 functions per module
5. Write one test function per module

Show the directory structure and key parts of each file.
"""
```

## Key Takeaways

**Functions are contracts** between different parts of your code. A well-designed function has a clear purpose, predictable behavior, and handles edge cases gracefully. The function's interface (parameters and return values) should make its purpose obvious.

**Scope rules (LEGB) determine variable visibility**. Understanding scope prevents bugs and helps you reason about code behavior. Avoid global variables when possible—they make code harder to test, debug, and parallelize.

**The mutable default argument trap** is a common source of bugs. Default arguments are evaluated once when the function is defined, not each time it's called. Always use `None` as a sentinel for mutable defaults.

**Functional programming concepts** like map, filter, and lambda functions can make code more concise and expressive. However, list comprehensions are often more Pythonic and readable than functional approaches.

**Modules organize related code** into reusable units. The `if __name__ == "__main__"` pattern makes modules both importable and executable. Packages group related modules together with a clear structure.

**Documentation and testing** aren't optional—they're essential for code that others (including future you) can trust and use. Good docstrings explain not just what a function does, but why, when, and how to use it.

**Performance matters** in scientific computing. Function call overhead is usually negligible, but can matter in tight loops with trivial functions. Memoization can dramatically speed up recursive or expensive functions.

## Quick Reference: Functions and Modules

| Concept | Syntax | Example |
|---------|--------|---------|
| Define function | `def name(params):` | `def add(x, y): return x + y` |
| Return value | `return expression` | `return x * 2` |
| Return multiple | `return a, b` | `return min_val, max_val` |
| Default argument | `param=default` | `def f(x, n=10):` |
| Variable args | `*args` | `def sum_all(*values):` |
| Keyword args | `**kwargs` | `def plot(**options):` |
| Lambda | `lambda params: expression` | `lambda x: x**2` |
| Map | `map(function, iterable)` | `map(abs, numbers)` |
| Filter | `filter(function, iterable)` | `filter(lambda x: x > 0, data)` |
| Import module | `import module` | `import math` |
| Import specific | `from module import name` | `from math import pi` |
| Import as alias | `import module as alias` | `import numpy as np` |
| Module check | `if __name__ == "__main__":` | Used for test code |

### Common Built-in Functions for Functional Programming

| Function | Purpose | Example |
|----------|---------|---------|
| `map()` | Apply function to all items | `list(map(str, [1,2,3]))` → `['1','2','3']` |
| `filter()` | Keep items where function is True | `list(filter(lambda x: x>0, [-1,1,2]))` → `[1,2]` |
| `reduce()` | Aggregate to single value | `reduce(operator.add, [1,2,3])` → `6` |
| `zip()` | Combine iterables | `list(zip([1,2], ['a','b']))` → `[(1,'a'), (2,'b')]` |
| `enumerate()` | Add indices | `list(enumerate(['a','b']))` → `[(0,'a'), (1,'b')]` |
| `sorted()` | Sort with key function | `sorted(data, key=lambda x: x[1])` |
| `any()` | True if any element is True | `any([False, True, False])` → `True` |
| `all()` | True if all elements are True | `all([True, True, False])` → `False` |

## Next Chapter Preview

With functions and modules mastered, Chapter 6 will introduce NumPy—the foundation of scientific computing in Python. You'll discover why NumPy arrays are 10-100x more efficient than Python lists for numerical work, learn about vectorization (computing on entire arrays without loops), and understand broadcasting (NumPy's powerful pattern for combining arrays of different shapes).

The functional programming concepts from this chapter directly prepare you for NumPy's vectorized operations, where you'll apply functions to entire arrays at once. The module organization skills will help you structure larger scientific projects. Most importantly, the performance awareness you've developed will help you understand when to transition from pure Python to NumPy for serious numerical work.