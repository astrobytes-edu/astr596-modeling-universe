# Chapter 6: Object-Oriented Programming - Organizing Scientific Code

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement classes that model scientific concepts and data
- Understand the relationship between classes and objects, and when OOP is appropriate
- Create methods that operate on object data and properties that compute derived values
- Apply inheritance and composition to build hierarchies of related scientific objects
- Implement special methods to make your objects behave like built-in Python types
- Debug common OOP-related errors using introspection tools
- Write effective tests for your classes
- Recognize OOP patterns in scientific libraries like NumPy and Astropy
- Choose between OOP, functional, and procedural approaches based on problem requirements

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Define and use functions with various parameter types (Chapter 5)
- ✓ Understand scope and namespaces (Chapter 5)
- ✓ Work with dictionaries and their methods (Chapter 4)
- ✓ Create and import modules (Chapter 5)
- ✓ Handle mutable vs immutable objects (Chapter 4)

## Chapter Overview

So far, we've organized code using functions and modules. But what happens when you need to model complex scientific systems where data and the operations on that data are intimately connected? This is where Object-Oriented Programming (OOP) shines. OOP lets you bundle data and functionality together into objects that model real-world (or abstract) concepts.

Consider tracking stars in a catalog. Each star has properties (position, magnitude, spectral type) and behaviors (calculate distance, determine visibility, evolve over time). With functions alone, you'd pass star data between dozens of functions, hoping you don't mix up which data belongs to which star. With OOP, each star is an object that knows its own data and what it can do. This organizational principle scales from simple data containers to complex simulations with thousands of interacting components.

This chapter teaches you to think in objects — not as a dogmatic paradigm, but as a powerful tool for organizing scientific code. You'll learn when OOP makes code clearer (modeling physical objects, managing complex state) and when it adds unnecessary complexity (simple calculations, functional transformations). By the end, you'll understand why NumPy arrays are objects with methods, how Matplotlib figures manage their state, and when to create your own classes versus using simpler approaches.

## 6.1 Classes and Objects: The Fundamentals

A class is a blueprint for creating objects. Think of it like the architectural plans for a house — the class defines what properties the house will have (rooms, doors, windows) and what it can do (open doors, turn on lights). An object is a specific instance created from that blueprint — an actual house built from those plans.

Before we dive into creating classes, let's understand that classes aren't entirely new — they're a way to organize concepts you already know. A class bundles together variables (which we call attributes) and functions (which we call methods) into a single coherent unit.

```{mermaid}
flowchart TD
    A[Traditional Programming] --> B[Variables]
    A --> C[Functions]
    
    D[Object-Oriented Programming] --> E[Class]
    E --> F[Attributes<br/>Variables attached to objects]
    E --> G[Methods<br/>Functions attached to objects]
    
    B -.->|becomes| F
    C -.->|becomes| G
    
    H[Object/Instance] --> I[Has its own attribute values]
    H --> J[Can call methods]
    E -->|creates| H
    
    style E fill:#f9f,stroke:#333,stroke-width:4px
    style H fill:#bbf,stroke:#333,stroke-width:2px
```

### Understanding Object Anatomy: What's Inside an Object?

An object contains members, which are simply the things that belong to that object. There are two types of members, and you already know both concepts by different names:

**Attributes** are variables that belong to an object. Just like variables store data, attributes store an object's data. The only difference is that attributes are attached to a specific object.

**Methods** are functions that belong to an object. Just like functions perform operations, methods perform operations — but they have access to the object's attributes and can operate on the object's data.

Let's see this connection explicitly:

```python
# You already know variables and functions:
temperature = 25.0  # Variable
def convert_to_fahrenheit(celsius):  # Function
    return celsius * 9/5 + 32

# In OOP, these become attributes and methods:
class Thermometer:
    def __init__(self):
        self.temperature = 25.0  # Attribute (like a variable, but belongs to object)
    
    def convert_to_fahrenheit(self):  # Method (like a function, but belongs to object)
        return self.temperature * 9/5 + 32

# The key difference: attributes and methods are organized together
therm = Thermometer()
print(therm.temperature)  # Access attribute through object
print(therm.convert_to_fahrenheit())  # Call method through object
```

This organization is powerful because related data and operations stay together. The thermometer object knows its own temperature and how to convert it — everything about temperature management is in one place.

### Types of Members: Public, Protected, and Private

Python has a unique approach to member visibility that's more about convention than enforcement. Unlike languages like Java or C++, Python trusts programmers to respect conventions rather than enforcing strict access controls. Here's how Python handles member visibility:

```{mermaid}
flowchart LR
    A[Class Members] --> B[Public<br/>No prefix<br/>External use]
    A --> C[Protected<br/>Single underscore _<br/>Internal + subclasses]
    A --> D[Private<br/>Double underscore __<br/>Name mangled]
    
    B --> E[sensor.reading]
    C --> F[sensor._calibration]
    D --> G[sensor.__secret]
    
    style B fill:#9f9,stroke:#333
    style C fill:#ff9,stroke:#333
    style D fill:#f99,stroke:#333
```

**Public members** (the default) can be accessed from anywhere. These have no special prefix and are meant for external use:

```python
class Sensor:
    def __init__(self):
        self.reading = 42.0  # Public attribute
    
    def get_reading(self):  # Public method
        return self.reading

sensor = Sensor()
print(sensor.reading)  # Fine to access directly
```

**Protected members** (single underscore prefix) are meant for internal use within the class and its subclasses. This is purely convention — Python doesn't prevent access, but the underscore signals "please don't use this from outside":

```python
class Instrument:
    def __init__(self):
        self._calibration_factor = 1.05  # Protected attribute
    
    def _apply_calibration(self, raw_value):  # Protected method
        """Internal method for calibration."""
        return raw_value * self._calibration_factor
    
    def measure(self):
        """Public method that uses protected members internally."""
        raw = self._get_raw_reading()
        return self._apply_calibration(raw)
    
    def _get_raw_reading(self):
        """Protected method to get uncalibrated reading."""
        return 100.0

# You CAN access protected members, but shouldn't
inst = Instrument()
print(inst._calibration_factor)  # Works but violates convention
```

**Private members** (double underscore prefix) trigger name mangling to make them harder to access accidentally. Python changes the name internally to include the class name:

```python
class SecureDevice:
    def __init__(self):
        self.__secret_key = "hidden"  # Private attribute
    
    def __internal_process(self):  # Private method
        """This method is truly internal."""
        return self.__secret_key

device = SecureDevice()
# print(device.__secret_key)  # AttributeError!
# Python mangles the name to _SecureDevice__secret_key
# You CAN still access it if determined:
print(device._SecureDevice__secret_key)  # Works but defeats the purpose
```

The Python philosophy is "we're all consenting adults" — these conventions communicate intent rather than enforce restrictions. Use public for your API, protected for internal implementation that subclasses might need, and private only when you really need to avoid name conflicts in inheritance.

### Naming Conventions: Python Style for Classes and Members

Python has strong naming conventions that make code readable and intentions clear. Following these conventions makes your code immediately understandable to other Python programmers:

```{mermaid}
flowchart TD
    A[Python Naming Conventions] --> B[Classes<br/>CamelCase/PascalCase]
    A --> C[Methods & Attributes<br/>snake_case]
    A --> D[Constants<br/>UPPER_SNAKE_CASE]
    A --> E[Special Methods<br/>__dunder__]
    
    B --> B1[StarCatalog<br/>DataProcessor<br/>OpticalTelescope]
    C --> C1[calculate_flux<br/>max_intensity<br/>wave_length]
    D --> D1[SPEED_OF_LIGHT<br/>MAX_ITERATIONS<br/>DEFAULT_TIMEOUT]
    E --> E1[__init__<br/>__str__<br/>__add__]
    
    style B fill:#f9f
    style C fill:#9ff
    style D fill:#ff9
    style E fill:#f99
```

**Classes** use CamelCase (also called PascalCase):
```python
class StarCatalog:  # Good
class DataProcessor:  # Good
class star_catalog:  # Bad - use CamelCase for classes
class dataprocessor:  # Bad - use CamelCase with word separation
```

**Methods and attributes** use snake_case:
```python
class SpectralAnalyzer:
    def __init__(self):
        self.wave_length = 500.0  # Bad - should be wavelength or wave_length
        self.wavelength = 500.0  # Good
        self.maxIntensity = 100  # Bad - use snake_case not camelCase
        self.max_intensity = 100  # Good
    
    def CalculateFlux(self):  # Bad - use snake_case
        pass
    
    def calculate_flux(self):  # Good
        pass
```

**Constants** (class-level attributes that shouldn't change) use UPPER_SNAKE_CASE:
```python
class PhysicalConstants:
    SPEED_OF_LIGHT = 2.998e10  # cm/s
    PLANCK_CONSTANT = 6.626e-27  # erg·s
    GRAVITATIONAL_CONSTANT = 6.674e-8  # cm³/g·s²
```

**Special methods** always use double underscores (dunders):
```python
class Vector:
    def __init__(self):  # Constructor
        pass
    
    def __str__(self):  # String representation
        pass
    
    def __add__(self, other):  # Addition operator
        pass
```

### Your First Class

Let's start with the simplest possible class and understand every component:

```python
In [1]: class Star:
   ...:     """A simple star class."""
   ...:     
   ...:     def __init__(self, name, magnitude):
   ...:         """Initialize a new Star object."""
   ...:         self.name = name
   ...:         self.magnitude = magnitude

In [2]: # Create an instance (object) of the Star class
In [3]: sirius = Star("Sirius", -1.46)

In [4]: # Access object attributes
In [5]: print(f"{sirius.name} has magnitude {sirius.magnitude}")
Sirius has magnitude -1.46
```

Let's dissect this class definition and see what happens when we create an object:

```{mermaid}
sequenceDiagram
    participant Code as Your Code
    participant Python
    participant Memory
    participant Object as Star Object
    
    Code->>Python: sirius = Star("Sirius", -1.46)
    Python->>Memory: Allocate space for new object
    Memory-->>Python: Memory allocated
    Python->>Object: Create empty Star instance
    Python->>Object: Call __init__(self, "Sirius", -1.46)
    Object->>Object: self.name = "Sirius"
    Object->>Object: self.magnitude = -1.46
    Object-->>Python: Initialization complete
    Python-->>Code: Return reference to object
    Note over Code: sirius now refers to the Star object
```

Breaking down the components:

1. **`class Star:`** - Defines a new class named Star (use CamelCase for class names)
2. **`__init__` method** - Special method called when creating new objects (the constructor)
3. **`self` parameter** - References the specific instance being created or operated on
4. **`self.name = name`** - Creates an instance attribute storing this object's data
5. **Instance creation** - `Star("Sirius", -1.46)` calls `__init__` to create a new object

The `self` parameter is crucial but often confusing. When you call `sirius = Star("Sirius", -1.46)`, Python essentially does:
1. Creates a new empty object
2. Calls `Star.__init__(new_object, "Sirius", -1.46)`
3. Returns the initialized object

The `self` parameter is how each object keeps track of its own data.

### ⚠️ **Common Bug Alert: Missing self Parameter**

```python
# WRONG - Forgetting self in method definition
class BadClass:
    def method():  # Missing self!
        return "something"

obj = BadClass()
obj.method()  # TypeError: method() takes 0 positional arguments but 1 was given

# CORRECT - Always include self as first parameter
class GoodClass:
    def method(self):  # self is required
        return "something"

# Why this error happens:
# Python automatically passes the object as the first argument
# obj.method() is actually like: GoodClass.method(obj)
# If you forget self, Python tries to pass obj but there's no parameter for it!
```

This is probably the most common OOP error for beginners. Remember: instance methods ALWAYS need `self` as their first parameter, even if they don't use it.

### Adding Methods: Functions That Operate on Objects

Methods are functions defined inside a class. They automatically receive the object they're called on as their first parameter (`self`):

```python
In [6]: class Star:
   ...:     """A star with observable properties."""
   ...:     
   ...:     def __init__(self, name, magnitude, distance_pc):
   ...:         """
   ...:         Initialize a star.
   ...:         
   ...:         Parameters
   ...:         ----------
   ...:         name : str
   ...:             Star designation
   ...:         magnitude : float
   ...:             Apparent magnitude
   ...:         distance_pc : float
   ...:             Distance in parsecs
   ...:         """
   ...:         self.name = name
   ...:         self.magnitude = magnitude
   ...:         self.distance_pc = distance_pc
   ...:     
   ...:     def absolute_magnitude(self):
   ...:         """Calculate absolute magnitude from apparent magnitude and distance."""
   ...:         import math
   ...:         return self.magnitude - 5 * math.log10(self.distance_pc) + 5
   ...:     
   ...:     def luminosity_solar(self):
   ...:         """Calculate luminosity relative to the Sun."""
   ...:         # Sun's absolute magnitude is 4.83
   ...:         abs_mag = self.absolute_magnitude()
   ...:         return 10**((4.83 - abs_mag) / 2.5)

In [7]: # Create a star object
In [8]: proxima = Star("Proxima Centauri", 11.13, 1.301)

In [9]: # Call methods on the object
In [10]: abs_mag = proxima.absolute_magnitude()
In [11]: print(f"Absolute magnitude: {abs_mag:.2f}")
Absolute magnitude: 15.56

In [12]: luminosity = proxima.luminosity_solar()
In [13]: print(f"Luminosity: {luminosity:.4f} solar luminosities")
Luminosity: 0.0017 solar luminosities
```

Methods provide behavior — they're actions the object can perform using its own data. Notice how methods access the object's attributes through `self`. This encapsulation means the object carries both its data and the operations on that data together.

### 🔍 **Check Your Understanding**

What's the difference between these two approaches?

```python
# Approach 1: Functions and variables
temperature = 25.0
humidity = 60.0

def calculate_heat_index(temp, hum):
    # Heat index calculation
    return temp + 0.5 * hum

# Approach 2: Class with attributes and methods
class WeatherStation:
    def __init__(self):
        self.temperature = 25.0
        self.humidity = 60.0
    
    def calculate_heat_index(self):
        return self.temperature + 0.5 * self.humidity
```

<details>
<summary>Answer</summary>

Both achieve the same calculation, but they organize code differently. In Approach 1, the data (temperature, humidity) and the function are separate — you must remember to pass the right variables to the function. In Approach 2, the data and method are bundled together in an object. The object "knows" its own temperature and humidity, so the method can access them directly through `self`.

The OOP approach becomes more valuable as complexity grows. Imagine tracking 50 weather measurements and 20 calculations — keeping track of which data goes with which function becomes error-prone. With objects, each weather station manages its own data and knows what operations it can perform.

</details>

### Instance vs Class Members: Understanding the Difference

One of the most important distinctions in OOP is between instance members (belonging to specific objects) and class members (shared by all objects of that class). This distinction affects both attributes and methods.

```{mermaid}
classDiagram
    class Satellite {
        <<class attributes>>
        +int total_satellites$
        +float EARTH_RADIUS_KM$
        
        <<instance attributes>>
        +str name
        +float altitude
        +bool operational
        
        <<instance methods>>
        +calculate_period()
        
        <<class methods>>
        +get_satellite_count()$
        
        <<static methods>>
        +km_to_miles()$
    }
    
    class ISS {
        name = "ISS"
        altitude = 408
        operational = true
    }
    
    class Hubble {
        name = "Hubble"
        altitude = 547
        operational = true
    }
    
    Satellite <|-- ISS : instance of
    Satellite <|-- Hubble : instance of
    
    note for Satellite "$ indicates class/static members\nshared by all instances"
```

**Instance Members** belong to individual objects:
```python
class Satellite:
    def __init__(self, name, altitude_km):
        # Instance attributes - each satellite has its own
        self.name = name
        self.altitude = altitude_km
        self.operational = True
    
    def calculate_period(self):
        # Instance method - uses this satellite's altitude
        import math
        earth_radius_km = 6371
        total_radius = earth_radius_km + self.altitude
        # Simplified calculation
        return 2 * math.pi * math.sqrt(total_radius**3 / 398600)

# Each satellite object has independent instance attributes
sat1 = Satellite("ISS", 408)
sat2 = Satellite("Hubble", 547)

print(f"{sat1.name} altitude: {sat1.altitude} km")  # ISS altitude: 408 km
print(f"{sat2.name} altitude: {sat2.altitude} km")  # Hubble altitude: 547 km

# Changing one doesn't affect the other
sat1.altitude = 410
print(f"{sat1.altitude=}, {sat2.altitude=}")  # sat1.altitude=410, sat2.altitude=547
```

**Class Members** are shared by all instances:
```python
class Satellite:
    # Class attributes - shared by all satellites
    total_satellites = 0
    EARTH_RADIUS_KM = 6371
    
    def __init__(self, name, altitude_km):
        self.name = name
        self.altitude = altitude_km
        # Increment the shared counter
        Satellite.total_satellites += 1
    
    @classmethod
    def get_satellite_count(cls):
        # Class method - operates on class, not instance
        return cls.total_satellites
    
    @classmethod
    def from_tle(cls, tle_string):
        """Alternative constructor - creates instance from TLE data."""
        # Parse TLE string to extract name and altitude
        name = tle_string.split('\n')[0].strip()
        # ... parsing logic ...
        altitude = 400  # Simplified
        return cls(name, altitude)  # Creates new instance
    
    @staticmethod
    def km_to_miles(km):
        # Static method - doesn't need class or instance
        return km * 0.621371

# Class attributes are shared
sat1 = Satellite("ISS", 408)
sat2 = Satellite("Hubble", 547)
sat3 = Satellite.from_tle("STARLINK-1234\n...")  # Alternative constructor

print(f"Total satellites: {Satellite.total_satellites}")  # 3
print(f"Via instance: {sat1.total_satellites}")  # Also 3 - same value!

# Changing class attribute affects all instances
Satellite.total_satellites = 10
print(f"sat1 sees: {sat1.total_satellites}")  # 10
print(f"sat2 sees: {sat2.total_satellites}")  # 10
```

Here's how Python resolves attribute access:

```{mermaid}
flowchart TD
    A[Access: object.attribute] --> B{Attribute in<br/>instance __dict__?}
    B -->|Yes| C[Return instance attribute]
    B -->|No| D{Attribute in<br/>class __dict__?}
    D -->|Yes| E[Return class attribute]
    D -->|No| F{Check parent<br/>classes MRO}
    F -->|Found| G[Return from parent]
    F -->|Not Found| H[AttributeError]
    
    style C fill:#9f9
    style E fill:#ff9
    style G fill:#9ff
    style H fill:#f99
```

**Method Types Summary**:
- **Instance methods** (most common): Receive `self`, operate on instance data
- **Class methods** (`@classmethod`): Receive `cls`, operate on class data, often used for alternative constructors
- **Static methods** (`@staticmethod`): Don't receive `self` or `cls`, utility functions that belong logically to the class

### Alternative Constructors with Class Methods

Class methods are particularly useful for creating alternative ways to construct objects:

```python
class DataSeries:
    """Time series data with multiple construction methods."""
    
    def __init__(self, times, values):
        """Standard constructor with times and values."""
        self.times = times
        self.values = values
    
    @classmethod
    def from_file(cls, filename):
        """Create DataSeries from a file."""
        times, values = [], []
        with open(filename) as f:
            for line in f:
                t, v = line.split()
                times.append(float(t))
                values.append(float(v))
        return cls(times, values)  # Call regular constructor
    
    @classmethod
    def zeros(cls, n_points, dt=1.0):
        """Create zero-filled series with regular spacing."""
        times = [i * dt for i in range(n_points)]
        values = [0.0] * n_points
        return cls(times, values)
    
    @classmethod
    def from_function(cls, func, t_start, t_end, n_points):
        """Create series by sampling a function."""
        import numpy as np
        times = np.linspace(t_start, t_end, n_points).tolist()
        values = [func(t) for t in times]
        return cls(times, values)

# Multiple ways to create the same type of object
series1 = DataSeries([1, 2, 3], [10, 20, 30])  # Direct
series2 = DataSeries.from_file('data.txt')  # From file
series3 = DataSeries.zeros(100)  # Pre-filled
series4 = DataSeries.from_function(lambda t: t**2, 0, 10, 50)  # From function
```

### 📦 **Computational Thinking Box: Objects as State Machines**

```
PATTERN: Object as State Container

Objects excel at maintaining and managing state over time.
Unlike functions that forget everything between calls,
objects remember their state and can evolve it.

Structure:
- State: Instance attributes hold current values
- Transitions: Methods modify state based on rules
- Queries: Methods return information about state

Real-world applications:
- Simulation particles tracking position/velocity
- Random number generators maintaining seed state
- File objects tracking read/write position
- Neural network layers holding weights
- Iterators remembering position in sequence

This pattern appears throughout scientific Python:
- NumPy arrays remember shape, dtype, data
- Matplotlib figures track all plot elements
- SciPy optimizers maintain convergence history
```

## 6.2 Properties and Encapsulation

Properties let you compute attributes dynamically and control access to object data. They're one of Python's most elegant features, allowing you to write code that looks like simple attribute access but actually runs methods behind the scenes.

### Computed Properties with @property

Sometimes an object's attribute should be calculated from other attributes rather than stored separately. Properties make this transparent:

```python
In [22]: class Rectangle:
   ...:     """A rectangle with computed properties."""
   ...:     
   ...:     def __init__(self, width, height):
   ...:         self.width = width
   ...:         self.height = height
   ...:     
   ...:     @property
   ...:     def area(self):
   ...:         """Area computed from width and height."""
   ...:         return self.width * self.height
   ...:     
   ...:     @property
   ...:     def perimeter(self):
   ...:         """Perimeter computed from width and height."""
   ...:         return 2 * (self.width + self.height)
   ...:     
   ...:     @property
   ...:     def diagonal(self):
   ...:         """Diagonal length."""
   ...:         return (self.width**2 + self.height**2)**0.5

In [23]: rect = Rectangle(3, 4)

In [24]: # Properties look like attributes but are computed
In [25]: print(f"Area: {rect.area}")  # No parentheses!
Area: 12

In [26]: print(f"Diagonal: {rect.diagonal}")
Diagonal: 5.0

In [27]: # When we change dimensions, properties update automatically
In [28]: rect.width = 5
In [29]: print(f"New area: {rect.area}")
New area: 20
```

Properties ensure data consistency. If area were a regular attribute, you'd have to remember to update it every time width or height changed. With properties, it's always correct because it's calculated on demand.

### Setters and Validation

Properties can also validate data when it's set, preventing invalid states:

```python
In [30]: class Temperature:
   ...:     """Temperature with automatic conversion and validation."""
   ...:     
   ...:     def __init__(self, celsius=0):
   ...:         self._celsius = celsius  # Note: underscore indicates "internal"
   ...:     
   ...:     @property
   ...:     def celsius(self):
   ...:         return self._celsius
   ...:     
   ...:     @celsius.setter
   ...:     def celsius(self, value):
   ...:         if value < -273.15:
   ...:             raise ValueError(f"Temperature below absolute zero: {value}°C")
   ...:         self._celsius = value
   ...:     
   ...:     @property
   ...:     def fahrenheit(self):
   ...:         return self._celsius * 9/5 + 32
   ...:     
   ...:     @fahrenheit.setter
   ...:     def fahrenheit(self, value):
   ...:         self.celsius = (value - 32) * 5/9  # Uses celsius setter!
   ...:     
   ...:     @property
   ...:     def kelvin(self):
   ...:         return self._celsius + 273.15
   ...:     
   ...:     @kelvin.setter
   ...:     def kelvin(self, value):
   ...:         self.celsius = value - 273.15  # Reuses validation

In [31]: temp = Temperature(25)

In [32]: # Access in any unit
In [33]: print(f"Celsius: {temp.celsius}°C")
In [34]: print(f"Fahrenheit: {temp.fahrenheit}°F")
In [35]: print(f"Kelvin: {temp.kelvin}K")

In [36]: # Set in any unit
In [37]: temp.fahrenheit = 100
In [38]: print(f"Celsius: {temp.celsius:.1f}°C")
Celsius: 37.8°C

In [39]: # Validation prevents invalid states
In [40]: temp.celsius = -300  # ValueError: Temperature below absolute zero
```

The underscore prefix (`_celsius`) is Python's convention for "internal" attributes. It signals to users "don't access this directly, use the property instead." Python doesn't enforce this — it's a social contract among programmers.

### Property Deletion

For completeness, properties can also have deleters, though they're rarely used:

```python
class ManagedResource:
    def __init__(self):
        self._resource = None
    
    @property
    def resource(self):
        if self._resource is None:
            self._resource = self._acquire_resource()
        return self._resource
    
    @resource.setter
    def resource(self, value):
        self._resource = value
    
    @resource.deleter
    def resource(self):
        """Clean up when resource is deleted."""
        if self._resource is not None:
            print(f"Releasing resource: {self._resource}")
            self._release_resource(self._resource)
            self._resource = None
    
    def _acquire_resource(self):
        print("Acquiring expensive resource...")
        return "ResourceHandle"
    
    def _release_resource(self, resource):
        print(f"Resource {resource} released")

# Usage
obj = ManagedResource()
obj.resource  # Acquires resource
del obj.resource  # Calls deleter, releases resource
```

### Understanding Descriptors: The Magic Behind Properties

For curious students, properties are actually implemented using descriptors, Python's mechanism for customizing attribute access. While you rarely need to write descriptors directly, understanding them demystifies how properties work:

```python
# This is what happens behind the scenes with @property
class Temperature:
    """Using property decorator (recommended approach)."""
    def __init__(self):
        self._celsius = 0
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value

# The above is syntactic sugar for descriptors:
class CelsiusDescriptor:
    """A descriptor that manages temperature (advanced concept)."""
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to class attribute."""
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        """Called when accessing the attribute."""
        if instance is None:
            return self
        return getattr(instance, self.name, 0)
    
    def __set__(self, instance, value):
        """Called when setting the attribute."""
        if value < -273.15:
            raise ValueError("Below absolute zero")
        setattr(instance, self.name, value)

class TemperatureWithDescriptor:
    """Using descriptor directly (rarely needed)."""
    celsius = CelsiusDescriptor()  # Descriptor instance
    
    def __init__(self):
        self.celsius = 0  # Calls descriptor's __set__

# Both approaches work identically:
t1 = Temperature()
t2 = TemperatureWithDescriptor()
t1.celsius = 25  # Uses property
t2.celsius = 25  # Uses descriptor

# Why understand descriptors?
# - They explain how @property works internally
# - They're used by many Python features (methods, classmethod, staticmethod)
# - Advanced libraries like SQLAlchemy use them extensively
# - You probably won't write them, but knowing they exist helps debugging
```

Descriptors are powerful but complex. Stick with `@property` for normal use, but knowing descriptors exist helps when debugging mysterious attribute behavior in advanced libraries.

### ⚠️ **Common Bug Alert: Property Recursion**

```python
# WRONG - Infinite recursion!
class BadClass:
    @property
    def value(self):
        return self.value  # Calls itself forever!
    
    @value.setter
    def value(self, val):
        self.value = val  # Calls setter forever!

# CORRECT - Use different internal name
class GoodClass:
    @property
    def value(self):
        return self._value  # Different name
    
    @value.setter
    def value(self, val):
        self._value = val

# Why this happens:
# self.value in the property calls the property again!
# Always use a different name (usually with underscore) for storage
```

**Remember**: The property name and the internal storage name MUST be different. Convention is to prefix the storage with underscore.

## 6.3 Inheritance: Building on Existing Classes

Inheritance lets you create new classes based on existing ones, inheriting their attributes and methods while adding or modifying functionality. This models "is-a" relationships: a WhiteDwarf is-a Star, so it inherits star properties while adding its own specific features.

```{mermaid}
classDiagram
    class CelestialBody {
        +name: str
        +mass: float
        +radius: float
        +surface_gravity()
        +density()
    }
    
    class Star {
        +temperature: float
        +luminosity()
        +density()
    }
    
    class Planet {
        +orbital_period: float
        +moons: int
        +orbital_velocity()
    }
    
    class WhiteDwarf {
        +cooling_age: float
        +crystallization_fraction()
    }
    
    class GasGiant {
        +ring_system: bool
        +cloud_layers()
    }
    
    class RockyPlanet {
        +plate_tectonics: bool
        +surface_temperature()
    }
    
    CelestialBody <|-- Star : inherits
    CelestialBody <|-- Planet : inherits
    Star <|-- WhiteDwarf : inherits
    Planet <|-- GasGiant : inherits
    Planet <|-- RockyPlanet : inherits
    
    note for CelestialBody "Base class with common properties"
    note for Star "Overrides density() method"
```

### Basic Inheritance

```python
In [41]: class CelestialBody:
   ...:     """Base class for astronomical objects."""
   ...:     
   ...:     def __init__(self, name, mass, radius):
   ...:         self.name = name
   ...:         self.mass = mass      # grams
   ...:         self.radius = radius  # cm
   ...:     
   ...:     def surface_gravity(self):
   ...:         """Calculate surface gravity in cm/s²."""
   ...:         G = 6.674e-8  # CGS units
   ...:         return G * self.mass / self.radius**2
   ...:     
   ...:     def density(self):
   ...:         """Calculate average density in g/cm³."""
   ...:         import math
   ...:         volume = 4/3 * math.pi * self.radius**3
   ...:         return self.mass / volume

In [42]: class Planet(CelestialBody):
   ...:     """A planet with additional properties."""
   ...:     
   ...:     def __init__(self, name, mass, radius, orbital_period, moons=0):
   ...:         # Call parent class constructor
   ...:         super().__init__(name, mass, radius)
   ...:         # Add planet-specific attributes
   ...:         self.orbital_period = orbital_period  # days
   ...:         self.moons = moons
   ...:     
   ...:     def orbital_velocity(self, star_mass):
   ...:         """Calculate orbital velocity around a star."""
   ...:         import math
   ...:         G = 6.674e-8
   ...:         # Kepler's third law to get semi-major axis
   ...:         period_sec = self.orbital_period * 86400
   ...:         a = (G * star_mass * period_sec**2 / (4 * math.pi**2))**(1/3)
   ...:         return 2 * math.pi * a / period_sec

In [43]: # Create a planet
In [44]: earth = Planet("Earth", 5.972e27, 6.371e8, 365.25, moons=1)

In [45]: # Planet inherits methods from CelestialBody
In [46]: print(f"Surface gravity: {earth.surface_gravity():.1f} cm/s²")
Surface gravity: 978.0 cm/s²

In [47]: print(f"Density: {earth.density():.2f} g/cm³")
Density: 5.51 g/cm³

In [48]: # And has its own methods
In [49]: v = earth.orbital_velocity(1.989e33)  # Sun's mass
In [50]: print(f"Orbital velocity: {v/1e5:.1f} km/s")
Orbital velocity: 29.8 km/s
```

### Understanding super()

The `super()` function is crucial for inheritance but often misunderstood. It calls methods from the parent class, but why use it instead of calling the parent directly?

```python
# Three ways to call parent methods:

class Child(Parent):
    def __init__(self, name, child_attr):
        # Method 1: Direct parent call (avoid this)
        Parent.__init__(self, name)
        
        # Method 2: super() without arguments (Python 3+ preferred)
        super().__init__(name)
        
        # Method 3: super() with arguments (Python 2 style, still works)
        super(Child, self).__init__(name)
        
        self.child_attr = child_attr
```

Why use `super()` instead of direct parent calls?

```python
# Problem with direct calls - breaks with multiple inheritance
class A:
    def __init__(self):
        print("A init")

class B(A):
    def __init__(self):
        A.__init__(self)  # Direct call
        print("B init")

class C(A):
    def __init__(self):
        A.__init__(self)  # Direct call
        print("C init")

class D(B, C):  # Multiple inheritance
    def __init__(self):
        B.__init__(self)
        C.__init__(self)
        print("D init")

# Creating D() prints:
# A init  (called by B)
# B init
# A init  (called by C again!)
# C init
# D init

# With super(), A.__init__ is called only once
class B(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("B init")

class C(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("C init")

class D(B, C):
    def __init__(self):
        super().__init__()  # Calls everything in right order
        print("D init")

# Now D() prints:
# A init  (called once!)
# C init
# B init
# D init
```

### Method Resolution Order (MRO)

When you call a method on an object, Python searches for it in a specific order called the Method Resolution Order (MRO). Understanding MRO is crucial for complex inheritance hierarchies:

```{mermaid}
flowchart TD
    A[Call: earth.density()] --> B{Method in<br/>Planet class?}
    B -->|No| C{Method in<br/>CelestialBody?}
    C -->|Yes| D[Use CelestialBody.density()]
    B -->|Yes| E[Use Planet.density()]
    
    F[Call: earth.orbital_velocity()] --> G{Method in<br/>Planet class?}
    G -->|Yes| H[Use Planet.orbital_velocity()]
    G -->|No| I{Method in<br/>CelestialBody?}
    I -->|No| J[AttributeError]
    
    style D fill:#9f9
    style E fill:#9f9
    style H fill:#9f9
    style J fill:#f99
```

You can inspect the MRO:
```python
In [51]: Planet.__mro__
Out[51]: (<class 'Planet'>, <class 'CelestialBody'>, <class 'object'>)

# Python searches in this order: Planet → CelestialBody → object

# For complex hierarchies:
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

### Method Overriding

Child classes can override parent methods to provide specialized behavior:

```python
In [52]: class Star(CelestialBody):
   ...:     """A star with luminosity."""
   ...:     
   ...:     def __init__(self, name, mass, radius, temperature):
   ...:         super().__init__(name, mass, radius)
   ...:         self.temperature = temperature  # Kelvin
   ...:     
   ...:     def luminosity(self):
   ...:         """Calculate luminosity using Stefan-Boltzmann law."""
   ...:         import math
   ...:         sigma = 5.670e-5  # CGS units
   ...:         return 4 * math.pi * self.radius**2 * sigma * self.temperature**4
   ...:     
   ...:     def density(self):
   ...:         """Override density to add warning for stellar densities."""
   ...:         avg_density = super().density()  # Call parent method
   ...:         if avg_density < 0.1:
   ...:             print("Note: This is average density, not core density")
   ...:         return avg_density

In [53]: sun = Star("Sun", 1.989e33, 6.96e10, 5778)
In [54]: density = sun.density()
In [55]: print(f"Solar density: {density:.2f} g/cm³")
Solar density: 1.41 g/cm³
```

### 🔍 **Check Your Understanding**

What's the difference between `super().__init__()` and `ParentClass.__init__(self)`? When might you use each?

<details>
<summary>Answer</summary>

`super().__init__()` is preferred because:
- It follows the Method Resolution Order (MRO), which is crucial for multiple inheritance
- It's more maintainable - if you change the parent class name, you don't need to update the child
- It ensures each parent class is initialized exactly once in complex hierarchies

`ParentClass.__init__(self)` might be used when:
- You need to skip a parent in the hierarchy (rare and usually indicates design issues)
- You're working with legacy Python 2 code that doesn't support `super()` properly
- You explicitly want to call a specific class's method regardless of MRO (very rare)

In practice, always use `super()` unless you have a very specific reason not to.

</details>

### Multiple Inheritance and Mixins

Python supports multiple inheritance, where a class can inherit from multiple parents. This is powerful but can be complex:

```python
# Mixin pattern - small classes that add specific functionality
class TimestampMixin:
    """Adds timestamp tracking to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import time
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def touch(self):
        """Update the timestamp."""
        import time
        self.updated_at = time.time()

class LoggingMixin:
    """Adds logging capability to any class."""
    
    def log(self, message):
        """Log a message with object info."""
        print(f"[{self.__class__.__name__}] {message}")

class Observatory(CelestialBody, TimestampMixin, LoggingMixin):
    """Observatory that tracks celestial bodies with logging."""
    
    def __init__(self, name, mass, radius, telescope_count):
        super().__init__(name, mass, radius)
        self.telescope_count = telescope_count
        self.log(f"Created observatory {name}")
    
    def observe(self, target):
        """Observe a target."""
        self.touch()  # From TimestampMixin
        self.log(f"Observing {target}")  # From LoggingMixin
        return f"Data from {target}"

# Usage
obs = Observatory("Keck", 1e10, 1e5, 2)
# Output: [Observatory] Created observatory Keck
obs.observe("M31")
# Output: [Observatory] Observing M31
```

### The Diamond Problem

Multiple inheritance can create ambiguity when the same method exists in multiple parent classes:

```{mermaid}
flowchart TD
    A[CelestialBody<br/>has density()]
    B[Star<br/>overrides density()]
    C[Planet<br/>overrides density()]
    D[BinaryComponent<br/>Which density()?]
    
    A --> B
    A --> C
    B --> D
    C --> D
    
    style A fill:#f9f
    style D fill:#f99
```

```python
# The diamond problem - which method gets called?
class CelestialBody:
    def density(self):
        return "CelestialBody density"

class Star(CelestialBody):
    def density(self):
        return "Star density"

class Planet(CelestialBody):
    def density(self):
        return "Planet density"

class BinaryComponent(Star, Planet):
    pass  # Which density() method do we inherit?

# Python uses MRO to resolve this
obj = BinaryComponent()
print(obj.density())  # "Star density" - Star comes first in inheritance list
print(BinaryComponent.__mro__)
# Shows the exact order Python will search for methods
```

### Abstract Base Classes

Sometimes you want to define a class that shouldn't be instantiated directly, only inherited:

```python
from abc import ABC, abstractmethod

class Detector(ABC):
    """Abstract base class for all detectors."""
    
    def __init__(self, name, efficiency):
        self.name = name
        self.efficiency = efficiency
    
    @abstractmethod
    def detect(self, photon):
        """Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def calibrate(self):
        """Must be implemented by subclasses."""
        pass
    
    def status(self):
        """Concrete method available to all subclasses."""
        return f"{self.name}: {self.efficiency:.1%} efficient"

class CCDDetector(Detector):
    def detect(self, photon):
        """Implement required method."""
        import random
        return random.random() < self.efficiency
    
    def calibrate(self):
        """Implement required method."""
        self.efficiency *= 0.99  # Degrades over time

# Can't instantiate abstract class
# detector = Detector("test", 0.9)  # TypeError!

# Must use concrete subclass
ccd = CCDDetector("CCD1", 0.85)
print(ccd.detect("photon"))  # Works
```

### A Note on Metaclasses

Metaclasses are classes whose instances are classes themselves. They're Python's mechanism for customizing class creation. While powerful, they're rarely needed in scientific programming:

```python
# This is advanced territory - most programmers never need this

# Normal class creation
class Star:
    pass

# What actually happens internally (simplified):
# Star = type('Star', (object,), {})

# type is the default metaclass
print(type(Star))  # <class 'type'>
print(type(type))  # <class 'type'> - type is its own metaclass!

# Example of when metaclasses might be used (rare):
class SingletonMeta(type):
    """Metaclass that ensures only one instance exists."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    """Only one database connection allowed."""
    def __init__(self):
        self.connected = True

# Both variables reference the same object
db1 = Database()
db2 = Database()
print(db1 is db2)  # True - same object!

# Why mention metaclasses?
# - They complete your understanding of Python's object model
# - Some advanced frameworks use them (Django, SQLAlchemy)
# - If you see "metaclass=" you'll know it's customizing class creation
# - 99% of the time, you DON'T need them - use simpler solutions

# Tim Peters' advice: "Metaclasses are deeper magic than 99% of users
# should ever worry about." Focus on regular classes and inheritance!
```

### Context Managers: Objects that Manage Resources

Context managers are objects that define what happens when entering and exiting a `with` statement. They're crucial for resource management in scientific computing:

```python
class DataFileReader:
    """A complete context manager for reading data files."""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.current_line = 0
        self.total_lines = 0
        
        # Count total lines for progress tracking
        try:
            with open(filename, 'r') as f:
                self.total_lines = sum(1 for line in f if not line.startswith('#'))
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            self.total_lines = 0
    
    def __enter__(self):
        """Called when entering 'with' block."""
        try:
            self.file = open(self.filename, 'r')
            return self  # Return self to be used as the variable after 'as'
        except FileNotFoundError:
            print(f"Error: Cannot open {self.filename}")
            raise
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called when leaving 'with' block.
        
        Parameters
        ----------
        exc_type : type or None
            Exception class if an exception occurred
        exc_value : Exception or None
            Exception instance if an exception occurred
        traceback : traceback or None
            Traceback object if an exception occurred
        
        Returns
        -------
        bool
            True to suppress exception, False to propagate
        """
        if self.file:
            self.file.close()
        
        # Handle specific exceptions if needed
        if exc_type is ValueError:
            print(f"Data parsing error in {self.filename}: {exc_value}")
            return True  # Suppress this specific exception
        
        return False  # Propagate other exceptions
    
    def next_data_line(self):
        """Read next non-comment line."""
        if not self.file:
            return None
        
        for line in self.file:
            if not line.startswith('#'):
                self.current_line += 1
                return line.strip()
        return None
    
    @property
    def progress(self):
        """Calculate reading progress as percentage."""
        if self.total_lines == 0:
            return 0.0
        return (self.current_line / self.total_lines) * 100
    
    def has_more(self):
        """Check if more data remains."""
        return self.current_line < self.total_lines

# Usage with automatic cleanup
with DataFileReader('observations.txt') as reader:
    while reader.has_more():
        line = reader.next_data_line()
        if line:
            print(f"Progress: {reader.progress:.1f}% - Data: {line}")
    # File automatically closed here, even if an error occurs

# The context manager pattern ensures:
# 1. Resources are acquired in __enter__
# 2. Resources are released in __exit__ (even if exceptions occur)
# 3. Exceptions can be handled or propagated as needed

# You can also use contextlib for simpler cases:
from contextlib import contextmanager

@contextmanager
def timing_context(label):
    """Simple context manager using decorator."""
    import time
    start = time.perf_counter()
    print(f"Starting {label}...")
    try:
        yield  # Code block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label} took {elapsed:.3f} seconds")

# Usage
with timing_context("data processing"):
    # Time this code block
    sum(i**2 for i in range(1000000))
```

Sometimes "has-a" relationships (composition) are better than "is-a" relationships (inheritance). Here's how to think about the choice:

```{mermaid}
flowchart TD
    A[Designing Object Relationships] --> B{Is it an 'is-a'<br/>relationship?}
    B -->|Yes| C[Use Inheritance]
    B -->|No| D{Is it a 'has-a'<br/>relationship?}
    D -->|Yes| E[Use Composition]
    D -->|No| F[Reconsider design]
    
    C --> G[Example:<br/>Planet is-a CelestialBody]
    E --> H[Example:<br/>Asteroid has-a Orbit]
    
    style C fill:#9f9
    style E fill:#9ff
```

```python
In [55]: class Orbit:
   ...:     """Orbital parameters (composition approach)."""
   ...:     
   ...:     def __init__(self, semi_major_axis, eccentricity, inclination):
   ...:         self.a = semi_major_axis  # cm
   ...:         self.e = eccentricity
   ...:         self.i = inclination      # radians
   ...:     
   ...:     def period(self, central_mass):
   ...:         """Calculate orbital period."""
   ...:         import math
   ...:         G = 6.674e-8
   ...:         return 2 * math.pi * math.sqrt(self.a**3 / (G * central_mass))

In [56]: class Asteroid:
   ...:     """Asteroid with orbital information (has-a orbit)."""
   ...:     
   ...:     def __init__(self, name, diameter, orbit):
   ...:         self.name = name
   ...:         self.diameter = diameter  # km
   ...:         self.orbit = orbit        # Orbit object
   ...:     
   ...:     def time_to_opposition(self, earth_position):
   ...:         """Calculate time until next opposition."""
   ...:         # Use self.orbit.a, self.orbit.e, etc.
   ...:         pass

In [57]: # Create asteroid with orbit
In [58]: ceres_orbit = Orbit(4.14e13, 0.0758, 0.185)
In [59]: ceres = Asteroid("Ceres", 939.4, ceres_orbit)

In [60]: # Access orbit properties through composition
In [61]: period = ceres.orbit.period(1.989e33)
In [62]: print(f"Orbital period: {period/86400/365.25:.1f} years")
Orbital period: 4.6 years
```

Use inheritance when objects share an "is-a" relationship (Planet is-a CelestialBody). Use composition when objects have a "has-a" relationship (Asteroid has-a Orbit).

## 6.4 Special Methods: Making Objects Pythonic

Special methods (also called magic methods or dunder methods) let your objects behave like built-in Python types. They're surrounded by double underscores and are called automatically by Python in specific situations.

```{mermaid}
flowchart LR
    A[Python Operation] --> B[Special Method Called]
    
    C[print(obj)] --> D[__str__]
    E[obj1 + obj2] --> F[__add__]
    G[len(obj)] --> H[__len__]
    I[obj bracket i bracket] --> J[__getitem__]
    K[for x in obj] --> L[__iter__]
    M[obj1 == obj2] --> N[__eq__]
    O[abs(obj)] --> P[__abs__]
    
    style A fill:#f9f
    style B fill:#9ff
```

### String Representation

The most important special methods control how objects appear when printed:

```python
In [63]: class Vector3D:
   ...:     """A 3D vector with special methods."""
   ...:     
   ...:     def __init__(self, x, y, z):
   ...:         self.x = x
   ...:         self.y = y
   ...:         self.z = z
   ...:     
   ...:     def __str__(self):
   ...:         """Human-readable string representation."""
   ...:         return f"Vector({self.x}, {self.y}, {self.z})"
   ...:     
   ...:     def __repr__(self):
   ...:         """Developer-friendly representation."""
   ...:         return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
   ...:     
   ...:     def __len__(self):
   ...:         """Return dimension (always 3 for 3D vectors)."""
   ...:         return 3
   ...:     
   ...:     def __abs__(self):
   ...:         """Return magnitude."""
   ...:         return (self.x**2 + self.y**2 + self.z**2)**0.5

In [64]: v = Vector3D(3, 4, 0)

In [65]: print(v)  # Calls __str__
Vector(3, 4, 0)

In [66]: v  # In IPython, calls __repr__
Out[66]: Vector3D(x=3, y=4, z=0)

In [67]: len(v)  # Calls __len__
Out[67]: 3

In [68]: abs(v)  # Calls __abs__
Out[68]: 5.0
```

The difference between `__str__` and `__repr__`:
- `__str__`: For end users, should be readable
- `__repr__`: For developers, should be unambiguous (ideally, valid Python to recreate object)

### Arithmetic Operations

Make your objects work with mathematical operators:

```{mermaid}
classDiagram
    class Vector3D {
        +float x
        +float y
        +float z
        +__add__(other) : Vector3D
        +__sub__(other) : Vector3D
        +__mul__(scalar) : Vector3D
        +__truediv__(scalar) : Vector3D
        +__eq__(other) : bool
        +__neg__() : Vector3D
        +__abs__() : float
    }
    
    note for Vector3D "Special methods enable:<br/>v1 + v2 → __add__<br/>v1 - v2 → __sub__<br/>v * 3 → __mul__<br/>v / 2 → __truediv__<br/>-v → __neg__<br/>abs(v) → __abs__"
```

```python
In [69]: class Vector3D:
   ...:     """Vector with arithmetic operations."""
   ...:     
   ...:     def __init__(self, x, y, z):
   ...:         self.x = x
   ...:         self.y = y
   ...:         self.z = z
   ...:     
   ...:     def __add__(self, other):
   ...:         """Vector addition with + operator."""
   ...:         return Vector3D(
   ...:             self.x + other.x,
   ...:             self.y + other.y,
   ...:             self.z + other.z
   ...:         )
   ...:     
   ...:     def __mul__(self, scalar):
   ...:         """Scalar multiplication with * operator."""
   ...:         return Vector3D(
   ...:             self.x * scalar,
   ...:             self.y * scalar,
   ...:             self.z * scalar
   ...:         )
   ...:     
   ...:     def __eq__(self, other):
   ...:         """Equality comparison with == operator."""
   ...:         return (self.x == other.x and 
   ...:                 self.y == other.y and 
   ...:                 self.z == other.z)
   ...:     
   ...:     def __str__(self):
   ...:         return f"({self.x}, {self.y}, {self.z})"

In [70]: v1 = Vector3D(1, 2, 3)
In [71]: v2 = Vector3D(4, 5, 6)

In [72]: v3 = v1 + v2  # Calls __add__
In [73]: print(f"v1 + v2 = {v3}")
v1 + v2 = (5, 7, 9)

In [74]: v4 = v1 * 2  # Calls __mul__
In [75]: print(f"v1 * 2 = {v4}")
v1 * 2 = (2, 4, 6)

In [76]: v1 == v2  # Calls __eq__
Out[76]: False
```

### Container Behavior

Make your objects act like containers (lists, dicts):

```python
In [77]: class DataSeries:
   ...:     """A container for time series data."""
   ...:     
   ...:     def __init__(self, values):
   ...:         self._values = list(values)
   ...:     
   ...:     def __len__(self):
   ...:         """Number of data points."""
   ...:         return len(self._values)
   ...:     
   ...:     def __getitem__(self, index):
   ...:         """Access with square brackets."""
   ...:         return self._values[index]
   ...:     
   ...:     def __setitem__(self, index, value):
   ...:         """Set with square brackets."""
   ...:         self._values[index] = value
   ...:     
   ...:     def __contains__(self, value):
   ...:         """Support 'in' operator."""
   ...:         return value in self._values
   ...:     
   ...:     def __iter__(self):
   ...:         """Make object iterable."""
   ...:         return iter(self._values)

In [78]: data = DataSeries([1, 2, 3, 4, 5])

In [79]: len(data)  # __len__
Out[79]: 5

In [80]: data[2]  # __getitem__
Out[80]: 3

In [81]: data[2] = 10  # __setitem__
In [82]: 10 in data  # __contains__
Out[82]: True

In [83]: # __iter__ makes it work in loops
In [84]: for value in data:
   ...:     print(value, end=' ')
1 2 10 4 5
```

### 🔍 **Check Your Understanding**

If you implement `__eq__` but not `__hash__`, what happens when you try to use your objects in a set? Why?

<details>
<summary>Answer</summary>

Objects that implement `__eq__` but not `__hash__` cannot be used in sets or as dictionary keys. Python will raise a TypeError if you try. Here's why:

```python
class BadClass:
    def __init__(self, value):
        self.value = value
    
    def __eq__(self, other):
        return self.value == other.value
    # No __hash__ defined!

obj1 = BadClass(1)
obj2 = BadClass(1)
print(obj1 == obj2)  # True - __eq__ works

# But can't use in set:
# my_set = {obj1}  # TypeError: unhashable type: 'BadClass'
```

The rule: If two objects are equal (according to `__eq__`), they must have the same hash value. When you override `__eq__`, Python sets `__hash__` to None to prevent you from accidentally breaking this rule.

To fix it, implement `__hash__`:
```python
class GoodClass:
    def __init__(self, value):
        self.value = value
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)  # Hash based on equality criteria

my_set = {GoodClass(1), GoodClass(1)}  # Works!
print(len(my_set))  # 1 - duplicates removed
```

</details>

### 📦 **Computational Thinking Box: The Protocol Pattern**

```
PATTERN: Duck Typing and Protocols

"If it walks like a duck and quacks like a duck, it's a duck"

Python doesn't care about object type, only behavior.
Objects that implement certain special methods can be used
anywhere that behavior is expected.

Common Protocols:
- Iterator: __iter__ and __next__
- Context Manager: __enter__ and __exit__
- Container: __len__, __getitem__, __contains__
- Numeric: __add__, __mul__, __abs__, etc.

This is why:
- Your objects can work with built-in functions
- You can use your objects in for loops
- Your objects can support operators like + and *

NumPy arrays implement these protocols, making them feel
like native Python despite being implemented in C.
This protocol approach is key to Python's flexibility.
```

## 6.5 Debugging and Introspecting Objects

Understanding how to debug and inspect objects is crucial for effective OOP development. Python provides powerful tools for examining objects at runtime.

### Essential Debugging Tools

Python offers several built-in functions for inspecting objects:

```python
class Example:
    """Example class for demonstrating introspection."""
    
    class_var = "shared"
    
    def __init__(self, value):
        self.instance_var = value
        self._protected = "internal"
    
    def method(self):
        return self.instance_var

obj = Example(42)

# 1. dir() - List all attributes and methods
print(dir(obj))
# ['__class__', '__init__', 'class_var', 'instance_var', 'method', ...]

# 2. vars() - Get instance __dict__
print(vars(obj))
# {'instance_var': 42, '_protected': 'internal'}

# 3. type() - Get object's class
print(type(obj))
# <class '__main__.Example'>

# 4. isinstance() - Check class membership
print(isinstance(obj, Example))  # True

# 5. hasattr() - Check if attribute exists
print(hasattr(obj, 'method'))  # True

# 6. getattr() - Get attribute safely
value = getattr(obj, 'missing', 'default')
print(value)  # 'default'

# 7. id() - Get unique object identifier
print(id(obj))  # Memory address

# 8. help() - Get documentation
help(Example)  # Prints class documentation
```

### Debugging AttributeError

AttributeError is one of the most common OOP errors. Here's how to debug it systematically:

```python
class Spacecraft:
    def __init__(self, name):
        self.name = name
        self.fuel = 100
    
    def launch(self):
        if self.fuel > 0:
            self.fuel -= 10
            return "Launched!"

ship = Spacecraft("Voyager")

# Common AttributeError scenarios:

# 1. Typo in attribute name
# print(ship.feul)  # AttributeError: 'feul' not 'fuel'

# Debugging approach:
def debug_attributes(obj, looking_for):
    """Helper to find similar attribute names."""
    attrs = dir(obj)
    print(f"Looking for: {looking_for}")
    print(f"Available attributes: {[a for a in attrs if not a.startswith('_')]}")
    
    # Find similar names
    similar = [a for a in attrs if looking_for.lower() in a.lower()]
    if similar:
        print(f"Did you mean: {similar}?")

# debug_attributes(ship, "feul")
# Output: Did you mean: ['fuel']?

# 2. Forgetting self in method
class BadSpacecraft:
    def __init__(self, name):
        name = name  # Forgot self!
        self.fuel = 100

# bad_ship = BadSpacecraft("Enterprise")
# print(bad_ship.name)  # AttributeError: no 'name'

# 3. Accessing before initialization
class OrderDependent:
    def __init__(self):
        self.computed = self.calculate()  # Uses base before it exists!
        self.base = 10
    
    def calculate(self):
        return self.base * 2  # AttributeError: no 'base' yet

# Debugging with __dict__
def inspect_object(obj):
    """Detailed object inspection."""
    print(f"Object type: {type(obj)}")
    print(f"Object ID: {id(obj)}")
    print("\nInstance attributes:")
    for key, value in vars(obj).items():
        print(f"  {key}: {value!r}")
    print("\nClass attributes:")
    for key in dir(obj.__class__):
        if not key.startswith('_') and not callable(getattr(obj.__class__, key)):
            print(f"  {key}: {getattr(obj.__class__, key)!r}")
```

### Debugging Inheritance Issues

When debugging inheritance, understanding the MRO is crucial:

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

# Debugging MRO
obj = D()
print(f"Method resolution order: {D.__mro__}")
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Which method gets called?
print(f"obj.method() returns: {obj.method()}")  # "B"

# Finding where a method comes from
def find_method_source(obj, method_name):
    """Find which class provides a method."""
    for cls in obj.__class__.__mro__:
        if method_name in cls.__dict__:
            return cls
    return None

source = find_method_source(obj, 'method')
print(f"method comes from: {source}")  # <class 'B'>
```

### ⚠️ **Common Bug Alert: Mutable Default Arguments in Classes**

```python
# DANGEROUS - Mutable default argument
class DataBuffer:
    def __init__(self, initial_data=[]):  # BAD!
        self.data = initial_data
    
    def add(self, value):
        self.data.append(value)

# All instances share the same list!
buffer1 = DataBuffer()
buffer1.add(1)
buffer2 = DataBuffer()
buffer2.add(2)
print(buffer1.data)  # [1, 2] - Both buffers share data!

# CORRECT - Use None and create new list
class DataBuffer:
    def __init__(self, initial_data=None):
        self.data = initial_data if initial_data is not None else []
    
    def add(self, value):
        self.data.append(value)

# Now each instance has its own list
buffer1 = DataBuffer()
buffer1.add(1)
buffer2 = DataBuffer()
buffer2.add(2)
print(buffer1.data)  # [1] - Separate lists!
```

This bug occurs because default arguments are evaluated once when the function is defined, not each time it's called. The same list object is reused for all instances!

## 6.6 Testing Object-Oriented Code

Testing classes requires special considerations beyond testing simple functions. Here's how to write effective tests for your OOP code.

### Basic Class Testing

```python
import unittest

class Star:
    def __init__(self, name, magnitude):
        self.name = name
        self.magnitude = magnitude
    
    def is_visible(self, limiting_magnitude=6.0):
        """Check if star is visible to naked eye."""
        return self.magnitude <= limiting_magnitude
    
    def brightness_ratio(self, other):
        """Calculate brightness ratio with another star."""
        return 10**((other.magnitude - self.magnitude) / 2.5)

class TestStar(unittest.TestCase):
    """Test cases for Star class."""
    
    def setUp(self):
        """Create test objects before each test."""
        self.sirius = Star("Sirius", -1.46)
        self.proxima = Star("Proxima Centauri", 11.13)
    
    def test_initialization(self):
        """Test object creation and attributes."""
        self.assertEqual(self.sirius.name, "Sirius")
        self.assertEqual(self.sirius.magnitude, -1.46)
    
    def test_visibility(self):
        """Test visibility calculation."""
        self.assertTrue(self.sirius.is_visible())
        self.assertFalse(self.proxima.is_visible())
        # Test with custom limit
        self.assertTrue(self.proxima.is_visible(limiting_magnitude=12))
    
    def test_brightness_ratio(self):
        """Test brightness comparison."""
        ratio = self.sirius.brightness_ratio(self.proxima)
        self.assertAlmostEqual(ratio, 8710.7, places=1)
    
    def tearDown(self):
        """Clean up after each test if needed."""
        pass

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Testing Inheritance

```python
class CelestialBody:
    def __init__(self, name, mass):
        self.name = name
        self.mass = mass
    
    def gravitational_parameter(self):
        G = 6.674e-8
        return G * self.mass

class Planet(CelestialBody):
    def __init__(self, name, mass, radius):
        super().__init__(name, mass)
        self.radius = radius
    
    def escape_velocity(self):
        import math
        gm = self.gravitational_parameter()
        return math.sqrt(2 * gm / self.radius)

class TestInheritance(unittest.TestCase):
    """Test inheritance relationships."""
    
    def test_parent_methods_available(self):
        """Child should have parent methods."""
        earth = Planet("Earth", 5.972e27, 6.371e8)
        # Test inherited method
        gm = earth.gravitational_parameter()
        self.assertAlmostEqual(gm, 3.986e20, delta=1e18)
    
    def test_child_extends_parent(self):
        """Child adds new functionality."""
        earth = Planet("Earth", 5.972e27, 6.371e8)
        v_esc = earth.escape_velocity()
        self.assertAlmostEqual(v_esc, 1.12e6, delta=1e4)  # ~11.2 km/s
    
    def test_isinstance_relationships(self):
        """Test class relationships."""
        earth = Planet("Earth", 5.972e27, 6.371e8)
        self.assertIsInstance(earth, Planet)
        self.assertIsInstance(earth, CelestialBody)
        self.assertIsInstance(earth, object)
```

### Testing Properties and Setters

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @property
    def kelvin(self):
        return self._celsius + 273.15

class TestProperties(unittest.TestCase):
    """Test property behavior."""
    
    def test_property_access(self):
        """Properties should work like attributes."""
        temp = Temperature(25)
        self.assertEqual(temp.celsius, 25)
        self.assertAlmostEqual(temp.kelvin, 298.15)
    
    def test_setter_validation(self):
        """Setter should validate input."""
        temp = Temperature()
        # Valid set
        temp.celsius = 100
        self.assertEqual(temp.celsius, 100)
        
        # Invalid set
        with self.assertRaises(ValueError):
            temp.celsius = -300
    
    def test_computed_property_updates(self):
        """Computed properties should update automatically."""
        temp = Temperature(0)
        self.assertEqual(temp.kelvin, 273.15)
        temp.celsius = 100
        self.assertEqual(temp.kelvin, 373.15)
```

### Mocking for Testing

When testing classes that depend on external resources or other complex objects:

```python
from unittest.mock import Mock, patch

class DataFetcher:
    def fetch_from_api(self, url):
        """Would normally make network request."""
        # In real code, this would hit an API
        pass

class Observatory:
    def __init__(self, fetcher):
        self.fetcher = fetcher
    
    def get_weather(self):
        data = self.fetcher.fetch_from_api("weather.api")
        return data['conditions']

class TestWithMocks(unittest.TestCase):
    """Test using mocks for dependencies."""
    
    def test_observatory_with_mock_fetcher(self):
        """Test without real API calls."""
        # Create mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.fetch_from_api.return_value = {
            'conditions': 'clear',
            'temperature': 15
        }
        
        # Test observatory with mock
        obs = Observatory(mock_fetcher)
        conditions = obs.get_weather()
        
        # Verify behavior
        self.assertEqual(conditions, 'clear')
        mock_fetcher.fetch_from_api.assert_called_once_with("weather.api")
    
    @patch('requests.get')
    def test_with_patch(self, mock_get):
        """Patch external dependencies."""
        mock_get.return_value.json.return_value = {'status': 'ok'}
        # Your test code here
        pass
```

### Best Practices for Testing Classes

1. **Test each layer separately**: Test initialization, methods, properties, and special methods independently
2. **Use setUp and tearDown**: Initialize common test objects in setUp, clean up in tearDown
3. **Test edge cases**: Empty inputs, boundary values, invalid inputs
4. **Test the interface, not implementation**: Test what methods do, not how they do it
5. **Mock external dependencies**: Don't let tests depend on files, networks, or databases

## 6.7 When to Use OOP

OOP isn't always the answer. Understanding when to use classes versus simpler approaches is crucial for writing clear, maintainable code.

```{mermaid}
flowchart TD
    A[Design Decision:<br/>How to organize code?] --> B{Does it maintain<br/>state over time?}
    
    B -->|Yes| C{Multiple related<br/>operations on<br/>same data?}
    B -->|No| D{Multiple related<br/>functions?}
    
    C -->|Yes| E[Use a Class]
    C -->|No| F{Complex data<br/>structure?}
    
    D -->|Yes| G[Use a Module]
    D -->|No| H[Use Functions]
    
    F -->|Yes| I[Use Class or<br/>NamedTuple]
    F -->|No| J[Use Dictionary<br/>or Tuple]
    
    E --> K[Examples:<br/>Simulation<br/>DataProcessor<br/>Particle]
    
    G --> L[Examples:<br/>math_utils.py<br/>conversions.py]
    
    H --> M[Examples:<br/>calculate_mean()<br/>convert_units()]
    
    style E fill:#9f9
    style G fill:#ff9
    style H fill:#9ff
```

### When OOP Shines

Use classes when you need to:

**1. Model entities with state and behavior:**
```python
class Simulation:
    """Track simulation state over time."""
    
    def __init__(self, particles, dt=0.01):
        self.particles = particles
        self.time = 0.0
        self.dt = dt
        self.history = []
    
    def step(self):
        """Advance simulation by one timestep."""
        # Update particles
        for p in self.particles:
            p.update(self.dt)
        self.time += self.dt
        self.history.append(self.get_state())
    
    def get_state(self):
        """Capture current state."""
        return {
            'time': self.time,
            'positions': [p.position for p in self.particles],
            'energies': [p.energy() for p in self.particles]
        }
```

**2. Create reusable data structures:**
```python
class TimeSeries:
    """Reusable time series container."""
    
    def __init__(self):
        self.times = []
        self.values = []
    
    def add_point(self, time, value):
        """Add a data point, maintaining time order."""
        # Find insertion point to keep sorted
        import bisect
        idx = bisect.bisect_left(self.times, time)
        self.times.insert(idx, time)
        self.values.insert(idx, value)
    
    def interpolate(self, time):
        """Linearly interpolate value at given time."""
        # Implementation here
        pass
```

**3. Build hierarchies of related concepts:**
```python
# Base class for all detectors
class Detector:
    def __init__(self, name, efficiency):
        self.name = name
        self.efficiency = efficiency
    
    def detect(self, photon):
        import random
        return random.random() < self.efficiency

# Specialized detector types
class CCDDetector(Detector):
    def __init__(self, name, efficiency, pixel_size, quantum_efficiency):
        super().__init__(name, efficiency)
        self.pixel_size = pixel_size
        self.quantum_efficiency = quantum_efficiency
```

### When to Avoid OOP

Don't use classes for:

```{mermaid}
flowchart LR
    A[Avoid Classes For] --> B[Simple Calculations]
    A --> C[Unrelated Utilities]
    A --> D[Pure Data]
    
    B --> E[Use Functions<br/>celsius_to_fahrenheit()]
    C --> F[Use Module<br/>math_utils.py]
    D --> G[Use NamedTuple<br/>or Dictionary]
    
    style A fill:#f99
    style E fill:#9f9
    style F fill:#9f9
    style G fill:#9f9
```

**1. Simple calculations without state:**
```python
# Unnecessary OOP
class TemperatureConverter:
    def celsius_to_fahrenheit(self, celsius):
        return celsius * 9/5 + 32

# Better as a simple function
def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32
```

**2. Collections of unrelated utilities:**
```python
# Awkward OOP
class MathUtils:
    @staticmethod
    def factorial(n):
        # ...
    
    @staticmethod
    def is_prime(n):
        # ...

# Better as module functions
# In math_utils.py:
def factorial(n):
    # ...

def is_prime(n):
    # ...
```

**3. Data without behavior:**
```python
# Overkill OOP for pure data
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Consider simpler alternatives
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Or even just tuples/dicts for very simple cases
point = (x, y)
point = {'x': x, 'y': y}
```

### OOP in Scientific Python Libraries

Understanding how major libraries use OOP helps you write code that integrates well:

```{mermaid}
flowchart TD
    A[Scientific Python OOP Patterns]
    
    A --> B[NumPy Arrays]
    B --> B1[Objects with methods:<br/>arr.mean(), arr.reshape()]
    
    A --> C[Matplotlib]
    C --> C1[Figure/Axes objects:<br/>fig.savefig(), ax.plot()]
    
    A --> D[SciPy]
    D --> D1[Result objects:<br/>result.x, result.fun]
    
    A --> E[Pandas]
    E --> E1[DataFrame objects:<br/>df.groupby(), df.merge()]
    
    style A fill:#f9f
```

**NumPy arrays are objects:**
```python
import numpy as np
arr = np.array([1, 2, 3])
# Methods operate on the array's data
arr.mean()  # Method call
arr.reshape(3, 1)  # Returns new view
arr.sort()  # Modifies in place
```

**Matplotlib uses OOP for complex plots:**
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()  # Create figure and axes objects
ax.plot([1, 2, 3], [1, 4, 9])  # Axes object has plot method
ax.set_xlabel('X')  # Configure axes properties
```

**SciPy optimizers maintain state:**
```python
from scipy.optimize import minimize
result = minimize(objective_func, x0)
# result is an object with attributes
result.x  # Solution
result.fun  # Function value
result.nit  # Number of iterations
```

### Creating NumPy-like Array Classes

Here's a minimal example of how you might create your own array-like class:

```python
class SimpleArray:
    """Minimal array-like class for demonstration."""
    
    def __init__(self, data):
        self._data = list(data)
        self.shape = (len(self._data),)
        self.dtype = type(self._data[0]) if self._data else float
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __setitem__(self, index, value):
        self._data[index] = value
    
    def __repr__(self):
        return f"SimpleArray({self._data})"
    
    def __str__(self):
        return f"[{', '.join(str(x) for x in self._data)}]"
    
    def mean(self):
        """Calculate mean like NumPy arrays."""
        return sum(self._data) / len(self._data)
    
    def sum(self):
        """Sum all elements."""
        return sum(self._data)
    
    def reshape(self, *new_shape):
        """Simplified reshape (only 1D to 2D)."""
        if len(new_shape) == 2:
            rows, cols = new_shape
            if rows * cols != len(self._data):
                raise ValueError("Shape incompatible with data")
            
            result = []
            for i in range(rows):
                row = self._data[i*cols:(i+1)*cols]
                result.append(row)
            return result
        return self._data

# Usage similar to NumPy
arr = SimpleArray([1, 2, 3, 4, 5, 6])
print(arr.mean())  # 3.5
print(arr.reshape(2, 3))  # [[1, 2, 3], [4, 5, 6]]
```

### 🛠️ **Debug This!**

This class has a subtle bug. Can you find it?

```python
class DataBuffer:
    def __init__(self, max_size=100, initial_data=[]):
        self.max_size = max_size
        self.data = initial_data
    
    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.max_size:
            self.data.pop(0)
    
    def get_mean(self):
        return sum(self.data) / len(self.data)

# Test code
buffer1 = DataBuffer()
buffer1.add(10)
buffer2 = DataBuffer()
buffer2.add(20)
print(f"Buffer1: {buffer1.data}")
print(f"Buffer2: {buffer2.data}")
```

<details>
<summary>Bug and Solution</summary>

**Bug**: Mutable default argument! Both buffers share the same list.

Output will be:
```
Buffer1: [10, 20]
Buffer2: [10, 20]
```

**Solution**: Use None as default and create new list in __init__:
```python
def __init__(self, max_size=100, initial_data=None):
    self.max_size = max_size
    self.data = initial_data if initial_data is not None else []
```

This bug appears in classes just like it does in functions!

</details>

## 6.8 Performance Considerations

Understanding how Python implements objects helps you write efficient OOP code.

### 🔊 **Performance Profile: Attribute Access**

Let's measure the cost of different attribute access patterns:

```python
In [85]: import time

In [86]: class SimpleClass:
   ...:     def __init__(self):
   ...:         self.value = 42
   ...:     
   ...:     def get_value(self):
   ...:         return self.value
   ...:     
   ...:     @property
   ...:     def computed_value(self):
   ...:         return self.value

In [87]: obj = SimpleClass()

In [88]: # Time different access methods
In [89]: %timeit obj.value  # Direct attribute
92.3 ns ± 1.2 ns per loop

In [90]: %timeit obj.get_value()  # Method call
118.5 ns ± 2.1 ns per loop

In [91]: %timeit obj.computed_value  # Property
125.7 ns ± 1.8 ns per loop

# Direct attribute access is fastest
# Methods and properties add ~30% overhead
```

### Memory Usage of Objects

Objects have memory overhead compared to simple data structures:

```python
In [92]: import sys

In [93]: # Compare memory usage
In [94]: class Point:
   ...:     def __init__(self, x, y):
   ...:         self.x = x
   ...:         self.y = y

In [95]: # Object approach
In [96]: point_obj = Point(3.0, 4.0)
In [97]: print(f"Object size: {sys.getsizeof(point_obj)} bytes")
In [98]: print(f"Dict size: {sys.getsizeof(point_obj.__dict__)} bytes")
Object size: 48 bytes
Dict size: 296 bytes  # Additional overhead for attribute storage!

In [99]: # Dictionary approach  
In [100]: point_dict = {'x': 3.0, 'y': 4.0}
In [101]: print(f"Dict size: {sys.getsizeof(point_dict)} bytes")
Dict size: 232 bytes

In [102]: # Tuple approach (most memory efficient)
In [103]: point_tuple = (3.0, 4.0)
In [104]: print(f"Tuple size: {sys.getsizeof(point_tuple)} bytes")
Tuple size: 56 bytes

In [105]: # For many points, NumPy is most efficient
In [106]: import numpy as np
In [107]: points_array = np.array([[3.0, 4.0]] * 1000)
In [108]: print(f"Array size for 1000 points: {points_array.nbytes} bytes")
In [109]: print(f"Per point: {points_array.nbytes/1000} bytes")
Array size for 1000 points: 16000 bytes
Per point: 16.0 bytes
```

### Using __slots__ for Memory Efficiency

For classes with many instances, `__slots__` can significantly reduce memory:

```python
In [110]: class RegularPoint:
   ...:     def __init__(self, x, y):
   ...:         self.x = x
   ...:         self.y = y

In [111]: class SlottedPoint:
   ...:     __slots__ = ['x', 'y']  # Fixed attributes
   ...:     
   ...:     def __init__(self, x, y):
   ...:         self.x = x
   ...:         self.y = y

In [112]: regular = RegularPoint(3, 4)
In [113]: slotted = SlottedPoint(3, 4)

In [114]: print(f"Regular: {sys.getsizeof(regular) + sys.getsizeof(regular.__dict__)} bytes")
Regular: 344 bytes

In [115]: print(f"Slotted: {sys.getsizeof(slotted)} bytes")
Slotted: 48 bytes

# Slotted uses 7x less memory!
# But loses flexibility:
In [116]: regular.z = 5  # Works - can add attributes
In [117]: slotted.z = 5  # AttributeError - can't add new attributes!
```

### Understanding __slots__ Tradeoffs

```python
class FlexibleClass:
    """Normal class - can add attributes dynamically."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

class RestrictedClass:
    """Slotted class - fixed attributes for memory efficiency."""
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Flexibility demonstration
flexible = FlexibleClass(1, 2)
flexible.z = 3  # OK - dynamic attribute
flexible.name = "point"  # OK

restricted = RestrictedClass(1, 2)
# restricted.z = 3  # AttributeError!

# Memory comparison for many instances
import sys
flexibles = [FlexibleClass(i, i) for i in range(1000)]
restricteds = [RestrictedClass(i, i) for i in range(1000)]

flex_memory = sum(sys.getsizeof(obj) + sys.getsizeof(obj.__dict__) 
                  for obj in flexibles)
restricted_memory = sum(sys.getsizeof(obj) for obj in restricteds)

print(f"Flexible total: {flex_memory:,} bytes")
print(f"Restricted total: {restricted_memory:,} bytes")
print(f"Savings: {(1 - restricted_memory/flex_memory)*100:.1f}%")
# Typical savings: 85-90%
```

Use `__slots__` when:
- You'll create many instances (thousands+)
- The attributes are fixed and known
- Memory usage is a concern
- You don't need dynamic attributes

Avoid `__slots__` when:
- You need to add attributes dynamically
- You're using multiple inheritance (gets complex)
- You're prototyping and design isn't final
- The class has few instances

## 6.9 Serialization: Saving and Loading Objects

In scientific computing, you often need to save objects to disk for later analysis or to share with colleagues. Python's `pickle` module handles object serialization, but some OOP features require special attention:

```python
import pickle

class SimulationState:
    """A class that can be saved and loaded."""
    
    def __init__(self, particles, time=0.0):
        self.particles = particles
        self.time = time
        self.history = []
        # Some attributes shouldn't be pickled
        self._cache = {}  # Temporary cache
        self._file_handle = None  # File handles can't be pickled
    
    def evolve(self, dt):
        """Advance simulation."""
        self.time += dt
        # Update particles...
        self.history.append(self.time)
    
    def save(self, filename):
        """Save object to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """Load object from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    # Customize pickling behavior if needed
    def __getstate__(self):
        """Control what gets pickled."""
        # Get the default state
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state.pop('_file_handle', None)
        # Optionally clear cache to save space
        state['_cache'] = {}
        return state
    
    def __setstate__(self, state):
        """Control how object is reconstructed."""
        # Restore state
        self.__dict__.update(state)
        # Reinitialize things that weren't pickled
        self._file_handle = None

# Usage
sim = SimulationState(particles=[1, 2, 3])
sim.evolve(0.1)
sim.evolve(0.1)

# Save the simulation
sim.save('simulation.pkl')

# Later, load it back
loaded_sim = SimulationState.load('simulation.pkl')
print(f"Loaded simulation at time {loaded_sim.time}")
print(f"History: {loaded_sim.history}")
```

### Common Pickling Issues and Solutions

```python
# Problem 1: Lambda functions can't be pickled
class BadClass:
    def __init__(self):
        self.transform = lambda x: x**2  # Can't pickle!

class GoodClass:
    def __init__(self):
        self.transform = self._square  # Regular method, can pickle
    
    def _square(self, x):
        return x**2

# Problem 2: Objects with file handles
class FileProcessor:
    def __init__(self, filename):
        self.filename = filename  # Save filename, not file object
        self._file = None
    
    @property
    def file(self):
        """Lazy file opening."""
        if self._file is None:
            self._file = open(self.filename, 'r')
        return self._file
    
    def __getstate__(self):
        """Don't pickle the file handle."""
        state = self.__dict__.copy()
        state['_file'] = None  # Clear file handle
        return state
    
    def __setstate__(self, state):
        """Restore without file handle."""
        self.__dict__ = state
        # File will be reopened on next access

# Problem 3: Classes with __slots__
class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Need special handling for __slots__
    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}
    
    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)

# Alternative: Use dill for more complex objects
# pip install dill
import dill

# dill can handle more object types than pickle
complex_obj = lambda x: x**2  # Lambda function
serialized = dill.dumps(complex_obj)
restored = dill.loads(serialized)
print(restored(5))  # 25

# Best practices for pickleable classes:
# 1. Avoid lambda functions, use regular methods
# 2. Don't store file handles or network connections
# 3. Implement __getstate__/__setstate__ for complex objects
# 4. Test that your objects round-trip correctly
# 5. Consider using dill for complex scientific objects
# 6. Version your pickled formats for long-term storage
```

### Alternatives to Pickle for Scientific Data

```python
# For numerical data, consider these alternatives:

import numpy as np
import json
import h5py  # For HDF5 files

class ScientificData:
    """Class with multiple serialization options."""
    
    def __init__(self, name, data_array, metadata):
        self.name = name
        self.data = data_array  # NumPy array
        self.metadata = metadata  # Dictionary
    
    def save_pickle(self, filename):
        """Standard pickle (good for complex objects)."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def save_numpy(self, filename):
        """NumPy format (efficient for arrays)."""
        np.savez(filename, 
                 data=self.data,
                 metadata=self.metadata,
                 name=self.name)
    
    def save_hdf5(self, filename):
        """HDF5 format (best for large scientific datasets)."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=self.data)
            f.attrs['name'] = self.name
            for key, value in self.metadata.items():
                f.attrs[key] = value
    
    def save_json(self, filename):
        """JSON format (human-readable, portable)."""
        # Note: Need to convert NumPy array to list
        data_dict = {
            'name': self.name,
            'data': self.data.tolist(),
            'metadata': self.metadata
        }
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    @classmethod
    def load_numpy(cls, filename):
        """Load from NumPy format."""
        with np.load(filename, allow_pickle=True) as data:
            return cls(
                name=str(data['name']),
                data_array=data['data'],
                metadata=data['metadata'].item()
            )

# Choose format based on needs:
# - Pickle: Complex Python objects, class instances
# - NumPy: Numerical arrays, fast I/O
# - HDF5: Large datasets, hierarchical data, cross-language
# - JSON: Human-readable, web APIs, configuration
```

### Exercise 6.1: Design a Measurement Class

Create a class for scientific measurements that handles uncertainty:

```python
"""
Design a Measurement class with these requirements:

1. Store a value and its uncertainty
2. Implement arithmetic operations that propagate uncertainty
3. Provide formatted string output with appropriate significant figures
4. Include a method to check if two measurements agree within uncertainty

Error propagation formulas:
- Addition: z = x + y, δz = sqrt(δx² + δy²)
- Subtraction: z = x - y, δz = sqrt(δx² + δy²)
- Multiplication: z = x * y, δz = |z| * sqrt((δx/x)² + (δy/y)²)
- Division: z = x / y, δz = |z| * sqrt((δx/x)² + (δy/y)²)

Pseudocode:
CLASS Measurement:
    INITIALIZE with value and uncertainty
    
    METHOD add(other):
        new_value = self.value + other.value
        new_uncertainty = sqrt(self.uncertainty^2 + other.uncertainty^2)
        RETURN new Measurement
    
    METHOD agrees_with(other):
        difference = abs(self.value - other.value)
        combined_uncertainty = sqrt(self.uncertainty^2 + other.uncertainty^2)
        RETURN difference <= combined_uncertainty

Test with:
- m1 = Measurement(10.0, 0.1)
- m2 = Measurement(10.05, 0.08)
- Check if they agree
- Add them and print result
"""

# Your implementation here
```

### Exercise 6.2: Build a File Reader Class

Create a class that reads data files and tracks reading progress:

```python
"""
Create a DataFileReader class that:

1. Opens a file and reads it line by line
2. Tracks current line number and percentage complete
3. Can skip comment lines (starting with #)
4. Implements context manager protocol (__enter__ and __exit__)
5. Provides methods to read next data line or all remaining lines

Requirements:
- Handle file not found gracefully
- Count total lines on initialization for progress tracking
- Parse numeric data from lines
- Properly handle exceptions in __exit__

Example usage:
with DataFileReader('data.txt') as reader:
    while reader.has_more():
        data = reader.next_data_line()
        print(f"Progress: {reader.progress:.1f}%")
"""

# Enhanced starter template with exception handling pattern:
class DataFileReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.current_line = 0
        self.total_lines = 0
        
        # Pre-count lines for progress tracking
        try:
            with open(filename, 'r') as f:
                self.total_lines = sum(1 for line in f 
                                     if not line.startswith('#'))
        except FileNotFoundError:
            # Handle missing file gracefully
            pass
    
    def __enter__(self):
        """
        Called when entering 'with' block.
        Should open resources and return self.
        """
        # Open file here
        # Handle potential FileNotFoundError
        # Return self for use with 'as' clause
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when leaving 'with' block.
        
        Parameters (automatically passed by Python):
        - exc_type: Exception class if error occurred, None otherwise
        - exc_val: Exception instance if error occurred, None otherwise
        - exc_tb: Traceback object if error occurred, None otherwise
        
        Return:
        - True to suppress the exception
        - False to let the exception propagate
        - None is treated as False
        """
        # Close file if open
        if self.file:
            self.file.close()
        
        # Example exception handling:
        if exc_type is ValueError:
            print(f"Data format error: {exc_val}")
            return True  # Suppress ValueError
        
        # Let other exceptions propagate
        return False
    
    def next_data_line(self):
        """Read next non-comment line."""
        # Skip comment lines
        # Update current_line counter
        # Return parsed data or None
        pass
    
    @property
    def progress(self):
        """Calculate progress as percentage."""
        # Avoid division by zero
        # Return percentage complete
        pass
    
    def has_more(self):
        """Check if more data available."""
        pass
    
    # Add other methods as needed

# Your implementation here
```

### Exercise 6.3: Inheritance Hierarchy

Design a class hierarchy for different types of astronomical observations:

```python
"""
Create these classes:

1. Observation (base class)
   - timestamp, observer, instrument
   - method: is_valid()

2. OpticalObservation(Observation)
   - wavelength, flux, exposure_time
   - method: signal_to_noise()

3. SpectroscopicObservation(OpticalObservation)  
   - wavelength_array, flux_array
   - method: find_emission_lines()

Each class should:
- Call parent __init__ properly
- Add its own attributes
- Override or extend parent methods where appropriate
- Include proper docstrings

Test the inheritance chain and method resolution order.
"""

# Your implementation here
```

### Exercise 6.4: Performance Comparison

Compare different approaches for storing and processing particle data:

```python
"""
Compare three approaches for handling 10,000 particles:

1. List of Particle objects (OOP)
2. Dictionary with lists (structured)
3. NumPy arrays (vectorized)

Each particle has: x, y, z position and vx, vy, vz velocity

Tasks to time:
- Creation/initialization
- Calculate all distances from origin
- Update all positions based on velocity
- Find particles within distance threshold

Measure both time and memory usage.
Which approach is best for which scenarios?
"""

# Example timing code:
import time
import sys

def time_operation(func, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

# Example memory measurement:
def get_size(obj):
    """Get memory size of object."""
    if hasattr(obj, '__dict__'):
        return sys.getsizeof(obj) + sys.getsizeof(obj.__dict__)
    return sys.getsizeof(obj)

# Your implementation here
```

## Key Definitions

**Attribute**: A variable that belongs to an object or class. Instance attributes belong to specific objects, while class attributes are shared by all instances.

**Class**: A blueprint for creating objects. Defines what attributes and methods objects will have.

**Class Method**: A method that receives the class as first argument (`@classmethod`). Often used for alternative constructors.

**Composition**: Design where objects contain other objects ("has-a" relationship).

**Constructor**: The `__init__` method that initializes new objects.

**Decorator**: Special syntax using `@` that modifies functions/methods (`@property`, `@staticmethod`).

**Duck Typing**: Python philosophy that an object's suitability is determined by its methods/attributes, not its type.

**Dunder Methods**: Special methods with double underscores (`__init__`, `__str__`) defining object behavior.

**Encapsulation**: Bundling data and methods within a single unit (class).

**Inheritance**: Mechanism where a class derives from another class ("is-a" relationship).

**Instance**: A specific object created from a class.

**Method**: A function defined inside a class.

**Method Resolution Order (MRO)**: Order Python searches for methods in class hierarchy.

**Object**: A specific instance of a class containing data and methods.

**Overriding**: Subclass providing its own implementation of a parent's method.

**Polymorphism**: Different objects responding to same method call differently.

**Property**: Special attribute executing methods when accessed (`@property`).

**Self**: First parameter of instance methods, referring to the instance.

**Static Method**: Method not receiving self or class (`@staticmethod`).

**Super**: Built-in function accessing parent class methods.

## Key Takeaways

✅ **Classes bundle data and behavior together** - When data and operations naturally go together, classes provide clean organization with state persistence between calls.

✅ **The self parameter is not magic** - It's just Python's way of passing the object to its methods. Forgetting it is the most common OOP error.

✅ **Properties provide controlled access** - Use `@property` for computed attributes, validation, and maintaining data consistency without explicit method calls.

✅ **Inheritance models "is-a", composition models "has-a"** - Choose inheritance for specialized versions, composition for objects that contain others.

✅ **Special methods make objects Pythonic** - Implementing `__str__`, `__len__`, `__add__` lets your objects work with built-in functions and operators.

✅ **super() ensures proper initialization** - Always use `super()` instead of direct parent calls to handle complex inheritance correctly.

✅ **Testing classes requires special consideration** - Test initialization, methods, properties separately. Mock external dependencies.

✅ **OOP has performance tradeoffs** - Objects use more memory than tuples, attribute access has overhead. Use `__slots__` for memory-critical applications.

✅ **Not everything needs to be a class** - Use functions for simple calculations, modules for utilities, classes for stateful objects with behavior.

✅ **Python's flexibility is powerful but requires discipline** - Duck typing and dynamic attributes enable powerful patterns but can hide bugs if misused.

## Quick Reference Tables

### Class Definition Syntax

| Concept | Syntax | Example |
|---------|--------|---------|
| Define class | `class Name:` | `class Star:` |
| Constructor | `def __init__(self):` | `def __init__(self, name):` |
| Instance attribute | `self.attr = value` | `self.mass = 1.989e33` |
| Class attribute | `attr = value` | `SPEED_OF_LIGHT = 3e10` |
| Instance method | `def method(self):` | `def luminosity(self):` |
| Class method | `@classmethod` | `@classmethod`<br>`def from_file(cls):` |
| Static method | `@staticmethod` | `@staticmethod`<br>`def validate():` |
| Property | `@property` | `@property`<br>`def area(self):` |
| Setter | `@attr.setter` | `@temperature.setter` |
| Inheritance | `class Child(Parent):` | `class Planet(CelestialBody):` |

### Common Special Methods

| Method | Purpose | Called by |
|--------|---------|-----------|
| `__init__` | Constructor | `Object()` |
| `__str__` | String for users | `str(obj)`, `print(obj)` |
| `__repr__` | String for developers | `repr(obj)` |
| `__len__` | Length | `len(obj)` |
| `__getitem__` | Get by index | `obj[index]` |
| `__setitem__` | Set by index | `obj[index] = val` |
| `__contains__` | Membership | `item in obj` |
| `__iter__` | Iteration | `for item in obj` |
| `__add__` | Addition | `obj1 + obj2` |
| `__eq__` | Equality | `obj1 == obj2` |
| `__lt__` | Less than | `obj1 < obj2` |
| `__hash__` | Hashing | `hash(obj)`, `{obj}` |

### Debugging Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `dir(obj)` | List all attributes | `dir(star)` |
| `vars(obj)` | Get __dict__ | `vars(star)` |
| `type(obj)` | Get class | `type(star)` |
| `isinstance(obj, cls)` | Check type | `isinstance(star, Star)` |
| `hasattr(obj, attr)` | Check attribute | `hasattr(star, 'mass')` |
| `getattr(obj, attr, default)` | Safe get | `getattr(star, 'age', 0)` |
| `obj.__class__.__mro__` | Get MRO | `Planet.__mro__` |

## Next Chapter Preview

With object-oriented programming mastered, Chapter 7 introduces NumPy — the foundation of scientific computing in Python. You'll discover:

- Why NumPy arrays are 10-100x faster than Python lists
- How vectorization eliminates explicit loops
- Broadcasting rules that enable elegant array operations
- Memory layout and why it matters for performance

The OOP concepts from this chapter directly explain NumPy's design:
- Arrays are objects with methods (`arr.mean()`, `arr.reshape()`)
- Special methods enable mathematical operators (`arr1 + arr2`)
- Properties provide computed attributes (`arr.shape`, `arr.T`)

Understanding objects prepares you to leverage NumPy's full power and eventually create your own scientific classes that integrate seamlessly with the ecosystem.