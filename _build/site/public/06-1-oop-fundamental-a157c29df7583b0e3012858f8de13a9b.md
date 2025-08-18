# Chapter 6: Object-Oriented Programming Fundamentals - Organizing Scientific Code

## Learning Objectives

By the end of this chapter, you will be able to:
- Transform functions and data into cohesive classes that model scientific concepts
- Distinguish between instance and class attributes and choose appropriately for scientific data
- Create methods that operate on object state while understanding the role of `self`
- Implement properties to compute derived values and validate scientific constraints
- Write special methods (`__init__`, `__str__`, `__repr__`) to make objects Pythonic
- Debug common OOP errors using introspection tools and error messages
- Recognize when OOP improves code organization versus when functions suffice
- Connect the transition from procedural to object-oriented thinking in scientific computing

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Define and call functions with various parameter types (Chapter 5)
- ✓ Understand scope and namespaces (Chapter 5)
- ✓ Work with dictionaries and their methods (Chapter 4)
- ✓ Create and import modules (Chapter 5)
- ✓ Handle mutable vs immutable objects (Chapter 4)

Quick diagnostic:
```{code-cell} python
# Can you predict what this prints?
def modify(data):
    data['value'] = data['value'] * 2
    return data

measurement = {'value': 100, 'unit': 'K'}
result = modify(measurement)
print(f"Original: {measurement['value']}")  # What value?
print(f"Result: {result['value']}")         # What value?
```

If you said both print "200", you're ready! Objects work similarly - they're mutable and passed by reference, which becomes crucial when methods modify object state.

## Chapter Overview

You've mastered functions to organize behavior and modules to organize related functions. But what happens when data and the functions that operate on it are inseparable? When tracking particles in a simulation, each particle has position, velocity, and mass, along with methods to update position, calculate kinetic energy, and check collisions. Passing all this data between separate functions becomes error-prone and verbose. This is where Object-Oriented Programming transforms your code from a collection of functions to a model of your problem domain.

Object-Oriented Programming (OOP) isn't just another way to organize code - it's a fundamental shift in how we think about programs. Instead of viewing code as a sequence of operations on data, we model it as interactions between objects that combine data and behavior. A thermometer knows its temperature and how to convert units. A dataset knows its values and how to calculate statistics. A simulation particle knows its state and how to evolve. This paradigm mirrors how we naturally think about scientific systems, making complex programs more intuitive and maintainable.

This chapter introduces OOP's essential concepts through practical scientific examples. You'll learn to create classes (blueprints for objects), instantiate objects (specific instances), and define methods (functions attached to objects). We'll explore how properties provide computed attributes and validation, ensuring your scientific constraints are always satisfied. Most importantly, you'll develop judgment about when OOP clarifies code (managing stateful systems, modeling entities) versus when it adds unnecessary complexity (simple calculations, stateless transformations). By the end, you'll understand why NumPy arrays are objects with methods, setting the foundation for leveraging Python's scientific ecosystem.

## 6.1 From Functions to Objects: The Conceptual Leap

Let's start with a problem you've already solved with functions, then transform it into objects to see the difference:

```{code-cell} python
# Approach 1: Functions with dictionaries (what you know)
def create_particle(mass, x, y, vx, vy):
    """Create a particle dictionary."""
    return {
        'mass': mass,  # grams
        'x': x, 'y': y,  # cm
        'vx': vx, 'vy': vy  # cm/s
    }

def kinetic_energy(particle):
    """Calculate kinetic energy in ergs."""
    v_squared = particle['vx']**2 + particle['vy']**2
    return 0.5 * particle['mass'] * v_squared

def update_position(particle, dt):
    """Update position based on velocity."""
    particle['x'] += particle['vx'] * dt
    particle['y'] += particle['vy'] * dt
    return particle

# Using functions
p1 = create_particle(1.0, 0, 0, 10, 5)
energy = kinetic_energy(p1)
p1 = update_position(p1, 0.1)
print(f"Energy: {energy:.1f} ergs")
```

Now let's see the same problem with OOP:

```{code-cell} python
# Approach 2: Object-Oriented (what you're learning)
class Particle:
    """A particle with position and velocity."""
    
    def __init__(self, mass, x, y, vx, vy):
        """Initialize particle state."""
        self.mass = mass  # grams
        self.x = x        # cm
        self.y = y        # cm
        self.vx = vx      # cm/s
        self.vy = vy      # cm/s
    
    def kinetic_energy(self):
        """Calculate kinetic energy in ergs."""
        v_squared = self.vx**2 + self.vy**2
        return 0.5 * self.mass * v_squared
    
    def update_position(self, dt):
        """Update position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt

# Using objects
p2 = Particle(1.0, 0, 0, 10, 5)
energy = p2.kinetic_energy()
p2.update_position(0.1)
print(f"Energy: {energy:.1f} ergs")
```

Both approaches solve the problem, but notice the differences:
- **Organization**: Data and methods stay together in the class
- **Syntax**: Methods are called on objects (`p2.kinetic_energy()`)
- **State**: The object maintains its own state between method calls
- **Clarity**: The object-oriented version reads more naturally

### 🎭 **The More You Know: How Objects Saved the Mars Rover**

In 2004, NASA's Spirit rover suddenly stopped responding, 18 days into its mission. The cause? Procedural code managing 250+ hardware components through global variables and scattered functions. When flash memory filled up, the initialization functions couldn't track which subsystems were already started, causing an infinite reboot loop.

The fix came from JPL engineer Jennifer Trosper, who had warned about this exact scenario. The team remotely patched the rover's software to use object-oriented design. Each hardware component became an object tracking its own state:

```python
class RoverComponent:
    def __init__(self, name):
        self.name = name
        self.initialized = False
        self.error_count = 0
    
    def initialize(self):
        if not self.initialized:
            # Safe initialization
            self.initialized = True
```

This simple change - objects knowing their own state - saved a $400 million mission. Spirit went on to operate for 6 years instead of the planned 90 days. When Curiosity launched in 2011, its entire control system used OOP from the start. Each instrument is an object, each motor is an object, even each wheel is an object with its own wear tracking.

You're learning the same pattern that keeps billion-dollar spacecraft alive on other planets!

## 6.2 Classes and Objects: Building Blocks

A **class** is a blueprint for creating objects. An **object** (or instance) is a specific realization of that blueprint. Think of a class as the concept "thermometer" and objects as specific thermometers in your lab.

```{code-cell} python
# Define a class (blueprint)
class Measurement:
    """A scientific measurement with uncertainty."""
    
    def __init__(self, value, error):
        """Initialize measurement."""
        self.value = value
        self.error = error
    
    def relative_error(self):
        """Calculate relative error as percentage."""
        if self.value == 0:
            return float('inf')
        return abs(self.error / self.value) * 100

# Create objects (instances)
temp = Measurement(273.15, 0.1)  # Temperature in Kelvin
pressure = Measurement(101325, 50)  # Pressure in Pascal

print(f"Temperature: {temp.value} ± {temp.error} K")
print(f"Relative error: {temp.relative_error():.2f}%")
```

### Understanding `self`

The `self` parameter is how each object keeps track of its own data. When you call `temp.relative_error()`, Python automatically passes `temp` as the first argument:

```{code-cell} python
class Counter:
    """Demonstrates how self works."""
    
    def __init__(self):
        self.count = 0  # Each object gets its own count
    
    def increment(self):
        self.count += 1  # self refers to the specific object
    
    def get_count(self):
        return self.count

# Each object maintains independent state
c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(f"Counter 1: {c1.get_count()}")  # 2
print(f"Counter 2: {c2.get_count()}")  # 1
```

### ⚠️ **Common Bug Alert: Forgetting self**

```{code-cell} python
# WRONG - Missing self parameter
class BadClass:
    def method():  # Missing self!
        return "something"

# This fails:
# obj = BadClass()
# obj.method()  # TypeError: takes 0 arguments but 1 given

# CORRECT - Always include self
class GoodClass:
    def method(self):  # self is required
        return "something"

obj = GoodClass()
print(obj.method())  # Works!
```

This is probably the most common OOP error. Remember: instance methods ALWAYS need `self` as their first parameter.

### Instance vs Class Attributes

Instance attributes belong to specific objects. Class attributes are shared by all instances:

```{code-cell} python
class Simulation:
    """Demonstrates instance vs class attributes."""
    
    # Class attribute - shared by all simulations
    speed_of_light = 2.998e10  # cm/s
    total_runs = 0
    
    def __init__(self, name, particles):
        # Instance attributes - unique to each simulation
        self.name = name
        self.particles = particles
        self.time = 0.0
        # Increment shared counter
        Simulation.total_runs += 1
    
    def advance(self, dt):
        """Advance simulation by dt seconds."""
        self.time += dt

# Create simulations
sim1 = Simulation("Test A", 1000)
sim2 = Simulation("Test B", 5000)

print(f"Total simulations: {Simulation.total_runs}")
print(f"Sim1 particles: {sim1.particles}")
print(f"Sim2 particles: {sim2.particles}")
print(f"Speed of light: {Simulation.speed_of_light} cm/s")
```

### 🔍 **Check Your Understanding**

What's the output of this code? Why?

```{code-cell} python
class DataPoint:
    count = 0  # Class attribute
    
    def __init__(self, value):
        self.value = value  # Instance attribute
        DataPoint.count += 1

p1 = DataPoint(10)
p2 = DataPoint(20)
p1.count = 100  # What happens here?

print(f"p1.count: {p1.count}")
print(f"p2.count: {p2.count}")
print(f"DataPoint.count: {DataPoint.count}")
```

<details>
<summary>Answer</summary>

Output:
- `p1.count: 100` 
- `p2.count: 2`
- `DataPoint.count: 2`

When you write `p1.count = 100`, you create a new instance attribute that shadows the class attribute for `p1` only. The class attribute remains unchanged at 2, and `p2` still sees the class attribute. This is a common source of confusion - instance attributes can hide class attributes with the same name!

</details>

## 6.3 Methods: Functions Attached to Objects

Methods are functions that belong to a class. They can access and modify the object's state through `self`:

```{code-cell} python
class Vector2D:
    """A 2D vector for physics calculations."""
    
    def __init__(self, x, y):
        """Initialize vector components."""
        self.x = x
        self.y = y
    
    def magnitude(self):
        """Calculate vector magnitude."""
        return (self.x**2 + self.y**2)**0.5
    
    def normalize(self):
        """Normalize vector in place."""
        mag = self.magnitude()
        if mag > 0:
            self.x /= mag
            self.y /= mag
    
    def dot(self, other):
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def angle_with(self, other):
        """Calculate angle with another vector in radians."""
        import math
        dot_product = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags == 0:
            return 0
        cos_angle = dot_product / mags
        # Clamp to avoid numerical errors
        cos_angle = max(-1, min(1, cos_angle))
        return math.acos(cos_angle)

# Using vector methods
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 0)

print(f"v1 magnitude: {v1.magnitude()}")
print(f"Dot product: {v1.dot(v2)}")
print(f"Angle: {v1.angle_with(v2):.2f} radians")

v1.normalize()
print(f"After normalization: ({v1.x:.2f}, {v1.y:.2f})")
```

### Method Types: Instance, Class, and Static

```{code-cell} python
class DataProcessor:
    """Demonstrates different method types."""
    
    version = "1.0"
    
    def __init__(self, data):
        self.data = data
    
    # Instance method - needs self
    def process(self):
        """Process this object's data."""
        return sum(self.data) / len(self.data)
    
    # Class method - gets class, not instance
    @classmethod
    def from_file(cls, filename):
        """Create instance from file."""
        # Simulate file reading
        data = [1, 2, 3, 4, 5]
        return cls(data)  # Create new instance
    
    # Static method - doesn't need self or cls
    @staticmethod
    def validate_data(data):
        """Check if data is valid."""
        return len(data) > 0 and all(isinstance(x, (int, float)) for x in data)

# Using different method types
processor = DataProcessor([10, 20, 30])
print(f"Average: {processor.process()}")

# Class method creates new instance
# processor2 = DataProcessor.from_file("data.txt")

# Static method works without instance
valid = DataProcessor.validate_data([1, 2, 3])
print(f"Data valid: {valid}")
```

### 💡 **Computational Thinking Box: Methods as Interface**

```
PATTERN: Public Interface vs Private Implementation

In scientific software, methods define how objects interact.
Think of methods as the object's "API" - what it promises
to do regardless of internal implementation.

Public Interface (what users see):
- particle.update_position(dt)
- measurement.get_uncertainty()
- simulation.run_steps(100)

Private Implementation (internal details):
- How position is stored (Cartesian? polar?)
- How uncertainty is calculated
- What algorithm updates the simulation

This separation allows you to change implementation without
breaking code that uses your objects. NumPy arrays exemplify
this: arr.mean() works the same whether the array is stored
in row-major or column-major order, in RAM or memory-mapped.

Best Practice: Start method names with underscore (_) to
indicate internal methods not meant for external use.
```

## 6.4 Properties: Smart Attributes

Properties let you compute attributes dynamically and validate data when it's set. They make objects safer and more intuitive:

```{code-cell} python
class Circle:
    """Circle with computed properties."""
    
    def __init__(self, radius):
        self._radius = radius  # Note: underscore for internal
    
    @property
    def radius(self):
        """Get radius."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set radius with validation."""
        if value <= 0:
            raise ValueError(f"Radius must be positive, got {value}")
        self._radius = value
    
    @property
    def area(self):
        """Computed area - always up to date."""
        import math
        return math.pi * self._radius**2
    
    @property
    def circumference(self):
        """Computed circumference."""
        import math
        return 2 * math.pi * self._radius

# Properties look like attributes but run code
circle = Circle(5)
print(f"Radius: {circle.radius} cm")
print(f"Area: {circle.area:.2f} cm²")
print(f"Circumference: {circle.circumference:.2f} cm")

# Changing radius automatically updates computed properties
circle.radius = 10
print(f"New area: {circle.area:.2f} cm²")

# Validation prevents invalid states
try:
    circle.radius = -5  # This will raise an error
except ValueError as e:
    print(f"Error: {e}")
```

### Properties for Unit Safety

```{code-cell} python
class Temperature:
    """Temperature with automatic unit conversion."""
    
    def __init__(self, kelvin):
        self._kelvin = kelvin
    
    @property
    def kelvin(self):
        return self._kelvin
    
    @kelvin.setter
    def kelvin(self, value):
        if value < 0:
            raise ValueError("Temperature cannot be below absolute zero")
        self._kelvin = value
    
    @property
    def celsius(self):
        return self._kelvin - 273.15
    
    @celsius.setter
    def celsius(self, value):
        self.kelvin = value + 273.15  # Use kelvin setter for validation
    
    @property
    def fahrenheit(self):
        return self._kelvin * 9/5 - 459.67
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.kelvin = (value + 459.67) * 5/9

# Same temperature, different units
temp = Temperature(300)
print(f"Water at {temp.kelvin:.1f} K")
print(f"  = {temp.celsius:.1f} °C")
print(f"  = {temp.fahrenheit:.1f} °F")

# Set in any unit, always consistent
temp.celsius = 100
print(f"Boiling: {temp.kelvin:.1f} K")
```

### 🌟 **Why This Matters: Mars Climate Orbiter**

In 1999, NASA's Mars Climate Orbiter burned up in Mars' atmosphere. The cause? One team used pound-force seconds, another used newton-seconds. The spacecraft's thrusters fired with 4.45× the intended force.

Modern spacecraft software uses properties to prevent such disasters:

```python
class Thruster:
    @property
    def thrust_newtons(self):
        return self._thrust_n
    
    @thrust_newtons.setter
    def thrust_newtons(self, value):
        self._thrust_n = value
    
    @property
    def thrust_pounds(self):
        return self._thrust_n * 0.224809
    
    @thrust_pounds.setter  
    def thrust_pounds(self, value):
        self._thrust_n = value / 0.224809
```

Properties ensure units are always consistent internally, regardless of what units the user provides. This pattern is now mandatory in NASA flight software.

### ⚠️ **Common Bug Alert: Property Recursion**

```{code-cell} python
# WRONG - Infinite recursion!
class BadExample:
    @property
    def value(self):
        return self.value  # Calls itself forever!

# CORRECT - Use different internal name
class GoodExample:
    def __init__(self):
        self._value = 0  # Underscore prefix
    
    @property
    def value(self):
        return self._value  # Different name

example = GoodExample()
print(f"Value: {example.value}")
```

Always use a different internal name (usually with underscore) for the actual storage.

## 6.5 Special Methods: Making Objects Pythonic

Special methods (also called "magic methods" or "dunder methods") let your objects work with Python's built-in functions and operators:

```{code-cell} python
class Fraction:
    """A fraction with arithmetic operations."""
    
    def __init__(self, numerator, denominator):
        """Initialize fraction."""
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.num = numerator
        self.den = denominator
        self._reduce()
    
    def _reduce(self):
        """Reduce fraction to lowest terms."""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        common = gcd(abs(self.num), abs(self.den))
        self.num //= common
        self.den //= common
    
    def __str__(self):
        """String for print() - human readable."""
        return f"{self.num}/{self.den}"
    
    def __repr__(self):
        """String for repr() - unambiguous."""
        return f"Fraction({self.num}, {self.den})"
    
    def __float__(self):
        """Convert to float."""
        return self.num / self.den
    
    def __add__(self, other):
        """Add two fractions."""
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den * other.den
        return Fraction(new_num, new_den)
    
    def __eq__(self, other):
        """Check equality."""
        return self.num * other.den == other.num * self.den

# Using special methods
f1 = Fraction(1, 2)
f2 = Fraction(1, 3)

print(f"f1 = {f1}")  # Calls __str__
print(f"f1 + f2 = {f1 + f2}")  # Calls __add__
print(f"f1 as float: {float(f1)}")  # Calls __float__
print(f"f1 == Fraction(2,4): {f1 == Fraction(2, 4)}")  # Calls __eq__
```

### Essential Special Methods

```{code-cell} python
class DataSet:
    """Collection that acts like a built-in container."""
    
    def __init__(self, values=None):
        self.values = values if values else []
    
    def __len__(self):
        """Support len(dataset)."""
        return len(self.values)
    
    def __getitem__(self, index):
        """Support dataset[index]."""
        return self.values[index]
    
    def __setitem__(self, index, value):
        """Support dataset[index] = value."""
        self.values[index] = value
    
    def __contains__(self, value):
        """Support 'value in dataset'."""
        return value in self.values
    
    def __iter__(self):
        """Support for loops."""
        return iter(self.values)

# Acts like a built-in container
data = DataSet([10, 20, 30, 40, 50])

print(f"Length: {len(data)}")
print(f"First: {data[0]}")
print(f"Contains 30: {30 in data}")

data[1] = 25
for value in data:
    print(value, end=" ")
```

### 🔍 **Check Your Understanding**

Which special method is called in each line?

```python
obj1 = MyClass(10)    # Line 1
print(obj1)           # Line 2  
result = obj1 + obj2  # Line 3
if obj1 == obj2:      # Line 4
    pass
value = obj1[0]       # Line 5
```

<details>
<summary>Answer</summary>

1. `__init__` - Initialize new object
2. `__str__` - Convert to string for printing
3. `__add__` - Addition operator
4. `__eq__` - Equality comparison
5. `__getitem__` - Index/key access

These special methods are what make Python objects feel "native" - they integrate seamlessly with the language's syntax.

</details>

### 💡 **Computational Thinking Box: Protocol-Based Design**

```
PATTERN: Duck Typing Through Special Methods

"If it walks like a duck and quacks like a duck, it's a duck"

Python doesn't check types - it checks capabilities. Any object
implementing the right special methods can be used anywhere:

Iterator Protocol:
- __iter__() and __next__() → works in for loops

Container Protocol:  
- __len__() and __getitem__() → works with len(), indexing

Numeric Protocol:
- __add__(), __mul__(), etc. → works with math operators

This is why your custom objects can work with built-in functions!
A DataSet with __len__ works with len(). A Vector with __add__
works with +. This protocol-based design is central to Python's
flexibility and why scientific libraries integrate so well.

Real-world example: Any object with .shape, .dtype, and 
__getitem__ can be used where NumPy expects an array-like object.
```

## 6.6 When to Use Objects vs Functions

Not every problem needs objects. Here's how to decide:

### Use Objects When:

1. **Managing State Over Time**
```{code-cell} python
class RunningStatistics:
    """Maintains statistics as data arrives."""
    
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_sq = 0
    
    def add_value(self, x):
        """Add new value to statistics."""
        self.count += 1
        self.sum += x
        self.sum_sq += x**2
    
    @property
    def mean(self):
        """Current mean."""
        return self.sum / self.count if self.count > 0 else 0
    
    @property
    def variance(self):
        """Current variance."""
        if self.count < 2:
            return 0
        mean = self.mean
        return (self.sum_sq - self.count * mean**2) / (self.count - 1)

# Object maintains state between calls
stats = RunningStatistics()
for value in [1, 2, 3, 4, 5]:
    stats.add_value(value)
    print(f"After {value}: mean={stats.mean:.1f}, var={stats.variance:.1f}")
```

2. **Modeling Real Entities**
```{code-cell} python
class Galaxy:
    """Model a galaxy with properties."""
    
    def __init__(self, name, distance_mpc, redshift):
        self.name = name
        self.distance_mpc = distance_mpc  # Megaparsecs
        self.redshift = redshift
    
    def recession_velocity(self):
        """Calculate recession velocity in km/s."""
        c = 3e5  # km/s
        return c * self.redshift  # Simple approximation
    
    def lookback_time_gyr(self):
        """Estimate lookback time in Gyr."""
        # Simplified: t ≈ z / H0
        H0 = 70  # km/s/Mpc
        return self.redshift * 1000 / H0

m31 = Galaxy("Andromeda", 0.78, -0.001)  # Blueshifted!
print(f"{m31.name}: v={m31.recession_velocity():.0f} km/s")
```

### Use Functions When:

1. **Simple Transformations**
```{code-cell} python
# No need for a class here
def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def calculate_orbital_period(semi_major_axis_au):
    """Kepler's third law."""
    return semi_major_axis_au ** 1.5  # years

# Simple functions are clearer than unnecessary classes
temp_k = celsius_to_kelvin(25)
period = calculate_orbital_period(1.0)  # Earth
print(f"Temperature: {temp_k} K")
print(f"Period: {period} years")
```

2. **Stateless Operations**
```{code-cell} python
# These don't need to remember anything between calls
def mean(values):
    """Calculate mean."""
    return sum(values) / len(values)

def standard_deviation(values):
    """Calculate standard deviation."""
    m = mean(values)
    variance = sum((x - m)**2 for x in values) / len(values)
    return variance**0.5

data = [1, 2, 3, 4, 5]
print(f"Mean: {mean(data)}")
print(f"Std: {standard_deviation(data):.2f}")
```

### 🌟 **Why This Matters: The NumPy Decision**

When Travis Oliphant designed NumPy in 2005, he faced this exact decision. Should arrays be simple functions operating on data, or objects with methods?

He chose objects, and it transformed scientific Python:

```python
# If NumPy used only functions:
array = create_array([1, 2, 3])
mean = calculate_mean(array)
reshaped = reshape_array(array, (3, 1))

# Because NumPy uses objects:
array = np.array([1, 2, 3])
mean = array.mean()
reshaped = array.reshape(3, 1)
```

The object approach won because arrays maintain state (shape, dtype, memory layout) and operations naturally belong to the data. This decision made NumPy intuitive and helped it become the foundation of scientific Python. You're learning to make the same architectural decisions!

## 6.7 Debugging Classes

Understanding how to inspect and debug objects is crucial:

```{code-cell} python
class Instrument:
    """Scientific instrument for debugging demo."""
    
    def __init__(self, name, wavelength_nm):
        self.name = name
        self.wavelength_nm = wavelength_nm
        self._calibrated = False
    
    def calibrate(self):
        """Calibrate instrument."""
        self._calibrated = True
        return f"{self.name} calibrated"

# Create instrument
spectrometer = Instrument("HARPS", 500)

# Introspection tools
print(f"Type: {type(spectrometer)}")
print(f"Class name: {spectrometer.__class__.__name__}")
print(f"Is Instrument?: {isinstance(spectrometer, Instrument)}")

# Check attributes
print(f"\nHas 'calibrate'?: {hasattr(spectrometer, 'calibrate')}")
print(f"Wavelength: {getattr(spectrometer, 'wavelength_nm', 'N/A')}")

# List all attributes (filtering out special ones)
attrs = [a for a in dir(spectrometer) if not a.startswith('_')]
print(f"\nPublic attributes: {attrs}")

# See the instance dictionary
print(f"\nInstance dict: {vars(spectrometer)}")
```

### 🐛 **Debug This!**

This code has a subtle but critical bug. Can you find it?

```{code-cell} python
class Observatory:
    def __init__(self, name, telescopes=[]):  # Bug here!
        self.name = name
        self.telescopes = telescopes
    
    def add_telescope(self, telescope):
        self.telescopes.append(telescope)

# Test the code
keck = Observatory("Keck")
keck.add_telescope("Keck I")

vlt = Observatory("VLT")
vlt.add_telescope("Antu")

print(f"Keck telescopes: {keck.telescopes}")
print(f"VLT telescopes: {vlt.telescopes}")  # Unexpected output!
```

<details>
<summary>Solution</summary>

The bug is the mutable default argument `telescopes=[]`. All instances share the same list! When you add a telescope to one observatory, it appears in all of them.

**Fix:**
```python
def __init__(self, name, telescopes=None):
    self.name = name
    self.telescopes = telescopes if telescopes is not None else []
```

This bug has caused real problems in production systems. Always use `None` as default for mutable arguments, then create a new object in the method.

</details>

## 6.8 Practice Exercises

### Exercise 1: Build a DataPoint Class

Let's start with a fundamental building block for data analysis:

```{code-cell} python
"""
Part A: Basic DataPoint class (5 minutes)
Create a class that stores a value with timestamp
"""

import time

class DataPoint:
    """Single measurement with timestamp."""
    
    def __init__(self, value, label=""):
        self.value = value
        self.label = label
        self.timestamp = time.time()
    
    def age_seconds(self):
        """How old is this data point?"""
        return time.time() - self.timestamp

# Test it
dp = DataPoint(42.5, "temperature")
print(f"Value: {dp.value}")
print(f"Label: {dp.label}")
time.sleep(0.1)  # Wait a bit
print(f"Age: {dp.age_seconds():.3f} seconds")
```

```{code-cell} python
"""
Part B: Add validation and properties (10 minutes)
Enhance with properties for safety
"""

class DataPoint:
    """Measurement with validation."""
    
    def __init__(self, value, error=0, label=""):
        self.value = value
        self.error = abs(error)  # Ensure positive
        self.label = label
        self.timestamp = time.time()
    
    @property
    def relative_error(self):
        """Relative error as percentage."""
        if self.value == 0:
            return float('inf')
        return (self.error / abs(self.value)) * 100
    
    @property
    def is_significant(self):
        """Check if measurement is significant."""
        return self.relative_error < 5.0  # Less than 5% error
    
    def __str__(self):
        """Nice string representation."""
        return f"{self.value} ± {self.error} ({self.label})"

# Test enhanced version
dp = DataPoint(100, 2, "voltage")
print(dp)
print(f"Relative error: {dp.relative_error:.1f}%")
print(f"Significant?: {dp.is_significant}")
```

```{code-cell} python
"""
Part C: Complete DataSeries class (15 minutes)
Container for multiple DataPoints with analysis
"""

class DataSeries:
    """Collection of DataPoints with statistics."""
    
    def __init__(self, name):
        self.name = name
        self._points = []
    
    def add_point(self, value, error=0):
        """Add new data point."""
        dp = DataPoint(value, error, self.name)
        self._points.append(dp)
    
    def __len__(self):
        """Number of points."""
        return len(self._points)
    
    def __getitem__(self, index):
        """Access points by index."""
        return self._points[index]
    
    @property
    def values(self):
        """Array of values."""
        return [p.value for p in self._points]
    
    @property
    def mean(self):
        """Calculate mean value."""
        if not self._points:
            return 0
        return sum(self.values) / len(self._points)
    
    @property
    def std_dev(self):
        """Calculate standard deviation."""
        if len(self._points) < 2:
            return 0
        m = self.mean
        variance = sum((x - m)**2 for x in self.values) / (len(self._points) - 1)
        return variance**0.5
    
    def __str__(self):
        return f"DataSeries '{self.name}': {len(self)} points, mean={self.mean:.2f}±{self.std_dev:.2f}"

# Test complete system
series = DataSeries("Temperature")
for temp in [20.1, 20.5, 19.8, 20.2, 20.0]:
    series.add_point(temp, error=0.1)

print(series)
print(f"First point: {series[0]}")
print(f"Last point: {series[-1]}")
print(f"All values: {series.values}")
```

### Exercise 2: Variable Star Class (Continuing Our Thread)

Building on Chapter 5's lightcurve analysis:

```{code-cell} python
"""
Transform our functional variable star analysis into OOP
This connects directly to what you learned in Chapter 5!
"""

class VariableStar:
    """A variable star with photometric properties."""
    
    def __init__(self, name, observations):
        """
        Initialize with observations.
        observations: list of (time, magnitude, error) tuples
        """
        self.name = name
        self.observations = sorted(observations)  # Sort by time
    
    @property
    def mean_magnitude(self):
        """Average magnitude."""
        mags = [obs[1] for obs in self.observations]
        return sum(mags) / len(mags) if mags else 0
    
    @property
    def amplitude(self):
        """Peak-to-peak amplitude."""
        if not self.observations:
            return 0
        mags = [obs[1] for obs in self.observations]
        return max(mags) - min(mags)
    
    @property
    def time_span(self):
        """Total observation time span in days."""
        if len(self.observations) < 2:
            return 0
        times = [obs[0] for obs in self.observations]
        return max(times) - min(times)
    
    def phase_fold(self, period):
        """Fold lightcurve at given period."""
        result = []
        for time, mag, err in self.observations:
            phase = (time % period) / period
            result.append((phase, mag, err))
        return sorted(result)
    
    def __str__(self):
        return (f"VariableStar({self.name}): "
                f"{len(self.observations)} obs, "
                f"<m>={self.mean_magnitude:.2f}, "
                f"amp={self.amplitude:.2f}")
    
    def __len__(self):
        """Number of observations."""
        return len(self.observations)

# Test with simulated Cepheid data
import math
observations = []
for day in range(20):
    time = day + 0.1 * day  # Non-uniform sampling
    # Sinusoidal variation
    phase = 2 * math.pi * time / 5.366  # Delta Cephei period
    magnitude = 4.0 + 0.5 * math.sin(phase)
    error = 0.01
    observations.append((time, magnitude, error))

delta_cep = VariableStar("Delta Cephei", observations)
print(delta_cep)
print(f"Time span: {delta_cep.time_span:.1f} days")

# Phase fold at known period
folded = delta_cep.phase_fold(5.366)
print(f"First folded point: phase={folded[0][0]:.3f}, mag={folded[0][1]:.2f}")
```

### Exercise 3: Performance Comparison

```{code-cell} python
"""
Compare OOP vs functional approaches for performance
Understanding the tradeoffs helps you choose wisely
"""

import time

# OOP Approach
class ParticleOOP:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def energy(self):
        return 0.5 * (self.vx**2 + self.vy**2)

# Functional Approach
def create_particle(x, y, vx, vy):
    return {'x': x, 'y': y, 'vx': vx, 'vy': vy}

def update_particle(p, dt):
    p['x'] += p['vx'] * dt
    p['y'] += p['vy'] * dt
    return p

def particle_energy(p):
    return 0.5 * (p['vx']**2 + p['vy']**2)

# Performance test
n = 10000
dt = 0.01

# OOP timing
start = time.perf_counter()
particles_oop = [ParticleOOP(i, i, 1, 1) for i in range(n)]
for p in particles_oop:
    p.update(dt)
    e = p.energy()
oop_time = time.perf_counter() - start

# Functional timing
start = time.perf_counter()
particles_func = [create_particle(i, i, 1, 1) for i in range(n)]
for p in particles_func:
    update_particle(p, dt)
    e = particle_energy(p)
func_time = time.perf_counter() - start

print(f"OOP approach: {oop_time*1000:.2f} ms")
print(f"Functional approach: {func_time*1000:.2f} ms")
print(f"Ratio: {oop_time/func_time:.2f}x")
print("\nNote: OOP is often slightly slower but provides")
print("better organization and maintainability for complex systems")
```

## Main Takeaways

You've just made a fundamental leap in how you think about programming. Object-Oriented Programming isn't just a different syntax – it's a different mental model. Instead of thinking "what operations do I need to perform on this data?", you now think "what *is* this thing and what can it *do*?" This shift from procedural to object-oriented thinking mirrors how we naturally conceptualize scientific systems. A particle isn't just three numbers for position; it's an entity with mass, velocity, and behaviors like moving and colliding.

The power of OOP becomes clear when managing complexity. That simple Particle class with five attributes and three methods might seem like overkill compared to a dictionary. But when your simulation has thousands of particles, each needing consistent updates, validation, and state tracking, the object-oriented approach prevents the chaos that killed the Mars Climate Orbiter mission. Properties ensure units stay consistent. Methods guarantee state updates follow physical laws. Special methods make your objects work seamlessly with Python's syntax. These aren't just programming conveniences – they're safety mechanisms that prevent billion-dollar disasters.

But perhaps the most important lesson is knowing when NOT to use objects. Not every function needs to become a method. Not every data structure needs to become a class. Simple calculations should stay as functions. Stateless transformations don't need objects. The art lies in recognizing when you're modeling entities with state and behavior (use classes) versus performing operations on data (use functions). This judgment will develop as you write more code, but now you have the conceptual framework to make these decisions thoughtfully.

Looking ahead, everything in Python's scientific stack builds on these concepts. NumPy arrays are objects with methods like `.mean()` and `.reshape()`. Every Matplotlib plot is an object maintaining state about axes, data, and styling. When you write `array.sum()` or `figure.savefig()`, you're using the same patterns you just learned. More importantly, you can now create your own scientific classes that integrate seamlessly with these tools. You're not just learning to use objects – you're learning to think in objects, and that's a superpower for scientific computing.

## Definitions

**Attribute**: A variable that belongs to an object. Instance attributes are unique to each object; class attributes are shared by all instances.

**Class**: A blueprint or template for creating objects. Defines what attributes and methods objects will have.

**Constructor**: The `__init__` method that initializes new objects when they're created.

**Encapsulation**: The bundling of data and methods that operate on that data within a single unit (class).

**Instance**: A specific object created from a class. Each instance has its own set of instance attributes.

**Method**: A function defined inside a class that operates on instances of that class.

**Object**: A specific instance of a class containing data (attributes) and behavior (methods).

**Property**: A special attribute that executes code when accessed or set, created with the `@property` decorator.

**Self**: The first parameter of instance methods, referring to the specific object being operated on.

**Setter**: A property method that validates and sets attribute values, defined with `@attribute.setter`.

**Special Method**: Methods with double underscores (like `__init__`, `__str__`) that define object behavior for built-in operations.

**Static Method**: A method that doesn't receive self or cls, defined with `@staticmethod`.

## Key Takeaways

✓ **Classes combine data and behavior** – Objects bundle related attributes and methods, keeping code organized and preventing errors from mismatched data and functions

✓ **The self parameter connects methods to objects** – It's automatically passed to methods and refers to the specific instance being operated on

✓ **Properties provide smart attributes** – Use `@property` for computed values and validation, ensuring data consistency without explicit method calls

✓ **Special methods make objects Pythonic** – Implementing `__str__`, `__len__`, `__add__` lets your objects work naturally with built-in functions and operators

✓ **Instance attributes belong to objects, class attributes are shared** – Choose instance for object-specific data, class for constants and shared state

✓ **Not everything needs to be a class** – Use objects for stateful entities with behavior, functions for simple calculations and transformations

✓ **Properties prevent unit disasters** – Validation in setters catches errors immediately, preventing Mars Climate Orbiter-style catastrophes

✓ **OOP is the foundation of scientific Python** – NumPy arrays, Matplotlib figures, and Pandas DataFrames all use these patterns

## Quick Reference Tables

### Class Definition Syntax

| Element | Syntax | Example |
|---------|--------|---------|
| Define class | `class Name:` | `class Particle:` |
| Constructor | `def __init__(self):` | `def __init__(self, mass):` |
| Instance attribute | `self.attr = value` | `self.mass = 1.67e-24` |
| Class attribute | `attr = value` | `SPEED_OF_LIGHT = 3e10` |
| Instance method | `def method(self):` | `def velocity(self):` |
| Property getter | `@property` | `@property def energy(self):` |
| Property setter | `@attr.setter` | `@energy.setter` |
| Class method | `@classmethod` | `@classmethod def from_file(cls):` |
| Static method | `@staticmethod` | `@staticmethod def validate():` |

### Essential Special Methods

| Method | Purpose | Called By |
|--------|---------|-----------|
| `__init__` | Initialize object | `MyClass()` |
| `__str__` | Human-readable string | `str(obj)`, `print(obj)` |
| `__repr__` | Developer string | `repr(obj)` |
| `__len__` | Get length | `len(obj)` |
| `__getitem__` | Get by index | `obj[i]` |
| `__setitem__` | Set by index | `obj[i] = val` |
| `__contains__` | Check membership | `x in obj` |
| `__iter__` | Make iterable | `for x in obj` |
| `__add__` | Addition | `obj1 + obj2` |
| `__eq__` | Equality test | `obj1 == obj2` |

### Debugging Object Tools

| Function | Purpose | Example |
|----------|---------|---------|
| `type(obj)` | Get object's class | `type(particle)` |
| `isinstance(obj, cls)` | Check if object is instance | `isinstance(p, Particle)` |
| `hasattr(obj, 'attr')` | Check if attribute exists | `hasattr(p, 'mass')` |
| `getattr(obj, 'attr')` | Get attribute safely | `getattr(p, 'mass', 0)` |
| `setattr(obj, 'attr', val)` | Set attribute | `setattr(p, 'mass', 1.0)` |
| `dir(obj)` | List all attributes | `dir(particle)` |
| `vars(obj)` | Get instance `__dict__` | `vars(particle)` |
| `help(obj)` | Get documentation | `help(Particle)` |

## Next Chapter Preview

In Chapter 7, you'll discover how object-oriented programming scales to advanced patterns. You'll master inheritance to model scientific hierarchies (Particle → ChargedParticle → Electron), use composition for complex systems (Telescope HAS-A Mount, Detector, and Filter), and implement context managers for resource safety. We'll explore performance optimization with `__slots__`, advanced special methods for full container behavior, and design patterns from production scientific software. Most importantly, you'll learn when these advanced techniques clarify versus complicate code. The same OOP foundation you just built powers frameworks like Astropy and SciPy – next chapter shows you how to architect at that level!