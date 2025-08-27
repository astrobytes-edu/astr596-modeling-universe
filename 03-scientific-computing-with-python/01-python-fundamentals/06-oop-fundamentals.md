---
title: "Chapter 6: OOP Fundamentals - Organizing Scientific Code"
subtitle: "Modeling the Universe | Python Fundamentals"
exports:
  - format: pdf
---

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] **(1) Transform functions and data** into cohesive classes that model scientific concepts
- [ ] **(2) Distinguish between** instance and class attributes and choose appropriately for scientific data
- [ ] **(3) Create methods that operate** on object state while understanding the role of `self`
- [ ] **(4) Implement properties** to compute derived values and validate scientific constraints
- [ ] **(5) Write special methods** (`__init__`, `__str__`, `__repr__`) to make objects Pythonic
- [ ] **(6) Debug** common OOP errors using introspection tools and error messages
- [ ] **(7) Recognize when OOP improves** code organization versus when functions suffice
- [ ] **(8) Connect the transition** from procedural to object-oriented thinking in scientific computing

## Prerequisites Check

Before starting this chapter, verify you can:

- [ ] Define and call functions with various parameter types (Chapter 5)
- [ ] Understand scope and namespaces (Chapter 5)
- [ ] Work with dictionaries and their methods (Chapter 4)
- [ ] Create and import modules (Chapter 5)
- [ ] Handle mutable vs immutable objects (Chapter 4)

**Quick diagnostic:**

```{code-cell} ipython3
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

**Object-Oriented Programming** (OOP) isn't just another way to organize code - it's a fundamental shift in how we think about programs. Instead of viewing code as a sequence of operations on data, we model it as interactions between objects that combine data and behavior. A thermometer knows its temperature and how to convert units. A dataset knows its values and how to calculate statistics. A simulation particle knows its state and how to evolve. This paradigm mirrors how we naturally think about scientific systems, making complex programs more intuitive and maintainable.

This chapter introduces OOP's essential concepts through practical scientific examples. You'll learn to create classes (blueprints for objects), instantiate objects (specific instances), and define methods (functions attached to objects). We'll explore how properties provide computed attributes and validation, ensuring your scientific constraints are always satisfied. Most importantly, you'll develop judgment about when OOP clarifies code (managing stateful systems, modeling entities) versus when it adds unnecessary complexity (simple calculations, stateless transformations). By the end, you'll understand why NumPy arrays are objects with methods, setting the foundation for leveraging Python's scientific ecosystem.

---

## 6.1 From Functions to Objects: The Conceptual Leap

:::{margin}
**Object-Oriented Programming**  
A programming paradigm that organizes code around objects (data) and methods (behavior) rather than functions and logic.
:::

Let's start with a problem you've already solved with functions, then transform it into objects to see the difference. In Python, everything is actually an object - even functions and modules! But some objects are more complex than others, and creating your own classes lets you model your specific problem domain.

```{code-cell} ipython3
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

```{code-cell} ipython3
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

:::{admonition} 🌟 The More You Know: How Objects Saved the Mars Rover
:class: history, dropdown

In 2004, NASA's Spirit rover suddenly stopped responding, 18 days into its mission. The cause? Procedural code managing 250+ hardware components through global variables and scattered functions. When flash memory filled up, the initialization functions couldn't track which subsystems were already started, causing an infinite reboot loop.

The fix required remotely clearing flash memory and implementing better state tracking. While the actual fix involved procedural error handling and filesystem limits, the incident highlighted why modern rovers use object-oriented design principles for state management. JPL engineer Jennifer Trosper, who had warned about potential state management issues, helped lead the recovery effort. The team's solution involved better encapsulation of subsystem states - a principle now implemented through OOP in modern missions:

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

This pattern - objects knowing their own state - is now standard in spacecraft software. Spirit went on to operate for 6 years instead of the planned 90 days. When Curiosity launched in 2011, its entire control system used OOP from the start. Each instrument is an object, each motor is an object, even each wheel is an object with its own wear tracking.

You're learning the same pattern that keeps billion-dollar spacecraft alive on other planets!

*[Source: Reeves, G., & Neilson, T. (2005). "The Mars Rover Spirit FLASH Anomaly." IEEE Aerospace Conference Proceedings.]*
:::

---

## 6.2 Classes and Objects: Building Blocks

:::{margin}
**Class**  
A blueprint or template for creating objects that defines attributes and methods.
:::

Before we dive into creating classes, let's understand why we need them beyond the simple example we just saw. As your programs grow, you face several challenges that classes elegantly solve:

**Namespace pollution**: Without classes, you might have functions like `calculate_star_luminosity()`, `calculate_planet_mass()`, `calculate_galaxy_distance()` - your namespace becomes cluttered with hundreds of related functions.

:::{margin}
**Object**  
A specific instance of a class containing data (attributes) and behavior (methods).
:::

**Data consistency**: When data and functions are separate, nothing prevents you from passing a galaxy's data to a star's calculation function, potentially causing silent errors or crashes.

:::{margin}
**Constructor**  
The `__init__` method that initializes new objects when they're created.
:::

**Code reusability**: With functions alone, similar behaviors must be duplicated. Every object type needs its own set of functions even when the logic is similar.

**Conceptual clarity**: We naturally think of stars, planets, and galaxies as entities with properties and behaviors. Classes let us model this intuition directly in code.

A **class** is a blueprint for creating objects. An **object** (or _instance_) is a specific realization of that blueprint. Think of a class as the concept "thermometer" and objects as specific thermometers in your lab.

```{code-cell} ipython3
# Define a class (blueprint)
class Measurement:
    """A scientific measurement with uncertainty."""
    
    def __init__(self, value, error):
        """Initialize measurement with value and error."""
        self.value = value
        self.error = error
    
    def relative_error(self):
        """Calculate relative error as percentage."""
        if self.value == 0:
            return float('inf')
        return abs(self.error / self.value) * 100

# Create objects (instances)
temp = Measurement(293.15, 0.1)  # Temperature in Kelvin
# Pressure in dyne/cm² (CGS unit for pressure)
# Note: CGS (centimeter-gram-second) is traditional in astronomy
pressure = Measurement(1.01325e6, 500)  # 1 atm = 1.01325e6 dyne/cm²

print(f"Temperature: {temp.value} ± {temp.error} K")
print(f"Pressure: {pressure.value:.2e} ± {pressure.error} dyne/cm²")
print(f"Pressure relative error: {pressure.relative_error():.3f}%")
```

### Understanding `self`

:::{margin}
**self**  
The first parameter of instance methods, referring to the specific object being operated on.
:::

The `self` parameter is how each object keeps track of its own data. When you call `temp.relative_error()`, Python automatically passes `temp` as the first argument. Here's what happens behind the scenes:

```{code-cell} ipython3
class Counter:
    """Demonstrates how self works."""
    
    def __init__(self):
        """Initialize counter at zero."""
        self.count = 0  # Each object gets its own count
    
    def increment(self):
        """Increment this counter by one."""
        self.count += 1  # self refers to the specific object
    
    def get_count(self):
        """Return current count value."""
        return self.count

# Each object maintains independent state
c1 = Counter()
c2 = Counter()

c1.increment()
c1.increment()
c2.increment()

print(f"Counter 1: {c1.get_count()}")  # 2
print(f"Counter 2: {c2.get_count()}")  # 1

# What Python actually does when you call c1.increment():
# Counter.increment(c1)  # c1 becomes 'self' in the method
```

This seemingly simple concept of bundling data with behavior revolutionized programming. Let me tell you how it started...

:::{admonition} 🌟 The More You Know: How Norwegian Scientists Invented OOP to Simulate Complex Systems
:class: history, dropdown

In 1962, Norwegian computer scientists Kristen Nygaard and Ole-Johan Dahl faced a mounting challenge at the Norwegian Computing Center. Nygaard had been developing simulations since 1957 - first for nuclear reactor calculations, then for operations research problems. Their early projects included analyzing factory layouts, airport departure systems, and harbor operations. The existing approaches using ALGOL 60 were becoming unwieldy for modeling these complex, interconnected systems with hundreds of interacting components.

Their revolutionary solution? Create "objects" that bundled data with behavior. In 1965, they successfully used [SIMULA](http://kristennygaard.org/FORSKNINGSDOK_MAPPE/F_OO_start.html) I to analyze the Raufoss ammunitions factory layout - determining optimal arrangements for cranes and storage points. The program, punched on 1,130 cards, could simulate 2.5 days of factory operations in just 22 seconds. Each crane, storage point, and workstation became an object that knew its own state and could respond to events.

```python
# Simplified concept in modern Python:
class FactoryStation:
    def __init__(self, name, capacity, processing_time):
        self.name = name
        self.capacity = capacity
        self.queue = []
    
    def receive_item(self, item):
        # Each station manages its own queue and processing
        self.queue.append(item)
```

By 1967, Simula 67 formalized these concepts into classes, inheritance, and virtual methods - the foundation of modern OOP. Alan Kay, influenced by Simula (along with Sketchpad and his biology background), coined "object-oriented programming" around the same time. He later explained: "I thought of objects being like biological cells and/or individual computers on a network, only able to communicate with messages."

The impact was profound but gradual. Simula influenced Smalltalk in the 1970s, then C++ in the 1980s, and eventually Java in the 1990s. The Norwegian Computing Center, initially focused on practical simulation problems, had accidentally created one of computing's most transformative paradigms. Today, when you create a Particle class or a Galaxy object, you're using concepts born from the need to simulate factories, airports, and harbors in 1960s Norway!

*[Sources: Dahl & Nygaard (1978). "The development of the SIMULA languages." ACM SIGPLAN Notices; Kay, A. (2003). Email to Stefan Ram on OOP definition]*
:::

:::{admonition} ⚠️ Common Bug Alert: Forgetting self
:class: warning

```{code-cell} ipython3
:tags: [raises-exception]

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
:::

### Instance vs Class Attributes

:::{margin}
**Instance Attribute**  
Data unique to each object, defined with `self.attribute`.
:::

Instance attributes belong to specific objects. Class attributes are shared by all instances. This bundling of data and methods is called **encapsulation** - a core principle of OOP:

:::{margin}
**Class Attribute**  
Data shared by all instances of a class, defined directly in the class body.
:::

```{code-cell} ipython3
class Simulation:
    """Demonstrates instance vs class attributes."""
    
    # Class attribute - shared by all simulations
    speed_of_light = 2.998e10  # cm/s (CGS)
    total_runs = 0
    
    def __init__(self, name, particle_count):
        """Initialize a new simulation."""
        # Instance attributes - unique to each simulation
        self.name = name
        self.particle_count = particle_count
        self.time = 0.0
        self.box_size_cm = 1.0e-4  # 1 micron box
        # Increment shared counter
        Simulation.total_runs += 1
    
    def advance(self, dt):
        """Advance simulation by dt seconds."""
        self.time += dt

# Create simulations
sim1 = Simulation("Test A", 1000)
sim2 = Simulation("Test B", 5000)

print(f"Total simulations: {Simulation.total_runs}")
print(f"Sim1 particles: {sim1.particle_count}")
print(f"Sim2 particles: {sim2.particle_count}")
print(f"Speed of light: {Simulation.speed_of_light:.2e} cm/s")
print(f"Box size: {sim1.box_size_cm:.2e} cm")
```

::::{admonition} 🔍 Check Your Understanding
:class: question

What's the output of this code? Why?

```{code-cell} ipython3
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

:::{dropdown} Answer
Output:
- `p1.count: 100` 
- `p2.count: 2`
- `DataPoint.count: 2`

When you write `p1.count = 100`, you create a new instance attribute that shadows the class attribute for `p1` only. The class attribute remains unchanged at 2, and `p2` still sees the class attribute. This is a common source of confusion - instance attributes can hide class attributes with the same name!
:::
::::

:::{margin}
**Encapsulation**  
The bundling of data and methods that operate on that data within a single unit (class).
:::

---

## 6.3 Methods: Functions Attached to Objects

:::{margin}
**Method**  
A function defined inside a class that operates on instances of that class.
:::

Methods are functions that belong to a class. They can access and modify the object's state through `self`. Let's build up from simple to complex.

```{code-cell} ipython3
# Note: In production code, always import at file top:
# import math
# Here we import locally for pedagogical clarity

import math  # Best practice: import at the top

# First: A simple class with basic methods
class Sample:
    """Scientific sample with basic operations."""
    
    def __init__(self, mass_g, volume_cm3):
        """Initialize with mass in grams, volume in cm³."""
        self.mass = mass_g
        self.volume = volume_cm3
    
    def density(self):
        """Calculate density in g/cm³."""
        return self.mass / self.volume
    
    def is_denser_than_water(self):
        """Check if denser than water (1 g/cm³)."""
        return self.density() > 1.0

# Using simple methods
iron = Sample(7.87, 1.0)  # Iron sample
print(f"Iron density: {iron.density()} g/cm³")
print(f"Sinks in water: {iron.is_denser_than_water()}")
```

Now let's advance to more complex mathematical methods:

```{code-cell} ipython3
class Vector2D:
    """A 2D vector for physics calculations."""
    
    def __init__(self, x, y):
        """Initialize vector components in cm."""
        self.x = x
        self.y = y
    
    def magnitude(self):
        """Calculate vector magnitude (length)."""
        return (self.x**2 + self.y**2)**0.5
    
    def normalize(self):
        """Scale vector to unit length (magnitude = 1)."""
        mag = self.magnitude()
        if mag > 0:
            self.x /= mag
            self.y /= mag
    
    def dot(self, other):
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def angle_with(self, other):
        """Calculate angle with another vector in radians."""
        dot_product = self.dot(other)
        mags = self.magnitude() * other.magnitude()
        if mags == 0:
            return 0
        cos_angle = dot_product / mags
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = max(-1, min(1, cos_angle))
        return math.acos(cos_angle)

# Using vector methods
v1 = Vector2D(3, 4)
v2 = Vector2D(1, 0)

print(f"v1 magnitude: {v1.magnitude()} cm")
print(f"Dot product: {v1.dot(v2)} cm²")
print(f"Angle: {v1.angle_with(v2):.2f} radians")

v1.normalize()  # Makes magnitude = 1
print(f"After normalization: ({v1.x:.2f}, {v1.y:.2f})")
print(f"New magnitude: {v1.magnitude():.2f}")
```

::::{admonition} 🔍 Check Your Understanding
:class: question

Why does `normalize()` modify the vector in place while `magnitude()` returns a value?

:::{dropdown} Answer
This follows a common convention in programming: methods that transform an object modify it in place (like `list.sort()`), while methods that calculate values return them without changing the object. The names hint at this pattern: "normalize" is a verb suggesting action on the object, while "magnitude" is a noun suggesting a property being measured. This convention helps users predict method behavior from the name alone.
:::
::::

### Method Types: Instance, Class, and Static

```{code-cell} ipython3
class DataProcessor:
    """Demonstrates different method types."""
    
    version = "1.0"
    
    def __init__(self, data):
        """Initialize with data to process."""
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
    
    # Static method with physics example
    @staticmethod
    def celsius_to_kelvin(celsius):
        """Convert temperature - no instance needed."""
        return celsius + 273.15

# Using different method types
processor = DataProcessor([10, 20, 30])
print(f"Average: {processor.process()}")

# Class method creates new instance
# processor2 = DataProcessor.from_file("data.txt")

# Static methods work without instance
valid = DataProcessor.validate_data([1, 2, 3])
print(f"Data valid: {valid}")

# Physics static method
temp_k = DataProcessor.celsius_to_kelvin(25)
print(f"Temperature: {temp_k} K")
```

:::{admonition} 💡 Computational Thinking Box: Methods as Interface
:class: tip

**PATTERN: Public Interface vs Private Implementation**

In scientific software, methods define how objects interact. Think of methods as the object's "API" - what it promises to do regardless of internal implementation.

**Public Interface** (what users see):
- `particle.update_position(dt)`
- `measurement.get_uncertainty()`
- `simulation.run_steps(100)`

**Private Implementation** (internal details):
- How position is stored (Cartesian? polar?)
- How uncertainty is calculated
- What algorithm updates the simulation

This separation allows you to change implementation without breaking code that uses your objects. NumPy arrays exemplify this: `arr.mean()` works the same whether the array is stored in row-major or column-major order, in RAM or memory-mapped.

**Best Practice**: Start method names with underscore (`_`) to indicate internal methods not meant for external use.
:::

:::{admonition} 🌟 Why This Matters: Real Astropy OOP
:class: important, dropdown

Here's how Astropy uses the OOP patterns you're learning:

```python
from astropy.coordinates import SkyCoord
from astropy import units as u

# SkyCoord is a class with properties and methods!
m31 = SkyCoord(ra=10.68*u.degree, dec=41.27*u.degree)
print(m31.galactic)  # Property with coordinate transformation
print(m31.separation(another_coord))  # Method for angular distance

# You're learning the same patterns that power professional astronomy code!
```
:::

---

## 6.4 Properties: Smart Attributes

:::{margin}
**Property**  
A special attribute that executes code when accessed or set, created with the `@property` decorator.
:::

:::{margin}
**Setter**  
A property method that validates and sets attribute values, defined with `@attribute.setter`.
:::

Properties let you compute attributes dynamically and validate data when it's set. They make objects safer and more intuitive:

```{code-cell} ipython3
class Circle:
    """Circle with computed properties."""
    
    def __init__(self, radius):
        """Initialize with radius, using validation."""
        # This now uses the setter for validation!
        self.radius = radius
    
    @property
    def radius(self):
        """Get radius in cm."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set radius with validation."""
        if value <= 0:
            raise ValueError(f"Radius must be positive, got {value}")
        self._radius = value
    
    @property
    def area(self):
        """Computed area in cm² using πr²."""
        return math.pi * self._radius**2
    
    @property
    def circumference(self):
        """Computed circumference in cm using 2πr."""
        return 2 * math.pi * self._radius

# Properties look like attributes but run code
circle = Circle(5)
print(f"Radius: {circle.radius} cm")
print(f"Area: {circle.area:.2f} cm²")
print(f"Circumference: {circle.circumference:.2f} cm")

# Changing radius automatically updates computed properties
circle.radius = 10
print(f"New area: {circle.area:.2f} cm²")

# Validation prevents invalid states (even in __init__ now!)
try:
    bad_circle = Circle(-5)  # This will raise an error
except ValueError as e:
    print(f"Error: {e}")
```

### Properties for Unit Safety

```{code-cell} ipython3
class Temperature:
    """Temperature with automatic unit conversion."""
    
    def __init__(self, kelvin):
        """Initialize with temperature in Kelvin."""
        self._kelvin = kelvin
    
    @property
    def kelvin(self):
        """Get temperature in Kelvin."""
        return self._kelvin
    
    @kelvin.setter
    def kelvin(self, value):
        """Set temperature with validation."""
        if value < 0:
            raise ValueError("Below absolute zero!")
        self._kelvin = value
    
    @property
    def celsius(self):
        """Get temperature in Celsius."""
        return self._kelvin - 273.15
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius."""
        # Use kelvin setter for validation
        self.kelvin = value + 273.15
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit."""
        return self._kelvin * 9/5 - 459.67

# Same temperature, different units
temp = Temperature(300)
print(f"Water at {temp.kelvin:.1f} K")
print(f"  = {temp.celsius:.1f} °C")
print(f"  = {temp.fahrenheit:.1f} °F")

# Set in any unit, always consistent
temp.celsius = 100
print(f"Boiling: {temp.kelvin:.1f} K")
```

### Astronomy-Specific Properties Example

```{code-cell} ipython3
class Observation:
    """Astronomical observation with computed conditions."""
    
    def __init__(self, target, altitude_deg, moon_days_since_new=0):
        """Initialize observation with target and conditions."""
        self.target = target
        self.altitude_deg = altitude_deg  # Object altitude in degrees
        self.moon_days = moon_days_since_new
    
    @property
    def airmass(self):
        """Calculate airmass from altitude (simplified Pickering 2002)."""
        import math
        if self.altitude_deg <= 0:
            return float('inf')
        # Convert to zenith angle
        zenith_deg = 90 - self.altitude_deg
        zenith_rad = math.radians(zenith_deg)
        # Simplified plane-parallel approximation
        # Real calculation uses Pickering's formula
        return 1.0 / math.cos(zenith_rad) if zenith_deg < 85 else 38.0
    
    @property
    def moon_phase(self):
        """Calculate moon illumination fraction (0=new, 1=full)."""
        import math
        # Simplified: actual needs ephemeris
        # Moon cycle is ~29.53 days
        phase_angle = 2 * math.pi * self.moon_days / 29.53
        # Illumination varies as (1 + cos(phase))/2
        return (1 + math.cos(phase_angle)) / 2
    
    @property
    def sky_brightness_category(self):
        """Categorize sky brightness based on moon."""
        if self.moon_phase < 0.25:
            return "Dark"
        elif self.moon_phase < 0.75:
            return "Grey"
        else:
            return "Bright"

# Example observation
obs = Observation("M31", altitude_deg=60, moon_days_since_new=7)
print(f"Observing {obs.target}:")
print(f"  Airmass: {obs.airmass:.2f}")
print(f"  Moon illumination: {obs.moon_phase:.1%}")
print(f"  Sky conditions: {obs.sky_brightness_category}")
```

:::{admonition} 🌟 Why This Matters: The Therac-25 and the Importance of Validation
:class: important, dropdown

Properties with validation could have prevented one of medical history's worst software disasters. Between 1985 and 1987, the Therac-25 radiation therapy machine caused at least six accidents where patients received massive radiation overdoses—up to 100 times the intended dose. Three patients died directly from the overdoses.

The root cause was a lack of validation in the software's state management. The machine could operate in electron-beam mode (low power) or X-ray mode (high power with metal target). A race condition meant the machine could be left in a lethal state: high power WITHOUT the metal target.

With proper validation using properties:

```python
class RadiationTherapyMachine:
    def __init__(self):
        self._mode = None
        self._power_level = 0
        self._target_in_place = False
    
    @property
    def power_level(self):
        return self._power_level
    
    @power_level.setter
    def power_level(self, value):
        # Validation prevents lethal configuration
        if self._mode == "electron" and value > ELECTRON_MAX:
            raise ValueError("Power too high for electron mode!")
        if self._mode == "xray" and not self._target_in_place:
            raise ValueError("X-ray mode requires target!")
        self._power_level = value
```

Today, medical device software uses extensive validation at every state change. Properties ensure that impossible values trigger immediate alerts, not patient deaths. Every validation in your setters follows safety practices written in the aftermath of preventable tragedies.

[Source: Leveson, N.G. & Turner, C.S. (1993). "An Investigation of the Therac-25 Accidents". IEEE Computer, 26(7), 18-41.]
:::

:::{admonition} 🌟 The More You Know: How Properties Could Have Saved Hubble's Mirror
:class: history, dropdown

In 1990, the Hubble Space Telescope reached orbit with a catastrophic flaw - its primary mirror was ground to the wrong shape by 2.2 micrometers, about 1/50th the width of a human hair. The error occurred because a measuring device called a null corrector had been assembled incorrectly, with one lens positioned 1.3mm out of place. But here's the tragic part: the computer software accepting test measurements had no validation. It accepted clearly impossible values without question.

During testing, technicians actually got measurements showing the mirror was wrong. But other tests (using the faulty null corrector) showed it was "perfect." The software happily stored both sets of contradictory data. No validation checks asked: "Why do these measurements disagree by orders of magnitude?" or "Is this curvature physically possible for a mirror this size?"

The servicing mission to install COSTAR cost over $600 million (the total impact including delays and lost science time exceeded $1.5 billion). The repair required a daring Space Shuttle mission in 1993 to install COSTAR - essentially giving Hubble "glasses." But the software fix was equally important. NASA completely rewrote their testing software with aggressive validation:

```python
# Simplified version of the validation concept:
class MirrorMeasurement:
    def __init__(self, expected_curvature):
        self.expected = expected_curvature
        self._curvature_mm = None
    
    @property
    def curvature_mm(self):
        return self._curvature_mm
    
    @curvature_mm.setter
    def curvature_mm(self, value):
        # Physical limits based on mirror specifications
        if not (2200.0 <= value <= 2400.0):  
            raise ValueError(f"Impossible curvature: {value}mm")
        
        # Check against expected value
        deviation = abs(value - self.expected) / self.expected
        if deviation > 0.001:  # 0.1% tolerance
            raise Warning(f"Curvature {value} deviates {deviation*100:.2f}% from expected")
        
        self._curvature_mm = value
```

Note: Modern NASA testing actually uses far more sophisticated validation including statistical process control, multiple sensor cross-validation, and machine learning-based anomaly detection - this example shows the core concept.

Today, every NASA mirror goes through validation software that checks measurements at the moment of entry. Properties ensure that impossible values trigger immediate alerts, not billion-dollar disasters. The James Webb Space Telescope, Hubble's successor, had its mirrors tested with software that validates every measurement against physical constraints, expected ranges, and cross-checks with redundant sensors.

When you write validation in your setters, you're implementing the same safeguards that now protect every space telescope from Hubble's fate. That simple `if value <= 0: raise ValueError()` in your code? That's the pattern that could have saved one of humanity's greatest scientific instruments from launching half-blind into space!

*[Sources: Allen, L. (1990). The Hubble Space Telescope Optical Systems Failure Report. NASA. Simplified technical details for pedagogical purposes.]*
:::

:::{admonition} 🌟 Why This Matters: Mars Climate Orbiter
:class: important, dropdown

In 1999, NASA's Mars Climate Orbiter burned up in Mars' atmosphere. The cause? One team used pound-force seconds, another used newton-seconds. The spacecraft's thrusters fired with 4.45× the intended force.

While Python wasn't used in the 1999 mission (spacecraft used Ada and C++), modern spacecraft software prevents such disasters using property-based validation - a pattern we can now demonstrate in Python:

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

Properties ensure units are always consistent internally, regardless of what units the user provides. This pattern is now mandatory in NASA's modern flight software, preventing the type of error that destroyed the Mars Climate Orbiter.
:::

::::{admonition} 🔍 Check Your Understanding
:class: question

What happens if you create a property without a setter but try to assign to it?

```python
class ReadOnly:
    @property
    def value(self):
        return 42

obj = ReadOnly()
obj.value = 100  # What happens?
```

:::{dropdown} Answer
You get an `AttributeError: can't set attribute`. Properties without setters are read-only. This is actually useful for computed values that should never be directly modified, like the area of a circle (which should only change when the radius changes). This pattern enforces data consistency by preventing invalid states.
:::
::::

:::{admonition} Common Bug Alert: Property Recursion
:class: warning

```{code-cell} ipython3
# WRONG - Infinite recursion!
class BadExample:
    @property
    def value(self):
        return self.value  # Calls itself forever!

# CORRECT - Use different internal name
class GoodExample:
    def __init__(self):
        """Initialize with internal storage."""
        self._value = 0  # Underscore prefix
    
    @property
    def value(self):
        """Access the value."""
        return self._value  # Different name

example = GoodExample()
print(f"Value: {example.value}")
```

Always use a different internal name (usually with underscore) for the actual storage.
:::

---

## 6.5 Special Methods: Making Objects Pythonic

:::{margin}
**Special Method**  
Methods with double underscores (like `__init__`, `__str__`) that define object behavior for built-in operations.
:::

Special methods (also called "magic methods" or "dunder methods") let your objects work with Python's built-in functions and operators. The term "duck typing" comes from the saying "If it walks like a duck and quacks like a duck, it's a duck" - meaning Python cares about what an object can do, not what type it is.

:::{margin}
**Duck Typing**  
Python's philosophy that an object's suitability is determined by its methods, not its type.
:::

:::{admonition} 🌟 The More You Know: How Python Democratized Programming
:class: history, dropdown

In December 1989, Guido van Rossum was frustrated. Working at CWI (Centrum Wiskunde & Informatica) in Amsterdam on the Amoeba distributed operating system, he found existing languages inadequate. ABC was too rigid and couldn't be extended. C was too low-level for rapid development. So during the Christmas vacation (he was bored and the office was closed), he started writing his own language, naming it after the British comedy group Monty Python's Flying Circus.

Guido made a radical decision that would change programming forever: instead of hiding object behavior behind compiler magic like C++ did, Python would expose everything through special methods that anyone could implement. Want your object to work with `len()`? Just add `__len__()`. Want it to support addition? Add `__add__()`. No special compiler support needed - just simple methods with funny names.

This transparency was revolutionary. In C++, only the compiler could decide what `+` meant for built-in types. In Python, ANY object could define it:

```python
# This wasn't possible in other languages of the time!
class Vector:
    def __add__(self, other):
        # YOU decide what + means for YOUR objects
        return Vector(self.x + other.x, self.y + other.y)

v1 + v2  # Calls YOUR __add__ method
```

The scientific community immediately saw the implications. Jim Hugunin created Numeric (NumPy's ancestor) in 1995, using special methods to make arrays feel like native Python objects:

```python
# Scientific arrays that felt built-in!
array1 + array2    # Element-wise addition via __add__
array[5:10]        # Slicing via __getitem__
len(array)         # Size via __len__
print(array)       # Readable output via __str__
```

Guido later reflected: "I wanted Python to be a bridge between the shell and C. I never imagined it would become the language of scientific computing" (paraphrased from various interviews). That bridge was built on special methods - the democratic principle that any object could be a first-class citizen.

Alex Martelli, who would later coin the term "duck typing" for Python's approach, explained it perfectly: "In Python, you don't check if it IS-a duck, you check if it QUACKS-like-a duck, WALKS-like-a duck" (2000, comp.lang.python newsgroup). This philosophy meant scientific libraries could create objects that integrated seamlessly with Python's syntax.

When you implement `__str__` or `__add__`, you're using the democratic principle that made Python the world's most popular scientific language: your objects are equals with Python's built-in types. No special privileges needed - just implement the methods, and Python treats your objects as first-class citizens!

*[Sources: Van Rossum, G. (1996). Foreword for "Programming Python" (1st ed.). Various interviews compiled. Martelli's "duck typing" coined circa 2000.]*
:::

```{code-cell} ipython3
class Fraction:
    """A fraction with arithmetic operations."""
    
    def __init__(self, numerator, denominator):
        """Initialize and reduce fraction."""
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.num = numerator
        self.den = denominator
        self._reduce()
    
    def _reduce(self):
        """Reduce to lowest terms using GCD (Euclidean algorithm)."""
        # Euclidean algorithm: repeatedly replace larger number 
        # with remainder when divided by smaller number
        a, b = abs(self.num), abs(self.den)
        while b:  # Continue until remainder is 0
            a, b = b, a % b  # GCD(a,b) = GCD(b, a mod b)
        self.num //= a
        self.den //= a
    
    def __str__(self):
        """Human-readable string for print()."""
        return f"{self.num}/{self.den}"
    
    def __repr__(self):
        """Unambiguous string for debugging."""
        return f"Fraction({self.num}, {self.den})"
    
    def __float__(self):
        """Convert to float."""
        return self.num / self.den
    
    def __add__(self, other):
        """Add two fractions: a/b + c/d = (ad+bc)/bd"""
        new_num = self.num * other.den + other.num * self.den
        new_den = self.den * other.den
        return Fraction(new_num, new_den)
    
    def __eq__(self, other):
        """Check equality: a/b == c/d if ad == bc"""
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

```{code-cell} ipython3
class DataSet:
    """Collection that acts like a built-in container."""
    
    def __init__(self, values=None):
        """Initialize with optional values."""
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
    
    def __bool__(self):
        """Support if dataset: (True if not empty)."""
        return len(self.values) > 0

# Acts like a built-in container
data = DataSet([10, 20, 30, 40, 50])

print(f"Length: {len(data)}")
print(f"First: {data[0]}")
print(f"Contains 30: {30 in data}")
print(f"Is non-empty: {bool(data)}")

data[1] = 25
for value in data:
    print(value, end=" ")
```

::::{admonition} 🔍 Check Your Understanding
:class: question

Which special method would you implement to make your object work with the `abs()` function?

:::{dropdown} Answer
You would implement `__abs__()`. Python's built-in `abs()` function calls the object's `__abs__()` method if it exists. For example:

```python
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __abs__(self):
        # Return magnitude for abs()
        return (self.x**2 + self.y**2)**0.5

v = Vector2D(3, 4)
print(abs(v))  # Prints 5.0
```

This pattern extends to many built-ins: `len()` calls `__len__()`, `str()` calls `__str__()`, etc. Understanding this connection helps you make objects that feel native to Python.
:::
::::

:::{admonition} 💡 Computational Thinking Box: Protocol-Based Design
:class: tip

**PATTERN: Duck Typing Through Special Methods** <br>
_"If it walks like a duck and quacks like a duck, it's a duck"_

Python doesn't check types - it checks capabilities. Any object implementing the right special methods can be used anywhere:

**Iterator Protocol:**
- `__iter__()` and `__next__()` → works in `for` loops

**Container Protocol:**
- `__len__()` and `__getitem__()` → works with `len()`, indexing

**Numeric Protocol:**
- `__add__()`, `__mul__()`, etc. → works with math operators

**Context Manager Protocol:**
- `__enter__()` and `__exit__()` → works with 'with' statement

**Array Protocol (NumPy):**
- `.shape`, `.dtype`, `__getitem__` → works where NumPy expects array-like

This is why your custom objects can work with built-in functions! A DataSet with `__len__` works with `len()`. A Vector with `__add__` works with `+`. This protocol-based design is central to Python's flexibility and why scientific libraries integrate so well.

**Real-world example:** Any object with `.shape`, `.dtype`, and `__getitem__` can be used where NumPy expects an array-like object. This is how libraries like PyTorch tensors work seamlessly with NumPy functions.
:::

---

## 6.6 When to Use Objects vs Functions

Now that you understand HOW to create classes with all their powerful features - properties for validation, special methods for integration, inheritance for code reuse (Chapter 10) - you need wisdom about WHEN to use them. Not every problem needs objects. Creating unnecessary classes can make code harder to understand, not easier. The art of programming lies in choosing the right tool for the right job.

Here's how to decide. Note: In Chapter 10, we'll explore the "is-a" relationship (inheritance) versus "has-a" relationship (composition) in detail. For now, focus on single classes.

### Use Objects When:

1. **Managing State Over Time**

```{code-cell} ipython3
class RunningStatistics:
    """Maintains statistics as data arrives."""
    
    def __init__(self):
        """Initialize empty statistics."""
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

```{code-cell} ipython3
class Galaxy:
    """Model a galaxy with properties."""
    
    def __init__(self, name, distance_mpc, redshift):
        """Initialize galaxy with observed properties."""
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

```{code-cell} ipython3
# No need for a class here
def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def calculate_orbital_period(semi_major_axis_au):
    """Kepler's third law: P² = a³ for period in years."""
    return semi_major_axis_au ** 1.5

# Simple functions are clearer than unnecessary classes
temp_k = celsius_to_kelvin(25)
period = calculate_orbital_period(1.0)  # Earth
print(f"Temperature: {temp_k} K")
print(f"Period: {period} years")
```

2. **Stateless Operations**

```{code-cell} ipython3
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

:::{admonition} 🌟 Why This Matters: The NumPy Decision
:class: important, dropdown

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

Note: For arrays with millions of elements (common in N-body simulations), the performance difference between OOP and functional approaches can matter. NumPy solves this by implementing operations in C while exposing an OOP interface - the best of both worlds!
:::

:::{admonition} 🌟 The More You Know: How OOP United Astronomy's Warring Packages
:class: history, dropdown

By 2011, Python astronomy had descended into chaos. Every research group had created their own packages with incompatible interfaces. There was PyFITS for reading FITS files, PyWCS for world coordinate systems, vo.table for Virtual Observatory tables, asciitable for text data, cosmolopy for cosmological calculations, and dozens more. Installing a working astronomy environment was a nightmare - each package had different conventions, different dependencies, and different ways of representing the same concepts.

Erik Tollerud, a graduate student at UC Irvine, described the situation: "I spent more time converting between data formats than doing science" (paraphrased from development discussions). A coordinate might be represented as a tuple in one package, a list in another, and a custom object in a third. Unit conversions were handled differently everywhere. Even reading a simple FITS file could require three different packages that didn't talk to each other.

At the 2011 Python in Astronomy conference, something remarkable happened. Thomas Robitaille, Perry Greenfield, Erik Tollerud, and developers from competing packages made a radical decision: merge everything into one coherent framework using consistent OOP principles. The design philosophy was simple but powerful:

1. **If it's an entity with state and behavior, make it a class** (SkyCoord for coordinates, Table for data, Quantity for values with units)
2. **If it's a simple transformation, keep it a function** (unit conversions, mathematical operations)
3. **Everything has units, always** (no more Mars Climate Orbiter disasters)
4. **One obvious way to do things** (borrowed from Python's philosophy)

The transformation was remarkable. This incompatible mess:

```python
# Old way - three packages, incompatible outputs
import pyfits
import pywcs  
import coords

data = pyfits.getdata('image.fits')  # Returns numpy array
header = pyfits.getheader('image.fits')  # Returns header object
wcs = pywcs.WCS(header)  # Different coordinate object
# Convert pixel to sky - returns plain numpy array, no units!
sky = wcs.wcs_pix2sky([[100, 200]], 1)  
# Now convert to different coordinate system - different package!
galactic = coords.Position((sky[0][0], sky[0][1])).galactic()
```

Became this unified interface:

```python
# Astropy way - one package, consistent OOP
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

hdu = fits.open('image.fits')[0]  # Unified HDU object
wcs = WCS(hdu.header)  # Same package, consistent interface
# Returns SkyCoord object with units and frame info!
sky = wcs.pixel_to_world(100, 200)  
galactic = sky.galactic  # Simple property access for conversion
```

The key insight? Objects should model astronomical concepts the way astronomers think about them. A coordinate isn't just numbers - it's a position with a reference frame, epoch, and possibly distance. A table isn't just an array - it has columns with units, metadata, and masks. A quantity isn't just a float - it has units that propagate through calculations.

Today, Astropy has over 10 million downloads and is astronomy's most-used package. The Large Synoptic Survey Telescope, the Event Horizon Telescope that imaged black holes, and the James Webb Space Telescope data pipelines all build on Astropy's OOP foundation. When you're deciding whether to use a class or function, you're making the same architectural decisions that unified an entire scientific community and enabled discoveries like gravitational waves and exoplanets!

*[Sources: Robitaille, T., et al. (2013). Astropy: A community Python package for astronomy. Astronomy & Astrophysics, 558, A33. Development history simplified for pedagogical purposes.]*
:::

## 6.7 Debugging Classes

Understanding how to inspect and debug objects is crucial. Python provides powerful introspection tools to examine objects at runtime:

```{code-cell} ipython3
class Instrument:
    """Scientific instrument for debugging demo."""
    
    def __init__(self, name, wavelength_nm):
        """Initialize instrument with name and wavelength."""
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

# Instance __dict__ vs dir()
# __dict__ shows instance attributes only
# dir() shows everything the object can access
print(f"\nInstance __dict__: {vars(spectrometer)}")
print(f"dir() has {len(dir(spectrometer))} items (includes inherited)")
print(f"__dict__ has {len(vars(spectrometer))} items (instance only)")
```

### Type Hints with Classes (Optional)

Python supports type hints to document expected types, making code clearer:

```{code-cell} ipython3
from typing import Optional, List

class ObservationLog:
    """Demonstrates type hints with classes."""
    
    def __init__(self, observer: str, target: str) -> None:
        """Initialize observation log with type hints."""
        self.observer: str = observer
        self.target: str = target
        self.entries: List[str] = []
        self.completion_time: Optional[float] = None
    
    def add_entry(self, entry: str) -> None:
        """Add log entry."""
        self.entries.append(entry)
    
    def get_entry_count(self) -> int:
        """Return number of entries."""
        return len(self.entries)
    
    def mark_complete(self, time_hours: float) -> None:
        """Mark observation complete."""
        self.completion_time = time_hours

# Type hints help IDEs and documentation
log = ObservationLog("J. Smith", "M31")
log.add_entry("Target acquired")
log.add_entry("Exposure started")
log.mark_complete(2.5)

print(f"Observer: {log.observer}")
print(f"Entries: {log.get_entry_count()}")
print(f"Duration: {log.completion_time} hours")
```

::::{admonition} 🛠️ Debug This!
:class: challenge

This code has a subtle but critical bug. Can you find it?

```{code-cell} ipython3
class Observatory:
    def __init__(self, name, telescopes=[]):  # Bug here!
        self.name = name
        self.telescopes = telescopes
    
    def add_telescope(self, telescope):
        """Add a telescope to this observatory."""
        self.telescopes.append(telescope)

# Test the code
keck = Observatory("Keck")
keck.add_telescope("Keck I")

vlt = Observatory("VLT")
vlt.add_telescope("Antu")

print(f"Keck telescopes: {keck.telescopes}")
print(f"VLT telescopes: {vlt.telescopes}")  # Unexpected output!
```

:::{dropdown} Solution
The bug is the mutable default argument `telescopes=[]`. All instances share the same list! When you add a telescope to one observatory, it appears in all of them.

**Fix:**
```python
def __init__(self, name, telescopes=None):
    self.name = name
    self.telescopes = telescopes if telescopes is not None else []
```

This bug has caused real problems in production systems. Always use `None` as default for mutable arguments, then create a new object in the method.
:::
::::

## Main Takeaways

You've just made a fundamental leap in how you think about programming. Object-Oriented Programming isn't just a different syntax—it's a different mental model. Instead of thinking "what operations do I need to perform on this data?", you now think "what *is* this thing and what can it *do*?" This shift from procedural to object-oriented thinking mirrors how we naturally conceptualize scientific systems. A particle isn't just three numbers for position; it's an entity with mass, velocity, and behaviors like moving and colliding. This conceptual alignment makes complex programs more intuitive and maintainable.

The power of OOP becomes clear when managing complexity. That simple Particle class with five attributes and three methods might seem like overkill compared to a dictionary. But when your simulation has thousands of particles, each needing consistent updates, validation, and state tracking, the object-oriented approach prevents the chaos that killed the Mars Climate Orbiter mission. Properties ensure units stay consistent. Methods guarantee state updates follow physical laws. Special methods make your objects work seamlessly with Python's syntax. These aren't just programming conveniences—they're safety mechanisms that prevent billion-dollar disasters. The Therac-25 radiation overdoses, Hubble's mirror error, and countless other failures could have been prevented with proper encapsulation and validation.

The historical journey from SIMULA to modern Python reveals how OOP emerged from practical needs. Norwegian scientists needed to simulate factories and harbors, leading them to bundle data with behavior—the birth of objects. This paradigm spread through Smalltalk, C++, and Java, eventually reaching Python where Guido van Rossum's radical transparency (special methods anyone can implement) democratized programming. When you write `__add__` to define addition for your objects, you're using the same mechanism that made NumPy arrays feel native to Python. This is why scientific libraries integrate so seamlessly—they all follow the same protocols.

But perhaps the most important lesson is knowing when NOT to use objects. Not every function needs to become a method. Not every data structure needs to become a class. Simple calculations should stay as functions. Stateless transformations don't need objects. The art lies in recognizing when you're modeling entities with state and behavior (use classes) versus performing operations on data (use functions). The NumPy decision to make arrays objects wasn't arbitrary—arrays maintain complex state (shape, dtype, memory layout) and operations naturally belong to the data. The Astropy unification succeeded because astronomical concepts map naturally to objects—a coordinate is more than numbers, it's a position with a reference frame.

Looking ahead, everything in Python's scientific stack builds on these concepts. NumPy arrays are objects with methods like `.mean()` and `.reshape()`. Every Matplotlib plot is an object maintaining state about axes, data, and styling. When you write `array.sum()` or `figure.savefig()`, you're using the same patterns you just learned. More importantly, you can now create your own scientific classes that integrate seamlessly with these tools. You're not just learning to use objects—you're learning to think in objects, and that's a superpower for scientific computing that will serve you throughout your career.

## Definitions

**attribute** - A variable that belongs to an object. Instance attributes are unique to each object; class attributes are shared by all instances

**class** - A blueprint or template for creating objects. Defines what attributes and methods objects will have

**class attribute** - Data shared by all instances of a class, defined directly in the class body

**constructor** - The `__init__` method that initializes new objects when they're created

**duck typing** - Python's philosophy that an object's suitability is determined by its methods and attributes, not its type

**encapsulation** - The bundling of data and methods that operate on that data within a single unit (class)

**instance** - A specific object created from a class. Each instance has its own set of instance attributes

**instance attribute** - Data unique to each object, defined with `self.attribute`

**method** - A function defined inside a class that operates on instances of that class

**object** - A specific instance of a class containing data (attributes) and behavior (methods)

**object-oriented programming** - A programming paradigm that organizes code around objects (data) and methods (behavior)

**property** - A special attribute that executes code when accessed or set, created with the `@property` decorator

**self** - The first parameter of instance methods, referring to the specific object being operated on

**setter** - A property method that validates and sets attribute values, defined with `@attribute.setter`

**special method** - Methods with double underscores (like `__init__`, `__str__`) that define object behavior for built-in operations

**static method** - A method that doesn't receive self or cls, defined with `@staticmethod`

## Key Takeaways

✓ **Classes combine data and behavior** – Objects bundle related attributes and methods, keeping code organized and preventing errors from mismatched data and functions

✓ **The self parameter connects methods to objects** – It's automatically passed to methods and refers to the specific instance being operated on

✓ **Properties provide smart attributes** – Use `@property` for computed values and validation, ensuring data consistency without explicit method calls

✓ **Special methods make objects Pythonic** – Implementing `__str__`, `__len__`, `__add__` lets your objects work naturally with built-in functions and operators

✓ **Instance attributes belong to objects, class attributes are shared** – Choose instance for object-specific data, class for constants and shared state

✓ **Not everything needs to be a class** – Use objects for stateful entities with behavior, functions for simple calculations and transformations

✓ **Properties prevent unit disasters** – Validation in setters catches errors immediately, preventing Mars Climate Orbiter-style catastrophes

✓ **Everything in Python is an object** – Even functions and modules are objects, making Python's object model consistent and powerful

✓ **Duck typing enables flexibility** – Objects work based on capabilities (methods) not types, allowing seamless integration with Python's protocols

✓ **OOP emerged from practical simulation needs** – SIMULA's factory simulations birthed the paradigm that now powers scientific computing

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
| `__bool__` | Truth value | `if obj:`, `bool(obj)` |
| `__call__` | Make callable | `obj()` |
| `__abs__` | Absolute value | `abs(obj)` |
| `__float__` | Convert to float | `float(obj)` |

### When to Use Classes vs Functions

| Use Classes When | Use Functions When |
|-----------------|-------------------|
| Managing state over time | Simple transformations |
| Modeling real entities | Stateless operations |
| Operations belong to data | One-way data flow |
| Need data validation | No state to maintain |
| Complex initialization | Simple input → output |
| Multiple related methods | Single operation |

### Debugging Object Tools

| Function | Purpose | Example |
|----------|---------|---------|
| `type(obj)` | Get object's class | `type(particle)` |
| `isinstance(obj, cls)` | Check if object is instance | `isinstance(p, Particle)` |
| `hasattr(obj, 'attr')` | Check if attribute exists | `hasattr(p, 'mass')` |
| `getattr(obj, 'attr')` | Get attribute safely | `getattr(p, 'mass', 0)` |
| `setattr(obj, 'attr', val)` | Set attribute | `setattr(p, 'mass', 1.0)` |
| `dir(obj)` | List all accessible attributes | `dir(particle)` |
| `vars(obj)` or `obj.__dict__` | Get instance attributes only | `vars(particle)` |
| `help(obj)` | Get documentation | `help(Particle)` |

## Next Chapter Preview

In Chapter 7: NumPy Fundamentals, you'll see how the OOP concepts you just learned power the foundation of scientific Python. NumPy arrays aren't just data containers—they're sophisticated objects with methods like `.reshape()`, `.mean()`, and `.dot()`. You'll discover how NumPy combines the intuitive OOP interface you now understand with blazing-fast C implementations, achieving the best of both worlds. We'll explore array creation, indexing, broadcasting, and vectorization—concepts that eliminate explicit loops and make calculations orders of magnitude faster. Most importantly, you'll see how NumPy's object-oriented design enables the entire scientific Python ecosystem, from plotting with Matplotlib to machine learning with scikit-learn. The objects you just learned to create? They're the same pattern that processes terabytes of astronomical data and simulates the universe's evolution!

## References

1. **Mars Exploration Rover Spirit Recovery (2004)**
   - NASA JPL. (2004). *Mars Exploration Rover Mission: Spirit Anomaly Report*. Jet Propulsion Laboratory.
   - Reeves, G., & Neilson, T. (2005). "The Mars Rover Spirit FLASH Anomaly." *IEEE Aerospace Conference Proceedings*.

2. **SIMULA and OOP Origins (1960s)**
   - Nygaard, K., & Dahl, O. J. (1978). "The development of the SIMULA languages." *ACM SIGPLAN Notices*, 13(8), 245-272.
   - Holmevik, J. R. (1994). "Compiling SIMULA: A Historical Study of Technological Genesis." *IEEE Annals of the History of Computing*, 16(4), 25-37.

3. **Therac-25 Radiation Accidents (1985-1987)**
   - Leveson, N.G. & Turner, C.S. (1993). "An Investigation of the Therac-25 Accidents." *IEEE Computer*, 26(7), 18-41.

4. **Hubble Space Telescope Mirror Error (1990)**
   - Allen, L. (1990). *The Hubble Space Telescope Optical Systems Failure Report*. NASA-TM-103443.
   - Chaisson, E. (1994). *The Hubble Wars*. New York: HarperCollins. ISBN 0-06-017114-6.

5. **Mars Climate Orbiter Loss (1999)**
   - Stephenson, A. G. et al. (1999). *Mars Climate Orbiter Mishap Investigation Board Report*. NASA.
   - Oberg, J. (1999). "Why the Mars Probe Went Off Course." *IEEE Spectrum*, 36(12), 34-39.

6. **NumPy Design Decisions (2005)**
   - Oliphant, T. E. (2006). *A guide to NumPy* (Vol. 1). USA: Trelgol Publishing.
   - Van Der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). "The NumPy array: a structure for efficient numerical computation." *Computing in Science & Engineering*, 13(2), 22-30.

7. **Astropy Unification (2011-2013)**
   - Robitaille, T., et al. (2013). "Astropy: A community Python package for astronomy." *Astronomy & Astrophysics*, 558, A33.
   - Greenfield, P. (2011). "What Python Can Do for Astronomy." *Proceedings of the 20th Annual Python in Science Conference*.

8. **Python Language Design**
   - Van Rossum, G. (1996). Foreword for "Programming Python" (1st ed.). O'Reilly Media.
   - Van Rossum, G., & Drake, F. L. (2009). *Python 3 Reference Manual*. CreateSpace.
   - Martelli, A. (2000). "The Python 'Duck Typing' Principle." comp.lang.python newsgroup archives.

9. **Object-Oriented Design Principles**
   - Kay, A. (1993). "The early history of Smalltalk." *ACM SIGPLAN Notices*, 28(3), 69-95.
   - Kay, A. (2003). Email correspondence to Stefan Ram on the definition of object-oriented programming.

10. **Python OOP Resources**
    - Lutz, M. (2013). *Learning Python* (5th ed.). O'Reilly Media.
    - Ramalho, L. (2015). *Fluent Python*. O'Reilly Media.
    - Beazley, D., & Jones, B. K. (2013). *Python Cookbook* (3rd ed.). O'Reilly Media.