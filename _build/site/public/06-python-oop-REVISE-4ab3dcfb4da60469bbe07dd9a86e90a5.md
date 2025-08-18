# Chapter 6: Object-Oriented Programming - Organizing Scientific Code

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement classes that model astronomical objects and scientific concepts
- Create methods that operate on object data and properties that compute derived values
- Apply inheritance and composition to build hierarchies of related astronomical objects
- Implement special methods to make your objects behave like built-in Python types
- Debug common OOP-related errors using introspection tools
- Write effective tests for your astronomical classes
- Recognize OOP patterns in NumPy, Astropy, and other scientific libraries
- Choose between OOP, functional, and procedural approaches based on problem requirements

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Define and use functions with various parameter types (Chapter 5)
- ✓ Understand scope and namespaces (Chapter 5)
- ✓ Work with dictionaries and their methods (Chapter 4)
- ✓ Create and import modules (Chapter 5)
- ✓ Handle mutable vs immutable objects (Chapter 4)

Quick diagnostic:
```{code-cell} python
# Can you predict what this prints?
def modify(data):
    data['key'] = 'modified'
    return data

original = {'key': 'original'}
result = modify(original)
print(original['key'])  # What value?
```

If you said "modified", you're ready! Objects work similarly - they're mutable and passed by reference.

## Chapter Overview

So far, we've organized code using functions and modules. But what happens when you need to model complex astronomical systems where data and the operations on that data are intimately connected? This is where Object-Oriented Programming (OOP) transforms your code from a collection of functions into a model of the universe itself.

Consider tracking a galaxy catalog. Each galaxy has properties (position, redshift, luminosity, morphology) and behaviors (calculate distance, determine stellar mass, check observability). With functions alone, you'd pass galaxy data between dozens of functions, hoping you don't mix up which data belongs to NGC 1365 versus M87. With OOP, each galaxy is an object that knows its own data and what it can do. This organizational principle scales from simple data containers to complex simulations with thousands of interacting components.

This chapter teaches you to think in objects — not as a dogmatic paradigm, but as a powerful tool for organizing scientific code. You'll learn when OOP makes code clearer (modeling physical objects, managing complex state) and when it adds unnecessary complexity (simple calculations, functional transformations). By the end, you'll understand why NumPy arrays are objects with methods, how Astropy's SkyCoord manages coordinates, and when to create your own classes versus using simpler approaches.

## 6.1 Classes and Objects: The Fundamentals

A class is a blueprint for creating objects. Think of it like the specification for a telescope — the class defines what properties the telescope will have (aperture, focal length, mount type) and what it can do (slew to coordinates, take exposures, auto-guide). An object is a specific instance created from that blueprint — an actual telescope built from those specifications.

Before we dive into creating classes, let's clarify terminology. When we organize code with classes, we use specific terms for familiar concepts:

**Attributes** are variables that belong to an object. Just like variables store data, attributes store an object's data. The only difference is that attributes are attached to a specific object.

**Methods** are functions that belong to an object. Just like functions perform operations, methods perform operations — but they have access to the object's attributes and can operate on the object's data.

Let's see this connection explicitly. You're about to transform from writing scripts to architecting software — the same transformation that took astronomy from simple data reduction to sophisticated pipelines like those processing JWST data:

```{code-cell} python
# You already know variables and functions:
temperature = 25.0  # Variable
def convert_to_fahrenheit(celsius):  # Function
    return celsius * 9/5 + 32

# In OOP, these become attributes and methods:
class Thermometer:
    def __init__(self):
        self.temperature = 25.0  # Attribute (variable attached to object)
    
    def convert_to_fahrenheit(self):  # Method (function attached to object)
        return self.temperature * 9/5 + 32

# The key difference: attributes and methods are organized together
therm = Thermometer()
print(therm.temperature)  # Access attribute through object
print(therm.convert_to_fahrenheit())  # Call method through object
```

This organization is powerful because related data and operations stay together. The thermometer object knows its own temperature and how to convert it — everything about temperature management is in one place.

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

### 🌟 **Why This Matters: Real Astronomy Software Architecture**

```
REAL-WORLD CONNECTION: Astropy's Design

Astropy, the core astronomy Python package, is built entirely 
on OOP principles. Every major component is a class:

- SkyCoord: Manages celestial coordinates
  Attributes: ra, dec, frame, distance
  Methods: transform_to(), separation(), match_to_catalog()

- Time: Handles astronomical time systems
  Attributes: jd, mjd, iso, scale
  Methods: to_datetime(), sidereal_time()

- Quantity: Numbers with units
  Attributes: value, unit
  Methods: to(), decompose(), si

When you write coord = SkyCoord(ra=10.5*u.deg, dec=41.2*u.deg),
you're creating an object that knows its position AND how to
transform between coordinate systems, calculate separations,
and precess to different epochs.

This is why OOP matters: it's how professional astronomy
software organizes complex data and operations together.
```

### Your First Class

Let's start with the simplest possible class and build up progressively:

```{code-cell} python
# Step 1: Basic class definition
class Star:
    """A simple star class."""
    
    def __init__(self, name, magnitude):
        """Initialize a new Star object."""
        self.name = name
        self.magnitude = magnitude

# Create an instance (object) of the Star class
sirius = Star("Sirius", -1.46)
print(f"{sirius.name} has magnitude {sirius.magnitude}")
```

🎉 **Congratulations! You just created a blueprint that could model any star in the universe!** From red dwarfs to blue supergiants, from Proxima Centauri to Betelgeuse, this simple class structure can represent them all. You're no longer just writing code — you're building computational models of astronomical objects. This is exactly how professional packages like Astropy started!

Now let's add methods to give our star behavior:

```{code-cell} python
# Step 2: Add methods for behavior
class Star:
    """A star with observable properties."""
    
    def __init__(self, name, magnitude, distance_pc):
        self.name = name
        self.magnitude = magnitude
        self.distance_pc = distance_pc
    
    def absolute_magnitude(self):
        """Calculate absolute magnitude."""
        import math
        return self.magnitude - 5 * math.log10(self.distance_pc) + 5
    
    def luminosity_solar(self):
        """Calculate luminosity relative to the Sun."""
        abs_mag = self.absolute_magnitude()
        return 10**((4.83 - abs_mag) / 2.5)
```

Finally, let's use our complete star class:

```{code-cell} python
# Step 3: Using the complete class
proxima = Star("Proxima Centauri", 11.13, 1.301)

# Call methods on the object
abs_mag = proxima.absolute_magnitude()
print(f"Absolute magnitude: {abs_mag:.2f}")

luminosity = proxima.luminosity_solar()
print(f"Luminosity: {luminosity:.4f} solar luminosities")
```

Let's visualize what happens when we create an object:

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

The `self` parameter is crucial but often confusing. When you call `sirius = Star("Sirius", -1.46)`, Python essentially does:
1. Creates a new empty object
2. Calls `Star.__init__(new_object, "Sirius", -1.46)`
3. Returns the initialized object

The `self` parameter is how each object keeps track of its own data.

### ⚠️ **Common Bug Alert: Missing self Parameter**

```{code-cell} python
# WRONG - Forgetting self in method definition
class BadClass:
    def method():  # Missing self!
        return "something"

# This will fail:
# obj = BadClass()
# obj.method()  # TypeError: takes 0 positional arguments but 1 was given

# CORRECT - Always include self as first parameter
class GoodClass:
    def method(self):  # self is required
        return "something"

# Why? Python automatically passes the object as the first argument
# obj.method() is actually like: GoodClass.method(obj)
```

This is probably the most common OOP error for beginners. Remember: instance methods ALWAYS need `self` as their first parameter, even if they don't use it.

### Instance vs Class Members

One of the most important distinctions in OOP is between instance members (belonging to specific objects) and class members (shared by all objects of that class):

```{code-cell} python
class Satellite:
    # Class attributes - shared by all satellites
    total_satellites = 0
    EARTH_RADIUS_KM = 6371
    
    def __init__(self, name, altitude_km):
        # Instance attributes - each satellite has its own
        self.name = name
        self.altitude = altitude_km
        # Increment the shared counter
        Satellite.total_satellites += 1
    
    def calculate_period(self):
        # Instance method - uses this satellite's altitude
        import math
        total_radius = self.EARTH_RADIUS_KM + self.altitude
        return 2 * math.pi * math.sqrt(total_radius**3 / 398600)
    
    @classmethod
    def get_satellite_count(cls):
        # Class method - operates on class, not instance
        return cls.total_satellites

# Each satellite has independent instance attributes
iss = Satellite("ISS", 408)
hubble = Satellite("Hubble", 547)

print(f"ISS period: {iss.calculate_period():.1f} minutes")
print(f"Total satellites: {Satellite.total_satellites}")
```

Performance tip: Accessing instance attributes is fast (one dictionary lookup), but be aware that each object stores its own copy. For large numbers of objects, consider using `__slots__` (covered in section 6.8).

### 🔍 **Check Your Understanding**

What's the difference between these two approaches?

```{code-cell} python
# Approach 1: Functions and data separate
star_data = {'name': 'Vega', 'magnitude': 0.03}
def get_brightness(data):
    return 10**(-data['magnitude']/2.5)

# Approach 2: Class with methods
class Star:
    def __init__(self, name, magnitude):
        self.name = name
        self.magnitude = magnitude
    
    def get_brightness(self):
        return 10**(-self.magnitude/2.5)
```

<details>
<summary>Answer</summary>

Both calculate the same value, but they organize code differently. In Approach 1, data and functions are separate — you must remember to pass the right dictionary to the function, and there's no guarantee the dictionary has the right keys. In Approach 2, the data and method are bundled together. The object "knows" its own magnitude, and the method is guaranteed to work with the object's data.

The OOP approach becomes more valuable as complexity grows. Imagine tracking 1000 stars with 20 different calculations — keeping track of which data goes with which function becomes error-prone. With objects, each star manages its own data and knows what operations it can perform.

</details>

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

Astronomical applications:
- Telescope objects tracking position, filters, exposure settings
- Observation objects accumulating photons over time
- Orbit objects evolving position and velocity
- Pipeline objects maintaining reduction state

This pattern appears throughout scientific Python:
- NumPy arrays remember shape, dtype, data
- Matplotlib figures track all plot elements
- SciPy optimizers maintain convergence history
- Astropy Tables track columns, metadata, units
```

## 6.2 Properties and Encapsulation

Properties let you compute attributes dynamically and control access to object data. They're one of Python's most elegant features, allowing you to write code that looks like simple attribute access but actually runs methods behind the scenes.

### Computed Properties with @property

Sometimes an object's attribute should be calculated from other attributes rather than stored separately. Properties make this transparent:

```{code-cell} python
class Detector:
    """CCD detector with computed properties."""
    
    def __init__(self, width_pixels, height_pixels, pixel_size_um):
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.pixel_size_um = pixel_size_um
    
    @property
    def total_pixels(self):
        """Total pixel count."""
        return self.width_pixels * self.height_pixels
    
    @property
    def area_mm2(self):
        """Sensor area in square millimeters."""
        width_mm = self.width_pixels * self.pixel_size_um / 1000
        height_mm = self.height_pixels * self.pixel_size_um / 1000
        return width_mm * height_mm
    
    @property
    def diagonal_mm(self):
        """Sensor diagonal in millimeters."""
        width_mm = self.width_pixels * self.pixel_size_um / 1000
        height_mm = self.height_pixels * self.pixel_size_um / 1000
        return (width_mm**2 + height_mm**2)**0.5

# Properties look like attributes but are computed
ccd = Detector(4096, 4096, 15)
print(f"Total pixels: {ccd.total_pixels:,}")
print(f"Sensor area: {ccd.area_mm2:.1f} mm²")
print(f"Diagonal: {ccd.diagonal_mm:.1f} mm")
```

Properties ensure data consistency. If total_pixels were a regular attribute, you'd have to remember to update it every time width or height changed. With properties, it's always correct because it's calculated on demand.

### Setters and Validation

Properties can also validate data when it's set, preventing invalid states. This is crucial for scientific code where physical constraints must be respected:

```{code-cell} python
class Filter:
    """Astronomical filter with wavelength validation."""
    
    def __init__(self, name, central_wavelength_nm):
        self.name = name
        self._wavelength = central_wavelength_nm  # Note: underscore for internal
    
    @property
    def wavelength(self):
        """Central wavelength in nanometers."""
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        """Set wavelength with validation."""
        if value < 100:  # Below UV
            raise ValueError(f"Wavelength {value} nm below UV range")
        if value > 30000:  # Beyond far-IR
            raise ValueError(f"Wavelength {value} nm beyond far-IR")
        self._wavelength = value
    
    @property
    def wavelength_angstrom(self):
        """Wavelength in Angstroms."""
        return self._wavelength * 10
    
    @property
    def frequency_hz(self):
        """Frequency in Hz."""
        c = 2.998e17  # Speed of light in nm/s
        return c / self._wavelength

# Validation prevents invalid states
v_filter = Filter("V", 550)
print(f"V band: {v_filter.wavelength} nm = {v_filter.wavelength_angstrom} Å")

# This would raise an error:
# v_filter.wavelength = 50  # ValueError: below UV range
```

The underscore prefix (`_wavelength`) is Python's convention for "internal" attributes. It signals to users "don't access this directly, use the property instead."

### 🌟 **Why This Matters: Preventing Spacecraft Disasters with Properties**

```
REAL-WORLD CONNECTION: Unit Safety in Astropy

Remember the Mars Climate Orbiter? It crashed because one team used 
metric units while another used imperial. Properties prevent such 
disasters in modern astronomy software.

Astropy's Quantity class uses properties to ensure unit consistency:

    >>> distance = 10 * u.parsec
    >>> distance.to(u.lightyear)  # Property converts safely
    <Quantity 32.6156 lyr>
    
    >>> velocity = 200 * u.km/u.s
    >>> velocity.si  # Property returns SI units
    <Quantity 200000. m/s>

Properties validate and convert units automatically:
- Setting invalid units raises an error immediately
- Conversions are handled consistently
- Unit metadata travels with the data

Every major space mission now uses similar property-based 
validation. The James Webb Space Telescope's command software 
validates every parameter through properties before uplink.

You're learning the same defensive programming that keeps 
billion-dollar missions safe!
```

### ⚠️ **Common Bug Alert: Property Recursion**

```{code-cell} python
# WRONG - Infinite recursion!
class BadClass:
    @property
    def value(self):
        return self.value  # Calls itself forever!

# CORRECT - Use different internal name
class GoodClass:
    @property
    def value(self):
        return self._value  # Different name

# Always use a different name (usually with underscore) for storage
```

## 6.3 Inheritance: Building on Existing Classes

Inheritance lets you create new classes based on existing ones, inheriting their attributes and methods while adding or modifying functionality. This models "is-a" relationships that are everywhere in astronomy: a WhiteDwarf is-a Star, a GlobularCluster is-a StellarPopulation, a SpacetelEscope is-a Telescope.

### 🌟 **Why This Matters: Chandra's Software Architecture**

```
REAL-WORLD CONNECTION: X-ray Observatory Software Hierarchy

The Chandra X-ray Observatory's data processing pipeline uses 
deep inheritance hierarchies to model its complex instruments:

ChardraDetector (base class)
├── ACIS (Advanced CCD Imaging Spectrometer)
│   ├── ACIS_I (Imaging array)
│   └── ACIS_S (Spectroscopy array)
└── HRC (High Resolution Camera)
    ├── HRC_I (Imaging detector)
    └── HRC_S (Spectroscopy detector)

Each detector inherits common methods:
- read_event_list()
- apply_calibration()
- detect_cosmic_rays()

But each subclass overrides specific behaviors:
- ACIS handles charge transfer inefficiency
- HRC handles timing with microsecond precision
- ACIS_S adds grating spectroscopy methods

This inheritance structure allows the pipeline to process
data from ANY detector using the same interface, while each
detector type handles its unique characteristics.

When astronomers write:
    detector.process_observation(data)

The correct processing happens automatically based on the
detector type - polymorphism in action! This design pattern
is why Chandra can still process new observation modes 25
years after launch.
```

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
        +spectral_class()
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
    
    CelestialBody <|-- Star : inherits
    CelestialBody <|-- Planet : inherits
    Star <|-- WhiteDwarf : inherits
    Planet <|-- GasGiant : inherits
    
    note for CelestialBody "Base class with common properties"
    note for Star "Adds stellar-specific features"
```

### 🌟 **Why This Matters: Building Astronomical Software**

```
REAL-WORLD CONNECTION: Inheritance in Action

Major astronomy packages use inheritance extensively:

Astropy coordinates:
- BaseCoordinateFrame (base)
  └── ICRS (International Celestial Reference System)
  └── Galactic (Galactic coordinates)
  └── AltAz (Horizontal coordinates)

Each frame inherits transformation methods but implements
its own specific coordinate system.

Photutils (photometry package):
- Aperture (base)
  └── CircularAperture
  └── EllipticalAperture
  └── RectangularAperture

All apertures inherit area calculation and plotting,
but each implements its own geometry.

This design lets you write code that works with ANY
coordinate system or ANY aperture shape!
```

### Basic Inheritance - Progressive Build

Let's build an inheritance hierarchy step by step:

```{code-cell} python
# Step 1: Define the base class
class CelestialBody:
    """Base class for astronomical objects."""
    
    def __init__(self, name, mass, radius):
        self.name = name
        self.mass = mass      # kg
        self.radius = radius  # meters
    
    def surface_gravity(self):
        """Calculate surface gravity in m/s²."""
        G = 6.674e-11  # SI units
        return G * self.mass / self.radius**2
    
    def escape_velocity(self):
        """Calculate escape velocity in m/s."""
        import math
        G = 6.674e-11
        return math.sqrt(2 * G * self.mass / self.radius)
```

```{code-cell} python
# Step 2: Create a derived class that extends the base
class Planet(CelestialBody):
    """A planet with additional properties."""
    
    def __init__(self, name, mass, radius, orbital_period, moons=0):
        # Call parent class constructor
        super().__init__(name, mass, radius)
        # Add planet-specific attributes
        self.orbital_period = orbital_period  # days
        self.moons = moons
    
    def day_length_hours(self):
        """Estimate day length (simplified)."""
        # This is a simplification for demonstration
        return 24 * (self.radius / 6.371e6)**0.5
```

```{code-cell} python
# Step 3: Use the inheritance hierarchy
earth = Planet("Earth", 5.972e24, 6.371e6, 365.25, moons=1)

# Planet inherits methods from CelestialBody
print(f"Surface gravity: {earth.surface_gravity():.1f} m/s²")
print(f"Escape velocity: {earth.escape_velocity()/1000:.1f} km/s")

# And has its own methods
print(f"Estimated day length: {earth.day_length_hours():.1f} hours")
print(f"Number of moons: {earth.moons}")
```

### Understanding super()

The `super()` function is crucial for inheritance. It ensures parent classes are properly initialized and is essential for complex inheritance hierarchies:

```{code-cell} python
class Star(CelestialBody):
    """A star with luminosity."""
    
    def __init__(self, name, mass, radius, temperature):
        super().__init__(name, mass, radius)  # Initialize parent
        self.temperature = temperature  # Kelvin
    
    def luminosity(self):
        """Calculate luminosity using Stefan-Boltzmann law."""
        import math
        sigma = 5.670e-8  # Stefan-Boltzmann constant
        return 4 * math.pi * self.radius**2 * sigma * self.temperature**4
    
    def spectral_class(self):
        """Determine spectral class from temperature."""
        if self.temperature > 30000: return 'O'
        elif self.temperature > 10000: return 'B'
        elif self.temperature > 7500: return 'A'
        elif self.temperature > 6000: return 'F'
        elif self.temperature > 5200: return 'G'
        elif self.temperature > 3700: return 'K'
        else: return 'M'

sun = Star("Sun", 1.989e30, 6.96e8, 5778)
print(f"Sun's spectral class: {sun.spectral_class()}")
print(f"Luminosity: {sun.luminosity():.2e} watts")
```

### Method Overriding

Child classes can override parent methods to provide specialized behavior:

```{code-cell} python
class WhiteDwarf(Star):
    """A white dwarf star with special density calculation."""
    
    def __init__(self, name, mass, radius, temperature, cooling_age_gyr):
        super().__init__(name, mass, radius, temperature)
        self.cooling_age = cooling_age_gyr  # Billion years
    
    def density(self):
        """Override density to add extreme density warning."""
        import math
        volume = 4/3 * math.pi * self.radius**3
        density = self.mass / volume
        
        # White dwarfs have extreme densities
        if density > 1e9:  # kg/m³
            print(f"⚠️ Extreme density: {density:.2e} kg/m³")
            print(f"   (A teaspoon would weigh {density*5e-6:.0f} kg!)")
        
        return density
    
    def crystallization_fraction(self):
        """Estimate crystallized fraction based on cooling age."""
        # Simplified model for demonstration
        if self.cooling_age < 1:
            return 0.0
        elif self.cooling_age > 10:
            return 0.9
        else:
            return self.cooling_age / 11

sirius_b = WhiteDwarf("Sirius B", 2.02e30, 5.8e6, 25000, 0.12)
density = sirius_b.density()
```

### 🔍 **Check Your Understanding**

What's the output of this code? Why?

```{code-cell} python
class A:
    def method(self): return "A"

class B(A):
    def method(self): return "B"

class C(A):
    def method(self): return "C"

class D(B, C):
    pass

obj = D()
print(obj.method())
print(D.__mro__)  # Method Resolution Order
```

<details>
<summary>Answer</summary>

Output is "B" because Python uses the Method Resolution Order (MRO) to determine which method to call. The MRO for class D is: D → B → C → A → object. Since D doesn't have its own method(), Python looks in B next and finds it there. The MRO ensures each class appears only once and respects the inheritance order you specified (B before C).

</details>

### Composition vs Inheritance

Sometimes "has-a" relationships (composition) are better than "is-a" relationships (inheritance):

```{code-cell} python
class Orbit:
    """Orbital parameters (composition approach)."""
    
    def __init__(self, semi_major_axis_au, eccentricity, period_years):
        self.a = semi_major_axis_au  # Astronomical units
        self.e = eccentricity
        self.period = period_years
    
    def perihelion(self):
        """Closest approach distance."""
        return self.a * (1 - self.e)
    
    def aphelion(self):
        """Farthest distance."""
        return self.a * (1 + self.e)

class Comet:
    """Comet with orbital information (has-a orbit)."""
    
    def __init__(self, name, orbit, magnitude):
        self.name = name
        self.orbit = orbit  # Composition: Comet HAS an Orbit
        self.magnitude = magnitude
    
    def is_visible(self):
        """Check if visible to naked eye."""
        return self.magnitude < 6

# Create comet with orbit
halley_orbit = Orbit(17.8, 0.967, 75.3)
halley = Comet("Halley", halley_orbit, 4.5)

print(f"{halley.name}'s perihelion: {halley.orbit.perihelion():.2f} AU")
print(f"Visible to naked eye: {halley.is_visible()}")
```

Use inheritance when objects share an "is-a" relationship (WhiteDwarf is-a Star). Use composition when objects have a "has-a" relationship (Comet has-a Orbit).

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
    
    style A fill:#f9f
    style B fill:#9ff
```

### String Representation and Arithmetic

```{code-cell} python
class Vector3D:
    """A 3D vector for astronomical coordinates."""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        """Human-readable string for print()."""
        return f"Vector({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def __repr__(self):
        """Developer-friendly representation."""
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
    
    def __add__(self, other):
        """Vector addition with + operator."""
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __abs__(self):
        """Magnitude with abs()."""
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    def __eq__(self, other):
        """Equality comparison with ==."""
        return (self.x == other.x and 
                self.y == other.y and 
                self.z == other.z)

# Using special methods
v1 = Vector3D(3, 4, 0)
v2 = Vector3D(1, 0, 0)

print(v1)  # Calls __str__
print(f"Magnitude: {abs(v1)}")  # Calls __abs__

v3 = v1 + v2  # Calls __add__
print(f"Sum: {v3}")
```

### Container Behavior

Make your objects act like containers:

```{code-cell} python
class ObservationLog:
    """A log of observations that acts like a list."""
    
    def __init__(self):
        self._observations = []
    
    def add(self, target, time, conditions):
        self._observations.append({
            'target': target,
            'time': time,
            'conditions': conditions
        })
    
    def __len__(self):
        """Number of observations."""
        return len(self._observations)
    
    def __getitem__(self, index):
        """Access with square brackets."""
        return self._observations[index]
    
    def __iter__(self):
        """Make iterable with for loops."""
        return iter(self._observations)
    
    def __contains__(self, target):
        """Support 'in' operator."""
        return any(obs['target'] == target for obs in self._observations)

# Use like a container
log = ObservationLog()
log.add("M31", "2024-01-15 22:30", "Clear")
log.add("M42", "2024-01-15 23:15", "Partly cloudy")

print(f"Total observations: {len(log)}")
print(f"First observation: {log[0]['target']}")
print(f"M31 in log: {'M31' in log}")

for obs in log:
    print(f"  {obs['target']} at {obs['time']}")
```

### 🔍 **Check Your Understanding**

If you implement `__eq__` but not `__hash__`, can your objects be dictionary keys? Why or why not?

<details>
<summary>Answer</summary>

No, they cannot be dictionary keys or added to sets! When you override `__eq__`, Python automatically sets `__hash__` to None to prevent a subtle bug. Here's why:

```python
class BadStar:
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return self.name == other.name
    # No __hash__ defined!

star1 = BadStar("Vega")
star2 = BadStar("Vega")

# These are equal according to __eq__
print(star1 == star2)  # True

# But can't be used as dictionary keys:
# catalog = {star1: "bright"}  # TypeError: unhashable type: 'BadStar'
```

The rule is: objects that are equal (according to `__eq__`) must have the same hash value. If you could use objects with only `__eq__` as dictionary keys, you could have two "equal" keys with different hashes, breaking the dictionary!

To fix it, implement both together:
```python
class GoodStar:
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)  # Hash based on equality criteria

# Now it works!
catalog = {GoodStar("Vega"): "bright"}
```

This is part of Python's "protocol" system - certain methods must work together. You're learning the same patterns that make NumPy arrays and Astropy coordinates work seamlessly with Python's built-in functions!

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

Astronomical example:
Any object with ra, dec, and transform_to() can be used
as a coordinate, regardless of its actual class. This is
why Astropy coordinates are so flexible - they follow
protocols rather than rigid inheritance.

This protocol approach is key to Python's flexibility and
why your custom classes can seamlessly integrate with
built-in functions and scientific libraries.
```

## 6.5 Context Managers

Context managers are objects that define setup and cleanup actions for `with` statements. They're crucial for resource management in scientific computing:

```{code-cell} python
# Step 1: Basic context manager structure
class TelescopeConnection:
    """Manages telescope connection lifecycle."""
    
    def __init__(self, telescope_id):
        self.telescope_id = telescope_id
        self.connected = False
    
    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"Connecting to {self.telescope_id}...")
        self.connected = True
        return self  # Return self for use with 'as'
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Called when leaving 'with' block."""
        print(f"Disconnecting from {self.telescope_id}")
        self.connected = False
        return False  # Don't suppress exceptions

# Usage ensures cleanup even if errors occur
with TelescopeConnection("Keck-1") as telescope:
    print(f"Connected: {telescope.connected}")
    # Connection automatically closed after this block
```

Now let's build a more complete example for data files:

```{code-cell} python
# Step 2: Complete context manager with error handling
class FITSReader:
    """Context manager for reading FITS files."""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.header = {}
        self.data = None
    
    def __enter__(self):
        """Open and read FITS file."""
        print(f"Opening {self.filename}")
        # Simulate file operations
        self.file = f"Handle for {self.filename}"
        self.header = {'TELESCOP': 'HST', 'EXPTIME': 1200}
        self.data = [1, 2, 3, 4]  # Simplified
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up resources."""
        if self.file:
            print(f"Closing {self.filename}")
            self.file = None
        
        # Handle specific exceptions
        if exc_type is ValueError:
            print(f"Data error in {self.filename}: {exc_value}")
            return True  # Suppress this exception
        
        return False  # Let other exceptions propagate
```

```{code-cell} python
# Step 3: Using the context manager
with FITSReader("observation.fits") as fits:
    print(f"Telescope: {fits.header['TELESCOP']}")
    print(f"Exposure: {fits.header['EXPTIME']}s")
    # File automatically closed even if an error occurs

print("File has been closed automatically")
```

Context managers ensure resources are properly released even when errors occur - essential for telescope connections, file handles, and database connections.

## 6.6 Debugging and Testing Classes

Understanding how to debug and test objects is crucial for reliable scientific software.

### Introspection Tools

Python provides powerful tools for examining objects:

```{code-cell} python
class Instrument:
    """Example instrument for debugging demonstration."""
    
    class_var = "shared"
    
    def __init__(self, name, wavelength):
        self.name = name
        self.wavelength = wavelength
        self._calibrated = False
    
    def calibrate(self):
        self._calibrated = True

# Create an instrument
spectrograph = Instrument("HIRES", 500)

# Introspection functions
print(f"Type: {type(spectrograph)}")
print(f"ID: {id(spectrograph)}")
print(f"Has 'calibrate': {hasattr(spectrograph, 'calibrate')}")

# List all attributes
attrs = [a for a in dir(spectrograph) if not a.startswith('__')]
print(f"Attributes: {attrs}")

# Get instance dictionary
print(f"Instance dict: {vars(spectrograph)}")
```

### ⚠️ **Common Bug Alert: Mutable Default Arguments in Classes**

```{code-cell} python
# DANGEROUS - Mutable default argument
class BadObservatory:
    def __init__(self, name, telescopes=[]):  # BAD!
        self.name = name
        self.telescopes = telescopes  # All instances share same list!

# CORRECT - Use None and create new list
class GoodObservatory:
    def __init__(self, name, telescopes=None):
        self.name = name
        self.telescopes = telescopes if telescopes is not None else []

# The bug in action:
obs1 = BadObservatory("Mauna Kea")
obs1.telescopes.append("Keck I")
obs2 = BadObservatory("La Silla")
print(f"La Silla telescopes: {obs2.telescopes}")  # Has Keck I!
```

### Testing Classes with unittest

```{code-cell} python
import unittest

class TestVector3D(unittest.TestCase):
    """Test cases for Vector3D class."""
    
    def setUp(self):
        """Create test vectors before each test."""
        self.v1 = Vector3D(3, 4, 0)
        self.v2 = Vector3D(1, 0, 0)
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        self.assertEqual(abs(self.v1), 5.0)
        self.assertEqual(abs(self.v2), 1.0)
    
    def test_addition(self):
        """Test vector addition."""
        v3 = self.v1 + self.v2
        self.assertEqual(v3.x, 4)
        self.assertEqual(v3.y, 4)
        self.assertEqual(v3.z, 0)
    
    def test_equality(self):
        """Test equality comparison."""
        v_copy = Vector3D(3, 4, 0)
        self.assertEqual(self.v1, v_copy)
        self.assertNotEqual(self.v1, self.v2)

# Run tests (in Jupyter, use this approach)
suite = unittest.TestLoader().loadTestsFromTestCase(TestVector3D)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
```

## 6.7 When to Use OOP

OOP isn't always the answer. Here's how to decide:

```{mermaid}
flowchart TD
    A[Design Decision] --> B{Maintains state<br/>over time?}
    B -->|Yes| C{Multiple operations<br/>on same data?}
    B -->|No| D[Use Functions]
    C -->|Yes| E[Use Classes]
    C -->|No| F[Consider Functions]
    
    E --> G[Examples:<br/>Telescope<br/>Observation<br/>Pipeline]
    D --> H[Examples:<br/>unit_conversion<br/>calculate_airmass]
    
    style E fill:#9f9
    style D fill:#9ff
```

### When OOP Shines

Use classes for:

1. **Objects with state that evolves:**
```{code-cell} python
class Observation:
    """Tracks observation state over time."""
    
    def __init__(self, target):
        self.target = target
        self.start_time = None
        self.photons = []
        self.total_exposure = 0
    
    def start_exposure(self, time):
        self.start_time = time
    
    def add_photons(self, count):
        self.photons.append(count)
        self.total_exposure += 1
```

2. **Complex data with associated operations:**
```{code-cell} python
class Spectrum:
    """Spectrum with analysis methods."""
    
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux
    
    def find_lines(self, threshold):
        """Find emission lines."""
        # Complex analysis
        pass
    
    def normalize(self):
        """Normalize flux."""
        # Modifies internal state
        pass
```

### When to Avoid OOP

Don't use classes for:

1. **Simple calculations:**
```{code-cell} python
# Unnecessary OOP
class Converter:
    def parsec_to_lightyear(self, parsec):
        return parsec * 3.26156

# Better as a simple function
def parsec_to_lightyear(parsec):
    return parsec * 3.26156
```

2. **Collections of unrelated utilities:**
```{code-cell} python
# Better as a module with functions, not a class
# In astro_utils.py:
def airmass(zenith_angle):
    """Calculate airmass."""
    import math
    return 1 / math.cos(math.radians(zenith_angle))

def julian_date(year, month, day):
    """Convert to Julian date."""
    # Calculation here
    pass
```

### 🛠️ **Debug This!**

This class has a subtle bug. Can you find it?

```{code-cell} python
class PhotonCounter:
    def __init__(self, dark_current=0.1, bins=[]):
        self.dark_current = dark_current
        self.bins = bins
    
    def add_count(self, bin_number, count):
        while len(self.bins) <= bin_number:
            self.bins.append(0)
        self.bins[bin_number] += count

# Test code
counter1 = PhotonCounter()
counter1.add_count(0, 100)
counter2 = PhotonCounter()
counter2.add_count(0, 200)
print(f"Counter1 bins: {counter1.bins}")
print(f"Counter2 bins: {counter2.bins}")
```

<details>
<summary>Bug and Solution</summary>

**Bug**: Mutable default argument! Both counters share the same bins list.

Output will be:
```
Counter1 bins: [300]
Counter2 bins: [300]
```

**Solution**: Use None as default and create new list in __init__:
```python
def __init__(self, dark_current=0.1, bins=None):
    self.dark_current = dark_current
    self.bins = bins if bins is not None else []
```

This bug is especially dangerous in scientific code where data integrity is crucial!

</details>

## 6.8 Performance Considerations

Understanding object performance helps you make informed design decisions.

### Memory and Speed Tradeoffs

```{code-cell} python
import sys
import time

# Compare different approaches
class StarObject:
    def __init__(self, name, mag):
        self.name = name
        self.mag = mag

# Object approach
star_obj = StarObject("Vega", 0.03)
print(f"Object size: {sys.getsizeof(star_obj) + sys.getsizeof(star_obj.__dict__)} bytes")

# Dictionary approach
star_dict = {"name": "Vega", "mag": 0.03}
print(f"Dict size: {sys.getsizeof(star_dict)} bytes")

# Tuple approach (most memory efficient)
star_tuple = ("Vega", 0.03)
print(f"Tuple size: {sys.getsizeof(star_tuple)} bytes")

# For many objects, consider NumPy
import numpy as np
star_array = np.array([("Vega", 0.03)], dtype=[('name', 'U10'), ('mag', 'f4')])
print(f"NumPy size per star: {star_array.nbytes} bytes")
```

### Using __slots__ for Memory Efficiency

For classes with many instances (like particles in a simulation), `__slots__` can save significant memory:

```{code-cell} python
class RegularParticle:
    """Normal class - flexible but uses more memory."""
    def __init__(self, x, y, z, vx, vy, vz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

class SlottedParticle:
    """Slotted class - fixed attributes, less memory."""
    __slots__ = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    
    def __init__(self, x, y, z, vx, vy, vz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

# Memory comparison
regular = RegularParticle(1, 2, 3, 0.1, 0.2, 0.3)
slotted = SlottedParticle(1, 2, 3, 0.1, 0.2, 0.3)

print(f"Regular: {sys.getsizeof(regular) + sys.getsizeof(regular.__dict__)} bytes")
print(f"Slotted: {sys.getsizeof(slotted)} bytes")
```

**When to use `__slots__`:**
- Thousands+ of instances (particle simulations, pixel data)
- Fixed, known attributes
- Memory is a constraint

**When to avoid `__slots__`:**
- Need dynamic attributes
- Using multiple inheritance
- Prototyping/design not final

## 6.9 Practice Exercises

### Quick Practice: Create a VariableStar Class

Let's start with a fundamental building block for photometry analysis. This simple class will become the foundation for more complex astronomical software:

```{code-cell} python
"""
Quick Practice: Variable Star Class (5-10 lines)

Create a VariableStar class that:
1. Stores name, period (days), min_mag, and max_mag
2. Has a method to calculate amplitude
3. Implements __str__ for readable output

This is how professional packages like lightkurve start - 
with simple, well-designed base classes!
"""

class VariableStar:
    """A variable star with photometric properties."""
    
    def __init__(self, name, period, min_mag, max_mag):
        self.name = name
        self.period = period
        self.min_mag = min_mag
        self.max_mag = max_mag
    
    def amplitude(self):
        """Calculate peak-to-peak amplitude."""
        return self.max_mag - self.min_mag
    
    def __str__(self):
        return f"{self.name}: P={self.period:.2f}d, Amp={self.amplitude():.2f}mag"

# Test with real Cepheid data
delta_cep = VariableStar("Delta Cephei", 5.366, 3.48, 4.37)
print(delta_cep)
print(f"Amplitude: {delta_cep.amplitude():.2f} magnitudes")

# You've just created the same kind of object used in
# professional variable star databases like AAVSO!
```

### Synthesis Exercise: Variable Star Type Hierarchy

Now let's build a realistic inheritance hierarchy that models different types of variable stars. This demonstrates how professional astronomy software organizes related but distinct objects:

```{code-cell} python
"""
Synthesis: Variable Star Type Hierarchy

Build an inheritance tree modeling real astrophysics:
- Each type has its own physics for absolute magnitude
- Uses super() to maintain parent functionality
- Demonstrates polymorphism in action

This mirrors how packages like PyAstronomy organize stellar types!
"""

import math

class VariableStar:
    """Base class for all variable stars."""
    
    def __init__(self, name, period, apparent_mag, distance_pc):
        self.name = name
        self.period = period  # days
        self.apparent_mag = apparent_mag
        self.distance_pc = distance_pc
    
    def calculate_absolute_magnitude(self):
        """Default distance modulus calculation."""
        return self.apparent_mag - 5 * math.log10(self.distance_pc) + 5
    
    def __str__(self):
        return f"{self.__class__.__name__} {self.name}"

class Cepheid(VariableStar):
    """Classical Cepheid with period-luminosity relation."""
    
    def __init__(self, name, period, apparent_mag, distance_pc, pulsation_mode='fundamental'):
        super().__init__(name, period, apparent_mag, distance_pc)
        self.pulsation_mode = pulsation_mode
    
    def calculate_absolute_magnitude(self):
        """Use Leavitt's period-luminosity relation."""
        # Classical relation: M_V = -2.81 * log(P) - 1.43
        if self.period > 0:
            return -2.81 * math.log10(self.period) - 1.43
        return super().calculate_absolute_magnitude()
    
    def distance_from_pl_relation(self):
        """Calculate distance using period-luminosity relation."""
        abs_mag = self.calculate_absolute_magnitude()
        distance = 10 ** ((self.apparent_mag - abs_mag + 5) / 5)
        return distance

class RRLyrae(VariableStar):
    """RR Lyrae star with metallicity dependence."""
    
    def __init__(self, name, period, apparent_mag, distance_pc, metallicity=-1.5):
        super().__init__(name, period, apparent_mag, distance_pc)
        self.metallicity = metallicity  # [Fe/H]
    
    def calculate_absolute_magnitude(self):
        """RR Lyrae have nearly constant absolute magnitude."""
        # M_V ≈ 0.23 * [Fe/H] + 0.93 (simplified relation)
        return 0.23 * self.metallicity + 0.93
    
    def is_oosterhoff_type_i(self):
        """Classify based on period."""
        return 0.4 < self.period < 0.7

class EclipsingBinary(VariableStar):
    """Eclipsing binary star system."""
    
    def __init__(self, name, period, apparent_mag, distance_pc, 
                 mass_ratio=1.0, inclination=90):
        super().__init__(name, period, apparent_mag, distance_pc)
        self.mass_ratio = mass_ratio  # M2/M1
        self.inclination = inclination  # degrees
    
    def calculate_absolute_magnitude(self):
        """Combined magnitude of both stars."""
        # For equal stars, combined magnitude is 0.75 mag brighter
        if self.mass_ratio > 0.8:
            base_mag = super().calculate_absolute_magnitude()
            return base_mag - 0.75
        return super().calculate_absolute_magnitude()
    
    def is_contact_binary(self):
        """Check if likely W UMa type."""
        return self.period < 1.0 and self.mass_ratio > 0.5

# Test the hierarchy with real variable stars
stars = [
    Cepheid("Polaris", 3.97, 1.98, 132.5),
    RRLyrae("RR Lyrae", 0.567, 7.1, 262, metallicity=-1.2),
    EclipsingBinary("Algol", 2.867, 2.12, 27.5, mass_ratio=0.68)
]

print("Variable Star Catalog:")
for star in stars:
    abs_mag = star.calculate_absolute_magnitude()
    print(f"  {star}: M_V = {abs_mag:.2f}")
    
    # Polymorphism - each type calculates differently!
    if isinstance(star, Cepheid):
        dist = star.distance_from_pl_relation()
        print(f"    P-L distance: {dist:.1f} pc")
    elif isinstance(star, RRLyrae):
        print(f"    Oosterhoff I: {star.is_oosterhoff_type_i()}")
    elif isinstance(star, EclipsingBinary):
        print(f"    Contact binary: {star.is_contact_binary()}")

# This is exactly how professional catalogs like Gaia's
# variable star database organize their data!
```

### Challenge Exercise: Complete Photometry Pipeline

This advanced exercise demonstrates professional-level OOP design, mimicking the structure of packages like `lightkurve` or `astropy.timeseries`:

```{code-cell} python
"""
Challenge: Professional Photometry Pipeline

Design a complete OOP system for time-series photometry:
- Uses composition (LightCurve HAS-A ObservationSet)
- Implements special methods for Pythonic behavior  
- Properties for computed values
- Context managers for file I/O
- Multiple analysis algorithms via strategy pattern

This is how NASA's Kepler/TESS pipelines are structured!
"""

import numpy as np
from typing import List, Optional
import contextlib

class Observation:
    """Single photometric observation."""
    
    def __init__(self, time: float, magnitude: float, error: float):
        self.time = time  # Julian Date
        self.magnitude = magnitude
        self.error = error
    
    def __repr__(self):
        return f"Obs(t={self.time:.2f}, m={self.magnitude:.2f}±{self.error:.3f})"

class ObservationSet:
    """Container for time-series observations."""
    
    def __init__(self):
        self._observations: List[Observation] = []
    
    def add(self, obs: Observation):
        """Add observation, maintaining time order."""
        import bisect
        # Keep sorted by time for efficiency
        times = [o.time for o in self._observations]
        idx = bisect.bisect_left(times, obs.time)
        self._observations.insert(idx, obs)
    
    def __len__(self):
        """Number of observations."""
        return len(self._observations)
    
    def __getitem__(self, idx):
        """Access observations by index."""
        return self._observations[idx]
    
    def __iter__(self):
        """Iterate over observations."""
        return iter(self._observations)
    
    @property
    def times(self):
        """Array of observation times."""
        return np.array([obs.time for obs in self._observations])
    
    @property
    def magnitudes(self):
        """Array of magnitudes."""
        return np.array([obs.magnitude for obs in self._observations])
    
    @property
    def errors(self):
        """Array of measurement errors."""
        return np.array([obs.error for obs in self._observations])

class LightCurve:
    """Light curve with analysis methods."""
    
    def __init__(self, obs_set: ObservationSet, star_name: str = "Unknown"):
        self.observations = obs_set  # Composition: HAS-A ObservationSet
        self.star_name = star_name
        self._period = None  # Cached period
    
    @property
    def mean_magnitude(self):
        """Weighted mean magnitude."""
        if len(self.observations) == 0:
            return np.nan
        weights = 1.0 / self.observations.errors**2
        return np.average(self.observations.magnitudes, weights=weights)
    
    @property
    def amplitude(self):
        """Peak-to-peak amplitude."""
        if len(self.observations) == 0:
            return 0.0
        return np.ptp(self.observations.magnitudes)
    
    def phase_fold(self, period: float):
        """Fold light curve at given period."""
        phases = (self.observations.times % period) / period
        return phases, self.observations.magnitudes
    
    def find_period(self, method='lomb-scargle'):
        """Find best period using specified algorithm."""
        if method == 'lomb-scargle':
            return self._lomb_scargle_period()
        elif method == 'string-length':
            return self._string_length_period()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _lomb_scargle_period(self):
        """Simplified Lomb-Scargle periodogram."""
        # In real code, use astropy.timeseries.LombScargle
        test_periods = np.linspace(0.1, 10, 1000)
        # Simplified - just find period with minimum scatter
        best_period = test_periods[0]
        min_scatter = float('inf')
        
        for period in test_periods:
            phases, mags = self.phase_fold(period)
            scatter = np.std(mags)
            if scatter < min_scatter:
                min_scatter = scatter
                best_period = period
        
        self._period = best_period
        return best_period
    
    def __str__(self):
        return (f"LightCurve({self.star_name}): "
                f"{len(self.observations)} obs, "
                f"<m>={self.mean_magnitude:.2f}, "
                f"Amp={self.amplitude:.2f}")

class PhotometryFileReader:
    """Context manager for reading photometry files."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.light_curve = None
    
    def __enter__(self):
        """Open file and read header."""
        print(f"Opening {self.filename}")
        self.file = open(self.filename, 'r')
        
        # Read star name from first line
        star_name = self.file.readline().strip().replace('#', '').strip()
        
        # Read observations
        obs_set = ObservationSet()
        for line in self.file:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                obs = Observation(
                    float(parts[0]),  # time
                    float(parts[1]),  # magnitude
                    float(parts[2])   # error
                )
                obs_set.add(obs)
        
        self.light_curve = LightCurve(obs_set, star_name)
        return self.light_curve
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if self.file:
            print(f"Closing {self.filename}")
            self.file.close()
        return False  # Don't suppress exceptions

# Demonstration of the complete pipeline
print("=== Professional Photometry Pipeline Demo ===\n")

# Create synthetic variable star data
obs_set = ObservationSet()
true_period = 2.5  # days
times = np.linspace(0, 20, 100)
for t in times:
    # Sinusoidal variation
    phase = 2 * np.pi * t / true_period
    magnitude = 10.0 + 0.5 * np.sin(phase)
    error = 0.01 + 0.02 * np.random.random()
    obs_set.add(Observation(t, magnitude, error))

# Create light curve and analyze
lc = LightCurve(obs_set, "Simulated Cepheid")
print(lc)
print(f"Mean magnitude: {lc.mean_magnitude:.3f}")
print(f"Amplitude: {lc.amplitude:.3f} mag")

# Find period
found_period = lc.find_period(method='lomb-scargle')
print(f"Found period: {found_period:.3f} days (true: {true_period} days)")

# Demonstrate special methods
print(f"\nLight curve has {len(lc.observations)} observations")
print(f"First observation: {lc.observations[0]}")
print(f"Last observation: {lc.observations[-1]}")

# Show how this would work with a file
# with PhotometryFileReader('cepheid_data.txt') as light_curve:
#     period = light_curve.find_period()
#     print(f"Period: {period:.3f} days")

print("\n🎉 Congratulations! You've built a photometry pipeline")
print("    using the same OOP patterns as lightkurve and")
print("    astropy.timeseries. This is professional-level")
print("    astronomical software design!")
```

These exercises demonstrate how OOP transforms astronomical data analysis from scattered functions into organized, reusable systems. You're learning the same design patterns used in packages that analyze data from Kepler, TESS, and ground-based surveys like ASAS-SN and ZTF!

### Exercise 6.1: Design a Measurement Class

Create a class for scientific measurements with uncertainty:

```{code-cell} python
"""
Design a Measurement class with:
1. Store value and uncertainty
2. Arithmetic operations with error propagation
3. String output with significant figures
4. Method to check agreement within uncertainty

Error propagation:
- Addition: δz = sqrt(δx² + δy²)
- Multiplication: δz/z = sqrt((δx/x)² + (δy/y)²)
"""

class Measurement:
    def __init__(self, value, uncertainty):
        self.value = value
        self.uncertainty = uncertainty
    
    def __str__(self):
        # Format with appropriate precision
        return f"{self.value:.2f} ± {self.uncertainty:.2f}"
    
    def __add__(self, other):
        # Implement error propagation for addition
        import math
        new_value = self.value + other.value
        new_unc = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return Measurement(new_value, new_unc)
    
    def agrees_with(self, other):
        # Check if measurements agree within uncertainty
        import math
        diff = abs(self.value - other.value)
        combined_unc = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return diff <= combined_unc

# Test
m1 = Measurement(10.0, 0.1)
m2 = Measurement(10.05, 0.08)
print(f"m1: {m1}")
print(f"m2: {m2}")
print(f"Agreement: {m1.agrees_with(m2)}")
print(f"Sum: {m1 + m2}")
```

### Exercise 6.2: Astronomical Observation Class

```{code-cell} python
"""
Create an observation class hierarchy:
1. Base Observation class
2. OpticalObservation subclass
3. SpectroscopicObservation subclass
"""

from datetime import datetime

class Observation:
    """Base observation class."""
    
    def __init__(self, target, instrument, observer):
        self.target = target
        self.instrument = instrument
        self.observer = observer
        self.timestamp = datetime.now()
    
    def is_valid(self):
        """Check if observation is valid."""
        return all([self.target, self.instrument, self.observer])

class OpticalObservation(Observation):
    """Optical observation with photometry."""
    
    def __init__(self, target, instrument, observer, 
                 filter_name, exposure_time, counts):
        super().__init__(target, instrument, observer)
        self.filter = filter_name
        self.exposure_time = exposure_time
        self.counts = counts
    
    def signal_to_noise(self):
        """Calculate SNR."""
        import math
        return math.sqrt(self.counts)

# Test the hierarchy
obs = OpticalObservation("M42", "CCD", "Student", "V", 30, 10000)
print(f"Valid: {obs.is_valid()}")
print(f"SNR: {obs.signal_to_noise():.1f}")
```

### Exercise 6.3: Performance Comparison

```{code-cell} python
"""
Compare OOP vs procedural for 1000 stars
"""

import time
import numpy as np

# OOP approach
class Star:
    def __init__(self, ra, dec, mag):
        self.ra = ra
        self.dec = dec
        self.mag = mag
    
    def flux(self):
        return 10**(-self.mag/2.5)

# Create and time OOP
start = time.perf_counter()
stars = [Star(i, i, 10) for i in range(1000)]
fluxes_oop = [s.flux() for s in stars]
oop_time = time.perf_counter() - start

# NumPy approach
start = time.perf_counter()
star_data = np.array([(i, i, 10) for i in range(1000)],
                     dtype=[('ra', 'f8'), ('dec', 'f8'), ('mag', 'f8')])
fluxes_np = 10**(-star_data['mag']/2.5)
np_time = time.perf_counter() - start

print(f"OOP time: {oop_time*1000:.2f} ms")
print(f"NumPy time: {np_time*1000:.2f} ms")
print(f"Speedup: {oop_time/np_time:.1f}x")
```

## Main Takeaways

You've just learned one of programming's most powerful organizational paradigms — object-oriented programming. This isn't just about syntax or following rules; it's about modeling the world (and universe!) in code. When you create a Star class with mass, temperature, and luminosity attributes, along with methods to calculate its spectral class or evolution, you're not just organizing data — you're creating a computational model that mirrors how we think about stars scientifically.

The beauty of OOP in astronomy is how naturally it maps to our domain. Telescopes are objects with properties (aperture, focal_length) and behaviors (slew_to_target, take_exposure). Observations have data and methods to reduce them. Galaxies contain stellar populations which contain stars. This hierarchical, object-based thinking is already how astronomers conceptualize the universe. OOP just gives us the tools to express it in code.

But perhaps the most important lesson is knowing when NOT to use OOP. Not every problem needs a class. Simple calculations should remain functions. Collections of utilities work better as modules. The art lies in recognizing when you have entities with state and behavior (use classes), versus pure transformations of data (use functions). This judgment comes with practice, but you now have the foundation to make these decisions thoughtfully.

Looking ahead, you'll see OOP everywhere in scientific Python. NumPy arrays are objects with methods like reshape() and mean(). Matplotlib figures maintain state and expose methods like savefig(). Astropy coordinates know their reference frame and can transform themselves. Understanding OOP isn't just about writing your own classes — it's about leveraging the rich ecosystem of scientific objects that others have created. You're now equipped to not just use these tools, but understand their design and even contribute to them.

## Definitions

**Attribute**: A variable that belongs to an object or class. Instance attributes belong to specific objects, while class attributes are shared by all instances.

**Class**: A blueprint for creating objects. Defines what attributes and methods objects will have.

**Class Method**: A method that receives the class as first argument (`@classmethod`). Often used for alternative constructors.

**Composition**: Design pattern where objects contain other objects ("has-a" relationship).

**Constructor**: The `__init__` method that initializes new objects when they're created.

**Context Manager**: Object implementing `__enter__` and `__exit__` methods for use with `with` statements.

**Decorator**: Special syntax using `@` that modifies functions or methods (`@property`, `@staticmethod`).

**Duck Typing**: Python philosophy that an object's suitability is determined by its methods/attributes, not its type.

**Encapsulation**: Bundling data and methods that operate on that data within a single unit (class).

**Inheritance**: Mechanism where a class derives attributes and methods from another class ("is-a" relationship).

**Instance**: A specific object created from a class.

**Method**: A function defined inside a class that operates on instances of that class.

**Method Resolution Order (MRO)**: The order Python searches through classes when looking for methods in inheritance hierarchies.

**Property**: Special attribute that executes methods when accessed (`@property` decorator).

**Self**: First parameter of instance methods, referring to the specific instance being operated on.

**Special Methods**: Methods with double underscores (`__init__`, `__str__`) that define object behavior for built-in operations.

**Super**: Built-in function for accessing parent class methods in inheritance hierarchies.

## Key Takeaways

✅ **Classes bundle data and behavior together** — When data and operations naturally belong together (like a star's properties and calculations), classes provide clean organization with state persistence.

✅ **The self parameter connects methods to their object** — It's just Python's way of passing the object to its methods. Forgetting it is the most common OOP error.

✅ **Properties provide controlled access** — Use `@property` for computed attributes, validation, and maintaining data consistency without explicit method calls.

✅ **Inheritance models "is-a", composition models "has-a"** — Choose inheritance for specialized versions (WhiteDwarf is-a Star), composition for contained objects (Telescope has-a Mount).

✅ **Special methods make objects Pythonic** — Implementing `__str__`, `__len__`, `__add__` lets your objects work seamlessly with built-in functions and operators.

✅ **Context managers ensure cleanup** — Use `__enter__` and `__exit__` for resources that need guaranteed cleanup (files, connections, locks).

✅ **Performance has tradeoffs** — Objects use more memory than tuples, but provide better organization. Use `__slots__` for memory-critical applications with many instances.

✅ **Not everything needs classes** — Use functions for simple calculations, modules for utilities, classes for stateful objects with behavior.

✅ **OOP is everywhere in scientific Python** — NumPy arrays, Matplotlib figures, and Astropy objects all use these principles. Understanding OOP helps you leverage the entire ecosystem.

✅ **Testing classes requires special consideration** — Test initialization, methods, and properties separately. Mock external dependencies for reliable tests.

## Quick Reference Tables

### Class Definition Syntax

| Concept | Syntax | Example |
|---------|--------|---------|
| Define class | `class Name:` | `class Star:` |
| Constructor | `def __init__(self):` | `def __init__(self, name):` |
| Instance attribute | `self.attr = value` | `self.mass = 1.989e30` |
| Class attribute | `attr = value` | `SPEED_OF_LIGHT = 3e8` |
| Instance method | `def method(self):` | `def luminosity(self):` |
| Class method | `@classmethod` | `@classmethod def from_file(cls):` |
| Static method | `@staticmethod` | `@staticmethod def validate():` |
| Property | `@property` | `@property def temperature(self):` |
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
| `__enter__` | Context entry | `with obj:` |
| `__exit__` | Context exit | End of `with` block |

### Debugging Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `dir(obj)` | List all attributes | `dir(star)` |
| `vars(obj)` | Get instance `__dict__` | `vars(star)` |
| `type(obj)` | Get object's class | `type(star)` |
| `isinstance(obj, cls)` | Check type | `isinstance(star, Star)` |
| `hasattr(obj, attr)` | Check attribute exists | `hasattr(star, 'mass')` |
| `getattr(obj, attr, default)` | Safe attribute access | `getattr(star, 'age', 0)` |
| `id(obj)` | Get memory address | `id(star)` |
| `Class.__mro__` | Method resolution order | `Planet.__mro__` |

## Next Chapter Preview

With object-oriented programming mastered, Chapter 7 introduces NumPy — the foundation of scientific computing in Python. You'll discover why NumPy arrays are 10-100x faster than Python lists and how vectorization eliminates explicit loops. The OOP concepts from this chapter directly explain NumPy's design: arrays are objects with methods (`arr.mean()`, `arr.reshape()`), special methods enable mathematical operators (`arr1 + arr2`), and properties provide computed attributes (`arr.shape`, `arr.T`). Understanding objects prepares you to leverage NumPy's full power and eventually create your own scientific classes that integrate seamlessly with the ecosystem.