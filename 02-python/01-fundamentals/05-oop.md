# Chapter 5: Object-Oriented Programming

## Learning Objectives
By the end of this chapter, you will:
- Understand the philosophy of object-oriented design
- Create classes that model astronomical objects
- Master inheritance for code reuse and specialization
- Implement special methods for Pythonic behavior
- Design class hierarchies for scientific simulations
- Apply SOLID principles to astronomical software

## 5.1 Why Objects? Modeling the Universe

### The Philosophy of OOP

```python
# Procedural approach - data and functions separate
star_mass = 1.0  # Solar masses
star_luminosity = 1.0  # Solar luminosities
star_temperature = 5778  # Kelvin

def calculate_radius(luminosity, temperature):
    """Calculate stellar radius from Stefan-Boltzmann law."""
    import numpy as np
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    solar_luminosity = 3.828e26  # Watts
    
    L = luminosity * solar_luminosity
    R = np.sqrt(L / (4 * np.pi * sigma * temperature**4))
    return R / 6.96e8  # Return in solar radii

radius = calculate_radius(star_luminosity, star_temperature)
print(f"Radius: {radius:.2f} R☉")

print("\n" + "="*50 + "\n")

# Object-oriented approach - data and behavior together
class Star:
    """A star encapsulates data and behavior."""
    
    def __init__(self, mass, luminosity, temperature):
        self.mass = mass  # Solar masses
        self.luminosity = luminosity  # Solar luminosities
        self.temperature = temperature  # Kelvin
    
    def radius(self):
        """Calculate radius from luminosity and temperature."""
        import numpy as np
        sigma = 5.67e-8
        solar_luminosity = 3.828e26
        
        L = self.luminosity * solar_luminosity
        R = np.sqrt(L / (4 * np.pi * sigma * self.temperature**4))
        return R / 6.96e8
    
    def info(self):
        """Display star information."""
        return (f"Star: M={self.mass:.1f} M☉, L={self.luminosity:.1f} L☉, "
                f"T={self.temperature:.0f} K, R={self.radius():.2f} R☉")

# Create and use a star object
sun = Star(mass=1.0, luminosity=1.0, temperature=5778)
print(sun.info())
```

### Classes as Blueprints

```python
class Galaxy:
    """
    A class is a blueprint for creating objects.
    
    Class attributes are shared by all instances.
    Instance attributes are unique to each object.
    """
    
    # Class attribute - shared by all galaxies
    speed_of_light = 299792.458  # km/s
    
    def __init__(self, name, galaxy_type, redshift):
        # Instance attributes - unique to each galaxy
        self.name = name
        self.type = galaxy_type
        self.redshift = redshift
    
    def recession_velocity(self):
        """Calculate recession velocity from redshift."""
        # For low redshift, v ≈ cz
        return self.speed_of_light * self.redshift
    
    def distance_mpc(self, h0=70):
        """Estimate distance using Hubble's law."""
        return self.recession_velocity() / h0

# Create multiple instances from the same class
m31 = Galaxy("Andromeda", "Spiral", -0.001)  # Negative = approaching!
m87 = Galaxy("M87", "Elliptical", 0.0042)

print(f"{m31.name}: v = {m31.recession_velocity():.0f} km/s")
print(f"{m87.name}: d = {m87.distance_mpc():.1f} Mpc")

# Class attributes are shared
print(f"\nSpeed of light from M31 object: {m31.speed_of_light}")
print(f"Speed of light from M87 object: {m87.speed_of_light}")
print(f"Same object? {m31.speed_of_light is m87.speed_of_light}")
```

## 5.2 Building Robust Classes

### Properties and Validation

```python
class Planet:
    """A planet with validation and computed properties."""
    
    def __init__(self, name, mass_earth, radius_earth):
        self.name = name
        self._mass = None  # Private attribute
        self._radius = None
        
        # Use properties for validation
        self.mass = mass_earth
        self.radius = radius_earth
    
    @property
    def mass(self):
        """Mass in Earth masses."""
        return self._mass
    
    @mass.setter
    def mass(self, value):
        """Set mass with validation."""
        if value <= 0:
            raise ValueError(f"Mass must be positive, got {value}")
        if value > 13 * 317.8:  # 13 Jupiter masses = brown dwarf limit
            raise ValueError(f"Mass {value} M⊕ exceeds planetary limit")
        self._mass = value
    
    @property
    def radius(self):
        """Radius in Earth radii."""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Set radius with validation."""
        if value <= 0:
            raise ValueError(f"Radius must be positive, got {value}")
        self._radius = value
    
    @property
    def density(self):
        """Calculate density relative to Earth."""
        return self.mass / (self.radius ** 3)
    
    @property
    def surface_gravity(self):
        """Calculate surface gravity relative to Earth."""
        return self.mass / (self.radius ** 2)
    
    def classify(self):
        """Classify planet type based on properties."""
        if self.mass < 2:
            return "Terrestrial"
        elif self.mass < 15:
            return "Super-Earth"
        elif self.mass < 50:
            return "Neptune-like"
        else:
            return "Jupiter-like"

# Create planets with automatic validation
earth = Planet("Earth", 1.0, 1.0)
jupiter = Planet("Jupiter", 317.8, 11.2)

print(f"{earth.name}: density = {earth.density:.2f}, type = {earth.classify()}")
print(f"{jupiter.name}: density = {jupiter.density:.2f}, type = {jupiter.classify()}")

# Try invalid planet
try:
    invalid = Planet("Too Big", 5000, 10)  # Too massive!
except ValueError as e:
    print(f"Error: {e}")
```

### Special Methods (Magic Methods)

```python
class SpectralLine:
    """A spectral line with Pythonic behavior."""
    
    def __init__(self, wavelength, element, transition):
        self.wavelength = wavelength  # nm
        self.element = element
        self.transition = transition
    
    def __str__(self):
        """Human-readable string representation."""
        return f"{self.element} {self.transition} at {self.wavelength:.1f} nm"
    
    def __repr__(self):
        """Developer-friendly representation."""
        return (f"SpectralLine(wavelength={self.wavelength}, "
                f"element='{self.element}', transition='{self.transition}')")
    
    def __eq__(self, other):
        """Enable equality comparison."""
        if not isinstance(other, SpectralLine):
            return False
        return abs(self.wavelength - other.wavelength) < 0.1
    
    def __lt__(self, other):
        """Enable sorting by wavelength."""
        return self.wavelength < other.wavelength
    
    def __hash__(self):
        """Make hashable for use in sets/dicts."""
        return hash((round(self.wavelength, 1), self.element))

# Create spectral lines
h_alpha = SpectralLine(656.3, "Hydrogen", "n=3→2")
h_beta = SpectralLine(486.1, "Hydrogen", "n=4→2")
he_line = SpectralLine(587.6, "Helium", "D3")

# Pythonic behavior
print(str(h_alpha))  # Calls __str__
print(repr(h_beta))  # Calls __repr__

# Sorting works automatically
lines = [he_line, h_alpha, h_beta]
lines.sort()  # Uses __lt__
print(f"\nSorted lines: {[str(l) for l in lines]}")

# Can use in sets
unique_lines = {h_alpha, h_beta, h_alpha}  # Duplicate removed
print(f"Unique lines: {len(unique_lines)}")
```

### Container Classes

```python
class ObservationLog:
    """A container for observations with list-like behavior."""
    
    def __init__(self):
        self._observations = []
    
    def add(self, target, exposure, seeing):
        """Add an observation."""
        obs = {
            'target': target,
            'exposure': exposure,
            'seeing': seeing,
            'timestamp': self._get_timestamp()
        }
        self._observations.append(obs)
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def __len__(self):
        """Enable len(log)."""
        return len(self._observations)
    
    def __getitem__(self, index):
        """Enable log[index] access."""
        return self._observations[index]
    
    def __iter__(self):
        """Enable iteration: for obs in log."""
        return iter(self._observations)
    
    def __contains__(self, target):
        """Enable 'target in log' checks."""
        return any(obs['target'] == target for obs in self._observations)
    
    def filter_by_seeing(self, max_seeing):
        """Get observations with good seeing."""
        return [obs for obs in self._observations if obs['seeing'] <= max_seeing]

# Use the container
log = ObservationLog()
log.add("M31", 300, 1.2)
log.add("M42", 600, 0.8)
log.add("NGC1234", 450, 1.5)

print(f"Total observations: {len(log)}")
print(f"First observation: {log[0]}")
print(f"M31 observed? {'M31' in log}")

print("\nGood seeing observations:")
for obs in log.filter_by_seeing(1.0):
    print(f"  {obs['target']}: {obs['seeing']}\" seeing")
```

## 5.3 Inheritance: Building Hierarchies

### Basic Inheritance

```python
class CelestialBody:
    """Base class for all celestial objects."""
    
    def __init__(self, name, mass, radius):
        self.name = name
        self.mass = mass  # kg
        self.radius = radius  # m
    
    def density(self):
        """Calculate density in kg/m³."""
        import numpy as np
        volume = (4/3) * np.pi * self.radius**3
        return self.mass / volume
    
    def surface_gravity(self):
        """Calculate surface gravity in m/s²."""
        G = 6.67430e-11  # m³/kg/s²
        return G * self.mass / self.radius**2
    
    def escape_velocity(self):
        """Calculate escape velocity in m/s."""
        import numpy as np
        return np.sqrt(2 * self.surface_gravity() * self.radius)

class Star(CelestialBody):
    """Star extends CelestialBody with stellar properties."""
    
    def __init__(self, name, mass, radius, temperature, luminosity):
        # Call parent constructor
        super().__init__(name, mass, radius)
        
        # Add star-specific attributes
        self.temperature = temperature  # K
        self.luminosity = luminosity  # W
    
    def spectral_class(self):
        """Determine spectral class from temperature."""
        if self.temperature > 30000:
            return 'O'
        elif self.temperature > 10000:
            return 'B'
        elif self.temperature > 7500:
            return 'A'
        elif self.temperature > 6000:
            return 'F'
        elif self.temperature > 5200:
            return 'G'
        elif self.temperature > 3700:
            return 'K'
        else:
            return 'M'
    
    def habitable_zone(self):
        """Calculate habitable zone boundaries in AU."""
        import numpy as np
        solar_luminosity = 3.828e26
        L_ratio = self.luminosity / solar_luminosity
        
        inner = 0.95 * np.sqrt(L_ratio)
        outer = 1.37 * np.sqrt(L_ratio)
        return inner, outer

class Planet(CelestialBody):
    """Planet extends CelestialBody with planetary properties."""
    
    def __init__(self, name, mass, radius, orbital_distance, host_star=None):
        super().__init__(name, mass, radius)
        
        self.orbital_distance = orbital_distance  # m
        self.host_star = host_star
    
    def orbital_period(self):
        """Calculate orbital period using Kepler's third law."""
        if not self.host_star:
            return None
        
        import numpy as np
        G = 6.67430e-11
        T_squared = (4 * np.pi**2 * self.orbital_distance**3) / (G * self.host_star.mass)
        return np.sqrt(T_squared)
    
    def is_in_habitable_zone(self):
        """Check if planet is in star's habitable zone."""
        if not self.host_star or not hasattr(self.host_star, 'habitable_zone'):
            return False
        
        inner, outer = self.host_star.habitable_zone()
        au_to_m = 1.496e11
        distance_au = self.orbital_distance / au_to_m
        
        return inner <= distance_au <= outer

# Create stellar system
sun = Star("Sun", 1.989e30, 6.96e8, 5778, 3.828e26)
earth = Planet("Earth", 5.972e24, 6.371e6, 1.496e11, host_star=sun)

print(f"{sun.name}: Spectral class {sun.spectral_class()}")
print(f"  Habitable zone: {sun.habitable_zone()[0]:.2f} - {sun.habitable_zone()[1]:.2f} AU")

print(f"\n{earth.name}:")
print(f"  Escape velocity: {earth.escape_velocity()/1000:.1f} km/s")
print(f"  Orbital period: {earth.orbital_period()/(86400*365.25):.2f} years")
print(f"  In habitable zone? {earth.is_in_habitable_zone()}")
```

### Multiple Inheritance and Mixins

```python
class Rotator:
    """Mixin for rotating objects."""
    
    def __init__(self, rotation_period):
        self.rotation_period = rotation_period  # seconds
    
    def angular_velocity(self):
        """Calculate angular velocity in rad/s."""
        import numpy as np
        return 2 * np.pi / self.rotation_period
    
    def rotational_kinetic_energy(self, moment_of_inertia):
        """Calculate rotational kinetic energy."""
        omega = self.angular_velocity()
        return 0.5 * moment_of_inertia * omega**2

class Magnetized:
    """Mixin for objects with magnetic fields."""
    
    def __init__(self, magnetic_field):
        self.magnetic_field = magnetic_field  # Tesla
    
    def magnetic_moment(self):
        """Estimate magnetic moment."""
        import numpy as np
        return self.magnetic_field * self.radius**3

class Pulsar(Star, Rotator, Magnetized):
    """Pulsar combines star properties with rotation and magnetism."""
    
    def __init__(self, name, mass, radius, temperature, luminosity,
                 rotation_period, magnetic_field):
        # Initialize all parent classes
        Star.__init__(self, name, mass, radius, temperature, luminosity)
        Rotator.__init__(self, rotation_period)
        Magnetized.__init__(self, magnetic_field)
    
    def spindown_luminosity(self):
        """Calculate energy loss from magnetic dipole radiation."""
        import numpy as np
        c = 3e8
        mu0 = 4 * np.pi * 1e-7
        
        omega = self.angular_velocity()
        B = self.magnetic_field
        R = self.radius
        
        return (2 * B**2 * R**6 * omega**4) / (3 * mu0 * c**3)
    
    def characteristic_age(self):
        """Estimate pulsar age from spindown."""
        P = self.rotation_period
        P_dot = 1e-15  # Typical period derivative
        return P / (2 * P_dot) / (365.25 * 86400)  # Years

# Create a pulsar
crab_pulsar = Pulsar(
    name="Crab Pulsar",
    mass=1.4 * 1.989e30,  # 1.4 solar masses
    radius=10000,  # 10 km
    temperature=1e6,  # 1 million K
    luminosity=1e31,  # W
    rotation_period=0.033,  # 33 ms
    magnetic_field=1e8  # 10^8 Tesla
)

print(f"{crab_pulsar.name}:")
print(f"  Rotation: {1/crab_pulsar.rotation_period:.1f} Hz")
print(f"  Spindown luminosity: {crab_pulsar.spindown_luminosity():.2e} W")
print(f"  Characteristic age: {crab_pulsar.characteristic_age():.0f} years")
```

## 5.4 Polymorphism and Abstract Classes

### Duck Typing in Python

```python
class OpticalTelescope:
    """Ground-based optical telescope."""
    
    def __init__(self, aperture, focal_length):
        self.aperture = aperture  # meters
        self.focal_length = focal_length
    
    def observe(self, target):
        """Observe a target."""
        resolution = 1.22 * 0.55e-6 / self.aperture  # Rayleigh criterion
        return f"Optical observation of {target}, resolution: {resolution*206265:.2f}\""

class RadioTelescope:
    """Radio telescope (single dish or array)."""
    
    def __init__(self, diameter, frequency):
        self.diameter = diameter  # meters
        self.frequency = frequency  # GHz
    
    def observe(self, target):
        """Observe a target."""
        wavelength = 0.3 / self.frequency  # meters
        resolution = 1.22 * wavelength / self.diameter
        return f"Radio observation of {target} at {self.frequency} GHz, beam: {resolution*206265:.1f}\""

class SpaceTelescope:
    """Space-based telescope."""
    
    def __init__(self, aperture, orbit):
        self.aperture = aperture
        self.orbit = orbit
    
    def observe(self, target):
        """Observe a target."""
        return f"Space observation of {target} from {self.orbit} orbit, no atmosphere!"

# Polymorphism - same interface, different implementations
def conduct_observations(telescopes, target):
    """Use any telescope that has an observe method."""
    results = []
    for telescope in telescopes:
        # Duck typing - if it has observe(), we can use it
        result = telescope.observe(target)
        results.append(result)
    return results

# Create different telescope types
keck = OpticalTelescope(10, 17.5)
vla = RadioTelescope(25, 10)
hubble = SpaceTelescope(2.4, "LEO")

# All work with the same function!
observations = conduct_observations([keck, vla, hubble], "M31")
for obs in observations:
    print(obs)
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod
import numpy as np

class Integrator(ABC):
    """Abstract base class for numerical integrators."""
    
    @abstractmethod
    def step(self, func, y, t, dt):
        """Take one integration step."""
        pass
    
    def integrate(self, func, y0, t_span, dt):
        """Integrate from t0 to tf with step size dt."""
        t0, tf = t_span
        t = t0
        y = y0
        
        times = [t0]
        states = [y0]
        
        while t < tf:
            y = self.step(func, y, t, dt)
            t += dt
            times.append(t)
            states.append(y)
        
        return np.array(times), np.array(states)

class EulerIntegrator(Integrator):
    """Simple Euler method."""
    
    def step(self, func, y, t, dt):
        """Euler step: y_new = y + dt * f(y, t)"""
        return y + dt * func(y, t)

class RK4Integrator(Integrator):
    """Fourth-order Runge-Kutta method."""
    
    def step(self, func, y, t, dt):
        """RK4 step with four function evaluations."""
        k1 = func(y, t)
        k2 = func(y + 0.5*dt*k1, t + 0.5*dt)
        k3 = func(y + 0.5*dt*k2, t + 0.5*dt)
        k4 = func(y + dt*k3, t + dt)
        
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

class LeapfrogIntegrator(Integrator):
    """Leapfrog integrator for Hamiltonian systems."""
    
    def step(self, func, y, t, dt):
        """Leapfrog step for [position, velocity] state."""
        # Assume y = [position, velocity]
        # and func returns [velocity, acceleration]
        pos, vel = y[0], y[1]
        
        # Half step velocity
        derivatives = func(y, t)
        vel_half = vel + 0.5 * dt * derivatives[1]
        
        # Full step position
        pos_new = pos + dt * vel_half
        
        # Half step velocity again
        y_temp = np.array([pos_new, vel_half])
        derivatives = func(y_temp, t + dt)
        vel_new = vel_half + 0.5 * dt * derivatives[1]
        
        return np.array([pos_new, vel_new])

# Test with simple harmonic oscillator
def harmonic_oscillator(y, t, omega=1.0):
    """dy/dt for harmonic oscillator: x'' + omega^2 * x = 0"""
    pos, vel = y
    return np.array([vel, -omega**2 * pos])

# Compare integrators
y0 = np.array([1.0, 0.0])  # Initial position and velocity
t_span = (0, 10)
dt = 0.01

euler = EulerIntegrator()
rk4 = RK4Integrator()
leapfrog = LeapfrogIntegrator()

# Can't instantiate abstract class
try:
    bad = Integrator()  # This will fail!
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# But can use concrete implementations
for integrator, name in [(euler, "Euler"), (rk4, "RK4"), (leapfrog, "Leapfrog")]:
    t, y = integrator.integrate(harmonic_oscillator, y0, t_span, dt)
    energy = 0.5 * (y[:, 1]**2 + y[:, 0]**2)  # Should be conserved
    print(f"{name}: Energy drift = {abs(energy[-1] - energy[0]):.6f}")
```

## 5.5 Design Patterns for Scientific Computing

### Factory Pattern

```python
class TelescopeFactory:
    """Factory for creating different telescope types."""
    
    @staticmethod
    def create_telescope(config):
        """Create telescope based on configuration."""
        telescope_type = config.get('type')
        
        if telescope_type == 'optical':
            return OpticalTelescope(
                aperture=config['aperture'],
                focal_length=config['focal_length']
            )
        elif telescope_type == 'radio':
            return RadioTelescope(
                diameter=config['diameter'],
                frequency=config['frequency']
            )
        elif telescope_type == 'space':
            return SpaceTelescope(
                aperture=config['aperture'],
                orbit=config['orbit']
            )
        else:
            raise ValueError(f"Unknown telescope type: {telescope_type}")

# Use factory to create telescopes from configuration
configs = [
    {'type': 'optical', 'aperture': 8.2, 'focal_length': 15},
    {'type': 'radio', 'diameter': 100, 'frequency': 1.4},
    {'type': 'space', 'aperture': 6.5, 'orbit': 'L2'}
]

telescopes = [TelescopeFactory.create_telescope(conf) for conf in configs]
for scope in telescopes:
    print(f"Created: {scope.__class__.__name__}")
```

### Singleton Pattern for Constants

```python
class PhysicalConstants:
    """Singleton for physical constants."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_constants()
        return cls._instance
    
    def _initialize_constants(self):
        """Initialize constants once."""
        print("Initializing constants...")
        self.c = 299792458  # m/s
        self.G = 6.67430e-11  # m³/kg/s²
        self.h = 6.62607015e-34  # J⋅s
        self.k = 1.380649e-23  # J/K
        self.sigma = 5.670374419e-8  # W/m²/K⁴
        
        # Astronomical constants
        self.solar_mass = 1.98892e30  # kg
        self.solar_radius = 6.96e8  # m
        self.solar_luminosity = 3.828e26  # W
        self.au = 1.495978707e11  # m
        self.parsec = 3.0857e16  # m
        self.year = 365.25 * 86400  # s

# Always get the same instance
const1 = PhysicalConstants()
const2 = PhysicalConstants()
print(f"Same instance? {const1 is const2}")
print(f"Speed of light: {const1.c} m/s")
```

### Observer Pattern for Data Updates

```python
class Observatory:
    """Observable that notifies subscribers of new data."""
    
    def __init__(self, name):
        self.name = name
        self._observers = []
        self._latest_observation = None
    
    def attach(self, observer):
        """Add an observer."""
        self._observers.append(observer)
    
    def detach(self, observer):
        """Remove an observer."""
        self._observers.remove(observer)
    
    def notify(self):
        """Notify all observers of new data."""
        for observer in self._observers:
            observer.update(self._latest_observation)
    
    def new_observation(self, data):
        """Record new observation and notify observers."""
        print(f"\n{self.name}: New observation received")
        self._latest_observation = data
        self.notify()

class DataProcessor:
    """Observer that processes new data."""
    
    def __init__(self, name):
        self.name = name
    
    def update(self, data):
        """Process new data."""
        print(f"  {self.name}: Processing {data['target']} data")

class AlertSystem:
    """Observer that checks for interesting events."""
    
    def __init__(self, threshold):
        self.threshold = threshold
    
    def update(self, data):
        """Check if alert should be triggered."""
        if data['magnitude'] < self.threshold:
            print(f"  ALERT: Bright object detected! Mag = {data['magnitude']}")

# Set up observer pattern
observatory = Observatory("SDSU Observatory")
processor = DataProcessor("Pipeline")
alert = AlertSystem(threshold=10.0)

observatory.attach(processor)
observatory.attach(alert)

# Simulate observations
observatory.new_observation({'target': 'M31', 'magnitude': 3.4})
observatory.new_observation({'target': 'SN2024abc', 'magnitude': 8.5})
```

## Try It Yourself

### Exercise 5.1: Complete Stellar Evolution Class
Build a comprehensive stellar evolution model.

```python
class EvolvingStar:
    """
    Model a star that evolves over time.
    
    Should include:
    - Main sequence lifetime calculation
    - Evolution through different stages
    - Mass loss for high-mass stars
    - Final fate based on initial mass
    """
    
    def __init__(self, initial_mass):
        """Initialize with ZAMS properties."""
        self.initial_mass = initial_mass  # Solar masses
        self.current_mass = initial_mass
        self.age = 0  # Years
        self.stage = "Main Sequence"
        
        # Your code here
        # Calculate initial luminosity, temperature, radius
        pass
    
    def main_sequence_lifetime(self):
        """Calculate MS lifetime from initial mass."""
        # Your code here
        # Use M-L relation and fuel consumption
        pass
    
    def evolve(self, time_step):
        """Evolve star by time_step years."""
        # Your code here
        # Update properties based on evolutionary stage
        pass
    
    def final_fate(self):
        """Determine final fate based on mass."""
        # Your code here
        # White dwarf, neutron star, or black hole?
        pass

# Test your implementation
star = EvolvingStar(10)  # 10 solar mass star
print(f"Main sequence lifetime: {star.main_sequence_lifetime():.2e} years")

# Evolve through lifetime
while star.age < star.main_sequence_lifetime() * 1.1:
    star.evolve(1e6)  # Million year steps
    if star.stage != "Main Sequence":
        print(f"Left main sequence at age {star.age:.2e} years")
        break

print(f"Final fate: {star.final_fate()}")
```

### Exercise 5.2: N-Body System with Inheritance
Create a flexible N-body simulation framework.

```python
class GravitationalBody:
    """Base class for gravitational bodies."""
    
    def __init__(self, mass, position, velocity):
        # Your code here
        pass
    
    def gravitational_force(self, other):
        """Calculate gravitational force from another body."""
        # Your code here
        pass

class PointMass(GravitationalBody):
    """Simple point mass."""
    pass

class ExtendedBody(GravitationalBody):
    """Body with finite size and tidal effects."""
    
    def __init__(self, mass, position, velocity, radius):
        # Your code here
        pass
    
    def tidal_force(self, other):
        """Calculate tidal force."""
        # Your code here
        pass

class NBodySystem:
    """Container for N-body simulation."""
    
    def __init__(self):
        self.bodies = []
        self.time = 0
    
    def add_body(self, body):
        """Add a body to the system."""
        # Your code here
        pass
    
    def calculate_forces(self):
        """Calculate all pairwise forces."""
        # Your code here
        pass
    
    def integrate(self, dt, method='leapfrog'):
        """Advance system by dt."""
        # Your code here
        pass
    
    def total_energy(self):
        """Calculate total system energy."""
        # Your code here
        # Should be conserved!
        pass

# Test: Sun-Earth-Moon system
system = NBodySystem()
sun = PointMass(1.989e30, [0, 0, 0], [0, 0, 0])
earth = PointMass(5.972e24, [1.496e11, 0, 0], [0, 29780, 0])
moon = PointMass(7.342e22, [1.496e11 + 3.844e8, 0, 0], [0, 29780 + 1022, 0])

system.add_body(sun)
system.add_body(earth)
system.add_body(moon)

initial_energy = system.total_energy()
system.integrate(86400)  # One day
final_energy = system.total_energy()

print(f"Energy conservation: {abs(final_energy - initial_energy)/initial_energy:.2e}")
```

### Exercise 5.3: Observation Planning System
Design an observation planning system using OOP principles.

```python
class ObservationRequest:
    """Request for telescope time."""
    
    def __init__(self, target, priority, constraints):
        # Your code here
        pass
    
    def is_observable(self, time, location):
        """Check if target is observable at given time and location."""
        # Your code here
        # Check altitude, moon separation, etc.
        pass

class Scheduler:
    """Schedule observations optimally."""
    
    def __init__(self, telescope):
        self.telescope = telescope
        self.requests = []
    
    def add_request(self, request):
        """Add observation request to queue."""
        # Your code here
        pass
    
    def optimize_schedule(self, night_start, night_end):
        """
        Create optimal observation schedule.
        
        Consider:
        - Target visibility
        - Priority
        - Slew time between targets
        - Instrument changes
        """
        # Your code here
        pass

# Test the scheduler
scheduler = Scheduler(telescope=keck)  # Use telescope from earlier

# Add observation requests
scheduler.add_request(
    ObservationRequest("M31", priority=1, 
                      constraints={'min_altitude': 30, 'max_moon_sep': 30})
)
scheduler.add_request(
    ObservationRequest("M42", priority=2,
                      constraints={'min_altitude': 40, 'max_airmass': 2.0})
)

schedule = scheduler.optimize_schedule("2024-01-15 19:00", "2024-01-16 06:00")
print(f"Optimized schedule: {len(schedule)} observations")
```

## Key Takeaways

✅ **Classes bundle data and behavior** - Keep related things together  
✅ **Properties provide validation** - Ensure data consistency  
✅ **Special methods enable Pythonic code** - `__str__`, `__eq__`, `__len__`  
✅ **Inheritance promotes code reuse** - But prefer composition over deep hierarchies  
✅ **Abstract classes define interfaces** - Ensure consistent APIs  
✅ **Polymorphism enables flexibility** - Same interface, different implementations  
✅ **Design patterns solve common problems** - Factory, Singleton, Observer  
✅ **SOLID principles apply to scientific code** - Single responsibility, open/closed  

## Next Chapter Preview
We'll explore testing and debugging techniques to ensure your astronomical software is reliable and correct, including unit tests, integration tests, and debugging strategies for numerical code.