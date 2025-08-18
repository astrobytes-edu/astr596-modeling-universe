# Chapter 8: Advanced Python Topics (Optional Sampler)

*This chapter provides a sampling of advanced Python topics. Choose sections relevant to your projects.*

## Learning Objectives
This optional chapter introduces:
- Async programming for concurrent I/O operations
- Advanced decorators and descriptors
- Context managers for resource management
- Type hints and static typing
- Metaclasses and introspection
- Package development and distribution

## 8.1 Async Programming for Concurrent Operations

### When Async Makes Sense

```python
import asyncio
import aiohttp
import time

# Synchronous version - slow for I/O bound tasks
def fetch_catalog_sync(catalog_ids):
    """Fetch multiple catalogs synchronously."""
    results = []
    for cat_id in catalog_ids:
        # Simulate HTTP request
        time.sleep(0.5)  # Network delay
        results.append(f"Catalog {cat_id} data")
    return results

# Asynchronous version - much faster for I/O
async def fetch_catalog_async(session, catalog_id):
    """Fetch single catalog asynchronously."""
    # Simulate async HTTP request
    await asyncio.sleep(0.5)  # Network delay
    return f"Catalog {catalog_id} data"

async def fetch_all_catalogs(catalog_ids):
    """Fetch multiple catalogs concurrently."""
    # In real code, use aiohttp.ClientSession()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for cat_id in catalog_ids:
            task = fetch_catalog_async(session, cat_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    return results

# Telescope control example
class AsyncTelescopeController:
    """Control telescope with async operations."""
    
    def __init__(self):
        self.position = {'ra': 0, 'dec': 0}
        self.filter = 'V'
        self.camera_ready = False
    
    async def slew_to(self, ra, dec):
        """Slew telescope to position."""
        print(f"Starting slew to RA={ra}, Dec={dec}")
        await asyncio.sleep(3)  # Simulate slew time
        self.position = {'ra': ra, 'dec': dec}
        print("Slew complete")
    
    async def change_filter(self, filter_name):
        """Change filter wheel."""
        print(f"Changing filter to {filter_name}")
        await asyncio.sleep(1)
        self.filter = filter_name
        print("Filter changed")
    
    async def prepare_camera(self):
        """Prepare CCD camera."""
        print("Preparing camera")
        await asyncio.sleep(2)
        self.camera_ready = True
        print("Camera ready")
    
    async def observe_target(self, ra, dec, filter_name, exposure):
        """Complete observation sequence."""
        # Run preparation tasks concurrently
        tasks = [
            self.slew_to(ra, dec),
            self.change_filter(filter_name),
            self.prepare_camera()
        ]
        
        await asyncio.gather(*tasks)
        
        # Take exposure
        print(f"Starting {exposure}s exposure")
        await asyncio.sleep(exposure / 10)  # Simulate faster
        print("Exposure complete")
        
        return {'ra': ra, 'dec': dec, 'filter': filter_name, 'data': 'image_data'}

# Example usage
async def observation_sequence():
    """Run a sequence of observations."""
    telescope = AsyncTelescopeController()
    
    targets = [
        (10.68, 41.27, 'V', 300),  # M31
        (5.58, -5.39, 'R', 600),   # M42
        (13.42, -11.16, 'B', 450)  # M104
    ]
    
    results = []
    for ra, dec, filt, exp in targets:
        result = await telescope.observe_target(ra, dec, filt, exp)
        results.append(result)
    
    return results

# Run async code
# asyncio.run(observation_sequence())
```

### Async for Real-Time Data Streams

```python
class AsyncDataStreamProcessor:
    """Process real-time data streams asynchronously."""
    
    async def generate_telemetry(self, n_points=100):
        """Simulate telemetry data stream."""
        for i in range(n_points):
            await asyncio.sleep(0.1)  # Data arrives every 100ms
            
            data = {
                'timestamp': time.time(),
                'temperature': 20 + np.random.randn(),
                'pressure': 1013 + np.random.randn() * 10,
                'seeing': 0.8 + np.random.random() * 0.5
            }
            yield data
    
    async def process_stream(self):
        """Process incoming data stream."""
        buffer = []
        
        async for data in self.generate_telemetry():
            buffer.append(data)
            
            # Process every 10 points
            if len(buffer) >= 10:
                await self.analyze_buffer(buffer)
                buffer = []
    
    async def analyze_buffer(self, buffer):
        """Analyze buffered data."""
        temps = [d['temperature'] for d in buffer]
        seeing = [d['seeing'] for d in buffer]
        
        print(f"Average temp: {np.mean(temps):.1f}°C, "
              f"seeing: {np.mean(seeing):.2f}\"")

# Run: asyncio.run(AsyncDataStreamProcessor().process_stream())
```

## 8.2 Advanced Decorators

### Parametrized Decorators

```python
def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    """
    Decorator to retry failed operations.
    
    Useful for network requests, hardware control, etc.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_observation_data(obs_id):
    """Fetch observation with automatic retry."""
    if np.random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network error")
    return f"Data for observation {obs_id}"

# Test retry decorator
# try:
#     data = fetch_observation_data("OBS123")
#     print(f"Success: {data}")
# except Exception as e:
#     print(f"Failed after retries: {e}")
```

### Class Decorators

```python
def add_validation(cls):
    """Class decorator to add validation to all setters."""
    
    original_setattr = cls.__setattr__
    
    def validated_setattr(self, name, value):
        # Check if there's a validator method
        validator_name = f'validate_{name}'
        if hasattr(self, validator_name):
            validator = getattr(self, validator_name)
            value = validator(value)
        
        original_setattr(self, name, value)
    
    cls.__setattr__ = validated_setattr
    return cls

@add_validation
class ValidatedStar:
    """Star with automatic validation."""
    
    def __init__(self, mass, temperature):
        self.mass = mass
        self.temperature = temperature
    
    def validate_mass(self, value):
        """Validate stellar mass."""
        if value <= 0:
            raise ValueError("Mass must be positive")
        if value > 150:
            raise ValueError("Mass exceeds stellar limit")
        return value
    
    def validate_temperature(self, value):
        """Validate temperature."""
        if value < 2000:
            raise ValueError("Temperature too low for star")
        if value > 50000:
            raise ValueError("Temperature too high")
        return value

# Validation happens automatically
star = ValidatedStar(1.0, 5778)
# star.mass = -1  # Raises ValueError
```

## 8.3 Descriptors and Properties

### Custom Descriptors

```python
class PhysicalQuantity:
    """Descriptor for physical quantities with units."""
    
    def __init__(self, name, unit, min_value=None, max_value=None):
        self.name = name
        self.unit = unit
        self.min_value = min_value
        self.max_value = max_value
        self.data = {}
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.data.get(id(instance), None)
    
    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value} {self.unit}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value} {self.unit}")
        
        self.data[id(instance)] = value
    
    def __delete__(self, instance):
        del self.data[id(instance)]

class Telescope:
    """Telescope with physical quantity descriptors."""
    
    aperture = PhysicalQuantity('aperture', 'm', min_value=0.1, max_value=100)
    focal_length = PhysicalQuantity('focal_length', 'm', min_value=0.1)
    elevation = PhysicalQuantity('elevation', 'degrees', min_value=0, max_value=90)
    
    def __init__(self, aperture, focal_length):
        self.aperture = aperture
        self.focal_length = focal_length
        self.elevation = 45  # Default
    
    @property
    def f_ratio(self):
        """Calculate f-ratio."""
        return self.focal_length / self.aperture

# Use descriptors
scope = Telescope(2.4, 57.6)  # Hubble
print(f"f-ratio: f/{scope.f_ratio:.1f}")
# scope.aperture = -1  # Raises ValueError
```

## 8.4 Context Managers

### Custom Context Managers

```python
class ObservationSession:
    """Context manager for observation sessions."""
    
    def __init__(self, observer, target):
        self.observer = observer
        self.target = target
        self.start_time = None
        self.log = []
    
    def __enter__(self):
        """Start observation session."""
        self.start_time = time.time()
        self.log.append(f"Session started for {self.target}")
        print(f"Beginning observation of {self.target}")
        
        # Initialize equipment
        self._initialize_equipment()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End observation session."""
        elapsed = time.time() - self.start_time
        
        if exc_type is not None:
            self.log.append(f"Session failed: {exc_val}")
            print(f"Error during observation: {exc_val}")
        else:
            self.log.append(f"Session completed successfully")
        
        self.log.append(f"Duration: {elapsed:.1f} seconds")
        
        # Cleanup
        self._shutdown_equipment()
        self._save_log()
        
        # Return False to propagate exceptions
        return False
    
    def _initialize_equipment(self):
        """Initialize telescope and camera."""
        print("  Initializing equipment...")
        time.sleep(0.5)
    
    def _shutdown_equipment(self):
        """Safely shutdown equipment."""
        print("  Shutting down equipment...")
        time.sleep(0.5)
    
    def _save_log(self):
        """Save observation log."""
        print(f"  Log saved: {len(self.log)} entries")
    
    def take_exposure(self, duration):
        """Take an exposure."""
        self.log.append(f"Exposure: {duration}s")
        time.sleep(duration / 10)  # Simulate
        return f"Image data for {duration}s exposure"

# Use context manager
with ObservationSession("Observer1", "M31") as session:
    image1 = session.take_exposure(30)
    image2 = session.take_exposure(60)
    print(f"  Captured 2 images")

# Equipment is automatically cleaned up
```

### Contextlib Utilities

```python
from contextlib import contextmanager, suppress

@contextmanager
def temporary_seed(seed):
    """Temporarily set random seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# Use temporary seed
print("Random without seed:", np.random.random())

with temporary_seed(42):
    print("Random with seed 42:", np.random.random())
    print("Again with seed 42:", np.random.random())

print("Random without seed:", np.random.random())

# Suppress specific exceptions
with suppress(FileNotFoundError):
    os.remove('nonexistent_file.txt')  # Doesn't raise error
```

## 8.5 Type Hints and Static Typing

### Advanced Type Hints

```python
from typing import (
    Union, Optional, List, Dict, Tuple, 
    TypeVar, Generic, Protocol, Literal,
    overload, cast
)
from typing_extensions import TypedDict

# Type variables for generics
T = TypeVar('T', bound=float)

class Spectrum(Generic[T]):
    """Generic spectrum class."""
    
    def __init__(self, wavelengths: List[T], fluxes: List[T]) -> None:
        self.wavelengths = wavelengths
        self.fluxes = fluxes
    
    def normalize(self) -> 'Spectrum[T]':
        """Normalize spectrum."""
        max_flux = max(self.fluxes)
        normalized = [f/max_flux for f in self.fluxes]
        return Spectrum(self.wavelengths, normalized)

# Typed dictionaries for structured data
class ObservationData(TypedDict):
    """Type hints for observation dictionary."""
    target: str
    ra: float
    dec: float
    filter: Literal['U', 'B', 'V', 'R', 'I']
    exposure: float
    airmass: Optional[float]

def process_observation(data: ObservationData) -> Dict[str, float]:
    """Process observation with type checking."""
    # Type checker knows the structure
    magnitude = -2.5 * np.log10(data['exposure'])
    
    if data['airmass'] is not None:
        magnitude += 0.2 * data['airmass']  # Extinction
    
    return {'magnitude': magnitude}

# Protocol for duck typing with types
class Plottable(Protocol):
    """Protocol for objects that can be plotted."""
    
    def get_x_data(self) -> List[float]: ...
    def get_y_data(self) -> List[float]: ...
    def get_label(self) -> str: ...

def plot_data(obj: Plottable) -> None:
    """Plot any object following Plottable protocol."""
    x = obj.get_x_data()
    y = obj.get_y_data()
    label = obj.get_label()
    # plt.plot(x, y, label=label)

# Function overloading
@overload
def load_data(filename: str) -> np.ndarray: ...

@overload
def load_data(filename: str, return_header: Literal[True]) -> Tuple[np.ndarray, Dict]: ...

def load_data(filename, return_header=False):
    """Load data with optional header."""
    data = np.random.randn(100, 100)
    header = {'OBJECT': 'M31', 'EXPTIME': 300}
    
    if return_header:
        return data, header
    return data
```

## 8.6 Metaclasses and Introspection

### Understanding Metaclasses

```python
class SingletonMeta(type):
    """Metaclass for creating singleton classes."""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ObservatoryConfig(metaclass=SingletonMeta):
    """Singleton configuration class."""
    
    def __init__(self):
        self.location = "Palomar"
        self.latitude = 33.36
        self.longitude = -116.86
        self.elevation = 1706
        print("Creating config instance")

# Only one instance ever created
config1 = ObservatoryConfig()
config2 = ObservatoryConfig()
print(f"Same instance? {config1 is config2}")

# Automatic registration metaclass
class RegisteredMeta(type):
    """Metaclass that registers all subclasses."""
    
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register non-base classes
        if bases:
            mcs.registry[name] = cls
        
        return cls

class Instrument(metaclass=RegisteredMeta):
    """Base instrument class."""
    pass

class CCD(Instrument):
    """CCD camera."""
    pass

class Spectrograph(Instrument):
    """Spectrograph."""
    pass

print(f"Registered instruments: {list(Instrument.registry.keys())}")
```

### Introspection and Reflection

```python
import inspect

class IntrospectableObject:
    """Object with introspection capabilities."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def introspect(self):
        """Examine object's attributes and methods."""
        print(f"Class: {self.__class__.__name__}")
        print(f"Module: {self.__class__.__module__}")
        
        print("\nAttributes:")
        for name, value in inspect.getmembers(self):
            if not name.startswith('_') and not inspect.ismethod(value):
                print(f"  {name}: {value}")
        
        print("\nMethods:")
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith('_'):
                sig = inspect.signature(method)
                print(f"  {name}{sig}")
    
    def describe_method(self, method_name):
        """Get detailed info about a method."""
        method = getattr(self, method_name)
        sig = inspect.signature(method)
        
        print(f"Method: {method_name}")
        print(f"Signature: {sig}")
        print(f"Parameters:")
        
        for param_name, param in sig.parameters.items():
            print(f"  {param_name}: {param.annotation if param.annotation != param.empty else 'Any'}")
        
        if method.__doc__:
            print(f"Docstring: {method.__doc__.strip()}")

# Dynamic class creation
def create_filter_class(filter_name, wavelength):
    """Dynamically create filter classes."""
    
    def __init__(self):
        self.name = filter_name
        self.wavelength = wavelength
    
    def info(self):
        return f"{self.name} filter at {self.wavelength}nm"
    
    # Create class dynamically
    FilterClass = type(
        f'{filter_name}Filter',
        (object,),
        {
            '__init__': __init__,
            'info': info,
            'filter_type': filter_name
        }
    )
    
    return FilterClass

# Create filter classes dynamically
VFilter = create_filter_class('V', 550)
RFilter = create_filter_class('R', 700)

v_filter = VFilter()
print(f"Dynamic class: {v_filter.info()}")

## 8.7 Packaging and Distribution

### Creating a Python Package

```python
# Example package structure for an astronomy library
"""
astrotools/
├── setup.py
├── setup.cfg
├── pyproject.toml
├── README.md
├── LICENSE
├── requirements.txt
├── astrotools/
│   ├── __init__.py
│   ├── photometry/
│   │   ├── __init__.py
│   │   ├── aperture.py
│   │   └── psf.py
│   ├── spectroscopy/
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   └── calibration.py
│   └── utils/
│       ├── __init__.py
│       └── coordinates.py
└── tests/
    ├── test_photometry.py
    └── test_spectroscopy.py
"""

# setup.py example
SETUP_PY = """
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="astrotools",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Astronomical data processing tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/astrotools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "astropy>=5.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "process-spectrum=astrotools.scripts.process:main",
        ],
    },
)
"""

# pyproject.toml for modern packaging
PYPROJECT_TOML = """
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "astrotools"
version = "0.1.0"
description = "Astronomical data processing tools"
authors = [{name = "Your Name", email = "your.email@example.com"}]
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "astropy>=5.0",
]

[project.optional-dependencies]
dev = ["pytest", "black", "mypy"]
docs = ["sphinx", "sphinx-rtd-theme"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
"""

print("Package structure examples saved to variables SETUP_PY and PYPROJECT_TOML")
```

## 8.8 Functional Programming Patterns

### Functional Approaches in Scientific Computing

```python
from functools import reduce, partial
from operator import add, mul
import itertools

class FunctionalAstronomy:
    """Functional programming patterns for astronomy."""
    
    @staticmethod
    def compose(*functions):
        """Compose multiple functions."""
        def inner(x):
            return reduce(lambda v, f: f(v), functions, x)
        return inner
    
    @staticmethod
    def curry(func):
        """Curry a function for partial application."""
        def curried(*args, **kwargs):
            if len(args) + len(kwargs) >= func.__code__.co_argcount:
                return func(*args, **kwargs)
            return partial(func, *args, **kwargs)
        return curried
    
    @staticmethod
    def pipeline_example():
        """Example: Processing pipeline with function composition."""
        
        # Define processing steps
        def load_spectrum(filename):
            """Load spectrum from file."""
            return np.random.randn(1000) + 100
        
        def remove_cosmic_rays(spectrum):
            """Remove cosmic ray hits."""
            cleaned = spectrum.copy()
            cleaned[cleaned > np.percentile(cleaned, 99)] = np.median(cleaned)
            return cleaned
        
        def normalize(spectrum):
            """Normalize to unit maximum."""
            return spectrum / np.max(spectrum)
        
        def smooth(window=5):
            """Return smoothing function."""
            def smoother(spectrum):
                from scipy.ndimage import uniform_filter1d
                return uniform_filter1d(spectrum, window)
            return smoother
        
        # Compose pipeline
        process = FunctionalAstronomy.compose(
            load_spectrum,
            remove_cosmic_rays,
            normalize,
            smooth(window=10)
        )
        
        # Process data
        result = process("spectrum.fits")
        return result
    
    @staticmethod
    def lazy_evaluation_example():
        """Demonstrate lazy evaluation with generators."""
        
        def fibonacci_orbit_periods():
            """Generate orbital periods following Fibonacci sequence."""
            a, b = 1, 1
            while True:
                yield a
                a, b = b, a + b
        
        # Take only what we need
        periods = itertools.islice(fibonacci_orbit_periods(), 10)
        return list(periods)
    
    @staticmethod
    @curry
    def redshift_wavelength(z, rest_wavelength):
        """Curried function for redshift calculation."""
        return rest_wavelength * (1 + z)

# Examples
fa = FunctionalAstronomy()

# Function composition
result = fa.pipeline_example()
print(f"Pipeline result shape: {result.shape}")

# Currying
redshift_z2 = fa.redshift_wavelength(2.0)  # Partial application
h_alpha_z2 = redshift_z2(656.3)
print(f"H-alpha at z=2: {h_alpha_z2:.1f} nm")

# Lazy evaluation
fib_periods = fa.lazy_evaluation_example()
print(f"Fibonacci periods: {fib_periods}")
```

## 8.9 Working with Binary Data

### Binary File Formats

```python
import struct

class BinaryDataHandler:
    """Handle binary astronomical data formats."""
    
    @staticmethod
    def write_binary_catalog(filename, catalog):
        """Write catalog to binary format."""
        # Format: int32(n_objects), then for each object:
        # float64(ra), float64(dec), float32(mag), int32(id)
        
        with open(filename, 'wb') as f:
            # Write header
            n_objects = len(catalog)
            f.write(struct.pack('i', n_objects))
            
            # Write each object
            for obj in catalog:
                data = struct.pack(
                    'ddfI',  # double, double, float, unsigned int
                    obj['ra'],
                    obj['dec'],
                    obj['mag'],
                    obj['id']
                )
                f.write(data)
    
    @staticmethod
    def read_binary_catalog(filename):
        """Read catalog from binary format."""
        catalog = []
        
        with open(filename, 'rb') as f:
            # Read header
            n_objects = struct.unpack('i', f.read(4))[0]
            
            # Read each object
            for _ in range(n_objects):
                data = f.read(struct.calcsize('ddfI'))
                ra, dec, mag, obj_id = struct.unpack('ddfI', data)
                
                catalog.append({
                    'ra': ra,
                    'dec': dec,
                    'mag': mag,
                    'id': obj_id
                })
        
        return catalog
    
    @staticmethod
    def memory_mapped_array_example():
        """Work with memory-mapped arrays."""
        
        # Create a large memory-mapped array
        filename = 'large_image.dat'
        shape = (4096, 4096)
        dtype = np.float32
        
        # Create and write
        fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        fp[:] = np.random.randn(*shape)
        del fp
        
        # Read specific sections without loading all
        fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
        
        # Only this section is loaded into memory
        cutout = fp[1000:1100, 2000:2100]
        print(f"Cutout shape: {cutout.shape}, mean: {np.mean(cutout):.3f}")
        
        del fp
        os.remove(filename)

# Test binary handling
handler = BinaryDataHandler()

# Create test catalog
test_catalog = [
    {'ra': 150.0 + i*0.1, 'dec': 30.0 + i*0.05, 'mag': 12.0 + i*0.2, 'id': i}
    for i in range(10)
]

# Write and read
handler.write_binary_catalog('test.cat', test_catalog)
loaded = handler.read_binary_catalog('test.cat')
print(f"Loaded {len(loaded)} objects from binary catalog")
os.remove('test.cat')

handler.memory_mapped_array_example()
```

## Try It Yourself

### Exercise 8.1: Async Observatory Controller
Build an async system for controlling multiple telescopes.

```python
class AsyncObservatory:
    """
    Control multiple telescopes asynchronously.
    
    Requirements:
    - Coordinate multiple telescopes
    - Handle concurrent observations
    - Manage shared resources (e.g., weather station)
    - Implement error recovery
    """
    
    def __init__(self, n_telescopes):
        self.telescopes = [f"T{i+1}" for i in range(n_telescopes)]
        # Your code here
    
    async def observe_target_list(self, targets):
        """
        Observe list of targets optimally using all telescopes.
        
        Should:
        - Distribute targets among telescopes
        - Handle failures gracefully
        - Optimize for minimal total time
        """
        # Your code here
        pass
    
    async def emergency_stop(self):
        """Emergency stop all telescopes."""
        # Your code here
        pass

# Test your implementation
# observatory = AsyncObservatory(3)
# targets = [("M31", 300), ("M42", 600), ("M51", 450)]
# asyncio.run(observatory.observe_target_list(targets))
```

### Exercise 8.2: Custom Descriptor System
Create a descriptor system for validated scientific data.

```python
class ScientificProperty:
    """
    Descriptor for scientific properties with:
    - Units and unit conversion
    - Validation ranges
    - Uncertainty tracking
    - Automatic documentation
    """
    
    def __init__(self, name, unit, uncertainty=None):
        # Your code here
        pass
    
    def __get__(self, instance, owner):
        # Your code here
        pass
    
    def __set__(self, instance, value):
        # Your code here
        pass

class Measurement:
    """Use your ScientificProperty descriptor."""
    
    # Your properties here
    # temperature = ScientificProperty('temperature', 'K', uncertainty=0.1)
    # pressure = ScientificProperty('pressure', 'Pa', uncertainty=10)
    
    pass

# Test your descriptor
# m = Measurement()
# m.temperature = (273.15, 0.05)  # Value with uncertainty
# print(f"T = {m.temperature}")
```

### Exercise 8.3: Metaclass for Data Validation
Create a metaclass that automatically validates data classes.

```python
class ValidatedMeta(type):
    """
    Metaclass that:
    - Automatically creates validators from type hints
    - Adds logging to all methods
    - Implements singleton pattern for config classes
    - Registers all subclasses
    """
    
    def __new__(mcs, name, bases, namespace):
        # Your code here
        pass

class ObservationData(metaclass=ValidatedMeta):
    """Your data class using the metaclass."""
    
    # Type hints that become validators
    ra: float  # Should validate 0 <= ra <= 360
    dec: float  # Should validate -90 <= dec <= 90
    magnitude: float  # Should validate reasonable range
    
    # Your implementation
    pass

# Test your metaclass
# obs = ObservationData()
# obs.ra = 361  # Should raise ValueError
```

## Key Takeaways

✅ **Async for I/O** - Use asyncio for telescope control, data streaming  
✅ **Decorators add functionality** - Retry logic, validation, caching  
✅ **Descriptors for properties** - Custom validation and unit handling  
✅ **Context managers for resources** - Automatic setup/cleanup  
✅ **Type hints improve code** - Catch errors early, better documentation  
✅ **Metaclasses are powerful but rare** - Use for frameworks, not applications  
✅ **Functional patterns have their place** - Composition, currying for data pipelines  
✅ **Binary data needs care** - Struct module, memory mapping for large files  

## Moving Forward

This sampler introduced advanced Python concepts. Remember:

1. **Use advanced features when they solve real problems** - Not just to be clever
2. **Simple code is often better** - Readability counts
3. **Profile before optimizing** - Measure, don't guess
4. **Test complex code thoroughly** - Advanced features can hide bugs
5. **Document why, not just what** - Especially for metaclasses and descriptors

## Next Steps

You're now ready for the Scientific Python Libraries section:
- **NumPy** - The foundation of scientific computing
- **Matplotlib** - Publication-quality visualization  
- **SciPy** - Scientific algorithms and tools
- **Pandas** - Data analysis and manipulation

Choose which advanced topics to explore based on your project needs!
```