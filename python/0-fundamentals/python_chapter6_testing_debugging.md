# Chapter 6: Testing, Debugging, and Error Handling

## Learning Objectives
By the end of this chapter, you will:
- Write comprehensive unit tests for scientific code
- Handle errors gracefully with exceptions
- Debug numerical algorithms effectively
- Validate scientific computations with property-based testing
- Profile and optimize code performance
- Implement logging for production astronomy software

## 6.1 Error Handling: Expecting the Unexpected

### The Exception Hierarchy

```python
import numpy as np
import warnings

def demonstrate_exception_hierarchy():
    """Show Python's exception hierarchy for scientific computing."""
    
    # Common exceptions in astronomical code
    exceptions = [
        (ValueError, "Invalid magnitude: -999", "Bad input values"),
        (TypeError, "Can't add string to float", "Type mismatch"),
        (IndexError, "Accessing spectrum[10000]", "Array bounds"),
        (KeyError, "Missing 'redshift' in dict", "Missing data"),
        (ZeroDivisionError, "Distance = 0 parsecs", "Division by zero"),
        (OverflowError, "exp(1000000)", "Numerical overflow"),
        (FileNotFoundError, "Missing FITS file", "I/O error"),
        (MemoryError, "10000x10000x10000 array", "Out of memory")
    ]
    
    for exc_type, example, description in exceptions:
        print(f"{exc_type.__name__:20} - {description:20} (e.g., {example})")

demonstrate_exception_hierarchy()
```

### Defensive Programming with Try-Except

```python
class SpectrumAnalyzer:
    """Analyze astronomical spectra with robust error handling."""
    
    def __init__(self, wavelengths, fluxes):
        """Initialize with validation."""
        if len(wavelengths) != len(fluxes):
            raise ValueError(f"Wavelength and flux arrays must have same length: "
                           f"{len(wavelengths)} != {len(fluxes)}")
        
        if len(wavelengths) == 0:
            raise ValueError("Cannot analyze empty spectrum")
        
        self.wavelengths = np.array(wavelengths)
        self.fluxes = np.array(fluxes)
    
    def find_continuum(self, method='median', window=50):
        """
        Estimate continuum level with error handling.
        
        Parameters
        ----------
        method : str
            'median', 'mean', or 'polynomial'
        window : int
            Window size for rolling estimate
        """
        try:
            if method == 'median':
                # Robust to outliers
                from scipy.ndimage import median_filter
                continuum = median_filter(self.fluxes, size=window)
            
            elif method == 'mean':
                # Faster but sensitive to lines
                from scipy.ndimage import uniform_filter1d
                continuum = uniform_filter1d(self.fluxes, size=window)
            
            elif method == 'polynomial':
                # Fit polynomial to spectrum
                coeffs = np.polyfit(self.wavelengths, self.fluxes, deg=3)
                continuum = np.polyval(coeffs, self.wavelengths)
            
            else:
                raise ValueError(f"Unknown method: {method}. "
                               f"Use 'median', 'mean', or 'polynomial'")
            
            return continuum
            
        except ImportError as e:
            # Handle missing scipy gracefully
            warnings.warn(f"SciPy not available, falling back to NumPy: {e}")
            return np.median(self.fluxes) * np.ones_like(self.fluxes)
        
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"Error in continuum finding: {e}")
            return self.fluxes.copy()
    
    def measure_line(self, wavelength, tolerance=1.0):
        """
        Measure spectral line properties with comprehensive error handling.
        
        Parameters
        ----------
        wavelength : float
            Expected line wavelength (nm)
        tolerance : float
            Search window (nm)
        
        Returns
        -------
        dict or None
            Line properties or None if not found
        """
        try:
            # Find wavelength range
            mask = np.abs(self.wavelengths - wavelength) < tolerance
            
            if not np.any(mask):
                warnings.warn(f"No data near {wavelength} nm")
                return None
            
            # Extract line region
            wave_region = self.wavelengths[mask]
            flux_region = self.fluxes[mask]
            
            # Avoid division by zero in equivalent width
            continuum = np.median(self.fluxes)
            if continuum == 0:
                raise ValueError("Continuum level is zero - cannot normalize")
            
            # Calculate line properties
            peak_idx = np.argmax(np.abs(flux_region - continuum))
            peak_wavelength = wave_region[peak_idx]
            peak_flux = flux_region[peak_idx]
            
            # Equivalent width with trapezoidal integration
            ew = np.trapz(1 - flux_region/continuum, wave_region)
            
            return {
                'wavelength': peak_wavelength,
                'flux': peak_flux,
                'equivalent_width': ew,
                'is_emission': peak_flux > continuum
            }
            
        except ValueError as e:
            print(f"Value error in line measurement: {e}")
            return None
        
        except (IndexError, TypeError) as e:
            print(f"Data access error: {e}")
            return None
        
        finally:
            # Always execute cleanup code
            # Useful for closing files, releasing resources
            pass

# Test error handling
wavelengths = np.linspace(400, 700, 1000)
fluxes = np.ones_like(wavelengths)

# Add emission line
line_center = 656.3  # H-alpha
fluxes += 2 * np.exp(-(wavelengths - line_center)**2 / 0.5**2)

analyzer = SpectrumAnalyzer(wavelengths, fluxes)
line = analyzer.measure_line(656.3)
if line:
    print(f"H-alpha detected: {line['wavelength']:.1f} nm, "
          f"EW = {line['equivalent_width']:.2f} nm")

# Test error cases
try:
    bad_analyzer = SpectrumAnalyzer([1, 2, 3], [4, 5])  # Mismatched lengths
except ValueError as e:
    print(f"Caught expected error: {e}")
```

### Custom Exceptions for Domain-Specific Errors

```python
class AstronomyError(Exception):
    """Base class for astronomy-specific exceptions."""
    pass

class CoordinateError(AstronomyError):
    """Invalid astronomical coordinates."""
    pass

class ObservabilityError(AstronomyError):
    """Target not observable."""
    pass

class ConvergenceError(AstronomyError):
    """Numerical method failed to converge."""
    def __init__(self, message, iterations=None, tolerance=None):
        super().__init__(message)
        self.iterations = iterations
        self.tolerance = tolerance

class Telescope:
    """Telescope with observation constraints."""
    
    def __init__(self, latitude, elevation_limit=10):
        self.latitude = latitude  # degrees
        self.elevation_limit = elevation_limit  # degrees
    
    def observe(self, ra, dec):
        """
        Attempt to observe target.
        
        Raises
        ------
        CoordinateError
            If coordinates are invalid
        ObservabilityError
            If target cannot be observed from this location
        """
        # Validate coordinates
        if not (0 <= ra <= 360):
            raise CoordinateError(f"RA must be 0-360 degrees, got {ra}")
        
        if not (-90 <= dec <= 90):
            raise CoordinateError(f"Dec must be -90 to 90 degrees, got {dec}")
        
        # Check observability (simplified)
        max_altitude = 90 - abs(self.latitude - dec)
        
        if max_altitude < self.elevation_limit:
            raise ObservabilityError(
                f"Target at Dec={dec}° never rises above {self.elevation_limit}° "
                f"from latitude {self.latitude}°"
            )
        
        return f"Observing target at RA={ra}, Dec={dec}"

# Use custom exceptions
palomar = Telescope(latitude=33.36, elevation_limit=20)

targets = [
    ("M31", 10.68, 41.27),  # Andromeda - observable
    ("LMC", 80.89, -69.76),  # Large Magellanic Cloud - too far south
    ("Invalid", 400, 0),     # Bad coordinates
]

for name, ra, dec in targets:
    try:
        result = palomar.observe(ra, dec)
        print(f"✓ {name}: {result}")
    except CoordinateError as e:
        print(f"✗ {name}: Invalid coordinates - {e}")
    except ObservabilityError as e:
        print(f"✗ {name}: Cannot observe - {e}")
```

## 6.2 Testing Scientific Code

### Unit Testing with pytest

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class StellarPhysics:
    """Class with physics calculations to test."""
    
    @staticmethod
    def luminosity_from_magnitude(magnitude, distance_pc):
        """Calculate luminosity from apparent magnitude and distance."""
        if distance_pc <= 0:
            raise ValueError("Distance must be positive")
        
        # Absolute magnitude
        abs_mag = magnitude - 5 * np.log10(distance_pc) + 5
        
        # Solar absolute magnitude in V band
        solar_abs_mag = 4.83
        
        # Luminosity in solar units
        luminosity = 10**((solar_abs_mag - abs_mag) / 2.5)
        
        return luminosity
    
    @staticmethod
    def schwarzschild_radius(mass_kg):
        """Calculate Schwarzschild radius."""
        G = 6.67430e-11
        c = 299792458
        return 2 * G * mass_kg / c**2
    
    @staticmethod
    def orbital_velocity(mass_kg, radius_m):
        """Calculate circular orbital velocity."""
        G = 6.67430e-11
        return np.sqrt(G * mass_kg / radius_m)

# Test file: test_stellar_physics.py
class TestStellarPhysics:
    """Test suite for stellar physics calculations."""
    
    def test_luminosity_sun(self):
        """Test that we get L=1 for the Sun."""
        # Sun's apparent magnitude from 10 pc
        sun_mag_10pc = 4.83
        L = StellarPhysics.luminosity_from_magnitude(sun_mag_10pc, 10)
        assert_allclose(L, 1.0, rtol=1e-10)
    
    def test_luminosity_distance_scaling(self):
        """Test inverse square law."""
        L1 = StellarPhysics.luminosity_from_magnitude(0, 10)
        L2 = StellarPhysics.luminosity_from_magnitude(5, 100)  # 5 mag fainter, 10x farther
        assert_allclose(L1, L2, rtol=1e-10)
    
    def test_luminosity_invalid_distance(self):
        """Test that negative distance raises error."""
        with pytest.raises(ValueError, match="Distance must be positive"):
            StellarPhysics.luminosity_from_magnitude(10, -5)
    
    def test_schwarzschild_radius_sun(self):
        """Test Schwarzschild radius of the Sun."""
        solar_mass = 1.98892e30
        rs = StellarPhysics.schwarzschild_radius(solar_mass)
        expected = 2953.25  # meters
        assert_allclose(rs, expected, rtol=0.01)
    
    def test_orbital_velocity_earth(self):
        """Test Earth's orbital velocity."""
        solar_mass = 1.98892e30
        au = 1.496e11
        v = StellarPhysics.orbital_velocity(solar_mass, au)
        expected = 29780  # m/s
        assert_allclose(v, expected, rtol=0.01)
    
    @pytest.mark.parametrize("mass,radius,expected", [
        (1.989e30, 1.496e11, 29780),  # Earth around Sun
        (5.972e24, 384400000, 1022),  # Moon around Earth
        (1.989e30, 7.78e11, 13070),   # Jupiter around Sun
    ])
    def test_orbital_velocity_multiple(self, mass, radius, expected):
        """Test orbital velocity for multiple systems."""
        v = StellarPhysics.orbital_velocity(mass, radius)
        assert_allclose(v, expected, rtol=0.01)

# Run tests (in practice, use pytest from command line)
# pytest test_stellar_physics.py -v
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

class Photometry:
    """Photometric calculations with properties to test."""
    
    @staticmethod
    def combine_magnitudes(magnitudes):
        """Combine multiple magnitudes (like binary star)."""
        if len(magnitudes) == 0:
            raise ValueError("Need at least one magnitude")
        
        fluxes = 10**(-0.4 * np.array(magnitudes))
        total_flux = np.sum(fluxes)
        return -2.5 * np.log10(total_flux)
    
    @staticmethod
    def color_index(mag1, mag2):
        """Calculate color index (e.g., B-V)."""
        return mag1 - mag2
    
    @staticmethod
    def extinction_correction(observed_mag, color_excess, R_v=3.1):
        """Correct for interstellar extinction."""
        A_v = R_v * color_excess
        return observed_mag - A_v

class TestPhotometryProperties:
    """Property-based tests for photometry."""
    
    @given(st.floats(min_value=-30, max_value=30))
    def test_single_magnitude_unchanged(self, mag):
        """Single magnitude should equal combined magnitude."""
        combined = Photometry.combine_magnitudes([mag])
        assert_allclose(combined, mag, rtol=1e-10)
    
    @given(
        arrays(np.float64, shape=st.integers(2, 10),
               elements=st.floats(min_value=-10, max_value=20))
    )
    def test_combined_brighter_than_any_component(self, mags):
        """Combined magnitude should be brighter than brightest component."""
        combined = Photometry.combine_magnitudes(mags)
        assert combined < np.min(mags)  # Lower magnitude = brighter
    
    @given(
        st.floats(min_value=-5, max_value=25),
        st.floats(min_value=-5, max_value=25)
    )
    def test_identical_binary_magnitude(self, mag1, mag2):
        """Two identical stars should be 0.75 mag brighter than one."""
        if abs(mag1 - mag2) < 1e-10:  # Identical
            combined = Photometry.combine_magnitudes([mag1, mag2])
            expected = mag1 - 2.5 * np.log10(2)
            assert_allclose(combined, expected, rtol=1e-5)
    
    @given(
        st.floats(min_value=0, max_value=20),
        st.floats(min_value=0, max_value=3)
    )
    def test_extinction_makes_fainter(self, obs_mag, color_excess):
        """Extinction correction should make object brighter (lower mag)."""
        if color_excess > 0:
            corrected = Photometry.extinction_correction(obs_mag, color_excess)
            assert corrected < obs_mag  # Should be brighter after correction

# Example of property-based test finding edge cases
def test_magnitude_combination_properties():
    """Test mathematical properties of magnitude combination."""
    
    # Commutative property
    mags1 = [10, 11, 12]
    mags2 = [12, 10, 11]  # Same values, different order
    assert_allclose(
        Photometry.combine_magnitudes(mags1),
        Photometry.combine_magnitudes(mags2)
    )
    
    # Adding very faint star shouldn't change result
    bright = [0.0]
    bright_plus_faint = [0.0, 30.0]  # 30 mag is essentially zero flux
    assert_allclose(
        Photometry.combine_magnitudes(bright),
        Photometry.combine_magnitudes(bright_plus_faint),
        rtol=1e-6
    )

test_magnitude_combination_properties()
print("Property tests passed!")
```

### Testing Numerical Algorithms

```python
class NumericalIntegrator:
    """Integration methods with convergence testing."""
    
    @staticmethod
    def adaptive_simpson(func, a, b, tol=1e-6, max_depth=10):
        """
        Adaptive Simpson's rule integration.
        
        Automatically subdivides intervals for accuracy.
        """
        def simpson_step(f, a, b):
            """Single Simpson's rule evaluation."""
            h = (b - a) / 2
            return h/3 * (f(a) + 4*f(a + h) + f(b))
        
        def adaptive_helper(f, a, b, tol, depth):
            """Recursive adaptive integration."""
            if depth > max_depth:
                raise ConvergenceError(
                    f"Max recursion depth {max_depth} reached",
                    iterations=depth
                )
            
            c = (a + b) / 2
            
            # Compute integral over whole interval and subintervals
            S_whole = simpson_step(f, a, b)
            S_left = simpson_step(f, a, c)
            S_right = simpson_step(f, c, b)
            S_split = S_left + S_right
            
            # Error estimate
            error = abs(S_split - S_whole) / 15
            
            if error < tol:
                # Accurate enough
                return S_split + (S_split - S_whole) / 15
            else:
                # Need more refinement
                left = adaptive_helper(f, a, c, tol/2, depth + 1)
                right = adaptive_helper(f, c, b, tol/2, depth + 1)
                return left + right
        
        return adaptive_helper(func, a, b, tol, 0)

class TestNumericalIntegration:
    """Test numerical integration accuracy."""
    
    def test_polynomial_exact(self):
        """Simpson's rule is exact for polynomials up to degree 3."""
        def cubic(x):
            return 2*x**3 - 3*x**2 + 4*x - 5
        
        # Analytical integral
        def cubic_integral(x):
            return 0.5*x**4 - x**3 + 2*x**2 - 5*x
        
        a, b = 0, 5
        expected = cubic_integral(b) - cubic_integral(a)
        result = NumericalIntegrator.adaptive_simpson(cubic, a, b, tol=1e-10)
        
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_gaussian_integral(self):
        """Test integral of Gaussian (erf function)."""
        def gaussian(x):
            return np.exp(-x**2)
        
        # Integral from 0 to 1 should be approximately erf(1) * sqrt(pi)/2
        from scipy.special import erf
        expected = erf(1) * np.sqrt(np.pi) / 2
        
        result = NumericalIntegrator.adaptive_simpson(gaussian, 0, 1)
        assert_allclose(result, expected, rtol=1e-6)
    
    def test_singularity_detection(self):
        """Test handling of singularities."""
        def singular(x):
            if abs(x) < 1e-10:
                return float('inf')
            return 1 / x
        
        # Should fail to converge
        with pytest.raises(ConvergenceError) as exc_info:
            NumericalIntegrator.adaptive_simpson(singular, -1, 1)
        
        assert exc_info.value.iterations is not None

# Run integration tests
tester = TestNumericalIntegration()
tester.test_polynomial_exact()
tester.test_gaussian_integral()
print("Integration tests passed!")
```

## 6.3 Debugging Scientific Code

### Common Numerical Bugs

```python
def demonstrate_numerical_bugs():
    """Common bugs in astronomical calculations."""
    
    print("1. Floating Point Comparison Bug:")
    redshift1 = 0.1 + 0.2
    redshift2 = 0.3
    print(f"   0.1 + 0.2 == 0.3? {redshift1 == redshift2}")  # False!
    print(f"   Use np.isclose: {np.isclose(redshift1, redshift2)}")
    
    print("\n2. Integer Division Bug (Python 2 legacy):")
    pixels = 4096
    binning = 3
    print(f"   4096 / 3 = {pixels / binning}")  # Float in Python 3
    print(f"   4096 // 3 = {pixels // binning}")  # Integer division
    
    print("\n3. Mutable Default Argument Bug:")
    def add_observation(target, obs_list=[]):  # BUG!
        obs_list.append(target)
        return obs_list
    
    list1 = add_observation("M31")
    list2 = add_observation("M42")  # M31 appears here too!
    print(f"   List 2 contains: {list2}")  # ['M31', 'M42']
    
    print("\n4. Array Broadcasting Bug:")
    ra = np.array([150, 151, 152])  # Shape (3,)
    dec = np.array([[30], [31]])    # Shape (2, 1)
    try:
        combined = ra + dec  # Broadcasts to (2, 3)
        print(f"   Unexpected shape: {combined.shape}")
    except ValueError as e:
        print(f"   Broadcasting error: {e}")
    
    print("\n5. Catastrophic Cancellation:")
    a = 1.0000001
    b = 1.0000000
    diff = a - b  # Lost most precision
    print(f"   Direct subtraction: {diff}")
    print(f"   Better formula needed for small differences")

demonstrate_numerical_bugs()
```

### Debugging Strategies

```python
class DebuggedOrbitIntegrator:
    """Orbit integrator with debugging capabilities."""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.call_count = 0
        self.energy_history = []
    
    def gravity_force(self, r, mass):
        """Calculate gravitational acceleration."""
        self.call_count += 1
        
        # Debugging: Check for invalid inputs
        if self.debug:
            if np.any(np.isnan(r)):
                raise ValueError(f"NaN in position: {r}")
            if mass <= 0:
                raise ValueError(f"Invalid mass: {mass}")
        
        r_mag = np.linalg.norm(r)
        
        # Debugging: Check for singularity
        if r_mag < 1e-10:
            if self.debug:
                print(f"WARNING: Near singularity at r = {r_mag}")
            r_mag = 1e-10  # Softening parameter
        
        G = 6.67430e-11
        return -G * mass * r / r_mag**3
    
    def integrate_orbit(self, initial_pos, initial_vel, mass, dt, steps):
        """Integrate orbit with energy monitoring."""
        pos = np.array(initial_pos)
        vel = np.array(initial_vel)
        
        positions = [pos.copy()]
        
        for step in range(steps):
            # Leapfrog integration
            acc = self.gravity_force(pos, mass)
            vel += acc * dt
            pos += vel * dt
            
            positions.append(pos.copy())
            
            # Debugging: Monitor energy conservation
            if self.debug:
                ke = 0.5 * np.linalg.norm(vel)**2
                pe = -6.67430e-11 * mass / np.linalg.norm(pos)
                total_energy = ke + pe
                self.energy_history.append(total_energy)
                
                if step > 0:
                    energy_change = abs(total_energy - self.energy_history[0])
                    if energy_change / abs(self.energy_history[0]) > 0.01:
                        print(f"WARNING: Energy drift at step {step}: "
                              f"{energy_change/abs(self.energy_history[0])*100:.2f}%")
            
            # Debugging: Check for escape
            if np.linalg.norm(pos) > 1e15:  # 1000 AU
                if self.debug:
                    print(f"Object escaped at step {step}")
                break
        
        if self.debug:
            print(f"Integration complete: {self.call_count} force evaluations")
            print(f"Final energy drift: "
                  f"{(self.energy_history[-1]/self.energy_history[0] - 1)*100:.4f}%")
        
        return np.array(positions)

# Test with debugging enabled
integrator = DebuggedOrbitIntegrator(debug=True)

# Earth orbit parameters
r0 = [1.496e11, 0, 0]  # 1 AU
v0 = [0, 29780, 0]  # Orbital velocity
solar_mass = 1.989e30

print("Integrating Earth orbit with debugging...")
trajectory = integrator.integrate_orbit(r0, v0, solar_mass, dt=3600, steps=100)
```

### Using Python Debugger (pdb)

```python
def find_spectral_peak(wavelengths, fluxes, line_center, window=5):
    """
    Find peak near expected line position.
    
    This function has a bug - let's debug it!
    """
    # Find wavelengths within window
    mask = np.abs(wavelengths - line_center) < window
    
    if not np.any(mask):
        return None
    
    # Extract region
    wave_region = wavelengths[mask]
    flux_region = fluxes[mask]
    
    # BUG: What if there are multiple peaks?
    peak_idx = np.argmax(flux_region)
    
    # Uncomment to debug:
    # import pdb; pdb.set_trace()
    # Now you can inspect variables:
    # (Pdb) p wave_region
    # (Pdb) p flux_region
    # (Pdb) p peak_idx
    
    peak_wavelength = wave_region[peak_idx]
    peak_flux = flux_region[peak_idx]
    
    return peak_wavelength, peak_flux

# Create test spectrum with double peak
wavelengths = np.linspace(650, 660, 100)
fluxes = np.ones_like(wavelengths)
fluxes += np.exp(-(wavelengths - 654)**2 / 0.1)  # First peak
fluxes += 2 * np.exp(-(wavelengths - 656.3)**2 / 0.1)  # Stronger peak (H-alpha)

result = find_spectral_peak(wavelengths, fluxes, 656.3, window=5)
if result:
    print(f"Found peak at {result[0]:.1f} nm with flux {result[1]:.2f}")
```

## 6.4 Logging and Profiling

### Structured Logging

```python
import logging
import time
from functools import wraps

# Configure logging for astronomy pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('astronomy_pipeline.log'),
        logging.StreamHandler()
    ]
)

class ObservationPipeline:
    """Data reduction pipeline with comprehensive logging."""
    
    def __init__(self, name="Pipeline"):
        self.logger = logging.getLogger(name)
        self.stats = {'processed': 0, 'failed': 0}
    
    def log_timing(self, func):
        """Decorator to log function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            self.logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self.logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
                return result
            
            except Exception as e:
                elapsed = time.perf_counter() - start
                self.logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    
    def process_observation(self, filename):
        """Process single observation with logging."""
        self.logger.info(f"Processing {filename}")
        
        try:
            # Simulate processing steps
            data = self.load_data(filename)
            calibrated = self.calibrate(data)
            result = self.extract_photometry(calibrated)
            
            self.stats['processed'] += 1
            self.logger.info(f"Successfully processed {filename}")
            return result
            
        except FileNotFoundError:
            self.logger.warning(f"File not found: {filename}")
            self.stats['failed'] += 1
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing {filename}: {e}")
            self.stats['failed'] += 1
            raise
    
    @log_timing
    def load_data(self, filename):
        """Load observation data."""
        # Simulate loading
        time.sleep(0.1)
        if 'bad' in filename:
            raise ValueError("Corrupted file")
        return {'filename': filename, 'data': np.random.randn(100, 100)}
    
    @log_timing
    def calibrate(self, data):
        """Apply calibration."""
        time.sleep(0.05)
        self.logger.debug(f"Calibrating with dark and flat frames")
        return data
    
    @log_timing
    def extract_photometry(self, data):
        """Extract photometric measurements."""
        time.sleep(0.02)
        flux = np.random.uniform(100, 1000)
        self.logger.debug(f"Extracted flux: {flux:.1f}")
        return {'flux': flux}
    
    def process_batch(self, filenames):
        """Process multiple observations."""
        self.logger.info(f"Starting batch processing of {len(filenames)} files")
        
        results = []
        for filename in filenames:
            try:
                result = self.process_observation(filename)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Skipping {filename}: {e}")
                continue
        
        self.logger.info(f"Batch complete: {self.stats['processed']} processed, "
                        f"{self.stats['failed']} failed")
        return results

# Test logging
pipeline = ObservationPipeline("TestPipeline")
files = ['obs001.fits', 'obs002.fits', 'bad_obs.fits', 'obs003.fits']
# results = pipeline.process_batch(files)  # Uncomment to see logging
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

class PerformanceProfiler:
    """Profile performance of astronomical calculations."""
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """Profile a single function call."""
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Get statistics
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        print(stream.getvalue())
        return result
    
    @staticmethod
    def benchmark_algorithms():
        """Compare different algorithm implementations."""
        import timeit
        
        # Compare different ways to calculate distances
        n_stars = 1000
        ra = np.random.uniform(0, 360, n_stars)
        dec = np.random.uniform(-90, 90, n_stars)
        
        def naive_distances():
            """Naive O(n²) implementation."""
            distances = np.zeros((n_stars, n_stars))
            for i in range(n_stars):
                for j in range(n_stars):
                    dra = ra[i] - ra[j]
                    ddec = dec[i] - dec[j]
                    distances[i, j] = np.sqrt(dra**2 + ddec**2)
            return distances
        
        def vectorized_distances():
            """Vectorized implementation using broadcasting."""
            dra = ra[:, np.newaxis] - ra[np.newaxis, :]
            ddec = dec[:, np.newaxis] - dec[np.newaxis, :]
            return np.sqrt(dra**2 + ddec**2)
        
        # Time both approaches
        naive_time = timeit.timeit(naive_distances, number=1)
        vector_time = timeit.timeit(vectorized_distances, number=1)
        
        print(f"Performance comparison for {n_stars} stars:")
        print(f"  Naive (loops): {naive_time:.3f}s")
        print(f"  Vectorized: {vector_time:.3f}s")
        print(f"  Speedup: {naive_time/vector_time:.1f}x")

# Memory profiling
def memory_usage_demo():
    """Demonstrate memory profiling."""
    import tracemalloc
    
    tracemalloc.start()
    
    # Simulate memory-intensive operation
    large_array = np.random.randn(1000, 1000)
    processed = np.fft.fft2(large_array)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

# Run profiling demos
PerformanceProfiler.benchmark_algorithms()
memory_usage_demo()
```

## Try It Yourself

### Exercise 6.1: Robust FITS File Reader
Create a FITS file reader with comprehensive error handling.

```python
class FITSReader:
    """
    Robust FITS file reader with error handling and validation.
    
    Should handle:
    - Missing files
    - Corrupted headers
    - Invalid data types
    - Memory limitations
    """
    
    def __init__(self, filename):
        self.filename = filename
        self.header = None
        self.data = None
    
    def read(self):
        """
        Read FITS file with comprehensive error handling.
        
        Returns
        -------
        tuple
            (header, data) or (None, None) on failure
        """
        # Your code here
        # Handle: FileNotFoundError, MemoryError, ValueError
        # Validate: Header keywords, data dimensions
        # Log: All operations and errors
        pass
    
    def validate_header(self, header):
        """Check required header keywords."""
        # Your code here
        pass
    
    def validate_data(self, data):
        """Check data integrity."""
        # Your code here
        # Check for NaN, Inf, reasonable ranges
        pass

# Test your implementation
reader = FITSReader("test.fits")
header, data = reader.read()
if data is not None:
    print(f"Successfully read {data.shape} array")
```

### Exercise 6.2: Test Suite for Cosmology Calculator
Write comprehensive tests for cosmological calculations.

```python
class CosmologyCalculator:
    """Calculate cosmological distances and times."""
    
    def __init__(self, H0=70, Om0=0.3, OL0=0.7):
        self.H0 = H0  # km/s/Mpc
        self.Om0 = Om0  # Matter density
        self.OL0 = OL0  # Dark energy density
    
    def luminosity_distance(self, z):
        """Calculate luminosity distance for redshift z."""
        # Your code here
        # Should handle z < 0, z > 1000
        pass
    
    def age_at_z(self, z):
        """Calculate age of universe at redshift z."""
        # Your code here
        pass
    
    def comoving_distance(self, z):
        """Calculate comoving distance."""
        # Your code here
        pass

class TestCosmology:
    """Test suite for cosmology calculator."""
    
    def test_distance_at_z0(self):
        """Distance at z=0 should be 0."""
        # Your code here
        pass
    
    def test_distance_increases_with_z(self):
        """Luminosity distance should increase with redshift."""
        # Your code here
        pass
    
    def test_age_decreases_with_z(self):
        """Universe was younger at higher redshift."""
        # Your code here
        pass
    
    def test_invalid_redshift(self):
        """Negative redshift should raise error."""
        # Your code here
        pass
    
    @pytest.mark.parametrize("z,expected_gyr", [
        (0, 13.8),
        (1, 5.9),
        (2, 3.3),
        (10, 0.5)
    ])
    def test_age_benchmarks(self, z, expected_gyr):
        """Test against known values."""
        # Your code here
        pass

# Run tests
cosmo = CosmologyCalculator()
print(f"Distance at z=1: {cosmo.luminosity_distance(1)} Mpc")
```

### Exercise 6.3: Debug and Optimize N-Body Code
Find and fix bugs in this N-body simulation.

```python
def buggy_nbody_simulation(masses, positions, velocities, dt, steps):
    """
    This N-body simulation has several bugs. Find and fix them!
    
    Bugs to find:
    1. Energy is not conserved
    2. Particles escape unexpectedly
    3. Performance is terrible for large N
    """
    n_bodies = len(masses)
    pos = positions.copy()  # BUG: Shallow copy issue?
    vel = velocities.copy()
    
    G = 6.67e-11
    
    for step in range(steps):
        forces = np.zeros_like(pos)
        
        # Calculate forces - BUG: Double counting?
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r)
                    # BUG: What if r_mag = 0?
                    forces[i] = G * masses[i] * masses[j] * r / r_mag**3
        
        # Update velocities and positions - BUG: Wrong integration order?
        pos += vel * dt
        vel += forces / masses[:, np.newaxis] * dt
    
    return pos, vel

# Debug this code:
# 1. Add energy conservation check
# 2. Fix the force calculation
# 3. Improve integration scheme
# 4. Add error handling
# 5. Profile and optimize

# Your improved version here:
def fixed_nbody_simulation(masses, positions, velocities, dt, steps):
    """Your fixed and optimized version."""
    # Your code here
    pass
```

## Key Takeaways

✅ **Handle errors gracefully** - Use try-except for predictable failures  
✅ **Create custom exceptions** - Make domain-specific errors clear  
✅ **Test edge cases** - Empty arrays, negative values, infinity  
✅ **Use property-based testing** - Let hypothesis find weird inputs  
✅ **Assert physical constraints** - Energy conservation, positive mass  
✅ **Log strategically** - Info for operations, debug for details, error for failures  
✅ **Profile before optimizing** - Measure, don't guess performance  
✅ **Debug systematically** - Use pdb, logging, and assertions  

## Next Chapter Preview
We'll explore performance optimization techniques including vectorization, Numba JIT compilation, parallel processing, and memory optimization for large astronomical datasets.